/**
 * Teach a stickman to walk, using every computer on the network.
 *
 *   node scripts/evolve-walker.ts [generations] [population] [--host <label>]
 *
 * `--host` pins every candidate to one machine. Normally you want the opposite — the
 * whole point is to spread the population — but pinning is how you measure what a single
 * machine contributes, which is the only way to answer "is a phone worth adding?"
 *
 * Each generation is one job: the population is split into slices, the fleet evaluates
 * them in parallel, and the winner becomes the parent of the next generation. The
 * platform schedules the work; the algorithm lives here, which is the division the
 * architecture asks for — the platform should not need to know what evolution is.
 */
import { readFileSync, existsSync } from 'node:fs'
import { evaluate, randomGenome, perturb } from '@dwp/protocol/walker.js'

const argv = process.argv.slice(2)
const positional = argv.filter(a => !a.startsWith('--') && !/^\d+$/.test(a) === false)
const GENERATIONS = Number(positional[0] ?? 300)
const POPULATION = Number(positional[1] ?? 240)
const HOST_LABEL = (() => {
  const i = argv.indexOf('--host')
  return i >= 0 ? argv[i + 1] : undefined
})()
const PER_TASK = 20
/** Never search finer than this. The old floor of 0.02 stalled every long run. */
const SIGMA_FLOOR = 0.045
/** Generations without improvement before the step size is pushed back up. */
const STAGNANT_BEFORE_BURST = 20
/** How much larger that exploratory step is. */
const BURST_FACTOR = 4
const STEPS = 1800
const EXPERIMENT = 'walker'

const PORT = Number(process.env.PORT ?? 8787)
const LOCAL = `http://127.0.0.1:${PORT}`

function env(key: string): string | undefined {
  if (process.env[key]) return process.env[key]
  if (!existsSync('.env')) return undefined
  for (const line of readFileSync('.env', 'utf8').split('\n')) {
    const m = new RegExp(`^${key}=(.*)$`).exec(line.trim())
    if (m) return m[1]
  }
  return undefined
}

const loginRes = await fetch(`${LOCAL}/auth/login`, {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: env('BOOTSTRAP_EMAIL'), password: env('BOOTSTRAP_PASSWORD') }),
})
if (!loginRes.ok) throw new Error(`could not sign in: ${loginRes.status}`)
const cookie = (loginRes.headers.getSetCookie?.()[0] ?? '').split(';')[0]!

const api = (path: string, body?: unknown): Promise<Response> =>
  fetch(`${LOCAL}${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { cookie, ...(body === undefined ? {} : { 'content-type': 'application/json' }) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })

let pinnedHostId: string | undefined
if (HOST_LABEL) {
  const { hosts } = await (await api('/hosts')).json() as
    { hosts: { id: string; label: string; online: boolean; adapters: string[] }[] }
  const match = hosts.find(h => h.online && h.label === HOST_LABEL)
  if (!match) {
    console.error(`\n  No online computer labelled "${HOST_LABEL}".\n  Online: ` +
      hosts.filter(h => h.online).map(h => h.label).join(', ') + '\n')
    process.exit(1)
  }
  if (!match.adapters?.includes('walker_evolution')) {
    console.error(`\n  "${HOST_LABEL}" cannot run walker_evolution — it advertises: ` +
      (match.adapters ?? []).join(', ') + '\n')
    process.exit(1)
  }
  pinnedHostId = match.id
  console.log(`  pinned to "${HOST_LABEL}" (${match.id.slice(0, 8)}…)`)
}

type TaskRow = { state: string; output: { results?: { seed: number; fitness: number; distance: number; ticks: number }[]; hostname?: string; evalsPerSecond?: number; evalMs?: number } | null; host_label: string | null }

const sleep = (ms: number): Promise<void> => new Promise(r => setTimeout(r, ms))

// Start from the best of a few random genomes so generation 1 is not pure noise.
let parent = randomGenome(1)
let best = evaluate(parent, STEPS)
for (let s = 2; s <= 60; s++) {
  const cand = randomGenome(s)
  const r = evaluate(cand, STEPS)
  if (r.fitness > best.fitness) { best = r; parent = cand }
}
console.log(`\n  seeded from 60 random gaits — fitness ${best.fitness.toFixed(2)}, ${best.distance}m\n`)

const history: { gen: number; fitness: number; distance: number; ticks: number; hosts: Record<string, number> }[] = []
let evaluations = 0
const startedAt = Date.now()

/**
 * How long the champion has gone without improving, and the best fitness seen.
 *
 * The original schedule decayed sigma 1.5% a generation against a floor of 0.02, which it
 * reached at generation 159. A 400-generation run therefore spent 60% of its compute
 * taking the smallest step it was allowed to, and measurably bought nothing with it:
 * fitness stopped moving within ten generations of the floor and did not move again.
 *
 * Decay is still right — big jumps early, refinement later. What was missing is a way
 * back out. Elitism makes that safe: seed 0 of every generation is the unperturbed
 * parent, so the champion can never be worse than the current best, and a larger step can
 * only cost a wasted generation, never a working gait.
 */
let bestSoFar = best.fitness
let stagnant = 0

for (let gen = 1; gen <= GENERATIONS; gen++) {
  // Shrink the step size as the gait improves: big jumps early to explore, small ones
  // later so a working gait is refined rather than destroyed.
  const decayed = 0.22 * Math.pow(0.985, gen)
  const bursting = stagnant >= STAGNANT_BEFORE_BURST
  const sigma = Math.min(0.30, Math.max(SIGMA_FLOOR, decayed) * (bursting ? BURST_FACTOR : 1))

  const tasks: Record<string, unknown>[] = []
  for (let offset = 0; offset < POPULATION; offset += PER_TASK) {
    const seeds: number[] = []
    for (let k = 0; k < PER_TASK && offset + k < POPULATION; k++) {
      // Seed 0 re-evaluates the parent unchanged, so the run can never go backwards.
      seeds.push(offset + k === 0 ? 0 : gen * 100_000 + offset + k)
    }
    tasks.push({ generation: gen, parent, sigma, seeds, steps: STEPS })
  }

  const res = await api('/jobs', {
    adapter: 'walker_evolution', tasks, ...(pinnedHostId ? { hostId: pinnedHostId } : {}),
  })
  if (!res.ok) { console.error(`  generation ${gen}: ${res.status} ${await res.text()}`); await sleep(3000); continue }
  const { jobId } = await res.json() as { jobId: string }

  type JobView = { job: { status: string }; tasks: TaskRow[] }
  let view: JobView | null = null
  for (let i = 0; i < 300; i++) {
    await sleep(400)
    view = await (await api(`/jobs/${jobId}`)).json() as JobView
    if (view.job.status !== 'running') break
  }
  if (!view) { console.error(`  generation ${gen}: no result`); continue }

  let champion: { seed: number; fitness: number; distance: number; ticks: number } | null = null
  const hosts: Record<string, number> = {}
  let genEvalMs = 0
  let genEvals = 0
  for (const t of view.tasks) {
    if (t.state !== 'succeeded' || !t.output?.results) continue
    hosts[t.host_label ?? '?'] = (hosts[t.host_label ?? '?'] ?? 0) + t.output.results.length
    genEvalMs += t.output.evalMs ?? 0
    genEvals += t.output.results.length
    for (const r of t.output.results) {
      evaluations += 1
      if (!champion || r.fitness > champion.fitness) champion = r
    }
  }
  if (!champion) { console.error(`  generation ${gen}: nothing came back`); continue }

  // Count stagnation before the parent moves, and clear it after a burst so exploration
  // comes in bursts rather than latching on permanently.
  if (champion.fitness > bestSoFar + 1e-9) { bestSoFar = champion.fitness; stagnant = 0 }
  else if (bursting) stagnant = 0
  else stagnant += 1

  // Rebuild the winner from its seed — the same arithmetic the host used.
  parent = perturb(parent, sigma, champion.seed)
  best = { fitness: champion.fitness, distance: champion.distance, ticks: champion.ticks, fell: false }
  history.push({ gen, fitness: champion.fitness, distance: champion.distance, ticks: champion.ticks, hosts })

  await api(`/experiments/${EXPERIMENT}`, {
    generation: gen,
    totalGenerations: GENERATIONS,
    population: POPULATION,
    steps: STEPS,
    sigma: Number(sigma.toFixed(4)),
    best: { fitness: champion.fitness, distance: champion.distance, ticks: champion.ticks },
    genome: parent,
    history: history.slice(-200),
    evaluations,
    elapsedSeconds: Math.round((Date.now() - startedAt) / 1000),
    hosts,
    updatedAt: new Date().toISOString(),
  })

  const split = Object.entries(hosts).map(([k, v]) => `${k.slice(0, 12)}:${v}`).join(' ')
  const rate = genEvalMs > 0 ? (genEvals / (genEvalMs / 1000)).toFixed(1) : '?'
  console.log(`  gen ${String(gen).padStart(3)}  fitness ${champion.fitness.toFixed(2).padStart(7)}  ` +
    `distance ${String(champion.distance).padStart(6)}m  upright ${String(champion.ticks).padStart(4)}/${STEPS}  ` +
    `${String(rate).padStart(6)} gaits/s  ${split}`)
}

console.log(`\n  done — ${evaluations} gait evaluations across the fleet in ` +
  `${Math.round((Date.now() - startedAt) / 1000)}s\n`)
