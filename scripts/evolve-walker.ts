/**
 * Teach a stickman to walk, using every computer on the network.
 *
 *   node scripts/evolve-walker.ts [generations] [population]
 *
 * Each generation is one job: the population is split into slices, the fleet evaluates
 * them in parallel, and the winner becomes the parent of the next generation. The
 * platform schedules the work; the algorithm lives here, which is the division the
 * architecture asks for — the platform should not need to know what evolution is.
 */
import { readFileSync, existsSync } from 'node:fs'
import { evaluate, randomGenome, perturb } from '@dwp/protocol/walker.js'

const GENERATIONS = Number(process.argv[2] ?? 300)
const POPULATION = Number(process.argv[3] ?? 240)
const PER_TASK = 20
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

type TaskRow = { state: string; output: { results?: { seed: number; fitness: number; distance: number; ticks: number }[]; hostname?: string } | null; host_label: string | null }

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

for (let gen = 1; gen <= GENERATIONS; gen++) {
  // Shrink the step size as the gait improves: big jumps early to explore, small ones
  // later so a working gait is refined rather than destroyed.
  const sigma = Math.max(0.02, 0.22 * Math.pow(0.985, gen))

  const tasks: Record<string, unknown>[] = []
  for (let offset = 0; offset < POPULATION; offset += PER_TASK) {
    const seeds: number[] = []
    for (let k = 0; k < PER_TASK && offset + k < POPULATION; k++) {
      // Seed 0 re-evaluates the parent unchanged, so the run can never go backwards.
      seeds.push(offset + k === 0 ? 0 : gen * 100_000 + offset + k)
    }
    tasks.push({ generation: gen, parent, sigma, seeds, steps: STEPS })
  }

  const res = await api('/jobs', { adapter: 'walker_evolution', tasks })
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
  for (const t of view.tasks) {
    if (t.state !== 'succeeded' || !t.output?.results) continue
    hosts[t.host_label ?? '?'] = (hosts[t.host_label ?? '?'] ?? 0) + t.output.results.length
    for (const r of t.output.results) {
      evaluations += 1
      if (!champion || r.fitness > champion.fitness) champion = r
    }
  }
  if (!champion) { console.error(`  generation ${gen}: nothing came back`); continue }

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
  console.log(`  gen ${String(gen).padStart(3)}  fitness ${champion.fitness.toFixed(2).padStart(7)}  ` +
    `distance ${String(champion.distance).padStart(6)}m  upright ${String(champion.ticks).padStart(4)}/${STEPS}  ${split}`)
}

console.log(`\n  done — ${evaluations} gait evaluations across the fleet in ` +
  `${Math.round((Date.now() - startedAt) / 1000)}s\n`)
