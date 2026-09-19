/**
 * End-to-end scenarios for the iOS agent, against a real control service.
 *
 *   node ios/test/e2e.ts                  every scenario
 *   node ios/test/e2e.ts mixed-fleet      one scenario
 *   node ios/test/e2e.ts --list           what each covers and why
 *
 * Nothing here is mocked. Each scenario gets a fresh database, a real control service, a
 * real Postgres, real agent processes, and a simulated network between them. The Swift
 * agent under test is the same core the phone runs — the shell differs, the decisions do
 * not — which is what makes "is the connection reliable?" answerable without a phone.
 *
 * Requires: pnpm db:up, and `cd ios/DWPAgentKit && swift build`.
 */
import { existsSync, rmSync } from 'node:fs'
import { startHarness, sleep, type Harness, type Agent } from '../../sim/harness.ts'

const GREEN = '\x1b[32m', RED = '\x1b[31m', DIM = '\x1b[2m', BOLD = '\x1b[1m', RESET = '\x1b[0m'
const SWIFT_AGENT = 'ios/DWPAgentKit/.build/debug/dwpagent'

type Ctx = {
  h: Harness
  check(name: string, ok: boolean, detail?: string): void
  note(text: string): void
}

type Scenario = { name: string; what: string; why: string; run(ctx: Ctx): Promise<void> }

const finished = async (h: Harness, jobId: string): Promise<boolean> =>
  (await h.job(jobId)).job.status !== 'running'

const scenarios: Scenario[] = [

  // ----------------------------------------------------------------------
  {
    name: 'pair-and-connect',
    what: 'The iOS agent enrolls with a pairing code and appears online.',
    why: 'The first thing that has to work, and the first thing a stranger sees. '
       + 'It also proves the Swift key format is one the control service accepts.',
    async run(ctx) {
      const phone = await ctx.h.enroll('phone', 'swift')
      ctx.check('pairing produced a host id', Boolean(phone.hostId), phone.hostId)

      ctx.h.start(phone)
      await ctx.h.waitFor('phone online', async () =>
        (await ctx.h.hosts()).some(x => x.id === phone.hostId && x.online))
      ctx.check('the phone appears online', true)

      const { hosts } = await (await ctx.h.api('/hosts')).json() as {
        hosts: { id: string; os: string; arch: string; adapters: string[]; agent_version: string }[]
      }
      const row = hosts.find(x => x.id === phone.hostId)!
      // hello was parsed and stored: if the capability schema had rejected it, the row
      // would still be blank here and the host would sit online but never get work.
      ctx.check('its capability record was accepted and stored',
        Boolean(row.os) && Boolean(row.arch), `os=${row.os} arch=${row.arch}`)
      ctx.check('it advertises the adapters it can actually run',
        row.adapters?.includes('echo') && row.adapters?.includes('cpu_inference_batch'),
        (row.adapters ?? []).join(', '))
      ctx.check('it does not advertise browser work',
        !row.adapters?.includes('remote_browser_session'),
        'Playwright cannot exist on this platform')
      ctx.check('it reports the iOS agent version', row.agent_version?.includes('ios'), row.agent_version)
    },
  },

  // ----------------------------------------------------------------------
  {
    name: 'echo-signed-result',
    what: 'The phone runs a task and returns a result the server can verify.',
    why: 'This is the provenance claim the whole system rests on: a result the control '
       + 'service did not compute and cannot forge. If the Swift signature were wrong, '
       + 'the server would record it as a forgery rather than a bug.',
    async run(ctx) {
      const phone = await ctx.h.enroll('phone', 'swift')
      ctx.h.start(phone)
      await ctx.h.waitFor('phone online', async () =>
        (await ctx.h.hosts()).some(x => x.id === phone.hostId && x.online))

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'each', count: 3, sleepMs: 20 })
      await ctx.h.waitFor('job finished', () => finished(ctx.h, jobId))

      const view = await ctx.h.job(jobId)
      const succeeded = view.tasks.filter(t => t.state === 'succeeded')
      ctx.check('every task succeeded', succeeded.length === view.job.total_items,
        `${succeeded.length}/${view.job.total_items}`)

      const outputs = succeeded.map(t => t.output as { hostId: string; os: string; arch: string; nonce: string })
      ctx.check('the phone identified itself in its own output',
        outputs.every(o => o.hostId === phone.hostId), outputs[0]?.hostId)
      ctx.check('each result carries its own distinct nonce',
        new Set(outputs.map(o => o.nonce)).size === outputs.length)

      // The decisive one: the server verified each signature before accepting it, so a
      // `signature_invalid` event anywhere means the Swift attestation is wrong.
      const { events } = await (await ctx.h.api(`/jobs/${jobId}/events`)).json() as
        { events: { type: string }[] }
      ctx.check('the server recorded no invalid signatures',
        !events.some(e => e.type === 'result.signature_invalid'),
        events.filter(e => e.type === 'result.signature_invalid').length + ' invalid')
      ctx.check('the server recorded the results as succeeded',
        events.filter(e => e.type === 'task.succeeded').length === view.job.total_items)
    },
  },

  // ----------------------------------------------------------------------
  {
    name: 'inference-agreement',
    what: 'The phone and a laptop run the same model over the same digits, and agree.',
    why: 'The real question about mobile compute is not whether it runs but whether it is '
       + 'right. Two different ONNX Runtime builds on two different chips must produce the '
       + 'same predictions, or a distributed result is worthless.',
    async run(ctx) {
      if (!existsSync('fixtures/manifest.json')) {
        ctx.note('no fixtures — run: node scripts/build-fixtures.ts')
        return
      }
      const phone = await ctx.h.enroll('phone', 'swift')
      const laptop = await ctx.h.enroll('laptop', 'node')
      ctx.h.start(phone); ctx.h.start(laptop)
      await ctx.h.waitFor('both online', async () =>
        (await ctx.h.hosts()).filter(x => x.online).length === 2)

      // Inference jobs are cut into slices by count/batchSize, not by host, so the way
      // to get the same digits onto both machines is one pinned job each. Identical
      // inputs, identical model, two different chips — which is the comparison worth
      // making and never how real work would be scheduled.
      const slice = { adapter: 'cpu_inference_batch', count: 200, batchSize: 200 }
      const phoneJob = await ctx.h.submitJob({ ...slice, hostId: phone.hostId })
      const laptopJob = await ctx.h.submitJob({ ...slice, hostId: laptop.hostId })

      await ctx.h.waitFor('phone finished its slice', () => finished(ctx.h, phoneJob), 240_000)
      await ctx.h.waitFor('laptop finished its slice', () => finished(ctx.h, laptopJob), 240_000)

      type Out = {
        from: number; count: number; predictions: number[]
        correct: number; logitChecksum: number; itemsPerSecond: number; modelLoadMs: number
      }
      const readOne = async (jobId: string): Promise<Out | null> => {
        const view = await ctx.h.job(jobId)
        const done = view.tasks.filter(t => t.state === 'succeeded')
        return done.length === 1 ? done[0]!.output as Out : null
      }

      const phoneResult = await readOne(phoneJob)
      const laptopResult = await readOne(laptopJob)

      ctx.check('both hosts finished their slice',
        phoneResult !== null && laptopResult !== null,
        `phone=${phoneResult ? 'ok' : 'missing'} laptop=${laptopResult ? 'ok' : 'missing'}`)
      if (!phoneResult || !laptopResult) return

      ctx.check('they were given the same slice',
        phoneResult.from === laptopResult.from && phoneResult.count === laptopResult.count,
        `[${phoneResult.from}, ${phoneResult.from + phoneResult.count})`)

      ctx.check('the phone classified every item it was given',
        phoneResult.predictions.length === phoneResult.count,
        `${phoneResult.predictions.length} predictions`)

      // MNIST test accuracy for this model is about 98%. Anything near chance means the
      // preprocessing or the tensor layout is wrong, not that the model is bad — and it
      // would still produce a perfectly well-signed, perfectly wrong result.
      const accuracy = phoneResult.correct / phoneResult.count
      ctx.check('the phone is as accurate as the model should be', accuracy > 0.9,
        `${(accuracy * 100).toFixed(1)}% on ${phoneResult.count} digits`)

      const agree = phoneResult.predictions.every((p, i) => p === laptopResult.predictions[i])
      ctx.check('the phone and the laptop predict identically', agree,
        agree ? `all ${phoneResult.count} items` : 'predictions diverge')

      // Floating-point CPU kernels are not bit-identical across microarchitectures, so
      // the gate is a published tolerance rather than equality — the same rule the
      // architecture applies to any two hosts (§6.1).
      const delta = Math.abs(phoneResult.logitChecksum - laptopResult.logitChecksum)
      const tolerance = Math.max(0.05, Math.abs(laptopResult.logitChecksum) * 1e-4)
      ctx.check('their logit checksums agree within tolerance', delta <= tolerance,
        `|Δ| = ${delta.toFixed(4)} against a tolerance of ${tolerance.toFixed(4)}`)

      ctx.note(`throughput — iOS core ${phoneResult.itemsPerSecond}/s, `
             + `Node agent ${laptopResult.itemsPerSecond}/s`)
      ctx.note(`model load — iOS core ${phoneResult.modelLoadMs}ms, `
             + `Node agent ${laptopResult.modelLoadMs}ms`)
    },
  },

  // ----------------------------------------------------------------------
  {
    name: 'mixed-fleet',
    what: 'A phone and two laptops share one queue of work.',
    why: 'A phone is only useful if it is an ordinary member of the fleet. The scheduler '
       + 'must not need to know which host is which.',
    async run(ctx) {
      const phone = await ctx.h.enroll('phone', 'swift')
      const a = await ctx.h.enroll('laptop-a', 'node')
      const b = await ctx.h.enroll('laptop-b', 'node')
      for (const agent of [phone, a, b]) ctx.h.start(agent)
      await ctx.h.waitFor('all three online', async () =>
        (await ctx.h.hosts()).filter(x => x.online).length === 3)

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'queue', count: 30, sleepMs: 40 })
      await ctx.h.waitFor('queue drained', () => finished(ctx.h, jobId), 120_000)

      const view = await ctx.h.job(jobId)
      const succeeded = view.tasks.filter(t => t.state === 'succeeded')
      ctx.check('every item was completed exactly once',
        succeeded.length === view.job.total_items,
        `${succeeded.length}/${view.job.total_items}`)

      const byHost = new Map<string, number>()
      for (const t of succeeded) byHost.set(t.host_label ?? '?', (byHost.get(t.host_label ?? '?') ?? 0) + 1)
      ctx.check('the phone took a share of the queue', (byHost.get('phone') ?? 0) > 0,
        [...byHost].map(([k, v]) => `${k}:${v}`).join(' '))
      ctx.check('all three hosts contributed', byHost.size === 3,
        [...byHost.keys()].join(', '))
    },
  },

  // ----------------------------------------------------------------------
  {
    name: 'reconnect-after-drop',
    what: 'The phone loses its connection and comes back by itself.',
    why: 'This is the normal case on a phone, not the exceptional one: it happens every '
       + 'time the device changes network, sleeps, or walks out of range.',
    async run(ctx) {
      const phone = await ctx.h.enroll('phone', 'swift')
      ctx.h.start(phone)
      await ctx.h.waitFor('phone online', async () =>
        (await ctx.h.hosts()).some(x => x.id === phone.hostId && x.online))

      // Cut every connection at the network, without telling either end.
      ctx.h.net.set({ blackhole: true })
      await ctx.h.waitFor('server notices the silence', async () =>
        !(await ctx.h.hosts()).find(x => x.id === phone.hostId)!.online, 40_000)
      ctx.check('the server marks the phone offline when it goes quiet', true)

      ctx.h.net.clear()
      await ctx.h.waitFor('phone reconnects on its own', async () =>
        (await ctx.h.hosts()).find(x => x.id === phone.hostId)!.online, 60_000)
      ctx.check('the phone reconnects without being touched', true)

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'each', count: 2 })
      await ctx.h.waitFor('work resumes', () => finished(ctx.h, jobId), 60_000)
      const view = await ctx.h.job(jobId)
      ctx.check('it takes work again after reconnecting',
        view.tasks.filter(t => t.state === 'succeeded').length === view.job.total_items)
    },
  },

  // ----------------------------------------------------------------------
  {
    name: 'task-reclaimed-when-phone-vanishes',
    what: 'The phone is killed mid-task; a laptop finishes the work.',
    why: 'A phone disappears as a matter of course — a call arrives, the screen locks, iOS '
       + 'suspends the app. The job must not notice.',
    async run(ctx) {
      const phone = await ctx.h.enroll('phone', 'swift')
      const laptop = await ctx.h.enroll('laptop', 'node')
      ctx.h.start(phone); ctx.h.start(laptop)
      await ctx.h.waitFor('both online', async () =>
        (await ctx.h.hosts()).filter(x => x.online).length === 2)

      // Long enough that killing the phone lands in the middle of its task.
      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'queue', count: 8, sleepMs: 1500 })
      await sleep(1200)
      ctx.h.stop(phone)
      ctx.note('phone killed mid-task (SIGKILL — no goodbye, exactly like iOS suspending it)')

      await ctx.h.waitFor('the laptop finishes the job alone', () => finished(ctx.h, jobId), 120_000)
      const view = await ctx.h.job(jobId)
      const succeeded = view.tasks.filter(t => t.state === 'succeeded')
      ctx.check('every item still completed', succeeded.length === view.job.total_items,
        `${succeeded.length}/${view.job.total_items}`)
      ctx.check('the abandoned work was retried, not lost',
        view.tasks.some(t => t.attempts > 1),
        `max attempts: ${Math.max(...view.tasks.map(t => t.attempts))}`)
    },
  },

  // ----------------------------------------------------------------------
  {
    name: 'server-restart',
    what: 'The control service restarts underneath a connected phone.',
    why: 'Deploys happen. An agent that needs a human to restart it after every one is '
       + 'not something you can leave on a friend\'s device.',
    async run(ctx) {
      const phone = await ctx.h.enroll('phone', 'swift')
      ctx.h.start(phone)
      await ctx.h.waitFor('phone online', async () =>
        (await ctx.h.hosts()).some(x => x.id === phone.hostId && x.online))

      await ctx.h.restartControl()
      ctx.note('control service killed and restarted')

      await ctx.h.waitFor('phone reconnects to the new process', async () =>
        (await ctx.h.hosts()).find(x => x.id === phone.hostId)?.online === true, 90_000)
      ctx.check('the phone reconnects after a restart', true)

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'each', count: 2 })
      await ctx.h.waitFor('work flows again', () => finished(ctx.h, jobId), 60_000)
      const view = await ctx.h.job(jobId)
      ctx.check('it is a working host again, not merely a connected one',
        view.tasks.filter(t => t.state === 'succeeded').length === view.job.total_items)
    },
  },

  // ----------------------------------------------------------------------
  {
    name: 'flaky-wifi',
    what: 'A connection that resets at random, with latency and jitter.',
    why: 'Café wifi, a train, a phone at the edge of a room. The agent must keep making '
       + 'progress rather than back off into uselessness.',
    async run(ctx) {
      const phone = await ctx.h.enroll('phone', 'swift')
      ctx.h.start(phone)
      await ctx.h.waitFor('phone online', async () =>
        (await ctx.h.hosts()).some(x => x.id === phone.hostId && x.online), 60_000)

      ctx.h.net.set({ latencyMs: 250, jitterMs: 120, resetProbability: 0.02 })
      ctx.note('250 ms latency, 120 ms jitter, 2% of writes reset the connection')

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'queue', count: 10, sleepMs: 50 })
      await ctx.h.waitFor('job completes despite the link', () => finished(ctx.h, jobId), 180_000)

      const view = await ctx.h.job(jobId)
      const succeeded = view.tasks.filter(t => t.state === 'succeeded').length
      ctx.check('the job still completed', succeeded === view.job.total_items,
        `${succeeded}/${view.job.total_items}`)

      ctx.h.net.clear()
      await ctx.h.waitFor('phone is healthy again', async () =>
        (await ctx.h.hosts()).find(x => x.id === phone.hostId)?.online === true, 60_000)
      ctx.check('it recovers to a clean state afterwards', true)
    },
  },

  // ----------------------------------------------------------------------
  {
    name: 'revocation-is-final',
    what: 'The owner revokes the phone; it stops and cannot come back.',
    why: 'Revocation is the answer to a lost or stolen phone, so it has to be immediate '
       + 'and permanent rather than advisory.',
    async run(ctx) {
      const phone = await ctx.h.enroll('phone', 'swift')
      ctx.h.start(phone)
      await ctx.h.waitFor('phone online', async () =>
        (await ctx.h.hosts()).some(x => x.id === phone.hostId && x.online))

      const res = await ctx.h.api(`/hosts/${phone.hostId}/revoke`, {})
      ctx.check('the revoke call succeeded', res.ok, `HTTP ${res.status}`)

      await ctx.h.waitFor('phone goes offline', async () => {
        const host = (await ctx.h.hosts()).find(x => x.id === phone.hostId)
        return !host || !host.online
      }, 60_000)
      ctx.check('the phone is dropped', true)

      // The real test is that it cannot dial back in: its assertion no longer verifies
      // against a key the server will look up.
      await sleep(8000)
      const host = (await ctx.h.hosts()).find(x => x.id === phone.hostId)
      ctx.check('it stays out after repeated retries', !host || !host.online,
        'a revoked key is refused at the upgrade, before any session exists')
    },
  },

  // ----------------------------------------------------------------------
  {
    name: 'paused-phone-declines',
    what: 'A paused phone stays connected but refuses work.',
    why: 'The owner\'s kill switch has to work without disconnecting — someone who pauses '
       + 'their phone should still see it in the fleet, just idle.',
    async run(ctx) {
      const phone = await ctx.h.enroll('phone', 'swift')
      const laptop = await ctx.h.enroll('laptop', 'node')
      ctx.h.start(phone); ctx.h.start(laptop)
      await ctx.h.waitFor('both online', async () =>
        (await ctx.h.hosts()).filter(x => x.online).length === 2)

      // Pause through the same local file the app's toggle writes.
      const { writeFileSync } = await import('node:fs')
      const { join } = await import('node:path')
      writeFileSync(join(phone.home, 'paused'), new Date().toISOString())
      ctx.note('phone paused via its local kill switch')

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'queue', count: 6, sleepMs: 30 })
      await ctx.h.waitFor('the laptop absorbs the work', () => finished(ctx.h, jobId), 90_000)

      const view = await ctx.h.job(jobId)
      ctx.check('the job still completed', 
        view.tasks.filter(t => t.state === 'succeeded').length === view.job.total_items)
      ctx.note('pause on this platform is the app\'s own toggle; the file is the test hook')
    },
  },

  // ----------------------------------------------------------------------
  {
    name: 'unfit-device-withdraws',
    what: 'An overheating phone takes itself out of the rotation instead of declining.',
    why: 'Declining offer by offer is a hot loop: the server releases the task and '
       + 're-offers it at once, so an unfit device spins, hammers the control service, '
       + 'and burns the battery it was trying to save. Withdrawing costs one message.',
    async run(ctx) {
      const laptop = await ctx.h.enroll('laptop', 'node')
      const phone = await ctx.h.enroll('phone', 'swift')
      ctx.h.start(laptop)

      // The agent reads this at spawn; there is no way to make a test machine genuinely
      // overheat, and the behaviour is worth proving.
      process.env.DWP_FORCE_UNFIT = '1'
      ctx.h.start(phone)
      delete process.env.DWP_FORCE_UNFIT
      ctx.note('phone started reporting thermal=critical')

      await ctx.h.waitFor('both online', async () =>
        (await ctx.h.hosts()).filter(x => x.online).length === 2, 60_000)

      // The server's own view: an unfit phone must read as paused, which is what takes
      // it out of dispatch entirely.
      await ctx.h.waitFor('the server sees the phone as paused', async () => {
        const { hosts } = await (await ctx.h.api('/hosts')).json() as
          { hosts: { id: string; paused: boolean }[] }
        return hosts.find(x => x.id === phone.hostId)?.paused === true
      }, 40_000)
      ctx.check('an unfit phone withdraws from the rotation', true)

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'queue', count: 6, sleepMs: 30 })
      await ctx.h.waitFor('the laptop does the work', () => finished(ctx.h, jobId), 90_000)

      const view = await ctx.h.job(jobId)
      const succeeded = view.tasks.filter(t => t.state === 'succeeded')
      ctx.check('the job still completed', succeeded.length === view.job.total_items,
        `${succeeded.length}/${view.job.total_items}`)
      ctx.check('none of it ran on the unfit phone',
        succeeded.every(t => t.host_label !== 'phone'),
        [...new Set(succeeded.map(t => t.host_label))].join(', '))

      // The decisive one: withdrawal must be quiet. A decline loop would show up here as
      // dozens of offers for a six-item job.
      const { events } = await (await ctx.h.api(`/jobs/${jobId}/events`)).json() as
        { events: { type: string }[] }
      const offers = events.filter(e => e.type === 'task.offered').length
      ctx.check('it did not spin the server with repeated offers', offers <= view.job.total_items * 2,
        `${offers} offers for ${view.job.total_items} items`)
    },
  },

  // ----------------------------------------------------------------------
  {
    name: 'walker-cross-host-agreement',
    what: 'A phone and a laptop score the same candidate gaits, and agree exactly.',
    why: 'The architecture treats cross-host agreement as a correctness check (§6.1). That '
       + 'only means something if two hosts genuinely compute the same thing — and before '
       + 'the shared deterministic math they did not: one ulp of difference in tanh '
       + 'compounded over 1800 physics steps into 33 fitness points and six metres, one '
       + 'stickman walking while the other fell. That would have read as evolutionary '
       + 'noise forever. This is the test that would have caught it.',
    async run(ctx) {
      const phone = await ctx.h.enroll('phone', 'swift')
      const laptop = await ctx.h.enroll('laptop', 'node')
      ctx.h.start(phone); ctx.h.start(laptop)
      await ctx.h.waitFor('both online', async () =>
        (await ctx.h.hosts()).filter(x => x.online).length === 2, 60_000)

      const { randomGenome, perturb, evaluate } =
        await import('@dwp/protocol/walker.js') as {
          randomGenome: (s: number) => number[]
          perturb: (p: number[], s: number, seed: number) => number[]
          evaluate: (g: number[], steps: number) => {
            fitness: number; distance: number; ticks: number; fell: boolean }
        }

      // A gait that survives the whole run. An untrained one falls in under a second,
      // which is exactly the regime where the bug stayed invisible.
      const STEPS = 1800
      let parent = randomGenome(1)
      let bestScore = evaluate(parent, STEPS)
      for (let s = 2; s <= 40; s++) {
        const c = randomGenome(s); const r = evaluate(c, STEPS)
        if (r.fitness > bestScore.fitness) { bestScore = r; parent = c }
      }
      for (let gen = 1; gen <= 40; gen++) {
        const sigma = Math.max(0.02, 0.22 * Math.pow(0.985, gen))
        for (let k = 1; k <= 40; k++) {
          const g = perturb(parent, sigma, gen * 100_000 + k)
          const r = evaluate(g, STEPS)
          if (r.fitness > bestScore.fitness) { bestScore = r; parent = g }
        }
      }
      ctx.note(`reference gait: fitness ${bestScore.fitness}, ${bestScore.distance}m, `
             + `${bestScore.ticks}/${STEPS} upright`)

      // Without this the test can pass while proving nothing. A gait that falls after a
      // few hundred steps agrees across platforms even with the platform's own libm —
      // divergence needs time to compound, so a short-lived reference makes every
      // assertion below vacuous rather than false.
      ctx.check('the reference gait survives long enough for divergence to show',
        bestScore.ticks > STEPS * 0.8,
        `${bestScore.ticks}/${STEPS} upright`)

      const seeds = [0, 900_001, 900_002, 900_003, 900_004, 900_005, 900_006, 900_007]
      const task = { generation: 1, parent, sigma: 0.05, seeds, steps: STEPS }

      // The same slice, pinned to each host in turn.
      const run = async (hostId: string): Promise<{ seed: number; fitness: number
                                                    distance: number; ticks: number; fell: boolean }[]> => {
        const jobId = await ctx.h.submitJob({ adapter: 'walker_evolution', tasks: [task], hostId })
        await ctx.h.waitFor(`walker slice on ${hostId.slice(0, 8)}`,
          () => finished(ctx.h, jobId), 240_000)
        const view = await ctx.h.job(jobId)
        const done = view.tasks.filter(t => t.state === 'succeeded')
        return done[0]
          ? (done[0]!.output as { results: { seed: number; fitness: number
              distance: number; ticks: number; fell: boolean }[] }).results
          : []
      }

      const phoneResults = await run(phone.hostId)
      const laptopResults = await run(laptop.hostId)

      ctx.check('both hosts returned a full slice',
        phoneResults.length === seeds.length && laptopResults.length === seeds.length,
        `phone ${phoneResults.length}, laptop ${laptopResults.length} of ${seeds.length}`)
      if (phoneResults.length !== seeds.length || laptopResults.length !== seeds.length) return

      let identical = 0
      let worstFitness = 0
      let survivors = 0
      for (let i = 0; i < seeds.length; i++) {
        const a = phoneResults[i]!, b = laptopResults[i]!
        if (a.fitness === b.fitness && a.distance === b.distance
            && a.ticks === b.ticks && a.fell === b.fell) identical += 1
        if (a.ticks > STEPS * 0.8) survivors += 1
        worstFitness = Math.max(worstFitness, Math.abs(a.fitness - b.fitness))
      }

      ctx.check('the phone and the laptop score every gait identically',
        identical === seeds.length, `${identical}/${seeds.length} exact, ${survivors} long survivors`)
      ctx.check('not one fitness point of drift', worstFitness === 0,
        `max |Δ fitness| = ${worstFitness}`)

      // And the scores must match what the driver computes locally, or the winner the
      // browser replays is not the winner that was scored.
      let matchesReference = 0
      for (const r of phoneResults) {
        const local = evaluate(perturb(parent, 0.05, r.seed), STEPS)
        if (local.fitness === r.fitness && local.ticks === r.ticks) matchesReference += 1
      }
      ctx.check('and both match the reference implementation the browser replays with',
        matchesReference === seeds.length, `${matchesReference}/${seeds.length}`)
    },
  },
]

// ---------------------------------------------------------------------------

const args = process.argv.slice(2)

if (args.includes('--list')) {
  console.log(`\n${BOLD}iOS agent scenarios${RESET}\n`)
  for (const s of scenarios) console.log(`  ${BOLD}${s.name}${RESET}\n    ${s.what}\n    ${DIM}${s.why}${RESET}\n`)
  process.exit(0)
}

if (!existsSync(SWIFT_AGENT)) {
  console.error(`\nNo Swift agent at ${SWIFT_AGENT}\n  Build it:  cd ios/DWPAgentKit && swift build\n`)
  process.exit(1)
}

const wanted = args.filter(a => !a.startsWith('--'))
const selected = wanted.length ? scenarios.filter(s => wanted.includes(s.name)) : scenarios
if (!selected.length) {
  console.error(`No scenario matched: ${wanted.join(', ')}\nTry --list`)
  process.exit(1)
}

// Compressed timings, as the network suite uses. The mechanisms are the same at any
// scale; only the clock changes.
process.env.DWP_HEARTBEAT_SECONDS = '2'
process.env.DWP_OFFLINE_AFTER_SECONDS = '6'
process.env.DWP_LEASE_SECONDS = '5'

let totalPassed = 0, totalFailed = 0
const summary: { name: string; passed: number; failed: number; ms: number; error?: string }[] = []

for (const scenario of selected) {
  const logDir = `.dwp/ios-logs/${scenario.name}`
  try { rmSync(logDir, { recursive: true, force: true }) } catch {}

  console.log(`\n${BOLD}${scenario.name}${RESET}  ${DIM}${scenario.what}${RESET}`)
  const started = Date.now()
  let passed = 0, failed = 0, error: string | undefined

  const ctx: Ctx = {
    h: null as never,
    check(name, ok, detail) {
      if (ok) { passed += 1; console.log(`    ${GREEN}ok${RESET}   ${name}${detail ? `  ${DIM}${detail}${RESET}` : ''}`) }
      else { failed += 1; console.log(`    ${RED}FAIL${RESET} ${name}${detail ? `  ${RED}${detail}${RESET}` : ''}`) }
    },
    note(text) { console.log(`    ${DIM}·    ${text}${RESET}`) },
  }

  let harness: Harness | undefined
  try {
    harness = await startHarness({ logDir, controlPort: 8893 })
    ctx.h = harness
    await scenario.run(ctx)
  } catch (err) {
    error = err instanceof Error ? err.message : String(err)
    failed += 1
    console.log(`    ${RED}ERROR${RESET} ${error}`)
  } finally {
    try { await harness?.teardown() } catch {}
  }

  totalPassed += passed; totalFailed += failed
  summary.push({ name: scenario.name, passed, failed, ms: Date.now() - started, error })
}

console.log(`\n${BOLD}Summary${RESET}`)
for (const r of summary) {
  const mark = r.failed === 0 ? `${GREEN}pass${RESET}` : `${RED}fail${RESET}`
  console.log(`  ${mark}  ${r.name.padEnd(34)} ${r.passed} ok, ${r.failed} failed  ${DIM}${(r.ms / 1000).toFixed(1)}s${RESET}`)
}
console.log(`\n${totalPassed} passed, ${totalFailed} failed\n`)
process.exit(totalFailed === 0 ? 0 : 1)
