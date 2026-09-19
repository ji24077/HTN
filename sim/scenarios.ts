import type { Harness, Agent } from './harness.ts'
import { sleep } from './harness.ts'

export type Ctx = {
  h: Harness
  /** Record a passing or failing assertion. Never throws — a scenario reports all of its checks. */
  check(name: string, ok: boolean, detail?: string): void
  note(text: string): void
}

export type Scenario = {
  name: string
  what: string
  /** Why this matters in the real world. */
  why: string
  run(ctx: Ctx): Promise<void>
}

const allSucceeded = async (h: Harness, jobId: string): Promise<boolean> =>
  (await h.job(jobId)).job.status !== 'running'

/** Every input has exactly one accepted result, and nothing was done twice. */
async function assertExactlyOnce(ctx: Ctx, jobId: string): Promise<void> {
  const view = await ctx.h.job(jobId)
  const succeeded = view.tasks.filter(t => t.state === 'succeeded')
  const nonces = new Set(succeeded.map(t => (t.output as { nonce?: string } | null)?.nonce).filter(Boolean))

  ctx.check('every task reached a terminal state',
    view.tasks.every(t => ['succeeded', 'failed', 'cancelled'].includes(t.state)),
    view.tasks.filter(t => !['succeeded', 'failed', 'cancelled'].includes(t.state)).map(t => `${t.seq}:${t.state}`).join(' '))

  ctx.check('every task has exactly one accepted result',
    succeeded.length === view.job.total_items,
    `${succeeded.length}/${view.job.total_items} succeeded`)

  ctx.check('no result was counted twice',
    nonces.size === succeeded.length,
    `${nonces.size} distinct nonces for ${succeeded.length} results`)
}

export const scenarios: Scenario[] = [
  // ------------------------------------------------------------------------
  {
    name: 'happy-path',
    what: 'Two computers join over a clean network and finish a job.',
    why: 'The baseline. If this fails, nothing below means anything.',
    async run(ctx) {
      const a = await ctx.h.enroll('alpha')
      const b = await ctx.h.enroll('bravo')
      ctx.h.start(a); ctx.h.start(b)

      await ctx.h.waitFor('both hosts online', async () =>
        (await ctx.h.hosts()).filter(x => x.online).length === 2)
      ctx.check('both computers appear online', true)

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'each', count: 4, sleepMs: 50 })
      await ctx.h.waitFor('job finished', () => allSucceeded(ctx.h, jobId))
      await assertExactlyOnce(ctx, jobId)

      const view = await ctx.h.job(jobId)
      const hostsUsed = new Set(view.tasks.map(t => t.host_label))
      ctx.check('work ran on both computers', hostsUsed.size === 2, [...hostsUsed].join(', '))
    },
  },

  // ------------------------------------------------------------------------
  {
    name: 'high-latency',
    what: 'A friend on the other side of the world: 400 ms each way, 80 ms of jitter.',
    why: 'Lease renewals and heartbeats are time-sensitive. A slow link must not look like a dead one.',
    async run(ctx) {
      const a = await ctx.h.enroll('distant')
      ctx.h.net.set({ latencyMs: 400, jitterMs: 80 })
      ctx.h.start(a)

      await ctx.h.waitFor('host online despite latency', async () =>
        (await ctx.h.hosts()).some(x => x.online), 45_000)
      ctx.check('a high-latency computer still completes the handshake', true, '~800ms round trip')

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'each', count: 3, sleepMs: 100 })
      await ctx.h.waitFor('job finished over slow link', () => allSucceeded(ctx.h, jobId), 60_000)
      await assertExactlyOnce(ctx, jobId)

      const view = await ctx.h.job(jobId)
      ctx.check('latency did not cause spurious retries',
        view.tasks.every(t => t.attempts === 1),
        `attempts: ${view.tasks.map(t => t.attempts).join(',')}`)
    },
  },

  // ------------------------------------------------------------------------
  {
    name: 'flaky-wifi',
    what: 'A connection that keeps dropping mid-job, roughly every 5 seconds.',
    why: 'The single most common real failure. Work must finish anyway, and never be double-counted.',
    async run(ctx) {
      const a = await ctx.h.enroll('flaky')
      const b = await ctx.h.enroll('steady')
      ctx.h.start(a); ctx.h.start(b)
      await ctx.h.waitFor('both online', async () => (await ctx.h.hosts()).filter(x => x.online).length === 2)

      // Sized so the job genuinely outlives several disruptions: ~20s of work across
      // 4 slots, against a drop every 5s. An earlier version used work that finished in
      // under a second and "passed" without ever being disrupted at all.
      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'queue', count: 100, sleepMs: 800 })

      // Cut every live connection repeatedly while the job runs.
      let cuts = 0
      let socketsCut = 0
      const chaos = setInterval(() => {
        const n = ctx.h.net.cutAll()
        if (n > 0) { cuts += 1; socketsCut += n }
      }, 5_000)
      try {
        await ctx.h.waitFor('job finished despite drops', () => allSucceeded(ctx.h, jobId), 120_000)
      } finally {
        clearInterval(chaos)
      }
      ctx.note(`connections were cut ${cuts} times (${socketsCut} sockets) during the run`)
      ctx.check('the network was genuinely disrupted', cuts >= 2, `${cuts} cut rounds, ${socketsCut} sockets`)
      await assertExactlyOnce(ctx, jobId)

      const view = await ctx.h.job(jobId)
      const retried = view.tasks.filter(t => t.attempts > 1).length
      ctx.note(`${retried} task(s) needed a second attempt`)
    },
  },

  // ------------------------------------------------------------------------
  {
    name: 'laptop-sleep',
    what: 'A laptop that closes its lid: the connection is silently swallowed, with no close.',
    why: 'The nastiest case. Both sides still believe the socket is fine, so only heartbeats reveal it.',
    async run(ctx) {
      const a = await ctx.h.enroll('sleeper')
      ctx.h.start(a)
      await ctx.h.waitFor('online', async () => (await ctx.h.hosts()).some(x => x.online))

      ctx.h.net.set({ blackhole: true })
      ctx.note('link is now a black hole: packets vanish, no FIN, no error')

      await ctx.h.waitFor('server notices the silence', async () =>
        (await ctx.h.hosts()).every(x => !x.online), 40_000)
      ctx.check('a silently dead connection is detected and marked offline', true,
        'detected by heartbeat timeout, not by a close event')

      ctx.h.net.clear()
      await ctx.h.waitFor('host comes back on its own', async () =>
        (await ctx.h.hosts()).some(x => x.online), 60_000)
      ctx.check('the computer reconnects by itself once the network returns', true)

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'each', count: 2, sleepMs: 50 })
      await ctx.h.waitFor('job after recovery', () => allSucceeded(ctx.h, jobId), 60_000)
      await assertExactlyOnce(ctx, jobId)
    },
  },

  // ------------------------------------------------------------------------
  {
    name: 'sleep-mid-task',
    what: 'The lid closes while the computer is holding work.',
    why: 'Held work must return to the queue and be finished by someone else, exactly once.',
    async run(ctx) {
      const a = await ctx.h.enroll('holder')
      const b = await ctx.h.enroll('rescuer')
      ctx.h.start(a); ctx.h.start(b)
      await ctx.h.waitFor('both online', async () => (await ctx.h.hosts()).filter(x => x.online).length === 2)

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'queue', count: 16, sleepMs: 400 })
      await ctx.h.waitFor('work in flight', async () =>
        (await ctx.h.job(jobId)).tasks.some(t => ['leased', 'running', 'offered'].includes(t.state)))

      ctx.h.stop(a)
      ctx.note('one computer was killed outright while holding leased work')

      await ctx.h.waitFor('job still completes', () => allSucceeded(ctx.h, jobId), 120_000)
      await assertExactlyOnce(ctx, jobId)

      const view = await ctx.h.job(jobId)
      ctx.check('abandoned work was picked up by the other computer',
        view.tasks.some(t => t.attempts > 1),
        `${view.tasks.filter(t => t.attempts > 1).length} task(s) re-attempted`)
    },
  },

  // ------------------------------------------------------------------------
  {
    name: 'blocked-websockets',
    what: 'A café or office network that allows HTTPS but blocks WebSocket upgrades.',
    why: 'A real and common restriction. It must fail loudly and recover on its own, not hang.',
    async run(ctx) {
      const a = await ctx.h.enroll('behind-proxy')
      ctx.h.net.set({ rejectUpgrade: true })
      ctx.h.start(a)

      await ctx.h.waitFor('agent has tried and been refused', async () =>
        ctx.h.net.stats.upgradesRejected >= 2, 40_000)
      ctx.check('blocked upgrades are refused rather than hanging',
        ctx.h.net.stats.upgradesRejected >= 2, `${ctx.h.net.stats.upgradesRejected} refusals`)
      ctx.check('the computer never appears online while blocked',
        (await ctx.h.hosts()).every(x => !x.online))

      ctx.h.net.clear()
      ctx.note('restriction lifted')
      await ctx.h.waitFor('recovers once unblocked', async () =>
        (await ctx.h.hosts()).some(x => x.online), 90_000)
      ctx.check('the computer joins by itself once the network allows it', true)
    },
  },

  // ------------------------------------------------------------------------
  {
    name: 'server-unreachable',
    what: 'The host machine is asleep or offline when a friend tries to join.',
    why: 'Agents must wait patiently and back off, not spin or give up permanently.',
    async run(ctx) {
      const a = await ctx.h.enroll('early-bird')
      ctx.h.net.set({ refuseConnections: true })
      ctx.h.start(a)

      await ctx.h.waitFor('several refused dials', async () => ctx.h.net.stats.refused >= 3, 40_000)
      ctx.check('connection attempts are refused while the server is away',
        ctx.h.net.stats.refused >= 3, `${ctx.h.net.stats.refused} refusals`)

      ctx.h.net.clear()
      await ctx.h.waitFor('joins when the server returns', async () =>
        (await ctx.h.hosts()).some(x => x.online), 90_000)
      ctx.check('the computer joins as soon as the server is back', true)
    },
  },

  // ------------------------------------------------------------------------
  {
    name: 'control-restart',
    what: 'The host restarts the server while friends are connected and working.',
    why: 'Restarting your own laptop must not require every friend to re-run anything.',
    async run(ctx) {
      const a = await ctx.h.enroll('friend-1')
      const b = await ctx.h.enroll('friend-2')
      ctx.h.start(a); ctx.h.start(b)
      await ctx.h.waitFor('both online', async () => (await ctx.h.hosts()).filter(x => x.online).length === 2)

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'queue', count: 20, sleepMs: 300 })
      await ctx.h.waitFor('work in flight', async () =>
        (await ctx.h.job(jobId)).tasks.some(t => t.state !== 'pending'))

      await ctx.h.restartControl()
      ctx.note('server was killed and restarted mid-job')

      await ctx.h.waitFor('both rejoin without help', async () =>
        (await ctx.h.hosts()).filter(x => x.online).length === 2, 90_000)
      ctx.check('every computer rejoins automatically after a server restart', true)

      await ctx.h.waitFor('job completes after restart', () => allSucceeded(ctx.h, jobId), 120_000)
      await assertExactlyOnce(ctx, jobId)
    },
  },

  // ------------------------------------------------------------------------
  {
    name: 'nat-idle-timeout',
    what: 'A home router that quietly reclaims idle connections every 8 seconds.',
    why: 'Cheap routers do this. Without heartbeats the connection dies whenever a host is idle.',
    async run(ctx) {
      const a = await ctx.h.enroll('behind-nat')
      ctx.h.net.set({ idleTimeoutMs: 8_000 })
      ctx.h.start(a)
      await ctx.h.waitFor('online', async () => (await ctx.h.hosts()).some(x => x.online), 45_000)

      // Idle for longer than the router's patience, doing nothing at all.
      await sleep(20_000)
      const stillOnline = (await ctx.h.hosts()).some(x => x.online)
      ctx.check('an idle computer survives a router that drops idle connections', stillOnline,
        stillOnline ? 'heartbeats kept the mapping alive or it reconnected' : 'went offline and did not return')

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'each', count: 2, sleepMs: 50 })
      await ctx.h.waitFor('still able to run work', () => allSucceeded(ctx.h, jobId), 60_000)
      await assertExactlyOnce(ctx, jobId)
    },
  },

  // ------------------------------------------------------------------------
  {
    name: 'reconnect-storm',
    what: 'Six computers all lose the network at the same moment, then it returns.',
    why: 'If they all retry on the same schedule they arrive together and hammer the server.',
    async run(ctx) {
      const agents: Agent[] = []
      for (let i = 0; i < 6; i++) agents.push(await ctx.h.enroll(`storm-${i}`))
      for (const a of agents) ctx.h.start(a)
      await ctx.h.waitFor('all six online', async () =>
        (await ctx.h.hosts()).filter(x => x.online).length === 6, 60_000)

      ctx.h.net.cutAll()
      const cutAt = Date.now()
      ctx.note('all six connections cut simultaneously')

      await ctx.h.waitFor('all six back', async () =>
        (await ctx.h.hosts()).filter(x => x.online).length === 6, 90_000)

      // Reconnections should be spread out by jitter, not arrive in lockstep.
      const spreadMs = Date.now() - cutAt
      ctx.check('all six recover without help', true, `${spreadMs}ms to full recovery`)

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'each', count: 1, sleepMs: 20 })
      await ctx.h.waitFor('every computer takes work again', () => allSucceeded(ctx.h, jobId), 60_000)
      const view = await ctx.h.job(jobId)
      const distinct = new Set(view.tasks.map(t => t.host_label)).size
      // Report the task count alongside, because two very different faults both show up
      // here as "fewer than six" and only this number tells them apart. 'each' mode pins
      // one task per host that is online *at the moment the job is submitted*, so:
      //   N distinct across N tasks  -> the online set shrank between the wait above and
      //     the submit. Agents that miss a heartbeat under load drop out in that gap; it
      //     is a timing artifact, and it is what you see when two suites run at once.
      //   N distinct across 6 tasks  -> six tasks existed and the work did not spread.
      //     That is the real defect this scenario exists to catch.
      // Keep the assertion at six either way. Relaxing it to hide the first case would
      // throw away the only check that catches the second.
      ctx.check('all six computers accept work after recovery',
        distinct === 6,
        `${distinct} distinct hosts across ${view.job.total_items} tasks`)
    },
  },

  // ------------------------------------------------------------------------
  {
    name: 'slow-link',
    what: 'A very slow connection: 32 KB/s with 150 ms of latency.',
    why: 'Tethered phones and rural links. Work should be slower, not broken.',
    async run(ctx) {
      const a = await ctx.h.enroll('tethered')
      ctx.h.net.set({ bandwidthBps: 32_000, latencyMs: 150 })
      ctx.h.start(a)
      await ctx.h.waitFor('online on a slow link', async () =>
        (await ctx.h.hosts()).some(x => x.online), 60_000)
      ctx.check('a slow computer still joins', true)

      const jobId = await ctx.h.submitJob({ adapter: 'echo', mode: 'each', count: 4, sleepMs: 50 })
      await ctx.h.waitFor('job finished on a slow link', () => allSucceeded(ctx.h, jobId), 90_000)
      await assertExactlyOnce(ctx, jobId)
    },
  },
]
