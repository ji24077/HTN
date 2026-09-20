/**
 * What this machine is willing to do, and how much of itself it will give.
 *
 * The window used to offer exactly one lever — pause — which is the difference between
 * lending a machine and not lending it. Everything between those two states had to be
 * expressed by turning the agent off when you wanted your laptop back, and turning it on
 * again when you remembered. That is not a limit, it is a manual switch.
 *
 * Every rule here is enforced at the moment an offer arrives, in the same place the agent
 * already decided whether it was eligible. That matters more than it sounds: it is the
 * only point where the decision is local, works offline, and cannot be overridden by the
 * server. A limit that depends on the control plane agreeing to honour it is a request,
 * not a limit.
 *
 * `decide` is pure and takes a snapshot of the world, so the rules can be tested without
 * a clock, a load average, or a history file. `gather` in the agent builds that snapshot
 * from the real ones.
 */

import { execFileSync } from 'node:child_process'
import { readdirSync, readFileSync } from 'node:fs'
import { cpus, loadavg, platform } from 'node:os'
import { busyMsSince } from './history.ts'
import { freeRamMb, isContainer } from './runtime.ts'

/** Days as JavaScript numbers them: 0 is Sunday. */
export type Weekday = 0 | 1 | 2 | 3 | 4 | 5 | 6

export type BudgetWindow = 'hour' | 'day'

export type Limits = {
  /**
   * Adapter ids this machine will accept. Absent means "whatever it can run".
   *
   * This one is not merely enforced here: it is also what the agent advertises at
   * handshake, and the scheduler filters offers on it (`spec->>'kind'=ANY(caps.kinds)`).
   * So a disallowed workload is never offered in the first place, and the check below is
   * the belt to that braces — it catches the window between changing the setting and the
   * reconnect that republishes it.
   */
  workloads?: string[]
  /** Ceiling on compute time in a rolling window. */
  budget?: { minutes: number; per: BudgetWindow }
  /** Local-time hours when work is accepted at all. */
  schedule?: { from: string; to: string; days?: Weekday[] }
  /** Refuse work when the machine is already under strain. */
  pressure?: {
    /** One-minute load average divided by cores. 1.0 means "fully committed". */
    maxLoadPerCore?: number
    minFreeRamMb?: number
    /** Host installs only; a container cannot see a battery. */
    notOnBattery?: boolean
  }
}

/** The world, as the rules need to see it. */
export type Conditions = {
  now: Date
  adapter: string
  /** What this machine is physically capable of, before any choice is applied. */
  available: string[]
  /** Compute milliseconds already spent in the named rolling window. */
  busyMs: (per: BudgetWindow) => number
  /** One-minute load average per core, or null where it cannot be read. */
  loadPerCore: number | null
  freeRamMb: number
  /** true on battery, false on mains, null where it cannot be known. */
  onBattery: boolean | null
}

export type Decision =
  | { ok: true }
  /**
   * `reason` is a stable tag for logs and tests; `detail` is the sentence a person reads
   * in the window. Keeping them separate stops the two drifting, which is what happens
   * when a log line is parsed out of a message written for a human.
   */
  | { ok: false; reason: 'workload' | 'schedule' | 'budget' | 'load' | 'memory' | 'battery'; detail: string }

/** Minutes past local midnight, or null if this is not "HH:MM". */
export function minutesOfDay(clock: string): number | null {
  const m = /^(\d{1,2}):(\d{2})$/.exec(clock.trim())
  if (!m) return null
  const hours = Number(m[1])
  const minutes = Number(m[2])
  if (hours > 23 || minutes > 59) return null
  return hours * 60 + minutes
}

/**
 * Is `now` inside the window?
 *
 * The wrap-around case is the one that matters and the one that is easy to get wrong:
 * "22:00 to 08:00" is the *useful* setting — lend the machine overnight — and naive
 * `from <= t && t < to` makes it match nothing at all, so the machine silently never
 * works and its owner concludes the feature is broken.
 *
 * `from === to` is treated as the whole day rather than as an empty window, because a
 * zero-width window is never what someone meant and "always" is the safe reading of it.
 */
export function withinSchedule(schedule: NonNullable<Limits['schedule']>, now: Date): boolean {
  const days = schedule.days
  if (days && days.length > 0 && !days.includes(now.getDay() as Weekday)) return false

  const from = minutesOfDay(schedule.from)
  const to = minutesOfDay(schedule.to)
  // An unreadable window must not silently stop the machine working.
  if (from === null || to === null) return true
  if (from === to) return true

  const t = now.getHours() * 60 + now.getMinutes()
  return from < to ? t >= from && t < to : t >= from || t < to
}

/** Which adapters this machine will accept, given what it can run and what was chosen. */
export function allowedWorkloads(limits: Limits | undefined, available: string[]): string[] {
  const chosen = limits?.workloads
  if (!chosen) return available
  const allowed = available.filter(a => chosen.includes(a))
  /**
   * Never advertise an empty set.
   *
   * `kinds` has `min_length=1` on the server, so a handshake with none is rejected
   * outright and the machine cannot connect at all — it would look like a broken agent
   * rather than like a machine that has been asked to run nothing. Turning everything off
   * is what Pause is for, and it says so in the window.
   */
  return allowed.length > 0 ? allowed : available
}

const plural = (n: number, one: string): string => `${n} ${one}${n === 1 ? '' : 's'}`

export function decide(limits: Limits | undefined, c: Conditions): Decision {
  if (!limits) return { ok: true }

  if (limits.workloads && !allowedWorkloads(limits, c.available).includes(c.adapter)) {
    return { ok: false, reason: 'workload', detail: `this machine is not set to run ${c.adapter}` }
  }

  if (limits.schedule && !withinSchedule(limits.schedule, c.now)) {
    return {
      ok: false,
      reason: 'schedule',
      detail: `outside the hours you set (${limits.schedule.from}–${limits.schedule.to})`,
    }
  }

  if (limits.budget && limits.budget.minutes > 0) {
    const spentMs = c.busyMs(limits.budget.per)
    const capMs = limits.budget.minutes * 60_000
    if (spentMs >= capMs) {
      return {
        ok: false,
        reason: 'budget',
        detail: `used this ${limits.budget.per}'s ${plural(limits.budget.minutes, 'minute')}`
          + ` of compute (${Math.round(spentMs / 60_000)} so far)`,
      }
    }
  }

  const pressure = limits.pressure
  if (pressure) {
    /**
     * Only refuse on a reading we actually have. `loadPerCore` is null where the
     * platform cannot answer — Windows reports a permanent zero from `loadavg()`, which
     * would read as a completely idle machine and make this guard silently inert rather
     * than absent. Null says so honestly and the rule stands aside.
     */
    if (pressure.maxLoadPerCore !== undefined && c.loadPerCore !== null
        && c.loadPerCore > pressure.maxLoadPerCore) {
      return {
        ok: false,
        reason: 'load',
        detail: `this machine is busy already (load ${c.loadPerCore.toFixed(2)} per core,`
          + ` limit ${pressure.maxLoadPerCore.toFixed(2)})`,
      }
    }
    if (pressure.minFreeRamMb !== undefined && c.freeRamMb < pressure.minFreeRamMb) {
      return {
        ok: false,
        reason: 'memory',
        detail: `only ${c.freeRamMb} MB free, and you asked to keep ${pressure.minFreeRamMb} MB`,
      }
    }
    if (pressure.notOnBattery && c.onBattery === true) {
      return { ok: false, reason: 'battery', detail: 'running on battery' }
    }
  }

  return { ok: true }
}

/**
 * Why this machine is not taking work *right now*, independent of any offer.
 *
 * The window has to be able to say "waiting until 22:00" while nothing is happening.
 * Asking `decide` needs an adapter, and there is no offer in hand — so this asks the
 * rules that do not depend on one.
 */
export function standingReason(limits: Limits | undefined, c: Omit<Conditions, 'adapter'>): Decision {
  return decide(limits, { ...c, adapter: '\u0000none' } as Conditions & { adapter: string })
}

// --------------------------------------------------------- reading the real world


const WINDOW_MS: Record<BudgetWindow, number> = { hour: 3_600_000, day: 86_400_000 }

/**
 * One-minute load average per core, or null where the figure is meaningless.
 *
 * Windows has no load average and Node reports a constant zero there. Passing that on
 * as a real reading would make a load guard permanently satisfied — switched on in the
 * window, doing nothing — which is worse than not offering it.
 *
 * In a container this reads the *host's* load, because /proc is the host's. That is the
 * right number anyway: the point of the guard is to leave the physical machine usable.
 */
export function loadPerCore(): number | null {
  if (platform() === 'win32') return null
  try {
    const cores = cpus().length || 1
    const one = loadavg()[0]
    return one !== undefined && Number.isFinite(one) ? one / cores : null
  } catch {
    return null
  }
}

/**
 * Is this machine on battery? null where that cannot be known.
 *
 * Cached, because the honest answer on macOS costs a subprocess and this is consulted on
 * every offer. Thirty seconds is far shorter than the time it takes anyone to notice a
 * charger has been unplugged, and it keeps a busy agent from spawning `pmset` in a loop.
 */
let batteryAnswer: { at: number; onBattery: boolean | null } | null = null

export function onBattery(): boolean | null {
  const now = Date.now()
  if (batteryAnswer && now - batteryAnswer.at < 30_000) return batteryAnswer.onBattery
  const answer = readBattery()
  batteryAnswer = { at: now, onBattery: answer }
  return answer
}

function readBattery(): boolean | null {
  // A container is given no power supply to look at, and spawning pmset there would
  // fail on every call. Say "cannot know" rather than guessing "on mains".
  if (isContainer()) return null
  try {
    if (platform() === 'linux') {
      const root = '/sys/class/power_supply'
      for (const name of readdirSync(root)) {
        // `online` is 1 when the mains adapter is connected. Absent means no adapter.
        if (/^(AC|ADP|ACAD)/i.test(name)) {
          const online = readFileSync(`${root}/${name}/online`, 'utf8').trim()
          return online === '0'
        }
      }
      return null
    }
    if (platform() === 'darwin') {
      const out = execFileSync('pmset', ['-g', 'batt'], { encoding: 'utf8', timeout: 2_000 })
      if (/AC Power/i.test(out)) return false
      if (/Battery Power/i.test(out)) return true
      return null
    }
  } catch {
    return null
  }
  return null
}

/** The snapshot `decide` needs, read from this machine as it is right now. */
export function liveConditions(adapter: string, available: string[]): Conditions {
  return {
    now: new Date(),
    adapter,
    available,
    busyMs: per => busyMsSince(WINDOW_MS[per]),
    loadPerCore: loadPerCore(),
    freeRamMb: freeRamMb(),
    onBattery: onBattery(),
  }
}

/** How much of a budget is left, for a window that wants to show a bar rather than a verdict. */
export function budgetState(limits: Limits | undefined): { usedMs: number; capMs: number; per: BudgetWindow } | null {
  const budget = limits?.budget
  if (!budget || budget.minutes <= 0) return null
  return { usedMs: busyMsSince(WINDOW_MS[budget.per]), capMs: budget.minutes * 60_000, per: budget.per }
}
