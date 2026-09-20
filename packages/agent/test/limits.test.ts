import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  allowedWorkloads, decide, minutesOfDay, standingReason, withinSchedule, type Conditions, type Limits,
} from '../src/limits.ts'

const AVAILABLE = ['echo', 'walker_evolution', 'cpu_inference_batch']

/** A machine with nothing wrong with it, so each test states only what it changes. */
function conditions(over: Partial<Conditions> = {}): Conditions {
  return {
    now: new Date('2026-09-19T14:00:00'),
    adapter: 'echo',
    available: AVAILABLE,
    busyMs: () => 0,
    loadPerCore: 0.1,
    freeRamMb: 8_000,
    onBattery: false,
    ...over,
  }
}

test('no limits means every offer is taken', () => {
  assert.deepEqual(decide(undefined, conditions()), { ok: true })
  assert.deepEqual(decide({}, conditions()), { ok: true })
})

// ------------------------------------------------------------------ workloads

test('a machine runs only the workloads it was set to run', () => {
  const limits: Limits = { workloads: ['echo'] }
  assert.deepEqual(decide(limits, conditions({ adapter: 'echo' })), { ok: true })
  const refused = decide(limits, conditions({ adapter: 'cpu_inference_batch' }))
  assert.equal(refused.ok, false)
  assert.equal(refused.ok === false && refused.reason, 'workload')
})

test('an empty or unavailable selection grants no permission to run', () => {
  for (const workloads of [[], ['nothing-real']]) {
    const limits = { workloads }
    assert.deepEqual(allowedWorkloads(limits, AVAILABLE), [])
    for (const adapter of AVAILABLE) assert.equal(decide(limits, conditions({ adapter })).ok, false)
    assert.equal(standingReason(limits, conditions()).ok, false)
  }
  assert.equal(standingReason({ workloads: ['echo'] }, conditions()).ok, true)
})

test('a choice is intersected with what the machine can actually run', () => {
  // Asking for ML on a machine without the runtime must not advertise ML.
  assert.deepEqual(
    allowedWorkloads({ workloads: ['echo', 'cpu_inference_batch'] }, ['echo', 'walker_evolution']),
    ['echo'],
  )
  assert.deepEqual(allowedWorkloads(undefined, AVAILABLE), AVAILABLE)
})

// ------------------------------------------------------------------- schedule

test('a clock string is read, or rejected without throwing', () => {
  assert.equal(minutesOfDay('22:00'), 22 * 60)
  assert.equal(minutesOfDay('7:05'), 7 * 60 + 5)
  assert.equal(minutesOfDay('00:00'), 0)
  for (const bad of ['24:00', '12:60', 'nine', '', '12', '12:5']) {
    assert.equal(minutesOfDay(bad), null, bad)
  }
})

test('an overnight window is the useful one and must actually work', () => {
  const overnight = { from: '22:00', to: '08:00' }
  const at = (h: number, m = 0) => new Date(2026, 8, 19, h, m)
  // Inside: late evening and early morning, across midnight.
  assert.equal(withinSchedule(overnight, at(22, 0)), true)
  assert.equal(withinSchedule(overnight, at(23, 59)), true)
  assert.equal(withinSchedule(overnight, at(0, 30)), true)
  assert.equal(withinSchedule(overnight, at(7, 59)), true)
  // Outside: the working day.
  assert.equal(withinSchedule(overnight, at(8, 0)), false)
  assert.equal(withinSchedule(overnight, at(14, 0)), false)
  assert.equal(withinSchedule(overnight, at(21, 59)), false)
})

test('a same-day window behaves the ordinary way', () => {
  const daytime = { from: '09:00', to: '17:00' }
  const at = (h: number) => new Date(2026, 8, 19, h, 0)
  assert.equal(withinSchedule(daytime, at(9)), true)
  assert.equal(withinSchedule(daytime, at(16)), true)
  assert.equal(withinSchedule(daytime, at(17)), false)
  assert.equal(withinSchedule(daytime, at(8)), false)
})

test('a zero-width window means all day, but an unreadable window grants no permission', () => {
  const at = new Date(2026, 8, 19, 3, 0)
  assert.equal(withinSchedule({ from: '09:00', to: '09:00' }, at), true)
  assert.equal(withinSchedule({ from: 'whenever', to: '09:00' }, at), false)
})

test('weekday overnight hours use the start day across midnight and week boundaries', () => {
  const weekdays: Limits['schedule'] = { from: '19:00', to: '07:00', days: [1, 2, 3, 4, 5] }
  assert.equal(withinSchedule(weekdays, new Date(2026, 8, 18, 19)), true) // Friday
  assert.equal(withinSchedule(weekdays, new Date(2026, 8, 19, 6, 59)), true) // Friday night
  assert.equal(withinSchedule(weekdays, new Date(2026, 8, 19, 7)), false)
  assert.equal(withinSchedule(weekdays, new Date(2026, 8, 19, 19)), false)
  assert.equal(withinSchedule(weekdays, new Date(2026, 8, 21, 6)), false) // Sunday night
  assert.equal(withinSchedule(weekdays, new Date(2026, 8, 21, 19)), true)
  assert.equal(withinSchedule({ from: '19:00', to: '07:00', days: [6] }, new Date(2026, 8, 20, 6)), true)
})

test('selecting no schedule days stops admission, including an all-day window', () => {
  assert.equal(withinSchedule({ from: '00:00', to: '00:00', days: [] }, conditions().now), false)
  assert.equal(withinSchedule({ from: '19:00', to: '07:00', days: [] }, new Date(2026, 8, 19, 22)), false)
})

test('standing admission automatically clears after pressure or schedule restrictions clear', () => {
  const limits: Limits = { workloads: ['echo'], schedule: { from: '19:00', to: '07:00' }, pressure: { minFreeRamMb: 2000 } }
  const night = conditions({ now: new Date(2026, 8, 19, 22) })
  assert.equal(standingReason(limits, conditions()).ok, false)
  assert.equal(standingReason(limits, { ...night, freeRamMb: 1000 }).ok, false)
  assert.equal(standingReason(limits, night).ok, true)
})

test('days restrict the window without replacing it', () => {
  // 2026-09-19 is a Saturday (day 6).
  const saturday = new Date(2026, 8, 19, 12, 0)
  assert.equal(withinSchedule({ from: '09:00', to: '17:00', days: [6] }, saturday), true)
  assert.equal(withinSchedule({ from: '09:00', to: '17:00', days: [1, 2, 3, 4, 5] }, saturday), false)
  // The hours still apply on a chosen day.
  const saturdayNight = new Date(2026, 8, 19, 22, 0)
  assert.equal(withinSchedule({ from: '09:00', to: '17:00', days: [6] }, saturdayNight), false)
})

test('outside its hours, the machine declines and says which hours', () => {
  const limits: Limits = { schedule: { from: '22:00', to: '08:00' } }
  const refused = decide(limits, conditions({ now: new Date(2026, 8, 19, 14, 0) }))
  assert.equal(refused.ok, false)
  assert.equal(refused.ok === false && refused.reason, 'schedule')
  assert.match(refused.ok === false ? refused.detail : '', /22:00/)
})

// --------------------------------------------------------------------- budget

test('a duty budget stops work once it is spent, and not before', () => {
  const limits: Limits = { budget: { minutes: 60, per: 'day' } }
  assert.deepEqual(decide(limits, conditions({ busyMs: () => 59 * 60_000 })), { ok: true })

  const spent = decide(limits, conditions({ busyMs: () => 60 * 60_000 }))
  assert.equal(spent.ok, false)
  assert.equal(spent.ok === false && spent.reason, 'budget')
  assert.match(spent.ok === false ? spent.detail : '', /60 minutes/)
})

test('the budget asks about the window it was configured with', () => {
  let asked: string | null = null
  decide({ budget: { minutes: 10, per: 'hour' } }, conditions({
    busyMs: per => { asked = per; return 0 },
  }))
  assert.equal(asked, 'hour')
})

test('a budget of zero minutes is ignored rather than stopping everything', () => {
  // Zero reads as "unset" in a number field, and a machine that silently refuses all
  // work because a field was cleared is indistinguishable from one that is broken.
  assert.deepEqual(decide({ budget: { minutes: 0, per: 'day' } }, conditions()), { ok: true })
})

// ------------------------------------------------------------------- pressure

test('a busy machine is left alone', () => {
  const limits: Limits = { pressure: { maxLoadPerCore: 0.7 } }
  assert.deepEqual(decide(limits, conditions({ loadPerCore: 0.69 })), { ok: true })
  const refused = decide(limits, conditions({ loadPerCore: 0.9 }))
  assert.equal(refused.ok === false && refused.reason, 'load')
})

test('a load guard stands aside where load cannot be read, rather than reading as idle', () => {
  /**
   * Windows reports a permanent zero from loadavg(). Treating that as a real reading
   * would make this guard silently inert — always passing — which is worse than absent,
   * because the window would show it switched on.
   */
  assert.deepEqual(
    decide({ pressure: { maxLoadPerCore: 0.1 } }, conditions({ loadPerCore: null })),
    { ok: true },
  )
})

test('memory the owner reserved is not taken', () => {
  const limits: Limits = { pressure: { minFreeRamMb: 2_000 } }
  assert.deepEqual(decide(limits, conditions({ freeRamMb: 2_000 })), { ok: true })
  const refused = decide(limits, conditions({ freeRamMb: 1_999 }))
  assert.equal(refused.ok === false && refused.reason, 'memory')
  assert.match(refused.ok === false ? refused.detail : '', /2000 MB/)
})

test('battery only stops work where a battery can actually be seen', () => {
  const limits: Limits = { pressure: { notOnBattery: true } }
  assert.equal(decide(limits, conditions({ onBattery: true })).ok, false)
  assert.deepEqual(decide(limits, conditions({ onBattery: false })), { ok: true })
  // A container has no battery; the rule must not refuse everything there.
  assert.deepEqual(decide(limits, conditions({ onBattery: null })), { ok: true })
})

test('the first rule that refuses is the one reported', () => {
  // Several wrong at once is the ordinary case; one clear reason beats a list.
  const limits: Limits = {
    schedule: { from: '22:00', to: '08:00' },
    budget: { minutes: 1, per: 'day' },
    pressure: { maxLoadPerCore: 0.1, minFreeRamMb: 99_000 },
  }
  const refused = decide(limits, conditions({
    now: new Date(2026, 8, 19, 14, 0),
    busyMs: () => 99 * 60_000,
    loadPerCore: 5,
    freeRamMb: 10,
  }))
  assert.equal(refused.ok === false && refused.reason, 'schedule')
})
