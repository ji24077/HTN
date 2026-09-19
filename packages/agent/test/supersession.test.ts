import assert from 'node:assert/strict'
import { test } from 'node:test'
import { SupersessionPolicy } from '../src/supersession.ts'

function fixture() {
  const queued = new Map<ReturnType<typeof setTimeout>, () => void>()
  let next = 0
  const policy = new SupersessionPolicy({
    random: () => 0,
    schedule: (callback, delay) => {
      assert.equal(delay, 60_000)
      const id = ++next as unknown as ReturnType<typeof setTimeout>
      queued.set(id, callback)
      return id
    },
    cancel: timer => { queued.delete(timer) },
  })
  const tick = () => {
    const first = [...queued.entries()][0]
    assert.ok(first, 'expected a scheduled retry')
    queued.delete(first[0])
    first[1]()
  }
  return { policy, queued, tick }
}

test('stops after three supersessions across successful automatic reconnects', () => {
  const { policy, queued, tick } = fixture()
  let connections = 1
  const reconnect = () => { connections += 1 }
  assert.equal(policy.standDown(reconnect).count, 1)
  tick()
  assert.equal(policy.standDown(reconnect).count, 2)
  tick()
  assert.deepEqual(policy.standDown(reconnect), { count: 3, giveUp: true, retryInMs: null })
  assert.equal(connections, 3)
  assert.equal(queued.size, 0)
})

test('manual takeover cancels the old timer and grants a fresh retry budget', () => {
  const { policy, queued } = fixture()
  policy.standDown(() => assert.fail('stale retry fired'))
  policy.reset()
  assert.equal(queued.size, 0)
  assert.equal(policy.standDown(() => {}).count, 1)
  assert.equal(queued.size, 1)
  policy.cancelPending()
  assert.equal(queued.size, 0)
})

test('there is at most one pending automatic takeover', () => {
  const { policy, queued } = fixture()
  policy.standDown(() => {})
  policy.standDown(() => {})
  assert.equal(queued.size, 1)
})
