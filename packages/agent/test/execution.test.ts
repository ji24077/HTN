import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ExecutionJournal } from '../src/execution.ts'
import { runEcho } from '../src/adapters/echo.ts'
import { runWalker } from '../src/adapters/walker.ts'

function fixture(t: test.TestContext) {
  const dir = mkdtempSync(join(tmpdir(), 'execution-'))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const journal = new ExecutionJournal('https://fleet.test', 'machine-1', dir)
  const records = () => readdirSync(journal.directory).filter(f => f.endsWith('.json'))
    .map(f => JSON.parse(readFileSync(join(journal.directory, f), 'utf8')))
  return { dir, journal, records }
}

test('successful adapters emit steps and progress, and output hooks persist redacted text', async t => {
  const { journal, records } = fixture(t)
  const report = journal.reporter('task-1', 1)
  journal.emit('task-1', 1, 'started', { adapter: 'echo' })
  await runEcho({ nonce: 'ok' }, 'machine-1', new AbortController().signal, report)
  await runWalker({ generation: 0, parent: Array(308).fill(0), sigma: 0.1, seeds: [1, 2], steps: 20 },
    'machine-1', new AbortController().signal, report)
  report.stdout('done successfully')
  report.stderr('Authorization: Bearer secret-test-value')
  journal.emit('task-1', 1, 'succeeded', { duration_ms: 2 })
  const events = records()[0].events
  assert.ok(events.some((e: any) => e.kind === 'step'))
  assert.ok(events.some((e: any) => e.kind === 'progress' && e.data.percent === 100))
  assert.equal(events.at(-1).kind, 'succeeded')
  assert.ok(JSON.stringify(events).includes('done successfully'))
  assert.ok(!JSON.stringify(events).includes('secret-test-value'))
})

test('unacknowledged events survive restart, acknowledged events stay local, and servers are isolated', t => {
  const { dir, journal, records } = fixture(t)
  journal.emit('task-1', 1, 'started')
  let batch: any
  journal.flush(value => { batch = value })
  assert.equal(batch.events[0].sequence, 1)
  const restarted = new ExecutionJournal('https://fleet.test', 'machine-1', dir)
  restarted.flush(value => { batch = value })
  assert.deepEqual(batch.events.map((e: any) => e.kind), ['started', 'interrupted'])
  restarted.acknowledge({ taskId: 'task-1', attempt: 1, sequences: [1, 2] })
  const again = new ExecutionJournal('https://fleet.test', 'machine-1', dir)
  again.flush(() => assert.fail('acknowledged events replayed'))
  const other = new ExecutionJournal('https://other.test', 'machine-1', dir)
  other.flush(() => assert.fail('events leaked to another server'))
  assert.equal(records()[0].events.length, 2)
})

test('ordinary output is coalesced on disk and the journal is bounded by bytes, not only count', async t => {
  const { journal, records } = fixture(t)
  const report = journal.reporter('task-1', 1)
  journal.emit('task-1', 1, 'started')
  report.stdout('x'.repeat(2048))
  assert.deepEqual(records()[0].events.map((e: any) => e.kind), ['started'])
  await new Promise(resolve => setTimeout(resolve, 400))
  assert.deepEqual(records()[0].events.map((e: any) => e.kind), ['started', 'stdout'])
  for (let i = 0; i < 700; i++) report.stdout('x'.repeat(2048))
  journal.emit('task-1', 1, 'succeeded')
  const events = records()[0].events
  assert.equal(events.filter((e: any) => e.kind === 'truncated').length, 1)
  assert.ok(events.length < 600)
  assert.equal(events.at(-1).kind, 'succeeded')
  const [file] = readdirSync(journal.directory).filter(f => f.endsWith('.json'))
  assert.ok(statSync(join(journal.directory, file!)).size < 1100 * 1024)
})

test('output limits retain an explicit truncation marker and final status', t => {
  const { journal, records } = fixture(t)
  for (let i = 0; i < 1010; i++) journal.reporter('task-1', 1).stdout('line')
  journal.emit('task-1', 1, 'failed', { error: { name: 'ExampleError', message: 'failure' } })
  const events = records()[0].events
  assert.ok(events.length <= 1000)
  assert.equal(events.filter((e: any) => e.kind === 'truncated').length, 1)
  assert.equal(events.at(-1).kind, 'failed')
})
