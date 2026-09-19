import assert from 'node:assert/strict'
import { test } from 'node:test'
import { appendFileSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { load, record, recent, summary, type RunRecord } from '../src/history.ts'

/**
 * The module holds one file and one array, which is right for an agent and awkward for
 * a test suite. `load(path)` is the seam: it points the module at a fresh temp file and
 * clears what it was holding, so each case starts from nothing.
 */
function freshFile(): string {
  const home = mkdtempSync(join(tmpdir(), 'dwp-history-'))
  const file = join(home, 'history.jsonl')
  load(file)
  return file
}

let n = 0
function run(over: Partial<RunRecord> = {}): RunRecord {
  n += 1
  return {
    taskId: 'task-' + n,
    jobId: 'job-1',
    adapter: 'echo',
    attempt: 1,
    startedAt: new Date(1_700_000_000_000 + n * 1000).toISOString(),
    finishedAt: new Date(1_700_000_000_000 + n * 1000 + 500).toISOString(),
    durationMs: 500,
    outcome: 'ok',
    cpuMs: 10,
    rssMb: 80,
    shared: false,
    agentVersion: '0.4.0+test',
    ...over,
  }
}

test('a recorded run round-trips through the file', () => {
  const file = freshFile()
  const written = run({ adapter: 'walker_evolution', outputBytes: 4096, message: 'fine' })
  record(written)

  assert.deepEqual(recent(10), [written])
  // A second reader — the next launch of the app — sees the same thing.
  assert.deepEqual(load(file), [written])
  assert.equal(readFileSync(file, 'utf8').trimEnd().split('\n').length, 1)
})

test('a damaged line is skipped and the runs around it still load', () => {
  const file = freshFile()
  const first = run()
  const third = run()
  writeFileSync(file, JSON.stringify(first) + '\n' + '{"taskId": "half-writ' + '\n' +
    JSON.stringify(third) + '\n')

  assert.deepEqual(load(file), [first, third])
  // And the module keeps working afterwards rather than refusing to write.
  const fourth = run()
  record(fourth)
  assert.deepEqual(recent(1), [fourth])
  assert.deepEqual(load(file), [first, third, fourth])
})

test('the cap trims to the newest 500 and the file matches memory', () => {
  const file = freshFile()
  const all: RunRecord[] = []
  for (let i = 0; i < 1001; i++) {
    const r = run()
    all.push(r)
    record(r)
  }

  const kept = recent(1000)
  assert.equal(kept.length, 500)
  assert.deepEqual(kept[0], all[1000])
  assert.deepEqual(kept[499], all[501])
  // The rewrite is the point: a trim that only touched memory would grow forever on
  // disk and come back at the next launch.
  assert.deepEqual(load(file), all.slice(501))
})

test('summary counts runs, busy time and per-adapter totals', () => {
  freshFile()
  record(run({ adapter: 'echo', durationMs: 200, cpuMs: 5 }))
  record(run({ adapter: 'echo', durationMs: 300, cpuMs: 7, outcome: 'aborted' }))
  record(run({ adapter: 'walker_evolution', durationMs: 4000, cpuMs: 3900 }))
  record(run({ adapter: 'walker_evolution', durationMs: 1000, cpuMs: 900, outcome: 'error' }))

  const s = summary()
  assert.equal(s.runs, 4)
  assert.equal(s.ok, 2)
  assert.equal(s.failed, 2)
  assert.equal(s.busyMs, 5500)
  assert.equal(s.cpuMs, 4812)
  assert.deepEqual(s.byAdapter, {
    echo: { runs: 2, busyMs: 500, ok: 1 },
    walker_evolution: { runs: 2, busyMs: 5000, ok: 1 },
  })
  assert.equal(s.since, recent(4)[3]?.startedAt)
})

test('an unwritable history file costs the run nothing', () => {
  const home = mkdtempSync(join(tmpdir(), 'dwp-history-'))
  // A file where the module expects a directory: every write below it fails, on every
  // platform, without needing to play with permissions as root might ignore.
  const blocker = join(home, 'blocked')
  appendFileSync(blocker, 'not a directory\n')
  load(join(blocker, 'history.jsonl'))

  const r = run()
  assert.doesNotThrow(() => { record(r) })
  // It still remembers the run for this session's window, it just cannot persist it.
  assert.deepEqual(recent(1), [r])
  assert.equal(summary().runs, 1)
})
