import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, chmodSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pythonCapability, programAvailable, runProgram } from '../src/adapters/program.ts'

const pythonRuntime = { version: '3.12.14', pytorch: '2.13.0+cpu' }

function bridge(t: test.TestContext, body: string) {
  const directory = mkdtempSync(join(tmpdir(), 'program-bridge-'))
  const python = join(directory, 'bridge')
  writeFileSync(python, `#!${process.execPath}\nif(process.argv.includes('--check')) { console.log('${JSON.stringify({ runtime: 'cpu', python: pythonRuntime })}'); process.exit(0) }\n${body}`)
  chmodSync(python, 0o700)
  const previous = process.env.DWP_PROGRAM_PYTHON
  process.env.DWP_PROGRAM_PYTHON = python
  t.after(() => {
    if (previous === undefined) delete process.env.DWP_PROGRAM_PYTHON
    else process.env.DWP_PROGRAM_PYTHON = previous
    rmSync(directory, { recursive: true, force: true })
  })
  const events: unknown[] = []
  const controller = new AbortController()
  const context = {
    taskId: 'task', jobId: 'job', hostId: 'worker', server: 'https://fleet.test', attempt: 2,
    signal: controller.signal,
    report: {
      step: (message: string) => events.push(message),
      stdout: (text: string) => events.push(text), stderr: (text: string) => events.push(text),
      progress: (done: number, total: number) => events.push({ done, total }),
    },
    cleaned: (data: object) => events.push({ cleaned: data }),
  }
  return { context, events, controller }
}

const input = { spec: { id: 'task', job_id: 'job', kind: 'python_project' } }

test('project bridge forwards identity and attempt, emits logs/cleanup, and strips agent environment', { skip: process.platform === 'win32' }, async t => {
  const { context, events } = bridge(t, `
    let text=''; process.stdin.on('data', b=>text+=b); process.stdin.on('end',()=>{
      const r=JSON.parse(text);
      console.log(JSON.stringify({kind:'progress',data:{done:4,total:8}}));
      console.log(JSON.stringify({kind:'cleaned',data:{workspace_removed:true,processes_stopped:true}}));
      console.log(JSON.stringify({kind:'result',data:{ok:true,attempt:r.attempt,worker:r.worker_id,server:r.server,hasCredential:!!process.env.DWP_CODE}}));
    });
  `)
  const previous = process.env.DWP_CODE
  process.env.DWP_CODE = 'test-private-code'
  t.after(() => { if (previous === undefined) delete process.env.DWP_CODE; else process.env.DWP_CODE = previous })
  assert.equal(programAvailable(), true)
  assert.deepEqual(pythonCapability(), pythonRuntime)
  assert.deepEqual(await runProgram(input, context), {
    ok: true, attempt: 2, worker: 'worker', server: 'https://fleet.test', hasCredential: false,
  })
  assert.ok(events.some(e => JSON.stringify(e) === '{"done":4,"total":8}'))
  assert.ok(events.some(e => JSON.stringify(e).includes('workspace_removed')))
  await assert.rejects(runProgram({ spec: { ...input.spec, id: 'other' } }, context), /envelope/)
})

test('cancellation waits for the bridge to acknowledge cleanup and exit', { skip: process.platform === 'win32' }, async t => {
  const { context, events, controller } = bridge(t, `
    process.on('SIGTERM',()=>{
      console.log(JSON.stringify({kind:'cleaned',data:{workspace_removed:true,processes_stopped:true}}));
      process.exit(0);
    });
    console.log(JSON.stringify({kind:'stdout',data:{text:'ready'}}));
    setInterval(()=>{},1000);
  `)
  const started = runProgram(input, context)
  const rejected = assert.rejects(started, /cancelled/)
  while (!events.includes('ready')) await new Promise(resolve => setTimeout(resolve, 10))
  controller.abort()
  await rejected
  assert.ok(events.some(e => JSON.stringify(e).includes('workspace_removed')))
})

test('a failed uploaded validator remains a failed project result', { skip: process.platform === 'win32' }, async t => {
  const { context } = bridge(t, `console.log(JSON.stringify({kind:'result',data:{ok:false,error:'invalid checkpoint'}}));`)
  assert.deepEqual(await runProgram(input, context), { ok: false, error: 'invalid checkpoint' })
})
