/**
 * What a machine is allowed to claim about its accelerator.
 *
 * The expensive mistake here is not failing to find a GPU -- it is claiming one that is
 * not there. A machine advertising `runtime: 'cuda'` is sent CUDA work by the scheduler,
 * and if the provider then fails to load, the task runs on the CPU and every timing
 * taken from it is quietly wrong. So these tests are mostly about refusals.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { detectAccelerator } from '../src/accelerator.ts'

test('a machine with no inference runtime says exactly that', () => {
  // The default image is built `--target agent` and has no ONNX runtime at all. This is
  // the overwhelmingly common case and it must not read as "no GPU found", which would
  // send somebody looking for a driver problem that does not exist.
  const report = detectAccelerator('auto')
  if (report.available) return   // this machine does have a runtime; covered below
  assert.equal(report.runtime, 'cpu')
  assert.equal(report.vramMib, 0)
  assert.ok(report.reason.length > 0, 'a refusal always says why')
})

test('the providers it would consider match the platform', () => {
  const report = detectAccelerator('auto')
  const expected = process.platform === 'darwin'
    ? ['coreml', 'cpu']
    : process.platform === 'win32' ? ['dml', 'cpu'] : ['cuda', 'cpu']
  assert.deepEqual(report.providers, expected)
})

test('a machine set to CPU reports cpu but still says what hardware it has', () => {
  // The operator turning a device down should be able to see what they turned down,
  // which is why the preference is applied to the reported runtime rather than the probe.
  const report = detectAccelerator('cpu')
  assert.equal(report.runtime, 'cpu')
  if (report.available) {
    assert.match(report.reason, /set to CPU/)
  }
})

test('cuda is never claimed without a visible NVIDIA device', () => {
  /**
   * The trap this guards. `listSupportedBackends()` reports cuda on every Linux build of
   * onnxruntime-node -- measured in a linux/amd64 container with no GPU and no CUDA
   * libraries, where it came back as `{name:'cuda', bundled:false}`. Believing it made an
   * ordinary CPU machine advertise a GPU.
   *
   * On a machine with a real NVIDIA card this assertion is satisfied by the other branch:
   * vram is reported and the claim is legitimate.
   */
  const report = detectAccelerator('auto')
  if (report.runtime === 'cuda') {
    assert.ok(report.vramMib > 0, 'claiming cuda requires nvidia-smi to have answered')
    assert.ok(report.device !== null, 'claiming cuda requires a named device')
  } else {
    assert.ok(report.vramMib === 0 || report.runtime === 'mps')
  }
})
