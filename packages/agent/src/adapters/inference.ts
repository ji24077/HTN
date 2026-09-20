import { createHash } from 'node:crypto'
import { mkdirSync, existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { hostname } from 'node:os'
import { InferenceInput, type InferenceOutput, mintAssertion } from '@dwp/protocol'
import type { KeyObject } from 'node:crypto'
import { AGENT_HOME } from '../paths.ts'
import type { ExecutionReporter } from '../execution.ts'
import { createLogger } from '@dwp/protocol'

const log = createLogger({ component: 'agent' })

/**
 * Batch inference over a pinned model.
 *
 * The host is given hashes, not data. It fetches the model and inputs, verifies them,
 * and runs only its assigned slice. Nothing is split mid-model: parallelism is across
 * independent items, which is the only kind that is arithmetically safe.
 */

const CACHE = join(AGENT_HOME, 'artifacts')
const PIXELS = 28 * 28

type Ort = typeof import('onnxruntime-node')
type Session = Awaited<ReturnType<Ort['InferenceSession']['create']>>

/** Sessions are expensive to build and safe to reuse, so keep them keyed by model hash. */
const sessions = new Map<string, Promise<Session>>()

/**
 * Which backend to ask for, in order of preference, always ending at the CPU.
 *
 * Named honestly, because the obvious name would be wrong: this is **not** GPU
 * acceleration. Measured on an M5 Pro against this MNIST model, with the ONNX Runtime
 * profiler and the GPU's own performance counters:
 *
 *   plain CPU provider            9,943 img/s
 *   coreml, default flags        11,720 img/s   1.18x
 *   coreml, USE_CPU_ONLY         11,753 img/s   1.18x  <- identical to default
 *   coreml, CPU_AND_GPU           8,597 img/s   0.86x  <- slower than plain CPU
 *
 * Forcing CoreML onto the CPU costs nothing, which means the default was never leaving
 * the CPU; the GPU's utilisation counter stayed at idle levels (3% against 2% idle)
 * throughout. The gain is entirely CoreML's own CPU kernels — Accelerate and the AMX
 * matrix unit — beating the ones ONNX Runtime ships. Explicitly allowing the GPU makes
 * it *slower*, because a 26 KB model spends more on dispatch than it saves on
 * arithmetic. WebGPU, tested the same way, was 4.4x slower still.
 *
 * So: prefer CoreML because it is measurably faster and returns bit-identical logits,
 * and do not set COREML_FLAG_USE_CPU_AND_GPU. Whether DirectML helps on Windows is
 * untested and, on this evidence, should be assumed to hurt until someone measures a
 * model large enough to be worth dispatching.
 *
 * `cpu` is always last, and that is the whole safety story: a machine whose preferred
 * backend is missing or broken still computes the right answer.
 *
 * DWP_ORT_PROVIDERS overrides it — set it to `cpu` to force the plain path, which is how
 * these numbers were produced.
 */
function preferredProviders(): string[] {
  const override = process.env.DWP_ORT_PROVIDERS
  if (override) return override.split(',').map(x => x.trim()).filter(Boolean)
  if (process.platform === 'darwin') return ['coreml', 'cpu']
  if (process.platform === 'win32') return ['dml', 'cpu']
  /**
   * Linux tries CUDA first, which until now it did not.
   *
   * `accelerator.ts` has always *advertised* cuda on Linux, so the scheduler would route
   * CUDA-runtime work to such a machine -- and this function then ran it on the CPU
   * without anybody being told. A machine that claims a GPU and quietly does not use it
   * is the worst of the three possible states.
   *
   * Safe to attempt unconditionally: `createSession` walks this list and a provider whose
   * library will not load throws at session creation, which is caught and logged before
   * falling through to `cpu`. Measured in a linux/amd64 container with no CUDA runtime:
   * "Failed to load libonnxruntime_providers_cuda.so ... libcublasLt.so.13" then a clean
   * CPU session.
   */
  return ['cuda', 'cpu']
}

/**
 * Build a session on the best backend this machine will actually accept.
 *
 * A provider that is not compiled into the installed build makes `create` throw outright
 * rather than degrade, so each is tried in turn. Whatever succeeds is logged once: the
 * difference between "the GPU is working" and "it quietly fell back to the CPU months
 * ago" is otherwise invisible, and it is exactly the sort of thing nobody notices until
 * they are comparing timings that no longer mean what they think.
 */
async function createSession(ort: Ort, modelBytes: Buffer): Promise<Session> {
  const wanted = preferredProviders()
  for (let i = 0; i < wanted.length; i += 1) {
    const providers = wanted.slice(i)
    try {
      const session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: providers as never,
      })
      // Not "the GPU": see above. This records which kernel library won, nothing more.
      log.info('inference.backend', { using: providers[0], requested: wanted })
      return session
    } catch (err) {
      log.warn('inference.backend_unavailable', {
        provider: providers[0],
        reason: err instanceof Error ? err.message.split('\n')[0] : String(err),
      })
    }
  }
  throw new Error(`no usable inference backend from ${wanted.join(', ')}`)
}

/**
 * Load ONNX Runtime on first use rather than at import.
 *
 * A machine that only runs echo or the walker should never have to download 85 MB of
 * inference runtime, and importing it at the top of this file would make the whole agent
 * fail to start when it is absent.
 */
let ortPromise: Promise<Ort> | null = null
function loadOrt(): Promise<Ort> {
  ortPromise ??= import('onnxruntime-node').then(m => m.default ?? m).catch(() => {
    throw new Error(
      'this computer does not have the machine-learning runtime installed.\n' +
      '  Add it with:  pnpm agent enable ml',
    )
  }) as Promise<Ort>
  return ortPromise
}

async function fetchArtifact(
  server: string, hostId: string, privateKey: KeyObject, sha256: string,
): Promise<Buffer> {
  mkdirSync(CACHE, { recursive: true, mode: 0o700 })
  const cached = join(CACHE, sha256)

  if (existsSync(cached)) {
    const bytes = readFileSync(cached)
    if (createHash('sha256').update(bytes).digest('hex') === sha256) return bytes
    // A cache entry that no longer matches its own name is corrupt; fall through.
  }

  const res = await fetch(`${server}/artifacts/${sha256}`, {
    headers: { authorization: `Bearer ${mintAssertion(hostId, privateKey)}` },
    signal: AbortSignal.timeout(120_000),
  })
  if (!res.ok) throw new Error(`could not fetch artifact ${sha256.slice(0, 12)}: HTTP ${res.status}`)

  const bytes = Buffer.from(await res.arrayBuffer())
  const actual = createHash('sha256').update(bytes).digest('hex')
  // A hash mismatch is a hard failure, never a warning: running the wrong model would
  // produce confident, wrong, and silently unverifiable results.
  if (actual !== sha256) throw new Error(`artifact hash mismatch: expected ${sha256.slice(0, 12)}, got ${actual.slice(0, 12)}`)

  writeFileSync(cached, bytes, { mode: 0o600 })
  return bytes
}

export async function runInference(
  rawInput: unknown,
  ctx: { hostId: string; server: string; privateKey: KeyObject; signal: AbortSignal; report?: ExecutionReporter },
): Promise<InferenceOutput> {
  const input = InferenceInput.parse(rawInput)
  ctx.report?.step('Loading inference runtime')
  const ort = await loadOrt()

  const loadStart = performance.now()
  ctx.report?.step('Fetching and verifying model and input artifacts')
  const [modelBytes, inputBytes] = await Promise.all([
    fetchArtifact(ctx.server, ctx.hostId, ctx.privateKey, input.modelHash),
    fetchArtifact(ctx.server, ctx.hostId, ctx.privateKey, input.inputsHash),
  ])

  if (!sessions.has(input.modelHash)) {
    sessions.set(input.modelHash, createSession(ort, modelBytes))
  }
  const session = await sessions.get(input.modelHash)!
  const modelLoadMs = performance.now() - loadStart
  ctx.report?.step('Model loaded; executing inference batch')

  if (inputBytes.subarray(0, 4).toString('ascii') !== 'DWPI') throw new Error('input artifact has an unexpected format')
  const total = inputBytes.readUInt32BE(4)
  if (input.from + input.count > total) {
    throw new Error(`slice ${input.from}+${input.count} exceeds the ${total} available items`)
  }

  const pixelsStart = 8
  const labelsStart = 8 + total * PIXELS
  const predictions: number[] = []
  let correct = 0
  let logitChecksum = 0

  const inferStart = performance.now()
  for (let i = 0; i < input.count; i++) {
    if (ctx.signal.aborted) throw new Error('cancelled')

    const offset = pixelsStart + (input.from + i) * PIXELS
    const pixels = new Float32Array(PIXELS)
    // Preprocessing v1, recorded in the manifest so results stay comparable over time.
    for (let p = 0; p < PIXELS; p++) pixels[p] = inputBytes[offset + p]! / 255

    const out = await session.run({
      [input.inputName]: new ort.Tensor('float32', pixels, [1, 1, 28, 28]),
    })
    const logits = out[input.outputName]!.data as Float32Array

    let best = 0
    for (let c = 1; c < logits.length; c++) if (logits[c]! > logits[best]!) best = c
    predictions.push(best)
    logitChecksum += logits[best]!
    if (inputBytes[labelsStart + input.from + i] === best) correct += 1
    ctx.report?.progress(i + 1, input.count)
  }
  const inferenceMs = performance.now() - inferStart

  return {
    from: input.from,
    count: input.count,
    predictions,
    correct,
    // Rounded so tiny floating-point differences between chip generations do not read as
    // disagreement; real divergence is far larger than this.
    logitChecksum: Number(logitChecksum.toFixed(3)),
    hostId: ctx.hostId,
    hostname: hostname(),
    modelLoadMs: Number(modelLoadMs.toFixed(1)),
    inferenceMs: Number(inferenceMs.toFixed(1)),
    itemsPerSecond: Number((input.count / (inferenceMs / 1000)).toFixed(1)),
  }
}
