import { createHash } from 'node:crypto'
import { mkdirSync, existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { hostname } from 'node:os'
import ort from 'onnxruntime-node'
import { InferenceInput, type InferenceOutput, mintAssertion } from '@dwp/protocol'
import type { KeyObject } from 'node:crypto'
import { AGENT_HOME } from '../paths.ts'

/**
 * Batch inference over a pinned model.
 *
 * The host is given hashes, not data. It fetches the model and inputs, verifies them,
 * and runs only its assigned slice. Nothing is split mid-model: parallelism is across
 * independent items, which is the only kind that is arithmetically safe.
 */

const CACHE = join(AGENT_HOME, 'artifacts')
const PIXELS = 28 * 28

/** Sessions are expensive to build and safe to reuse, so keep them keyed by model hash. */
const sessions = new Map<string, Promise<ort.InferenceSession>>()

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
  ctx: { hostId: string; server: string; privateKey: KeyObject; signal: AbortSignal },
): Promise<InferenceOutput> {
  const input = InferenceInput.parse(rawInput)

  const loadStart = performance.now()
  const [modelBytes, inputBytes] = await Promise.all([
    fetchArtifact(ctx.server, ctx.hostId, ctx.privateKey, input.modelHash),
    fetchArtifact(ctx.server, ctx.hostId, ctx.privateKey, input.inputsHash),
  ])

  if (!sessions.has(input.modelHash)) {
    sessions.set(input.modelHash, ort.InferenceSession.create(modelBytes))
  }
  const session = await sessions.get(input.modelHash)!
  const modelLoadMs = performance.now() - loadStart

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
