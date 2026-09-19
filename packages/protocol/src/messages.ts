import { z } from 'zod'

// ---------------------------------------------------------------- capability

/**
 * What a phone reports about itself beyond its hardware.
 *
 * A laptop's capability is static between heartbeats; a phone's is not. It gets hot, it
 * gets unplugged, it drops into Low Power Mode, and each of those changes whether it is
 * sensible to hand it another hundred items. Optional, because desktop hosts have no
 * answer for any of it — and absent is meaningfully different from false here.
 */
export const MobileState = z.object({
  thermal: z.enum(['nominal', 'fair', 'serious', 'critical', 'unknown']),
  lowPowerMode: z.boolean(),
  availableMemoryMb: z.number().int().nonnegative(),
  /** The device's own verdict, so the scheduler need not re-derive the policy. */
  fitForWork: z.boolean(),
  batteryLevel: z.number().min(0).max(1).optional(),
  charging: z.boolean().optional(),
})
export type MobileState = z.infer<typeof MobileState>

export const CapabilityRecord = z.object({
  agentVersion: z.string(),
  os: z.enum(['darwin', 'win32', 'linux', 'ios', 'android']),
  arch: z.string(),
  cpuModel: z.string(),
  logicalCores: z.number().int().positive(),
  totalRamMb: z.number().int().positive(),
  freeRamMb: z.number().int().nonnegative(),
  adapters: z.array(z.string()),
  mobile: MobileState.optional(),
})
export type CapabilityRecord = z.infer<typeof CapabilityRecord>

export const ConsentState = z.object({
  paused: z.boolean(),
  allowCompute: z.boolean(),
  allowBrowser: z.boolean(),
  maxConcurrency: z.number().int().positive(),
})
export type ConsentState = z.infer<typeof ConsentState>

// ------------------------------------------------------------ agent → control

export const Hello = z.object({ capability: CapabilityRecord, consent: ConsentState })
export const Heartbeat = z.object({
  freeRamMb: z.number().nonnegative(),
  running: z.number().int().nonnegative(),
  /** Present only from mobile hosts, where this changes minute to minute. */
  mobile: MobileState.optional(),
})
export const TaskAccept = z.object({ taskId: z.string(), leaseId: z.string() })
export const TaskDecline = z.object({ taskId: z.string(), leaseId: z.string(), reason: z.string() })
export const LeaseRenew = z.object({ taskId: z.string(), leaseId: z.string() })
export const TaskProgress = z.object({ taskId: z.string(), done: z.number().int(), total: z.number().int() })

/**
 * A result the control service did not compute and cannot forge.
 *
 * `signature` is Ed25519 over the canonical attestation string (see attestation.ts),
 * made with the host's private key. It is what makes the `distribution` gate provable
 * to a reviewer who does not trust the operator.
 */
export const TaskResult = z.object({
  taskId: z.string(),
  leaseId: z.string(),
  attempt: z.number().int().positive(),
  output: z.unknown(),
  outputHash: z.string(),
  startedAt: z.string(),
  finishedAt: z.string(),
  hostReportedMs: z.number().nonnegative(),
  signature: z.string(),
})
export const TaskError = z.object({
  taskId: z.string(),
  leaseId: z.string(),
  errorClass: z.string(),
  message: z.string(),
})
export const ConsentUpdate = ConsentState

// ------------------------------------------------------------ control → agent

export const HelloAck = z.object({
  hostId: z.string(),
  serverTime: z.string(),
  heartbeatSeconds: z.number(),
  /** The release the server is currently offering, so agents can notice they are behind. */
  releaseVersion: z.string().nullable().default(null),
})
export const TaskOffer = z.object({
  taskId: z.string(),
  jobId: z.string(),
  adapter: z.string(),
  attempt: z.number().int().positive(),
  input: z.unknown(),
  leaseId: z.string(),
  leaseSeconds: z.number().int().positive(),
  wallClockMs: z.number().int().positive(),
})
export const TaskCancel = z.object({ taskId: z.string(), reason: z.string() })
export const Revoked = z.object({ reason: z.string() })

// ------------------------------------------------------------------- adapters

/** Echo: the Pass 1 adapter. Proves a specific host really executed a specific task. */
export const EchoInput = z.object({ nonce: z.string(), sleepMs: z.number().int().nonnegative().default(0) })
export const EchoOutput = z.object({
  nonce: z.string(),
  hostId: z.string(),
  hostname: z.string(),
  os: z.string(),
  arch: z.string(),
  elapsedMs: z.number().nonnegative(),
})
export type EchoInput = z.infer<typeof EchoInput>
export type EchoOutput = z.infer<typeof EchoOutput>

/**
 * A slice of a batch inference job.
 *
 * The host is told which model and inputs to use by hash, and which items of that input
 * set are its share. It never receives the data inline — it fetches and verifies the
 * artifacts itself, so every host provably ran the same model over the same bytes.
 */
export const InferenceInput = z.object({
  modelHash: z.string().length(64),
  inputsHash: z.string().length(64),
  inputName: z.string(),
  outputName: z.string(),
  /** Index of the first item of this slice, and how many items it covers. */
  from: z.number().int().nonnegative(),
  count: z.number().int().positive(),
  preprocessing: z.literal('v1'),
})
export type InferenceInput = z.infer<typeof InferenceInput>

export const InferenceOutput = z.object({
  from: z.number().int().nonnegative(),
  count: z.number().int().positive(),
  /** Predicted class per item, in slice order. */
  predictions: z.array(z.number().int()),
  /** Correct predictions, from the labels shipped with the inputs. */
  correct: z.number().int().nonnegative(),
  /** Sum of each item's top logit — a cheap fingerprint for cross-host agreement. */
  logitChecksum: z.number(),
  hostId: z.string(),
  hostname: z.string(),
  modelLoadMs: z.number(),
  inferenceMs: z.number(),
  itemsPerSecond: z.number(),
})
export type InferenceOutput = z.infer<typeof InferenceOutput>

/**
 * One slice of a generation: evaluate these candidate gaits.
 *
 * Only the parent genome and a list of seeds travel over the wire. Each host rebuilds the
 * candidates itself from parent + seed, so a generation of hundreds costs one genome of
 * bandwidth rather than hundreds, and any host can reproduce any candidate exactly.
 */
export const WalkerInput = z.object({
  generation: z.number().int().nonnegative(),
  parent: z.array(z.number()),
  sigma: z.number().positive(),
  seeds: z.array(z.number().int()),
  steps: z.number().int().positive().max(5000),
})
export type WalkerInput = z.infer<typeof WalkerInput>

export const WalkerOutput = z.object({
  generation: z.number().int().nonnegative(),
  results: z.array(z.object({
    seed: z.number().int(),
    fitness: z.number(),
    distance: z.number(),
    ticks: z.number().int(),
    fell: z.boolean(),
  })),
  hostId: z.string(),
  hostname: z.string(),
  evalMs: z.number(),
  evalsPerSecond: z.number(),
})
export type WalkerOutput = z.infer<typeof WalkerOutput>

export const AGENT_TO_CONTROL = {
  hello: Hello,
  heartbeat: Heartbeat,
  'consent.update': ConsentUpdate,
  'task.accept': TaskAccept,
  'task.decline': TaskDecline,
  'lease.renew': LeaseRenew,
  'task.progress': TaskProgress,
  'task.result': TaskResult,
  'task.error': TaskError,
} as const

export const CONTROL_TO_AGENT = {
  'hello.ack': HelloAck,
  'task.offer': TaskOffer,
  'task.cancel': TaskCancel,
  revoked: Revoked,
} as const
