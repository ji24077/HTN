import { z } from 'zod'

// ---------------------------------------------------------------- capability

export const CapabilityRecord = z.object({
  agentVersion: z.string(),
  os: z.enum(['darwin', 'win32', 'linux']),
  arch: z.string(),
  cpuModel: z.string(),
  logicalCores: z.number().int().positive(),
  totalRamMb: z.number().int().positive(),
  freeRamMb: z.number().int().nonnegative(),
  adapters: z.array(z.string()),
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
export const Heartbeat = z.object({ freeRamMb: z.number().nonnegative(), running: z.number().int().nonnegative() })
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

export const HelloAck = z.object({ hostId: z.string(), serverTime: z.string(), heartbeatSeconds: z.number() })
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
