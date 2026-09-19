import * as Sentry from '@sentry/node'
import type { NodeOptions } from '@sentry/node'
import { WorkerTelemetry } from '@dwp/protocol'

const REDACTED = '[redacted]'
const SENSITIVE_KEY = /authorization|cookie|token|secret|passw|credential|private[-_]?key|api[-_]?key|signature|dsn|^code$|pair(?:ing)?[-_]?code/i
const rememberedSecrets = new Set<string>()

/** Pairing codes are supplied through both CLI arguments and the desktop window. */
export function rememberTelemetrySecret(value: string): void {
  if (value.length >= 4) rememberedSecrets.add(value)
}

function scrubText(value: string, secrets: readonly string[]): string {
  for (const secret of secrets) value = value.split(secret).join(REDACTED)
  return value
    .replace(/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g, REDACTED)
    .replace(/Bearer\s+[A-Za-z0-9._~+/=-]+/gi, REDACTED)
    .replace(/eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+/g, REDACTED)
    .replace(/([?&#](?:code|token|access_token|refresh_token|secret)=)[^&#\s]*/gi, `$1${REDACTED}`)
    .replace(/(\b(?:code|token|secret|private[-_]?key)\s*[:=]\s*)[^\s,;]+/gi, `$1${REDACTED}`)
    .replace(/(https?:\/\/)[^\s/@]+@/gi, `$1${REDACTED}@`)
}

/** Scrub every serialized event field, including errors, nested data and header pairs. */
export function scrubTelemetry<T>(value: T, env: NodeJS.ProcessEnv = process.env): T {
  const secrets = [...rememberedSecrets,
    ...Object.entries(env).filter(([key, item]) => SENSITIVE_KEY.test(key) && item && item.length >= 8)
      .map(([, item]) => item!),
  ].sort((a, b) => b.length - a.length)
  const walk = (item: unknown, depth: number): unknown => {
    if (depth > 12) return REDACTED
    if (typeof item === 'string') return scrubText(item, secrets)
    if (item instanceof Error) return walk({ name: item.name, message: item.message, stack: item.stack }, depth + 1)
    if (Array.isArray(item)) {
      if (item.length === 2 && typeof item[0] === 'string' && SENSITIVE_KEY.test(item[0])) {
        return [item[0], REDACTED]
      }
      return item.map(child => walk(child, depth + 1))
    }
    if (item && typeof item === 'object') {
      return Object.fromEntries(Object.entries(item).map(([key, child]) => [
        key, SENSITIVE_KEY.test(key) && child != null ? REDACTED : walk(child, depth + 1),
      ]))
    }
    return item
  }
  return walk(value, 0) as T
}

export function telemetryOptions(env: NodeJS.ProcessEnv = process.env, saved?: unknown): NodeOptions | null {
  const parsed = WorkerTelemetry.safeParse(saved)
  const remote = parsed.success ? parsed.data : undefined
  // Presence matters: an explicitly empty local DSN is the operator's opt-out.
  const dsn = (env.SENTRY_DSN !== undefined ? env.SENTRY_DSN : remote?.dsn)?.trim()
  if (!dsn) return null
  return {
    dsn,
    // Blank values (a template .env) fall through; only the DSN treats blank as a choice.
    environment: env.SENTRY_ENVIRONMENT || remote?.environment || 'development',
    release: env.SENTRY_RELEASE || env.RELEASE || remote?.release || undefined,
    sendDefaultPii: false,
    maxBreadcrumbs: 0,
    // The worker reports exceptions without recording task inputs or network bodies.
    defaultIntegrations: false,
    integrations: [Sentry.onUncaughtExceptionIntegration(), Sentry.onUnhandledRejectionIntegration()],
    skipOpenTelemetrySetup: true,
    beforeSend: event => scrubTelemetry(event, { ...env, SENTRY_DSN: dsn }),
    initialScope: { tags: { component: 'worker', runtime: 'desktop-agent' } },
  }
}

let activeSettings: string | undefined

const PROCESS_EVENTS = ['uncaughtException', 'unhandledRejection'] as const
type ProcessListener = (...args: any[]) => void
let installedListeners: Array<[typeof PROCESS_EVENTS[number], ProcessListener]> = []
const emitter: NodeJS.EventEmitter = process

/** Sentry registers crash handlers per client and never removes them; we do, so DSN rotation cannot stack them. */
function removeProcessListeners(): void {
  for (const [event, listener] of installedListeners) emitter.removeListener(event, listener)
  installedListeners = []
}

function trackProcessListeners(init: () => void): void {
  const before = new Map(PROCESS_EVENTS.map(event => [event, new Set(emitter.listeners(event))]))
  try {
    init()
  } finally {
    for (const event of PROCESS_EVENTS) {
      for (const listener of emitter.listeners(event)) {
        if (!before.get(event)!.has(listener)) installedListeners.push([event, listener as ProcessListener])
      }
    }
  }
}

export function initTelemetry(saved?: unknown): boolean {
  const options = telemetryOptions(process.env, saved)
  const fingerprint = JSON.stringify(options ? [options.dsn, options.environment, options.release] : null)
  if (fingerprint === activeSettings && (!options || Sentry.isEnabled())) return Boolean(options)
  const previous = Sentry.getClient()
  Sentry.getCurrentScope().setClient(undefined)
  if (previous) void Promise.resolve(previous.close(2_000)).catch(() => {})
  removeProcessListeners()
  activeSettings = undefined
  if (!options) {
    activeSettings = fingerprint
    return false
  }
  try {
    trackProcessListeners(() => Sentry.init(options))
    activeSettings = fingerprint
    return true
  } catch {
    // Reporting is optional and must not prevent a worker from starting.
    return false
  }
}

export function captureWorkloadFailure(
  error: unknown,
  context: { workerId: string; taskId: string; adapter: string; attempt: number; jobId?: string },
): void {
  if (!Sentry.isEnabled()) return
  Sentry.withScope(scope => {
    scope.setTags({
      worker_id: context.workerId,
      task_id: context.taskId,
      job_id: context.jobId,
      adapter: context.adapter,
      attempt: String(context.attempt),
      execution_id: `${context.taskId}:${context.attempt}`,
      reservation_id: `${context.taskId}:${context.attempt}`,
    })
    Sentry.captureException(error)
  })
}
