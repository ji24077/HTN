import * as Sentry from '@sentry/node'
import type { NodeOptions } from '@sentry/node'

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

export function telemetryOptions(env: NodeJS.ProcessEnv = process.env): NodeOptions | null {
  const dsn = env.SENTRY_DSN?.trim()
  if (!dsn) return null
  return {
    dsn,
    environment: env.SENTRY_ENVIRONMENT || 'development',
    release: env.SENTRY_RELEASE || env.RELEASE || undefined,
    sendDefaultPii: false,
    maxBreadcrumbs: 0,
    // The worker reports exceptions without recording task inputs or network bodies.
    defaultIntegrations: false,
    integrations: [Sentry.onUncaughtExceptionIntegration(), Sentry.onUnhandledRejectionIntegration()],
    skipOpenTelemetrySetup: true,
    beforeSend: event => scrubTelemetry(event, env),
    initialScope: { tags: { component: 'worker', runtime: 'desktop-agent' } },
  }
}

export function initTelemetry(): boolean {
  const options = telemetryOptions()
  if (!options) return false
  try {
    Sentry.init(options)
    return true
  } catch {
    // Reporting is optional and must not prevent a worker from starting.
    return false
  }
}

export function captureWorkloadFailure(
  error: unknown,
  context: { workerId: string; taskId: string; adapter: string; attempt: number },
): void {
  if (!Sentry.isEnabled()) return
  Sentry.withScope(scope => {
    scope.setTags({
      worker_id: context.workerId,
      task_id: context.taskId,
      adapter: context.adapter,
      attempt: String(context.attempt),
    })
    Sentry.captureException(error)
  })
}
