import { appendFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'

/**
 * Structured logging for a system whose failures happen on someone else's machine.
 *
 * Every line goes two places: a human-readable line on the console, and one JSON object
 * per line in a file. The file is what you read after a friend says "it didn't work" —
 * it can be filtered, correlated by host, and diffed between runs, which a wall of
 * console text cannot.
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error'

const LEVEL_ORDER: Record<LogLevel, number> = { debug: 10, info: 20, warn: 30, error: 40 }

/**
 * Field names whose values must never be written to a log file.
 *
 * Logs get pasted into chats and issue trackers. Anything matching this is replaced by
 * its length, which is almost always the diagnostically useful part anyway (was it
 * empty? was it truncated?).
 */
const SECRET_KEY = /pass(word|phrase)?|secret|token|signature|privatekey|cookie|authorization|^code$|paircode/i

function redact(value: unknown, key: string): unknown {
  if (SECRET_KEY.test(key)) {
    if (value === undefined || value === null) return value
    const s = String(value)
    return `[redacted ${s.length}ch]`
  }
  if (value instanceof Error) return { name: value.name, message: value.message }
  return value
}

function sanitize(fields: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(fields)) {
    out[k] = v !== null && typeof v === 'object' && !Array.isArray(v) && !(v instanceof Error)
      ? sanitize(v as Record<string, unknown>)
      : redact(v, k)
  }
  return out
}

export type Logger = {
  readonly runId: string
  debug(event: string, fields?: Record<string, unknown>): void
  info(event: string, fields?: Record<string, unknown>): void
  warn(event: string, fields?: Record<string, unknown>): void
  error(event: string, fields?: Record<string, unknown>): void
  /** A logger that stamps every line with extra fields, e.g. a hostId. */
  child(bound: Record<string, unknown>): Logger
}

export type LoggerOptions = {
  component: string
  /** Directory for JSONL files. Set DWP_LOG_DIR to override; '' disables file output. */
  dir?: string
  level?: LogLevel
  /** Set false for processes whose stdout is parsed by something else. */
  console?: boolean
  runId?: string
}

const PAD: Record<LogLevel, string> = { debug: 'DBG', info: 'INF', warn: 'WRN', error: 'ERR' }

export function createLogger(options: LoggerOptions): Logger {
  const dir = options.dir ?? process.env.DWP_LOG_DIR ?? '.dwp/logs'
  const level = options.level ?? (process.env.DWP_LOG_LEVEL as LogLevel | undefined) ?? 'info'
  const useConsole = options.console ?? true
  const runId = options.runId ?? crypto.randomUUID().slice(0, 8)

  let file: string | null = null
  if (dir) {
    try {
      mkdirSync(dir, { recursive: true })
      file = join(dir, `${options.component}-${new Date().toISOString().slice(0, 10)}.jsonl`)
    } catch {
      file = null   // logging must never be the reason the process dies
    }
  }

  const make = (bound: Record<string, unknown>): Logger => {
    const emit = (lvl: LogLevel, event: string, fields: Record<string, unknown> = {}): void => {
      if (LEVEL_ORDER[lvl] < LEVEL_ORDER[level]) return
      const record = {
        ts: new Date().toISOString(),
        lvl,
        comp: options.component,
        runId,
        evt: event,
        ...sanitize({ ...bound, ...fields }),
      }

      if (file) {
        try { appendFileSync(file, JSON.stringify(record) + '\n') } catch {}
      }
      if (useConsole) {
        const extras = Object.entries(record)
          .filter(([k]) => !['ts', 'lvl', 'comp', 'runId', 'evt'].includes(k))
          .map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : String(v)}`)
          .join(' ')
        const line = `${record.ts.slice(11, 23)} ${PAD[lvl]} ${options.component}/${event}${extras ? '  ' + extras : ''}`
        if (lvl === 'error') console.error(line)
        else if (lvl === 'warn') console.warn(line)
        else console.log(line)
      }
    }

    return {
      runId,
      debug: (e, f) => emit('debug', e, f),
      info: (e, f) => emit('info', e, f),
      warn: (e, f) => emit('warn', e, f),
      error: (e, f) => emit('error', e, f),
      child: extra => make({ ...bound, ...extra }),
    }
  }

  return make({})
}
