/**
 * In-memory connection history, for answering "what actually happened to that host?"
 *
 * The database records durable facts; this records the noisy operational detail that
 * is only interesting for a while — flap counts, last close code, why an auth failed.
 * Bounded on purpose: this must never be the reason a long-running server grows.
 */
const MAX_EVENTS = 500
const MAX_AUTH_FAILURES = 100

export type ConnectionRecord = {
  hostId: string
  label: string
  remoteAddr: string
  connectedAt: string
  disconnectedAt: string | null
  lastCloseCode: number | null
  connectCount: number
  /** Reconnects within a short window — the signature of an unstable link. */
  flapCount: number
}

export type AuthFailure = {
  at: string
  hostId: string | null
  reason: string
  remoteAddr: string
}

const byHost = new Map<string, ConnectionRecord>()
const authFailures: AuthFailure[] = []
const timeline: { at: string; hostId: string; event: string; detail?: string }[] = []

const FLAP_WINDOW_MS = 60_000

function push(hostId: string, event: string, detail?: string): void {
  timeline.push({ at: new Date().toISOString(), hostId, event, ...(detail ? { detail } : {}) })
  if (timeline.length > MAX_EVENTS) timeline.splice(0, timeline.length - MAX_EVENTS)
}

export function noteConnection(hostId: string, label: string, remoteAddr: string): void {
  const now = Date.now()
  const existing = byHost.get(hostId)
  const reconnectedQuickly =
    existing?.disconnectedAt != null && now - Date.parse(existing.disconnectedAt) < FLAP_WINDOW_MS

  byHost.set(hostId, {
    hostId,
    label,
    remoteAddr,
    connectedAt: new Date(now).toISOString(),
    disconnectedAt: null,
    lastCloseCode: existing?.lastCloseCode ?? null,
    connectCount: (existing?.connectCount ?? 0) + 1,
    flapCount: (existing?.flapCount ?? 0) + (reconnectedQuickly ? 1 : 0),
  })
  push(hostId, 'connected', remoteAddr)
}

export function noteDisconnection(hostId: string, closeCode: number): void {
  const rec = byHost.get(hostId)
  if (rec) {
    rec.disconnectedAt = new Date().toISOString()
    rec.lastCloseCode = closeCode
  }
  push(hostId, 'disconnected', `close=${closeCode}`)
}

export function noteAuthFailure(hostId: string | null, reason: string, remoteAddr: string): void {
  authFailures.push({ at: new Date().toISOString(), hostId, reason, remoteAddr })
  if (authFailures.length > MAX_AUTH_FAILURES) authFailures.shift()
  push(hostId ?? 'unknown', 'auth_rejected', reason)
}

export function snapshot() {
  return {
    now: new Date().toISOString(),
    connections: [...byHost.values()].sort((a, b) => b.connectedAt.localeCompare(a.connectedAt)),
    recentAuthFailures: authFailures.slice(-25).reverse(),
    timeline: timeline.slice(-100).reverse(),
  }
}
