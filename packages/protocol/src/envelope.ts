import { z } from 'zod'

/**
 * Every WebSocket message on every link is this shape.
 *
 * `ts` is the SENDER's clock and is used for durations only. Ordering comes from
 * the server's monotonic run_events.seq — a Mac and a Windows box will disagree
 * about the time, and a run map built on sender timestamps is nonsense.
 */
export const Envelope = z.object({
  v: z.literal(1),
  id: z.string().min(1),
  replyTo: z.string().min(1).optional(),
  ts: z.string(),
  type: z.string().min(1),
  payload: z.unknown(),
})
export type Envelope = z.infer<typeof Envelope>

/**
 * The most JSON one task payload or one task result may be, in bytes.
 *
 * Half of the 128 KiB frame the server will read, because a `task.result` frame carries
 * the output twice over — once as the value and once inside the hash and signature it is
 * attested with — plus timestamps and lease identifiers.
 *
 * It is the agent's business as much as the server's: a result over this is refused, and
 * refused deterministically, so a host that ships one has thrown its own work away. Keep
 * in step with JSON_LIMIT in backend/src/orchestrator/shared/protocol.py.
 */
export const JSON_LIMIT = 64 * 1024

/**
 * Whether a result can actually be delivered, measured the way the server measures it.
 *
 * Canonical JSON on the server sorts keys and escapes non-ASCII, so this can read a
 * little under for output containing non-ASCII text. That is why the server still checks:
 * this exists to stop a host spending minutes on a slice and only then discovering the
 * answer will not fit, not to be the only thing standing between the two.
 */
export function resultTooLarge(output: unknown): boolean {
  return Buffer.byteLength(JSON.stringify(output ?? null), 'utf8') > JSON_LIMIT
}

export function envelope(type: string, payload: unknown, replyTo?: string): Envelope {
  return { v: 1, id: crypto.randomUUID(), ts: new Date().toISOString(), type, payload, ...(replyTo ? { replyTo } : {}) }
}

/** Parse an inbound frame. Returns null rather than throwing — a peer must never crash us. */
export function decode(raw: string | Buffer): Envelope | null {
  try {
    const parsed = Envelope.safeParse(JSON.parse(raw.toString('utf8')))
    return parsed.success ? parsed.data : null
  } catch {
    return null
  }
}
