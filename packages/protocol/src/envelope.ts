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
