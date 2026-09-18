/**
 * A small fixed-window limiter for the two endpoints reachable without a session.
 *
 * In-memory and per-process, which is the right size for a single laptop hosting a
 * handful of friends. A multi-instance deployment would need shared state.
 */
type Bucket = { count: number; resetAt: number }

export class RateLimiter {
  readonly #buckets = new Map<string, Bucket>()
  readonly #limit: number
  readonly #windowMs: number

  constructor(limit: number, windowMs: number) {
    this.#limit = limit
    this.#windowMs = windowMs
    setInterval(() => {
      const now = Date.now()
      for (const [key, b] of this.#buckets) if (b.resetAt < now) this.#buckets.delete(key)
    }, windowMs).unref()
  }

  /** Returns seconds to wait when the caller is over its limit, or null when allowed. */
  check(key: string): number | null {
    const now = Date.now()
    const bucket = this.#buckets.get(key)
    if (!bucket || bucket.resetAt < now) {
      this.#buckets.set(key, { count: 1, resetAt: now + this.#windowMs })
      return null
    }
    bucket.count += 1
    return bucket.count > this.#limit ? Math.ceil((bucket.resetAt - now) / 1000) : null
  }
}
