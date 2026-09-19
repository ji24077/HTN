type Timer = ReturnType<typeof setTimeout>
type Scheduler = {
  schedule: (callback: () => void, delayMs: number) => Timer
  cancel: (timer: Timer) => void
  random: () => number
}

/** Bound automatic takeovers of one identity, and own their single retry timer. */
export class SupersessionPolicy {
  private count = 0
  private pending: Timer | undefined
  private readonly scheduler: Scheduler

  constructor(scheduler: Scheduler = { schedule: setTimeout, cancel: clearTimeout, random: Math.random }) {
    this.scheduler = scheduler
  }

  standDown(reconnect: () => void): { count: number; giveUp: boolean; retryInMs: number | null } {
    this.cancelPending()
    this.count += 1
    const giveUp = this.count >= 3
    const retryInMs = giveUp ? null : 60_000 + this.scheduler.random() * 60_000
    if (retryInMs !== null) {
      this.pending = this.scheduler.schedule(() => {
        this.pending = undefined
        reconnect()
      }, retryInMs)
    }
    return { count: this.count, giveUp, retryInMs }
  }

  cancelPending(): void {
    if (this.pending !== undefined) this.scheduler.cancel(this.pending)
    this.pending = undefined
  }

  /** Only an explicit takeover resets the budget; reopening a socket does not. */
  reset(): void {
    this.cancelPending()
    this.count = 0
  }
}
