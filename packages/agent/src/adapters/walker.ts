import { hostname } from 'node:os'
import { WalkerInput, type WalkerOutput } from '@dwp/protocol'
import { perturb, evaluate } from '@dwp/protocol/walker.js'

/**
 * Evaluate a slice of one generation of candidate gaits.
 *
 * Each candidate is rebuilt locally from the parent genome and its seed, simulated, and
 * scored. The simulation is deterministic, so the same seed produces the same score on
 * any machine — which is what lets scores from different hosts be compared at all.
 */
export async function runWalker(
  rawInput: unknown,
  hostId: string,
  signal: AbortSignal,
): Promise<WalkerOutput> {
  const input = WalkerInput.parse(rawInput)
  const started = performance.now()
  const results: WalkerOutput['results'] = []

  for (const seed of input.seeds) {
    if (signal.aborted) throw new Error('cancelled')
    const genome = perturb(input.parent, input.sigma, seed)
    const r = evaluate(genome, input.steps)
    results.push({ seed, fitness: r.fitness, distance: r.distance, ticks: r.ticks, fell: r.fell })
    // Yield occasionally so cancellation and lease renewals are not starved by a long
    // synchronous run of simulations.
    if (results.length % 8 === 0) await new Promise(resolve => setImmediate(resolve))
  }

  const evalMs = performance.now() - started
  return {
    generation: input.generation,
    results,
    hostId,
    hostname: hostname(),
    evalMs: Number(evalMs.toFixed(1)),
    evalsPerSecond: Number((input.seeds.length / (evalMs / 1000)).toFixed(1)),
  }
}
