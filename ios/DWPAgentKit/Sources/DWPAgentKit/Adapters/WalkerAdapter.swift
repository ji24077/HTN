import Foundation

/**
 * Evaluate a slice of one generation of candidate gaits.
 *
 * Each candidate is rebuilt locally from the parent genome and its seed, simulated, and
 * scored. Because the simulation uses `JSMath` rather than the platform's, the same seed
 * produces bit-identical motion here and on a laptop — which is what lets a score from a
 * phone be compared with one from a Mac, and what lets the browser replay the winner
 * without showing something other than what was actually scored.
 */
public enum WalkerAdapter {
    public static let name = "walker_evolution"

    public static func run(_ rawInput: JSONValue, context: ExecContext) async throws -> JSONValue {
        guard let input = WalkerInput.parse(rawInput) else {
            throw AgentError.adapter("walker input was not the expected shape")
        }

        let started = DispatchTime.now().uptimeNanoseconds
        var results: [JSONValue] = []
        results.reserveCapacity(input.seeds.count)

        for (i, seed) in input.seeds.enumerated() {
            try Task.checkCancellation()
            let genome = Walker.perturb(parent: input.parent, sigma: input.sigma, seed: seed)
            let r = Walker.evaluate(genome, steps: input.steps)
            results.append(.object([
                ("seed", .int(seed)),
                ("fitness", .double(r.fitness)),
                ("distance", .double(r.distance)),
                ("ticks", .int(r.ticks)),
                ("fell", .bool(r.fell)),
            ]))
            // Progress is not decoration here: iOS expires a continued background task
            // that reports none, so a long slice on a locked phone depends on this.
            if (i + 1) % 4 == 0 || i + 1 == input.seeds.count {
                context.onProgress(i + 1, input.seeds.count)
            }
            // Yield so cancellation and lease renewals are not starved by a long run of
            // simulations, which are pure CPU and would otherwise never give up the thread.
            if (i + 1) % 8 == 0 { await Task.yield() }
        }

        let evalMs = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
        return .object([
            ("generation", .int(input.generation)),
            ("results", .array(results)),
            ("hostId", .string(context.hostId)),
            ("hostname", .string(deviceHostname)),
            ("evalMs", .double(roundTo(evalMs, 1))),
            ("evalsPerSecond", .double(roundTo(Double(input.seeds.count) / (evalMs / 1000), 1))),
        ])
    }
}
