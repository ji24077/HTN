import Foundation
import OnnxRuntimeBindings

/**
 * Batch inference over a pinned model, one independent item at a time.
 *
 * Nothing is split mid-model — parallelism is across items only, which is the only kind
 * that is arithmetically safe and, not coincidentally, the only kind a phone can serve
 * without becoming a liability. A slice is abandonable at item granularity, so a device
 * that gets hot, gets locked, or walks out of wifi costs the job one slice rather than
 * a stall.
 */
public actor InferenceAdapter {
    public static let name = "cpu_inference_batch"
    public static let shared = InferenceAdapter()

    private var env: ORTEnv?
    /// Sessions are expensive to build and safe to reuse, so they are kept by model hash.
    private var sessions: [String: ORTSession] = [:]

    private static let pixels = 28 * 28

    /**
     * Deliberately not `processorCount`.
     *
     * Saturating every core on a phone is how a demo turns into a thermal plateau: the
     * measured pattern on this hardware is a ~40% throughput loss within a few minutes of
     * sustained all-core load. Two threads keeps the device responsive, keeps the clocks
     * up, and finishes a slice of this size well inside a lease.
     */
    private var threadCount: Int32 {
        #if os(iOS)
        return 2
        #else
        return Int32(max(1, min(4, ProcessInfo.processInfo.processorCount / 2)))
        #endif
    }

    private func session(for modelHash: String, path: URL) throws -> ORTSession {
        if let existing = sessions[modelHash] { return existing }
        let environment = try env ?? ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        env = environment

        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(threadCount)
        let created = try ORTSession(env: environment, modelPath: path.path, sessionOptions: options)
        sessions[modelHash] = created
        return created
    }

    /// Frees model memory when the app is backgrounded or the system asks for space.
    public func releaseSessions() {
        sessions.removeAll()
    }

    public func run(_ rawInput: JSONValue, context: ExecContext) async throws -> JSONValue {
        guard let input = InferenceInput.parse(rawInput) else {
            throw AgentError.adapter("inference input was not the expected shape")
        }

        let loadStart = DispatchTime.now().uptimeNanoseconds
        async let modelURL = context.artifacts.fetchFile(
            sha256: input.modelHash, server: context.server,
            hostId: context.hostId, identity: context.identity)
        async let inputBytes = context.artifacts.fetch(
            sha256: input.inputsHash, server: context.server,
            hostId: context.hostId, identity: context.identity)

        let model = try await modelURL
        let bytes = try await inputBytes
        let session = try session(for: input.modelHash, path: model)
        let modelLoadMs = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1_000_000

        // ------------------------------------------------------------- layout

        guard bytes.count >= 8, bytes[0] == 0x44, bytes[1] == 0x57, bytes[2] == 0x50, bytes[3] == 0x49 else {
            throw AgentError.adapter("input artifact has an unexpected format")
        }
        let total = Int(bytes[4]) << 24 | Int(bytes[5]) << 16 | Int(bytes[6]) << 8 | Int(bytes[7])
        guard input.from + input.count <= total else {
            throw AgentError.adapter("slice \(input.from)+\(input.count) exceeds the \(total) available items")
        }
        let pixelsStart = 8
        let labelsStart = 8 + total * Self.pixels
        guard bytes.count >= labelsStart + total else {
            throw AgentError.adapter("input artifact is shorter than its own header claims")
        }

        // ----------------------------------------------------------- inference

        var predictions: [JSONValue] = []
        predictions.reserveCapacity(input.count)
        var correct = 0
        var logitChecksum = 0.0

        let inferStart = DispatchTime.now().uptimeNanoseconds
        let outputNames: Set<String> = [input.outputName]

        for i in 0..<input.count {
            try Task.checkCancellation()

            let offset = pixelsStart + (input.from + i) * Self.pixels
            var floats = [Float](repeating: 0, count: Self.pixels)
            // Preprocessing v1, recorded in the manifest so results stay comparable.
            for p in 0..<Self.pixels {
                floats[p] = Float(bytes[offset + p]) / 255.0
            }

            let tensorData = NSMutableData(bytes: floats, length: floats.count * MemoryLayout<Float>.size)
            let tensor = try ORTValue(tensorData: tensorData, elementType: .float,
                                      shape: [1, 1, 28, 28])
            let outputs = try session.run(withInputs: [input.inputName: tensor],
                                          outputNames: outputNames, runOptions: nil)
            guard let value = outputs[input.outputName] else {
                throw AgentError.adapter("model produced no output named \(input.outputName)")
            }
            let raw = try value.tensorData() as Data
            let logits = raw.withUnsafeBytes { buffer in
                Array(buffer.bindMemory(to: Float.self))
            }
            guard !logits.isEmpty else { throw AgentError.adapter("model produced an empty output") }

            var best = 0
            for c in 1..<logits.count where logits[c] > logits[best] { best = c }
            predictions.append(.int(best))
            logitChecksum += Double(logits[best])
            if Int(bytes[labelsStart + input.from + i]) == best { correct += 1 }

            if (i + 1) % 25 == 0 || i + 1 == input.count {
                context.onProgress(i + 1, input.count)
            }
        }
        let inferenceMs = Double(DispatchTime.now().uptimeNanoseconds - inferStart) / 1_000_000

        return .object([
            ("from", .int(input.from)),
            ("count", .int(input.count)),
            ("predictions", .array(predictions)),
            ("correct", .int(correct)),
            // Rounded so tiny floating-point differences between chip generations do not
            // read as disagreement. An A-series and an M-series will not produce
            // bit-identical logits; real divergence is far larger than this.
            ("logitChecksum", .double(roundTo(logitChecksum, 3))),
            ("hostId", .string(context.hostId)),
            ("hostname", .string(deviceHostname)),
            ("modelLoadMs", .double(roundTo(modelLoadMs, 1))),
            ("inferenceMs", .double(roundTo(inferenceMs, 1))),
            ("itemsPerSecond", .double(roundTo(Double(input.count) / (inferenceMs / 1000), 1))),
        ])
    }
}
