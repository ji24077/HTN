import Foundation

/**
 * The wire protocol, mirrored from `packages/protocol/src`.
 *
 * Kept as explicit constructors rather than Codable structs because object key order is
 * load-bearing for anything that gets hashed, and because a decoder that throws on a
 * peer's malformed frame would hand a remote party a crash. Every parse here returns nil
 * instead — the same rule `decode()` follows on the TypeScript side.
 */

public struct Envelope: Sendable {
    public let v: Int
    public let id: String
    public let replyTo: String?
    public let ts: String
    public let type: String
    public let payload: JSONValue

    public init(type: String, payload: JSONValue, replyTo: String? = nil) {
        self.v = 1
        self.id = UUID().uuidString.lowercased()
        self.ts = ISO8601.string(from: Date())
        self.type = type
        self.payload = payload
        self.replyTo = replyTo
    }

    private init(v: Int, id: String, replyTo: String?, ts: String, type: String, payload: JSONValue) {
        self.v = v; self.id = id; self.replyTo = replyTo; self.ts = ts; self.type = type; self.payload = payload
    }

    public var json: JSONValue {
        var pairs: [(key: String, value: JSONValue)] = [
            ("v", .int(v)), ("id", .string(id)), ("ts", .string(ts)),
            ("type", .string(type)), ("payload", payload),
        ]
        if let replyTo { pairs.append(("replyTo", .string(replyTo))) }
        return .object(pairs)
    }

    public var text: String { json.stringify() }

    /// Returns nil rather than throwing: a peer must never be able to crash us.
    public static func decode(_ data: Data) -> Envelope? {
        guard let value = JSONValue.parse(data),
              case .object = value,
              let v = value["v"]?.intValue, v == 1,
              let id = value["id"]?.stringValue, !id.isEmpty,
              let ts = value["ts"]?.stringValue,
              let type = value["type"]?.stringValue, !type.isEmpty
        else { return nil }
        return Envelope(v: v, id: id, replyTo: value["replyTo"]?.stringValue,
                        ts: ts, type: type, payload: value["payload"] ?? .null)
    }
}

// ------------------------------------------------------------------ capability

public struct CapabilityRecord: Sendable {
    public var agentVersion: String
    public var os: String
    public var arch: String
    public var cpuModel: String
    public var logicalCores: Int
    public var totalRamMb: Int
    public var freeRamMb: Int
    public var adapters: [String]

    public var json: JSONValue {
        .object([
            ("agentVersion", .string(agentVersion)),
            ("os", .string(os)),
            ("arch", .string(arch)),
            ("cpuModel", .string(cpuModel)),
            ("logicalCores", .int(logicalCores)),
            ("totalRamMb", .int(totalRamMb)),
            ("freeRamMb", .int(freeRamMb)),
            ("adapters", .array(adapters.map { .string($0) })),
        ])
    }
}

public struct ConsentState: Sendable {
    public var paused: Bool
    public var allowCompute: Bool
    public var allowBrowser: Bool
    public var maxConcurrency: Int

    public var json: JSONValue {
        .object([
            ("paused", .bool(paused)),
            ("allowCompute", .bool(allowCompute)),
            ("allowBrowser", .bool(allowBrowser)),
            ("maxConcurrency", .int(maxConcurrency)),
        ])
    }
}

// -------------------------------------------------------------- control → agent

public struct HelloAck: Sendable {
    public let hostId: String
    public let serverTime: String
    public let heartbeatSeconds: Double
    public let releaseVersion: String?

    public static func parse(_ v: JSONValue) -> HelloAck? {
        guard let hostId = v["hostId"]?.stringValue,
              let serverTime = v["serverTime"]?.stringValue,
              let heartbeat = v["heartbeatSeconds"]?.doubleValue
        else { return nil }
        return HelloAck(hostId: hostId, serverTime: serverTime,
                        heartbeatSeconds: heartbeat, releaseVersion: v["releaseVersion"]?.stringValue)
    }
}

public struct TaskOffer: Sendable {
    public let taskId: String
    public let jobId: String
    public let adapter: String
    public let attempt: Int
    public let input: JSONValue
    public let leaseId: String
    public let leaseSeconds: Int
    public let wallClockMs: Int

    public static func parse(_ v: JSONValue) -> TaskOffer? {
        guard let taskId = v["taskId"]?.stringValue,
              let jobId = v["jobId"]?.stringValue,
              let adapter = v["adapter"]?.stringValue,
              let attempt = v["attempt"]?.intValue, attempt > 0,
              let leaseId = v["leaseId"]?.stringValue,
              let leaseSeconds = v["leaseSeconds"]?.intValue, leaseSeconds > 0,
              let wallClockMs = v["wallClockMs"]?.intValue, wallClockMs > 0
        else { return nil }
        return TaskOffer(taskId: taskId, jobId: jobId, adapter: adapter, attempt: attempt,
                         input: v["input"] ?? .null, leaseId: leaseId,
                         leaseSeconds: leaseSeconds, wallClockMs: wallClockMs)
    }
}

// -------------------------------------------------------------- adapter inputs

public struct EchoInput: Sendable {
    public let nonce: String
    public let sleepMs: Int

    public static func parse(_ v: JSONValue) -> EchoInput? {
        guard let nonce = v["nonce"]?.stringValue else { return nil }
        let sleepMs = v["sleepMs"]?.intValue ?? 0
        guard sleepMs >= 0 else { return nil }
        return EchoInput(nonce: nonce, sleepMs: sleepMs)
    }
}

public struct InferenceInput: Sendable {
    public let modelHash: String
    public let inputsHash: String
    public let inputName: String
    public let outputName: String
    public let from: Int
    public let count: Int

    public static func parse(_ v: JSONValue) -> InferenceInput? {
        guard let modelHash = v["modelHash"]?.stringValue, modelHash.count == 64,
              let inputsHash = v["inputsHash"]?.stringValue, inputsHash.count == 64,
              let inputName = v["inputName"]?.stringValue,
              let outputName = v["outputName"]?.stringValue,
              let from = v["from"]?.intValue, from >= 0,
              let count = v["count"]?.intValue, count > 0,
              v["preprocessing"]?.stringValue == "v1"
        else { return nil }
        return InferenceInput(modelHash: modelHash, inputsHash: inputsHash,
                              inputName: inputName, outputName: outputName,
                              from: from, count: count)
    }
}

/**
 * One slice of a generation: evaluate these candidate gaits.
 *
 * Only the parent genome and a list of seeds travel. Each host rebuilds the candidates
 * itself from parent + seed, so a generation of hundreds costs one genome of bandwidth
 * rather than hundreds — which matters more on a phone than anywhere else in the fleet.
 */
public struct WalkerInput: Sendable {
    public let generation: Int
    public let parent: [Double]
    public let sigma: Double
    public let seeds: [Int]
    public let steps: Int

    public static func parse(_ v: JSONValue) -> WalkerInput? {
        guard let generation = v["generation"]?.intValue, generation >= 0,
              let parentRaw = v["parent"]?.arrayValue,
              let sigma = v["sigma"]?.doubleValue, sigma > 0,
              let seedsRaw = v["seeds"]?.arrayValue,
              let steps = v["steps"]?.intValue, steps > 0, steps <= 5000
        else { return nil }
        let parent = parentRaw.compactMap { $0.doubleValue }
        let seeds = seedsRaw.compactMap { $0.intValue }
        guard parent.count == parentRaw.count, seeds.count == seedsRaw.count else { return nil }
        return WalkerInput(generation: generation, parent: parent, sigma: sigma,
                           seeds: seeds, steps: steps)
    }
}
