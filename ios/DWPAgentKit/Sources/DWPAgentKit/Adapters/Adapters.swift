import Foundation

/// Rounds the way `Number(x.toFixed(n))` does, so a phone's timings serialize the same
/// shape as a laptop's and the two can be compared without special-casing either.
func roundTo(_ value: Double, _ places: Int) -> Double {
    guard value.isFinite else { return 0 }
    let factor = pow(10.0, Double(places))
    return (value * factor).rounded() / factor
}

public struct ExecContext: Sendable {
    public let hostId: String
    public let server: String
    public let identity: HostIdentity
    public let artifacts: ArtifactStore
    /// Called as items complete, so the UI and `BGContinuedProcessingTask` both have
    /// something real to report. iOS expires a continued task that goes quiet.
    public let onProgress: @Sendable (Int, Int) -> Void

    public init(hostId: String, server: String, identity: HostIdentity,
                artifacts: ArtifactStore,
                onProgress: @escaping @Sendable (Int, Int) -> Void = { _, _ in }) {
        self.hostId = hostId
        self.server = server
        self.identity = identity
        self.artifacts = artifacts
        self.onProgress = onProgress
    }
}

public var deviceHostname: String {
    #if os(iOS)
    // The device model, not `ProcessInfo.hostName`. On a phone that property is either
    // useless or actively wrong — in the Simulator it returns the Mac's hostname, which
    // would put a laptop's name on results a phone computed. The model identifier is
    // honest, stable, and not a name the owner chose, so it leaks nothing personal.
    return Capability.deviceModel
    #else
    let name = ProcessInfo.processInfo.hostName
    if name.isEmpty || name == "localhost" { return Capability.deviceModel }
    return name
    #endif
}

// ------------------------------------------------------------------------ echo

/**
 * Computes nothing on purpose.
 *
 * Its whole job is to prove that this specific physical device executed a specific
 * server-issued nonce — which is the only claim a first connection actually needs to
 * make, and the fastest way to tell a working phone from a plausible-looking one.
 */
public enum EchoAdapter {
    public static let name = "echo"

    public static func run(_ rawInput: JSONValue, hostId: String) async throws -> JSONValue {
        guard let input = EchoInput.parse(rawInput) else {
            throw AgentError.adapter("echo input was not the expected shape")
        }
        let started = DispatchTime.now().uptimeNanoseconds
        if input.sleepMs > 0 {
            try await Task.sleep(nanoseconds: UInt64(input.sleepMs) * 1_000_000)
        }
        let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000

        return .object([
            ("nonce", .string(input.nonce)),
            ("hostId", .string(hostId)),
            ("hostname", .string(deviceHostname)),
            ("os", .string(Capability.osName)),
            ("arch", .string(Capability.architecture)),
            ("elapsedMs", .double(roundTo(elapsedMs, 3))),
        ])
    }
}
