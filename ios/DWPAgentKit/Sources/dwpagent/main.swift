import Foundation
import DWPAgentKit

/**
 * The same agent core, with a terminal instead of a screen.
 *
 * This exists so the iOS agent can be tested the way the desktop one already is — by the
 * network simulator, against a real control service, over a real socket, with real
 * failures injected. Every line of decision-making it exercises is the same code the
 * phone runs; only the shell differs. Without it, "is the connection reliable?" could
 * only be answered by holding a phone and watching.
 */

let arguments = Array(CommandLine.arguments.dropFirst())
let command = arguments.first ?? ""

func flag(_ name: String) -> String? {
    guard let i = arguments.firstIndex(of: "--\(name)"), i + 1 < arguments.count else { return nil }
    return arguments[i + 1]
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

let store = FileKeyStore()

switch command {
case "pair":
    guard let server = flag("server"), let code = flag("code") else {
        fail("usage: dwpagent pair --server <url> --code <CODE> [--label <name>]")
    }
    let label = flag("label") ?? Capability.deviceModel
    do {
        let result = try await Pairing.pair(server: server, code: code, label: label, store: store)
        print("Paired as \"\(result.config.label)\"")
        print("  host id : \(result.config.hostId)")
        print("  control : \(result.config.server)")
    } catch {
        fail("\(error)")
    }

case "run":
    guard let config = try store.loadConfig(), let identity = try store.loadIdentity() else {
        fail("No agent config. Pair first:\n  dwpagent pair --server <url> --code <CODE>")
    }
    print("[agent] \(config.label) (\(config.hostId)) v\(agentVersion) pid=\(ProcessInfo.processInfo.processIdentifier)")

    let artifactDir = store.home.appendingPathComponent("artifacts")
    let connection = AgentConnection(
        config: config, identity: identity,
        artifacts: ArtifactStore(directory: artifactDir))

    // Line-buffered and timestamped, so the simulator's log reader can follow a Swift
    // agent the same way it follows a Node one.
    await connection.observe(
        state: { state in
            FileHandle.standardOutput.write(Data("[state] \(state)\n".utf8))
        },
        events: { event in
            FileHandle.standardOutput.write(
                Data("[\(ISO8601.string(from: event.at))] \(event.type) \(event.detail)\n".utf8))
        },
        progress: { taskId, done, total in
            FileHandle.standardOutput.write(Data("[progress] \(taskId.prefix(8)) \(done)/\(total)\n".utf8))
        })

    await connection.start()

    // Park forever; the connection owns its own lifecycle from here.
    while true {
        try? await Task.sleep(nanoseconds: 1_000_000_000)
    }

case "whoami":
    guard let config = try store.loadConfig(), let identity = try store.loadIdentity() else {
        fail("not paired")
    }
    print(JSONValue.object([
        ("hostId", .string(config.hostId)),
        ("label", .string(config.label)),
        ("server", .string(config.server)),
        ("publicKey", .string(identity.publicKeySpkiBase64)),
        ("os", .string(Capability.osName)),
        ("adapters", .array(AgentConnection.adapters.map { .string($0) })),
    ]).stringify())

/// Emits the vectors the Node conformance test verifies. Signing is the one place where
/// "looks right" and "is right" are indistinguishable without checking across languages.
case "selftest-vectors":
    let identity = try HostIdentity.fromPEM(
        String(data: FileHandle.standardInput.readDataToEndOfFile(), encoding: .utf8) ?? "")
    let claim = Attestation.Claim(
        taskId: flag("task") ?? "task-1", attempt: Int(flag("attempt") ?? "1") ?? 1,
        hostId: flag("host") ?? "host-1", outputHash: flag("hash") ?? String(repeating: "a", count: 64),
        startedAt: flag("started") ?? "2026-09-18T00:00:00.000Z",
        finishedAt: flag("finished") ?? "2026-09-18T00:00:01.000Z")
    print(JSONValue.object([
        ("publicKey", .string(identity.publicKeySpkiBase64)),
        ("assertion", .string(try Assertion.mint(hostId: flag("host") ?? "host-1", identity: identity))),
        ("attestation", .string(try Attestation.sign(claim, identity: identity))),
    ]).stringify())

/// Round-trips a JSON document through the ordered parser and serializer, so Node can
/// diff the result against its own `JSON.stringify`.
case "selftest-json":
    let input = FileHandle.standardInput.readDataToEndOfFile()
    guard let value = JSONValue.parse(input) else { fail("could not parse stdin as JSON") }
    print(value.stringify())

default:
    print("""
    dwpagent — the iOS agent core, on the command line

      pair --server <url> --code <CODE> [--label <name>]
      run
      whoami
      selftest-vectors      signing vectors, for cross-language checks
      selftest-json         ordered JSON round-trip, for cross-language checks

    DWP_HOME isolates state, so several can run side by side.
    """)
}
