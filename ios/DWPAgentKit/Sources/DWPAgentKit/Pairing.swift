import Foundation

/**
 * Enrollment: exchange a short-lived code for a host identity.
 *
 * The private half of the keypair is generated here and never transmitted — the server
 * is told only the SPKI public key, which is why a read of the control database cannot
 * impersonate this device or forge its results.
 */
public enum Pairing {

    /// What the owner actually sends: `https://host/join?code=ABCD-1234`.
    ///
    /// Typing a code on a phone keyboard is exactly the friction this app exists to
    /// remove, so the link is the primary path and the manual fields are the fallback.
    public struct Invite: Sendable, Equatable {
        public let server: String
        public let code: String
    }

    public static func parseInvite(_ text: String) -> Invite? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed),
              let scheme = url.scheme, scheme == "http" || scheme == "https",
              let host = url.host
        else { return nil }

        let components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        guard let code = components?.queryItems?.first(where: { $0.name == "code" })?.value,
              !code.isEmpty
        else { return nil }

        var origin = "\(scheme)://\(host)"
        if let port = url.port { origin += ":\(port)" }
        return Invite(server: origin, code: code.uppercased())
    }

    public struct Result: Sendable {
        public let config: AgentConfig
        public let identity: HostIdentity
    }

    public static func pair(server: String, code: String, label: String,
                            store: KeyStore, session: URLSession = .shared) async throws -> Result {
        let origin = server.replacingOccurrences(of: "/+$", with: "", options: .regularExpression)

        // Reuse an existing identity if this device has one. Generating a second keypair
        // would enroll the same phone twice and leave an orphan host row online forever.
        let identity = try store.loadIdentity() ?? {
            let fresh = HostIdentity.generate()
            try store.saveIdentity(fresh)
            return fresh
        }()

        guard let url = URL(string: "\(origin)/hosts/pair") else {
            throw AgentError.pairing("that does not look like a server address")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 30
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = JSONValue.object([
            ("code", .string(code.trimmingCharacters(in: .whitespaces).uppercased())),
            ("publicKey", .string(identity.publicKeySpkiBase64)),
            ("label", .string(label)),
        ]).data

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw AgentError.pairing(
                "Could not reach \(origin).\n\nCheck the address is right and that this device is on a network that can reach it.")
        }

        let status = (response as? HTTPURLResponse)?.statusCode ?? -1
        guard status == 200 else {
            switch status {
            case 400:
                throw AgentError.pairing(
                    "That pairing code was not accepted.\n\nCodes expire ten minutes after they are created and work only once. Ask for a fresh link.")
            case 429:
                throw AgentError.pairing("Too many attempts. Wait a few minutes and try again.")
            default:
                throw AgentError.pairing("Pairing failed (HTTP \(status)).")
            }
        }

        guard let body = JSONValue.parse(data),
              let hostId = body["hostId"]?.stringValue,
              let assigned = body["label"]?.stringValue,
              let wsUrl = body["wsUrl"]?.stringValue
        else { throw AgentError.pairing("The server's reply was not in the expected form.") }

        let config = AgentConfig(
            server: origin, wsUrl: wsUrl, hostId: hostId, label: assigned,
            allowCompute: true, allowBrowser: false,
            // One at a time, deliberately. A phone that accepts two slices at once heats
            // up faster and finishes neither sooner.
            maxConcurrency: 1, paused: false)
        try store.saveConfig(config)
        return Result(config: config, identity: identity)
    }
}
