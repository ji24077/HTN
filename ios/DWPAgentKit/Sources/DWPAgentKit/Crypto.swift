import Foundation
import CryptoKit

// ------------------------------------------------------------------- base64url

extension Data {
    public var base64URLEncoded: String {
        base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    public init?(base64URLEncoded s: String) {
        var b = s.replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        while b.count % 4 != 0 { b += "=" }
        self.init(base64Encoded: b)
    }
}

// ------------------------------------------------------------------ timestamps

/**
 * `Date().toISOString()`, byte for byte.
 *
 * Always UTC, always exactly three fractional digits. The attestation payload is a
 * newline-joined string containing two of these, so a formatter that dropped a trailing
 * zero would produce a signature the control service computes differently and rejects
 * as forged — a failure that would read as an attack rather than a formatting bug.
 */
public enum ISO8601 {
    public static func string(from date: Date) -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let c = calendar.dateComponents([.year, .month, .day, .hour, .minute, .second, .nanosecond], from: date)
        let millis = Int((Double(c.nanosecond ?? 0) / 1_000_000).rounded(.down))
        return String(format: "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
                      c.year!, c.month!, c.day!, c.hour!, c.minute!, c.second!, millis)
    }

    public static func date(from string: String) -> Date? {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: string) ?? {
            let plain = ISO8601DateFormatter()
            plain.formatOptions = [.withInternetDateTime]
            return plain.date(from: string)
        }()
    }
}

// ------------------------------------------------------------------- assertion

/**
 * A short-lived EdDSA assertion, minted fresh for every connection and artifact fetch.
 *
 * There is no long-lived bearer token anywhere in this protocol: the control database
 * holds only public keys, so reading it cannot impersonate this device.
 */
public enum Assertion {
    public static let ttlSeconds = 120

    public static func mint(hostId: String, identity: HostIdentity,
                            audience: String = "dwp-control",
                            now: Date = Date()) throws -> String {
        let iat = Int(now.timeIntervalSince1970)
        let header = JSONValue.object([("alg", .string("EdDSA")), ("typ", .string("JWT"))])
        let claims = JSONValue.object([
            ("iss", .string(hostId)),
            ("aud", .string(audience)),
            ("iat", .int(iat)),
            ("exp", .int(iat + ttlSeconds)),
            ("jti", .string(UUID().uuidString.lowercased())),
        ])
        let signingInput = "\(header.data.base64URLEncoded).\(claims.data.base64URLEncoded)"
        let signature = try identity.sign(Data(signingInput.utf8))
        return "\(signingInput).\(signature.base64URLEncoded)"
    }
}

// ----------------------------------------------------------------- attestation

/**
 * The bytes a host signs when it reports a result.
 *
 * Field order is fixed and mirrors `attestation.ts` exactly. Both sides build this string
 * independently and a mismatch reads as a forged result, so this is the one place in the
 * iOS agent where "close enough" is indistinguishable from an attack.
 */
public enum Attestation {
    public struct Claim: Sendable {
        public let taskId: String
        public let attempt: Int
        public let hostId: String
        public let outputHash: String
        public let startedAt: String
        public let finishedAt: String

        public init(taskId: String, attempt: Int, hostId: String,
                    outputHash: String, startedAt: String, finishedAt: String) {
            self.taskId = taskId
            self.attempt = attempt
            self.hostId = hostId
            self.outputHash = outputHash
            self.startedAt = startedAt
            self.finishedAt = finishedAt
        }
    }

    public static func payload(_ c: Claim) -> Data {
        Data("dwp-attest/v1\n\(c.taskId)\n\(c.attempt)\n\(c.hostId)\n\(c.outputHash)\n\(c.startedAt)\n\(c.finishedAt)\n".utf8)
    }

    public static func sign(_ c: Claim, identity: HostIdentity) throws -> String {
        try identity.sign(payload(c)).base64URLEncoded
    }

    /// Present so the agent can check its own work in tests, and so the conformance
    /// suite can verify in both directions rather than only outward.
    public static func verify(_ c: Claim, signatureBase64URL: String, publicKeySpkiBase64: String) -> Bool {
        guard let spki = Data(base64Encoded: publicKeySpkiBase64), spki.count == 44,
              let signature = Data(base64URLEncoded: signatureBase64URL),
              let key = try? Curve25519.Signing.PublicKey(rawRepresentation: spki.suffix(32))
        else { return false }
        return key.isValidSignature(signature, for: payload(c))
    }
}

/// `sha256(JSON.stringify(output ?? null))`, hex — the same string the desktop agent hashes.
public func hashOutput(_ output: JSONValue) -> String {
    SHA256.hash(data: output.data).map { String(format: "%02x", $0) }.joined()
}

public func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
