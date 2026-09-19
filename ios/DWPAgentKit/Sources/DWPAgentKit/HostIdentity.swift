import Foundation
import CryptoKit

/**
 * This host's identity: an Ed25519 keypair whose private half never leaves the device.
 *
 * Secure Enclave is deliberately not used. It only holds P-256 keys, and the control
 * service verifies Ed25519 — so an Enclave-backed key would mean a second signature
 * algorithm in a protocol that has exactly one. The trade is stated plainly rather than
 * hidden: on iOS the key is a software key in the Keychain, protected by
 * `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly` (never synced, never backed up,
 * unreadable until the device has been unlocked once since boot). That is weaker than
 * hardware isolation and stronger than the mode-0600 file the desktop agent uses.
 */
public struct HostIdentity: Sendable {
    public let privateKey: Curve25519.Signing.PrivateKey

    public init(privateKey: Curve25519.Signing.PrivateKey) {
        self.privateKey = privateKey
    }

    /// The 44-byte SPKI DER encoding, base64 — exactly what the control service stores.
    public var publicKeySpkiBase64: String {
        Self.spki(from: privateKey.publicKey).base64EncodedString()
    }

    public func sign(_ message: Data) throws -> Data {
        try privateKey.signature(for: message)
    }

    // ------------------------------------------------------------------- DER

    /// `SEQUENCE { SEQUENCE { OID 1.3.101.112 }, BIT STRING { rawKey } }`
    static func spki(from publicKey: Curve25519.Signing.PublicKey) -> Data {
        var der = Data([0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00])
        der.append(publicKey.rawRepresentation)
        return der
    }

    /// `SEQUENCE { INTEGER 0, SEQUENCE { OID 1.3.101.112 }, OCTET STRING { OCTET STRING { seed } } }`
    static func pkcs8(from privateKey: Curve25519.Signing.PrivateKey) -> Data {
        var der = Data([0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70,
                        0x04, 0x22, 0x04, 0x20])
        der.append(privateKey.rawRepresentation)
        return der
    }

    static func seed(fromPKCS8 der: Data) -> Data? {
        // The prefix is fixed for Ed25519, so a length check plus a prefix match is a
        // complete parse — there are no optional fields to skip.
        let prefix: [UInt8] = [0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70,
                               0x04, 0x22, 0x04, 0x20]
        guard der.count == prefix.count + 32, Array(der.prefix(prefix.count)) == prefix else { return nil }
        return der.suffix(32)
    }

    // ------------------------------------------------------------------- PEM

    public var privateKeyPEM: String {
        let body = Self.pkcs8(from: privateKey).base64EncodedString()
        let wrapped = stride(from: 0, to: body.count, by: 64).map { offset -> String in
            let start = body.index(body.startIndex, offsetBy: offset)
            let end = body.index(start, offsetBy: min(64, body.count - offset))
            return String(body[start..<end])
        }.joined(separator: "\n")
        return "-----BEGIN PRIVATE KEY-----\n\(wrapped)\n-----END PRIVATE KEY-----\n"
    }

    /// Read the same PKCS#8 PEM the desktop agent writes, so a key can be moved or compared.
    public static func fromPEM(_ pem: String) throws -> HostIdentity {
        let body = pem
            .replacingOccurrences(of: "-----BEGIN PRIVATE KEY-----", with: "")
            .replacingOccurrences(of: "-----END PRIVATE KEY-----", with: "")
            .filter { !$0.isWhitespace }
        guard let der = Data(base64Encoded: body), let seed = seed(fromPKCS8: der) else {
            throw AgentError.badKey("not an Ed25519 PKCS#8 private key")
        }
        return HostIdentity(privateKey: try Curve25519.Signing.PrivateKey(rawRepresentation: seed))
    }

    public static func generate() -> HostIdentity {
        HostIdentity(privateKey: Curve25519.Signing.PrivateKey())
    }
}

public enum AgentError: Error, CustomStringConvertible {
    case badKey(String)
    case pairing(String)
    case artifact(String)
    case adapter(String)
    case cancelled
    case notConfigured

    public var description: String {
        switch self {
        case .badKey(let m): return "key: \(m)"
        case .pairing(let m): return m
        case .artifact(let m): return m
        case .adapter(let m): return m
        case .cancelled: return "cancelled"
        case .notConfigured: return "this device has not been paired yet"
        }
    }
}
