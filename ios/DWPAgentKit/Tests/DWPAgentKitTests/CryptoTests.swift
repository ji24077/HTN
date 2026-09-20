import XCTest
import CryptoKit
@testable import DWPAgentKit

/**
 * Identity and signing.
 *
 * These assert byte-level shapes rather than "it signs something", because every one of
 * these values is consumed by a different language. A key exported in the wrong DER
 * envelope pairs successfully and then fails every connection afterwards, which is a
 * miserable thing to debug from a phone.
 */
final class CryptoTests: XCTestCase {

    func testSPKIIsTheCanonical44ByteEd25519Encoding() {
        let identity = HostIdentity.generate()
        guard let der = Data(base64Encoded: identity.publicKeySpkiBase64) else {
            return XCTFail("public key was not base64")
        }
        XCTAssertEqual(der.count, 44)
        XCTAssertEqual(Array(der.prefix(12)),
                       [0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00])
        XCTAssertEqual(Array(der.suffix(32)), Array(identity.privateKey.publicKey.rawRepresentation))
    }

    func testPEMRoundTripsThroughTheDesktopAgentsFormat() throws {
        let original = HostIdentity.generate()
        let restored = try HostIdentity.fromPEM(original.privateKeyPEM)
        XCTAssertEqual(restored.publicKeySpkiBase64, original.publicKeySpkiBase64)
        XCTAssertTrue(original.privateKeyPEM.hasPrefix("-----BEGIN PRIVATE KEY-----\n"))
        XCTAssertTrue(original.privateKeyPEM.hasSuffix("-----END PRIVATE KEY-----\n"))
    }

    func testPEMRejectsThingsThatAreNotEd25519Keys() {
        XCTAssertThrowsError(try HostIdentity.fromPEM("-----BEGIN PRIVATE KEY-----\nQUJD\n-----END PRIVATE KEY-----"))
        XCTAssertThrowsError(try HostIdentity.fromPEM("not a key at all"))
        XCTAssertThrowsError(try HostIdentity.fromPEM(""))
    }

    func testAttestationPayloadIsExactlyTheDocumentedString() {
        let claim = Attestation.Claim(
            taskId: "t1", attempt: 2, hostId: "h1", outputHash: "abc",
            startedAt: "2026-09-18T00:00:00.000Z", finishedAt: "2026-09-18T00:00:01.000Z")
        let expected = "dwp-attest/v1\nt1\n2\nh1\nabc\n2026-09-18T00:00:00.000Z\n2026-09-18T00:00:01.000Z\n"
        XCTAssertEqual(String(data: Attestation.payload(claim), encoding: .utf8), expected)
    }

    func testAttestationVerifiesAndRejectsTampering() throws {
        let identity = HostIdentity.generate()
        let claim = Attestation.Claim(
            taskId: "t1", attempt: 1, hostId: "h1", outputHash: "f00d",
            startedAt: "2026-09-18T00:00:00.000Z", finishedAt: "2026-09-18T00:00:01.000Z")
        let signature = try Attestation.sign(claim, identity: identity)

        XCTAssertTrue(Attestation.verify(claim, signatureBase64URL: signature,
                                         publicKeySpkiBase64: identity.publicKeySpkiBase64))

        // Every field is covered by the signature, so changing any one must break it.
        let tampered = Attestation.Claim(
            taskId: "t1", attempt: 1, hostId: "h1", outputHash: "f00e",
            startedAt: "2026-09-18T00:00:00.000Z", finishedAt: "2026-09-18T00:00:01.000Z")
        XCTAssertFalse(Attestation.verify(tampered, signatureBase64URL: signature,
                                          publicKeySpkiBase64: identity.publicKeySpkiBase64))

        // A different host's key must not validate this host's work.
        let other = HostIdentity.generate()
        XCTAssertFalse(Attestation.verify(claim, signatureBase64URL: signature,
                                          publicKeySpkiBase64: other.publicKeySpkiBase64))
    }

    func testAttestationVerifyIsTotalOnGarbageInput() {
        let claim = Attestation.Claim(taskId: "t", attempt: 1, hostId: "h", outputHash: "x",
                                      startedAt: "a", finishedAt: "b")
        XCTAssertFalse(Attestation.verify(claim, signatureBase64URL: "!!!", publicKeySpkiBase64: "!!!"))
        XCTAssertFalse(Attestation.verify(claim, signatureBase64URL: "", publicKeySpkiBase64: ""))
        XCTAssertFalse(Attestation.verify(claim, signatureBase64URL: "AAAA", publicKeySpkiBase64: "AAAA"))
    }

    func testAssertionHasThreePartsAndPinsTheAlgorithm() throws {
        let identity = HostIdentity.generate()
        let token = try Assertion.mint(hostId: "host-abc", identity: identity)
        let parts = token.split(separator: ".").map(String.init)
        XCTAssertEqual(parts.count, 3)

        guard let headerData = Data(base64URLEncoded: parts[0]),
              let header = JSONValue.parse(headerData),
              let claimsData = Data(base64URLEncoded: parts[1]),
              let claims = JSONValue.parse(claimsData)
        else { return XCTFail("token parts were not base64url JSON") }

        XCTAssertEqual(header["alg"]?.stringValue, "EdDSA")
        XCTAssertEqual(header["typ"]?.stringValue, "JWT")
        XCTAssertEqual(claims["iss"]?.stringValue, "host-abc")
        XCTAssertEqual(claims["aud"]?.stringValue, "dwp-control")

        let iat = try XCTUnwrap(claims["iat"]?.intValue)
        let exp = try XCTUnwrap(claims["exp"]?.intValue)
        XCTAssertEqual(exp - iat, Assertion.ttlSeconds)

        // The signature must cover exactly "header.claims".
        let signature = try XCTUnwrap(Data(base64URLEncoded: parts[2]))
        XCTAssertTrue(identity.privateKey.publicKey.isValidSignature(
            signature, for: Data("\(parts[0]).\(parts[1])".utf8)))
    }

    func testAssertionUsesAFreshNonceEachTime() throws {
        let identity = HostIdentity.generate()
        let first = try Assertion.mint(hostId: "h", identity: identity)
        let second = try Assertion.mint(hostId: "h", identity: identity)
        // The control service rejects a replayed jti, so two connects in the same second
        // must not produce the same token.
        XCTAssertNotEqual(first, second)
    }

    func testBase64URLHasNoPaddingOrUnsafeCharacters() {
        for _ in 0..<200 {
            let bytes = Data((0..<Int.random(in: 1...64)).map { _ in UInt8.random(in: 0...255) })
            let encoded = bytes.base64URLEncoded
            XCTAssertFalse(encoded.contains("="))
            XCTAssertFalse(encoded.contains("+"))
            XCTAssertFalse(encoded.contains("/"))
            XCTAssertEqual(Data(base64URLEncoded: encoded), bytes)
        }
    }

    func testISO8601MatchesToISOStringShape() {
        let date = Date(timeIntervalSince1970: 1_789_000_000.5)
        let text = ISO8601.string(from: date)
        XCTAssertEqual(text.count, 24)
        XCTAssertTrue(text.hasSuffix("Z"))
        // Always three fractional digits, even when they are zeros.
        XCTAssertEqual(ISO8601.string(from: Date(timeIntervalSince1970: 0)), "1970-01-01T00:00:00.000Z")
        XCTAssertNotNil(ISO8601.date(from: text))
    }

    func testHashOutputIsSHA256OfTheSerializedForm() {
        let output = JSONValue.object([("a", .int(1)), ("b", .string("x"))])
        XCTAssertEqual(hashOutput(output), sha256Hex(Data(#"{"a":1,"b":"x"}"#.utf8)))
        XCTAssertEqual(hashOutput(output).count, 64)
    }
}
