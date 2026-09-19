import XCTest
@testable import DWPAgentKit

/**
 * Frame handling.
 *
 * Every one of these inputs is something a compromised or simply broken control service
 * could put on the wire. The contract is that none of them produce a crash and none of
 * them produce a half-valid object the agent then acts on.
 */
final class ProtocolTests: XCTestCase {

    func testEnvelopeRoundTrip() throws {
        let sent = Envelope(type: "hello", payload: .object([("a", .int(1))]))
        let received = try XCTUnwrap(Envelope.decode(Data(sent.text.utf8)))
        XCTAssertEqual(received.type, "hello")
        XCTAssertEqual(received.v, 1)
        XCTAssertEqual(received.id, sent.id)
        XCTAssertEqual(received.payload["a"]?.intValue, 1)
    }

    func testEnvelopeCarriesReplyToOnlyWhenSet() {
        XCTAssertFalse(Envelope(type: "x", payload: .null).text.contains("replyTo"))
        XCTAssertTrue(Envelope(type: "x", payload: .null, replyTo: "abc").text.contains(#""replyTo":"abc""#))
    }

    func testEnvelopeRejectsFramesThatAreNotOurs() {
        let bad = [
            "",                                            // empty
            "null",                                        // valid JSON, wrong shape
            "[]",
            #"{"v":2,"id":"a","ts":"t","type":"x"}"#,       // wrong version
            #"{"id":"a","ts":"t","type":"x"}"#,             // no version
            #"{"v":1,"id":"","ts":"t","type":"x"}"#,        // empty id
            #"{"v":1,"id":"a","ts":"t","type":""}"#,        // empty type
            #"{"v":1,"id":"a","ts":"t"}"#,                  // no type
            "not json at all",
        ]
        for frame in bad {
            XCTAssertNil(Envelope.decode(Data(frame.utf8)), "expected nil for \(frame.debugDescription)")
        }
    }

    func testEnvelopeToleratesAnAbsentPayload() throws {
        let decoded = try XCTUnwrap(Envelope.decode(Data(#"{"v":1,"id":"a","ts":"t","type":"ping"}"#.utf8)))
        XCTAssertEqual(decoded.payload, .null)
    }

    // ------------------------------------------------------------- task offers

    private func offerJSON(_ overrides: [String: JSONValue] = [:]) -> JSONValue {
        var pairs: [(key: String, value: JSONValue)] = [
            ("taskId", .string("t1")), ("jobId", .string("j1")), ("adapter", .string("echo")),
            ("attempt", .int(1)), ("input", .object([("nonce", .string("n"))])),
            ("leaseId", .string("l1")), ("leaseSeconds", .int(30)), ("wallClockMs", .int(60000)),
        ]
        for (key, value) in overrides {
            if let i = pairs.firstIndex(where: { $0.key == key }) { pairs[i] = (key, value) }
        }
        return .object(pairs)
    }

    func testTaskOfferParsesAValidOffer() throws {
        let offer = try XCTUnwrap(TaskOffer.parse(offerJSON()))
        XCTAssertEqual(offer.taskId, "t1")
        XCTAssertEqual(offer.adapter, "echo")
        XCTAssertEqual(offer.leaseSeconds, 30)
    }

    func testTaskOfferRejectsOutOfRangeValues() {
        // A zero lease or a zero wall clock would mean "you have no time to do this",
        // which is not a smaller task — it is a malformed one.
        XCTAssertNil(TaskOffer.parse(offerJSON(["attempt": .int(0)])))
        XCTAssertNil(TaskOffer.parse(offerJSON(["leaseSeconds": .int(0)])))
        XCTAssertNil(TaskOffer.parse(offerJSON(["wallClockMs": .int(0)])))
        XCTAssertNil(TaskOffer.parse(offerJSON(["taskId": .int(5)])))
        XCTAssertNil(TaskOffer.parse(.null))
        XCTAssertNil(TaskOffer.parse(.object([])))
    }

    // -------------------------------------------------------- adapter inputs

    func testEchoInputDefaultsSleepToZero() throws {
        let input = try XCTUnwrap(EchoInput.parse(.object([("nonce", .string("abc"))])))
        XCTAssertEqual(input.sleepMs, 0)
        XCTAssertNil(EchoInput.parse(.object([("sleepMs", .int(5))])))
        XCTAssertNil(EchoInput.parse(.object([("nonce", .string("a")), ("sleepMs", .int(-1))])))
    }

    private func inferenceJSON(_ overrides: [String: JSONValue] = [:]) -> JSONValue {
        let hash = String(repeating: "a", count: 64)
        var pairs: [(key: String, value: JSONValue)] = [
            ("modelHash", .string(hash)), ("inputsHash", .string(hash)),
            ("inputName", .string("Input3")), ("outputName", .string("Plus214_Output_0")),
            ("from", .int(0)), ("count", .int(10)), ("preprocessing", .string("v1")),
        ]
        for (key, value) in overrides {
            if let i = pairs.firstIndex(where: { $0.key == key }) { pairs[i] = (key, value) }
        }
        return .object(pairs)
    }

    func testInferenceInputParses() throws {
        let input = try XCTUnwrap(InferenceInput.parse(inferenceJSON()))
        XCTAssertEqual(input.from, 0)
        XCTAssertEqual(input.count, 10)
    }

    func testInferenceInputRejectsBadHashesAndRanges() {
        XCTAssertNil(InferenceInput.parse(inferenceJSON(["modelHash": .string("short")])))
        XCTAssertNil(InferenceInput.parse(inferenceJSON(["count": .int(0)])))
        XCTAssertNil(InferenceInput.parse(inferenceJSON(["from": .int(-1)])))
        // An unknown preprocessing version means the results would not be comparable with
        // every other host's, which is worse than not running it at all.
        XCTAssertNil(InferenceInput.parse(inferenceJSON(["preprocessing": .string("v2")])))
    }

    func testHelloAckParses() throws {
        let ack = try XCTUnwrap(HelloAck.parse(.object([
            ("hostId", .string("h")), ("serverTime", .string("2026-09-18T00:00:00.000Z")),
            ("heartbeatSeconds", .int(15)), ("releaseVersion", .null),
        ])))
        XCTAssertEqual(ack.heartbeatSeconds, 15)
        XCTAssertNil(ack.releaseVersion)
        XCTAssertNil(HelloAck.parse(.object([("hostId", .string("h"))])))
    }

    // ----------------------------------------------------------------- invites

    func testInviteLinkParsing() {
        let invite = Pairing.parseInvite("https://demo.example.com/join?code=abcd-1234")
        XCTAssertEqual(invite?.server, "https://demo.example.com")
        XCTAssertEqual(invite?.code, "ABCD-1234")

        // A port must survive; a tunnel or a laptop on the LAN will have one.
        XCTAssertEqual(Pairing.parseInvite("http://192.168.1.5:8787/join?code=AAAA-BBBB")?.server,
                       "http://192.168.1.5:8787")
        // Whitespace, because this arrives via paste.
        XCTAssertEqual(Pairing.parseInvite("  https://x.dev/join?code=Q-1  \n")?.code, "Q-1")
    }

    func testInviteLinkRejectsAnythingWithoutACode() {
        XCTAssertNil(Pairing.parseInvite("https://demo.example.com/join"))
        XCTAssertNil(Pairing.parseInvite("not a url"))
        XCTAssertNil(Pairing.parseInvite(""))
        // Non-web schemes would let a link point the agent somewhere it cannot dial.
        XCTAssertNil(Pairing.parseInvite("file:///etc/passwd?code=X"))
        XCTAssertNil(Pairing.parseInvite("javascript:alert(1)?code=X"))
    }
}
