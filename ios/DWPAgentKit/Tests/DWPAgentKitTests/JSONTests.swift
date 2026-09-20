import XCTest
@testable import DWPAgentKit

/**
 * The serializer has one job that matters: produce the same bytes `JSON.stringify` does.
 *
 * Everything the agent signs is derived from those bytes, so a difference here does not
 * show up as a formatting nit — it shows up on the dashboard as `result.signature_invalid`,
 * which is the control service's word for "someone is forging results".
 */
final class JSONTests: XCTestCase {

    func testObjectKeysKeepInsertionOrder() {
        let value = JSONValue.object([
            ("zebra", .int(1)), ("apple", .int(2)), ("middle", .int(3)),
        ])
        XCTAssertEqual(value.stringify(), #"{"zebra":1,"apple":2,"middle":3}"#)
    }

    func testIntegralDoublesDropTheDecimalPoint() {
        // The single most likely divergence: Swift writes 1.0, JavaScript writes 1.
        XCTAssertEqual(jsNumber(1.0), "1")
        XCTAssertEqual(jsNumber(-4.0), "-4")
        XCTAssertEqual(jsNumber(0.0), "0")
        XCTAssertEqual(jsNumber(1000.0), "1000")
    }

    func testFractionalDoublesRoundTrip() {
        XCTAssertEqual(jsNumber(12.345), "12.345")
        XCTAssertEqual(jsNumber(0.1), "0.1")
        XCTAssertEqual(jsNumber(-0.5), "-0.5")
        XCTAssertEqual(jsNumber(1234.567), "1234.567")
    }

    func testNonFiniteBecomesNullAsJSONStringifyDoes() {
        XCTAssertEqual(jsNumber(.nan), "null")
        XCTAssertEqual(jsNumber(.infinity), "null")
    }

    func testStringEscapingMatchesJSONStringify() {
        XCTAssertEqual(jsString("a\"b"), #""a\"b""#)
        XCTAssertEqual(jsString("line\nbreak"), #""line\nbreak""#)
        XCTAssertEqual(jsString("tab\there"), #""tab\there""#)
        // JSON.stringify leaves the solidus alone and emits non-ASCII raw.
        XCTAssertEqual(jsString("a/b"), #""a/b""#)
        XCTAssertEqual(jsString("café"), "\"café\"")
        XCTAssertEqual(jsString("\u{01}"), #""\u0001""#)
    }

    func testParseRoundTripPreservesOrder() {
        let source = #"{"b":1,"a":{"d":[1,2.5,"x",true,null],"c":"y"}}"#
        guard let parsed = JSONValue.parse(Data(source.utf8)) else {
            return XCTFail("did not parse")
        }
        XCTAssertEqual(parsed.stringify(), source)
    }

    func testParseHandlesSurrogatePairs() {
        guard let parsed = JSONValue.parse(Data(#"{"e":"\ud83d\ude00"}"#.utf8)) else {
            return XCTFail("did not parse")
        }
        XCTAssertEqual(parsed["e"]?.stringValue, "😀")
    }

    /// A peer must never be able to crash us — the same rule `decode()` follows in the
    /// TypeScript protocol. These are all inputs a hostile or broken server could send.
    func testMalformedInputReturnsNilRatherThanTrapping() {
        let bad = ["", "{", "[1,", "{\"a\"}", "{\"a\":}", "nul", "\"unterminated",
                   "{\"a\":1}}", "[[[[[[[[[[", "\\", "{\"a\":\"\\u00\"}"]
        for source in bad {
            XCTAssertNil(JSONValue.parse(Data(source.utf8)), "expected nil for \(source.debugDescription)")
        }
    }

    func testDeeplyNestedInputDoesNotBlowTheStack() {
        // 512 is well inside what a recursive parser handles and well past anything the
        // real protocol contains; the point is that a big input fails cleanly.
        let deep = String(repeating: "[", count: 512) + String(repeating: "]", count: 512)
        XCTAssertNotNil(JSONValue.parse(Data(deep.utf8)))
    }

    func testAccessorsAreForgivingAboutIntegerRepresentation() {
        // The server writes `1` where zod declared a number; both must read as Int.
        XCTAssertEqual(JSONValue.parse(Data(#"{"n":5}"#.utf8))?["n"]?.intValue, 5)
        XCTAssertEqual(JSONValue.parse(Data(#"{"n":5.0}"#.utf8))?["n"]?.intValue, 5)
        XCTAssertEqual(JSONValue.parse(Data(#"{"n":5}"#.utf8))?["n"]?.doubleValue, 5.0)
    }
}
