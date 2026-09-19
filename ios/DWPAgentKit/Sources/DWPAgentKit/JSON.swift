import Foundation

/**
 * A JSON value with ordered objects, and a serializer that matches `JSON.stringify`.
 *
 * Ordering matters here for one reason: the agent hashes its own output and signs that
 * hash, and the whole provenance story rests on anyone being able to recompute it. A
 * dictionary would reorder keys per process launch, so two runs of the same task on the
 * same phone would produce different hashes for identical results.
 *
 * Number formatting is the other half of the same problem — JavaScript writes `1` where
 * Swift writes `1.0`, and a reviewer recomputing the hash in Node would find a mismatch
 * that looks exactly like a forged result. `jsNumber` below is the reconciliation, and
 * `NumberFormatConformanceTests` checks it against real `JSON.stringify` output.
 */
public enum JSONValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    /// Ordered: the array preserves the key order the caller wrote.
    case object([(key: String, value: JSONValue)])

    public static func == (a: JSONValue, b: JSONValue) -> Bool {
        switch (a, b) {
        case (.null, .null): return true
        case let (.bool(x), .bool(y)): return x == y
        case let (.int(x), .int(y)): return x == y
        case let (.double(x), .double(y)): return x == y
        case let (.int(x), .double(y)): return Double(x) == y
        case let (.double(x), .int(y)): return x == Double(y)
        case let (.string(x), .string(y)): return x == y
        case let (.array(x), .array(y)): return x == y
        case let (.object(x), .object(y)):
            guard x.count == y.count else { return false }
            for (lhs, rhs) in zip(x, y) where lhs.key != rhs.key || lhs.value != rhs.value { return false }
            return true
        default: return false
        }
    }
}

// ------------------------------------------------------------------ serialize

/**
 * Format a Double the way JavaScript's `Number.prototype.toString` does.
 *
 * The cases that actually differ: an integral value prints without `.0`, and JS switches
 * to exponent notation at different thresholds than Swift. Everything the adapters emit
 * is a rounded, modest magnitude, so the common path is the integral check and the
 * shortest-round-trip description Swift already produces.
 */
public func jsNumber(_ d: Double) -> String {
    if d.isNaN || d.isInfinite { return "null" }   // JSON.stringify writes null for both
    if d == 0 { return "0" }                       // and 0 for -0

    // Integers the format represents exactly. Above 2^53 this shortcut is wrong: the
    // exact stored value of 1.2345678901234568e20 is 123456789012345683968, while
    // JavaScript prints the shortest decimal that round-trips, 123456789012345680000.
    // Printing the exact one is not a rounding nit — it is a different hash.
    if d == d.rounded(), abs(d) < 9_007_199_254_740_992 {
        if let exact = Int64(exactly: d.rounded()) { return String(exact) }
    }

    let swiftShortest = "\(d)"
    guard swiftShortest.contains("e") else {
        var plain = swiftShortest
        if plain.hasSuffix(".0") { plain.removeLast(2) }
        return plain
    }
    return jsFromShortest(swiftShortest)
}

/**
 * ECMAScript's `Number::toString`, applied to Swift's shortest round-trip digits.
 *
 * Swift and JavaScript agree on *which* digits are shortest; they disagree about when to
 * write them as plain decimal and when to use an exponent. This is that decision, taken
 * from the spec rather than guessed: `n` is the position of the decimal point and `k` the
 * number of significant digits.
 */
private func jsFromShortest(_ description: String) -> String {
    let halves = description.split(separator: "e", maxSplits: 1)
    guard halves.count == 2, let exponent = Int(halves[1]) else { return description }

    var mantissa = String(halves[0])
    let negative = mantissa.hasPrefix("-")
    if negative { mantissa.removeFirst() }

    var digits = mantissa
    var pointPosition = mantissa.count
    if let dot = mantissa.firstIndex(of: ".") {
        pointPosition = mantissa.distance(from: mantissa.startIndex, to: dot)
        digits = mantissa.replacingOccurrences(of: ".", with: "")
    }
    while digits.count > 1 && digits.hasSuffix("0") { digits.removeLast() }

    let n = pointPosition + exponent   // value == 0.<digits> x 10^n
    let k = digits.count

    var out: String
    if k <= n && n <= 21 {
        out = digits + String(repeating: "0", count: n - k)
    } else if 0 < n && n <= 21 {
        let split = digits.index(digits.startIndex, offsetBy: n)
        out = String(digits[..<split]) + "." + String(digits[split...])
    } else if -6 < n && n <= 0 {
        out = "0." + String(repeating: "0", count: -n) + digits
    } else {
        let e = n - 1
        let mantissaText = k == 1 ? digits : String(digits.first!) + "." + String(digits.dropFirst())
        out = "\(mantissaText)e\(e >= 0 ? "+" : "-")\(abs(e))"
    }
    return negative ? "-" + out : out
}

/// Escape a string the way `JSON.stringify` does: control characters only, plus `"` and `\`.
public func jsString(_ value: String) -> String {
    var out = "\""
    out.reserveCapacity(value.count + 2)
    for scalar in value.unicodeScalars {
        switch scalar {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        case "\u{08}": out += "\\b"
        case "\u{0C}": out += "\\f"
        default:
            if scalar.value < 0x20 {
                out += String(format: "\\u%04x", scalar.value)
            } else {
                out.unicodeScalars.append(scalar)
            }
        }
    }
    return out + "\""
}

extension JSONValue {
    /// Serialize exactly as `JSON.stringify` would, with object keys in insertion order.
    public func stringify() -> String {
        switch self {
        case .null: return "null"
        case .bool(let b): return b ? "true" : "false"
        case .int(let i): return String(i)
        case .double(let d): return jsNumber(d)
        case .string(let s): return jsString(s)
        case .array(let items):
            return "[" + items.map { $0.stringify() }.joined(separator: ",") + "]"
        case .object(let pairs):
            return "{" + pairs.map { jsString($0.key) + ":" + $0.value.stringify() }.joined(separator: ",") + "}"
        }
    }

    public var data: Data { Data(stringify().utf8) }
}

// -------------------------------------------------------------------- parse

extension JSONValue {
    /// Parse arbitrary JSON. Object key order is preserved as written in the source text.
    public static func parse(_ data: Data) -> JSONValue? {
        var parser = JSONParser(bytes: [UInt8](data))
        guard let value = parser.parseValue() else { return nil }
        parser.skipWhitespace()
        return parser.atEnd ? value : nil
    }

    public subscript(key: String) -> JSONValue? {
        guard case .object(let pairs) = self else { return nil }
        return pairs.first { $0.key == key }?.value
    }

    public var stringValue: String? { if case .string(let s) = self { return s }; return nil }
    public var intValue: Int? {
        switch self {
        case .int(let i): return i
        case .double(let d): return Int(exactly: d.rounded()) 
        default: return nil
        }
    }
    public var doubleValue: Double? {
        switch self {
        case .int(let i): return Double(i)
        case .double(let d): return d
        default: return nil
        }
    }
    public var boolValue: Bool? { if case .bool(let b) = self { return b }; return nil }
    public var arrayValue: [JSONValue]? { if case .array(let a) = self { return a }; return nil }
}

/**
 * A minimal recursive-descent JSON parser.
 *
 * Foundation's JSONSerialization would be less code, but it hands back an unordered
 * dictionary — and this type exists precisely to keep order. It also throws on input a
 * peer controls; returning nil everywhere keeps the "a peer must never crash us" rule
 * that `decode` in the TypeScript protocol already follows.
 */
private struct JSONParser {
    let bytes: [UInt8]
    var i = 0

    init(bytes: [UInt8]) { self.bytes = bytes }
    var atEnd: Bool { i >= bytes.count }

    mutating func skipWhitespace() {
        while i < bytes.count, bytes[i] == 0x20 || bytes[i] == 0x09 || bytes[i] == 0x0A || bytes[i] == 0x0D { i += 1 }
    }

    mutating func parseValue() -> JSONValue? {
        skipWhitespace()
        guard i < bytes.count else { return nil }
        switch bytes[i] {
        case UInt8(ascii: "{"): return parseObject()
        case UInt8(ascii: "["): return parseArray()
        case UInt8(ascii: "\""): return parseString().map { .string($0) }
        case UInt8(ascii: "t"): return literal("true") ? .bool(true) : nil
        case UInt8(ascii: "f"): return literal("false") ? .bool(false) : nil
        case UInt8(ascii: "n"): return literal("null") ? .null : nil
        default: return parseNumber()
        }
    }

    private mutating func literal(_ word: String) -> Bool {
        let w = [UInt8](word.utf8)
        guard i + w.count <= bytes.count, Array(bytes[i..<(i + w.count)]) == w else { return false }
        i += w.count
        return true
    }

    private mutating func parseObject() -> JSONValue? {
        i += 1   // {
        var pairs: [(key: String, value: JSONValue)] = []
        skipWhitespace()
        if i < bytes.count, bytes[i] == UInt8(ascii: "}") { i += 1; return .object(pairs) }
        while true {
            skipWhitespace()
            guard let key = parseString() else { return nil }
            skipWhitespace()
            guard i < bytes.count, bytes[i] == UInt8(ascii: ":") else { return nil }
            i += 1
            guard let value = parseValue() else { return nil }
            pairs.append((key, value))
            skipWhitespace()
            guard i < bytes.count else { return nil }
            if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
            if bytes[i] == UInt8(ascii: "}") { i += 1; return .object(pairs) }
            return nil
        }
    }

    private mutating func parseArray() -> JSONValue? {
        i += 1   // [
        var items: [JSONValue] = []
        skipWhitespace()
        if i < bytes.count, bytes[i] == UInt8(ascii: "]") { i += 1; return .array(items) }
        while true {
            guard let value = parseValue() else { return nil }
            items.append(value)
            skipWhitespace()
            guard i < bytes.count else { return nil }
            if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
            if bytes[i] == UInt8(ascii: "]") { i += 1; return .array(items) }
            return nil
        }
    }

    private mutating func parseString() -> String? {
        guard i < bytes.count, bytes[i] == UInt8(ascii: "\"") else { return nil }
        i += 1
        var out: [UInt8] = []
        while i < bytes.count {
            let b = bytes[i]
            if b == UInt8(ascii: "\"") { i += 1; return String(decoding: out, as: UTF8.self) }
            if b == UInt8(ascii: "\\") {
                i += 1
                guard i < bytes.count else { return nil }
                switch bytes[i] {
                case UInt8(ascii: "\""): out.append(UInt8(ascii: "\""))
                case UInt8(ascii: "\\"): out.append(UInt8(ascii: "\\"))
                case UInt8(ascii: "/"): out.append(UInt8(ascii: "/"))
                case UInt8(ascii: "b"): out.append(0x08)
                case UInt8(ascii: "f"): out.append(0x0C)
                case UInt8(ascii: "n"): out.append(0x0A)
                case UInt8(ascii: "r"): out.append(0x0D)
                case UInt8(ascii: "t"): out.append(0x09)
                case UInt8(ascii: "u"):
                    guard let scalar = parseUnicodeEscape() else { return nil }
                    out.append(contentsOf: Array(String(scalar).utf8))
                    continue
                default: return nil
                }
                i += 1
                continue
            }
            out.append(b)
            i += 1
        }
        return nil
    }

    /// Handles surrogate pairs, which appear in any payload carrying an emoji.
    private mutating func parseUnicodeEscape() -> Unicode.Scalar? {
        guard let high = readHex4() else { return nil }
        if high >= 0xD800 && high <= 0xDBFF {
            guard i + 1 < bytes.count, bytes[i] == UInt8(ascii: "\\"), bytes[i + 1] == UInt8(ascii: "u") else {
                return Unicode.Scalar(0xFFFD)
            }
            // Step over the backslash only: readHex4 expects to start on the "u".
            i += 1
            guard let low = readHex4(), low >= 0xDC00, low <= 0xDFFF else { return Unicode.Scalar(0xFFFD) }
            let combined = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
            return Unicode.Scalar(combined) ?? Unicode.Scalar(0xFFFD)
        }
        return Unicode.Scalar(high) ?? Unicode.Scalar(0xFFFD)
    }

    private mutating func readHex4() -> Int? {
        // Caller leaves `i` on the "u".
        guard i < bytes.count, bytes[i] == UInt8(ascii: "u"), i + 4 < bytes.count else { return nil }
        i += 1
        var value = 0
        for _ in 0..<4 {
            let b = bytes[i]
            let digit: Int
            switch b {
            case UInt8(ascii: "0")...UInt8(ascii: "9"): digit = Int(b - UInt8(ascii: "0"))
            case UInt8(ascii: "a")...UInt8(ascii: "f"): digit = Int(b - UInt8(ascii: "a")) + 10
            case UInt8(ascii: "A")...UInt8(ascii: "F"): digit = Int(b - UInt8(ascii: "A")) + 10
            default: return nil
            }
            value = value * 16 + digit
            i += 1
        }
        return value
    }

    private mutating func parseNumber() -> JSONValue? {
        let start = i
        if i < bytes.count, bytes[i] == UInt8(ascii: "-") { i += 1 }
        var isInteger = true
        while i < bytes.count {
            let b = bytes[i]
            if b >= UInt8(ascii: "0") && b <= UInt8(ascii: "9") { i += 1; continue }
            if b == UInt8(ascii: ".") || b == UInt8(ascii: "e") || b == UInt8(ascii: "E")
                || b == UInt8(ascii: "+") || b == UInt8(ascii: "-") {
                isInteger = false
                i += 1
                continue
            }
            break
        }
        guard start < i else { return nil }
        let text = String(decoding: bytes[start..<i], as: UTF8.self)
        if isInteger, let n = Int(text) { return .int(n) }
        guard let d = Double(text) else { return nil }
        return .double(d)
    }
}
