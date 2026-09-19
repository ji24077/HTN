import XCTest
@testable import DWPAgentKit

/// Serves canned bytes so the store's verification rules can be tested without a server.
final class StubProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var body: Data = Data()
    nonisolated(unsafe) static var status = 200
    nonisolated(unsafe) static var requestCount = 0
    nonisolated(unsafe) static var lastAuthorization: String?
    nonisolated(unsafe) static var delay: TimeInterval = 0

    static func reset() {
        body = Data(); status = 200; requestCount = 0; lastAuthorization = nil; delay = 0
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requestCount += 1
        Self.lastAuthorization = request.value(forHTTPHeaderField: "authorization")
        let body = Self.body
        let status = Self.status
        let url = request.url!
        let work: @Sendable () -> Void = {
            let response = HTTPURLResponse(url: url, statusCode: status,
                                           httpVersion: "HTTP/1.1", headerFields: nil)!
            self.client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            self.client?.urlProtocol(self, didLoad: body)
            self.client?.urlProtocolDidFinishLoading(self)
        }
        if Self.delay > 0 {
            DispatchQueue.global().asyncAfter(deadline: .now() + Self.delay, execute: work)
        } else {
            work()
        }
    }

    override func stopLoading() {}
}

/**
 * The store's whole reason to exist is refusing bytes that are not what was asked for.
 *
 * A phone on someone else's wifi is the most plausible place in this system for a
 * middlebox to hand back something other than the requested model, so "verify before use"
 * is tested here as a hard failure rather than as a log line.
 */
final class ArtifactStoreTests: XCTestCase {

    private var directory: URL!
    private var session: URLSession!
    private let identity = HostIdentity.generate()

    override func setUp() {
        super.setUp()
        StubProtocol.reset()
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dwp-artifact-tests-\(UUID().uuidString)")
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubProtocol.self]
        session = URLSession(configuration: configuration)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: directory)
        super.tearDown()
    }

    private func store() -> ArtifactStore {
        ArtifactStore(directory: directory, session: session)
    }

    func testFetchesAndVerifiesGoodBytes() async throws {
        let payload = Data("the model".utf8)
        StubProtocol.body = payload
        let hash = sha256Hex(payload)

        let bytes = try await store().fetch(sha256: hash, server: "https://example.test",
                                            hostId: "h1", identity: identity)
        XCTAssertEqual(bytes, payload)
    }

    func testRejectsBytesThatDoNotMatchTheirHash() async {
        StubProtocol.body = Data("something else entirely".utf8)
        let requested = sha256Hex(Data("the model".utf8))

        do {
            _ = try await store().fetch(sha256: requested, server: "https://example.test",
                                        hostId: "h1", identity: identity)
            XCTFail("a hash mismatch must be fatal, never a warning")
        } catch {
            XCTAssertTrue("\(error)".contains("hash mismatch"), "got: \(error)")
        }
    }

    func testDoesNotCacheBytesItRejected() async {
        StubProtocol.body = Data("wrong".utf8)
        let requested = sha256Hex(Data("right".utf8))
        _ = try? await store().fetch(sha256: requested, server: "https://example.test",
                                     hostId: "h1", identity: identity)
        let cached = await store().cached(requested)
        XCTAssertFalse(cached, "rejected bytes must not be left on disk to be trusted later")
    }

    func testSecondFetchIsServedFromCache() async throws {
        let payload = Data("cache me".utf8)
        StubProtocol.body = payload
        let hash = sha256Hex(payload)
        let store = store()

        _ = try await store.fetch(sha256: hash, server: "https://example.test", hostId: "h1", identity: identity)
        _ = try await store.fetch(sha256: hash, server: "https://example.test", hostId: "h1", identity: identity)
        XCTAssertEqual(StubProtocol.requestCount, 1, "the second read should not have touched the network")
    }

    func testACorruptCacheEntryIsRefetchedRatherThanTrusted() async throws {
        let payload = Data("real content".utf8)
        let hash = sha256Hex(payload)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // A file that no longer matches its own name: truncated download, disk corruption,
        // or tampering — indistinguishable from here, and all handled the same way.
        try Data("corrupt".utf8).write(to: directory.appendingPathComponent(hash))

        StubProtocol.body = payload
        let bytes = try await store().fetch(sha256: hash, server: "https://example.test",
                                            hostId: "h1", identity: identity)
        XCTAssertEqual(bytes, payload)
        XCTAssertEqual(StubProtocol.requestCount, 1)
    }

    func testHTTPFailureSurfacesTheStatus() async {
        StubProtocol.status = 404
        do {
            _ = try await store().fetch(sha256: String(repeating: "a", count: 64),
                                        server: "https://example.test", hostId: "h1", identity: identity)
            XCTFail("expected a failure")
        } catch {
            XCTAssertTrue("\(error)".contains("404"), "got: \(error)")
        }
    }

    func testFetchIsAuthenticatedWithAFreshAssertion() async throws {
        let payload = Data("x".utf8)
        StubProtocol.body = payload
        _ = try await store().fetch(sha256: sha256Hex(payload), server: "https://example.test",
                                    hostId: "host-42", identity: identity)

        let header = try XCTUnwrap(StubProtocol.lastAuthorization)
        XCTAssertTrue(header.hasPrefix("Bearer "))
        let claims = try XCTUnwrap(
            Data(base64URLEncoded: String(header.dropFirst(7)).split(separator: ".")[1].description))
        XCTAssertEqual(JSONValue.parse(claims)?["iss"]?.stringValue, "host-42")
    }

    /// Two slices of one job start together and want the same 7.8 MB of inputs. On a phone
    /// fetching it twice is not merely wasteful — it is two copies resident at once.
    func testConcurrentFetchesOfTheSameArtifactShareOneRequest() async throws {
        let payload = Data(repeating: 0x7, count: 4096)
        StubProtocol.body = payload
        StubProtocol.delay = 0.2
        let hash = sha256Hex(payload)
        let store = store()
        // A local copy: the test case's own property is task-isolated, and sending it
        // into two concurrent children is exactly the race the compiler is describing.
        let key = identity

        async let first = store.fetch(sha256: hash, server: "https://example.test", hostId: "h", identity: key)
        async let second = store.fetch(sha256: hash, server: "https://example.test", hostId: "h", identity: key)
        let (a, b) = try await (first, second)

        XCTAssertEqual(a, payload)
        XCTAssertEqual(b, payload)
        XCTAssertEqual(StubProtocol.requestCount, 1)
    }
}
