import XCTest
@testable import DWPAgentKit

/**
 * The iOS agent's version constant must match the build it shipped in.
 *
 * It is hand-written on purpose — see `agentVersion` — but a hand-written constant is
 * exactly what drifted on the desktop side, and then drifted here: the phone reported
 * `0.2.0-ios` in production handshakes while the repo had moved to `0.3.0`. Nothing
 * failed, which is what made it survive; the fleet view simply described the device
 * wrongly.
 */
final class VersionTests: XCTestCase {

    func testVersionMatchesTheRepositoryPackage() throws {
        // From #filePath, not an absolute path: this has to work in any checkout.
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // DWPAgentKitTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // DWPAgentKit
            .deletingLastPathComponent()   // ios
            .deletingLastPathComponent()   // repo root
        let packageURL = repoRoot.appendingPathComponent("packages/agent/package.json")

        guard let data = try? Data(contentsOf: packageURL),
              let json = JSONValue.parse(data),
              let desktop = json["version"]?.stringValue
        else {
            throw XCTSkip("packages/agent/package.json not readable from \(packageURL.path) — "
                        + "expected when the package is built outside the repo")
        }

        // Only the numeric prefix. The `-ios` suffix is what tells the two
        // implementations apart in a handshake, so a whole-string comparison could never
        // pass and the obvious "fix" would be to delete the thing that distinguishes them.
        let numeric = { (s: String) in String(s.prefix(while: { $0.isNumber || $0 == "." })) }
        XCTAssertEqual(numeric(agentVersion), numeric(desktop),
                       "iOS reports \(agentVersion) while packages/agent is \(desktop) — "
                     + "bump `agentVersion` in Capability.swift in the same change")
    }

    func testVersionKeepsThePlatformSuffix() {
        // Dropping this makes an iPhone indistinguishable from a laptop in the fleet view,
        // and `ios/test/e2e.ts` asserts on it.
        XCTAssertTrue(agentVersion.hasSuffix("-ios"), "got \(agentVersion)")
    }
}
