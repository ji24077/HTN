import XCTest
@testable import DWPAgentKit

/**
 * The device's self-assessment.
 *
 * `fitForWork` is the only place the agent refuses work for a reason the control service
 * cannot see, so the policy is pinned here rather than left to be rediscovered from
 * behaviour on a hot phone.
 */
final class CapabilityTests: XCTestCase {

    private func state(thermal: String = "nominal", lowPower: Bool = false,
                       battery: Double? = nil, charging: Bool? = nil) -> Capability.MobileState {
        Capability.MobileState(thermal: thermal, lowPowerMode: lowPower,
                               batteryLevel: battery, charging: charging,
                               availableMemoryMb: 2048)
    }

    func testAHealthyDeviceTakesWork() {
        XCTAssertTrue(state().fitForWork)
        XCTAssertTrue(state(thermal: "fair").fitForWork)
        XCTAssertTrue(state(thermal: "serious").fitForWork,
                      "serious is warm, not dangerous — declining here would idle the device most of the time")
    }

    func testACriticalDeviceRefuses() {
        XCTAssertFalse(state(thermal: "critical").fitForWork)
    }

    func testLowPowerModeIsTreatedAsAnInstruction() {
        // The user has explicitly asked the system to stop doing optional work. Taking a
        // slice anyway is the behaviour that gets an app deleted.
        XCTAssertFalse(state(lowPower: true).fitForWork)
    }

    func testNearlyFlatAndUnpluggedRefuses() {
        XCTAssertFalse(state(battery: 0.1, charging: false).fitForWork)
        XCTAssertTrue(state(battery: 0.1, charging: true).fitForWork,
                      "plugged in is the whole point; a low reading while charging is fine")
        XCTAssertTrue(state(battery: 0.9, charging: false).fitForWork)
    }

    func testUnknownBatteryDoesNotBlockWork() {
        // The simulator and a Mac report nothing here. Refusing on absent data would mean
        // the test fleet never takes a task.
        XCTAssertTrue(state(battery: nil, charging: nil).fitForWork)
    }

    func testMobileStateSerializesTheOptionalFieldsOnlyWhenKnown() {
        XCTAssertFalse(state().json.stringify().contains("batteryLevel"))
        XCTAssertTrue(state(battery: 0.5, charging: true).json.stringify().contains(#""batteryLevel":0.5"#))
        XCTAssertTrue(state().json.stringify().contains(#""fitForWork":true"#))
    }

    func testProbeReportsAPlatformTheControlServiceAccepts() {
        let record = Capability.probe(adapters: AgentConnection.adapters)
        // The zod enum is the gate: an unknown value makes `hello` unparseable, the host
        // row is never updated, and the device connects but is never given work.
        XCTAssertTrue(["ios", "darwin", "win32", "linux", "android"].contains(record.os))
        XCTAssertGreaterThan(record.logicalCores, 0)
        XCTAssertGreaterThan(record.totalRamMb, 0)
        XCTAssertGreaterThanOrEqual(record.freeRamMb, 0)
        XCTAssertFalse(record.cpuModel.isEmpty)
    }

    func testProbeAdvertisesOnlyAdaptersThisPlatformHas() {
        let record = Capability.probe(adapters: AgentConnection.adapters)
        XCTAssertTrue(record.adapters.contains("echo"))
        XCTAssertTrue(record.adapters.contains("cpu_inference_batch"))
        // Playwright cannot exist here. Advertising it would earn work the device can
        // only fail, which reads as a broken host rather than an absent option.
        XCTAssertFalse(record.adapters.contains("remote_browser_session"))
    }

    func testCapabilityJSONKeysMatchTheSchemaOrder() {
        let json = Capability.probe(adapters: ["echo"]).json.stringify()
        XCTAssertTrue(json.hasPrefix(#"{"agentVersion":"#))
        for key in ["os", "arch", "cpuModel", "logicalCores", "totalRamMb", "freeRamMb", "adapters"] {
            XCTAssertTrue(json.contains("\"\(key)\":"), "missing \(key)")
        }
    }

    func testConsentNeverAdvertisesBrowserWork() {
        let config = AgentConfig(server: "s", wsUrl: "w", hostId: "h", label: "l", allowBrowser: true)
        // Set true by the caller and forced false by the initialiser — there is no
        // browser adapter on this platform and consent must not claim otherwise.
        XCTAssertFalse(config.allowBrowser)
    }
}
