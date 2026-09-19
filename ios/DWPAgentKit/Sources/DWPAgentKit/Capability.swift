import Foundation
#if canImport(UIKit)
import UIKit
#endif

public let agentVersion = "0.2.0-ios"

/**
 * What this device is, and — the part that matters on a phone — what state it is in.
 *
 * A desktop host's capability is essentially static: cores and RAM do not change between
 * heartbeats. A phone's does. It gets hot, it gets unplugged, it drops to Low Power Mode,
 * and each of those should change whether it is sensible to send it a hundred items of
 * work. So the record carries a `mobile` block alongside the fixed hardware facts, and
 * the scheduler is free to use it or ignore it.
 */
public enum Capability {

    public static func probe(adapters: [String]) -> CapabilityRecord {
        CapabilityRecord(
            agentVersion: agentVersion,
            os: osName,
            arch: architecture,
            cpuModel: deviceModel,
            logicalCores: ProcessInfo.processInfo.processorCount,
            totalRamMb: Int(ProcessInfo.processInfo.physicalMemory / 1024 / 1024),
            freeRamMb: availableMemoryMb,
            adapters: adapters
        )
    }

    public static var osName: String {
        #if os(iOS)
        return "ios"
        #elseif os(macOS)
        return "darwin"
        #else
        return "linux"
        #endif
    }

    public static var architecture: String {
        #if arch(arm64)
        return "arm64"
        #elseif arch(x86_64)
        return "x64"
        #else
        return "unknown"
        #endif
    }

    /// `iPhone17,1` style identifier — the honest answer to "which phone is this".
    public static var deviceModel: String {
        #if targetEnvironment(simulator)
        if let simulated = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] {
            return "\(simulated) (simulator)"
        }
        return "iOS Simulator"
        #else
        var info = utsname()
        uname(&info)
        let identifier = withUnsafePointer(to: &info.machine) { pointer in
            pointer.withMemoryRebound(to: CChar.self, capacity: MemoryLayout.size(ofValue: pointer)) {
                String(validatingCString: $0) ?? ""
            }
        }
        return identifier.isEmpty ? "Apple device" : identifier
        #endif
    }

    /**
     * Headroom before this process is killed, not free RAM on the device.
     *
     * On iOS the meaningful limit is the per-app jetsam budget — roughly 4 GB of an 8 GB
     * phone — and the OS kills instantly at the boundary with no chance to save state.
     * Reporting device-wide free memory would tell the scheduler a number that has almost
     * nothing to do with whether the next slice fits.
     */
    public static var availableMemoryMb: Int {
        #if os(iOS)
        return Int(os_proc_available_memory() / 1024 / 1024)
        #else
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return Int(ProcessInfo.processInfo.physicalMemory / 1024 / 1024 / 4)
        }
        let pageSize = UInt64(vm_kernel_page_size)
        let free = (UInt64(stats.free_count) + UInt64(stats.inactive_count)) * pageSize
        return Int(free / 1024 / 1024)
        #endif
    }

    // -------------------------------------------------------------- mobile state

    public struct MobileState: Sendable, Equatable {
        public var thermal: String
        public var lowPowerMode: Bool
        public var batteryLevel: Double?
        public var charging: Bool?
        public var availableMemoryMb: Int

        /// Is it reasonable to accept sustained work right now?
        ///
        /// Deliberately conservative: a phone that is hot, unplugged and nearly flat is a
        /// host that will drop its task and annoy its owner, and a declined offer costs
        /// the scheduler a requeue while an abandoned one costs a whole lease timeout.
        public var fitForWork: Bool {
            if thermal == "critical" { return false }
            if lowPowerMode { return false }
            if let charging, let batteryLevel, !charging, batteryLevel < 0.2 { return false }
            return true
        }

        public var json: JSONValue {
            var pairs: [(key: String, value: JSONValue)] = [
                ("thermal", .string(thermal)),
                ("lowPowerMode", .bool(lowPowerMode)),
                ("availableMemoryMb", .int(availableMemoryMb)),
                ("fitForWork", .bool(fitForWork)),
            ]
            if let batteryLevel { pairs.append(("batteryLevel", .double(batteryLevel))) }
            if let charging { pairs.append(("charging", .bool(charging))) }
            return .object(pairs)
        }
    }

    public static func mobileState() -> MobileState {
        let thermal: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermal = "nominal"
        case .fair: thermal = "fair"
        case .serious: thermal = "serious"
        case .critical: thermal = "critical"
        @unknown default: thermal = "unknown"
        }

        var level: Double?
        var charging: Bool?
        #if os(iOS)
        // Monitoring is off by default and returns -1 until enabled; the agent turns it
        // on at startup, so a nil here means genuinely unavailable rather than unread.
        if UIDevice.current.isBatteryMonitoringEnabled {
            let raw = UIDevice.current.batteryLevel
            if raw >= 0 { level = Double(raw) }
            switch UIDevice.current.batteryState {
            case .charging, .full: charging = true
            case .unplugged: charging = false
            default: charging = nil
            }
        }
        #endif

        return MobileState(
            thermal: thermal,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            batteryLevel: level,
            charging: charging,
            availableMemoryMb: availableMemoryMb
        )
    }
}
