import Foundation
#if canImport(Security)
import Security
#endif

/**
 * Where the host key and config live.
 *
 * Two backings, chosen by platform rather than by preference: the Keychain on iOS, and a
 * mode-0600 file under `DWP_HOME` on macOS. The macOS path exists so the existing network
 * simulator can run several Swift agents side by side exactly as it does Node ones —
 * without it, every reliability test would need a phone in a drawer.
 */
public protocol KeyStore: Sendable {
    func loadIdentity() throws -> HostIdentity?
    func saveIdentity(_ identity: HostIdentity) throws
    func loadConfig() throws -> AgentConfig?
    func saveConfig(_ config: AgentConfig) throws
    func clear() throws
}

public struct AgentConfig: Codable, Sendable, Equatable {
    public var server: String
    public var wsUrl: String
    public var hostId: String
    public var label: String
    public var allowCompute: Bool
    public var allowBrowser: Bool
    public var maxConcurrency: Int
    /// Local kill switch. A server round-trip would be useless exactly when it matters.
    public var paused: Bool

    public init(server: String, wsUrl: String, hostId: String, label: String,
                allowCompute: Bool = true, allowBrowser: Bool = false,
                maxConcurrency: Int = 1, paused: Bool = false) {
        self.server = server
        self.wsUrl = wsUrl
        self.hostId = hostId
        self.label = label
        self.allowCompute = allowCompute
        // Always false, and not configurable: there is no browser adapter on this
        // platform and advertising one would earn work the device can only fail.
        self.allowBrowser = false
        self.maxConcurrency = maxConcurrency
        self.paused = paused
    }
}

// ------------------------------------------------------------------ file store

/// Used on macOS for tests and the CLI, and as the iOS simulator's backing.
public struct FileKeyStore: KeyStore {
    public let home: URL

    public init(home: URL? = nil) {
        if let home {
            self.home = home
        } else if let override = ProcessInfo.processInfo.environment["DWP_HOME"] {
            self.home = URL(fileURLWithPath: override)
        } else {
            self.home = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".dwp-ios")
        }
    }

    private var keyPath: URL { home.appendingPathComponent("agent.key") }
    private var configPath: URL { home.appendingPathComponent("config.json") }

    private func ensureHome() throws {
        try FileManager.default.createDirectory(
            at: home, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
    }

    public func loadIdentity() throws -> HostIdentity? {
        guard let pem = try? String(contentsOf: keyPath, encoding: .utf8) else { return nil }
        return try HostIdentity.fromPEM(pem)
    }

    public func saveIdentity(_ identity: HostIdentity) throws {
        try ensureHome()
        try identity.privateKeyPEM.write(to: keyPath, atomically: true, encoding: .utf8)
        // `atomically` replaces the file, which drops any mode set on the previous one.
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: keyPath.path)
    }

    public func loadConfig() throws -> AgentConfig? {
        guard let data = try? Data(contentsOf: configPath) else { return nil }
        return try JSONDecoder().decode(AgentConfig.self, from: data)
    }

    public func saveConfig(_ config: AgentConfig) throws {
        try ensureHome()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(config).write(to: configPath, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: configPath.path)
    }

    public func clear() throws {
        try? FileManager.default.removeItem(at: keyPath)
        try? FileManager.default.removeItem(at: configPath)
    }
}

// -------------------------------------------------------------- keychain store

#if canImport(Security) && !targetEnvironment(simulator)
/// The iOS backing. Device-only and non-syncing, so the identity cannot follow a backup
/// onto a second phone and quietly become two hosts claiming one id.
public struct KeychainKeyStore: KeyStore {
    public let service: String

    public init(service: String = "com.dwp.agent") { self.service = service }

    private func query(_ account: String) -> [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: account]
    }

    private func read(_ account: String) -> Data? {
        var q = query(account)
        q[kSecReturnData as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var out: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess else { return nil }
        return out as? Data
    }

    private func write(_ account: String, _ data: Data) throws {
        let q = query(account)
        SecItemDelete(q as CFDictionary)
        var add = q
        add[kSecValueData as String] = data
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let status = SecItemAdd(add as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw AgentError.badKey("keychain write failed (OSStatus \(status))")
        }
    }

    public func loadIdentity() throws -> HostIdentity? {
        guard let data = read("identity"), let pem = String(data: data, encoding: .utf8) else { return nil }
        return try HostIdentity.fromPEM(pem)
    }

    public func saveIdentity(_ identity: HostIdentity) throws {
        try write("identity", Data(identity.privateKeyPEM.utf8))
    }

    public func loadConfig() throws -> AgentConfig? {
        guard let data = read("config") else { return nil }
        return try JSONDecoder().decode(AgentConfig.self, from: data)
    }

    public func saveConfig(_ config: AgentConfig) throws {
        try write("config", try JSONEncoder().encode(config))
    }

    public func clear() throws {
        SecItemDelete(query("identity") as CFDictionary)
        SecItemDelete(query("config") as CFDictionary)
    }
}
#endif

/// The right store for wherever this is running.
public func defaultKeyStore() -> KeyStore {
    #if canImport(Security) && !targetEnvironment(simulator) && os(iOS)
    return KeychainKeyStore()
    #else
    return FileKeyStore()
    #endif
}
