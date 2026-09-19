import Foundation
import SwiftUI
import BackgroundTasks
import UIKit
import DWPAgentKit

/**
 * The app's single piece of state, wrapping the shared agent core.
 *
 * Everything that decides anything lives in DWPAgentKit and is tested there. This type
 * does three things the package cannot: it owns the SwiftUI lifecycle, it turns battery
 * and thermal notifications into agent state, and it holds the `BGContinuedProcessingTask`
 * that lets a running slice survive the user locking the screen.
 */
@MainActor
final class AgentModel: ObservableObject {

    @Published private(set) var state: ConnectionState = .idle
    @Published private(set) var config: AgentConfig?
    @Published private(set) var events: [AgentEvent] = []
    @Published private(set) var progress: (done: Int, total: Int)?
    @Published private(set) var completedTasks = 0
    @Published var paused = false
    @Published var pairingError: String?
    @Published var isPairing = false

    /// Set once per launch, because `supportedResources` is a device fact, not a setting.
    @Published private(set) var backgroundNote: String = ""

    private let store: KeyStore
    private var identity: HostIdentity?
    private var connection: AgentConnection?

    /// Registered in `init`; a request whose identifier is not permitted is rejected.
    static let backgroundTaskIdentifier = "com.dwp.agent.work"

    /**
     * Held as `AnyObject` so the deployment target can stay at iOS 17.
     *
     * `BGContinuedProcessingTask` is iOS 26 only, but an iPhone on iOS 18 is still a
     * perfectly good host while the app is open — raising the floor to 26 to get a
     * background nicety would exclude devices that can do the actual work.
     */
    private var backgroundTaskBox: AnyObject?

    @available(iOS 26.0, *)
    private var backgroundTask: BGContinuedProcessingTask? {
        get { backgroundTaskBox as? BGContinuedProcessingTask }
        set { backgroundTaskBox = newValue }
    }

    init(store: KeyStore = defaultKeyStore()) {
        self.store = store
        self.config = try? store.loadConfig()
        self.identity = try? store.loadIdentity()
        self.paused = config?.paused ?? false

        UIDevice.current.isBatteryMonitoringEnabled = true
        registerBackgroundTask()
    }

    var isPaired: Bool { config != nil && identity != nil }

    // ------------------------------------------------------------------ pairing

    func pair(link: String, label: String) async {
        pairingError = nil
        guard let invite = Pairing.parseInvite(link) else {
            pairingError = "That does not look like an invite link. It should look like "
                         + "https://their-address/join?code=ABCD-1234"
            return
        }
        await pair(server: invite.server, code: invite.code, label: label)
    }

    func pair(server: String, code: String, label: String) async {
        isPairing = true
        defer { isPairing = false }
        do {
            let result = try await Pairing.pair(server: server, code: code, label: label, store: store)
            config = result.config
            identity = result.identity
            pairingError = nil
        } catch {
            pairingError = "\(error)"
        }
    }

    func clearError() { pairingError = nil }

    func forget() async {
        await stop()
        try? store.clear()
        config = nil
        identity = nil
        events = []
        completedTasks = 0
    }

    // --------------------------------------------------------------- connection

    func start() async {
        guard let config, let identity, connection == nil else { return }

        let connection = AgentConnection(config: config, identity: identity)
        self.connection = connection

        await connection.observe(
            state: { [weak self] state in
                Task { @MainActor in self?.state = state }
            },
            events: { [weak self] event in
                Task { @MainActor in self?.apply(event) }
            },
            progress: { [weak self] _, done, total in
                Task { @MainActor in
                    self?.progress = (done, total)
                    // Reporting progress is not cosmetic: iOS expires a continued task
                    // that goes quiet, so this is what keeps a slice alive on a locked phone.
                    if #available(iOS 26.0, *) {
                        self?.backgroundTask?.progress.totalUnitCount = Int64(total)
                        self?.backgroundTask?.progress.completedUnitCount = Int64(done)
                    }
                }
            })
        await connection.start()
    }

    func stop() async {
        await connection?.stop()
        connection = nil
        state = .stopped
        progress = nil
        endBackgroundTask(success: true)
    }

    func setPaused(_ value: Bool) async {
        paused = value
        if var config {
            config.paused = value
            try? store.saveConfig(config)
            self.config = config
        }
        await connection?.setPaused(value)
    }

    private func apply(_ event: AgentEvent) {
        events.insert(event, at: 0)
        if events.count > 200 { events.removeLast(events.count - 200) }

        switch event.type {
        case "task.accepted":
            beginBackgroundTask(describing: event.detail)
        case "task.succeeded":
            completedTasks += 1
            progress = nil
            endBackgroundTask(success: true)
        case "task.failed", "task.cancelled":
            progress = nil
            endBackgroundTask(success: false)
        default:
            break
        }
    }

    // -------------------------------------------------------- background running

    private func registerBackgroundTask() {
        guard #available(iOS 26.0, *) else {
            backgroundNote = "This phone runs iOS \(UIDevice.current.systemVersion). "
                           + "Work continues only while the app is on screen; iOS 26 adds background slices."
            return
        }
        let registered = BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.backgroundTaskIdentifier, using: .main
        ) { [weak self] task in
            guard let task = task as? BGContinuedProcessingTask else { return }
            Task { @MainActor in self?.adopt(task) }
        }

        // The Simulator has no background processing at all, and says so through a
        // submission error rather than here. Recording the reason keeps "it did not keep
        // running" from looking like a bug in the agent.
        #if targetEnvironment(simulator)
        backgroundNote = "The Simulator does not run background tasks — work stops when this app leaves the screen."
        #else
        if !registered {
            backgroundNote = "This build cannot register background work; slices will stop when the app is backgrounded."
        } else if BGTaskScheduler.supportedResources.contains(.gpu) {
            backgroundNote = "Background work is available, including GPU."
        } else {
            backgroundNote = "Background work is available (CPU only — background GPU is iPad-only today)."
        }
        #endif
    }

    /**
     * Keep the current slice running after the user leaves the app.
     *
     * Submitted per slice rather than once for the whole session, deliberately. A
     * continued task is defined by a measurable goal the user can watch and cancel; an
     * agent idling for offers has no goal, and iOS expires a task that reports no
     * progress. So a slice survives backgrounding, and the connection between slices does
     * not — which is the honest shape of what this platform offers.
     */
    private func beginBackgroundTask(describing detail: String) {
        guard #available(iOS 26.0, *), backgroundTaskBox == nil else { return }
        let request = BGContinuedProcessingTaskRequest(
            identifier: Self.backgroundTaskIdentifier,
            title: "Contributing compute",
            subtitle: detail.isEmpty ? "Running a slice of work" : detail)
        // Queue rather than fail: a slice that starts a moment later is still useful,
        // where a refused one has to be handed back to the fleet.
        request.strategy = .queue

        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            // Not fatal. The slice still runs; it just will not survive backgrounding.
            events.insert(AgentEvent("background.unavailable", "\(error.localizedDescription)"), at: 0)
        }
    }

    @available(iOS 26.0, *)
    private func adopt(_ task: BGContinuedProcessingTask) {
        backgroundTask = task
        if let progress {
            task.progress.totalUnitCount = Int64(progress.total)
            task.progress.completedUnitCount = Int64(progress.done)
        }
        task.expirationHandler = { [weak self] in
            Task { @MainActor in
                // The system is reclaiming the time. Stopping cleanly returns the slice to
                // the fleet through the normal lease path instead of letting it time out.
                self?.events.insert(AgentEvent("background.expired", "the system reclaimed this task"), at: 0)
                await self?.stop()
            }
        }
    }

    private func endBackgroundTask(success: Bool) {
        if #available(iOS 26.0, *) { backgroundTask?.setTaskCompleted(success: success) }
        backgroundTaskBox = nil
    }
}
