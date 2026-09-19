import Foundation

/// What the UI shows, and what the tests assert on.
public enum ConnectionState: Sendable, Equatable {
    case idle
    case connecting(attempt: Int)
    case connected(since: Date)
    case waiting(retryInSeconds: Double, reason: String)
    case revoked(reason: String)
    case stopped
}

public struct AgentEvent: Sendable {
    public let type: String
    public let detail: String
    public let at: Date
    public init(_ type: String, _ detail: String = "") {
        self.type = type; self.detail = detail; self.at = Date()
    }
}

/**
 * The agent's connection to the control service.
 *
 * A deliberate mirror of `packages/agent/src/transport.ts` — the same backoff, the same
 * silence detection, the same lease renewal — because two implementations that drift
 * apart produce a fleet where a phone and a laptop disagree about what "connected" means
 * and only one of them is right.
 *
 * What is genuinely different is stated where it happens: this side cannot observe the
 * server's ping frames (URLSession answers them without telling us), so liveness is
 * inferred from our own ping's pong; and an offer is declined when the device is in no
 * state to finish it, which no desktop host ever needs to do.
 */
public actor AgentConnection {

    // Fallbacks only. The server announces the real cadence at handshake and a lease
    // duration with every offer; a hardcoded interval would drift out of agreement the
    // moment either is tuned.
    private static let defaultHeartbeat: TimeInterval = 15
    /// A connection that survives this many heartbeats counts as healthy and resets the
    /// backoff. Relative rather than absolute, so it scales with however the server is tuned.
    private static let stableHeartbeats: Double = 2
    private static let maxBackoff: TimeInterval = 30
    private static let handshakeTimeout: TimeInterval = 15

    private let config: AgentConfig
    private let identity: HostIdentity
    private let artifacts: ArtifactStore
    private let session: URLSession
    private let clock: @Sendable () -> Date

    private var socket: URLSessionWebSocketTask?
    private var stopped = false
    private var attempt = 0
    private var backoff: TimeInterval = 1
    private var everConnected = false
    private var connectedSince: Date?
    private var heartbeatInterval = AgentConnection.defaultHeartbeat
    private var lastInbound = Date()
    private var running: [String: RunningTask] = [:]
    private var paused: Bool
    /// The device's own verdict, kept separate from the owner's switch so that recovering
    /// from a hot phone does not silently un-pause one the owner paused deliberately.
    private var deviceFit = true
    private var loop: Task<Void, Never>?

    private struct RunningTask {
        let leaseId: String
        let work: Task<Void, Never>
        let renew: Task<Void, Never>
    }

    public private(set) var state: ConnectionState = .idle {
        didSet { if state != oldValue { onStateChange?(state) } }
    }

    private var onStateChange: (@Sendable (ConnectionState) -> Void)?
    private var onEvent: (@Sendable (AgentEvent) -> Void)?
    private var onProgress: (@Sendable (String, Int, Int) -> Void)?

    public init(config: AgentConfig, identity: HostIdentity,
                artifacts: ArtifactStore = ArtifactStore(),
                session: URLSession = .shared,
                clock: @escaping @Sendable () -> Date = { Date() }) {
        self.config = config
        self.identity = identity
        self.artifacts = artifacts
        self.session = session
        self.clock = clock
        self.paused = config.paused
    }

    public func observe(state: (@Sendable (ConnectionState) -> Void)? = nil,
                        events: (@Sendable (AgentEvent) -> Void)? = nil,
                        progress: (@Sendable (String, Int, Int) -> Void)? = nil) {
        self.onStateChange = state
        self.onEvent = events
        self.onProgress = progress
    }

    private func emit(_ type: String, _ detail: String = "") {
        onEvent?(AgentEvent(type, detail))
    }

    // ---------------------------------------------------------------- lifecycle

    public func start() {
        guard loop == nil else { return }
        stopped = false
        loop = Task { [weak self] in
            guard let self else { return }
            await self.runForever()
        }
    }

    public func stop() {
        stopped = true
        loop?.cancel()
        loop = nil
        abandonRunning()
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
        state = .stopped
    }

    /// The owner's kill switch. Local, so it works when the control service does not.
    public func setPaused(_ value: Bool) async {
        guard value != paused else { return }
        paused = value
        if value { abandonRunning() }
        send("consent.update", consentJSON())
        emit(value ? "paused" : "resumed")
    }

    public var isPaused: Bool { paused }
    public var runningCount: Int { running.count }

    private func runForever() async {
        while !stopped && !Task.isCancelled {
            attempt += 1
            state = .connecting(attempt: attempt)
            let dialStarted = clock()

            let outcome = await connectOnce()

            let held = connectedSince.map { clock().timeIntervalSince($0) } ?? 0
            connectedSince = nil
            abandonRunning()

            if stopped || Task.isCancelled { break }
            if case .revoked(let reason) = state {
                emit("revoked", reason)
                return
            }

            // Reset the backoff only for a connection that actually held. Resetting on
            // every open turns a flapping link into a hot reconnect loop: connect, drop,
            // retry in a second, forever — which hammers the server hardest exactly when
            // it is least able to cope.
            if held >= heartbeatInterval * Self.stableHeartbeats {
                backoff = 1
                attempt = 0
            }

            // Full jitter: a fleet reconnecting after an outage must not arrive in lockstep.
            let delay = Double.random(in: 0...backoff)
            backoff = min(backoff * 2, Self.maxBackoff)
            state = .waiting(retryInSeconds: delay, reason: outcome)
            emit("connect.retry", String(format: "in %.1fs after %@ (held %.1fs)", delay, outcome, held))
            _ = dialStarted
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
        }
        if !stopped { state = .stopped }
    }

    // ------------------------------------------------------------------ connect

    private func connectOnce() async -> String {
        guard let url = URL(string: config.wsUrl) else { return "bad ws url" }
        var request = URLRequest(url: url)
        request.timeoutInterval = Self.handshakeTimeout
        do {
            request.setValue("Bearer \(try Assertion.mint(hostId: config.hostId, identity: identity))",
                             forHTTPHeaderField: "authorization")
        } catch {
            return "could not sign the connection assertion"
        }

        let task = session.webSocketTask(with: request)
        socket = task
        task.resume()

        deviceFit = Capability.mobileState().fitForWork
        lastInbound = clock()
        connectedSince = clock()
        everConnected = true
        state = .connected(since: clock())
        emit("connect.established", config.wsUrl)

        send("hello", .object([
            ("capability", capabilityJSON()),
            ("consent", consentJSON()),
        ]))

        let reason = await withTaskGroup(of: String?.self, returning: String.self) { group in
            group.addTask { [weak self] in await self?.receiveLoop(task) ?? "gone" }
            group.addTask { [weak self] in await self?.heartbeatLoop(task) ?? nil }

            // Whichever finishes first ends the connection; the other is cancelled.
            var result = "closed"
            for await value in group {
                if let value { result = value; break }
            }
            group.cancelAll()
            return result
        }

        task.cancel(with: .goingAway, reason: nil)
        socket = nil
        return reason
    }

    private func receiveLoop(_ task: URLSessionWebSocketTask) async -> String {
        while !Task.isCancelled {
            do {
                let message = try await task.receive()
                noteInbound()
                switch message {
                case .string(let text): await handle(Data(text.utf8))
                case .data(let data): await handle(data)
                @unknown default: break
                }
            } catch {
                if Task.isCancelled { return "cancelled" }
                return (error as NSError).localizedDescription
            }
        }
        return "cancelled"
    }

    private func noteInbound() { lastInbound = clock() }

    private func heartbeatLoop(_ task: URLSessionWebSocketTask) async -> String? {
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: UInt64(heartbeatInterval * 1_000_000_000))
            if Task.isCancelled { return nil }

            // A link can fail in one direction only — writes vanish into it while the
            // socket still looks open — so the absence of inbound traffic is the only
            // reliable signal there is.
            let silent = clock().timeIntervalSince(lastInbound)
            if silent > heartbeatInterval * 3 {
                emit("server.went_silent", String(format: "%.1fs", silent))
                return "server went silent"
            }

            let mobile = Capability.mobileState()

            /**
             * Withdraw from the rotation rather than declining offer by offer.
             *
             * Declining is the wrong mechanism here: the server releases the task and
             * re-offers it immediately, so an unfit device spins in a decline loop that
             * hammers the control service and burns exactly the battery Low Power Mode
             * was asked to save. Consent already means "do not send me work", so say that
             * once and say it again when the device recovers.
             */
            if mobile.fitForWork != deviceFit {
                deviceFit = mobile.fitForWork
                send("consent.update", consentJSON())
                emit(deviceFit ? "device.rejoined" : "device.withdrew",
                     deviceFit ? "ready for work again" : mobile.unfitReason)
                if !deviceFit { abandonRunning() }
            }

            send("heartbeat", .object([
                ("freeRamMb", .int(Capability.availableMemoryMb)),
                ("running", .int(running.count)),
                ("mobile", mobile.json),
            ]))

            // Our own ping, so a server that has nothing to say is still distinguishable
            // from one that has gone away. URLSession answers the server's pings without
            // telling us, so this is the only liveness signal we can actually see.
            task.sendPing { [weak self] error in
                guard error == nil, let self else { return }
                Task { await self.noteInbound() }
            }
        }
        return nil
    }

    // ------------------------------------------------------------------ inbound

    private func handle(_ data: Data) async {
        guard let message = Envelope.decode(data) else { return }

        switch message.type {
        case "hello.ack":
            guard let ack = HelloAck.parse(message.payload) else { return }
            if ack.heartbeatSeconds > 0 { heartbeatInterval = ack.heartbeatSeconds }
            let skew = clock().timeIntervalSince(ISO8601.date(from: ack.serverTime) ?? clock())
            emit("handshake.complete", String(format: "server skew %.0fms", skew * 1000))
            if abs(skew) > 60 {
                emit("clock.skewed", "this device disagrees with the server by over a minute")
            }

        case "revoked":
            let reason = message.payload["reason"]?.stringValue ?? "unknown"
            state = .revoked(reason: reason)
            stopped = true
            socket?.cancel(with: .normalClosure, reason: nil)

        case "task.cancel":
            if let taskId = message.payload["taskId"]?.stringValue {
                cancelRunning(taskId)
                emit("task.cancelled", taskId)
            }

        case "task.offer":
            guard let offer = TaskOffer.parse(message.payload) else { return }
            await accept(offer)

        default:
            break
        }
    }

    // -------------------------------------------------------------------- work

    private func eligibility(for offer: TaskOffer) -> String? {
        if paused { return "paused" }
        if !config.allowCompute { return "compute-not-allowed" }
        if !Self.adapters.contains(offer.adapter) { return "no-such-adapter" }
        if running.count >= config.maxConcurrency { return "at-capacity" }
        // A backstop, not the mechanism. Withdrawal happens through consent in the
        // heartbeat; this only catches an offer already in flight when that was sent.
        let mobile = Capability.mobileState()
        if !mobile.fitForWork { return "device-not-fit(\(mobile.unfitReason))" }
        return nil
    }

    public static let adapters = [EchoAdapter.name, InferenceAdapter.name, WalkerAdapter.name]

    private func accept(_ offer: TaskOffer) async {
        if let reason = eligibility(for: offer) {
            send("task.decline", .object([
                ("taskId", .string(offer.taskId)),
                ("leaseId", .string(offer.leaseId)),
                ("reason", .string(reason)),
            ]))
            emit("task.declined", "\(offer.adapter) — \(reason)")
            return
        }

        send("task.accept", .object([
            ("taskId", .string(offer.taskId)),
            ("leaseId", .string(offer.leaseId)),
        ]))
        emit("task.accepted", "\(offer.adapter) #\(offer.taskId.prefix(8))")

        let renewInterval = max(1.0, Double(offer.leaseSeconds) / 3)
        let renew = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(renewInterval * 1_000_000_000))
                if Task.isCancelled { return }
                await self?.send("lease.renew", .object([
                    ("taskId", .string(offer.taskId)),
                    ("leaseId", .string(offer.leaseId)),
                ]))
            }
        }

        let work = Task { [weak self] in
            guard let self else { return }
            await self.execute(offer)
        }
        running[offer.taskId] = RunningTask(leaseId: offer.leaseId, work: work, renew: renew)
    }

    private func execute(_ offer: TaskOffer) async {
        let startedAt = ISO8601.string(from: clock())
        let t0 = DispatchTime.now().uptimeNanoseconds

        // The wall clock the server granted. A task that blows through it is going to be
        // reclaimed anyway, so stopping here keeps the device from burning battery on
        // work nobody is waiting for any more.
        let deadline = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(offer.wallClockMs) * 1_000_000)
            if !Task.isCancelled { await self?.cancelRunning(offer.taskId) }
        }
        defer { deadline.cancel() }

        let taskId = offer.taskId
        let context = ExecContext(
            hostId: config.hostId, server: config.server, identity: identity,
            artifacts: artifacts,
            onProgress: { [weak self] done, total in
                Task { await self?.report(taskId: taskId, done: done, total: total) }
            })

        do {
            let output: JSONValue
            switch offer.adapter {
            case InferenceAdapter.name:
                output = try await InferenceAdapter.shared.run(offer.input, context: context)
            case WalkerAdapter.name:
                output = try await WalkerAdapter.run(offer.input, context: context)
            default:
                output = try await EchoAdapter.run(offer.input, hostId: config.hostId)
            }

            let finishedAt = ISO8601.string(from: clock())
            let outputHash = hashOutput(output)
            let signature = try Attestation.sign(
                Attestation.Claim(taskId: offer.taskId, attempt: offer.attempt, hostId: config.hostId,
                                  outputHash: outputHash, startedAt: startedAt, finishedAt: finishedAt),
                identity: identity)

            send("task.result", .object([
                ("taskId", .string(offer.taskId)),
                ("leaseId", .string(offer.leaseId)),
                ("attempt", .int(offer.attempt)),
                ("output", output),
                ("outputHash", .string(outputHash)),
                ("startedAt", .string(startedAt)),
                ("finishedAt", .string(finishedAt)),
                ("hostReportedMs", .double(roundTo(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000, 3))),
                // Signed here, on this device, with a key the server has never seen.
                ("signature", .string(signature)),
            ]))
            emit("task.succeeded", "#\(offer.taskId.prefix(8))")
        } catch {
            let cancelled = error is CancellationError
            send("task.error", .object([
                ("taskId", .string(offer.taskId)),
                ("leaseId", .string(offer.leaseId)),
                ("errorClass", .string(cancelled ? "aborted" : "adapter_error")),
                ("message", .string("\(error)")),
            ]))
            emit("task.failed", "#\(offer.taskId.prefix(8)) — \(error)")
        }

        finish(offer.taskId)
    }

    private func report(taskId: String, done: Int, total: Int) {
        onProgress?(taskId, done, total)
        send("task.progress", .object([
            ("taskId", .string(taskId)),
            ("done", .int(done)),
            ("total", .int(total)),
        ]))
    }

    private func finish(_ taskId: String) {
        guard let entry = running.removeValue(forKey: taskId) else { return }
        entry.renew.cancel()
    }

    private func cancelRunning(_ taskId: String) {
        guard let entry = running.removeValue(forKey: taskId) else { return }
        entry.work.cancel()
        entry.renew.cancel()
    }

    private func abandonRunning() {
        for (_, entry) in running {
            entry.work.cancel()
            entry.renew.cancel()
        }
        running.removeAll()
    }

    // ------------------------------------------------------------------ outbound

    private func send(_ type: String, _ payload: JSONValue) {
        guard let socket else { return }
        let text = Envelope(type: type, payload: payload).text
        socket.send(.string(text)) { _ in
            // A failed write shows up as a receive error on the next turn of the loop,
            // which is where reconnection is already handled. Nothing useful to do here.
        }
    }

    private func capabilityJSON() -> JSONValue {
        var record = Capability.probe(adapters: Self.adapters).json
        if case .object(var pairs) = record {
            pairs.append(("mobile", Capability.mobileState().json))
            record = .object(pairs)
        }
        return record
    }

    /// The server sees one flag. It is the union of the owner's switch and the device's
    /// own readiness — both mean "do not send me work", and the server need not care which.
    private func consentJSON() -> JSONValue {
        ConsentState(paused: paused || !deviceFit, allowCompute: config.allowCompute,
                     allowBrowser: false, maxConcurrency: config.maxConcurrency).json
    }
}
