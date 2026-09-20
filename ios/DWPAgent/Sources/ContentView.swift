import SwiftUI
import DWPAgentKit

struct ContentView: View {
    @EnvironmentObject private var model: AgentModel

    var body: some View {
        NavigationStack {
            if model.isPaired { StatusView() } else { PairingView() }
        }
    }
}

// ---------------------------------------------------------------------- pairing

/**
 * Pasting a link, not typing a code.
 *
 * Typing `ABCD-1234` on a phone keyboard is exactly the friction a native app exists to
 * remove, so the link the owner already sends is the primary path and the manual fields
 * are the fallback for when someone reads a code aloud.
 */
struct PairingView: View {
    @EnvironmentObject private var model: AgentModel
    @State private var link = ""
    @State private var label = UIDevice.current.name
    @State private var manual = false
    @State private var server = ""
    @State private var code = ""

    var body: some View {
        Form {
            Section {
                Text("Join a fleet")
                    .font(.title2.weight(.semibold))
                Text("Paste the invite link you were sent. This phone generates its own key; "
                   + "the private half never leaves the device.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            if manual {
                Section("Server") {
                    TextField("https://their-address", text: $server)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                }
                Section("Pairing code") {
                    TextField("ABCD-1234", text: $code)
                        .textInputAutocapitalization(.characters)
                        .autocorrectionDisabled()
                }
            } else {
                Section("Invite link") {
                    TextField("https://…/join?code=…", text: $link)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                }
            }

            Section("This device will appear as") {
                TextField("Name", text: $label)
            }

            if let error = model.pairingError {
                Section {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            }

            Section {
                Button {
                    Task {
                        if manual {
                            await model.pair(server: server, code: code, label: label)
                        } else {
                            await model.pair(link: link, label: label)
                        }
                    }
                } label: {
                    HStack {
                        if model.isPairing { ProgressView().padding(.trailing, 6) }
                        Text(model.isPairing ? "Pairing…" : "Pair this phone")
                    }
                }
                .disabled(model.isPairing || (manual ? server.isEmpty || code.isEmpty : link.isEmpty))

                Button(manual ? "I have a link instead" : "Enter a code by hand") {
                    manual.toggle()
                    model.clearError()
                }
                .font(.footnote)
            }
        }
        .navigationTitle("Distributed Work")
    }
}

// ----------------------------------------------------------------------- status

struct StatusView: View {
    @EnvironmentObject private var model: AgentModel
    @State private var showForget = false

    var body: some View {
        List {
            Section {
                HStack(spacing: 12) {
                    Circle()
                        .fill(statusColor)
                        .frame(width: 10, height: 10)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(statusText).font(.headline)
                        if let config = model.config {
                            Text(config.label).font(.footnote).foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                    Text("\(model.completedTasks)")
                        .font(.title3.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)

                if let progress = model.progress {
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: Double(progress.done), total: Double(max(progress.total, 1)))
                        Text("\(progress.done) of \(progress.total) items")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Section("This device") {
                let device = Capability.mobileState()
                LabeledContent("Thermal", value: device.thermal)
                LabeledContent("Low Power Mode", value: device.lowPowerMode ? "on" : "off")
                if let level = device.batteryLevel {
                    LabeledContent("Battery",
                                   value: "\(Int(level * 100))%\(device.charging == true ? ", charging" : "")")
                }
                LabeledContent("Memory headroom", value: "\(device.availableMemoryMb) MB")
                if !device.fitForWork {
                    Text("Holding back: \(device.unfitReason)")
                        .font(.footnote)
                        .foregroundStyle(.orange)
                }
            }

            Section {
                Toggle("Paused", isOn: Binding(
                    get: { model.paused },
                    set: { value in Task { await model.setPaused(value) } }))
                Text("Pausing stops new work immediately, without disconnecting. It is stored "
                   + "on this phone, so it works even when the server cannot be reached.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if !model.backgroundNote.isEmpty {
                Section("Running in the background") {
                    Text(model.backgroundNote).font(.footnote).foregroundStyle(.secondary)
                    Text("A slice that is already running continues when you leave the app, with "
                       + "progress you can watch and cancel. Between slices, iOS suspends the app.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section("Activity") {
                if model.events.isEmpty {
                    Text("Nothing yet.").font(.footnote).foregroundStyle(.secondary)
                }
                ForEach(Array(model.events.prefix(40).enumerated()), id: \.offset) { _, event in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(event.type).font(.footnote.weight(.medium))
                        if !event.detail.isEmpty {
                            Text(event.detail).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }

            Section {
                Button("Forget this fleet", role: .destructive) { showForget = true }
            }
        }
        .navigationTitle("Contributing")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button(isRunning ? "Stop" : "Start") {
                    Task { isRunning ? await model.stop() : await model.start() }
                }
            }
        }
        .confirmationDialog("Forget this fleet?", isPresented: $showForget, titleVisibility: .visible) {
            Button("Forget", role: .destructive) { Task { await model.forget() } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This deletes the key and pairing on this phone. You will need a new invite to rejoin.")
        }
    }

    private var isRunning: Bool {
        switch model.state {
        case .idle, .stopped, .revoked: return false
        default: return true
        }
    }

    private var statusColor: Color {
        switch model.state {
        case .connected: return .green
        case .connecting, .waiting: return .orange
        case .revoked: return .red
        case .idle, .stopped: return .secondary
        }
    }

    private var statusText: String {
        switch model.state {
        case .idle: return "Not started"
        case .connecting(let attempt): return attempt > 1 ? "Connecting (attempt \(attempt))" : "Connecting"
        case .connected: return "Connected"
        case .waiting(let seconds, let reason): return String(format: "Reconnecting in %.0fs — %@", seconds, reason)
        case .revoked(let reason): return "Revoked — \(reason)"
        case .stopped: return "Stopped"
        }
    }
}
