import SwiftUI
import DWPAgentKit

@main
struct DWPAgentApp: App {
    @StateObject private var model = AgentModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
                .task { await autoStartIfRequested() }
        }
    }

    /**
     * Launch-argument driving, for the end-to-end suite.
     *
     * `xcrun simctl launch` can pass `-dwpServer … -dwpCode … -dwpAutoStart 1`, which
     * UserDefaults exposes directly. Without it the only way to check that the real iOS
     * build — not the shared core on a Mac — pairs and takes work would be to tap through
     * it by hand every time, which means it would stop being checked.
     */
    private func autoStartIfRequested() async {
        let defaults = UserDefaults.standard
        if !model.isPaired,
           let server = defaults.string(forKey: "dwpServer"),
           let code = defaults.string(forKey: "dwpCode") {
            let label = defaults.string(forKey: "dwpLabel") ?? "iPhone"
            await model.pair(server: server, code: code, label: label)
        }
        if defaults.bool(forKey: "dwpAutoStart") && model.isPaired {
            await model.start()
        }
    }
}
