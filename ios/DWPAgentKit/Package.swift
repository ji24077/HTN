// swift-tools-version: 6.0
import PackageDescription

/**
 * The iOS agent's core, as a package rather than an app target.
 *
 * Everything that decides anything lives here — protocol encoding, host identity,
 * transport, adapters — so it can be exercised by `swift test` on a Mac instead of
 * only by tapping a phone. The iOS app is a thin shell around this, and the macOS
 * `dwpagent` executable is the same core with a terminal instead of a screen, which
 * is what lets the existing simulator drive it like any other host.
 */
let package = Package(
    name: "DWPAgentKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "DWPAgentKit", targets: ["DWPAgentKit"]),
        .executable(name: "dwpagent", targets: ["dwpagent"]),
    ],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", exact: "1.24.2"),
    ],
    targets: [
        .target(
            name: "DWPAgentKit",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ]
        ),
        .executableTarget(name: "dwpagent", dependencies: ["DWPAgentKit"]),
        .testTarget(name: "DWPAgentKitTests", dependencies: ["DWPAgentKit"]),
    ]
)
