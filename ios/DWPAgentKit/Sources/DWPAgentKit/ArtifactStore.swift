import Foundation

/**
 * Content-addressed artifact cache.
 *
 * The device is told a hash, never the data. It fetches, verifies, and only then uses —
 * and a hash mismatch is a hard failure rather than a warning, because running the wrong
 * model produces confident, wrong, silently unverifiable results. That rule is copied
 * from the desktop adapter deliberately: the two must agree about what is fatal, or the
 * cross-host agreement check compares one machine's caution against another's.
 */
public actor ArtifactStore {
    private let directory: URL
    private let session: URLSession
    /// Two slices of one job start together and want the same 7.8 MB of inputs. Without
    /// this they would each fetch it.
    private var inFlight: [String: Task<Data, Error>] = [:]

    public init(directory: URL? = nil, session: URLSession = .shared) {
        if let directory {
            self.directory = directory
        } else {
            let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            self.directory = base.appendingPathComponent("dwp-artifacts")
        }
        self.session = session
    }

    private func path(_ sha256: String) -> URL { directory.appendingPathComponent(sha256) }

    public func cached(_ sha256: String) -> Bool {
        FileManager.default.fileExists(atPath: path(sha256).path)
    }

    public func evictAll() {
        try? FileManager.default.removeItem(at: directory)
    }

    /// Same fetch, but hands back the verified file on disk.
    ///
    /// ONNX Runtime's Objective-C API opens a model by path rather than from memory, and
    /// writing a 26 MB model to a second temporary file just to satisfy that would double
    /// the storage cost on the most storage-constrained host we have.
    public func fetchFile(sha256: String, server: String, hostId: String,
                          identity: HostIdentity) async throws -> URL {
        _ = try await fetch(sha256: sha256, server: server, hostId: hostId, identity: identity)
        return path(sha256)
    }

    public func fetch(sha256: String, server: String, hostId: String, identity: HostIdentity) async throws -> Data {
        if let existing = inFlight[sha256] { return try await existing.value }

        let task = Task<Data, Error> { [directory, session] in
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let file = directory.appendingPathComponent(sha256)

            if let bytes = try? Data(contentsOf: file) {
                if sha256Hex(bytes) == sha256 { return bytes }
                // A cache entry that no longer matches its own name is corrupt. Drop it
                // and refetch rather than trusting it or failing the task.
                try? FileManager.default.removeItem(at: file)
            }

            guard let url = URL(string: "\(server)/artifacts/\(sha256)") else {
                throw AgentError.artifact("bad artifact URL for \(sha256.prefix(12))")
            }
            var request = URLRequest(url: url)
            request.timeoutInterval = 120
            request.setValue("Bearer \(try Assertion.mint(hostId: hostId, identity: identity))",
                             forHTTPHeaderField: "authorization")

            let (bytes, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                let code = (response as? HTTPURLResponse)?.statusCode ?? -1
                throw AgentError.artifact("could not fetch artifact \(sha256.prefix(12)): HTTP \(code)")
            }
            let actual = sha256Hex(bytes)
            guard actual == sha256 else {
                throw AgentError.artifact(
                    "artifact hash mismatch: expected \(sha256.prefix(12)), got \(actual.prefix(12))")
            }
            try bytes.write(to: file, options: .atomic)
            return bytes
        }

        inFlight[sha256] = task
        defer { inFlight[sha256] = nil }
        return try await task.value
    }
}
