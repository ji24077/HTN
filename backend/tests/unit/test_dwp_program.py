import unittest

from orchestrator.worker.executors.python_project import PythonProjectExecutor


class PublicArtifactOriginTests(unittest.TestCase):
    def test_public_https_artifact_transfers_preserve_tls(self):
        for scheme in ("https", "wss"):
            executor = PythonProjectExecutor(
                f"{scheme}://fleet.example:8443/agent/connect",
                "paired-worker",
                artifact_prefix="/agent/v1",
            )
            self.assertEqual(executor.origin, "https://fleet.example:8443")
            self.assertEqual(executor.artifact_prefix, "/agent/v1")

    def test_existing_private_worker_routes_are_unchanged(self):
        executor = PythonProjectExecutor("ws://127.0.0.1:8080/v1/worker", "worker")
        self.assertEqual(executor.origin, "http://127.0.0.1:8080")
        self.assertEqual(executor.artifact_prefix, "/v1")
