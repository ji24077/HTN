import os
import unittest
from pathlib import Path
from unittest.mock import patch

from orchestrator.shared.worker_telemetry import worker_telemetry
from orchestrator.worker.setup import worker_environment


class WorkerTelemetryTests(unittest.TestCase):
    def test_only_public_ingest_settings_are_distributed(self):
        with patch.dict(
            os.environ,
            {
                "SENTRY_DSN": "https://public@example.invalid/123",
                "SENTRY_ENVIRONMENT": "fleet",
                "SENTRY_RELEASE": "version-1",
                "SENTRY_AUTH_TOKEN": "do-not-send",
                "DATABASE_URL": "postgres://user:secret@db/main",
            },
            clear=True,
        ):
            self.assertEqual(
                worker_telemetry(),
                {
                    "dsn": "https://public@example.invalid/123",
                    "environment": "fleet",
                    "release": "version-1",
                },
            )

    def test_worker_override_disable_and_invalid_credentials(self):
        for value, expected in (
            ("https://worker@example.invalid/456", "https://worker@example.invalid/456"),
            ("", None),
            ("https://public:secret@example.invalid/123", None),
            ("https://public:@example.invalid/123", None),
            ("http://public@example.invalid/123", None),
            ("https://public@example.invalid/123?token=secret", None),
        ):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {
                        "SENTRY_DSN": "https://backend@example.invalid/123",
                        "SENTRY_WORKER_DSN": value,
                    },
                    clear=True,
                ),
            ):
                self.assertEqual(worker_telemetry()["dsn"], expected)

    def test_python_enrollment_profile_includes_telemetry_without_api_credentials(self):
        data = {
            "worker_id": "worker-123",
            "worker_token": "test-worker-token-1234567890",
            "tailscale_auth_key": "tskey-auth-test",
            "server_url": "wss://backend.example.ts.net/v1/worker",
            "telemetry": {
                "dsn": "https://public@example.invalid/123",
                "environment": "fleet",
                "release": "v1",
                "auth_token": "do-not-save",
            },
        }
        env = worker_environment(data, Path("/tmp/profile"), "/tmp/helper")
        self.assertEqual(env["SENTRY_DSN"], data["telemetry"]["dsn"])
        self.assertEqual(env["SENTRY_ENVIRONMENT"], "fleet")
        self.assertEqual(env["SENTRY_RELEASE"], "v1")
        self.assertNotIn("do-not-save", str(env))
        self.assertNotIn("TS_AUTHKEY", env)
        data["telemetry"]["dsn"] = "https://key:secret@example.invalid/123"
        self.assertEqual(worker_environment(data, Path("/tmp"), "/tmp/helper")["SENTRY_DSN"], "")
        del data["telemetry"]
        self.assertNotIn("SENTRY_DSN", worker_environment(data, Path("/tmp"), "/tmp/helper"))
