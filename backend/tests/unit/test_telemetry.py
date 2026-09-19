"""Check that credentials never survive the Sentry scrubber."""

import json
import os
import unittest
from unittest.mock import patch

from orchestrator.shared.telemetry import REDACTED, Scrubber, init_sentry, secret_values

ADMIN = "admin-token-0123456789abcdefghij"
WORKER = "worker-token-0123456789abcdefghi"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlLXZhbHVl"
ENV = {
    "ADMIN_TOKEN": ADMIN,
    "WORKER_TOKENS": json.dumps({"worker-1": WORKER}),
    "DATABASE_URL": "postgresql://postgres:db-password-123@db.example:5432/postgres",
    "TAILSCALE_OAUTH_CLIENT_SECRET": "tskey-client-abc123-secretsecret",
    "SENTRY_DSN": "",
}


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, ENV)
        env.start()
        self.addCleanup(env.stop)
        self.scrubber = Scrubber(secret_values())

    def test_blank_dsn_disables_reporting(self):
        self.assertFalse(init_sentry("server"))

    def test_configured_secrets_are_collected_by_value(self):
        secrets = secret_values()
        for value in (ADMIN, WORKER, "db-password-123", ENV["TAILSCALE_OAUTH_CLIENT_SECRET"]):
            self.assertIn(value, secrets)

    def test_event_is_scrubbed_but_keeps_debugging_context(self):
        event = {
            "request": {
                "headers": {"Authorization": f"Bearer {JWT}", "Cookie": "a=b", "Host": "x.test"},
                "data": {"tasks": [{"id": "task-1", "payload": {"label": "demo"}}]},
                "query_string": [["access_token", JWT], ["page", "2"]],
            },
            "exception": {
                "values": [
                    {
                        "value": f"connect failed for Bearer {WORKER}",
                        "stacktrace": {
                            "frames": [
                                {
                                    "vars": {
                                        "config": f"WorkerConfig(token='{WORKER}')",
                                        "worker_token": "anything",
                                        "url": ENV["DATABASE_URL"],
                                        "progress": 40.0,
                                    }
                                }
                            ]
                        },
                    }
                ]
            },
        }
        cleaned = self.scrubber.event(event)
        text = json.dumps(cleaned)
        for secret in (ADMIN, WORKER, JWT, "db-password-123"):
            self.assertNotIn(secret, text)
        request = cleaned["request"]
        self.assertEqual(request["headers"]["Authorization"], REDACTED)
        self.assertEqual(request["headers"]["Host"], "x.test")
        self.assertEqual(request["data"]["tasks"][0]["payload"], {"label": "demo"})
        self.assertEqual(request["query_string"], [["access_token", REDACTED], ["page", "2"]])
        variables = cleaned["exception"]["values"][0]["stacktrace"]["frames"][0]["vars"]
        self.assertEqual(variables["worker_token"], REDACTED)
        self.assertEqual(variables["progress"], 40.0)
        self.assertIn("db.example:5432", variables["url"])
