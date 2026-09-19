"""Check that credentials never survive the Sentry scrubber."""

import json
import os
import unittest
from datetime import UTC, datetime
from unittest.mock import patch

import sentry_sdk
from sentry_sdk.transport import Transport

from orchestrator.shared.protocol import Task, TaskSpec
from orchestrator.shared.telemetry import REDACTED, Scrubber, init_sentry, secret_values
from orchestrator.worker.agent import execute
from orchestrator.worker.executors import StubExecutor

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

    def test_device_invites_and_signing_keys_are_redacted(self):
        cleaned = self.scrubber.event(
            {
                "code": "SINGLE-USE-INVITE",
                "privateKey": "private-material",
                "message": "-----BEGIN PRIVATE KEY-----\nprivate-material\n-----END PRIVATE KEY-----",
                "url": "https://fleet.example/join?code=SINGLE-USE-INVITE",
            }
        )
        self.assertNotIn("SINGLE-USE-INVITE", json.dumps(cleaned))
        self.assertNotIn("private-material", json.dumps(cleaned))

    def test_invite_code_in_serialized_frame_locals_is_redacted(self):
        code = "0123456789abcdef0123456789abcdef"
        for value in (
            f"PairRequest(code='{code}')",
            repr(bytearray(json.dumps({"code": code}).encode())),
        ):
            self.assertNotIn(code, self.scrubber.text(value))

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


class WorkerTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_does_not_send_credentials_in_model_locals(self):
        envelopes = []

        class MemoryTransport(Transport):
            def capture_envelope(self, envelope):
                envelopes.append(envelope)

        real_init = sentry_sdk.init

        def init_with_memory_transport(**options):
            return real_init(**options, transport=MemoryTransport)

        credential = "synthetic-runtime-credential-0123456789"
        task = Task(
            spec=TaskSpec(
                id="task-1",
                job_id="job-1",
                kind="stub",
                payload={"fail": True, "value": {"api_key": credential}},
                requirements={"runtime": "cpu", "vram_mib": 0},
                max_attempts=1,
                timeout_seconds=10,
            ),
            state="running",
            generation=1,
            created_at=datetime.now(UTC),
        )
        previous_client = sentry_sdk.get_client()
        try:
            with (
                patch.dict(os.environ, {"SENTRY_DSN": "https://public@example.test/1"}),
                patch(
                    "orchestrator.shared.telemetry.sentry_sdk.init",
                    side_effect=init_with_memory_transport,
                ),
            ):
                self.assertTrue(init_sentry("worker"))
                result = await execute(StubExecutor(), task, lambda _progress: None)
                sentry_sdk.flush()
            self.assertEqual(result.type, "failed")
            events = [
                item.payload.json
                for envelope in envelopes
                for item in envelope.items
                if item.headers.get("type") == "event"
            ]
            self.assertEqual(len(events), 1)
            frames = events[0]["exception"]["values"][0]["stacktrace"]["frames"]
            self.assertTrue(any(frame["function"] == "execute" for frame in frames))
            self.assertTrue(all(not frame.get("vars") for frame in frames))
            self.assertNotIn(credential.encode(), b"".join(e.serialize() for e in envelopes))
        finally:
            sentry_sdk.get_client().close()
            sentry_sdk.get_global_scope().set_client(previous_client)
