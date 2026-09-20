"""Exercise success/failure log export through the real Sentry SDK without a network."""

import json
import os
import unittest
from datetime import UTC, datetime
from unittest.mock import patch

import sentry_sdk
from sentry_sdk.transport import Transport

from orchestrator.preprocessing.training_telemetry import publish, record
from orchestrator.shared.telemetry import init_sentry


def job():
    return {
        "job_id": "synthetic-preparation-job",
        "original_hash": "original",
        "data": {
            "program_plan": {
                "worker_id": "nvidia-target",
                "preparation": {
                    "source_worker_id": "amd-source",
                    "source_vendor": "amd",
                    "target_vendor": "nvidia",
                },
            },
            "limits": {"adaptations": 3},
            "training_preparation": {
                "attempts": {"optimization": 1, "migration": 1},
                "accepted_hash": "original",
                "candidate_hash": "candidate",
                "started_at": datetime.now(UTC).isoformat(),
            },
        },
    }


class TrainingTelemetryTests(unittest.TestCase):
    def test_persisted_events_are_deduplicated_scrubbed_and_keep_evidence(self):
        value = job()
        credential = "synthetic-credential-0123456789"
        with patch.dict(os.environ, {"API_KEY": credential}):
            for _ in range(2):
                record(value, "migration", "rejected", evidence={"error": credential})
        events = value["data"]["training_preparation"]["telemetry"]
        self.assertEqual(len(events), 1)
        self.assertNotIn(credential, json.dumps(events))
        record(value, "migration", "passed", evidence={"report": {"loss_history": [0.3]}})
        self.assertEqual(events[1]["loss_final"], 0.3)
        self.assertEqual(events[1]["evidence_kind"], "worker_execution")
        self.assertNotEqual(events[0]["preparation_event_id"], events[1]["preparation_event_id"])

    def test_failed_sentry_handler_cannot_fail_the_pipeline(self):
        value = job()
        record(value, "optimization", "kept_baseline")
        events = value["data"]["training_preparation"]["telemetry"]
        with patch(
            "orchestrator.preprocessing.training_telemetry.log.log",
            side_effect=RuntimeError("offline"),
        ):
            publish(events)
        self.assertEqual(events[0]["candidate_hash"], "original")

    def test_success_and_failure_reach_sentry_logs_with_stage_attributes(self):
        envelopes = []

        class MemoryTransport(Transport):
            def capture_envelope(self, envelope):
                envelopes.append(envelope)

        real_init = sentry_sdk.init

        def initialize(**options):
            return real_init(**options, transport=MemoryTransport)

        previous = sentry_sdk.get_client()
        try:
            with (
                patch.dict(os.environ, {"SENTRY_DSN": "https://public@example.test/1"}),
                patch("orchestrator.shared.telemetry.sentry_sdk.init", side_effect=initialize),
            ):
                self.assertTrue(init_sentry("server"))
                value = job()
                record(value, "optimization", "passed")
                record(value, "migration", "rejected")
                publish(value["data"]["training_preparation"]["telemetry"])
                sentry_sdk.flush()
            logs = [
                log
                for envelope in envelopes
                for item in envelope.items
                if item.headers.get("type") == "log"
                for log in json.loads(item.get_bytes())["items"]
            ]
            self.assertEqual(len(logs), 2)
            self.assertEqual(
                [
                    (log["attributes"]["stage"]["value"], log["attributes"]["outcome"]["value"])
                    for log in logs
                ],
                [("optimization", "passed"), ("migration", "rejected")],
            )
            self.assertTrue(
                all(log["attributes"]["source_vendor"]["value"] == "amd" for log in logs)
            )
        finally:
            sentry_sdk.get_client().close()
            sentry_sdk.get_global_scope().set_client(previous)
