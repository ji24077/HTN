"""Actual HTTP/WebSocket exchange across the Jack-to-Python scheduler bridge."""

import base64
import hashlib
import json
import os
import time
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from orchestrator.server.app import create_public_app, create_worker_app
from orchestrator.server.db.store import Conflict, StaleAssignment, StaleSession
from orchestrator.shared.protocol import Task, TaskSpec, task_ref

ADMIN = "test-admin-token-0123456789012345"
USER = "f19ed307-6827-4b38-9a22-fc291727dd97"
ORIGIN = "https://fleet.example.com"


def frame(kind, payload):
    return {
        "v": 1,
        "id": str(uuid4()),
        "ts": "2026-09-19T12:00:00.000Z",
        "type": kind,
        "payload": payload,
    }


def hello(*, paused=False, allow=True):
    return frame(
        "hello",
        {
            "capability": {
                "adapters": ["echo", "walker_evolution", "untrusted-command"],
                "agentVersion": "test",
                "os": "ios",
                "arch": "arm64",
            },
            "consent": {
                "paused": paused,
                "allowCompute": allow,
                "allowBrowser": False,
                "maxConcurrency": 8,
            },
        },
    )


def task(number):
    return Task(
        spec=TaskSpec(
            id=f"task-{number}",
            job_id="job-1",
            kind="echo",
            payload={"nonce": number},
            requirements={"runtime": "cpu", "vram_mib": 0},
            max_attempts=3,
            timeout_seconds=60,
        ),
        state="queued",
        generation=0,
        created_at=datetime.now(timezone.utc),
    )


class FakeStore:
    """Model durable ownership/session contracts; cryptography stays real."""

    def __init__(self, public_key):
        self.worker_id = str(uuid4())
        self.public_key = public_key
        self.key_lookups = 0
        self.jtis = set()
        self.codes = set()
        self.tasks = [task(1), task(2)]
        self.session = None
        self.registered = []
        self.renewals = []
        self.finished = []
        self.disconnected = []
        self.paused = False
        self.execution_batches = []

    async def append_execution_events(self, worker_id, session, batch):
        self.assert_session(session)
        current = await self.task(batch.taskId)
        if current.worker_id != worker_id or current.generation != batch.attempt:
            raise StaleAssignment("not assigned")
        self.execution_batches.append(batch)
        return [item.sequence for item in batch.events]

    async def create_pair_code(self, owner_id):
        self.owner = owner_id
        code = uuid4().hex.upper()
        self.codes.add(code)
        return code

    async def pair_device(self, code, public_key, label):
        if code not in self.codes:
            raise Conflict("expired")
        self.codes.remove(code)
        self.public_key = public_key
        return self.worker_id

    async def device_key(self, worker_id):
        self.key_lookups += 1
        return self.public_key if worker_id == self.worker_id else None

    async def use_assertion_jti(self, worker_id, jti, expires_at):
        if jti in self.jtis:
            return False
        self.jtis.add(jti)
        return True

    async def register(self, worker_id, session, capabilities, *, expected_device_key=None):
        if expected_device_key != self.public_key:
            raise StaleSession("device revoked")
        for current in self.tasks:
            if current.state in {"assigned", "running"}:
                current.state = "queued"
        self.session = session
        self.registered.append(capabilities)

    def assert_session(self, session):
        if session != self.session:
            raise StaleSession("superseded")

    async def heartbeat(self, worker_id, session, active, paused, progress=0):
        self.assert_session(session)
        self.paused = paused
        self.renewals.extend(active)
        return [
            ref
            for ref in active
            if any(
                task_ref(current) == ref and current.state == "running" for current in self.tasks
            )
        ]

    async def claim(self, worker_id, session):
        self.assert_session(session)
        if self.paused or any(t.state in {"assigned", "running"} for t in self.tasks):
            return None
        current = next((t for t in self.tasks if t.state == "queued"), None)
        if current:
            current.state = "assigned"
            current.generation += 1
            current.worker_id, current.session_id = worker_id, session
            current.lease_until = datetime.now(timezone.utc) + timedelta(seconds=45)
            current.deadline = datetime.now(timezone.utc) + timedelta(seconds=60)
        return current

    async def task(self, task_id):
        return next(t for t in self.tasks if t.spec.id == task_id)

    async def ack(self, worker_id, session, ref):
        self.assert_session(session)
        current = await self.task(ref.task_id)
        if task_ref(current) != ref or current.state not in {"assigned", "running"}:
            raise StaleAssignment("cancelled")
        current.state = "running"

    async def finish(
        self, worker_id, session, ref, result, failure, retryable, *, attestation=None
    ):
        self.assert_session(session)
        current = await self.task(ref.task_id)
        if task_ref(current) != ref or current.state != "running":
            raise StaleAssignment("cancelled")
        current.state = "queued" if failure and retryable else "succeeded"
        self.finished.append((ref, result, failure, attestation))

    async def disconnect(self, worker_id, session):
        self.disconnected.append(session)


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.private = Ed25519PrivateKey.generate()
        self.public = base64.b64encode(
            self.private.public_key().public_bytes(
                serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
            )
        ).decode()
        self.store = FakeStore(self.public)
        self.app = create_public_app()
        self.app.state.config = SimpleNamespace(
            admin_token=ADMIN,
            public_origin=ORIGIN,
            supabase_admin_ids={USER},
            supabase_admin_emails=frozenset(),
        )
        self.app.state.store = self.store

        async def verify(token):
            if token != "approved":
                raise HTTPException(401)
            return {"sub": USER}

        self.app.state.supabase_auth = SimpleNamespace(verify=AsyncMock(side_effect=verify))
        self.client = TestClient(self.app, base_url=ORIGIN)
        releases = patch(
            "orchestrator.server.dwp.release_info",
            return_value={
                "releaseKey": None,
                "releaseVersion": None,
            },
        )
        releases.start()
        self.addCleanup(releases.stop)

    def token(self, **overrides):
        now = int(time.time())
        claims = {
            "iss": self.store.worker_id,
            "aud": "dwp-control",
            "iat": now,
            "exp": now + 120,
            "jti": str(uuid4()),
            **overrides,
        }
        return jwt.encode(claims, self.private, algorithm="EdDSA")

    def connect(self, token=None):
        return self.client.websocket_connect(
            "/agent/connect", headers={"Authorization": "Bearer " + (token or self.token())}
        )

    def start(self, socket):
        socket.send_json(hello())
        ack = socket.receive_json()
        self.assertEqual(ack["type"], "hello.ack")
        self.assertEqual(ack["payload"]["heartbeatSeconds"], 5)
        offer = socket.receive_json()
        self.assertEqual(offer["type"], "task.offer")
        return offer["payload"]

    def ref(self, offer):
        return {"taskId": offer["taskId"], "leaseId": offer["leaseId"]}

    def result(self, offer):
        output = {"nonce": 1, "value": 0.0000001, "label": "東京"}
        encoded = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        start, end = "2026-09-19T12:00:00.000Z", "2026-09-19T12:00:00.001Z"
        signed = (
            f"dwp-attest/v1\n{offer['taskId']}\n{offer['attempt']}\n"
            f"{self.store.worker_id}\n{digest}\n{start}\n{end}\n"
        )
        signature = (
            base64.urlsafe_b64encode(self.private.sign(signed.encode())).decode().rstrip("=")
        )
        return frame(
            "task.result",
            {
                **self.ref(offer),
                "attempt": offer["attempt"],
                "output": output,
                "outputHash": digest,
                "startedAt": start,
                "finishedAt": end,
                "hostReportedMs": 1,
                "signature": signature,
            },
        )

    def send_result(self, socket, result):
        socket.send_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")))

    def test_pairing_requires_admin_then_one_use_code_and_valid_key(self):
        self.assertEqual(self.client.post("/v1/device-invites").status_code, 401)
        issued = self.client.post(
            "/v1/device-invites", headers={"Authorization": "Bearer approved"}
        )
        self.assertEqual(issued.status_code, 201)
        self.assertEqual(issued.headers["cache-control"], "no-store")
        self.assertEqual(self.store.owner, USER)
        self.assertEqual(issued.json()["server"], ORIGIN)
        body = {"code": issued.json()["code"].lower(), "publicKey": self.public, "label": "Phone"}
        bad = self.client.post("/hosts/pair", json={**body, "publicKey": "not-a-public-key-at-all"})
        self.assertEqual(bad.status_code, 400)
        paired = self.client.post("/hosts/pair", json=body)
        self.assertEqual(paired.status_code, 200)
        self.assertEqual(paired.json()["wsUrl"], "wss://fleet.example.com/agent/connect")
        self.assertEqual(paired.json()["hostId"], self.store.worker_id)
        self.assertEqual(self.client.post("/hosts/pair", json=body).status_code, 400)

    def test_pairing_and_reconnect_deliver_current_public_worker_telemetry(self):
        with patch.dict(
            os.environ,
            {
                "SENTRY_DSN": "https://backend@example.invalid/1",
                "SENTRY_WORKER_DSN": "https://worker@example.invalid/2",
                "SENTRY_ENVIRONMENT": "fleet",
                "SENTRY_AUTH_TOKEN": "must-not-send",
            },
            clear=True,
        ):
            issued = self.client.post(
                "/v1/device-invites", headers={"Authorization": "Bearer approved"}
            )
            paired = self.client.post(
                "/hosts/pair",
                json={
                    "code": issued.json()["code"],
                    "publicKey": self.public,
                    "label": "Desktop",
                },
            )
            expected = {
                "dsn": "https://worker@example.invalid/2",
                "environment": "fleet",
                "release": None,
            }
            self.assertEqual(paired.json()["telemetry"], expected)
            self.assertNotIn("must-not-send", paired.text)
            self.assertEqual(paired.headers["cache-control"], "no-store")
            with self.connect() as socket:
                socket.send_json(hello(paused=True))
                self.assertEqual(socket.receive_json()["payload"]["telemetry"], expected)
            os.environ["SENTRY_WORKER_DSN"] = ""
            with self.connect() as socket:
                socket.send_json(hello(paused=True))
                self.assertIsNone(socket.receive_json()["payload"]["telemetry"]["dsn"])

    def test_pairing_body_and_attempts_are_bounded(self):
        response = self.client.post("/hosts/pair", content="x" * 5000)
        self.assertEqual(response.status_code, 413)
        for _ in range(19):
            self.assertEqual(self.client.post("/hosts/pair", json={}).status_code, 400)
        self.assertEqual(self.client.post("/hosts/pair", json={}).status_code, 429)

    def test_malformed_issuer_rejected_before_lookup_and_assertion_replay_denied(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.connect(self.token(iss="not-a-uuid")):
                self.fail("invalid issuer connected")
        self.assertEqual(self.store.key_lookups, 0)
        token = self.token()
        with self.connect(token) as socket:
            self.start(socket)
        with self.assertRaises(WebSocketDisconnect):
            with self.connect(token):
                self.fail("replayed assertion connected")

    def test_signed_result_persists_proof_and_single_slot_lease_binding(self):
        with self.connect() as socket:
            offer = self.start(socket)
            socket.send_json(frame("task.accept", self.ref(offer)))
            socket.send_json(frame("heartbeat", {"freeRamMb": 200, "running": 0}))
            socket.send_json(frame("lease.renew", {"taskId": "unknown", "leaseId": "wrong"}))
            self.assertEqual(socket.receive_json()["type"], "task.cancel")
            self.assertEqual(self.store.renewals, [])
            self.assertEqual(sum(t.state in {"assigned", "running"} for t in self.store.tasks), 1)
            socket.send_json(frame("lease.renew", self.ref(offer)))
            self.send_result(socket, self.result(offer))
            next_offer = socket.receive_json()
            self.assertEqual(next_offer["type"], "task.offer")
            self.assertEqual(next_offer["payload"]["taskId"], "task-2")
        self.assertEqual(len(self.store.renewals), 1)
        _, output, failure, proof = self.store.finished[0]
        self.assertEqual(output["label"], "東京")
        self.assertEqual(failure, "")
        self.assertEqual(proof["hostId"], self.store.worker_id)
        self.assertIn('"label":"東京"', proof["rawOutput"])
        self.assertEqual(self.store.registered[0].runtime, "cpu")
        self.assertEqual(self.store.registered[0].kinds, ["echo", "walker_evolution"])

    def test_tampered_output_or_signature_never_finishes_assignment(self):
        for field in ("output", "signature"):
            with self.subTest(field=field):
                self.store.tasks = [task(1)]
                with self.connect() as socket:
                    offer = self.start(socket)
                    socket.send_json(frame("task.accept", self.ref(offer)))
                    result = self.result(offer)
                    if field == "output":
                        result["payload"]["output"]["nonce"] = "tampered"
                    else:
                        result["payload"]["signature"] = "AAAA"
                    self.send_result(socket, result)
                    with self.assertRaises(WebSocketDisconnect):
                        socket.receive_json()
                self.assertEqual(self.store.finished, [])

    def test_cancelled_assignment_is_cancelled_on_device_and_cannot_finish(self):
        with self.connect() as socket:
            offer = self.start(socket)
            socket.send_json(frame("task.accept", self.ref(offer)))
            # Invalid task reference provides an ordering barrier for the preceding ack.
            socket.send_json(frame("lease.renew", {"taskId": "unknown", "leaseId": "wrong"}))
            socket.receive_json()
            self.store.tasks[0].state = "cancelled"
            socket.send_json(frame("heartbeat", {"freeRamMb": 200, "running": 1}))
            self.assertEqual(socket.receive_json()["type"], "task.cancel")
            next_offer = socket.receive_json()["payload"]
            self.assertEqual(next_offer["taskId"], "task-2")
            self.send_result(socket, self.result(offer))
            self.assertEqual(socket.receive_json()["type"], "task.cancel")
        self.assertEqual(self.store.finished, [])

    def test_reconnect_supersedes_old_session_and_changes_generation_and_lease(self):
        with self.connect() as old:
            first = self.start(old)
            old.send_json(frame("task.accept", self.ref(first)))
            old.send_json(frame("lease.renew", {"taskId": "unknown", "leaseId": "wrong"}))
            old.receive_json()
            with self.connect() as new:
                second = self.start(new)
                self.assertEqual(second["attempt"], first["attempt"] + 1)
                self.assertNotEqual(second["leaseId"], first["leaseId"])
                old.send_json(frame("lease.renew", self.ref(first)))
                with self.assertRaises(WebSocketDisconnect) as closed:
                    old.receive_json()
                self.assertEqual(closed.exception.code, 4000)
                new.send_json(frame("task.accept", self.ref(second)))
                self.send_result(new, self.result(first))
                self.send_result(new, self.result(second))
                self.assertEqual(new.receive_json()["payload"]["taskId"], "task-2")
        self.assertEqual(len(self.store.finished), 1)
        self.assertEqual(self.store.finished[0][0].generation, 2)

    def test_consent_pause_blocks_dispatch_and_errors_report_without_device_text(self):
        with patch("orchestrator.server.dwp.sentry_sdk.capture_message") as capture:
            with self.connect() as socket:
                socket.send_json(hello(paused=True))
                self.assertEqual(socket.receive_json()["type"], "hello.ack")
                socket.send_json(frame("lease.renew", {"taskId": "unknown", "leaseId": "wrong"}))
                self.assertEqual(socket.receive_json()["type"], "task.cancel")
                self.assertEqual(self.store.tasks[0].state, "queued")
                socket.send_json(frame("consent.update", hello()["payload"]["consent"]))
                offer = socket.receive_json()["payload"]
                socket.send_json(frame("task.accept", self.ref(offer)))
                socket.send_json(
                    frame(
                        "task.error",
                        {
                            **self.ref(offer),
                            "errorClass": "adapter_error",
                            "message": "secret-from-device",
                        },
                    )
                )
                self.assertEqual(socket.receive_json()["type"], "task.offer")
            self.assertEqual(self.store.finished[0][2], "device_execution_failed")
            capture.assert_called_once_with("Paired device execution failed", level="error")

    def test_listener_boundaries_and_invalid_frames(self):
        private = TestClient(create_worker_app())
        self.assertEqual(private.post("/hosts/pair").status_code, 404)
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("/v1/worker"):
                self.fail("public bearer worker route exposed")
        for invalid in ({"v": 99}, {"v": True}, {"payload": []}):
            with self.connect() as socket:
                socket.send_json({**hello(), **invalid})
                with self.assertRaises(WebSocketDisconnect):
                    socket.receive_json()

    def test_execution_events_are_acknowledged_and_do_not_complete_work(self):
        with self.connect() as socket:
            offer = self.start(socket)
            socket.send_json(frame("task.accept", self.ref(offer)))
            batch = {
                "taskId": offer["taskId"],
                "attempt": offer["attempt"],
                "events": [
                    {
                        "sequence": 1,
                        "at": "2026-09-19T12:00:00Z",
                        "kind": "stdout",
                        "data": {"text": "successful output"},
                    },
                ],
            }
            socket.send_json(frame("task.events", batch))
            ack = socket.receive_json()
            self.assertEqual(ack["type"], "task.events.ack")
            self.assertEqual(ack["payload"]["sequences"], [1])
            self.assertFalse(ack["payload"]["rejected"])
            self.assertEqual(self.store.tasks[0].state, "running")
            socket.send_json(frame("task.events", {**batch, "attempt": 2}))
            self.assertTrue(socket.receive_json()["payload"]["rejected"])
        self.assertEqual(len(self.store.execution_batches), 1)

    def test_unstorable_execution_events_never_close_the_task_connection(self):
        with self.connect() as socket:
            offer = self.start(socket)
            socket.send_json(frame("task.accept", self.ref(offer)))
            malformed = {
                "taskId": offer["taskId"],
                "attempt": offer["attempt"],
                "events": [{"sequence": 5000, "at": "2026-09-19T12:00:00", "kind": "stdout"}],
            }
            socket.send_json(frame("task.events", malformed))
            ack = socket.receive_json()
            self.assertEqual(ack["type"], "task.events.ack")
            self.assertEqual(
                ack["payload"],
                {
                    "taskId": offer["taskId"],
                    "attempt": offer["attempt"],
                    "sequences": [5000],
                    "rejected": True,
                },
            )
            self.assertEqual(self.store.execution_batches, [])

            async def busy(*args):
                raise TimeoutError("advisory lock timeout")

            self.store.append_execution_events = busy
            valid = {
                **malformed,
                "events": [
                    {"sequence": 1, "at": "2026-09-19T12:00:00Z", "kind": "stdout", "data": {}}
                ],
            }
            socket.send_json(frame("task.events", valid))
            # No acknowledgement: the device replays after its resend window.
            socket.send_json(frame("lease.renew", self.ref(offer)))
            self.send_result(socket, self.result(offer))
            self.assertEqual(socket.receive_json()["type"], "task.offer")
        self.assertEqual(len(self.store.finished), 1)


if __name__ == "__main__":
    unittest.main()
