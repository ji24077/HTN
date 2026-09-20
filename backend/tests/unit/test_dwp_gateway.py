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
        self.declined = []
        self.disconnected = []
        self.paused = False
        self.execution_batches = []
        self.preference = "auto"
        self.runtime_acks = []

    async def append_execution_events(self, worker_id, session, batch):
        self.assert_session(session)
        current = await self.task(batch.taskId)
        if current.worker_id != worker_id or current.generation != batch.attempt:
            raise StaleAssignment("not assigned")
        self.execution_batches.append(batch)
        return [item.sequence for item in batch.events]

    async def create_pair_code(self, owner_id, *, max_per_hour=10, max_devices=100):
        self.owner = owner_id
        # Recorded so a test can assert the route passes the operator's ceilings through
        # rather than letting the store's defaults quietly stand in for them.
        self.pair_code_limits = (max_per_hour, max_devices)
        code = uuid4().hex.upper()
        self.codes.add(code)
        return code

    async def runtime_preference(self, worker_id):
        return self.preference

    async def set_runtime_preference(self, worker_id, preference):
        self.preference = preference
        return True

    async def record_runtime_ack(
        self, worker_id, preference, applied, detail, *, runtime=None, vram_mib=None
    ):
        self.runtime_acks.append((preference, applied, detail, runtime))
        self.preference = preference

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
        self.finished.append((ref, result, failure, attestation, retryable))

    async def decline(self, worker_id, session, ref):
        self.assert_session(session)
        current = await self.task(ref.task_id)
        if task_ref(current) != ref or current.state != "assigned":
            raise StaleAssignment("only unaccepted offers can be declined")
        current.state = "queued"
        current.worker_id = current.session_id = None
        current.lease_until = current.deadline = None
        self.declined.append(ref)

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
            self_serve_join=False,
            self_serve_max_per_hour=10,
            self_serve_max_devices=100,
        )
        self.app.state.store = self.store
        self.app.state.device_connections = {}

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

    def result(self, offer, output=None):
        output = {"nonce": 1, "value": 0.0000001, "label": "東京"} if output is None else output
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

    def test_python_capability_survives_device_handshake(self):
        report = {"version": "3.12.14", "pytorch": "2.13.0+cpu"}
        message = hello()
        message["payload"]["capability"]["python"] = report
        message["payload"]["capability"]["adapters"].extend(["python_project", "python_program"])
        with self.connect() as socket:
            socket.send_json(message)
            self.assertEqual(socket.receive_json()["type"], "hello.ack")
            self.assertEqual(self.store.registered[0].python.model_dump(), report)
            self.assertIn("python_program", self.store.registered[0].kinds)

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

    def test_self_serve_invites_are_absent_until_enabled_then_mint_unowned_codes(self):
        """Off, the endpoint does not exist; on, it mints a code nobody had to approve.

        The two assertions that matter are about what does *not* change. A self-serve
        code is owned by nobody, so it draws on the unowned quota rather than a person's,
        and it is redeemed by the same `/hosts/pair` with the same single-use rule as a
        code an admin minted. If either stopped holding, this endpoint would be a way to
        get a *better* invite than the admin path hands out.
        """
        self.assertEqual(self.client.post("/v1/join-requests").status_code, 404)
        self.app.state.config.self_serve_join = True
        issued = self.client.post("/v1/join-requests")
        self.assertEqual(issued.status_code, 201)
        self.assertEqual(issued.headers["cache-control"], "no-store")
        self.assertEqual(issued.json()["server"], ORIGIN)
        self.assertIsNone(self.store.owner)
        # The ceilings are the operator's, not the store's defaults. Ten an hour is right
        # for a link left on the internet and wrong for a room at an event, where the
        # eleventh person to ask is told to come back in an hour; the route is the only
        # place that knows which of the two this deployment is.
        self.app.state.config.self_serve_max_per_hour = 250
        self.app.state.config.self_serve_max_devices = 400
        self.assertEqual(self.client.post("/v1/join-requests").status_code, 201)
        self.assertEqual(self.store.pair_code_limits, (250, 400))
        body = {
            "code": issued.json()["code"].lower(),
            "publicKey": self.public,
            "label": "Windows laptop",
        }
        paired = self.client.post("/hosts/pair", json=body)
        self.assertEqual(paired.status_code, 200)
        self.assertEqual(paired.json()["label"], "Windows laptop")
        self.assertEqual(self.client.post("/hosts/pair", json=body).status_code, 400)

    def test_self_serve_invite_limit_tells_the_reader_what_to_do(self):
        """A volunteer reads this, not an operator: it has to say what to do next."""
        from orchestrator.server.db.store import EnrollmentLimit

        self.app.state.config.self_serve_join = True
        with patch.object(self.store, "create_pair_code", AsyncMock(side_effect=EnrollmentLimit)):
            response = self.client.post("/v1/join-requests")
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "3600")
        self.assertIn("Try again", response.json()["detail"])

    def test_runtime_toggle_is_stored_then_pushed_and_the_device_answer_wins(self):
        """Stored first, pushed second, and the machine's answer overwrites the request.

        The order matters more than the push: a machine that is offline when the switch
        is flipped still has to arrive at the setting, which it does from storage at its
        next hello. And a machine that refuses -- no device to switch to -- must leave
        the record saying 'cpu', not the 'auto' that was asked for, or the dashboard
        shows a GPU setting on a machine that just explained it has no GPU.
        """
        with self.connect() as socket:
            self.start(socket)
            pushed = self.client.post(
                f"/v1/machines/{self.store.worker_id}/runtime",
                json={"runtimePreference": "cpu"},
                headers={"Authorization": "Bearer " + ADMIN},
            )
            self.assertEqual(pushed.status_code, 200)
            self.assertTrue(pushed.json()["delivered"])
            self.assertEqual(self.store.preference, "cpu")
            update = socket.receive_json()
            self.assertEqual(update["type"], "settings.update")
            self.assertEqual(update["payload"]["runtimePreference"], "cpu")

            # The device refuses `auto`: it has no device to switch to.
            socket.send_json(
                frame(
                    "settings.ack",
                    {
                        "runtimePreference": "cpu",
                        "applied": False,
                        "detail": "no inference runtime in this image",
                        "accelerator": {
                            "runtime": "cpu",
                            "vramMib": 0,
                            "available": False,
                            "reason": "no inference runtime in this image",
                            "device": None,
                            "providers": ["cuda", "cpu"],
                        },
                    },
                )
            )
            # Frames are handled in order, so a request that always answers is a
            # barrier: once task.cancel arrives, settings.ack has already been processed.
            socket.send_json(frame("lease.renew", {"taskId": "unknown", "leaseId": "wrong"}))
            self.assertEqual(socket.receive_json()["type"], "task.cancel")
        self.assertEqual(self.store.runtime_acks[-1][0], "cpu")
        self.assertFalse(self.store.runtime_acks[-1][1])

    def test_runtime_toggle_survives_a_machine_being_offline(self):
        """An unreachable machine is still set; it collects the value at its next hello."""
        response = self.client.post(
            f"/v1/machines/{self.store.worker_id}/runtime",
            json={"runtimePreference": "cpu"},
            headers={"Authorization": "Bearer " + ADMIN},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["delivered"])
        self.assertEqual(self.store.preference, "cpu")
        self.assertEqual(
            self.client.post(
                f"/v1/machines/{self.store.worker_id}/runtime",
                json={"runtimePreference": "gpu"},
                headers={"Authorization": "Bearer " + ADMIN},
            ).status_code,
            400,
        )
        self.assertEqual(
            self.client.post(
                f"/v1/machines/{self.store.worker_id}/runtime",
                json={"runtimePreference": "cpu"},
            ).status_code,
            401,
        )

    def test_reported_accelerator_is_recorded_rather_than_assumed(self):
        """A machine that reports a GPU is stored as having one; silence is not a denial."""
        with self.connect() as socket:
            socket.send_json(
                frame(
                    "hello",
                    {
                        "capability": {
                            "adapters": ["echo"],
                            "agentVersion": "test",
                            "os": "linux",
                            "arch": "x64",
                            "accelerator": {
                                "runtime": "cuda",
                                "vramMib": 8192,
                                "available": True,
                                "reason": "",
                                "device": "NVIDIA GeForce RTX 4060",
                                "providers": ["cuda", "cpu"],
                            },
                            "runtimePreference": "auto",
                        },
                        "consent": {
                            "paused": False,
                            "allowCompute": True,
                            "allowBrowser": False,
                            "maxConcurrency": 8,
                        },
                    },
                )
            )
            ack = socket.receive_json()
            self.assertEqual(ack["type"], "hello.ack")
        recorded = self.store.registered[-1]
        self.assertEqual(recorded.runtime, "cuda")
        self.assertEqual(recorded.vram_mib, 8192)
        self.assertTrue(recorded.accelerator.available)
        self.assertEqual(recorded.accelerator.device, "NVIDIA GeForce RTX 4060")

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
        _, output, failure, proof, _ = self.store.finished[0]
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

    def test_oversized_result_fails_one_task_and_keeps_the_connection(self):
        """A slice too big to report must cost that slice, not the host.

        Refusing it by closing the socket is what made a fleet flap: the worker was
        marked unhealthy, everything it held was cancelled as `worker_reconnected`, and
        the slice was re-queued onto the next machine to be refused there too.
        """
        big = {"results": ["x" * 512 for _ in range(200)]}
        with self.connect() as socket:
            offer = self.start(socket)
            socket.send_json(frame("task.accept", self.ref(offer)))
            self.send_result(socket, self.result(offer, output=big))
            # Still talking, and already being given the next piece of work.
            self.assertEqual(socket.receive_json()["type"], "task.cancel")
            following = socket.receive_json()
            self.assertEqual(following["type"], "task.offer")
            self.assertEqual(following["payload"]["taskId"], "task-2")
        ref, stored, failure, proof, retryable = self.store.finished[0]
        self.assertEqual(ref.task_id, "task-1")
        self.assertEqual(failure, "result_too_large")
        self.assertIsNone(stored)
        self.assertIsNone(proof)
        # Deterministic adapters reproduce the same oversized output everywhere, so a
        # retry only moves the failure to the next machine.
        self.assertFalse(retryable)

    def test_device_reporting_result_too_large_is_not_retried(self):
        """An agent that measures its own output first gets the same verdict."""
        with patch("sentry_sdk.capture_message") as capture:
            with self.connect() as socket:
                offer = self.start(socket)
                socket.send_json(frame("task.accept", self.ref(offer)))
                socket.send_json(
                    frame(
                        "task.error",
                        {**self.ref(offer), "errorClass": "result_too_large", "message": "too big"},
                    )
                )
                self.assertEqual(socket.receive_json()["type"], "task.offer")
            self.assertEqual(self.store.finished[0][2], "result_too_large")
            self.assertFalse(self.store.finished[0][4])
            # Nothing is broken; this is a job that was split too coarsely.
            capture.assert_not_called()

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

    def test_declines_back_off_without_starting_work_and_keep_generation_fencing(self):
        with self.connect() as socket:
            offer = self.start(socket)
            connection = self.app.state.device_connections[self.store.worker_id]
            for number in range(12):
                generation = offer["attempt"]
                socket.send_json(frame("task.decline", {**self.ref(offer), "reason": "schedule"}))
                # Heartbeats must not cause another offer during cooldown. The
                # stale-lease response orders all previous frames without sleeping.
                for _ in range(3):
                    socket.send_json(frame("heartbeat", {"freeRamMb": 200, "running": 0}))
                socket.send_json(frame("lease.renew", {"taskId": "barrier", "leaseId": "none"}))
                self.assertEqual(socket.receive_json()["type"], "task.cancel")
                self.assertEqual(self.store.tasks[0].state, "queued")
                self.assertEqual(self.store.tasks[0].generation, generation)
                self.assertEqual(len(self.store.declined), number + 1)
                self.assertEqual(self.store.finished, [])
                remaining = connection.offer_after - time.monotonic()
                self.assertGreater(remaining, 0)
                self.assertLessEqual(remaining, min(300, 30 * 2 ** number) + 0.000001)
                self.assertLessEqual(connection.decline_delay, 300)
                # Advance only this connection's cooldown, not the event-loop clock.
                connection.offer_after = 0
                socket.send_json(frame("heartbeat", {"freeRamMb": 200, "running": 0}))
                next_offer = socket.receive_json()["payload"]
                self.assertEqual(next_offer["attempt"], generation + 1)
                self.assertNotEqual(next_offer["leaseId"], offer["leaseId"])
                offer = next_offer
            socket.send_json(frame("task.accept", self.ref(offer)))
            self.send_result(socket, self.result(offer))
            self.assertEqual(socket.receive_json()["payload"]["taskId"], "task-2")
            self.assertEqual(connection.decline_delay, 30)
        self.assertEqual(len(self.store.finished), 1)

    def test_declining_already_started_work_still_counts_as_execution_failure(self):
        with self.connect() as socket:
            offer = self.start(socket)
            socket.send_json(frame("task.accept", self.ref(offer)))
            socket.send_json(frame("task.decline", {**self.ref(offer), "reason": "agent-stopping"}))
            self.assertEqual(socket.receive_json()["type"], "task.offer")
        self.assertEqual(self.store.declined, [])
        self.assertEqual(self.store.finished[0][2], "device_declined")

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
