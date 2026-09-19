"""Enrollment authorization, provider scope, failure cleanup, and CLI secret boundaries."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from orchestrator.server.app import create_public_app, create_worker_app
from orchestrator.server.db.store import Conflict, EnrollmentLimit
from orchestrator.server.enrollment import EnrollmentUnavailable, TailscaleEnrollment
from orchestrator.worker.setup import (
    SetupError,
    https_origin,
    login_and_enroll,
    setup,
    worker_environment,
)

USER = "f19ed307-6827-4b38-9a22-fc291727dd97"
ORIGIN = "https://fleet.example.com"


class EnrollmentAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = create_public_app()
        self.app.state.config = SimpleNamespace(
            admin_token="test-admin-token",
            supabase_admin_ids={USER},
            supabase_admin_emails=frozenset(),
            public_origin=ORIGIN,
            worker_gateway_url="wss://backend.example.ts.net:8443/v1/worker",
        )

        async def verify(token):
            if token == "expired":
                raise HTTPException(401)
            return {"sub": USER if token == "approved" else str(uuid4())}

        self.app.state.supabase_auth = SimpleNamespace(verify=AsyncMock(side_effect=verify))
        self.store = self.app.state.store = SimpleNamespace(
            reserve_enrollment=AsyncMock(), finish_enrollment=AsyncMock()
        )
        self.issuer = self.app.state.enrollment = SimpleNamespace(
            create=AsyncMock(return_value=("key-id", "tskey-auth-test-only")), revoke=AsyncMock()
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url=ORIGIN
        )
        self.addAsyncCleanup(self.client.aclose)

    async def enroll(self, token="approved", **kwargs):
        return await self.client.post(
            "/v1/worker-enrollments",
            headers={"Authorization": "Bearer " + token},
            json=kwargs or {"request_id": str(uuid4()), "name": "test computer"},
        )

    async def test_only_approved_users_can_enroll(self):
        for token, status in (("expired", 401), ("unapproved", 403), ("test-admin-token", 403)):
            self.assertEqual((await self.enroll(token)).status_code, status)
        self.issuer.create.assert_not_awaited()

    async def test_success_returns_credentials_once_with_no_cache(self):
        result = await self.enroll()
        self.assertEqual(result.status_code, 201)
        self.assertEqual(result.headers["cache-control"], "no-store")
        payload = result.json()
        self.assertEqual(payload["tailscale_auth_key"], "tskey-auth-test-only")
        self.assertGreaterEqual(len(payload["worker_token"]), 32)
        self.assertNotIn("client_secret", result.text)
        self.assertEqual(self.store.reserve_enrollment.await_args.args[2], USER)
        self.store.finish_enrollment.assert_awaited_once_with(payload["worker_id"], "key-id")

    async def test_replay_and_rate_limit_do_not_issue_keys(self):
        for error, status in ((Conflict("Already used"), 409), (EnrollmentLimit(), 429)):
            self.store.reserve_enrollment.side_effect = error
            self.assertEqual((await self.enroll()).status_code, status)
        self.issuer.create.assert_not_awaited()

    async def test_provider_failure_is_sanitized_and_pending_record_closed(self):
        self.issuer.create.side_effect = EnrollmentUnavailable("secret must not escape")
        result = await self.enroll()
        self.assertEqual(result.status_code, 502)
        self.assertNotIn("secret", result.text)
        self.assertIsNone(self.store.finish_enrollment.await_args.args[1])

    async def test_database_failure_revokes_unused_key(self):
        self.store.finish_enrollment.side_effect = [RuntimeError("db failed"), None]
        self.assertEqual((await self.enroll()).status_code, 502)
        self.issuer.revoke.assert_awaited_once_with("key-id")

    async def test_disabled_and_invalid_requests_never_enroll(self):
        self.assertEqual((await self.enroll(name="x", request_id="invalid")).status_code, 400)
        self.assertEqual(
            (await self.enroll(name="x" * 5000, request_id=str(uuid4()))).status_code, 413
        )
        self.app.state.enrollment = None
        self.assertEqual((await self.enroll()).status_code, 503)
        self.issuer.create.assert_not_awaited()

    async def test_cookie_enrollment_requires_same_origin(self):
        self.client.cookies.set("__Host-orchestrator_session", "approved")
        result = await self.client.post(
            "/v1/worker-enrollments",
            json={"name": "x", "request_id": str(uuid4())},
            headers={"Origin": "https://evil.example"},
        )
        self.assertEqual(result.status_code, 401)
        self.issuer.create.assert_not_awaited()

    async def test_worker_gateway_has_no_enrollment_route(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_worker_app()), base_url=ORIGIN
        ) as client:
            self.assertEqual((await client.post("/v1/worker-enrollments")).status_code, 404)


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_use_short_lived_tagged_key_and_revocation(self):
        calls = []

        def handle(request):
            calls.append(request)
            if request.url.path.endswith("/oauth/token"):
                self.assertIn(b"scope=auth_keys", request.content)
                return httpx.Response(200, json={"access_token": "oauth-test"})
            self.assertEqual(request.headers["authorization"], "Bearer oauth-test")
            if request.method == "DELETE":
                return httpx.Response(200, json={})
            d = json.loads(request.content)
            self.assertEqual(d["expirySeconds"], 600)
            self.assertEqual(
                d["capabilities"]["devices"]["create"],
                {
                    "reusable": False,
                    "ephemeral": False,
                    "preauthorized": True,
                    "tags": ["tag:htn-worker"],
                },
            )
            return httpx.Response(200, json={"id": "key-id", "key": "tskey-auth-test"})

        config = SimpleNamespace(
            tailscale_oauth_client_id="client",
            tailscale_oauth_client_secret="secret",
            tailscale_enrollment_tag="tag:htn-worker",
        )
        p = TailscaleEnrollment(config, transport=httpx.MockTransport(handle))
        try:
            key_id, key = await p.create()
            await p.revoke(key_id)
            self.assertEqual(key, "tskey-auth-test")
            self.assertEqual(len(calls), 4)
        finally:
            await p.close()


class SetupTests(unittest.IsolatedAsyncioTestCase):
    async def test_setup_saves_private_profile_after_join_without_secret_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "helper"
            helper.touch()
            output = Path(directory) / "worker.env"
            args = SimpleNamespace(
                server=ORIGIN,
                helper=str(helper),
                output=str(output),
                email="test@example.com",
                ca_file=None,
                name="Test",
                no_start=True,
            )
            data = {
                "worker_id": "worker-123",
                "worker_token": "test-worker-token-1234567890",
                "tailscale_auth_key": "tskey-auth-sensitive",
                "server_url": "wss://backend.example.ts.net:8443/v1/worker",
            }
            tunnel = AsyncMock()
            with (
                patch("orchestrator.worker.setup.getpass.getpass", return_value="private-password"),
                patch(
                    "orchestrator.worker.setup.login_and_enroll", new=AsyncMock(return_value=data)
                ),
                patch("orchestrator.worker.setup.EmbeddedTunnel", return_value=tunnel),
                patch.dict(os.environ, {}, clear=True),
            ):
                await setup(args)
                self.assertNotIn("TS_AUTHKEY", os.environ)
                with self.assertRaises(SetupError):
                    await setup(args)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            text = output.read_text()
            self.assertIn("WORKER_TOKEN=", text)
            for secret in ("tskey-auth-sensitive", "private-password", "TS_AUTHKEY"):
                self.assertNotIn(secret, text)

    async def test_password_only_goes_to_supabase_and_session_only_to_backend(self):
        def handle(request):
            if request.url.path == "/auth/config":
                return httpx.Response(
                    200,
                    json={
                        "url": "https://test.supabase.co",
                        "publishableKey": "sb_publishable_test",
                    },
                )
            if request.url.host == "test.supabase.co":
                self.assertEqual(request.headers["apikey"], "sb_publishable_test")
                self.assertEqual(json.loads(request.content)["password"], "private-password")
                return httpx.Response(
                    200, json={"access_token": "private-session", "refresh_token": "never-save"}
                )
            self.assertEqual(request.url.host, "fleet.example.com")
            self.assertEqual(request.headers["authorization"], "Bearer private-session")
            self.assertNotIn(b"private-password", request.content)
            return httpx.Response(201, json={"worker_id": "worker-123"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await login_and_enroll(
                client, ORIGIN, "test@example.com", "private-password", "Test", str(uuid4())
            )
            self.assertEqual(result, {"worker_id": "worker-123"})

    async def test_enrollment_redirect_is_not_followed(self):
        calls = []

        def handle(request):
            calls.append(str(request.url))
            return httpx.Response(302, headers={"location": "https://evil.example"})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle), follow_redirects=False
        ) as client:
            with self.assertRaises(SetupError):
                await login_and_enroll(client, ORIGIN, "a", "b", "c", str(uuid4()))
        self.assertEqual(len(calls), 1)

    async def test_profile_excludes_user_and_enrollment_secrets(self):

        data = {
            "worker_id": "worker-123",
            "worker_token": "test-worker-token-1234567890",
            "tailscale_auth_key": "tskey-auth-test",
            "server_url": "wss://backend.example.ts.net:8443/v1/worker",
            "access_token": "must-not-save",
        }
        env = worker_environment(data, Path("/tmp/profile"), "/tmp/helper")
        self.assertNotIn("TS_AUTHKEY", env)
        self.assertNotIn("must-not-save", str(env))
        for origin in (
            "http://localhost:5174",
            "https://user:pass@fleet.example",
            "https://fleet.example/path",
        ):
            with self.assertRaises(SetupError):
                https_origin(origin)


class DynamicWorkerTests(unittest.TestCase):
    def test_gateway_uses_database_enrollment_without_restart(self):
        app = create_worker_app()
        app.state.config = SimpleNamespace(worker_tokens={})
        app.state.store = SimpleNamespace(worker_authorized=AsyncMock(return_value=True))
        app.state.cache = None

        async def connected(socket, *args):
            await socket.accept()
            await socket.send_json({"authenticated": True})
            await socket.close()

        with patch("orchestrator.server.app.serve_worker", side_effect=connected):
            with TestClient(app).websocket_connect(
                "/v1/worker",
                headers={"X-Worker-ID": "worker-new", "Authorization": "Bearer test-worker-secret"},
            ) as socket:
                self.assertTrue(socket.receive_json()["authenticated"])
        app.state.store.worker_authorized.assert_awaited_once_with(
            "worker-new", "Bearer test-worker-secret"
        )
