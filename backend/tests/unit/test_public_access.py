"""Public browser sessions and isolation of the private worker listener."""

import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from orchestrator.server.app import create_public_app, create_worker_app
from orchestrator.server.auth import SESSION_COOKIE
from orchestrator.server.config import ServerConfig

ORIGIN = "https://fleet.example.com"
ADMIN = "unit-test-admin-credential-123456"
WORKER = "unit-test-worker-credential-12345"
USER_ID = "f19ed307-6827-4b38-9a22-fc291727dd97"
USER_TOKEN = "mock-supabase-user-token"


def configured(app):
    app.state.config = SimpleNamespace(
        admin_token=ADMIN,
        worker_tokens={"worker-1": WORKER},
        public_origin=ORIGIN,
        supabase_admin_ids=frozenset({USER_ID}),
        supabase_url="https://project.supabase.co",
        supabase_publishable_key="sb_publishable_test",
    )
    app.state.store = SimpleNamespace(workers=AsyncMock(return_value=[]))

    async def verify(token):
        if token == "unapproved-user":
            return {"sub": "other-user", "exp": int(time.time()) + 3600}
        if token != USER_TOKEN:
            raise HTTPException(status_code=401)
        return {"sub": USER_ID, "exp": int(time.time()) + 3600}

    app.state.supabase_auth = SimpleNamespace(verify=AsyncMock(side_effect=verify))
    app.state.cache = None
    return app


class PublicAccessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = configured(create_public_app())
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url=ORIGIN
        )
        self.addAsyncCleanup(self.client.aclose)

    async def login(self, token=USER_TOKEN, origin=ORIGIN):
        return await self.client.post(
            "/auth/session", headers={"Authorization": f"Bearer {token}", "Origin": origin}
        )

    async def test_anonymous_shell_does_not_grant_api_access(self):
        response = await self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("set-cookie", response.headers)
        for path in ("/auth/session", "/v1/workers", "/v1/updates"):
            self.assertEqual((await self.client.get(path)).status_code, 401)
        # Even DEMO_UI and proxy-supplied loopback/HTTPS headers cannot enable demo auth.
        with patch.dict(os.environ, {"DEMO_UI": "true"}):
            response = await self.client.get(
                "/demo/session",
                headers={
                    "Host": "localhost",
                    "X-Forwarded-For": "127.0.0.1",
                    "X-Forwarded-Proto": "https",
                },
            )
            self.assertEqual(response.status_code, 404)

    async def test_login_cookie_and_logout(self):
        self.assertEqual((await self.login("incorrect")).status_code, 401)
        self.assertEqual((await self.login(WORKER)).status_code, 401)
        response = await self.login()
        self.assertEqual(response.status_code, 200)
        cookie = response.headers["set-cookie"]
        for flag in ("Secure", "HttpOnly", "SameSite=strict", "Path=/", "Max-Age="):
            self.assertIn(flag, cookie)
        self.assertNotIn(ADMIN, cookie)
        self.assertEqual((await self.client.get("/auth/session")).json()["mode"], "public")
        self.assertEqual((await self.client.get("/v1/workers")).status_code, 200)
        self.assertEqual(
            (await self.client.delete("/auth/session", headers={"Origin": ORIGIN})).status_code, 200
        )
        self.assertEqual((await self.client.get("/v1/workers")).status_code, 401)

    async def test_cookie_mutations_require_exact_origin(self):
        await self.login()
        for headers in (
            {},
            {"Origin": "https://evil.example"},
            {"Origin": "http://fleet.example.com"},
        ):
            response = await self.client.post("/v1/tasks", json={"tasks": []}, headers=headers)
            self.assertEqual(response.status_code, 401)
        # An authorized request reaches body validation (rather than failing auth).
        response = await self.client.post(
            "/v1/tasks", json={"tasks": []}, headers={"Origin": ORIGIN}
        )
        self.assertEqual(response.status_code, 400)
        response = await self.client.get("/v1/workers", headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(response.status_code, 401)

    async def test_login_rejects_cross_origin_and_wrong_host(self):
        self.assertEqual((await self.login(origin="https://evil.example")).status_code, 403)
        response = await self.client.post(
            "/auth/session", headers={"Authorization": f"Bearer {ADMIN}"}
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            (await self.client.get("/", headers={"Host": "evil.example"})).status_code, 404
        )

    async def test_unapproved_user_cannot_get_session(self):
        response = await self.login("unapproved-user")
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("set-cookie", response.headers)
        self.client.cookies.set(SESSION_COOKIE, "expired-token")
        self.assertEqual((await self.client.get("/auth/session")).status_code, 401)

    async def test_only_publishable_config_is_exposed(self):
        response = await self.client.get("/auth/config")
        self.assertEqual(
            response.json(),
            {
                "url": "https://project.supabase.co",
                "publishableKey": "sb_publishable_test",
            },
        )
        self.assertNotIn(ADMIN, response.text)
        self.assertNotIn(WORKER, response.text)

    async def test_bearer_client_still_works(self):
        response = await self.client.get(
            "/v1/workers", headers={"Authorization": f"Bearer {ADMIN}"}
        )
        self.assertEqual(response.status_code, 200)


class ListenerTests(unittest.TestCase):
    def test_public_listener_rejects_websocket_even_with_valid_credentials(self):
        client = TestClient(configured(create_public_app()))
        with self.assertRaises(WebSocketDisconnect):
            with client.websocket_connect(
                "/v1/worker",
                headers={
                    "Authorization": f"Bearer {WORKER}",
                    "X-Worker-ID": "worker-1",
                },
            ):
                self.fail("Public listener accepted a worker")

    def test_private_listener_only_exposes_authenticated_worker_route(self):
        client = TestClient(configured(create_worker_app()))
        for path in ("/", "/auth/session", "/v1/workers", "/v1/tasks", "/v1/updates"):
            self.assertEqual(
                client.get(path, headers={"Authorization": f"Bearer {ADMIN}"}).status_code, 404
            )
        for token in ("wrong", ADMIN):
            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect(
                    "/v1/worker",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "X-Worker-ID": "worker-1",
                    },
                ):
                    self.fail("Private listener accepted the wrong credential")

        async def connected(socket, *_args):
            await socket.accept()
            await socket.send_json({"authenticated": True})
            await socket.close()

        with patch("orchestrator.server.app.serve_worker", side_effect=connected):
            with client.websocket_connect(
                "/v1/worker",
                headers={
                    "Authorization": f"Bearer {WORKER}",
                    "X-Worker-ID": "worker-1",
                },
            ) as socket:
                self.assertTrue(socket.receive_json()["authenticated"])

    def test_public_origin_validation(self):
        env = {
            "DATABASE_URL": "postgres://test",
            "ADMIN_TOKEN": ADMIN,
            "WORKER_TOKENS": '{"worker-1":"' + WORKER + '"}',
        }
        for origin in (
            "http://fleet.example.com",
            "https://a/path",
            "https://user:pw@a",
            "https://a?x=1",
        ):
            with patch.dict(os.environ, {**env, "PUBLIC_ORIGIN": origin}):
                with self.assertRaises(ValueError):
                    ServerConfig.from_env()


if __name__ == "__main__":
    unittest.main()
