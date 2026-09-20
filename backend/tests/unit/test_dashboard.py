"""Check packaged UI routes without starting PostgreSQL or worker processes."""

import os
import re
import unittest
from importlib.resources import files
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from orchestrator.server.app import create_app


class DashboardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        env = patch.dict(os.environ, {"DEMO_UI": "true"})
        env.start()
        self.addCleanup(env.stop)
        html = files("orchestrator.server").joinpath("web", "index.html").read_text()
        self.assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', html)
        self.script = next(path for path in self.assets if path.endswith(".js"))
        self.app = create_app()
        self.app.state.ui_session = "test-ui-session"
        self.app.state.config = SimpleNamespace(admin_token="test-admin-token")
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app, client=("127.0.0.1", 12345)),
            base_url="http://127.0.0.1",
        )
        self.addAsyncCleanup(self.client.aclose)

    async def test_dashboard_loads_its_separate_assets(self):
        page = await self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn('id="root"', page.text)
        self.assertEqual(len(self.assets), 2)
        for path in self.assets:
            self.assertIn(path, page.text)
            content_type = "text/css" if path.endswith(".css") else "application/javascript"
            response = await self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith(content_type))
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertTrue(response.text)

    async def test_assets_require_auth_and_reject_cross_site_requests(self):
        self.assertEqual((await self.client.get(self.script)).status_code, 401)
        await self.client.get("/")
        response = await self.client.get(self.script, headers={"Origin": "https://other.example"})
        self.assertEqual(response.status_code, 401)

    async def test_assets_are_allowlisted_and_hidden_outside_demo(self):
        await self.client.get("/")
        self.assertEqual((await self.client.get("/assets/schema.sql")).status_code, 404)
        with patch.dict(os.environ, {"DEMO_UI": "false"}):
            self.assertEqual((await self.client.get("/")).status_code, 404)
            response = await self.client.get(
                self.script, headers={"Authorization": "Bearer test-admin-token"}
            )
            self.assertEqual(response.status_code, 404)

    async def test_local_session_bootstrap_and_proxy_host(self):
        response = await self.client.get("/demo/session")
        self.assertEqual(response.status_code, 200)
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        response = await self.client.get(
            "/demo/session", headers={"Host": "127.0.0.1:5173", "Origin": "http://127.0.0.1:5173"}
        )
        self.assertEqual(response.status_code, 200)
        response = await self.client.get(
            "/demo/session", headers={"Origin": "https://other.example"}
        )
        self.assertEqual(response.status_code, 403)

    async def test_remote_clients_cannot_open_demo(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app, client=("192.0.2.1", 12345)),
            base_url="http://127.0.0.1",
        ) as remote:
            self.assertEqual((await remote.get("/")).status_code, 404)
            self.assertEqual((await remote.get("/demo/session")).status_code, 404)


if __name__ == "__main__":
    unittest.main()
