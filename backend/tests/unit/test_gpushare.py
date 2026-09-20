"""The GPU panel reads gpushare through a fixed set of paths, or says why it cannot."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orchestrator.gpushare.client import GpushareClient, GpushareUnavailable
from orchestrator.gpushare.routes import router
from orchestrator.server.config import ServerConfig

AUTH = {"authorization": "Bearer test-admin"}


def _app(gpushare) -> TestClient:
    app = FastAPI()
    app.state.config = SimpleNamespace(admin_token="test-admin", public_origin="")
    app.state.gpushare = gpushare
    app.include_router(router)
    return TestClient(app)


class RouteTests(unittest.TestCase):
    def test_unconfigured_reports_disabled_rather_than_failing(self):
        client = _app(None)
        self.assertEqual(client.get("/v1/gpushare/config", headers=AUTH).json(), {"enabled": False})

    def test_unconfigured_reads_are_503(self):
        client = _app(None)
        self.assertEqual(client.get("/v1/gpushare/state", headers=AUTH).status_code, 503)

    def test_reads_pass_the_service_payload_through(self):
        served = {"pods": [{"id": "abc", "gpu": "RTX 3090"}]}
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=served))
        client = _app(GpushareClient("http://127.0.0.1:8080", transport=transport))
        response = client.get("/v1/gpushare/pods", headers=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), served)

    def test_unreachable_service_is_502_not_503(self):
        # 503 would claim this server is unconfigured; the operator needs to
        # know the configured service is the thing that did not answer.
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        client = _app(
            GpushareClient("http://127.0.0.1:8080", transport=httpx.MockTransport(refuse))
        )
        response = client.get("/v1/gpushare/state", headers=AUTH)
        self.assertEqual(response.status_code, 502)
        self.assertIn("did not answer", response.json()["detail"])

    def test_reads_require_admin(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        client = _app(GpushareClient("http://127.0.0.1:8080", transport=transport))
        self.assertEqual(client.get("/v1/gpushare/state").status_code, 401)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_allowlisted_resources_are_reachable(self):
        """The client is not an open proxy: an unknown name never becomes a request."""
        asked: list[str] = []

        def record(request: httpx.Request) -> httpx.Response:
            asked.append(request.url.path)
            return httpx.Response(200, json={})

        client = GpushareClient("http://127.0.0.1:8080", transport=httpx.MockTransport(record))
        with self.assertRaises(ValueError):
            await client.read("../../etc/passwd")
        with self.assertRaises(ValueError):
            await client.read("serve")  # a real gpushare route, but a write one
        self.assertEqual(asked, [])
        await client.aclose()

    async def test_a_non_object_body_is_unavailable_not_a_type_error(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[1, 2]))
        client = GpushareClient("http://127.0.0.1:8080", transport=transport)
        with self.assertRaises(GpushareUnavailable):
            await client.read("jobs")
        await client.aclose()


class ConfigTests(unittest.TestCase):
    def _config(self, url: str) -> ServerConfig:
        with patch.dict(
            os.environ, {"DATABASE_URL": "postgres://localhost/x", "GPUSHARE_URL": url}
        ):
            return ServerConfig.from_env()

    def test_localhost_http_is_allowed(self):
        self.assertEqual(
            self._config("http://127.0.0.1:8080/").gpushare_url, "http://127.0.0.1:8080"
        )

    def test_remote_http_is_refused(self):
        with self.assertRaises(ValueError):
            self._config("http://gpu.example.com")

    def test_a_path_is_refused(self):
        with self.assertRaises(ValueError):
            self._config("https://gpu.example.com/api")

    def test_unset_leaves_the_panel_off(self):
        self.assertEqual(self._config("").gpushare_url, "")


if __name__ == "__main__":
    unittest.main()
