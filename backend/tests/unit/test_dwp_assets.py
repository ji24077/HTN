"""Device downloads stay bounded to published files and authenticated artifacts."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from orchestrator.server.dwp_assets import release_info, router


class DeviceAssetsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        env = patch.dict(
            os.environ,
            {
                "DWP_RELEASES_DIR": str(self.root),
                "DWP_FIXTURES_DIR": str(self.root),
            },
        )
        env.start()
        self.addCleanup(env.stop)
        self.app = FastAPI()
        self.app.state.config = SimpleNamespace(admin_token="test-admin", public_origin="")
        self.app.state.store = SimpleNamespace()
        self.client = TestClient(self.app)
        self.app.include_router(router)

    def test_no_release_does_not_pin_a_key(self):
        self.assertEqual(release_info(), {"releaseKey": None, "releaseVersion": None})
        self.assertEqual(self.client.get("/release/latest").status_code, 404)

    def test_only_files_in_binary_index_are_downloadable(self):
        directory = self.root / "binaries"
        directory.mkdir()
        (directory / "agent.exe").write_bytes(b"fixture")
        (directory / "private.key").write_bytes(b"secret")
        (directory / "index.json").write_text(
            json.dumps(
                {
                    "manifest": {"version": "test"},
                    "signature": "signature",
                    "publicKey": "key",
                    "binaries": [{"target": "win32-x64", "file": "agent.exe"}],
                    "apps": [],
                }
            )
        )
        self.assertEqual(self.client.get("/download/agent.exe").content, b"fixture")
        self.assertEqual(self.client.get("/download/private.key").status_code, 404)
        self.assertEqual(self.client.get("/download/..%5Cprivate.key").status_code, 404)

    def test_app_downloads_are_served_from_the_apps_directory(self):
        """The index lists apps and binaries together; they are not stored together.

        Serving both out of "binaries" made every zip on the /join page a link that
        rendered and then 404ed, while the raw executables beside them worked -- so it
        read as a broken download rather than a server looking in the wrong directory.
        """
        binaries = self.root / "binaries"
        apps = self.root / "apps"
        binaries.mkdir()
        apps.mkdir()
        (binaries / "agent.exe").write_bytes(b"executable")
        (apps / "Agent-Windows.zip").write_bytes(b"zipped app")
        # An app name that exists only under binaries must still not be reachable.
        (binaries / "Decoy.zip").write_bytes(b"not published")
        (binaries / "index.json").write_text(
            json.dumps(
                {
                    "manifest": {"version": "test"},
                    "signature": "signature",
                    "publicKey": "key",
                    "binaries": [{"target": "win32-x64", "file": "agent.exe"}],
                    "apps": [{"target": "windows-x64", "file": "Agent-Windows.zip"}],
                }
            )
        )
        self.assertEqual(
            self.client.get("/download/Agent-Windows.zip").content, b"zipped app"
        )
        self.assertEqual(self.client.get("/download/agent.exe").content, b"executable")
        self.assertEqual(self.client.get("/download/Decoy.zip").status_code, 404)

    def test_artifact_requires_verified_device_assertion(self):
        content = b"model fixture"
        digest = hashlib.sha256(content).hexdigest()
        (self.root / digest).write_bytes(content)
        with patch(
            "orchestrator.server.dwp.authenticate_device", AsyncMock(side_effect=ValueError)
        ):
            self.assertEqual(self.client.get(f"/artifacts/{digest}").status_code, 401)
        with patch("orchestrator.server.dwp.authenticate_device", AsyncMock(return_value="device")):
            self.assertEqual(self.client.get(f"/artifacts/{digest}").content, content)
            self.assertEqual(self.client.get("/artifacts/not-a-hash").status_code, 404)
            (self.root / digest).write_bytes(b"corrupt bytes")
            self.assertEqual(self.client.get(f"/artifacts/{digest}").status_code, 503)

    def test_join_does_not_reflect_or_leak_invite_in_html(self):
        response = self.client.get("/join?code=PRIVATE-CODE")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("PRIVATE-CODE", response.text)
        self.assertEqual(response.headers["referrer-policy"], "no-referrer")
