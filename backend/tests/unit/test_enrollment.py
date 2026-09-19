"""Enrollment authorization, provider scope, failure cleanup, and CLI secret boundaries."""

import asyncio
import contextlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

import orchestrator
from orchestrator.server.app import create_public_app, create_worker_app
from orchestrator.server.db.store import Conflict, EnrollmentLimit, NotFound
from orchestrator.server.enrollment import EnrollmentUnavailable, TailscaleEnrollment
from orchestrator.worker.setup import (
    SetupError,
    discard_profile,
    dotenv_quote,
    enrollment_error,
    finish_cleanup,
    https_origin,
    login_and_enroll,
    reserve_profile,
    setup,
    withdraw_enrollment,
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
            reserve_enrollment=AsyncMock(),
            finish_enrollment=AsyncMock(return_value=True),
            cancel_enrollment=AsyncMock(return_value="key-id"),
            clear_enrollment_key=AsyncMock(),
        )
        self.issuer = self.app.state.enrollment = SimpleNamespace(
            create=AsyncMock(return_value=("key-id", "tskey-auth-test-only", "oauth-test")),
            revoke=AsyncMock(return_value=True),
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

    async def withdraw(self, worker="worker-" + "a" * 20, token="approved"):
        return await self.client.delete(
            "/v1/worker-enrollments/" + worker, headers={"Authorization": "Bearer " + token}
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
        self.assertNotIn("oauth-test", result.text)
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
        self.issuer.revoke.assert_awaited_once_with("key-id", "oauth-test")
        self.assertEqual(self.store.finish_enrollment.await_args.kwargs, {"retain_key": None})

    async def test_unrevoked_key_is_retained_for_later_withdrawal(self):
        self.store.finish_enrollment.side_effect = [RuntimeError("db failed"), None]
        self.issuer.revoke.return_value = False
        self.assertEqual((await self.enroll()).status_code, 502)
        self.assertEqual(self.store.finish_enrollment.await_args.kwargs, {"retain_key": "key-id"})

    async def test_expired_reservation_is_never_activated(self):
        self.store.finish_enrollment.return_value = False
        self.assertEqual((await self.enroll()).status_code, 502)
        self.issuer.revoke.assert_awaited_once_with("key-id", "oauth-test")

    async def test_cleanup_failure_still_reports_provisioning_failure(self):
        self.store.finish_enrollment.side_effect = TimeoutError()
        result = await self.enroll()
        self.assertEqual(result.status_code, 502)
        self.assertNotIn("database", result.text)
        self.assertEqual(self.store.finish_enrollment.await_count, 2)
        self.issuer.revoke.assert_awaited_once()

    async def test_slow_provider_hits_provisioning_deadline(self):
        async def slow():
            await asyncio.sleep(5)

        self.issuer.create.side_effect = slow
        with patch("orchestrator.server.enrollment.PROVISIONING_SECONDS", 0.05):
            self.assertEqual((await self.enroll()).status_code, 502)
        self.assertIsNone(self.store.finish_enrollment.await_args.args[1])

    async def test_withdraw_releases_unused_enrollment_and_revokes_key(self):
        self.assertEqual((await self.withdraw()).status_code, 204)
        self.store.cancel_enrollment.assert_awaited_once_with("worker-" + "a" * 20, USER)
        self.issuer.revoke.assert_awaited_once_with("key-id")
        self.store.clear_enrollment_key.assert_awaited_once_with("worker-" + "a" * 20)
        self.store.cancel_enrollment.return_value = None
        self.assertEqual((await self.withdraw()).status_code, 204)
        self.issuer.revoke.assert_awaited_once()

    async def test_repeated_withdraw_retries_a_failed_revocation(self):
        self.issuer.revoke.return_value = False
        self.assertEqual((await self.withdraw()).status_code, 204)
        self.store.clear_enrollment_key.assert_not_awaited()
        self.issuer.revoke.return_value = True
        self.assertEqual((await self.withdraw()).status_code, 204)
        self.assertEqual(self.issuer.revoke.await_count, 2)
        self.store.clear_enrollment_key.assert_awaited_once()

    async def test_withdraw_is_owner_checked_and_refuses_connected_workers(self):
        for error, status in ((NotFound("missing"), 404), (Conflict("connected"), 409)):
            self.store.cancel_enrollment.side_effect = error
            self.assertEqual((await self.withdraw()).status_code, status)
        self.store.cancel_enrollment.side_effect = None
        for token in ("unapproved", "test-admin-token"):
            self.assertEqual((await self.withdraw(token=token)).status_code, 403)
        self.assertEqual((await self.withdraw(worker="worker-not-hex")).status_code, 404)
        self.assertEqual(self.store.cancel_enrollment.await_count, 2)
        self.issuer.revoke.assert_not_awaited()

    async def test_disabled_and_invalid_requests_never_enroll(self):
        self.assertEqual((await self.enroll(name="x", request_id="invalid")).status_code, 400)
        self.assertEqual(
            (await self.enroll(name="x" * 5000, request_id=str(uuid4()))).status_code, 413
        )
        self.app.state.enrollment = None
        result = await self.enroll()
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.json()["detail"]["code"], "enrollment_not_configured")
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
            key_id, key, token = await p.create()
            await p.revoke(key_id, token)
            self.assertEqual(key, "tskey-auth-test")
            self.assertEqual(len(calls), 3)  # Cleanup reuses the token that created the key.
            await p.revoke(key_id)
            self.assertEqual(len(calls), 5)
        finally:
            await p.close()


class SetupTests(unittest.IsolatedAsyncioTestCase):
    def assert_profile_permissions(self, output):
        if os.name == "posix":
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        else:
            # Windows inherits the containing directory's ACL; stat() mode bits
            # cannot describe its owner/group access policy.
            self.assertTrue(os.access(output, os.R_OK | os.W_OK))

    def test_profile_reservation_never_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "worker.env"
            output.write_text("existing identity")
            with self.assertRaises(OSError):
                reserve_profile(output)
            self.assertEqual(output.read_text(), "existing identity")

    def test_discard_reserved_profile_removes_it_after_handle_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "worker.env"
            fd = reserve_profile(output)
            try:
                os.write(fd, b"unfinished profile")
                discard_profile(fd, output)
            finally:
                os.close(fd)
            self.assertFalse(output.exists())

    def test_successful_reservation_keeps_complete_profile_after_close(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "worker.env"
            fd = reserve_profile(output)
            try:
                os.write(fd, b"WORKER_ID=worker-test\n")
            finally:
                os.close(fd)
            self.assertEqual(output.read_text(), "WORKER_ID=worker-test\n")
            self.assert_profile_permissions(output)

    @unittest.skipUnless(os.name == "nt", "Windows reserves file deletion by handle")
    def test_windows_reservation_blocks_path_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "worker.env"
            replacement = Path(directory) / "replacement.env"
            replacement.write_text("unrelated identity")
            fd = reserve_profile(output)
            try:
                with self.assertRaises(PermissionError):
                    output.unlink()
                with self.assertRaises(PermissionError):
                    os.replace(replacement, output)
                self.assertEqual(replacement.read_text(), "unrelated identity")
                discard_profile(fd, output)
            finally:
                os.close(fd)
            self.assertFalse(output.exists())
            self.assertEqual(replacement.read_text(), "unrelated identity")

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
                    "orchestrator.worker.setup.login_and_enroll",
                    new=AsyncMock(return_value=(data, "private-session")),
                ),
                patch("orchestrator.worker.setup.EmbeddedTunnel", return_value=tunnel),
                patch.dict(os.environ, {}, clear=True),
            ):
                await setup(args)
                self.assertNotIn("TS_AUTHKEY", os.environ)
                with self.assertRaises(SetupError):
                    await setup(args)
            self.assert_profile_permissions(output)
            text = output.read_text()
            self.assertIn("WORKER_TOKEN=", text)
            for secret in (
                "tskey-auth-sensitive",
                "private-password",
                "TS_AUTHKEY",
                "private-session",
            ):
                self.assertNotIn(secret, text)

    async def test_setup_withdraws_enrollment_when_join_fails(self):
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
            tunnel.__aenter__.side_effect = RuntimeError("Tailscale helper exited")
            withdraw = AsyncMock(return_value=True)
            with (
                patch("orchestrator.worker.setup.getpass.getpass", return_value="private-password"),
                patch(
                    "orchestrator.worker.setup.login_and_enroll",
                    new=AsyncMock(return_value=(data, "private-session")),
                ),
                patch("orchestrator.worker.setup.EmbeddedTunnel", return_value=tunnel),
                patch("orchestrator.worker.setup.withdraw_enrollment", withdraw),
                patch.dict(os.environ, {}, clear=True),
            ):
                with self.assertRaises(RuntimeError):
                    await setup(args)
                self.assertNotIn("TS_AUTHKEY", os.environ)
            withdraw.assert_awaited_once()
            self.assertEqual(
                withdraw.await_args.args[1:], (ORIGIN, "private-session", "worker-123")
            )
            self.assertFalse(output.exists())

    async def test_setup_keeps_profile_when_withdrawal_fails(self):
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
            tunnel.__aenter__.side_effect = RuntimeError("Tailscale helper exited")
            with (
                patch("orchestrator.worker.setup.getpass.getpass", return_value="private-password"),
                patch(
                    "orchestrator.worker.setup.login_and_enroll",
                    new=AsyncMock(return_value=(data, "private-session")),
                ),
                patch("orchestrator.worker.setup.EmbeddedTunnel", return_value=tunnel),
                patch(
                    "orchestrator.worker.setup.withdraw_enrollment",
                    new=AsyncMock(return_value=False),
                ),
                patch.dict(os.environ, {}, clear=True),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
                self.assertRaises(RuntimeError),
            ):
                await setup(args)
            self.assertIn("Kept", stderr.getvalue())
            self.assert_profile_permissions(output)
            text = output.read_text()
            self.assertIn("WORKER_TOKEN=", text)
            for secret in ("tskey-auth-sensitive", "private-password", "private-session"):
                self.assertNotIn(secret, text)

    async def test_setup_rejects_unrepresentable_paths_before_enrolling(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "hel\nper"
            if os.name != "nt":
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
            login = AsyncMock()
            with (
                patch("orchestrator.worker.setup.login_and_enroll", new=login),
                self.assertRaises(SetupError),
            ):
                await setup(args)
            login.assert_not_awaited()
            self.assertFalse(output.exists())

    async def test_cleanup_survives_a_second_interrupt_and_stays_bounded(self):
        async def slow_release():
            await asyncio.sleep(0.05)
            return True

        task = asyncio.ensure_future(finish_cleanup(slow_release()))
        await asyncio.sleep(0.01)
        task.cancel()
        self.assertTrue(await task)

        async def never():
            await asyncio.sleep(10)

        self.assertFalse(await finish_cleanup(never(), seconds=0.05))
        await asyncio.sleep(0)

    @unittest.skipIf(sys.platform == "win32", "POSIX signal delivery")
    def test_cleanup_survives_a_real_second_ctrl_c_under_asyncio_run(self):
        code = """
import asyncio
from orchestrator.worker.setup import finish_cleanup

async def cleanup():
    await asyncio.sleep(0.4)
    return True

async def main():
    print("READY", flush=True)
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        print("RESULT", await finish_cleanup(cleanup()), flush=True)
        raise

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("INTERRUPTED", flush=True)
"""
        source = str(Path(orchestrator.__file__).parent.parent)
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": source},
        )
        try:
            self.assertEqual(process.stdout.readline().strip(), "READY")
            time.sleep(0.1)
            process.send_signal(signal.SIGINT)
            time.sleep(0.1)
            process.send_signal(signal.SIGINT)
            out, err = process.communicate(timeout=20)
        finally:
            process.kill()
        self.assertIn("RESULT True", out)
        self.assertIn("INTERRUPTED", out)
        self.assertIn("Finishing cleanup", err)
        self.assertEqual(process.returncode, 0)

    async def test_setup_recovers_profile_when_write_and_withdrawal_both_fail(self):
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
            real_fdopen = os.fdopen
            calls = []

            class FullDisk:
                def __init__(self, stream):
                    self.stream = stream

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return self.stream.__exit__(*args)

                def write(self, text):
                    raise OSError("disk full")

            def failing_fdopen(fd, *args, **kwargs):
                calls.append(fd)
                stream = real_fdopen(fd, *args, **kwargs)
                return FullDisk(stream) if len(calls) == 1 else stream

            with (
                patch("orchestrator.worker.setup.getpass.getpass", return_value="private-password"),
                patch(
                    "orchestrator.worker.setup.login_and_enroll",
                    new=AsyncMock(return_value=(data, "private-session")),
                ),
                patch("orchestrator.worker.setup.EmbeddedTunnel", return_value=AsyncMock()),
                patch("orchestrator.worker.setup.os.fdopen", side_effect=failing_fdopen),
                patch(
                    "orchestrator.worker.setup.withdraw_enrollment",
                    new=AsyncMock(return_value=False),
                ),
                patch.dict(os.environ, {}, clear=True),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
                self.assertRaises(OSError),
            ):
                await setup(args)
            self.assertEqual(len(calls), 2)
            self.assertIn("may still be enrolled", stderr.getvalue())
            self.assertIn("WORKER_TOKEN=", output.read_text())
            self.assert_profile_permissions(output)

    async def test_recovery_never_writes_through_a_replaced_profile_path(self):
        await self._assert_replacement_preserved(symlink=True, released=False)

    async def test_cleanup_preserves_symlink_after_withdrawal(self):
        await self._assert_replacement_preserved(symlink=True, released=True)

    async def test_cleanup_preserves_replacement_regular_file(self):
        for released in (False, True):
            with self.subTest(released=released):
                await self._assert_replacement_preserved(symlink=False, released=released)

    async def _assert_replacement_preserved(self, *, symlink, released):
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "helper"
            helper.touch()
            victim = Path(directory) / "victim"
            victim.write_text("untouched")
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
            real_fdopen = os.fdopen
            replacement_blocked = False

            class Swapped:
                def __init__(self, stream):
                    self.stream = stream

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return self.stream.__exit__(*args)

                def write(self, text):
                    # Between reservation and the write, an attacker points the path elsewhere.
                    nonlocal replacement_blocked
                    try:
                        output.unlink()
                    except PermissionError:
                        if os.name != "nt":
                            raise
                        # Windows refuses the attack while the reservation is held;
                        # still simulate the failed profile write and exercise recovery.
                        replacement_blocked = True
                        raise OSError("disk full") from None
                    if symlink:
                        output.symlink_to(victim)
                    else:
                        victim.rename(output)
                    raise OSError("disk full")

            first_write = True

            def swapping_fdopen(fd, *args, **kwargs):
                nonlocal first_write
                stream = real_fdopen(fd, *args, **kwargs)
                if first_write:
                    first_write = False
                    return Swapped(stream)
                return stream

            with (
                patch("orchestrator.worker.setup.getpass.getpass", return_value="private-password"),
                patch(
                    "orchestrator.worker.setup.login_and_enroll",
                    new=AsyncMock(return_value=(data, "private-session")),
                ),
                patch("orchestrator.worker.setup.EmbeddedTunnel", return_value=AsyncMock()),
                patch("orchestrator.worker.setup.os.fdopen", side_effect=swapping_fdopen),
                patch(
                    "orchestrator.worker.setup.withdraw_enrollment",
                    new=AsyncMock(return_value=released),
                ),
                patch.dict(os.environ, {}, clear=True),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
                self.assertRaises(OSError),
            ):
                await setup(args)
            if os.name == "nt":
                self.assertTrue(replacement_blocked)
                self.assertEqual(victim.read_text(), "untouched")
                if released:
                    self.assertFalse(output.exists())
                else:
                    self.assertIn("Kept", stderr.getvalue())
                    self.assertIn("WORKER_TOKEN=", output.read_text())
                return
            if not released:
                self.assertIn("Could not keep", stderr.getvalue())
            self.assertEqual(output.read_text(), "untouched")
            self.assertEqual(output.is_symlink(), symlink)
            if symlink:
                self.assertEqual(victim.read_text(), "untouched")

    async def test_withdraw_request_uses_session_and_reports_outcome(self):
        status = 204

        def handle(request):
            self.assertEqual(request.method, "DELETE")
            self.assertEqual(str(request.url), ORIGIN + "/v1/worker-enrollments/worker-123")
            self.assertEqual(request.headers["authorization"], "Bearer private-session")
            return httpx.Response(status)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            for status, expected in ((204, "was withdrawn"), (500, "could not be withdrawn")):
                with contextlib.redirect_stderr(io.StringIO()) as stderr:
                    await withdraw_enrollment(client, ORIGIN, "private-session", "worker-123")
                self.assertIn(expected, stderr.getvalue())
                self.assertNotIn("private-session", stderr.getvalue())

    def test_enrollment_error_distinguishes_unconfigured_from_outage(self):
        unconfigured = {"detail": {"code": "enrollment_not_configured", "message": "off"}}
        self.assertIn("not configured", enrollment_error(httpx.Response(503, json=unconfigured)))
        outage = {"error": "database unavailable"}
        self.assertIn("temporarily unavailable", enrollment_error(httpx.Response(503, json=outage)))
        self.assertIn(
            "temporarily unavailable", enrollment_error(httpx.Response(503, text="<html>"))
        )
        self.assertIn("temporarily unavailable", enrollment_error(httpx.Response(503, json=[1])))
        self.assertIn("not approved", enrollment_error(httpx.Response(403, json={})))

    @unittest.skipUnless(shutil.which("uv"), "uv is not installed")
    def test_profile_values_round_trip_through_uv_env_file(self):
        values = {
            "TAILSCALE_HELPER": "/Users/José/tools/$HOME/orchestrator-tunnel",
            "TAILSCALE_STATE_DIR": r"C:\Users\José\tools\$HOME\tailscale",
            "WORKER_TOKEN": 'q"uo\\te\\\\${HOME}',
            "SERVER_URL": "wss://backend.example.ts.net:8443/v1/worker",
        }
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "worker.env"
            profile.write_text(
                "".join(f"{k}={dotenv_quote(v)}\n" for k, v in values.items()), encoding="utf-8"
            )
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "--no-cache",
                    "--offline",
                    "--python",
                    sys.executable,
                    "--env-file",
                    profile.as_posix(),
                    "python",
                    "-c",
                    "import json,os,sys;print(json.dumps({k: os.environ[k] for k in sys.argv[1:]}))",
                    *values,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=directory,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), values)
        with self.assertRaises(SetupError):
            dotenv_quote("a\nb")

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
            self.assertEqual(result, ({"worker_id": "worker-123"}, "private-session"))

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

        seen = []

        async def connected(socket, *args):
            seen.append(args)
            await socket.accept()
            await socket.send_json({"authenticated": True})
            await socket.close()

        with (
            patch("orchestrator.server.app.serve_worker", side_effect=connected),
            TestClient(app).websocket_connect(
                "/v1/worker",
                headers={"X-Worker-ID": "worker-new", "Authorization": "Bearer test-worker-secret"},
            ) as socket,
        ):
            self.assertTrue(socket.receive_json()["authenticated"])
        app.state.store.worker_authorized.assert_awaited_once_with(
            "worker-new", "Bearer test-worker-secret"
        )
        self.assertIs(
            seen[0][-1], True
        )  # Registration knows the credential came from the database.
