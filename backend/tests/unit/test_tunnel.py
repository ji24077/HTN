"""Exercise helper lifecycle and TLS identity over the worker's supplied socket."""

import asyncio
import datetime
import os
import socket
import ssl
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from websockets.asyncio.server import serve

from orchestrator.shared.protocol import Capabilities, Message
from orchestrator.worker.agent import Agent, DirectConnect
from orchestrator.worker.config import WorkerConfig
from orchestrator.worker.executors import StubExecutor
from orchestrator.worker.tunnel import EmbeddedTunnel


def config(url="wss://gateway.example.ts.net:8443/v1/worker"):
    return WorkerConfig(
        url,
        "worker-test",
        "test-only-worker-credential-123456",
        Capabilities(runtime="cpu", vram_mib=0, kinds=["stub"]),
    )


class TunnelLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.helper = self.root / "helper"
        self.env = patch.dict(
            os.environ,
            {
                "TAILSCALE_HELPER": str(self.helper),
                "TAILSCALE_STATE_DIR": str(self.root / "state"),
                "WORKER_TOKEN": "must-not-enter-the-helper",
                "DATABASE_URL": "must-not-enter-the-helper",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        if sys.platform == "win32":
            # Windows cannot execute the test fixture's shebang. Keep a real
            # child process and production's arguments, pipes and filtered env.
            create_process = asyncio.create_subprocess_exec

            async def launch_helper(program, *args, **kwargs):
                if Path(program).resolve() == self.helper.resolve():
                    return await create_process(sys.executable, program, *args, **kwargs)
                return await create_process(program, *args, **kwargs)

            launcher = patch(
                "orchestrator.worker.tunnel.asyncio.create_subprocess_exec",
                new=launch_helper,
            )
            launcher.start()
            self.addCleanup(launcher.stop)

    def script(self, body):
        self.helper.write_text(f"#!{sys.executable}\n" + body)
        self.helper.chmod(0o700)

    async def test_helper_receives_no_application_secrets_and_closes(self):
        self.script("""import json, os, sys
assert "WORKER_TOKEN" not in os.environ
assert "DATABASE_URL" not in os.environ
print(json.dumps({"event": "ready", "port": 12345}), flush=True)
sys.stdin.read()
""")
        async with EmbeddedTunnel(config()) as tunnel:
            self.assertEqual(tunnel.port, 12345)
            if sys.platform != "win32":
                self.assertEqual(tunnel.state.stat().st_mode & 0o777, 0o700)
            self.assertIsNone(tunnel.process.returncode)
        self.assertIsNotNone(tunnel.process.returncode)

    async def test_invalid_readiness_cleans_up_child(self):
        self.script('import sys\nprint("invalid", flush=True)\nsys.stdin.read()\n')
        tunnel = EmbeddedTunnel(config())
        with self.assertRaisesRegex(RuntimeError, "readiness"):
            async with tunnel:
                self.fail("invalid helper started")
        self.assertIsNotNone(tunnel.process.returncode)

    async def test_exit_is_reported_without_direct_fallback(self):
        self.script('print(\'{"event":"ready","port":12345}\', flush=True)\n')
        async with EmbeddedTunnel(config()) as tunnel:
            with self.assertRaisesRegex(RuntimeError, "without a direct fallback"):
                await tunnel.wait()
            with self.assertRaises(RuntimeError):
                await tunnel.open_socket()

    async def test_rejects_plaintext_and_non_tailnet_targets(self):
        self.script("")
        for url in (
            "ws://localhost/v1/worker",
            "wss://example.com/v1/worker",
            "wss://evil.ts.net.example.com/v1/worker",
            "wss://x.ts.net/?token=bad",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                EmbeddedTunnel(config(url))


class TunnelTLSChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "gateway.example.ts.net")])
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName("gateway.example.ts.net")]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        certfile, keyfile = root / "cert.pem", root / "key.pem"
        certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        keyfile.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        self.server_ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.server_ssl.load_cert_chain(certfile, keyfile)
        self.client_ssl = ssl.create_default_context(cafile=str(certfile))

    async def exercise(self, hostname):
        received = []

        async def handler(ws):
            received.append(ws.request.headers["Authorization"])
            await ws.recv()
            await ws.send(Message(type="welcome", session_id="test-session").model_dump_json())
            await ws.wait_closed()

        async with serve(handler, "127.0.0.1", 0, ssl=self.server_ssl) as server:
            port = server.sockets[0].getsockname()[1]

            async def open_socket():
                sock = socket.socket()
                sock.setblocking(False)
                await asyncio.get_running_loop().sock_connect(sock, ("127.0.0.1", port))
                return sock

            tunnel = AsyncMock()
            tunnel.open_socket.side_effect = open_socket
            agent = Agent(config(f"wss://{hostname}:8443/v1/worker"), StubExecutor(), tunnel)
            agent.work_loop = AsyncMock()

            # Only supply a test trust root; URI and server_hostname follow production code.
            def connect(*args, **kwargs):
                return DirectConnect(*args, ssl=self.client_ssl, **kwargs)

            with patch("orchestrator.worker.agent.DirectConnect", side_effect=connect):
                if hostname == "gateway.example.ts.net":
                    await agent.session()
                    agent.work_loop.assert_awaited_once()
                    self.assertEqual(received, [f"Bearer {agent.config.token}"])
                else:
                    with self.assertRaises(ssl.SSLCertVerificationError):
                        await agent.session()
                    self.assertEqual(received, [])

    async def test_original_gateway_hostname_verified_over_tunnel(self):
        await self.exercise("gateway.example.ts.net")

    async def test_wrong_certificate_rejected_before_worker_token_sent(self):
        await self.exercise("impostor.example.ts.net")
