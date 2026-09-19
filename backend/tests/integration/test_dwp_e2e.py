"""Real Node device -> Python HTTP/WS -> isolated PostgreSQL integration.

Opt in with RUN_DEVICE_E2E=1 and, if necessary, NODE24_BINARY=/path/to/node.
Needs pgserver (demo extra), Node >=24, and the root pnpm dependencies installed.
Set RUN_DEVICE_INFERENCE_E2E=1 to additionally require a working native ONNX runtime
and exercise ten pinned MNIST inputs through authenticated artifact downloads.
No DATABASE_URL, Supabase, existing device profile, or .env file is consumed.
"""

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from orchestrator.shared.dwp import verify_result

ROOT = Path(__file__).resolve().parents[3]
ADMIN = "ephemeral-device-e2e-admin-token-01234567890123"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def available_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def stop(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@unittest.skipUnless(
    os.getenv("RUN_DEVICE_E2E") == "1", "Set RUN_DEVICE_E2E=1; needs Node24 + pgserver"
)
class DeviceE2ETests(unittest.TestCase):
    def wait_for(self, name, predicate, *, processes=(), timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for process in processes:
                if process.poll() is not None:
                    self.fail(f"{name}: subprocess exited with {process.returncode}")
            try:
                result = predicate()
                if result:
                    return result
            except (httpx.TransportError, KeyError):
                pass
            time.sleep(0.1)
        self.fail(f"Timed out waiting for {name}")

    def test_real_agent_echo_walker_reconnect_and_cancellation(self):
        import pgserver

        node = os.getenv("NODE24_BINARY") or shutil.which("node")
        self.assertIsNotNone(node, "Install Node >=24 and set NODE24_BINARY")
        version = subprocess.run(
            [node, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=NO_WINDOW,
            check=True,
        ).stdout.strip()
        self.assertGreaterEqual(
            int(version.lstrip("v").split(".")[0]),
            24,
            "Set NODE24_BINARY to a Node >=24 executable",
        )
        self.assertTrue((ROOT / "node_modules").exists(), "Install root pnpm dependencies first")
        with tempfile.TemporaryDirectory(prefix="orchestrator-device-e2e-") as directory:
            temporary = Path(directory)
            database = pgserver.get_server(temporary / "postgres", cleanup_mode="stop")
            control = device = None
            log_files = []
            try:
                run_inference = os.getenv("RUN_DEVICE_INFERENCE_E2E") == "1"
                if run_inference:
                    fixture_dir = temporary / "fixtures"
                    fixture_dir.mkdir()
                    manifest = json.loads((ROOT / "fixtures" / "manifest.json").read_text())
                    shutil.copyfile(
                        ROOT / "fixtures" / "manifest.json", fixture_dir / "manifest.json"
                    )
                    for artifact in (manifest["model"], manifest["inputs"]):
                        digest = artifact["hash"]
                        self.assertEqual(len(digest), 64)
                        self.assertTrue(all(char in "0123456789abcdef" for char in digest))
                        shutil.copyfile(ROOT / "fixtures" / digest, fixture_dir / digest)
                # Start from only OS/runtime settings. No existing application credentials
                # are inherited, and neither subprocess reads an environment file.
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if key.upper()
                    in {
                        "PATH",
                        "PATHEXT",
                        "SYSTEMROOT",
                        "WINDIR",
                        "COMSPEC",
                        "TEMP",
                        "TMP",
                        "USERPROFILE",
                        "HOME",
                        "APPDATA",
                        "LOCALAPPDATA",
                        "LANG",
                        "LC_ALL",
                    }
                }
                env.update(
                    {
                        "DATABASE_URL": database.get_uri(),
                        "DATABASE_SCHEMA": "device_e2e",
                        "ADMIN_TOKEN": ADMIN,
                        "WORKER_TOKENS": "{}",
                        "SENTRY_DSN": "",
                        "PUBLIC_ORIGIN": "",
                        "REDIS_URL": "",
                        "SUPABASE_URL": "",
                        "DWP_HOME": str(temporary / "agent"),
                        "DWP_DNS_FALLBACK": "0",
                        "DWP_RELEASES_DIR": str(temporary / "releases"),
                        "DWP_FIXTURES_DIR": str(temporary / "fixtures"),
                        "PYTHONPATH": str(ROOT / "backend" / "src"),
                        "PYTHONUNBUFFERED": "1",
                    }
                )
                port = available_port()
                server = f"http://127.0.0.1:{port}"
                control_log = (temporary / "control.log").open("w", encoding="utf-8")
                agent_log = (temporary / "agent.log").open("w", encoding="utf-8")
                log_files.extend((control_log, agent_log))
                control = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "orchestrator.server.app:create_app",
                        "--factory",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--ws",
                        "websockets",
                        "--ws-max-size",
                        "131072",
                        "--log-level",
                        "warning",
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=control_log,
                    stderr=subprocess.STDOUT,
                    creationflags=NO_WINDOW,
                )
                with httpx.Client(
                    base_url=server, timeout=5, headers={"Authorization": "Bearer " + ADMIN}
                ) as client:
                    self.wait_for(
                        "isolated server",
                        lambda: client.get("/healthz").status_code == 200,
                        processes=(control,),
                    )
                    invite = client.post("/v1/device-invites")
                    invite.raise_for_status()
                    paired = subprocess.run(
                        [
                            node,
                            "packages/agent/src/index.ts",
                            "pair",
                            "--server",
                            server,
                            "--code",
                            invite.json()["code"],
                            "--label",
                            "Node E2E device",
                        ],
                        cwd=ROOT,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        creationflags=NO_WINDOW,
                    )
                    self.assertEqual(paired.returncode, 0, paired.stderr)
                    config = json.loads((temporary / "agent" / "config.json").read_text())
                    worker_id = config["hostId"]

                    def launch():
                        return subprocess.Popen(
                            [node, "packages/agent/src/index.ts", "run"],
                            cwd=ROOT,
                            env=env,
                            stdout=agent_log,
                            stderr=subprocess.STDOUT,
                            creationflags=NO_WINDOW,
                        )

                    def current_worker():
                        return next(
                            (
                                w
                                for w in client.get("/v1/workers").json()
                                if w["id"] == worker_id and w["state"] == "alive"
                            ),
                            None,
                        )

                    def submit(identifier, kind, payload):
                        response = client.post(
                            "/v1/tasks",
                            json={
                                "tasks": [
                                    {
                                        "id": identifier,
                                        "job_id": "device-e2e",
                                        "kind": kind,
                                        "payload": payload,
                                        "requirements": {"runtime": "cpu", "vram_mib": 0},
                                        "max_attempts": 3,
                                        "timeout_seconds": 60,
                                        "target_worker_id": worker_id,
                                    }
                                ]
                            },
                        )
                        self.assertEqual(response.status_code, 200, response.text)

                    def wait_task(identifier, state):
                        def accepted():
                            item = client.get("/v1/tasks/" + identifier).json()
                            if item["state"] == "failed":
                                self.fail(f"{identifier}: {item['failure']}")
                            return item if item["state"] == state else None

                        return self.wait_for(
                            identifier + " " + state, accepted, processes=(control, device)
                        )

                    def check_proof(item):
                        proof = item["attestation"]
                        self.assertEqual(proof["hostId"], worker_id)
                        self.assertEqual(proof["attempt"], item["generation"])
                        verified = verify_result(
                            {**proof, "output": item["result"]},
                            proof["rawOutput"],
                            proof["publicKey"],
                            task_id=item["spec"]["id"],
                            attempt=item["generation"],
                            worker_id=worker_id,
                        )
                        self.assertEqual(verified["outputHash"], proof["outputHash"])

                    device = launch()
                    first = self.wait_for(
                        "real agent hello", current_worker, processes=(control, device)
                    )
                    self.assertIn("walker_evolution", first["capabilities"]["kinds"])
                    submit("echo-e2e", "echo", {"nonce": "real-node-echo", "sleepMs": 50})
                    echo = wait_task("echo-e2e", "succeeded")
                    self.assertEqual(echo["result"]["nonce"], "real-node-echo")
                    check_proof(echo)

                    # Exercise the shipped independent audit command, including its
                    # failure status after changing accepted output but retaining proof.
                    exported = temporary / "echo-task.json"
                    exported.write_text(json.dumps(echo), encoding="utf-8")

                    def audit():
                        return subprocess.run(
                            [
                                sys.executable,
                                str(ROOT / "scripts" / "verify-device-result.py"),
                                str(exported),
                            ],
                            cwd=ROOT,
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=10,
                            creationflags=NO_WINDOW,
                        )

                    checked = audit()
                    self.assertEqual(checked.returncode, 0, checked.stderr)
                    self.assertTrue(json.loads(checked.stdout)["verified"])
                    tampered = json.loads(json.dumps(echo))
                    tampered["result"]["nonce"] = "altered-after-acceptance"
                    exported.write_text(json.dumps(tampered), encoding="utf-8")
                    rejected = audit()
                    self.assertEqual(rejected.returncode, 1, rejected.stdout)
                    self.assertIn("Verification failed", rejected.stderr)

                    submit(
                        "walker-e2e",
                        "walker_evolution",
                        {
                            "generation": 0,
                            "parent": [0.0] * 308,
                            "sigma": 0.1,
                            "seeds": [1, 2, 3],
                            "steps": 240,
                        },
                    )
                    walker = wait_task("walker-e2e", "succeeded")
                    self.assertEqual(len(walker["result"]["results"]), 3)
                    check_proof(walker)

                    if run_inference:
                        self.assertIn("cpu_inference_batch", first["capabilities"]["kinds"])
                        catalog = client.get("/v1/workloads")
                        catalog.raise_for_status()
                        inputs = catalog.json()["inference"]
                        self.assertIsNotNone(inputs, "Pinned inference fixtures were not loaded")
                        inputs["count"] = 10
                        cache = temporary / "agent" / "artifacts"
                        for digest in (inputs["modelHash"], inputs["inputsHash"]):
                            self.assertFalse((cache / digest).exists())
                            self.assertEqual(client.get("/artifacts/" + digest).status_code, 401)
                        submit("inference-e2e", "cpu_inference_batch", inputs)
                        inference = wait_task("inference-e2e", "succeeded")
                        result = inference["result"]
                        self.assertEqual(result["count"], 10)
                        self.assertEqual(len(result["predictions"]), 10)
                        self.assertTrue(
                            all(
                                type(value) is int and 0 <= value <= 9
                                for value in result["predictions"]
                            )
                        )
                        for digest in (inputs["modelHash"], inputs["inputsHash"]):
                            self.assertEqual(
                                hashlib.sha256((cache / digest).read_bytes()).hexdigest(), digest
                            )
                        data = (cache / inputs["inputsHash"]).read_bytes()
                        self.assertEqual(data[:4], b"DWPI")
                        label_start = 8 + int.from_bytes(data[4:8], "big") * 28 * 28
                        labels = data[label_start : label_start + 10]
                        correct = sum(
                            predicted == label
                            for predicted, label in zip(result["predictions"], labels, strict=True)
                        )
                        self.assertEqual(result["correct"], correct)
                        self.assertGreaterEqual(correct, 8)
                        check_proof(inference)
                        print(
                            f"\nNative ONNX inference: {correct}/10 MNIST inputs correct; "
                            "authenticated artifacts and signed output verified."
                        )

                    # A durable cancellation reaches the actual agent; its late abort/error
                    # cannot turn the cancelled task back into a retry or accepted result.
                    submit("cancel-e2e", "echo", {"nonce": "cancel", "sleepMs": 30000})
                    wait_task("cancel-e2e", "running")
                    cancelled = client.post("/v1/tasks/cancel-e2e/cancel")
                    self.assertEqual(cancelled.json()["state"], "cancelled")
                    submit("after-cancel-e2e", "echo", {"nonce": "after-cancel", "sleepMs": 10})
                    check_proof(wait_task("after-cancel-e2e", "succeeded"))
                    self.assertEqual(
                        client.get("/v1/tasks/cancel-e2e").json()["state"], "cancelled"
                    )

                    # Kill and restart the real process while a lease is held. Registration
                    # fences the first session, retries with generation+1, and signs anew.
                    submit("reconnect-e2e", "echo", {"nonce": "reconnect", "sleepMs": 2000})
                    running = wait_task("reconnect-e2e", "running")
                    previous_generation = running["generation"]
                    previous_session = running["session_id"]
                    stop(device)
                    device = launch()

                    def replaced():
                        worker = current_worker()
                        return (
                            worker if worker and worker["session_id"] != previous_session else None
                        )

                    self.wait_for(
                        "replacement device session", replaced, processes=(control, device)
                    )
                    completed = wait_task("reconnect-e2e", "succeeded")
                    self.assertEqual(completed["generation"], previous_generation + 1)
                    self.assertNotEqual(completed["session_id"], previous_session)
                    check_proof(completed)
            except BaseException:
                for handle in log_files:
                    handle.flush()
                for name in ("control.log", "agent.log"):
                    path = temporary / name
                    if path.exists():
                        print(
                            f"\n{name}:\n{path.read_text(encoding='utf-8')[-8000:]}",
                            file=sys.stderr,
                        )
                raise
            finally:
                stop(device)
                stop(control)
                for handle in log_files:
                    handle.close()
                database.cleanup()


if __name__ == "__main__":
    unittest.main()
