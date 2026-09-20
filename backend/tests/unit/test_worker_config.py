import unittest
from unittest.mock import patch

from orchestrator.shared.protocol import Capabilities, Machine
from orchestrator.worker.config import WorkerConfig, machine_specs


class MachineSpecsTests(unittest.TestCase):
    def test_reports_cores_and_ram_in_mib(self):
        with (
            patch("orchestrator.worker.config.os.cpu_count", return_value=8),
            patch(
                "orchestrator.worker.config.os.sysconf", side_effect=[4194304, 4096], create=True
            ),
        ):
            machine = machine_specs()
        self.assertEqual(machine.logical_cores, 8)
        self.assertEqual(machine.total_ram_mb, 16384)

    def test_missing_platform_details_are_unknown(self):
        with (
            patch("orchestrator.worker.config.os.cpu_count", return_value=None),
            patch("orchestrator.worker.config.os.sysconf", side_effect=AttributeError, create=True),
        ):
            machine = machine_specs()
        self.assertIsNone(machine.logical_cores)
        self.assertIsNone(machine.total_ram_mb)

    def test_worker_registration_includes_machine_specs(self):
        with patch.dict(
            "os.environ",
            {
                "WORKER_ID": "worker-a",
                "WORKER_TOKEN": "fixture-worker-token-123456789",
                "SERVER_URL": "ws://localhost:8080/v1/worker",
            },
            clear=True,
        ):
            config = WorkerConfig.from_env("stub")
        self.assertIsNotNone(config.capabilities.machine)

    def test_gpu_registration_retains_machine_specs_and_startup_runtime_controls(self):
        with (
            patch.dict(
                "os.environ",
                {
                    "WORKER_ID": "worker-a",
                    "WORKER_TOKEN": "fixture-worker-token-123456789",
                    "WORKER_RUNTIME": "auto",
                    "SERVER_URL": "ws://localhost:8080/v1/worker",
                },
                clear=True,
            ),
            patch(
                "orchestrator.worker.config.capabilities",
                return_value=Capabilities(
                    runtime="cuda",
                    vram_mib=8192,
                    kinds=["python_service"],
                ),
            ),
            patch(
                "orchestrator.worker.config.machine_specs",
                return_value=Machine(
                    logical_cores=8,
                    total_ram_mb=16384,
                ),
            ),
        ):
            config = WorkerConfig.from_env("python_service")
        self.assertEqual(config.capabilities.runtime, "cuda")
        self.assertEqual(config.capabilities.vram_mib, 8192)
        self.assertEqual(config.capabilities.machine.total_ram_mb, 16384)
        self.assertEqual(config.capabilities.machine.logical_cores, 8)
        self.assertEqual(config.capabilities.machine.max_concurrency, 1)
        self.assertEqual(config.capabilities.machine.runtime_control, "startup")
