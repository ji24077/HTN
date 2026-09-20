import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from orchestrator.worker.devices import capabilities


class WorkerDeviceTests(unittest.TestCase):
    def probe(self, report, requested="auto"):
        with patch(
            "orchestrator.worker.devices.subprocess.run",
            return_value=SimpleNamespace(stdout=json.dumps(report)),
        ):
            return capabilities("python_service", requested)

    def test_missing_runtime_reports_reason_and_cannot_claim_gpu(self):
        caps = self.probe({"runtime": "cpu", "reason": "PyTorch is not installed."}, "mps")
        self.assertEqual(caps.runtime, "cpu")
        self.assertFalse(caps.accelerator.available)
        self.assertIn("not installed", caps.accelerator.reason)

    def test_cpu_choice_preserves_detected_device_without_advertising_gpu_runtime(self):
        caps = self.probe({"runtime": "cuda", "vram_mib": 8192, "device": "GPU"}, "cpu")
        self.assertTrue(caps.accelerator.available)
        self.assertEqual((caps.runtime, caps.vram_mib, caps.runtime_preference), ("cpu", 0, "cpu"))

    def test_auto_reports_mps_without_inventing_dedicated_vram(self):
        caps = self.probe({"runtime": "mps", "device": "Apple GPU"})
        self.assertEqual((caps.runtime, caps.vram_mib), ("mps", 0))
        self.assertEqual(caps.machine.runtime_control, "startup")

    def test_probe_timeout_reports_unavailable(self):
        with patch(
            "orchestrator.worker.devices.subprocess.run",
            side_effect=subprocess.TimeoutExpired("probe", 30),
        ):
            caps = capabilities("python_service", "auto")
        self.assertFalse(caps.accelerator.available)
        self.assertEqual(caps.runtime, "cpu")
