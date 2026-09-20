import contextlib
import io
import json
import runpy
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from orchestrator.worker.devices import PROBE, capabilities


class WorkerDeviceTests(unittest.TestCase):
    def test_example_selects_available_device_and_never_downgrades_explicit_gpu(self):
        source = Path(__file__).resolve().parents[3] / "examples/projects/pytorch/device.py"
        for cuda, mps, expected in (
            (True, True, "cuda"),
            (False, True, "mps"),
            (False, False, "cpu"),
        ):
            torch = SimpleNamespace(
                device=lambda value: value,
                version=SimpleNamespace(hip=None),
                cuda=SimpleNamespace(is_available=lambda: cuda),
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
            )
            with patch.dict("sys.modules", {"torch": torch}):
                select = runpy.run_path(str(source))["select_device"]
                self.assertEqual(select("auto"), expected)
                self.assertEqual(select("cpu"), "cpu")
                if not cuda:
                    with self.assertRaisesRegex(RuntimeError, "fallback is disabled"):
                        select("cuda")
                if not mps:
                    with self.assertRaisesRegex(RuntimeError, "fallback is disabled"):
                        select("mps")

    def test_cuda_build_without_gpu_and_failed_gpu_probe_fall_back_to_cpu(self):
        for available in (False, True):
            devices = []

            def ones(count, *, device):
                devices.append(device)
                if device != "cpu":
                    raise RuntimeError("driver unavailable")
                return SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: count))

            torch = SimpleNamespace(
                __version__="2.13.0+cu130",
                ones=ones,
                version=SimpleNamespace(hip=None),
                cuda=SimpleNamespace(is_available=lambda: available),
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
            )
            output = io.StringIO()
            with patch.dict("sys.modules", {"torch": torch}), contextlib.redirect_stdout(output):
                exec(PROBE, {})
            report = json.loads(output.getvalue())
            self.assertEqual(report["runtime"], "cpu")
            self.assertEqual(report["pytorch"], "2.13.0+cu130")
            self.assertEqual(devices, ["cpu", "cuda"] if available else ["cpu"])
            caps = self.probe(report)
            self.assertFalse(caps.accelerator.available)
            self.assertIsNotNone(caps.python)

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

    def test_rocm_probe_uses_torch_cuda_api_but_reports_amd_provider(self):
        torch = SimpleNamespace(
            __version__="2.10.0+rocm7.1",
            version=SimpleNamespace(hip="7.1"),
            ones=lambda count, **_: SimpleNamespace(
                sum=lambda: SimpleNamespace(item=lambda: count)
            ),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: None,
                get_device_properties=lambda _: SimpleNamespace(
                    name="AMD GPU", total_memory=8 * 1024**3
                ),
            ),
        )
        output = io.StringIO()
        with patch.dict("sys.modules", {"torch": torch}), contextlib.redirect_stdout(output):
            exec(PROBE, {})  # noqa: S102 - Exercise the trusted worker probe verbatim.
        caps = self.probe(json.loads(output.getvalue()))
        self.assertEqual(caps.runtime, "cuda")
        self.assertEqual(caps.accelerator.providers, ["rocm"])
        self.assertEqual(caps.vram_mib, 8192)
