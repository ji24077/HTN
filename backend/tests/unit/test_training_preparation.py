"""Preparation contracts and numerical/performance gates; synthetic evidence, no GPUs."""

import json
import tempfile
import unittest
from importlib.resources import files as package_files
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from orchestrator.preprocessing.artifacts import encoded
from orchestrator.preprocessing.portability.scripts import translate_script
from orchestrator.preprocessing.projects import ProgramPlan, validate_plan
from orchestrator.preprocessing.training import (
    compatible,
    digest_native,
    performance_gate,
    validate_result,
)
from orchestrator.shared.protocol import Requirements
from orchestrator.worker.executors.native_benchmark import SIZES
from orchestrator.worker.executors.program_runner import run

SOURCE = (
    package_files("orchestrator.preprocessing.portability")
    .joinpath("native_training.py")
    .read_text()
)
VALIDATOR = "print('fixture validator')\n"


def plan(*, optimize=True, source_vendor="nvidia", target_vendor="amd", target="worker-b"):
    return {
        "summary": "Optimize the native stage, migrate if needed, then train the requested steps.",
        "worker_id": target,
        "requirements": {"runtime": "cuda", "vram_mib": 100},
        "entrypoint": "train.py",
        "working_directory": ".",
        "probe_args": [],
        "run_args": [
            "--steps",
            "16",
            "--expect-vendor",
            target_vendor,
            "--out",
            "{output_dir}/report.json",
            "--checkpoint",
            "{output_dir}/checkpoint.pt",
        ],
        "probe_timeout_seconds": 120,
        "run_timeout_seconds": 120,
        "validator": "validate.py",
        "validation_args": [],
        "outputs": [
            {"path": "checkpoint.pt", "kind": "checkpoint"},
            {"path": "metrics.json", "kind": "file"},
        ],
        "metrics": [{"name": "loss", "minimum": 0, "maximum": 1}],
        "preparation": {
            "rationale": "Native CUDA code must run on the target GPU.",
            "optimize": optimize,
            "source_worker_id": "worker-a",
            "source_vram_mib": 100,
            "source_vendor": source_vendor,
            "target_vendor": target_vendor,
            "validation_steps": 2,
        },
    }


def report(script=SOURCE, vendor="nvidia"):
    return {
        "operation": "synthetic-fixture",
        "gpu": "fixture GPU",
        "memory_scope": "allocator",
        "timing_scope": "fixture",
        "vendor": vendor,
        "parameters_on_gpu": True,
        "precision": "float32",
        "input_sha256": "1" * 64,
        "native_kernel_source_sha256": digest_native(script),
        "global_step": 2,
        "start_step": 0,
        "steps_executed": 2,
        "training_config": {"seed": 7, "learning_rate": 0.01, "batch_size": 257},
        "predictions": [0.2, 0.3],
        "loss_history": [0.5, 0.3],
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
        "training_seconds": 1,
        "custom_kernel_shape_checks": [
            {"elements": 0, "max_abs_error": 0},
            {"elements": 257, "max_abs_error": 0},
        ],
        "custom_kernel_max_abs_error": 0,
        "custom_kernel_calls": 3,
        "torch_version": "2.10.0",
    }


def benchmark(original=SOURCE, candidate=SOURCE, *, factor=0.8, count=21):
    return {
        "correctness_verified": True,
        "same_gpu_and_process": True,
        "vendor": "nvidia",
        "source_sha256": {
            "baseline": digest_native(original),
            "candidate": digest_native(candidate),
        },
        "cases": [
            {
                "elements": size,
                "correctness_verified": True,
                "baseline_sample_seconds": [0.01] * count,
                "candidate_sample_seconds": [0.01 * factor] * count,
            }
            for size in SIZES
        ],
    }


class TrainingPreparationContracts(unittest.TestCase):
    def test_only_native_changes_with_fixed_training_harness_are_accepted(self):
        files = {"train.py": encoded(SOURCE), "validate.py": encoded(VALIDATOR)}
        validate_plan(ProgramPlan.model_validate(plan()), files, "training", 600)
        migrated = translate_script(SOURCE, "amd")["source"]
        files["train.py"] = encoded(migrated)
        validate_plan(ProgramPlan.model_validate(plan(source_vendor="amd")), files, "training", 600)
        for changed in (SOURCE.replace("ALPHA = 0.375", "ALPHA = 0.9"), "print('not the harness')"):
            files["train.py"] = encoded(changed)
            with self.assertRaises(ValueError):
                validate_plan(ProgramPlan.model_validate(plan()), files, "training", 600)

    def test_full_run_cannot_skip_training_or_target_wrong_vendor(self):
        files = {"train.py": encoded(SOURCE), "validate.py": encoded(VALIDATOR)}
        for arguments in (
            ["--inspect"],
            plan()["run_args"] + ["--resume", "something.pt"],
            ["0" if x == "16" else x for x in plan()["run_args"]],
            ["nvidia" if x == "amd" else x for x in plan()["run_args"]],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_plan(
                    ProgramPlan.model_validate({**plan(), "run_args": arguments}),
                    files,
                    "training",
                    600,
                )

    def test_report_must_match_candidate_vendor_work_and_numerical_reference(self):
        baseline = report()
        target = translate_script(SOURCE, "amd")["source"]
        value = report(target, "amd")
        validate_result({"ok": True, "preparation": {"report": value}}, baseline, "amd", target, 2)
        for key, bad in (
            ("vendor", "nvidia"),
            ("native_kernel_source_sha256", "0" * 64),
            ("start_step", 1),
            ("predictions", [1, 1]),
            ("torch_version", "2.11.0"),
            ("parameters_on_gpu", False),
            ("input_sha256", "2" * 64),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_result(
                    {"ok": True, "preparation": {"report": {**value, key: bad}}},
                    baseline,
                    "amd",
                    target,
                    2,
                )

    def test_performance_claim_requires_repeated_paired_improvement_on_all_shapes(self):
        performance_gate(benchmark(), SOURCE, SOURCE, "nvidia")
        cases = [benchmark(factor=1.1), benchmark(factor=0.99), benchmark(count=1)]
        invalid = benchmark()
        invalid["cases"] = invalid["cases"][:-1]
        cases.append(invalid)
        invalid = benchmark()
        invalid["source_sha256"]["baseline"] = "wrong"
        cases.append(invalid)
        for value in cases:
            with self.assertRaises(ValueError):
                performance_gate(value, SOURCE, SOURCE, "nvidia")

    def test_rocm_is_not_mistaken_for_nvidia_even_with_cuda_runtime(self):
        worker = {
            "capabilities": {
                "runtime": "cuda",
                "vram_mib": 200,
                "kinds": ["python_project", "python_program"],
                "accelerator": {"providers": ["rocm"]},
            }
        }
        requirements = Requirements(runtime="cuda", vram_mib=100)
        self.assertTrue(compatible(worker, requirements, "amd"))
        self.assertFalse(compatible(worker, requirements, "nvidia"))

    def test_worker_returns_report_and_paired_evidence_without_publishing_probe_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "__dispatch_outputs__"
            output.mkdir()
            request = {
                "program": {"probe": True},
                "entrypoint": "train.py",
                "args": [],
                "preparation": {"stage": "optimization", "vendor": "nvidia"},
            }
            torch = SimpleNamespace(
                version=SimpleNamespace(hip=None),
                cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
            )

            def execute(*_):
                (output / "report.json").write_text(json.dumps(report()))
                (output / "checkpoint.pt").write_bytes(b"fixture")

            paired = SimpleNamespace(benchmark=lambda *_: benchmark())
            with (
                patch.dict(
                    "sys.modules", {"torch": torch, "__dispatch_native_benchmark__": paired}
                ),
                patch(
                    "orchestrator.worker.executors.program_runner.run_script", side_effect=execute
                ),
            ):
                result = run(request, root)
            self.assertTrue(result["ok"])
            self.assertEqual(result["preparation"]["report"], report())
            self.assertIn("benchmark", result["preparation"])
            self.assertNotIn("files", result)
