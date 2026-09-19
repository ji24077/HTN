"""Skill-guided native-code proposals, fixed workload checks and a bounded repair loop.

The first executable contract is demo/train.py. Model output is data, never a
local command. Remote execution belongs to an explicitly supplied validator.
Prepared files are not advertised as compiled, correct, faster, or GPU-tested.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from gpushare.portability.kernels import convert_kernel
from gpushare.portability.providers import ModelError
from gpushare.portability.scripts import (
    get_native_source,
    replace_native_source,
    translate_script,
    validate_demo_edit,
)


def _save(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class EngineeringAgent:
    def __init__(self, client, skills: str, *, max_attempts: int = 3, event: Callable | None = None):
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be 1..5 per phase")
        self.client, self.skills, self.max_attempts = client, skills, max_attempts
        self.event = event or (lambda value: None)

    def _propose(self, native: str, phase: str, source_backend: str, target_backend: str, feedback: str) -> dict:
        system = self.skills + """

You are the code-editing step of a verified GPU engineering loop.
Return JSON only with exactly these fields:
{"native_source": "complete replacement CUDA/HIP C++ source", "reason": "short explanation", "lesson": "candidate lesson or empty"}.
You can change only native_source. Its existing exported C ABI and calculation
must be preserved. Never change Python, test tolerances, reports, CPU reference,
precision, data, model, batch, optimizer or step count. Do not replace GPU math
with CPU math or add process/filesystem/network operations. The host validates
your proposal; you cannot declare it passed. Use the supported core GPU runtime
API and ordinary kernels. Keep error checks and release all owned allocations.
The C ABI has no cleanup/destruction hook: release every owned GPU allocation
before each call returns, including error paths. Do not retain device buffers
in static/global pointers or objects. Reducing the number of per-call allocations
is a useful option. Avoid architecture-specific intrinsics, launch bounds,
fast-math or reduced precision. Do not claim bit-identical output without tests.
Only numerical and timing measurements may support a success/speedup claim.
"""
        result = self.client.propose(system, {
            "phase": phase, "source_backend": source_backend, "target_backend": target_backend,
            "native_source": native, "feedback": feedback[-10000:],
            "workload": "float32 custom GPU preprocessing feeds fixed PyTorch full-batch Adam training; include zero-length and block-boundary correctness",
        })
        if set(result) != {"native_source", "reason", "lesson"}:
            raise ModelError("proposal must contain only native_source, reason, and lesson")
        if not isinstance(result["native_source"], str) or not 1 <= len(result["native_source"].encode()) <= 60000:
            raise ModelError("proposal native source must be a bounded nonempty string")
        for key, size in (("reason", 2000), ("lesson", 4000)):
            if not isinstance(result[key], str) or len(result[key]) > size:
                raise ModelError(f"proposal {key} is invalid")
        return result

    def run(self, source: str, destination: Path, *, target: str = "amd", validator=None) -> dict:
        if target not in {"amd", "nvidia"}:
            raise ValueError("target must be amd or nvidia")
        if len(source.encode()) > 80000:
            raise ValueError("script exceeds the first-demo size limit")
        # Python source uses universal newlines. Make hashes and frozen files
        # identical across Windows editing, local checks, and Linux GPU hosts.
        source = source.replace("\r\n", "\n").replace("\r", "\n")
        target_backend = "hip" if target == "amd" else "cuda"
        source_backend = "cuda" if target == "amd" else "hip"
        # Parsing and the supported-source check finish before any API request.
        native = get_native_source(source)
        original_check = convert_kernel(native, source_backend, target_backend)
        if any(item["severity"] == "blocker" for item in original_check["findings"]):
            raise ValueError("source contains unsupported GPU constructs; inspect translation findings first")
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        original_path = destination / "original.py"
        original_path.write_bytes(source.encode("utf-8"))
        (destination / "skills-snapshot.md").write_text(self.skills, encoding="utf-8")
        result = {"status": "running", "target": target, "source_sha256": _sha(source),
                  "source_hash_scope": "UTF-8 Python source with canonical LF newlines",
                  "skills_sha256": _sha(self.skills), "gpu_verified": False,
                  "performance_verified": False, "attempts": [], "lessons": [],
                  "scope": "demo-compatible scripts; only the embedded native GPU source may change",
                  "directory": str(destination)}

        def record():
            result["model_usage"] = self.client.summary()
            _save(destination / "result.json", result)

        record()
        try:
            if validator is not None:
                self.event({"phase": "baseline", "status": "running"})
                baseline = validator(original_path, "baseline", 0)
                result["baseline"] = baseline
                if baseline.get("passed") is not True or baseline.get("gpu_verified") is not True:
                    result.update(status="blocked", error="original GPU baseline did not pass")
                    return result
            current = source
            for phase in ("optimize", "translate"):
                feedback = ""
                phase_backend = source_backend if phase == "optimize" else target_backend
                if phase == "translate":
                    # Deterministic core mappings are a starting point for the agent,
                    # not proof that the result compiled or ran correctly.
                    seed = translate_script(current, target)
                    native = get_native_source(seed["source"])
                    feedback = "Initial supported API mappings are applied; review them and preserve semantics."
                else:
                    native = get_native_source(current)
                accepted = False
                for attempt in range(1, self.max_attempts + 1):
                    self.event({"phase": phase, "attempt": attempt, "status": "proposing"})
                    item = {"phase": phase, "attempt": attempt, "gpu_verified": False}
                    result["attempts"].append(item)
                    proposal = self._propose(native, phase, source_backend, phase_backend, feedback)
                    candidate = replace_native_source(current, proposal["native_source"])
                    guard = validate_demo_edit(source, candidate, phase)
                    kernel_check = convert_kernel(proposal["native_source"], phase_backend, phase_backend)
                    findings = [*guard, *kernel_check["findings"]]
                    item.update(reason=proposal["reason"], findings=findings, source_sha256=_sha(candidate))
                    candidate_path = destination / f"{phase}-{attempt}.py"
                    candidate_path.write_bytes(candidate.encode("utf-8"))
                    item["candidate"] = str(candidate_path)
                    if any(f["severity"] == "blocker" for f in findings):
                        item["status"] = "rejected_static_checks"
                        feedback = json.dumps(findings)
                        record()
                        continue
                    if validator is None:
                        item["status"] = "prepared_unverified"
                    else:
                        self.event({"phase": phase, "attempt": attempt, "status": "validating"})
                        validation = validator(candidate_path, phase, attempt)
                        item["validation"] = validation
                        if validation.get("passed") is not True or validation.get("gpu_verified") is not True:
                            item["status"] = "rejected_gpu_checks"
                            feedback = json.dumps(validation, ensure_ascii=False)[-10000:]
                            native = proposal["native_source"]
                            record()
                            continue
                        item.update(status="verified_correctness", gpu_verified=True)
                    current = candidate
                    item["lesson"] = proposal["lesson"]
                    if proposal["lesson"]:
                        result["lessons"].append({"phase": phase, "text": proposal["lesson"],
                                                  "status": "candidate_pending_regression",
                                                  "evidence_source_sha256": _sha(candidate)})
                    accepted = True
                    record()
                    break
                if not accepted:
                    result.update(status="blocked", error=f"{phase} exhausted its verified repair attempts")
                    return result
            output_path = destination / "translated.py"
            output_path.write_bytes(current.encode("utf-8"))
            result.update(status="verified_correctness" if validator else "prepared_unverified",
                          gpu_verified=validator is not None, output=str(output_path), output_sha256=_sha(current))
            result["performance_note"] = "No speedup is claimed from a short correctness run; matched repeated benchmarks are required."
            return result
        except Exception as exc:
            result.update(status="failed", error_type=type(exc).__name__,
                          error=str(exc) if isinstance(exc, (ValueError, ModelError)) else "engineering loop failed; no raw provider exception exposed")
            return result
        finally:
            record()
