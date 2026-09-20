"""Bounded SSH execution on caller-owned GPU hosts; never rents or changes hosts.

The caller supplies owned endpoints and handles budget enforcement and teardown.
Only an AST-protected demo script is uploaded. Native image Python/Torch/compiler
are used as-is; no dependencies or provider credentials are installed or sent.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import ipaddress
import json
import math
import os
import re
import shlex
import subprocess
import uuid
from pathlib import Path, PurePosixPath

TIMEOUT_SECONDS = 300
_PHASES = {"baseline", "optimize", "translate", "target_optimize"}
_SECRETS = re.compile(r"\b(?:sk-[\w-]{16,}|rpa_[\w-]{16,}|gh[pousr]_[\w-]{16,})")
_REMOTE_ENV = (
    "if [ -f /etc/gpushare-gpu-env ]; then . /etc/gpushare-gpu-env || exit $?; fi; "
    'export PATH="/opt/venv/bin:/opt/conda/bin:/opt/rocm/bin:/usr/local/cuda/bin:${PATH:-/usr/bin:/bin}"; '
    'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/opt/rocm/lib:/usr/local/cuda/lib64"; '
)
_PROBE = """
import json,os,shutil,sys
visibility_keys=("CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES",
                 "ROCR_VISIBLE_DEVICES","GPU_DEVICE_ORDINAL")
r={"vendor":sys.argv[1],"passed":False,"python_executable":sys.executable,
   "visibility_env":{key:os.environ[key] for key in visibility_keys if key in os.environ}}
try:
 import torch
 backend="amd" if torch.version.hip else "nvidia" if torch.version.cuda else "cpu"
 r.update(vendor=backend,torch_version=str(torch.__version__),cuda=torch.version.cuda,hip=torch.version.hip,
          torch_file=torch.__file__,visible_device_count=torch.cuda.device_count())
 if r["visible_device_count"] != 1: raise RuntimeError("exactly one visible GPU is required")
 if backend != sys.argv[1] or not torch.cuda.is_available(): raise RuntimeError("wrong or unavailable GPU backend")
 r["gpu"]=torch.cuda.get_device_name(0)
 free,total=torch.cuda.mem_get_info(0)
 r.update(free_memory_bytes=free,total_memory_bytes=total)
 if free <= 0: raise RuntimeError("GPU has zero free memory")
 r["compiler"]=shutil.which("hipcc" if backend=="amd" else "nvcc")
 if not r["compiler"]: raise RuntimeError("native GPU compiler is unavailable")
 if not shutil.which("timeout"): raise RuntimeError("remote timeout utility is unavailable")
 torch.cuda.reset_peak_memory_stats()
 x=torch.randn((32,32),device="cuda",dtype=torch.float32,requires_grad=True)
 loss=(x@x).square().mean();loss.backward();torch.cuda.synchronize()
 if not torch.isfinite(loss).item() or not torch.isfinite(x.grad).all().item(): raise RuntimeError("GPU math smoke failed")
 r.update(passed=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
except Exception as e: r["error"]=str(e)
print(json.dumps(r),flush=True)
"""


def _redact(value: str) -> str:
    return _SECRETS.sub("[REDACTED]", value)


def _source_bytes(path: Path) -> bytes:
    """Match the engine's canonical UTF-8/LF source hash and artifact scope."""
    return Path(path).read_text(encoding="utf-8").encode("utf-8")


def _transport_environment() -> dict[str, str]:
    # Windows OpenSSH exits 255 before connecting if PROGRAMDATA is absent.
    # Keep platform essentials only; `ssh -F none` also disables SendEnv config.
    allowed = {
        "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE",
        "LANG", "LC_ALL", "PROGRAMDATA",
    }
    return {key: value for key, value in os.environ.items() if key.upper() in allowed}


def _kernel_digest(script: str) -> str:
    literals = []
    for node in ast.parse(script).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"CUDA_SOURCE", "HIP_SOURCE"}
            for target in node.targets
        ):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                literals.append(node.value.value)
    if len(literals) != 1:
        raise ValueError("demo must contain exactly one literal CUDA_SOURCE or HIP_SOURCE")
    return hashlib.sha256(literals[0].encode()).hexdigest()


class RemoteDemoValidator:
    def __init__(self, source_info: dict, target_info: dict, work_dir: str,
                 local_dir: Path, baseline_script: Path, steps: int = 8, *,
                 source_vendor: str = "nvidia", target_vendor: str = "amd"):
        if {source_vendor, target_vendor} != {"nvidia", "amd"}:
            raise ValueError("source_vendor and target_vendor must be opposite nvidia/amd backends")
        self.source_vendor, self.target_vendor = source_vendor, target_vendor
        if type(steps) is not int or not 1 <= steps <= 10000:
            raise ValueError("steps must be between 1 and 10000")
        path = PurePosixPath(work_dir)
        if (not re.fullmatch(r"/workspace/gpushare-translation/[A-Za-z0-9_-]+", work_dir)
                or ".." in path.parts):
            raise ValueError("work_dir must be /workspace/gpushare-translation/<session>")
        self.work_dir, self.steps = work_dir, steps
        self.local_dir = Path(local_dir).resolve()
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.source_info = self._validate_endpoint(source_info)
        self.target_info = self._validate_endpoint(target_info)
        self.baseline_bytes = _source_bytes(baseline_script)
        self.baseline_source = self.baseline_bytes.decode("utf-8")
        self.baseline_sha256 = hashlib.sha256(self.baseline_bytes).hexdigest()
        self._baseline = None
        self._baseline_attempted = False
        self._evidence_roots = [self.local_dir]
        self.frozen_dir = self.local_dir / "baseline"
        if self.frozen_dir.exists():
            raise ValueError("baseline artifacts already exist; use a new validation directory")
        # Load the trusted repository comparator, never anything in a candidate.
        comparator = Path(__file__).resolve().parents[3] / "demo" / "compare.py"
        spec = importlib.util.spec_from_file_location("_trusted_demo_compare", comparator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._compare = module.compare_reports

    def restore_baseline(self, previous_validation_dir: Path) -> None:
        """Import trusted local artifacts from an earlier validator, without GPU work.

        Intended for a fresh follow-up process using the same caller-owned pods.
        The original script and all frozen artifact hashes must still match.
        This is integrity validation, not authentication of arbitrary user files.
        """
        if self._baseline is not None or self._baseline_attempted:
            raise ValueError("baseline already established or attempted")
        previous = Path(previous_validation_dir).resolve()
        frozen = previous / "baseline"
        report, files = self._read_frozen(frozen)
        # Require the earlier successful execution record, not just a report.
        proven = False
        for attempt in previous.glob("baseline-*"):
            try:
                outcome = json.loads((attempt / "validation.json").read_text(encoding="utf-8"))
                proven = (outcome.get("passed") is True and outcome.get("gpu_verified") is True
                          and (attempt / "report.json").read_bytes() == files["report.json"]
                          and (attempt / "train.py").read_bytes() == self.baseline_bytes)
                if proven:
                    break
            except (OSError, ValueError):
                continue
        if not proven:
            raise ValueError("verified baseline execution record is missing")
        self.frozen_dir.mkdir(exist_ok=False)
        for name, payload in files.items():
            with (self.frozen_dir / name).open("xb") as stream:
                stream.write(payload)
        (self.frozen_dir / "manifest.json").write_bytes((frozen / "manifest.json").read_bytes())
        self._baseline, self._baseline_attempted = report, True
        self._evidence_roots.append(previous)

    def _read_frozen(self, directory: Path) -> tuple[dict, dict[str, bytes]]:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        names = {"train.py", "report.json", "checkpoint.pt"}
        if set(manifest) != names:
            raise ValueError("invalid frozen baseline manifest")
        files = {name: (directory / name).read_bytes() for name in names}
        if any(not payload or hashlib.sha256(payload).hexdigest() != manifest[name]
               for name, payload in files.items()):
            raise ValueError("frozen baseline artifact hash mismatch")
        report = json.loads(files["report.json"])
        self._compare(report, report)
        if (files["train.py"] != self.baseline_bytes or report.get("vendor") != self.source_vendor
                or report.get("global_step") != self.steps or report.get("start_step") != 0
                or report.get("native_kernel_source_sha256") != _kernel_digest(self.baseline_source)):
            raise ValueError("frozen baseline does not match original script, vendor, or steps")
        return report, files

    def _require_verified_translation(self, payload: bytes) -> None:
        for root in self._evidence_roots:
            for pattern in ("translate-*", "target_optimize-*"):
                for attempt in root.glob(pattern):
                    try:
                        outcome = json.loads((attempt / "validation.json").read_text(encoding="utf-8"))
                        report = json.loads((attempt / "report.json").read_text(encoding="utf-8"))
                        if (outcome.get("passed") is True and outcome.get("gpu_verified") is True
                                and (attempt / "train.py").read_bytes() == payload
                                and report.get("vendor") == self.target_vendor
                                and report.get("start_step") == 0
                                and report.get("native_kernel_source_sha256") == _kernel_digest(payload.decode("utf-8"))
                                and self._compare(self._baseline, report)["status"] == "passed"):
                            return
                    except (OSError, ValueError, KeyError):
                        continue
        raise ValueError("candidate needs a successful target-GPU correctness validation before resume")

    def verify_resume(self, candidate: Path, phase: str = "translate", attempt: int = 0,
                      *, final_steps: int | None = None) -> dict:
        """Resume one frozen checkpoint on both backends and compare continued training.

        Defaults to four additional steps. No baseline or correctness runs are
        inferred: __call__ must have verified both first (or restore_baseline
        imports their trusted artifacts). Caller owns costs and pod cleanup.
        """
        output = {"passed": False, "gpu_verified": False, "report_path": None,
                  "comparison": None, "error": None, "diagnostics": "", "reports": {}}
        diagnostics = []
        run_dir = None
        try:
            if phase not in {"translate", "target_optimize"} or type(attempt) is not int or attempt < 0:
                raise ValueError("invalid resume phase or attempt")
            final_steps = self.steps + 4 if final_steps is None else final_steps
            if type(final_steps) is not int or not self.steps < final_steps <= 10000:
                raise ValueError("final_steps must exceed baseline steps and be at most 10000")
            if self._baseline is None:
                raise ValueError("verified frozen baseline is required before resume")
            baseline, frozen = self._read_frozen(self.frozen_dir)
            if baseline != self._baseline:
                raise ValueError("frozen baseline changed after validation")
            payload = _source_bytes(candidate)
            from gpushare.portability.scripts import validate_demo_edit

            findings = validate_demo_edit(self.baseline_source, payload.decode("utf-8"), phase)
            if any(finding["severity"] == "blocker" for finding in findings):
                raise ValueError("candidate violates demo edit guard")
            self._require_verified_translation(payload)
            name = f"resume-{phase}-{attempt:03d}-{uuid.uuid4().hex[:12]}"
            run_dir = self.local_dir / name
            run_dir.mkdir()
            checkpoint_hash = hashlib.sha256(frozen["checkpoint.pt"]).hexdigest()
            output.update(checkpoint_input_sha256=checkpoint_hash, start_step=self.steps,
                          global_step=final_steps, steps_executed=final_steps - self.steps)
            probes = {}
            for vendor, endpoint in ((self.target_vendor, self.target_info), (self.source_vendor, self.source_info)):
                probes[vendor] = self._preflight(endpoint, vendor, run_dir / f"{vendor}-preflight.json", diagnostics)
                if probes[vendor]["torch_version"].split("+", 1)[0] != baseline["torch_version"].split("+", 1)[0]:
                    raise ValueError("native Torch base version differs from frozen baseline")
            reports = {}
            # Run target first, then an independent original-source continuation.
            for role, vendor, endpoint, script in (
                ("target", self.target_vendor, self.target_info, payload),
                ("source", self.source_vendor, self.source_info, self.baseline_bytes),
            ):
                directory = run_dir / role
                directory.mkdir()
                (directory / "train.py").write_bytes(script)
                (directory / "input-checkpoint.pt").write_bytes(frozen["checkpoint.pt"])
                remote = f"{self.work_dir}/{name}/{role}"
                if self._ssh(endpoint, shlex.join(["mkdir", "-p", remote]), diagnostics).returncode:
                    raise RuntimeError("could not create resume directory")
                for filename in ("train.py", "input-checkpoint.pt"):
                    result = self._execute([*self._transport(endpoint, scp=True), str(directory / filename),
                                            f"{self._host(endpoint)}:{remote}/{filename}"], diagnostics)
                    if result.returncode:
                        raise RuntimeError(f"could not upload resume {filename}")
                result = self._ssh(endpoint, shlex.join(["sha256sum", f"{remote}/input-checkpoint.pt"]), diagnostics)
                if result.returncode or not result.stdout.split() or result.stdout.split()[0] != checkpoint_hash:
                    raise ValueError("remote resume checkpoint hash mismatch")
                command = f"cd {shlex.quote(remote)} && " + shlex.join([
                    "timeout", "--signal=TERM", "--kill-after=5s", "280s", "python3", "-u", "train.py",
                    "--expect-vendor", vendor, "--resume", "input-checkpoint.pt", "--steps", str(final_steps),
                    "--out", "report.json", "--checkpoint", "checkpoint.pt",
                ])
                run_error = None
                try:
                    result = self._ssh(endpoint, command, diagnostics)
                    if result.returncode:
                        run_error = f"{role} resume compile/run exited {result.returncode}"
                except Exception as exc:
                    run_error = str(exc)
                # Keep both outputs even on timeout or a failing numerical gate.
                for filename in ("report.json", "checkpoint.pt"):
                    try:
                        self._download(endpoint, f"{remote}/{filename}", directory / filename, diagnostics)
                    except Exception as exc:
                        diagnostics.append(_redact(str(exc)))
                report_path = directory / "report.json"
                if report_path.is_file():
                    output["reports"][role] = str(report_path)
                    if role == "target":
                        output["report_path"] = str(report_path)
                (directory / "execution.log").write_text("\n".join(diagnostics), encoding="utf-8")
                if run_error:
                    raise RuntimeError(run_error)
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self._compare(report, report)
                if (report.get("vendor") != vendor or report.get("gpu") != probes[vendor]["gpu"]
                        or report.get("torch_version") != probes[vendor]["torch_version"]
                        or report.get("native_kernel_source_sha256") != _kernel_digest(script.decode("utf-8"))
                        or report.get("start_step") != self.steps or report.get("global_step") != final_steps
                        or report.get("steps_executed") != final_steps - self.steps
                        or report.get("optimizer_restored") is not True
                        or type(report.get("optimizer_state_entries")) is not int
                        or report["optimizer_state_entries"] <= 0
                        or report.get("peak_allocated_bytes", 0) <= 0 or report.get("peak_reserved_bytes", 0) <= 0
                        or any(report.get(key) != baseline.get(key) for key in
                               ("input_sha256", "training_config", "precision", "operation"))):
                    raise ValueError(f"{role} report lacks matching GPU, code, optimizer, input, or resume-step evidence")
                error = report.get("resume_prediction_max_abs_error")
                initial = report.get("initial_predictions")
                if (type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 1e-5
                        or not isinstance(initial, list) or len(initial) != len(baseline["predictions"])
                        or any(type(a) not in (int, float) or not math.isfinite(a)
                               or not math.isclose(a, b, abs_tol=1e-5, rel_tol=1e-4)
                               for a, b in zip(initial, baseline["predictions"], strict=True))):
                    raise ValueError(f"{role} initial predictions do not match frozen checkpoint")
                checkpoint = directory / "checkpoint.pt"
                if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                    raise ValueError(f"{role} resumed checkpoint is missing or empty")
                reports[role] = report
            output["gpu_verified"] = True
            output["comparison"] = self._compare(reports["source"], reports["target"])
            if output["comparison"]["status"] != "passed":
                raise ValueError("cross-backend resumed training failed trusted numerical comparison")
            self._read_frozen(self.frozen_dir)
            output["passed"] = True
        except Exception as exc:
            output["error"] = _redact(str(exc))[:10000]
        output["diagnostics"] = "\n".join(diagnostics)[-10000:]
        if run_dir is not None:
            (run_dir / "validation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        return output

    @staticmethod
    def _validate_endpoint(info: dict) -> dict:
        ip = str(ipaddress.ip_address(info["ip"]))
        port = info["port"]
        key = Path(info["key"]).resolve()
        if type(port) is not int or not 1 <= port <= 65535 or not key.is_file():
            raise ValueError("endpoint requires a valid port and existing SSH private-key file")
        return {"ip": ip, "port": port, "key": str(key)}

    def _transport(self, info: dict, *, scp: bool = False) -> list[str]:
        return ["scp" if scp else "ssh", "-F", "none", "-i", info["key"],
                "-P" if scp else "-p", str(info["port"]),
                "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={self.local_dir / 'known_hosts'}",
                "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
                "-o", "ServerAliveCountMax=2"]

    @staticmethod
    def _host(info: dict) -> str:
        ip = info["ip"]
        return f"root@[{ip}]" if ":" in ip else f"root@{ip}"

    def _execute(self, argv: list[str], diagnostics: list[str]):
        # Do not give SSH provider/API tokens through inherited SendEnv settings.
        environment = _transport_environment()
        try:
            result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=TIMEOUT_SECONDS, env=environment)
        except subprocess.TimeoutExpired as exc:
            for stream in (exc.stdout, exc.stderr):
                if stream:
                    diagnostics.append(_redact(stream.decode(errors="replace") if isinstance(stream, bytes) else stream))
            raise RuntimeError("remote operation timed out after 300 seconds") from exc
        diagnostics.append(_redact(result.stdout + result.stderr))
        return result

    def _ssh(self, info: dict, command: str, diagnostics: list[str]):
        # The caller's trusted startup writes this fixed-path environment file.
        # SSH otherwise loses the image's venv, library and device visibility settings.
        return self._execute([*self._transport(info), self._host(info), _REMOTE_ENV + command], diagnostics)

    def _download(self, info: dict, remote: str, destination: Path, diagnostics: list[str]):
        return self._execute([*self._transport(info, scp=True),
                              f"{self._host(info)}:{remote}", str(destination)], diagnostics)

    def _preflight(self, info: dict, vendor: str, destination: Path, diagnostics: list[str]) -> dict:
        try:
            result = self._ssh(info, shlex.join(["timeout", "--kill-after=5s", "120s",
                                                "python3", "-c", _PROBE, vendor]), diagnostics)
            if result.returncode:
                raise RuntimeError(f"native preflight command exited {result.returncode}")
            report = json.loads(result.stdout)
        except Exception as exc:
            report = {"passed": False, "vendor": vendor, "error": _redact(str(exc))}
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if (report.get("passed") is not True or report.get("vendor") != vendor
                or type(report.get("visible_device_count")) is not int or report["visible_device_count"] != 1
                or not report.get("python_executable") or not report.get("torch_file")
                or not report.get("compiler") or report.get("free_memory_bytes", 0) <= 0
                or report.get("peak_allocated_bytes", 0) <= 0):
            raise RuntimeError(f"{vendor} native preflight failed: {report.get('error', 'invalid GPU evidence')}")
        return report

    def __call__(self, candidate: Path, phase: str, attempt: int) -> dict:
        output = {"passed": False, "gpu_verified": False, "report_path": None,
                  "comparison": None, "error": None, "diagnostics": ""}
        diagnostics = []
        run_dir = None
        remote = None
        info = None
        try:
            if phase not in _PHASES or type(attempt) is not int or attempt < 0:
                raise ValueError("invalid validation phase or attempt")
            if phase != "baseline" and self._baseline is None:
                raise ValueError("a verified frozen source-GPU baseline is required first")
            if phase == "baseline" and self._baseline_attempted:
                raise ValueError("baseline may only execute once")
            payload = _source_bytes(candidate)
            source = payload.decode("utf-8")
            from gpushare.portability.scripts import validate_demo_edit

            findings = validate_demo_edit(self.baseline_source, source, phase)
            if any(finding["severity"] == "blocker" for finding in findings):
                raise ValueError(f"candidate violates demo edit guard: {findings}")
            if phase == "baseline":
                if payload != self.baseline_bytes:
                    raise ValueError("baseline candidate must equal the original script bytes")
                self._baseline_attempted = True
            digest = _kernel_digest(source)
            name = f"{phase}-{attempt:03d}-{uuid.uuid4().hex[:12]}"
            run_dir = self.local_dir / name
            run_dir.mkdir()
            (run_dir / "train.py").write_bytes(payload)
            # Target first: reject unavailable target memory before doing source work.
            probes = {}
            for vendor, endpoint in ((self.target_vendor, self.target_info), (self.source_vendor, self.source_info)):
                probes[vendor] = self._preflight(endpoint, vendor, run_dir / f"{vendor}-preflight.json", diagnostics)
            versions = [probe["torch_version"].split("+", 1)[0] for probe in probes.values()]
            if len(set(versions)) != 1:
                raise ValueError(f"GPU hosts require matching base Torch versions, got {versions}")
            if self._baseline and versions[0] != self._baseline["torch_version"].split("+", 1)[0]:
                raise ValueError("native Torch base version changed after the frozen baseline")
            vendor = self.source_vendor if phase in {"baseline", "optimize"} else self.target_vendor
            info = self.source_info if vendor == self.source_vendor else self.target_info
            remote = f"{self.work_dir}/{name}"
            result = self._ssh(info, shlex.join(["mkdir", "-p", remote]), diagnostics)
            if result.returncode:
                raise RuntimeError("could not create remote attempt directory")
            result = self._execute([*self._transport(info, scp=True), str(run_dir / "train.py"),
                                    f"{self._host(info)}:{remote}/train.py"], diagnostics)
            if result.returncode:
                raise RuntimeError("could not stage candidate script")
            command = f"cd {shlex.quote(remote)} && " + shlex.join([
                "timeout", "--signal=TERM", "--kill-after=5s", "280s", "python3", "-u", "train.py",
                "--expect-vendor", vendor, "--steps", str(self.steps),
                "--out", "report.json", "--checkpoint", "checkpoint.pt",
            ])
            run_error = None
            try:
                result = self._ssh(info, command, diagnostics)
                if result.returncode:
                    run_error = f"candidate compile/run exited {result.returncode}"
            except Exception as exc:
                run_error = str(exc)
            # Preserve any report that exists, even if compilation/execution failed.
            local_report = run_dir / "report.json"
            fetched = self._download(info, f"{remote}/report.json", local_report, diagnostics)
            if local_report.is_file():
                output["report_path"] = str(local_report)
            if run_error:
                raise RuntimeError(run_error)
            if fetched.returncode or not local_report.is_file():
                raise RuntimeError("candidate did not produce a downloadable report")
            report = json.loads(local_report.read_text(encoding="utf-8"))
            comparison = self._compare(self._baseline or report, report)
            output["comparison"] = comparison
            if (report.get("vendor") != vendor or report.get("gpu") != probes[vendor]["gpu"]
                    or report.get("torch_version") != probes[vendor]["torch_version"]
                    or report.get("global_step") != self.steps or report.get("start_step") != 0
                    or report.get("native_kernel_source_sha256") != digest
                    or report.get("peak_allocated_bytes", 0) <= 0
                    or report.get("peak_reserved_bytes", 0) <= 0):
                raise ValueError("candidate report does not match native GPU, Torch, code, steps, or allocation evidence")
            output["gpu_verified"] = True
            if comparison["status"] != "passed":
                raise ValueError("candidate failed trusted numerical comparison")
            checkpoint = run_dir / "checkpoint.pt"
            fetched = self._download(info, f"{remote}/checkpoint.pt", checkpoint, diagnostics)
            if fetched.returncode or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise RuntimeError("candidate checkpoint is missing or empty")
            if phase == "baseline":
                self.frozen_dir.mkdir(exist_ok=False)
                for name in ("train.py", "report.json", "checkpoint.pt"):
                    with (self.frozen_dir / name).open("xb") as stream:
                        stream.write((run_dir / name).read_bytes())
                manifest = {name: hashlib.sha256((self.frozen_dir / name).read_bytes()).hexdigest()
                            for name in ("train.py", "report.json", "checkpoint.pt")}
                with (self.frozen_dir / "manifest.json").open("x", encoding="utf-8") as stream:
                    json.dump(manifest, stream, indent=2)
                self._baseline = report
                output["report_path"] = str(self.frozen_dir / "report.json")
            output["passed"] = True
        except Exception as exc:
            output["error"] = _redact(str(exc))[:10000]
        output["diagnostics"] = "\n".join(diagnostics)[-10000:]
        if run_dir is not None:
            (run_dir / "validation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        return output
