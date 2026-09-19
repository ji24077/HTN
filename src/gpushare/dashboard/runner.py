"""Background jobs that turn the research dashboard into a real control plane.

The browser never runs a pretend timer.  Every long-running card starts one of
these jobs and polls its recorded state.  Data generation runs locally; GPU
work is copied to a selected RunPod and executed there over SSH.  Secrets are
loaded into child-process environments but are never returned by the API.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[3]
STATE_ROOT = ROOT / ".gpushare"
JOB_ROOT = STATE_ROOT / "jobs"
RUN_ROOT = STATE_ROOT / "runs"
LATEST_PATH = STATE_ROOT / "latest.json"
REMOTE_ROOT = "/workspace/gpushare-ui"

_POD_ID = re.compile(r"^[a-zA-Z0-9_-]{4,64}$")

# Imported lazily-by-value so runner has no import cycle with agent.task.
MODEL_ID_FOR_SERVE = "Qwen/Qwen2.5-0.5B"
LONG_CONTEXT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


class JobError(RuntimeError):
    """A user-facing execution error.  It is safe to return its text."""


# Job logs are child-process stdout, and _project_env() hands those children
# BASETEN_API_KEY / OPENAI_API_KEY / GPUSHARE_RUNPOD_API_KEY. One SDK traceback
# that echoes its own credential would put a live key in the browser. Two
# defences, because either alone has a gap: match the shapes we know, and blank
# any exact value currently in the environment (which catches a key shape we
# have not seen).
_SECRET_SHAPES = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}"  # OpenAI
    r"|rpa_[A-Za-z0-9]{16,}"  # RunPod
    r"|gh[pousr]_[A-Za-z0-9]{16,}"  # GitHub
    r"|[A-Za-z0-9]{8}\.[A-Za-z0-9]{24,})"  # Baseten
)
_SECRET_ENV_KEYS = (
    "BASETEN_API_KEY",
    "BASETEN_MCP_API_KEY",
    "OPENAI_API_KEY",
    "GPUSHARE_RUNPOD_API_KEY",
    "RUNPOD_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
)


@functools.lru_cache(maxsize=1)
def _known_secrets() -> tuple[str, ...]:
    """Cached: this runs per log line, and re-reading .env each time would make
    a long training log quadratic in disk I/O."""
    env = _project_env()
    # Short values would match far too much of ordinary output; a real key is
    # never this short.
    return tuple(v for k in _SECRET_ENV_KEYS if (v := env.get(k)) and len(v) >= 12)


def _redact(text: str) -> str:
    out = _SECRET_SHAPES.sub("***REDACTED***", text)
    for value in _known_secrets():
        out = out.replace(value, "***REDACTED***")
    return out


@dataclass
class Job:
    id: str
    kind: str
    params: dict[str, Any]
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False
    _process: subprocess.Popen[str] | None = field(default=None, repr=False, compare=False)

    def public(self) -> dict[str, Any]:
        # A huge training log makes polling increasingly expensive. The full
        # job record is persisted, while the browser only needs the live tail.
        return {
            "id": self.id,
            "kind": self.kind,
            "params": self.params,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "logs": self.logs[-300:],
            "result": self.result,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        JOB_ROOT.mkdir(parents=True, exist_ok=True)
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        for path in JOB_ROOT.glob("*.json"):
            try:
                raw = json.loads(path.read_text())
                raw.pop("_process", None)
                job = Job(**raw)
                if job.status in {"queued", "running", "cancelling"}:
                    job.status = "interrupted"
                    job.error = "The dashboard stopped while this job was running."
                    job.finished_at = time.time()
                self._jobs[job.id] = job
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue

    def _persist(self, job: Job) -> None:
        path = JOB_ROOT / f"{job.id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job.public(), indent=2, ensure_ascii=False))
        tmp.replace(path)

    def create(
        self,
        kind: str,
        params: dict[str, Any],
        work: Callable[[Job], dict[str, Any]],
    ) -> Job:
        # Artifact-producing work shares data/ and the selected-config record.
        # Serialising it is much clearer than letting two buttons race.
        with self._lock:
            busy = next((j for j in self._jobs.values() if j.status == "running"), None)
            if busy:
                raise JobError(f"{busy.kind} job {busy.id[:8]} is already running")
            job = Job(id=uuid.uuid4().hex, kind=kind, params=params)
            self._jobs[job.id] = job
            self._persist(job)

        thread = threading.Thread(target=self._run, args=(job, work), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job, work: Callable[[Job], dict[str, Any]]) -> None:
        with self._lock:
            job.status = "running"
            job.started_at = time.time()
            self._persist(job)
        try:
            result = work(job)
            with self._lock:
                if job.cancel_requested:
                    job.status = "cancelled"
                else:
                    job.status = "complete"
                    job.progress = 100
                    job.stage = "complete"
                    job.result = result
        except Exception as exc:  # noqa: BLE001 - this is the job boundary
            with self._lock:
                job.status = "cancelled" if job.cancel_requested else "failed"
                job.error = str(exc)
                self.log(job, f"ERROR: {exc}")
        finally:
            with self._lock:
                job.finished_at = time.time()
                job._process = None
                self._persist(job)

    def get(self, job_id: str) -> Job:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise JobError(f"unknown job {job_id}") from exc

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.public() for j in jobs[:30]]

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self.get(job_id)
            if job.status not in {"queued", "running"}:
                return job
            job.cancel_requested = True
            job.status = "cancelling"
            proc = job._process
            self._persist(job)
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return job

    def update(self, job: Job, stage: str, progress: int) -> None:
        with self._lock:
            job.stage = stage
            job.progress = max(0, min(100, progress))
            self._persist(job)

    def log(self, job: Job, text: str) -> None:
        clean = _redact(text.rstrip())
        if not clean:
            return
        with self._lock:
            job.logs.append(clean[-2000:])
            if len(job.logs) > 2000:
                del job.logs[:500]
            self._persist(job)


JOBS = JobManager()


def _project_env() -> dict[str, str]:
    env = os.environ.copy()
    env_file = ROOT / ".env"
    if env_file.exists():
        for key, value in dotenv_values(env_file).items():
            if value is not None:
                env.setdefault(key, value)
    if env.get("GPUSHARE_RUNPOD_API_KEY"):
        env.setdefault("RUNPOD_API_KEY", env["GPUSHARE_RUNPOD_API_KEY"])
    return env


def _run(
    job: Job,
    args: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    allow_failure: bool = False,
) -> int:
    JOBS.log(job, "$ " + shlex.join(args))
    proc = subprocess.Popen(
        args,
        cwd=cwd,
        env=env or _project_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    with JOBS._lock:
        job._process = proc
    assert proc.stdout is not None
    for line in proc.stdout:
        JOBS.log(job, line)
        if job.cancel_requested:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            break
    code = proc.wait()
    with JOBS._lock:
        job._process = None
    if job.cancel_requested:
        raise JobError("cancelled")
    if code and not allow_failure:
        raise JobError(f"command exited with status {code}: {args[0]}")
    return code


def _capture(args: list[str], *, timeout: int = 30) -> str:
    try:
        done = subprocess.run(
            args,
            cwd=ROOT,
            env=_project_env(),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise JobError(f"could not run {args[0]}: {exc}") from exc
    if done.returncode:
        detail = (done.stderr or done.stdout).strip()
        raise JobError(detail or f"{args[0]} exited with {done.returncode}")
    return done.stdout


def _runpodctl() -> str:
    path = shutil.which("runpodctl")
    if not path:
        raise JobError("runpodctl is not installed; run `runpodctl doctor` first")
    return path


_pod_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])
_pod_lock = threading.Lock()


def list_pods(*, refresh: bool = False) -> list[dict[str, Any]]:
    """Return a deliberately small, secret-free view of the user's pods."""
    global _pod_cache
    with _pod_lock:
        if not refresh and time.time() - _pod_cache[0] < 10:
            return _pod_cache[1]
        raw = json.loads(_capture([_runpodctl(), "pod", "list", "--all", "-o", "json"]))
        out = []
        for item in raw:
            pod_id = str(item.get("id", ""))
            if not _POD_ID.fullmatch(pod_id):
                continue
            detail = item
            try:
                detail = json.loads(
                    _capture(
                        [_runpodctl(), "pod", "get", pod_id, "--include-machine", "-o", "json"]
                    )
                )
            except JobError:
                pass
            machine = detail.get("machine") or {}
            gpu = machine.get("gpuId") or item.get("gpuId") or "Unknown GPU"
            vendor = (
                "amd" if any(x in gpu.lower() for x in ("amd", "radeon", "mi300")) else "nvidia"
            )
            out.append(
                {
                    "id": pod_id,
                    "name": item.get("name") or detail.get("name") or pod_id,
                    "status": item.get("runtimeStatus") or detail.get("runtimeStatus") or "unknown",
                    "gpu": gpu,
                    "vendor": vendor,
                    "gpu_count": item.get("gpuCount") or detail.get("gpuCount") or 1,
                    "cost_per_hour": item.get("costPerHr") or detail.get("costPerHr"),
                    "datacenter": machine.get("dataCenterId"),
                    "uptime_seconds": item.get("uptimeSeconds") or detail.get("uptimeSeconds"),
                }
            )
        _pod_cache = (time.time(), out)
        return out


def _pod(pod_id: str) -> dict[str, Any]:
    if not _POD_ID.fullmatch(pod_id):
        raise JobError("invalid pod id")
    for pod in list_pods(refresh=True):
        if pod["id"] == pod_id:
            if pod["status"] != "running":
                raise JobError(f"pod {pod_id} is {pod['status']}, not running")
            return pod
    raise JobError(f"pod {pod_id} was not found")


def _ssh_info(pod_id: str) -> dict[str, Any]:
    raw = json.loads(_capture([_runpodctl(), "ssh", "info", pod_id, "-o", "json"]))
    ip = str(raw.get("ip", ""))
    port = int(raw.get("port", 0))
    key = Path((raw.get("ssh_key") or {}).get("path", ""))
    if not ip or not 1 <= port <= 65535 or not key.is_file():
        raise JobError(f"pod {pod_id} has no usable SSH endpoint")
    return {"ip": ip, "port": port, "key": str(key)}


def _ssh_args(info: dict[str, Any], remote_command: str) -> list[str]:
    return [
        "ssh",
        "-i",
        info["key"],
        "-p",
        str(info["port"]),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"root@{info['ip']}",
        remote_command,
    ]


def _remote(job: Job, info: dict[str, Any], argv: list[str], *, allow_failure: bool = False) -> int:
    command = f'cd {shlex.quote(REMOTE_ROOT)} && export PATH="$HOME/.local/bin:$PATH" && '
    command += shlex.join(argv)
    return _run(job, _ssh_args(info, command), allow_failure=allow_failure)


def _rsync_transport(info: dict[str, Any]) -> str:
    return shlex.join(
        [
            "ssh",
            "-i",
            info["key"],
            "-p",
            str(info["port"]),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
    )


def _sync_project(job: Job, info: dict[str, Any]) -> None:
    # Create only our dedicated directory; never sync --delete into /workspace.
    _run(job, _ssh_args(info, f"mkdir -p {shlex.quote(REMOTE_ROOT)}"))
    args = [
        "rsync",
        "-az",
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=.env",
        "--exclude=.gpushare",
        "--exclude=ckpt",
        "--exclude=eval",
        "--exclude=__pycache__",
        "--exclude=*.safetensors",
        "-e",
        _rsync_transport(info),
        f"{ROOT}/",
        f"root@{info['ip']}:{REMOTE_ROOT}/",
    ]
    _run(job, args)


# A training run on a 24 GB card needs the card. Anything above this is another
# tenant — usually our own resident inference server, which holds the weights
# plus a multi-GB prefix KV cache and leaves nothing behind.
GPU_BUSY_GB = 2.0

# train.py's default; placement has to assume it because TrainRequest does not
# expose seq_len, and memory scales with it.
TRAIN_SEQ_LEN = 192

# The memory half of the cost model came in 1.4% under a measured 13.80 GB peak
# — far better than its time half, which was out by nearly 7x. Still a margin:
# one agreement on one config is not a guarantee, and being wrong here means an
# OOM two minutes in rather than a slightly late ETA.
VRAM_MARGIN = 1.20


def _free_vram_gb(info: dict[str, Any]) -> float:
    out = _capture(
        _ssh_args(info, "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
    )
    return min((int(x) for x in re.findall(r"\d+", out)), default=0) / 1024


def _gpu_occupants(info: dict[str, Any]) -> str:
    try:
        return (
            _capture(
                _ssh_args(
                    info,
                    "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader",
                )
            )
            .strip()
            .replace("\n", "; ")
        )
    except Exception:  # noqa: BLE001 — the reason to move on is the memory, not the listing
        return ""


def predicted_train_vram_gb(
    *, dtype: str, attention: str, micro_batch: int, grad_accum: int
) -> float:
    """What this training config will need, from the shared cost model.

    Built through JobConfig rather than by reaching into the formula, so a
    change to how peak memory is computed cannot silently stop applying to
    placement.
    """
    from gpushare.agent.calibrate import COLD_START, DEFAULTS
    from gpushare.agent.simulate import predict_peak_vram_gb
    from gpushare.agent.specs import MODELS
    from gpushare.contracts import JobConfig

    cfg = JobConfig(
        job_id="placement",
        model="qwen2.5-0.5b",
        total_steps=1,
        seq_len=TRAIN_SEQ_LEN,
        dtype=dtype,
        attention=attention,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        H=190,
        workers=["w0"],
        global_batch_tokens=micro_batch * grad_accum * TRAIN_SEQ_LEN,
    )
    cal = DEFAULTS.get("ada_24gb", COLD_START)
    return predict_peak_vram_gb(cfg=cfg, model=MODELS["qwen2.5-0.5b"], cal=cal) * VRAM_MARGIN


def _place(job: Job, *, need_gb: float, preferred_pod_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick a pod with room, preferring the one that was asked for.

    Placement is the product: the point of pooling GPUs is that a job lands
    where it fits without a person tracking who is holding what. Refusing and
    naming the occupant, which is what this did before, still left the person
    to do the choosing.

    Free memory is read off each GPU rather than tracked here. Our own record of
    what is loaded lives in process memory and does not survive a dashboard
    restart, and it cannot see anything we did not start — a stale server from
    an earlier session was found holding 17.1 GB this way.
    """
    pods = {p["id"]: p for p in list_pods() if p.get("status") == "running"}
    if preferred_pod_id not in pods:
        raise JobError(f"pod {preferred_pod_id} is not running")

    order = [preferred_pod_id] + sorted(
        (i for i in pods if i != preferred_pod_id),
        key=lambda i: pods[i].get("cost_per_hour") or 0.0,
    )
    surveyed: list[str] = []
    for pod_id in order:
        pod = pods[pod_id]
        try:
            info = _ssh_info(pod_id)
            free = _free_vram_gb(info)
        except Exception as e:  # noqa: BLE001 — an unreachable pod is a candidate we skip
            surveyed.append(f"{pod['name']}: unreachable ({type(e).__name__})")
            continue
        if free >= need_gb:
            where = "as requested" if pod_id == preferred_pod_id else "moved from the requested pod"
            JOBS.log(job, f"placing on {pod['name']}: {free:.1f} GB free, needs {need_gb:.1f} GB ({where})")
            return pod, info
        surveyed.append(f"{pod['name']}: {free:.1f} GB free{(' — ' + o) if (o := _gpu_occupants(info)) else ''}")

    raise JobError(
        f"no pod has room for this job (needs {need_gb:.1f} GB). "
        + " | ".join(surveyed)
        + " — stop an inference server, or rent another pod"
    )


def _require_idle_gpu(job: Job, info: dict[str, Any]) -> None:
    """Refuse rather than relocate. For jobs pinned to one pod's disk.

    Inference optimisation reads the checkpoint that training left on that
    machine, so moving it elsewhere would fail later and more confusingly than
    stopping here.
    """
    free = _free_vram_gb(info)
    if free >= GPU_BUSY_GB:
        return
    who = _gpu_occupants(info)
    raise JobError(
        f"this pod has only {free:.1f} GB free"
        + (f" ({who})" if who else "")
        + " and its checkpoint cannot be used from another pod"
        " — stop the inference server here first"
    )


def _setup_pod(job: Job, info: dict[str, Any], vendor: str) -> None:
    extra = "rocm" if vendor == "amd" else "cuda"
    bootstrap = (
        "command -v uv >/dev/null 2>&1 || python3 -m pip install --user uv; "
        'export PATH="$HOME/.local/bin:$PATH"; '
        f"cd {shlex.quote(REMOTE_ROOT)}; uv sync --extra {extra} --extra dev"
    )
    _run(job, _ssh_args(info, bootstrap))


def _pull(
    job: Job,
    info: dict[str, Any],
    remote: str,
    local: Path,
    *,
    exclude_weights: bool = False,
) -> None:
    local.mkdir(parents=True, exist_ok=True)
    excludes = ["--exclude=*.safetensors"] if exclude_weights else []
    _run(
        job,
        [
            "rsync",
            "-az",
            *excludes,
            "-e",
            _rsync_transport(info),
            f"root@{info['ip']}:{remote.rstrip('/')}/",
            f"{local}/",
        ],
    )


def _push(job: Job, info: dict[str, Any], local: Path, remote: str) -> None:
    _run(job, _ssh_args(info, f"mkdir -p {shlex.quote(remote)}"))
    src = f"{local}/" if local.is_dir() else str(local)
    dst = f"root@{info['ip']}:{remote.rstrip('/')}/"
    _run(job, ["rsync", "-az", "-e", _rsync_transport(info), src, dst])


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise JobError(f"missing or invalid result: {path}") from exc


def _quality(before: dict[str, Any], after: dict[str, Any], tol: float = 0.02) -> dict[str, Any]:
    dp = after["json_parse_rate"] - before["json_parse_rate"]
    de = after["exact_match_rate"] - before["exact_match_rate"]
    ok = dp >= -tol and de >= -tol
    return {
        "status": "ok" if ok else "regressed",
        "tolerance": tol,
        "delta_parse": dp,
        "delta_exact": de,
        "detail": "model quality preserved" if ok else "model quality regressed",
    }


def _write_latest(value: dict[str, Any]) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = LATEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(LATEST_PATH)


def latest_run() -> dict[str, Any] | None:
    try:
        return json.loads(LATEST_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def start_data_generation(*, total: int, heldout: int, workers: int) -> Job:
    if not 100 <= total <= 10_000:
        raise JobError("total records must be between 100 and 10,000")
    if not 20 <= heldout < total:
        raise JobError("held-out count must be at least 20 and smaller than total")
    if not 1 <= workers <= 16:
        raise JobError("workers must be between 1 and 16")

    def work(job: Job) -> dict[str, Any]:
        JOBS.update(job, "generating SFT data with Baseten", 10)
        _run(
            job,
            [
                sys.executable,
                "scripts/gen_data.py",
                "--n",
                str(total),
                "--heldout",
                str(heldout),
                "--workers",
                str(workers),
            ],
        )
        JOBS.update(job, "checking generated data", 90)
        train_n = sum(1 for line in (ROOT / "data/train.jsonl").read_text().splitlines() if line)
        held_n = sum(1 for line in (ROOT / "data/heldout.jsonl").read_text().splitlines() if line)
        return {"train": train_n, "heldout": held_n, "requested": total}

    return JOBS.create(
        "generate-data", {"total": total, "heldout": heldout, "workers": workers}, work
    )


def start_training(
    *,
    pod_id: str,
    steps: int,
    dtype: str,
    attention: str,
    micro_batch: int,
    grad_accum: int,
) -> Job:
    if not 10 <= steps <= 10_000:
        raise JobError("steps must be between 10 and 10,000")
    if dtype not in {"bf16", "fp16", "fp32"} or attention not in {"sdpa", "eager"}:
        raise JobError("unsupported training configuration")
    if not 1 <= micro_batch <= 128 or not 1 <= grad_accum <= 128:
        raise JobError("batch values must be between 1 and 128")

    params = {
        "pod_id": pod_id,
        "steps": steps,
        "dtype": dtype,
        "attention": attention,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
    }

    def work(job: Job) -> dict[str, Any]:
        JOBS.update(job, "choosing a GPU with room", 3)
        need = predicted_train_vram_gb(
            dtype=dtype, attention=attention, micro_batch=micro_batch, grad_accum=grad_accum
        )
        pod, info = _place(job, need_gb=need, preferred_pod_id=pod_id)
        # Everything downstream keys off where the job actually landed, not
        # where it was asked to go — including the record of which machine holds
        # the checkpoint, which serving reads back.
        placed_id = pod["id"]
        JOBS.update(job, "syncing code and data", 8)
        _sync_project(job, info)
        JOBS.update(job, "preparing GPU environment", 15)
        _setup_pod(job, info, pod["vendor"])

        remote_run = f"{REMOTE_ROOT}/.runs/{job.id}"
        eval_dir = f"{remote_run}/eval"
        ckpt_dir = f"{remote_run}/ckpt"
        _run(job, _ssh_args(info, f"mkdir -p {shlex.quote(eval_dir)} {shlex.quote(ckpt_dir)}"))

        n_held = sum(1 for line in (ROOT / "data/heldout.jsonl").read_text().splitlines() if line)
        if not n_held:
            raise JobError("data/heldout.jsonl is empty; generate data first")

        JOBS.update(job, "baseline evaluation", 22)
        _remote(
            job,
            info,
            [
                "uv",
                "run",
                "python",
                "scripts/evaluate.py",
                "--model",
                "Qwen/Qwen2.5-0.5B",
                "--out",
                f".runs/{job.id}/eval/base.json",
                "--n",
                str(n_held),
            ],
        )

        JOBS.update(job, "fine-tuning Qwen2.5-0.5B", 45)
        _remote(
            job,
            info,
            [
                "uv",
                "run",
                "python",
                "scripts/train.py",
                "--out",
                f".runs/{job.id}/ckpt",
                "--steps",
                str(steps),
                "--dtype",
                dtype,
                "--attention",
                attention,
                "--micro-batch",
                str(micro_batch),
                "--grad-accum",
                str(grad_accum),
            ],
        )

        JOBS.update(job, "automatic held-out evaluation", 78)
        _remote(
            job,
            info,
            [
                "uv",
                "run",
                "python",
                "scripts/evaluate.py",
                "--model",
                f".runs/{job.id}/ckpt",
                "--out",
                f".runs/{job.id}/eval/after.json",
                "--n",
                str(n_held),
            ],
        )

        local = RUN_ROOT / job.id
        JOBS.update(job, "downloading measurements", 92)
        _pull(job, info, f"{remote_run}/eval", local / "eval")
        # Never pull the ~1 GB weight file back merely to draw the dashboard.
        _pull(job, info, f"{remote_run}/ckpt", local / "ckpt", exclude_weights=True)

        before, after = _json(local / "eval/base.json"), _json(local / "eval/after.json")
        meta = _json(local / "ckpt/meta.json")
        validation = _quality(before, after)
        result = {
            "pod": pod,
            "local_dir": str(local),
            "remote_checkpoint": ckpt_dir,
            "before": before,
            "after": after,
            "train": meta,
            "validation": validation,
        }
        _write_latest(
            {
                "job_id": job.id,
                "pod_id": placed_id,
                "pod": pod,
                "local_dir": str(local),
                "remote_checkpoint": ckpt_dir,
            }
        )
        return result

    return JOBS.create("train-and-evaluate", params, work)


def _checkpoint_for(pod_id: str) -> dict[str, Any]:
    latest = latest_run()
    if latest and latest.get("pod_id") == pod_id:
        return latest
    # Compatibility with the first real 4090 experiment, which predates the UI.
    return {
        "pod_id": pod_id,
        "remote_checkpoint": "/workspace/gpushare/ckpt/run",
        "local_dir": str(ROOT),
        "legacy": True,
    }


def start_training_optimization(*, pod_id: str) -> Job:
    def work(job: Job) -> dict[str, Any]:
        JOBS.update(job, "choosing a GPU with room", 3)
        pod, info = _place(
            job,
            need_gb=predicted_train_vram_gb(
                dtype="bf16", attention="sdpa", micro_batch=16, grad_accum=1
            ),
            preferred_pod_id=pod_id,
        )
        JOBS.update(job, "syncing benchmark", 8)
        _sync_project(job, info)
        JOBS.update(job, "preparing GPU environment", 15)
        _setup_pod(job, info, pod["vendor"])

        # micro_batch x grad_accum is constant, so every candidate does the same
        # work per optimizer step. The actual token count is NOT written here —
        # it depends on seq_len, which is train.py's own flag. It gets read back
        # out of the measurements and checked for agreement below.
        candidates = [(16, 1), (8, 2), (4, 4)]
        remote_base = f"{REMOTE_ROOT}/.runs/{job.id}/training-bench"
        measured = []
        for idx, (mb, ga) in enumerate(candidates):
            JOBS.update(job, f"measuring batch {mb} x accum {ga}", 20 + idx * 20)
            name = f"mb{mb}-ga{ga}"
            code = _remote(
                job,
                info,
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/train.py",
                    "--out",
                    f".runs/{job.id}/training-bench/{name}",
                    "--steps",
                    "40",
                    "--warmup-measure",
                    "10",
                    "--dtype",
                    "bf16",
                    "--attention",
                    "sdpa",
                    "--micro-batch",
                    str(mb),
                    "--grad-accum",
                    str(ga),
                    "--no-save",
                ],
                allow_failure=True,
            )
            measured.append({"name": name, "micro_batch": mb, "grad_accum": ga, "code": code})

        local = RUN_ROOT / job.id / "training-bench"
        JOBS.update(job, "selecting fastest measured config", 86)
        _pull(job, info, remote_base, local)
        valid = []
        for row in measured:
            path = local / row["name"] / "meta.json"
            if row["code"] == 0 and path.exists():
                meta = _json(path)
                if math.isfinite(meta["final_loss"]):
                    valid.append(
                        {
                            **row,
                            **meta,
                            "tokens_per_second": meta["tokens_per_step"] / meta["t_step_median_s"],
                        }
                    )
        if not valid:
            raise JobError("every training candidate failed to produce a finite loss")

        # The fixed-work invariant, CHECKED rather than assumed. Comparing step
        # times across configs that moved different amounts of data would make
        # the fastest one simply the one that did least.
        counts = {row["tokens_per_step"] for row in valid}
        if len(counts) != 1:
            raise JobError(
                f"candidates did different amounts of work ({sorted(counts)} tokens/step); "
                "the speed comparison would be meaningless"
            )
        fixed_tokens = counts.pop()

        chosen = min(valid, key=lambda x: x["t_step_median_s"])
        selection = {
            "dtype": "bf16",
            "attention": "sdpa",
            "micro_batch": chosen["micro_batch"],
            "grad_accum": chosen["grad_accum"],
        }
        (STATE_ROOT / "selected-config.json").write_text(json.dumps(selection, indent=2))

        JOBS.update(job, "validating the chosen config on held-out data", 90)
        validation, quality = _validate_selection(job, info, selection, fixed_tokens)

        return {
            "pod": pod,
            "candidates": valid,
            "chosen": selection,
            "fixed_tokens_per_step": fixed_tokens,
            "quality": quality,
            "validation": validation,
        }

    return JOBS.create("optimize-training-speed", {"pod_id": pod_id}, work)


def _validate_selection(
    job: Job, info: dict[str, Any], selection: dict[str, Any], fixed_tokens: int
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Train the chosen config and score it against the reference run.

    A speed benchmark says nothing about whether the model is still right, and
    "same tokens per step, finite loss" is not evidence that it is. Until this
    function has an eval to compare, the action reports `not_validated` — never
    `ok`. A dashboard that calls an unvalidated change validated is worse than
    one that says it does not know.

    The step count is taken from the reference run: scoring a 40-step model
    against a 500-step baseline would read as a catastrophic regression caused
    by the config, when it was caused by training for a twelfth as long.
    """
    reference = latest_run()
    ref_after = Path(reference["local_dir"]) / "eval/after.json" if reference else None
    ref_meta = Path(reference["local_dir"]) / "ckpt/meta.json" if reference else None
    if not (ref_after and ref_after.exists() and ref_meta and ref_meta.exists()):
        return {
            "status": "not_validated",
            "detail": "no reference run to compare against — train once from the "
            "dashboard first, then re-run this to validate the chosen config",
        }, None

    before = _json(ref_after)
    steps = int(_json(ref_meta).get("steps", 0))
    if steps <= 0:
        return {"status": "not_validated", "detail": "reference run records no step count"}, None

    run = f"{REMOTE_ROOT}/.runs/{job.id}/validate"
    _run(job, _ssh_args(info, f"mkdir -p {shlex.quote(run)}/eval {shlex.quote(run)}/ckpt"))
    code = _remote(
        job,
        info,
        [
            "uv",
            "run",
            "python",
            "scripts/train.py",
            "--out",
            f".runs/{job.id}/validate/ckpt",
            "--steps",
            str(steps),
            "--dtype",
            selection["dtype"],
            "--attention",
            selection["attention"],
            "--micro-batch",
            str(selection["micro_batch"]),
            "--grad-accum",
            str(selection["grad_accum"]),
        ],
        allow_failure=True,
    )
    if code != 0:
        return {"status": "not_validated", "detail": f"validation training exited {code}"}, None

    code = _remote(
        job,
        info,
        [
            "uv",
            "run",
            "python",
            "scripts/evaluate.py",
            "--model",
            f".runs/{job.id}/validate/ckpt",
            "--out",
            f".runs/{job.id}/validate/eval/after.json",
        ],
        allow_failure=True,
    )
    if code != 0:
        return {"status": "not_validated", "detail": f"validation eval exited {code}"}, None

    local = RUN_ROOT / job.id / "validate"
    _pull(job, info, f"{run}/eval", local / "eval")
    after_path = local / "eval/after.json"
    if not after_path.exists():
        return {"status": "not_validated", "detail": "no eval output came back"}, None

    after = _json(after_path)
    gate = _quality(before, after)
    gate["reference_steps"] = steps
    gate["fixed_tokens_per_step"] = fixed_tokens
    gate["detail"] = (
        f"trained {steps} steps with the chosen config and scored it on the same "
        f"held-out set: {gate['detail']}"
    )
    return gate, {"reference": before, "chosen": after}


def start_inference_optimization(*, pod_id: str) -> Job:
    def work(job: Job) -> dict[str, Any]:
        pod, info = _pod(pod_id), _ssh_info(pod_id)
        checkpoint = _checkpoint_for(pod_id)
        JOBS.update(job, "checking the GPU is free", 5)
        _require_idle_gpu(job, info)
        JOBS.update(job, "syncing inference benchmark", 10)
        _sync_project(job, info)
        JOBS.update(job, "preparing GPU environment", 20)
        _setup_pod(job, info, pod["vendor"])
        JOBS.update(job, "measuring sequential and batched decoding", 45)
        _remote(
            job,
            info,
            [
                "uv",
                "run",
                "python",
                "scripts/benchmark_inference.py",
                "--model",
                checkpoint["remote_checkpoint"].replace(f"{REMOTE_ROOT}/", ""),
                "--out",
                f".runs/{job.id}/inference.json",
                "--n",
                "64",
                "--batch",
                "16",
            ],
        )
        local = RUN_ROOT / job.id
        _pull(job, info, f"{REMOTE_ROOT}/.runs/{job.id}", local)
        result = _json(local / "inference.json")
        result["pod"] = pod
        return result

    return JOBS.create("optimize-inference-speed", {"pod_id": pod_id}, work)


def start_migration(*, kind: str, source_pod_id: str, target_pod_id: str) -> Job:
    if kind not in {"migrate-amd-nvidia", "migrate-nextgen"}:
        raise JobError("unknown migration kind")
    if source_pod_id == target_pod_id:
        raise JobError("source and target pods must be different")

    params = {"source_pod_id": source_pod_id, "target_pod_id": target_pod_id}

    def work(job: Job) -> dict[str, Any]:
        source, target = _pod(source_pod_id), _pod(target_pod_id)
        if kind == "migrate-amd-nvidia" and not (
            source["vendor"] == "amd" and target["vendor"] == "nvidia"
        ):
            raise JobError("AMD→NVIDIA requires an AMD source pod and an NVIDIA target pod")
        if kind == "migrate-nextgen" and not (
            source["vendor"] == target["vendor"] == "nvidia" and source["gpu"] != target["gpu"]
        ):
            raise JobError("next-generation migration requires two different NVIDIA GPUs")

        src_info, dst_info = _ssh_info(source_pod_id), _ssh_info(target_pod_id)
        checkpoint = _checkpoint_for(source_pod_id)
        local_source = Path(checkpoint["local_dir"])
        source_eval = local_source / "eval/after.json"
        if not source_eval.exists():
            source_eval = ROOT / "eval/after.json"
        baseline = _json(source_eval)

        with tempfile.TemporaryDirectory(prefix="gpushare-migration-") as tmp:
            transfer = Path(tmp) / "ckpt"
            JOBS.update(job, "saving and downloading safetensors", 15)
            _pull(job, src_info, checkpoint["remote_checkpoint"], transfer)
            if not (transfer / "model.safetensors").exists():
                raise JobError("source checkpoint has no model.safetensors")

            JOBS.update(job, "preparing target pod", 35)
            _sync_project(job, dst_info)
            _setup_pod(job, dst_info, target["vendor"])
            target_ckpt = f"{REMOTE_ROOT}/.runs/{job.id}/ckpt"
            target_eval = f"{REMOTE_ROOT}/.runs/{job.id}/eval"
            JOBS.update(job, "transferring checkpoint", 55)
            _push(job, dst_info, transfer, target_ckpt)
            _push(job, dst_info, source_eval, target_eval)

        n = int(baseline["n"])
        result_name = f"after-{target['vendor']}.json"
        JOBS.update(job, "automatic validation on target GPU", 78)
        _remote(
            job,
            dst_info,
            [
                "uv",
                "run",
                "python",
                "scripts/evaluate.py",
                "--model",
                f".runs/{job.id}/ckpt",
                "--out",
                f".runs/{job.id}/eval/{result_name}",
                "--n",
                str(n),
            ],
        )
        local = RUN_ROOT / job.id
        _pull(job, dst_info, target_eval, local / "eval")
        after = _json(local / "eval" / result_name)
        validation = _quality(baseline, after)
        _write_latest(
            {
                "job_id": job.id,
                "pod_id": target_pod_id,
                "pod": target,
                "local_dir": str(local),
                "remote_checkpoint": target_ckpt,
                "migrated_from": source_pod_id,
            }
        )
        return {
            "source": source,
            "target": target,
            "before": baseline,
            "after": after,
            "remote_checkpoint": target_ckpt,
            "validation": validation,
        }

    return JOBS.create(kind, params, work)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive inference: one resident model, reached through an SSH forward
# ─────────────────────────────────────────────────────────────────────────────
SERVE_PORT = 8100
_serve_lock = threading.Lock()
_serve: dict[str, Any] = {}


def available_models() -> list[dict[str, Any]]:
    """The three things a person actually wants to compare.

    `base` is the untrained model and is listed first on purpose: it is the
    control. A before/after claim with no before is not a claim, and the run we
    have scores base at json_parse_rate 0.000 — which is what makes the trained
    number mean anything.
    """
    out = [
        {
            "id": "base",
            "label": "Qwen2.5-0.5B (before training)",
            "ref": MODEL_ID_FOR_SERVE,
            "kind": "base",
            "detail": "has never seen the JSON format",
        }
    ]
    # The long-context serving model. It is NOT the fine-tuned task model and is
    # not compared against it on accuracy: it is here because prefill cost scales
    # with parameters x context, and 0.5B cannot produce a prefill worth
    # optimising — measured, it tops out near 0.5s at its 32K ceiling. Quality
    # claims stay with the 0.5B run; this one carries latency claims only.
    out.append(
        {
            "id": "longctx",
            "label": "Qwen3-4B Instruct (long-context serving)",
            "ref": LONG_CONTEXT_MODEL,
            "kind": "longctx",
            "detail": "40K context - for prefill/prefix-cache measurement, not accuracy",
        }
    )
    latest = latest_run()
    if latest and latest.get("remote_checkpoint"):
        out.append(
            {
                "id": "finetuned",
                "label": "fine-tuned (latest dashboard run)",
                "ref": latest["remote_checkpoint"],
                "kind": "finetuned",
                "job_id": latest.get("job_id"),
                "detail": "trained from this dashboard",
            }
        )
    else:
        # The first real 4090 experiment predates the UI and lives outside it.
        out.append(
            {
                "id": "finetuned",
                "label": "fine-tuned (first 4090 experiment)",
                "ref": "/workspace/gpushare/ckpt/run",
                "kind": "finetuned",
                "legacy": True,
                "detail": "500 steps, json_parse_rate 1.000 / exact_match 0.910",
            }
        )
    return out


def _model_ref(model_id: str, pod_id: str) -> str:
    for m in available_models():
        if m["id"] == model_id:
            return m["ref"]
    raise JobError(f"unknown model {model_id!r}")


def _tunnel_up(timeout: float = 3.0) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{SERVE_PORT}/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def stop_inference_server() -> dict[str, Any]:
    with _serve_lock:
        for key in ("tunnel", "remote"):
            proc = _serve.pop(key, None)
            if proc is not None:
                with contextlib.suppress(Exception):
                    proc.terminate()
        was = _serve.pop("model_id", None)
        _serve.clear()
    return {"stopped": was}


def serving() -> dict[str, Any]:
    """What the chat box is currently talking to, if anything."""
    with _serve_lock:
        if not _serve.get("model_id"):
            return {"running": False}
        return {
            "running": _tunnel_up(timeout=1.0),
            "model_id": _serve["model_id"],
            "model_ref": _serve["model_ref"],
            "pod_id": _serve["pod_id"],
            "dtype": _serve["dtype"],
        }


def start_inference_server(*, pod_id: str, model_id: str, dtype: str = "bf16") -> Job:
    """Load one model on the pod and hold it there.

    Loading Qwen costs ten to twenty seconds. Paying that per message would
    make the chat unusable AND would drown the latency number we want to show,
    so the model stays resident and the dashboard reaches it over an SSH local
    forward — the pod exposes port 22 and nothing else.
    """
    ref = _model_ref(model_id, pod_id)

    def work(job: Job) -> dict[str, Any]:
        stop_inference_server()
        pod, info = _pod(pod_id), _ssh_info(pod_id)
        JOBS.update(job, "syncing serve script", 10)
        _sync_project(job, info)
        JOBS.update(job, "preparing GPU environment", 25)
        _setup_pod(job, info, pod["vendor"])

        remote_ref = ref if ref.startswith("/") else ref
        # The kill runs in its OWN ssh call, and that separation is the whole
        # point. Folded into the launch command, `pkill -f` matches against the
        # full command line — which contains "scripts/serve.py" in the launch
        # half — so the wrapper shell kills itself and ssh returns 255 with no
        # output. The usual `[s]cripts` bracket trick does NOT save it here,
        # because the un-bracketed spelling is still present further along the
        # same line. Both were observed; this is the only shape that works.
        JOBS.update(job, "stopping any previous server", 40)
        _run(
            job,
            # The bracket stops this command's OWN argv from matching its own
            # pattern — which only works because the launch, which spells it
            # unbracketed, is a separate call. Verified against a live server:
            # 2 processes before, 0 after.
            _ssh_args(info, 'pkill -f "serve[.]py" || true'),
            allow_failure=True,
        )

        # setsid detaches so ssh can return; stdin must go to /dev/null as well
        # as stdout and stderr, or ssh waits on the inherited descriptor forever.
        cmd = (
            f'export PATH="$HOME/.local/bin:$PATH"; cd {shlex.quote(REMOTE_ROOT)}; '
            f"setsid nohup uv run python scripts/serve.py --model {shlex.quote(remote_ref)} "
            f"--dtype {shlex.quote(dtype)} --port {SERVE_PORT} "
            f"< /dev/null > /tmp/serve.log 2>&1 & echo started"
        )
        JOBS.update(job, f"loading {model_id} on the GPU", 45)
        _run(job, _ssh_args(info, cmd))

        JOBS.update(job, "opening SSH forward", 65)
        tunnel = subprocess.Popen(
            [
                "ssh",
                "-i",
                info["key"],
                "-p",
                str(info["port"]),
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ExitOnForwardFailure=yes",
                "-N",
                "-L",
                f"{SERVE_PORT}:127.0.0.1:{SERVE_PORT}",
                f"root@{info['ip']}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # The model load dominates; poll rather than guess a sleep.
        JOBS.update(job, "waiting for the model to finish loading", 80)
        for _attempt in range(60):
            if job.cancel_requested:
                tunnel.terminate()
                raise JobError("cancelled")
            if _tunnel_up(timeout=2.0):
                break
            time.sleep(2)
        else:
            tunnel.terminate()
            tail = _capture(_ssh_args(info, "tail -20 /tmp/serve.log"), timeout=20)
            raise JobError(f"model never became ready. serve.log:\n{tail}")

        with _serve_lock:
            _serve.update(
                tunnel=tunnel,
                model_id=model_id,
                model_ref=ref,
                pod_id=pod_id,
                dtype=dtype,
            )
        return {
            "pod": pod,
            "model_id": model_id,
            "model_ref": ref,
            "dtype": dtype,
            "port": SERVE_PORT,
        }

    return JOBS.create("serve-model", {"pod_id": pod_id, "model_id": model_id}, work)


def generate_stream(
    *, sentence: str, max_new_tokens: int = 64, greedy: bool = True, no_cache: bool = False
):
    """Proxy the pod's SSE stream through, one frame at a time.

    Every hop has to stay unbuffered or the feature is cosmetic: the pod sends
    no Content-Length, the SSH forward is raw TCP and transparent, and the
    caller must hand each line onward rather than collecting them. Reading the
    whole response here — the obvious `r.read()` — would rebuild exactly the
    blocking behaviour this exists to remove.
    """
    import urllib.error
    import urllib.request

    with _serve_lock:
        state = dict(_serve)
    if not state.get("model_id"):
        raise JobError("no model is loaded — start one from the model picker first")

    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVE_PORT}/generate/stream",
        data=json.dumps(
            {
                "sentence": sentence,
                "max_new_tokens": max_new_tokens,
                "greedy": greedy,
                "no_cache": no_cache,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        # Matches the blocking path: a cache-bypassed long-context request
        # re-runs the full prefill, which outlasts the old 180s budget.
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data: "):
                    yield line[6:]
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        yield json.dumps({"done": True, "error": f"the model server is unreachable ({e})"})


def set_prefix(*, prefix: str) -> dict[str, Any]:
    """Install (or clear) the cached static head of the prompt.

    Building the cache runs one full prefill, so the call takes as long as a
    cold request does — that returned `build_s` IS the cold baseline, which is
    why it is measured here rather than assumed from an earlier run.
    """
    import urllib.error
    import urllib.request

    with _serve_lock:
        state = dict(_serve)
    if not state.get("model_id"):
        raise JobError("no model is loaded — start one from the model picker first")

    body = json.dumps({"prefix": prefix}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVE_PORT}/prefix",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        # Generous: a 30K-token prefill on a cheap GPU is the slow case this
        # whole feature exists to remove, so the timeout must outlast it.
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise JobError(f"prefix failed: {e.read().decode()[:300]}") from e
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise JobError(f"the model server is unreachable ({e}); restart it") from e


def generate(
    *, sentence: str, max_new_tokens: int = 64, greedy: bool = True, no_cache: bool = False
) -> dict[str, Any]:
    """One interactive request against the resident model.

    `latency_s` is what a person feels at batch 1. It is NOT evidence about the
    inference optimization, which is a concurrency win — at batch 1 there is
    nothing to batch. benchmark_inference.py is what measures that, at 64
    concurrent, and the UI must keep the two claims apart.
    """
    import urllib.error
    import urllib.request

    with _serve_lock:
        state = dict(_serve)
    if not state.get("model_id"):
        raise JobError("no model is loaded — start one from the model picker first")

    body = json.dumps(
        {
            "sentence": sentence,
            "max_new_tokens": max_new_tokens,
            "greedy": greedy,
            "no_cache": no_cache,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVE_PORT}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        # 120s was sized for a short prompt. A cache-bypassed long-context
        # request re-runs the whole prefill and legitimately takes far longer.
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise JobError(f"generate failed: {e.read().decode()[:300]}") from e
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise JobError(f"the model server is unreachable ({e}); restart it") from e

    # Server-side generation time vs what the round trip cost. Showing only the
    # first would hide the SSH hop; showing only the second would blame the GPU
    # for the network.
    out["roundtrip_s"] = time.perf_counter() - started
    out["model_id"] = state["model_id"]
    return out
