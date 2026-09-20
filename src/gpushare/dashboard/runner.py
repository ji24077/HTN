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
import logging
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from gpushare.agent.profiles import profile_for
from gpushare.agent.sixseven import PROMPT as SIXSEVEN_PROMPT
from gpushare.agent.task import PROMPT as TASK_PROMPT

ROOT = Path(__file__).resolve().parents[3]
STATE_ROOT = ROOT / ".gpushare"
JOB_ROOT = STATE_ROOT / "jobs"
RUN_ROOT = STATE_ROOT / "runs"
LATEST_PATH = STATE_ROOT / "latest.json"
SAVED_PATH = STATE_ROOT / "saved-models.json"
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
                raw = json.loads(path.read_text(encoding="utf-8"))
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
        # Windows readers, antivirus, and sync clients can briefly deny replacement.
        # Keep the old complete record until a closed, uniquely named file is ready.
        tmp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=JOB_ROOT,
                prefix=f".{job.id}-", suffix=".tmp", delete=False,
            ) as stream:
                tmp = Path(stream.name)
                json.dump(job.public(), stream, indent=2, ensure_ascii=False)
            for attempt in range(8):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(min(0.05 * 2**attempt, 0.5))
        finally:
            if tmp is not None:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)

    def create(
        self,
        kind: str,
        params: dict[str, Any],
        work: Callable[[Job], dict[str, Any]],
    ) -> Job:
        # Artifact-producing work shares data/ and the selected-config record.
        # Serialising it is much clearer than letting two buttons race.
        with self._lock:
            # "running" alone leaves a window: a fresh Job defaults to
            # "queued" and only flips to "running" inside _run(), which has
            # to re-acquire this same lock from its own thread — so a second
            # create() call arriving before that thread runs would see no
            # "running" job and pass the check too. Both jobs then launch GPU
            # work on the same pod concurrently. Counting "queued" here closes
            # that window, since the new job is inserted under this same lock
            # before it is released.
            busy = next((j for j in self._jobs.values() if j.status in ("running", "queued")), None)
            if busy:
                raise JobError(f"{busy.kind} job {busy.id[:8]} is already running")
            job = Job(id=uuid.uuid4().hex, kind=kind, params=params)
            self._jobs[job.id] = job
            try:
                self._persist(job)
            except Exception:
                del self._jobs[job.id]
                raise

        thread = threading.Thread(target=self._run, args=(job, work), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job, work: Callable[[Job], dict[str, Any]]) -> None:
        try:
            with self._lock:
                job.status = "running"
                job.started_at = time.time()
                self._persist(job)
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
                job.error = _redact(str(exc))
                # The original exception may itself be a persistence failure.
                # Append in memory; only the terminal write below should persist.
                job.logs.append(f"ERROR: {job.error}"[-2000:])
        finally:
            with self._lock:
                job.finished_at = time.time()
                job._process = None
                try:
                    self._persist(job)
                except Exception as exc:  # noqa: BLE001 - preserve terminal state
                    message = _redact(f"Could not persist final job state: {exc}")
                    job.status = "failed"
                    job.stage = "failed"
                    job.error = f"{job.error}; {message}" if job.error else message
                    job.logs.append(f"ERROR: {message}"[-2000:])
                    # If storage recovers, record the failed status, not a stale
                    # running/success record. A permanent failure stays visible in
                    # memory and server logs without recursive error handling.
                    try:
                        self._persist(job)
                    except Exception:  # noqa: BLE001 - no further writes can help
                        logging.getLogger(__name__).error("Job %s: %s", job.id, message)

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
                _terminate_local_process(proc)
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


def _terminate_local_process(proc) -> None:
    if os.name == "nt":
        proc.terminate()
    else:
        os.killpg(proc.pid, signal.SIGTERM)


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
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=os.name != "nt",
    )
    with JOBS._lock:
        job._process = proc
    assert proc.stdout is not None
    for line in proc.stdout:
        JOBS.log(job, line)
        if job.cancel_requested:
            try:
                _terminate_local_process(proc)
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
            encoding="utf-8",
            errors="replace",
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
    key = Path(_project_env().get("GPUSHARE_SSH_KEY") or (raw.get("ssh_key") or {}).get("path", ""))
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
        "-o",
        f"UserKnownHostsFile={_known_hosts_path()}",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        f"root@{info['ip']}",
        remote_command,
    ]


def _known_hosts_path() -> str:
    path = Path(_project_env().get("GPUSHARE_SSH_KNOWN_HOSTS") or STATE_ROOT / "known_hosts")
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())


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
            "-o",
            f"UserKnownHostsFile={_known_hosts_path()}",
        ]
    )


def _sync_project(job: Job, info: dict[str, Any]) -> None:
    # Create only our dedicated directory; never sync --delete into /workspace.
    _run(job, _ssh_args(info, f"mkdir -p {shlex.quote(REMOTE_ROOT)}"))
    if not shutil.which("rsync") or os.name == "nt":
        with tempfile.TemporaryDirectory(prefix="gpushare-project-") as tmp:
            archive = Path(tmp) / "project.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for top in (
                    "src",
                    "scripts",
                    "data",
                    "pyproject.toml",
                    "uv.lock",
                    ".python-version",
                ):
                    path = ROOT / top
                    paths = path.rglob("*") if path.is_dir() else [path]
                    for item in paths:
                        relative = item.relative_to(ROOT)
                        if (
                            not item.is_file()
                            or item.is_symlink()
                            or "__pycache__" in relative.parts
                            or item.suffix in {".pyc", ".safetensors"}
                            or item.name.startswith(".env")
                        ):
                            continue
                        tar.add(item, arcname=relative.as_posix(), recursive=False)
            remote_archive = f"/tmp/gpushare-project-{uuid.uuid4().hex}.tar.gz"
            _run(job, _scp_args(info, str(archive), _scp_remote(info, remote_archive)))
            command = (
                f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(REMOTE_ROOT)}"
                f" && rm -f {shlex.quote(remote_archive)}"
            )
            _run(job, _ssh_args(info, command))
        return
    args = [
        "rsync",
        "-az",
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=.env",
        "--exclude=.env*",
        "--exclude=.gpushare",
        "--exclude=.cache",
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


# A 0.5B checkpoint is ~1.2 GB, and the base model has to be fetched before it
# can be written. Both land on the same overlay, so placement has to clear the
# sum with room to spare: a run that trains for two minutes and then cannot
# serialise has cost the GPU time for nothing.
DISK_NEED_GB = 6.0


def _free_disk_gb(info: dict[str, Any], path: str = REMOTE_ROOT) -> float:
    out = _capture(_ssh_args(info, f"df -BM --output=avail {shlex.quote(path)} 2>/dev/null | tail -1"))
    return max((int(x) for x in re.findall(r"\d+", out)), default=0) / 1024


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


def _reclaim(job: Job, info: dict[str, Any], pod: dict[str, Any]) -> bool:
    """Free a pod by dropping what this dashboard put there. Never anything else.

    Two reclaimable things, in the order they cost least:

    - Checkpoints from earlier runs. Only the newest is ever read back, so the
      rest are ~1.2 GB each of nothing. The newest is kept because serving
      resolves the model through it.
    - Our own resident inference server. This is the destructive one: it is very
      likely the model someone is demonstrating, which is why it is a last
      resort rather than a routine "clear the GPU before every job".

    A process we did not start is left alone. On rented hardware that could be
    another tenant, and killing it would be neither ours to do nor recoverable.
    """
    freed = False
    latest = (latest_run() or {}).get("job_id", "")
    stale = _capture(
        _ssh_args(
            info,
            f"ls -1d {REMOTE_ROOT}/.runs/*/ckpt 2>/dev/null | grep -v {shlex.quote(latest or 'none')} || true",
        )
    ).split()
    if stale:
        _capture(_ssh_args(info, "rm -rf " + " ".join(shlex.quote(d) for d in stale)))
        JOBS.log(job, f"reclaimed {len(stale)} old checkpoint(s) on {pod['name']}")
        freed = True

    if _capture(_ssh_args(info, 'pgrep -f "serve[.]py" || true')).strip():
        JOBS.log(
            job,
            f"stopping the inference server on {pod['name']} — no other pod had room. "
            "Reload the model from the picker when the job finishes.",
        )
        _capture(_ssh_args(info, 'pkill -f "serve[.]py" || true'))
        time.sleep(3)  # the allocator returns the memory to the driver on exit
        freed = True
    return freed


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
            free, disk = _free_vram_gb(info), _free_disk_gb(info)
        except Exception as e:  # noqa: BLE001 — an unreachable pod is a candidate we skip
            surveyed.append(f"{pod['name']}: unreachable ({type(e).__name__})")
            continue
        if free >= need_gb and disk >= DISK_NEED_GB:
            where = "as requested" if pod_id == preferred_pod_id else "moved from the requested pod"
            JOBS.log(
                job,
                f"placing on {pod['name']}: {free:.1f} GB VRAM and {disk:.1f} GB disk free, "
                f"needs {need_gb:.1f} GB / {DISK_NEED_GB:.0f} GB ({where})",
            )
            return pod, info
        if free < need_gb:
            surveyed.append(
                f"{pod['name']}: {free:.1f} GB VRAM free"
                + (f" — {o}" if (o := _gpu_occupants(info)) else "")
            )
        else:
            # Named separately because the fix is different: this pod has the
            # GPU but not the room to write the result.
            surveyed.append(f"{pod['name']}: only {disk:.1f} GB disk free")

    # Nothing fits as-is. Only now is it worth taking something away, and the
    # requested pod goes first: if the user has to lose a loaded model, lose the
    # one on the machine they actually asked for.
    JOBS.log(job, "no pod had room as-is — reclaiming space")
    for pod_id in order:
        pod = pods[pod_id]
        try:
            info = _ssh_info(pod_id)
            if not _reclaim(job, info, pod):
                continue
            free, disk = _free_vram_gb(info), _free_disk_gb(info)
        except Exception:  # noqa: BLE001 — a pod that fails to clear is one we skip
            continue
        if free >= need_gb and disk >= DISK_NEED_GB:
            JOBS.log(job, f"placing on {pod['name']} after reclaiming: {free:.1f} GB VRAM, {disk:.1f} GB disk")
            return pod, info

    raise JobError(
        f"no pod has room for this job (needs {need_gb:.1f} GB VRAM, {DISK_NEED_GB:.0f} GB disk), "
        "even after reclaiming. "
        + " | ".join(surveyed)
        + " — rent another pod, or use a smaller batch"
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
    extra = profile_for(vendor).project_extra
    bootstrap = (
        "command -v uv >/dev/null 2>&1 || python3 -m pip install --user uv; "
        'export PATH="$HOME/.local/bin:$PATH"; '
        f"cd {shlex.quote(REMOTE_ROOT)}; uv sync --extra {extra} --extra dev"
    )
    _run(job, _ssh_args(info, bootstrap))


def _serve_python(vendor: str) -> str:
    """How to invoke python for serving on this vendor.

    AMD cannot use the uv path: the image has no uv, `pip install --user uv`
    is refused by Ubuntu 24.04's PEP 668 guard, and this project's lock pins
    the rocm7.0 wheel index against a 7.1.1 host. The migration venv already
    solves exactly this — matched torch from rocm7.1 — so serving borrows it.

    NVIDIA keeps `uv run`, which is what the working demo uses; there is no
    reason to move it and a live path to break if it moves.
    """
    return profile_for(vendor).serve_python


def _setup_migration_pod(job: Job, info: dict[str, Any], vendor: str) -> None:
    """Isolated matched versions; never resolve the regular CUDA/ROCm lock."""
    wheel = profile_for(vendor).wheel_index
    commands = [
        ["python3", "-m", "venv", ".migration-venv"],
        [
            ".migration-venv/bin/python",
            "-m",
            "pip",
            "install",
            "torch==2.10.0",
            "--index-url",
            f"https://download.pytorch.org/whl/{wheel}",
        ],
        [
            ".migration-venv/bin/python",
            "-m",
            "pip",
            "install",
            "transformers==5.17.0",
            "peft==0.21.0",
            "safetensors",
            "numpy",
            "pydantic>=2.9",
            "pydantic-settings>=2.6",
        ],
        [".migration-venv/bin/python", "-m", "pip", "install", "--no-deps", "-e", "."],
    ]
    for argv in commands:
        _remote(job, info, argv)


def _pull(
    job: Job,
    info: dict[str, Any],
    remote: str,
    local: Path,
    *,
    exclude_weights: bool = False,
) -> None:
    local.mkdir(parents=True, exist_ok=True)
    if not shutil.which("rsync") or os.name == "nt":
        remote_archive = f"/tmp/gpushare-transfer-{uuid.uuid4().hex}.tar.gz"
        exclude = " --exclude='*.safetensors'" if exclude_weights else ""
        command = f"tar -czf {shlex.quote(remote_archive)}{exclude} -C {shlex.quote(remote)} ."
        _run(job, _ssh_args(info, command))
        with tempfile.TemporaryDirectory(prefix="gpushare-download-") as tmp:
            archive = Path(tmp) / "download.tar.gz"
            _run(job, _scp_args(info, _scp_remote(info, remote_archive), str(archive)))
            _extract_transfer_archive(archive, local)
        _run(job, _ssh_args(info, f"rm -f {shlex.quote(remote_archive)}"))
        return
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
    if not shutil.which("rsync") or os.name == "nt":
        if local.is_file():
            _run(job, _scp_args(info, str(local), _scp_remote(info, remote.rstrip("/") + "/")))
            return
        with tempfile.TemporaryDirectory(prefix="gpushare-upload-") as tmp:
            archive = Path(tmp) / "upload.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for path in sorted(local.rglob("*")):
                    if path.is_symlink():
                        raise JobError("checkpoint transfer cannot contain symbolic links")
                    if path.is_file():
                        tar.add(path, arcname=path.relative_to(local).as_posix(), recursive=False)
            remote_archive = f"/tmp/gpushare-transfer-{uuid.uuid4().hex}.tar.gz"
            _run(job, _scp_args(info, str(archive), _scp_remote(info, remote_archive)))
            _run(
                job,
                _ssh_args(
                    info,
                    f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote)}"
                    f" && rm -f {shlex.quote(remote_archive)}",
                ),
            )
        return
    src = f"{local}/" if local.is_dir() else str(local)
    dst = f"root@{info['ip']}:{remote.rstrip('/')}/"
    _run(job, ["rsync", "-az", "-e", _rsync_transport(info), src, dst])


def _scp_args(info: dict[str, Any], source: str, target: str) -> list[str]:
    # Native OpenSSH accepts argument arrays on Windows; no local shell/rsync.
    return [
        "scp",
        "-i",
        info["key"],
        "-P",
        str(info["port"]),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={_known_hosts_path()}",
        "-o",
        "ConnectTimeout=20",
        source,
        target,
    ]


def _scp_remote(info: dict[str, Any], path: str) -> str:
    # Transfer paths are generated internally and intentionally have no spaces
    # or shell metacharacters; this also works with legacy SCP implementations.
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", path):
        raise JobError("remote transfer path must be an absolute simple POSIX path")
    return f"root@{info['ip']}:{path}"


def _extract_transfer_archive(archive: Path, local: Path) -> None:
    root = local.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or not (member.isdir() or member.isfile()):
                raise JobError("unsafe path or link in checkpoint transfer archive")
        for member in members:
            target = root / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = tar.extractfile(member)
                if stream is None:
                    raise JobError("unreadable file in checkpoint transfer archive")
                with stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output)


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JobError(f"missing or invalid result: {path}") from exc


@dataclass(frozen=True)
class TaskSpec:
    """What an agent needs to know to optimise a task rather than a guess.

    The two optimisation agents used to name `scripts/train.py` and
    `scripts/evaluate.py` with no data flag at all, so they always measured the
    extraction set — whatever model you pointed them at. Running one against
    the 6-7 checkpoint produced real numbers about a workload that had nothing
    to do with it, which is worse than no numbers.

    `metrics` is the part a gate reads. The two tasks do not share a scoring
    vocabulary — one reports json_parse_rate, the other trigger_accuracy — and
    a delta over the wrong key silently reads as zero, which passes.
    """

    name: str
    train_data: str
    heldout_data: str
    evaluator: str
    prompt: str
    metrics: tuple[str, ...]


TASKS: dict[str, TaskSpec] = {
    "extraction": TaskSpec(
        name="extraction",
        train_data="data/train.jsonl",
        heldout_data="data/heldout.jsonl",
        evaluator="scripts/evaluate.py",
        prompt=TASK_PROMPT,
        metrics=("json_parse_rate", "exact_match_rate"),
    ),
    "sixseven": TaskSpec(
        name="sixseven",
        train_data="data/sixseven-train.jsonl",
        heldout_data="data/sixseven-heldout.jsonl",
        evaluator="scripts/eval_sixseven.py",
        prompt=SIXSEVEN_PROMPT,
        metrics=("accuracy", "trigger_accuracy", "non_trigger_accuracy"),
    ),
}


def task_for(name: str) -> TaskSpec:
    """Refuse an unknown task rather than quietly measuring the default one."""
    try:
        return TASKS[name]
    except KeyError:
        known = ", ".join(sorted(TASKS))
        raise JobError(f"no task named {name!r} — known: {known}") from None


def _task_name_for_prompt(prompt: str) -> str:
    """Which task owns this prompt shape. Unknown shapes stay on extraction,
    which is what every model in the registry predates the flag as."""
    for spec in TASKS.values():
        if spec.prompt == prompt:
            return spec.name
    return "extraction"


def _quality(
    before: dict[str, Any], after: dict[str, Any], tol: float = 0.02, *, task: str = "extraction"
) -> dict[str, Any]:
    """Did this action leave the model as good as it was, on ITS OWN metrics."""
    spec = task_for(task)
    # A missing metric is not a zero. Treating it as one turns "these two runs
    # measured different things" into a confident verdict: scoring a 6-7 run
    # against the extraction baseline produced deltas identical to the raw
    # after-values, and a field_accuracy the 6-7 evaluator never writes came
    # out as -0.985 and failed the gate. The reference run is simply whichever
    # training ran last, which says nothing about which task it was.
    missing = [k for k in spec.metrics if before.get(k) is None or after.get(k) is None]
    if missing:
        return {
            "status": "not_comparable",
            "task": spec.name,
            "tolerance": tol,
            "deltas": {},
            "delta_fields": {},
            "missing_metrics": missing,
            "detail": (
                f"the two runs do not report the same metrics ({', '.join(missing)}); "
                f"compare against a {spec.name} run"
            ),
        }
    deltas = {key: after[key] - before[key] for key in spec.metrics}
    before_fields = before.get("field_accuracy", {}) or {}
    after_fields = after.get("field_accuracy", {}) or {}
    # Only fields both runs scored. One side reporting per-field accuracy and
    # the other not is the cross-task case above, not a regression.
    field_deltas = {
        key: after_fields[key] - value
        for key, value in before_fields.items()
        if key in after_fields
    }
    # The epsilon is not slack in the policy, it is the difference between a
    # policy and its floating-point shadow: 0.98 - 1.00 is -0.020000000000000018,
    # which is "worse than 2 points" only to a computer, and a gate whose
    # verdict turns on the seventeenth decimal is not one anybody can reason
    # about.
    floor = -tol - 1e-9
    ok = all(d >= floor for d in deltas.values()) and all(
        d >= floor for d in field_deltas.values()
    )
    out = {
        "status": "ok" if ok else "regressed",
        "task": spec.name,
        "tolerance": tol,
        "deltas": deltas,
        "delta_fields": field_deltas,
        "detail": "model quality preserved" if ok else "model quality regressed",
    }
    # The extraction gate has been reporting these two names since it was
    # written, and the dashboard reads them by name.
    if spec.name == "extraction":
        out["delta_parse"] = deltas.get("json_parse_rate", 0.0)
        out["delta_exact"] = deltas.get("exact_match_rate", 0.0)
    return out


def _write_latest(value: dict[str, Any]) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = LATEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(LATEST_PATH)


def latest_run() -> dict[str, Any] | None:
    try:
        return json.loads(LATEST_PATH.read_text(encoding="utf-8"))
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
    save_as: str = "",
    base: str = MODEL_ID_FOR_SERVE,
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
        # Saved last, and only on success: a name in the picker should mean a
        # checkpoint that finished and passed its own evaluation, never one a
        # run was part way through writing.
        if save_as:
            result["saved"] = save_model(
                name=save_as,
                ref=ckpt_dir,
                kind="trained",
                pod_id=placed_id,
                base=base,
                metrics={
                    "exact_match_rate": (after or {}).get("exact_match_rate"),
                    "json_parse_rate": (after or {}).get("json_parse_rate"),
                    "held_out_loss": (after or {}).get("held_out_loss"),
                    "steps": steps,
                },
            )
        return result

    return JOBS.create("train-and-evaluate", params, work)


def _checkpoint_for(pod_id: str) -> dict[str, Any]:
    """Where this pod's trained weights are, or a refusal.

    There used to be a fallback here to a hardcoded path from the first
    experiment. On any pod that never ran it, that path does not exist, and
    transformers reads a missing local path as a Hugging Face repo id — so the
    failure surfaced minutes later as "Repo id must be in the form
    'namespace/repo_name'", which says nothing about the actual problem. A
    checkpoint that is not there is worth saying plainly.
    """
    latest = latest_run()
    if latest and latest.get("pod_id") == pod_id:
        return latest
    where = (latest or {}).get("pod_id")
    raise JobError(
        f"pod {pod_id} has no trained checkpoint"
        + (f" — the last run left one on {where}; pick that pod" if where
           else " — run training first")
    )


def start_training_optimization(*, pod_id: str, task: str = "extraction") -> Job:
    spec = task_for(task)

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
                    "--data",
                    spec.train_data,
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
        validation, quality = _validate_selection(job, info, selection, fixed_tokens, spec)

        # The validation run is a complete re-train from the same base model,
        # not merely a benchmark. Keep its checkpoint address so the guided UI
        # can serve the agent-trained model beside the manual baseline.
        remote_checkpoint = (
            f"{REMOTE_ROOT}/.runs/{job.id}/validate/ckpt" if quality is not None else None
        )

        return {
            "pod": pod,
            "candidates": valid,
            "chosen": selection,
            "fixed_tokens_per_step": fixed_tokens,
            "quality": quality,
            "validation": validation,
            "remote_checkpoint": remote_checkpoint,
        }

    return JOBS.create("optimize-training-speed", {"pod_id": pod_id}, work)


def _validate_selection(
    job: Job,
    info: dict[str, Any],
    selection: dict[str, Any],
    fixed_tokens: int,
    spec: TaskSpec,
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
            "--data",
            spec.train_data,
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
            spec.evaluator,
            "--model",
            f".runs/{job.id}/validate/ckpt",
            "--data",
            spec.heldout_data,
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
    gate = _quality(before, after, task=spec.name)
    if gate["status"] == "not_comparable":
        # This function's whole rule: never call something validated that was
        # not. The reference run is whichever training ran last, which may
        # belong to another task entirely, and there is no verdict to give.
        return {"status": "not_validated", "detail": gate["detail"]}, None
    gate["reference_steps"] = steps
    gate["fixed_tokens_per_step"] = fixed_tokens
    gate["detail"] = (
        f"trained {steps} steps with the chosen config and scored it on the same "
        f"held-out set: {gate['detail']}"
    )
    return gate, {"reference": before, "chosen": after}


def start_inference_optimization(
    *, pod_id: str, task: str = "extraction", model_id: str = ""
) -> Job:
    spec = task_for(task)

    def work(job: Job) -> dict[str, Any]:
        pod, info = _pod(pod_id), _ssh_info(pod_id)
        # Named model first. Asking latest.json instead means asking "what did
        # the last training run leave behind", which is a different question
        # from "which model am I optimising" the moment there is more than one
        # — and it is the question that hid a saved 6-7 checkpoint sitting on
        # the very pod being measured.
        if model_id:
            remote_ref = _model_ref(model_id, pod_id)
        else:
            remote_ref = _checkpoint_for(pod_id)["remote_checkpoint"]
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
                remote_ref.replace(f"{REMOTE_ROOT}/", ""),
                "--data",
                spec.heldout_data,
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
        result["task"] = spec.name
        return result

    return JOBS.create("optimize-inference-speed", {"pod_id": pod_id}, work)


def start_migration(
    *,
    kind: str,
    source_pod_id: str,
    target_pod_id: str,
    total_steps: int = 8,
    stop_after: int = 4,
    eval_n: int = 50,
    initial_adapter: str | None = None,
    prepare_pods: bool = True,
) -> Job:
    if kind not in {"migrate-amd-nvidia", "migrate-nextgen", "migrate-nvidia-amd"}:
        raise JobError("unknown migration kind")
    if source_pod_id == target_pod_id:
        raise JobError("source and target pods must be different")

    params = {"source_pod_id": source_pod_id, "target_pod_id": target_pod_id}
    if kind == "migrate-nvidia-amd":
        if not 0 < stop_after < total_steps <= 10000 or not 1 <= eval_n <= 2000:
            raise JobError("invalid bounded migration step/evaluation counts")
        params.update(total_steps=total_steps, stop_after=stop_after, eval_n=eval_n)

    def work(job: Job) -> dict[str, Any]:
        source, target = _pod(source_pod_id), _pod(target_pod_id)
        if kind == "migrate-nvidia-amd":
            from gpushare.dashboard.migration import run_resume_migration

            adapter = (ROOT / initial_adapter).resolve() if initial_adapter else None
            if adapter is not None and not adapter.is_relative_to(ROOT.resolve()):
                raise JobError("initial adapter must be inside the project")
            return run_resume_migration(
                job,
                source=source,
                target=target,
                src_info=_ssh_info(source_pod_id),
                dst_info=_ssh_info(target_pod_id),
                total_steps=total_steps,
                stop_after=stop_after,
                eval_n=eval_n,
                initial_adapter=adapter,
                prepare_pods=prepare_pods,
            )
        if kind == "migrate-amd-nvidia" and not (
            source["vendor"] == "amd" and target["vendor"] == "nvidia"
        ):
            raise JobError("AMD→NVIDIA requires an AMD source pod and an NVIDIA target pod")
        if kind == "migrate-nextgen" and not (
            source["vendor"] == target["vendor"] == "nvidia" and source["gpu"] != target["gpu"]
        ):
            raise JobError("next-generation migration requires two different NVIDIA GPUs")

        src_info, dst_info = _ssh_info(source_pod_id), _ssh_info(target_pod_id)
        # The target runs the validation eval, so it needs the card. Migration
        # is pinned at both ends — the source holds the checkpoint and the
        # target is the one being argued about — so it refuses rather than
        # relocating, unlike training.
        JOBS.update(job, "checking the target GPU is free", 5)
        _require_idle_gpu(job, dst_info)
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
            if not any(
                (transfer / name).is_file()
                for name in (
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "adapter_model.safetensors",
                )
            ):
                raise JobError("source checkpoint has no model or adapter safetensors")

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
        if validation["status"] != "ok":
            raise JobError("weights-only migration failed the quality gate; latest run unchanged")
        _write_latest(
            {
                "job_id": job.id,
                "pod_id": target_pod_id,
                "pod": target,
                "local_dir": str(local),
                "remote_checkpoint": target_ckpt,
                "migrated_from": source_pod_id,
                "migration_mode": "weights_only",
            }
        )
        return {
            "migration_mode": "weights_only",
            "training_resumed": False,
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


# The base models a run can start from. Saving one puts it in the picker under
# a name of your choosing; it is a Hugging Face id, so the pod fetches it and
# no weights are copied here.
BASE_CATALOG = [
    {
        "ref": MODEL_ID_FOR_SERVE,
        "label": "Qwen2.5-0.5B",
        "detail": "small task model — the one the JSON fine-tune uses",
    },
    {
        "ref": LONG_CONTEXT_MODEL,
        "label": "Qwen3-4B-Instruct",
        "detail": "40K context — carries the prefill/prefix-cache story, not accuracy",
    },
]


def saved_models() -> list[dict[str, Any]]:
    try:
        data = json.loads(SAVED_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _write_saved(entries: list[dict[str, Any]]) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    SAVED_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def save_model(
    *,
    name: str,
    ref: str,
    kind: str,
    pod_id: str | None = None,
    base: str | None = None,
    metrics: dict[str, Any] | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Put a model in the picker under a name a person chose.

    Names, not slots. The registry this replaces had exactly one "finetuned"
    entry derived from the last run, so training twice silently redefined what
    that word pointed at — and a chat labelled with it kept answering from
    whichever checkpoint happened to be latest. A name that was typed on
    purpose cannot be reassigned by a later run.

    A trained entry records the pod holding the weights. Nothing is copied
    here: the rsync back deliberately excludes *.safetensors, so this is a
    pointer, and it stops meaning anything when that pod is released. The UI
    says which entries those are rather than discovering it at serve time.
    """
    name = (name or "").strip()
    if not name:
        raise JobError("give the model a name")
    if len(name) > 60:
        raise JobError("model name must be 60 characters or fewer")
    entries = [e for e in saved_models() if e["name"] != name]
    entry = {
        "name": name,
        "ref": ref,
        "kind": kind,
        "pod_id": pod_id,
        "base": base,
        "metrics": metrics or {},
        # Which prompt shape this checkpoint was trained under. Serving it
        # with another one is not a worse answer, it is a different question —
        # a model trained on "Q:/A:" and served with the extraction template
        # answers in the extraction schema and never emits what it learned.
        "prompt": prompt or TASK_PROMPT,
        "saved_at": time.time(),
    }
    entries.insert(0, entry)
    _write_saved(entries)
    return entry


def forget_model(*, name: str) -> dict[str, Any]:
    entries = saved_models()
    kept = [e for e in entries if e["name"] != name]
    if len(kept) == len(entries):
        raise JobError(f"no saved model named {name!r}")
    _write_saved(kept)
    return {"forgotten": name}


def available_models() -> list[dict[str, Any]]:
    """Exactly the models a person saved, in the order they saved them.

    This used to synthesise the list instead: two hard-coded entries plus a
    "finetuned" one derived from whichever run was latest. Training twice
    therefore changed what "finetuned" meant without anyone choosing that, and
    the chat went on showing the old label over the new weights. The picker now
    shows saved names only, so what it offers is what somebody decided to keep.
    """
    out = []
    for entry in saved_models():
        out.append(
            {
                "id": entry["name"],
                "label": entry["name"],
                "ref": entry["ref"],
                "kind": entry.get("kind", "base"),
                "pod_id": entry.get("pod_id"),
                "base": entry.get("base"),
                "metrics": entry.get("metrics") or {},
                "prompt": entry.get("prompt") or TASK_PROMPT,
                # Derived, not stored: the prompt a checkpoint was trained
                # under is what decides which data an agent may measure it on,
                # so the two cannot drift apart.
                "task": _task_name_for_prompt(entry.get("prompt") or TASK_PROMPT),
                "saved_at": entry.get("saved_at"),
                "detail": _saved_detail(entry),
            }
        )
    return out


def _saved_detail(entry: dict[str, Any]) -> str:
    if entry.get("kind") == "base":
        return f"base model · {entry['ref']}"
    metrics = entry.get("metrics") or {}
    exact = metrics.get("exact_match_rate")
    bits = [f"trained from {entry.get('base') or '?'}"]
    if exact is not None:
        bits.append(f"exact {exact:.0%}")
    # Weights live on the pod: the sync back excludes *.safetensors, so this
    # name stops resolving when that pod is released. Saying so in the picker
    # beats finding out at serve time.
    if entry.get("pod_id"):
        bits.append(f"weights on pod {entry['pod_id'][:8]}")
    return " · ".join(bits)


def _model_ref(model_id: str, pod_id: str) -> str:
    """Where to load this model from, on this pod.

    The bookkeeping records one pod per checkpoint — the last one to write it.
    A checkpoint can live on several, and after a migration it does: that is
    the whole point of migrating. Serving the same weights on two chips to
    compare them is the demo, not an edge case.

    So a mismatch asks the pod instead of refusing on the record. Same lesson
    as the VRAM and disk checks: our note is a memory of one moment, the
    machine is the fact.
    """
    for m in available_models():
        if m["id"] != model_id:
            continue
        ref, required_pod = m["ref"], m.get("pod_id")
        if not required_pod or required_pod == pod_id or not ref.startswith("/"):
            return ref
        try:
            info = _ssh_info(pod_id)
            present = _capture(
                _ssh_args(info, f"test -f {shlex.quote(ref)}/model.safetensors && echo yes || echo no")
            ).strip()
        except Exception as e:  # noqa: BLE001 — an unreachable pod cannot serve either
            raise JobError(f"pod {pod_id} is unreachable ({type(e).__name__})") from e
        if present.endswith("yes"):
            return ref
        raise JobError(
            f"{m['label']} is not on pod {pod_id} — the last run left it on "
            f"{required_pod}. Migrate it here, or pick that pod."
        )
    raise JobError(f"unknown model {model_id!r}")


def _server_health(timeout: float = 3.0) -> dict[str, Any] | None:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{SERVE_PORT}/health", timeout=timeout) as r:
            return json.loads(r.read()) if r.status == 200 else None
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
        return None


def _tunnel_up(timeout: float = 3.0) -> bool:
    return _server_health(timeout) is not None


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


SERVING_PATH = STATE_ROOT / "serving.json"


def _remember_serving(**fields: Any) -> None:
    with contextlib.suppress(OSError):
        SERVING_PATH.parent.mkdir(parents=True, exist_ok=True)
        SERVING_PATH.write_text(json.dumps(fields, indent=2))


def restore_serving() -> None:
    """Re-adopt a model that is still resident after a dashboard restart.

    Which model is loaded lived only in this process's memory, so every restart
    made the UI report "no model is loaded" while the pod was still holding it —
    and the recovery was to reload an 8 GB model and rebuild a 30K-token cache
    that were both already there. That cost minutes each time.

    Adoption is conditional on the server agreeing: the tunnel is reached and
    its /health must name the same weights the note claims. A stale note whose
    server has since died, or been replaced by a different model, is discarded
    rather than trusted.
    """
    if _serve.get("model_id"):
        return
    try:
        note = json.loads(SERVING_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return
    health = _server_health(timeout=2.0)
    if not health or health.get("model") != note.get("model_ref"):
        return
    with _serve_lock:
        # No `tunnel` handle: the process that owned it is gone. The forward
        # itself outlives it, so requests work; stop_inference_server will kill
        # the remote server and leave the orphaned ssh, which is the lesser of
        # the two problems it used to have.
        _serve.update(
            model_id=note.get("model_id"),
            model_ref=note.get("model_ref"),
            pod_id=note.get("pod_id"),
            dtype=note.get("dtype", "bf16"),
        )
    print(f"adopted the model already serving on {note.get('pod_id')}", file=sys.stderr)


def serving() -> dict[str, Any]:
    """What the chat box is currently talking to, if anything."""
    with _serve_lock:
        if not _serve.get("model_id"):
            return {"running": False}
        health = _server_health(timeout=1.0) or {}
        # The note says which weights we asked for; /health says which are
        # answering. They disagree whenever a forward outlived the process that
        # opened it, and then every latency and every answer on the page is
        # attributed to the wrong model. restore_serving() already refuses a
        # note its server contradicts — this reports the same disagreement
        # instead of printing the note over it.
        live_ref = health.get("model")
        stale = bool(health) and live_ref != _serve["model_ref"]
        return {
            "running": bool(health),
            "model_id": _serve["model_id"],
            "model_ref": _serve["model_ref"],
            "pod_id": _serve["pod_id"],
            "dtype": _serve["dtype"],
            "live_model_ref": live_ref,
            "stale": stale,
            # Read off the server, not remembered here. A page that cannot see
            # whether the document is loaded will happily run the "before" case
            # without it — which returns in a fraction of a second and looks
            # like the optimization already happened.
            "prefix_tokens": health.get("prefix_tokens", 0),
            "prefix_build_s": health.get("prefix_build_s"),
        }


def _clear_stale_forward() -> None:
    """Free SERVE_PORT of a forward this process does not own.

    stop_inference_server() can only terminate a tunnel it has a handle on, so
    one opened by an earlier dashboard process outlives every restart. With the
    port held, ssh -o ExitOnForwardFailure=yes exits immediately and the new
    forward never exists — while the old one keeps answering, which is what let
    a serve job succeed against the wrong pod.

    Only forwards matching this exact spec are killed: the port belongs to this
    feature, but somebody else's unrelated listener is not ours to close.
    """
    spec = f"{SERVE_PORT}:127.0.0.1:{SERVE_PORT}"
    try:
        listeners = _capture(["lsof", f"-tiTCP:{SERVE_PORT}", "-sTCP:LISTEN"], timeout=10)
    except Exception:  # noqa: BLE001 — no lsof is not a reason to fail a serve
        return
    for pid in [line.strip() for line in listeners.splitlines() if line.strip().isdigit()]:
        try:
            argv = _capture(["ps", "-o", "command=", "-p", pid], timeout=10)
        except Exception:  # noqa: BLE001
            continue
        if "ssh" in argv and spec in argv:
            with contextlib.suppress(Exception):
                os.kill(int(pid), signal.SIGTERM)


def start_inference_server(*, pod_id: str, model_id: str, dtype: str = "bf16") -> Job:
    """Load one model on the pod and hold it there.

    Loading Qwen costs ten to twenty seconds. Paying that per message would
    make the chat unusable AND would drown the latency number we want to show,
    so the model stays resident and the dashboard reaches it over an SSH local
    forward — the pod exposes port 22 and nothing else.
    """
    ref = _model_ref(model_id, pod_id)
    template = next(
        (m.get("prompt") or TASK_PROMPT for m in available_models() if m["id"] == model_id),
        TASK_PROMPT,
    )

    def work(job: Job) -> dict[str, Any]:
        stop_inference_server()
        pod, info = _pod(pod_id), _ssh_info(pod_id)
        JOBS.update(job, "syncing serve script", 10)
        _sync_project(job, info)
        JOBS.update(job, "preparing GPU environment", 25)
        # Whoever _serve_python() points at the isolated venv for needs it
        # built first. Branching on the vendor name here (as this used to)
        # is exactly what commit 551dd8a's ChipProfile table replaced
        # everywhere else — a third vendor needing the same migration venv
        # would silently fall into uv sync and fail with the PEP 668 error
        # the AMD row exists to document.
        if _serve_python(pod["vendor"]) != "uv run python":
            _setup_migration_pod(job, info, pod["vendor"])
        else:
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
            f"setsid nohup {_serve_python(pod['vendor'])} scripts/serve.py "
            f"--model {shlex.quote(remote_ref)} "
            f"--dtype {shlex.quote(dtype)} --port {SERVE_PORT} "
            f"--prompt-template {shlex.quote(template)} "
            f"< /dev/null > /tmp/serve.log 2>&1 & echo started"
        )
        JOBS.update(job, f"loading {model_id} on the GPU", 45)
        _run(job, _ssh_args(info, cmd))

        JOBS.update(job, "opening SSH forward", 65)
        _clear_stale_forward()
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
            # Not "is anything answering on this port" — that question is
            # answered "yes" by a forward left behind by an earlier serve,
            # whose own ssh exits instantly under ExitOnForwardFailure because
            # the port is taken. The job then reported success while every
            # request went on reaching the previous pod's model. Ask who is
            # answering instead.
            health = _server_health(timeout=2.0)
            if health and health.get("model") == remote_ref:
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
        _remember_serving(model_id=model_id, model_ref=ref, pod_id=pod_id, dtype=dtype)
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
