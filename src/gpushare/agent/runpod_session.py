"""Bounded Runpod REST v2 sessions with a local, best-effort cleanup watchdog.

Runpod CLI 2.14 and the current REST v2 schema expose no server-side Pod TTL.
The independent watchdog survives a controller exit, but cannot guarantee a
spending limit if the local machine sleeps, loses connectivity, or powers off.
No account key or container environment is written to the session journal.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

from gpushare.agent.gpu_matrix import ALL_TARGETS

API_URL = "https://api.runpod.io/v2"
MAX_BUDGET = 15.0
MAX_DURATION = 5400
SESSION_RE = re.compile(r"[a-z0-9][a-z0-9-]{7,47}\Z")
POD_ID_RE = re.compile(r"[a-zA-Z0-9_-]{5,64}\Z")
ROLES = {"source", "target"}
PROFILES = {
    "migration": {
        "source": ("NVIDIA GeForce RTX 4090", "SECURE", ["EU-RO-1"], .74),
        "target": ("AMD Instinct MI300X OAM", "SECURE", ["EU-RO-1"], 2.39),
    },
    "inference-latency": {
        "source": ("NVIDIA GeForce RTX 3090", "COMMUNITY", [], .22),
        "target": ("NVIDIA GeForce RTX 4090", "SECURE", ["EU-RO-1"], .74),
        "amd": ("AMD Instinct MI300X OAM", "SECURE", ["EU-RO-1"], 2.39),
    },
    "inference-latency-3090": {
        "source": ("NVIDIA GeForce RTX 3090", "COMMUNITY", [], .22),
    },
    "inference-latency-4090": {
        "source": ("NVIDIA GeForce RTX 4090", "SECURE", ["EU-RO-1"], .74),
    },
    "inference-latency-amd": {
        "target": ("AMD Instinct MI300X OAM", "SECURE", ["EU-RO-1"], 2.39),
    },
}

# Each new benchmark model has its own bounded session. A campaign additionally
# reserves their combined worst-case cost with gpu_matrix.expansion_plan.
PROFILES.update({
    target.profile: {"source": (target.gpu_id, target.cloud, list(target.regions), target.hourly_limit)}
    for target in ALL_TARGETS if target.profile.startswith("hardware-")
})


def profile_config(value):
    if not isinstance(value, str) or value not in PROFILES:
        raise SessionError("Unknown approved rental profile")
    return PROFILES[value]


class SessionError(RuntimeError):
    """Safe error message, containing no raw API response."""


class APIError(SessionError):
    def __init__(self, status: int):
        self.status = status
        super().__init__(f"Runpod API returned HTTP {status}")


class UncertainRequest(SessionError):
    """A request may have reached the provider; do not retry a create."""


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def timestamp(value: str) -> float:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone missing")
        return parsed.timestamp()
    except (AttributeError, TypeError, ValueError) as exc:
        raise SessionError("Expected an ISO 8601 timestamp with timezone") from exc


def utc(value: float) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat()


def number(value: object, label: str, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SessionError(f"Invalid {label}")
    if not math.isfinite(value) or value < minimum:
        raise SessionError(f"Invalid {label}")
    return float(value)


def valid_id(value: object) -> bool:
    return isinstance(value, str) and POD_ID_RE.fullmatch(value) is not None


def pod_name(session_id: str, role: str) -> str:
    return f"gpushare-{session_id}-{role}"


class RunpodClient:
    """Fixed-origin HTTPS client. Its errors deliberately omit response bodies."""

    def __init__(self, api_key: str, *, timeout: float = 20, opener=None):
        if not api_key:
            raise SessionError("RUNPOD_API_KEY is required")
        self._key = api_key
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def request(self, method: str, path: str, body: dict | None = None):
        if not path.startswith("/") or path.startswith("//"):
            raise SessionError("Invalid API path")
        data = canonical(body) if body is not None else None
        if data is not None and self._key.encode() in data:
            raise SessionError("Refusing to send account credentials in a Pod payload")
        request = urllib.request.Request(
            API_URL + path, data=data, method=method,
            headers={"Authorization": f"Bearer {self._key}",
                     "Content-Type": "application/json", "User-Agent": "gpushare-session/1"},
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read()
                if response.status == 204 or not raw:
                    return None
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise APIError(exc.code) from None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise UncertainRequest("Runpod request did not return a confirmed result") from exc

    def list_pods(self) -> list[dict]:
        pods = []
        cursor = None
        for _ in range(100):
            query = "?limit=100" + ("&cursor=" + urllib.parse.quote(cursor) if cursor else "")
            page = self.request("GET", "/pods" + query)
            if not isinstance(page, dict) or not isinstance(page.get("pods"), list):
                raise SessionError("Unexpected Pod list schema")
            pods.extend(page["pods"])
            pagination = page.get("pagination", {})
            if not pagination.get("hasNextPage"):
                return pods
            next_cursor = pagination.get("nextCursor")
            if not isinstance(next_cursor, str) or next_cursor == cursor:
                raise SessionError("Invalid Pod list pagination")
            cursor = next_cursor
        raise SessionError("Pod list pagination exceeded its bound")

    def get_pod(self, pod_id: str):
        if not valid_id(pod_id):
            raise SessionError("Invalid Pod ID")
        try:
            return self.request("GET", f"/pods/{pod_id}")
        except APIError as exc:
            if exc.status == 404:
                return None
            raise

    def create_pod(self, payload: dict):
        return self.request("POST", "/pods", payload)

    def terminate_pod(self, pod_id: str):
        if not valid_id(pod_id):
            raise SessionError("Invalid Pod ID")
        try:
            return self.request("DELETE", f"/pods/{pod_id}")
        except APIError as exc:
            if exc.status != 404:
                raise
        return None

    def check_quote(self, record: dict, *, allow_unavailable_target_probe: bool = False):
        if allow_unavailable_target_probe and not (
            record.get("role") == "target" and record.get("gpu_id") == "AMD Instinct MI300X OAM"
            and record.get("gpu_count") == 1 and record.get("cloud") == "SECURE"
            and record.get("data_center_ids") == ["EU-RO-1"]
        ):
            raise SessionError("Unavailable allocation probe is restricted to the approved AMD target")
        filters = {"include": "AVAILABILITY", "product": "POD",
                   "count": record["gpu_count"], "cloud": record["cloud"]}
        if record.get("min_cuda_version"):
            filters["minCudaVersion"] = record["min_cuda_version"]
        query = urllib.parse.urlencode(filters)
        response = self.request("GET", "/catalog/gpus?" + query)
        matches = [gpu for gpu in response.get("gpus", []) if gpu.get("id") == record["gpu_id"]]
        if len(matches) != 1:
            raise SessionError("Quoted GPU is absent from the current catalog")
        gpu = matches[0]
        if gpu.get(record["cloud"].lower()) is False:
            raise SessionError("Quoted GPU does not support the approved cloud")
        rate = gpu.get("price", {}).get(record["cloud"].lower())
        if rate is None or not math.isclose(
            number(rate, "catalog price") * record["gpu_count"], record["hourly_gpu_usd"],
            rel_tol=0, abs_tol=0.000001,
        ):
            raise SessionError("GPU quote changed; prepare a new reviewed plan")
        available = {"LOW", "MEDIUM", "HIGH"}
        if gpu.get("availability") not in available and not allow_unavailable_target_probe:
            raise SessionError("Quoted GPU is no longer available")
        centers = {dc.get("id") for dc in gpu.get("dataCenters", [])
                   if dc.get("availability") in available}
        center_available = bool(centers.intersection(record["data_center_ids"])) or (
            record["cloud"] == "COMMUNITY" and record["data_center_ids"] == []
            and gpu.get("availability") in available
        )
        if not center_available and not allow_unavailable_target_probe:
            raise SessionError("Quoted GPU is unavailable in the approved data center")
        return {"availability": gpu.get("availability"),
                "approved_data_center_available": center_available,
                "availability_probe_permitted": allow_unavailable_target_probe}


@contextlib.contextmanager
def file_lock(path: Path):
    """OS advisory locking releases automatically if either process exits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        end = time.monotonic() + 10
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= end:
                    raise SessionError("Session journal is busy") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Session:
    def __init__(self, directory: Path, session_id: str, client=None, *, clock=time.time,
                 sleep=time.sleep):
        if not SESSION_RE.fullmatch(session_id):
            raise SessionError("Session ID must be 8-48 lowercase letters, numbers or hyphens")
        self.directory = Path(directory).resolve()
        self.session_id = session_id
        self.path = self.directory / f"{session_id}.json"
        self.lock_path = self.directory / f"{session_id}.lock"
        self.client = client
        self.clock = clock
        self.sleep = sleep

    def _validate(self, journal: dict):
        try:
            if journal["version"] != 1 or journal["session_id"] != self.session_id:
                raise SessionError("Journal ownership mismatch")
            start = number(journal["created_at"], "creation time")
            deadline = number(journal["deadline"], "deadline")
            if not 0 < deadline - start <= MAX_DURATION:
                raise SessionError("Unsafe journal deadline")
            if not 0 < number(journal["budget_usd"], "budget") <= MAX_BUDGET:
                raise SessionError("Unsafe journal budget")
            number(journal["storage_allowance_per_hour"], "storage allowance", 0.10)
            baseline = journal["baseline_ids"]
            if not isinstance(baseline, list) or not all(valid_id(item) for item in baseline):
                raise SessionError("Unsafe baseline IDs")
            records = journal["pods"]
            configuration = profile_config(journal.get("profile", "migration"))
            if len(records) != len(configuration) or {r["role"] for r in records} != set(configuration):
                raise SessionError("Unsafe session Pod roles")
            ids = []
            permission = journal.get("allow_unavailable_target_probe", False)
            if not isinstance(permission, bool):
                raise SessionError("Invalid unavailable-target probe permission")
            for record in records:
                if record["owner_session"] != self.session_id:
                    raise SessionError("Pod owner reference mismatch")
                if record["name"] != pod_name(self.session_id, record["role"]):
                    raise SessionError("Pod name does not belong to this session")
                gpu, cloud, centers, max_rate = configuration[record["role"]]
                if (record["gpu_count"] != 1 or record["cloud"] != cloud
                        or record["gpu_id"] != gpu or record["data_center_ids"] != centers
                        or number(record["hourly_gpu_usd"], "GPU quote") > max_rate):
                    raise SessionError("Unsafe GPU configuration")
                probe = record.get("allow_unavailable_target_probe", False)
                if not isinstance(probe, bool) or (probe and not (
                    permission and record["role"] == "target"
                    and record["gpu_id"] == "AMD Instinct MI300X OAM"
                    and record["data_center_ids"] == ["EU-RO-1"]
                )):
                    raise SessionError("Unsafe unavailable-target probe permission")
                number(record["hourly_gpu_usd"], "GPU quote", 0.001)
                minimum_cuda = record.get("min_cuda_version")
                if minimum_cuda is not None and (
                    not isinstance(minimum_cuda, str) or not re.fullmatch(r"\d+\.\d+", minimum_cuda)
                    or not gpu.startswith("NVIDIA")
                ):
                    raise SessionError("Unsafe CUDA version constraint")
                if record.get("id") is not None:
                    if not valid_id(record["id"]) or record["id"] in baseline:
                        raise SessionError("Unsafe journal Pod ID")
                    ids.append(record["id"])
            if len(ids) != len(set(ids)):
                raise SessionError("Duplicate journal Pod IDs")
        except (KeyError, TypeError) as exc:
            raise SessionError("Invalid session journal schema") from exc

    def _load(self):
        try:
            journal = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SessionError("Cannot read session journal") from exc
        self._validate(journal)
        return journal

    def _save(self, journal):
        self._validate(journal)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(canonical(journal))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def read(self):
        with file_lock(self.lock_path):
            return self._load()

    def mutate(self, operation):
        with file_lock(self.lock_path):
            journal = self._load()
            operation(journal)
            self._save(journal)
            return journal

    def prepare(self, plan: dict, *, allow_unavailable_target_probe: bool = False):
        """Prepare a bounded run; the explicit probe flag bypasses AMD stock only."""
        if not isinstance(plan, dict):
            raise SessionError("Plan must be a JSON object")
        if not isinstance(allow_unavailable_target_probe, bool):
            raise SessionError("Unavailable-target probe permission must be a boolean")
        now = self.clock()
        budget = number(plan.get("budget_usd"), "budget", 0.01)
        duration = number(plan.get("max_duration_seconds"), "duration", 1)
        storage = number(plan.get("storage_allowance_per_hour"), "storage allowance", 0.10)
        if budget > MAX_BUDGET or duration > MAX_DURATION:
            raise SessionError("Plan exceeds the authorized $15 / 90 minute limits")
        profile = plan.get("profile", "migration")
        configuration = profile_config(profile)
        if profile != "migration" and allow_unavailable_target_probe:
            raise SessionError("Stock bypass is not supported for this profile")
        pods = plan.get("pods", [])
        if (not isinstance(pods, list) or len(pods) != len(configuration)
                or not all(isinstance(pod, dict) for pod in pods)
                or {pod.get("role") for pod in pods} != set(configuration)):
            raise SessionError("Exactly the approved profile Pod roles are required")
        records = []
        for item in pods:
            role, payload = item["role"], item.get("payload")
            if not isinstance(payload, dict):
                raise SessionError("Each Pod requires a REST v2 payload object")
            allowed = {"name", "image", "gpu", "cloud", "disk", "dataCenterIds", "ports",
                       "env", "entrypoint", "cmd", "args", "startSsh", "globalNetworking"}
            if set(payload) - allowed or not isinstance(payload.get("image"), str):
                raise SessionError("Use an explicit image and supported bounded Pod fields")
            if payload.get("name") != pod_name(self.session_id, role):
                raise SessionError("Payload name must match the unique session and role")
            expected_gpu, expected_cloud, centers, max_rate = configuration[role]
            if (payload.get("cloud") != expected_cloud or not isinstance(payload.get("gpu"), dict)
                    or payload["gpu"].get("count") != 1):
                raise SessionError("Only the approved single-GPU cloud configuration is supported")
            if payload["gpu"].get("id") != expected_gpu:
                raise SessionError("Payload GPU differs from the approved pair")
            if payload.get("dataCenterIds", []) != centers:
                raise SessionError("Only the approved data center configuration is supported")
            if not 1 <= number(payload.get("disk"), "disk") <= 100 or payload.get("mounts"):
                raise SessionError("Use at most 100 GB container disk and no separate volumes")
            probe = allow_unavailable_target_probe and role == "target"
            if item.get("available") is not True and not (probe and item.get("available") is False):
                raise SessionError("Quoted GPU was unavailable")
            quoted_at = timestamp(item.get("quoted_at_utc"))
            if not 0 <= now - quoted_at <= 900:
                raise SessionError("Plan quote must be from the past 15 minutes")
            env = payload.get("env", {})
            if not isinstance(env, dict) or any(
                re.search(r"SECRET|TOKEN|PASSWORD|API.?KEY", key, re.I) for key in env
            ):
                raise SessionError("Do not include credentials in the Pod environment")
            rate = number(item.get("hourly_gpu_usd"), "GPU quote", 0.001)
            if rate > max_rate:
                raise SessionError("GPU quote exceeds the approved rate")
            records.append({"role": role, "owner_session": self.session_id,
                            "name": payload["name"], "gpu_id": expected_gpu, "gpu_count": 1,
                            "cloud": expected_cloud, "data_center_ids": centers,
                            "hourly_gpu_usd": rate, "quoted_at": quoted_at,
                            "min_cuda_version": payload["gpu"].get("minCudaVersion"),
                            "quoted_available": item["available"],
                            "allow_unavailable_target_probe": probe,
                            "payload_hash": digest(payload), "state": "planned", "id": None})
        journal = {"version": 1, "profile": profile, "session_id": self.session_id, "plan_hash": digest(plan),
                   "budget_usd": budget, "storage_allowance_per_hour": storage,
                   "created_at": now, "deadline": now + duration, "pods": records,
                   "baseline_ids": [], "baseline_recorded": False, "state": "prepared",
                   "allow_unavailable_target_probe": allow_unavailable_target_probe,
                   "cleanup_requested": False, "watchdog_heartbeat_at": None}
        self._budget(journal, now)
        with file_lock(self.lock_path):
            if self.path.exists():
                raise SessionError("Session already exists; choose a new unique session ID")
            self._save(journal)
        return journal

    def _budget(self, journal, now):
        remaining = max(0, journal["deadline"] - now) / 3600
        rates = sum(record["hourly_gpu_usd"] for record in journal["pods"])
        storage = journal["storage_allowance_per_hour"]
        spent_bound = sum(max(0, now - record["attempt_at"]) / 3600 * record["hourly_gpu_usd"]
                          for record in journal["pods"] if record.get("attempt_at") is not None)
        spent_bound += max(0, now - journal["created_at"]) / 3600 * storage
        required = spent_bound + remaining * (rates + storage)
        if required > journal["budget_usd"] + 0.000001:
            raise SessionError("Worst-case remaining session spend exceeds the approved budget")
        return required

    def heartbeat(self):
        return self.mutate(lambda j: j.update(watchdog_heartbeat_at=self.clock()))

    def _require_create(self, journal):
        now = self.clock()
        if journal["cleanup_requested"] or now >= journal["deadline"]:
            raise SessionError("Session deadline passed or cleanup has started")
        heartbeat = journal.get("watchdog_heartbeat_at")
        if heartbeat is None or not 0 <= now - heartbeat <= 30:
            raise SessionError("Start the independent watchdog before creating Pods")
        self._budget(journal, now)

    def _matches(self, pod, record, journal):
        if not isinstance(pod, dict) or not valid_id(pod.get("id")):
            return False
        if pod["id"] in journal["baseline_ids"] or pod.get("name") != record["name"]:
            return False
        gpu = pod.get("gpu", {})
        if gpu.get("id") != record["gpu_id"] or gpu.get("count") != record["gpu_count"]:
            return False
        if pod.get("cloud") != record["cloud"]:
            return False
        try:
            created = timestamp(pod.get("createdAt"))
        except SessionError:
            return False
        return journal["created_at"] - 5 <= created <= self.clock() + 60

    def _adopt(self, role, pod):
        def update(journal):
            record = next(r for r in journal["pods"] if r["role"] == role)
            if not self._matches(pod, record, journal):
                raise SessionError("Returned Pod does not match this session's ownership")
            if record["id"] not in (None, pod["id"]):
                raise SessionError("Refusing to replace an already recorded Pod ID")
            record.update(id=pod["id"], state="created", confirmed_at=self.clock())
            # Keep ownership first, even if the billable rate is unexpectedly high.
            # Cleanup also adopts uncertain creates and must remain able to delete them.
            cost = pod.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost):
                record["observed_hourly_usd"] = cost
        self.mutate(update)

    def _reconcile(self, role):
        journal = self.read()
        record = next(r for r in journal["pods"] if r["role"] == role)
        matches = [pod for pod in self.client.list_pods() if pod.get("name") == record["name"]]
        if len(matches) > 1:
            raise SessionError("Multiple Pods share the session name; manual reconciliation required")
        if matches:
            self._adopt(role, matches[0])
            return True
        return False

    def create(self, plan: dict):
        journal = self.read()
        if digest(plan) != journal["plan_hash"]:
            raise SessionError("Plan changed after preparation")
        if journal["state"] != "prepared":
            raise SessionError("Creation can be invoked only once; reconcile existing attempts")
        self._require_create(journal)
        # Snapshot all existing IDs before the first create; never adopt or delete them.
        existing = self.client.list_pods()
        if any(pod.get("name") in {r["name"] for r in journal["pods"]} for pod in existing):
            raise SessionError("A session Pod name already exists; choose another session ID")
        baseline = [pod["id"] for pod in existing if valid_id(pod.get("id"))]

        def begin(j):
            self._require_create(j)
            if j["state"] != "prepared":
                raise SessionError("Another controller already started creation")
            j.update(state="creating", baseline_ids=baseline, baseline_recorded=True)
        self.mutate(begin)
        try:
            for item in plan["pods"]:
                role = item["role"]
                record = next(r for r in self.read()["pods"] if r["role"] == role)
                if record.get("allow_unavailable_target_probe", False):
                    observed_quote = self.client.check_quote(
                        record, allow_unavailable_target_probe=True
                    )
                else:
                    observed_quote = self.client.check_quote(record)
                if isinstance(observed_quote, dict):
                    self.mutate(lambda j, selected_role=role, observed=observed_quote: next(
                        r for r in j["pods"] if r["role"] == selected_role
                    ).update(last_catalog_check=observed))

                def intent(j, selected_role=role):
                    self._require_create(j)
                    r = next(r for r in j["pods"] if r["role"] == selected_role)
                    r.update(state="creating", attempt_at=self.clock())
                self.mutate(intent)
                try:
                    pod = self.client.create_pod(item["payload"])
                except (UncertainRequest, APIError) as exc:
                    uncertain = isinstance(exc, UncertainRequest) or exc.status >= 500 or exc.status == 408
                    if not uncertain:
                        self.mutate(lambda j, selected_role=role: next(
                            r for r in j["pods"] if r["role"] == selected_role
                        ).update(state="rejected"))
                        raise
                    self.mutate(lambda j, selected_role=role: next(
                        r for r in j["pods"] if r["role"] == selected_role
                    ).update(state="uncertain"))
                    found = False
                    for attempt in range(3):
                        if attempt:
                            self.sleep(1)
                        if self._reconcile(role):
                            found = True
                            break
                    if not found:
                        raise SessionError("Create result remains uncertain; watchdog will reconcile") from None
                else:
                    self._adopt(role, pod)
                adopted = next(r for r in self.read()["pods"] if r["role"] == role)
                cost = adopted.get("observed_hourly_usd")
                if cost is not None and number(cost, "Pod cost") > (
                    record["hourly_gpu_usd"] + journal["storage_allowance_per_hour"] / len(journal["pods"])
                ):
                    raise SessionError("Created Pod cost exceeds its approved allowance")
                self._require_create(self.read())
            def activate(j):
                self._require_create(j)
                j["state"] = "active"
            self.mutate(activate)
            return self.read()
        except BaseException:
            self.mutate(lambda j: j.update(cleanup_requested=True))
            self.cleanup()
            raise

    def cleanup(self):
        journal = self.mutate(lambda j: j.update(cleanup_requested=True, state="cleaning"))
        errors = []
        for initial in journal["pods"]:
            role = initial["role"]
            try:
                if initial["id"] is None and initial["state"] in {"creating", "uncertain"}:
                    self._reconcile(role)
                current = self.read()
                record = next(r for r in current["pods"] if r["role"] == role)
                if record["id"] is None:
                    continue
                pod = self.client.get_pod(record["id"])
                if pod is not None:
                    if not self._matches(pod, record, current) or pod["id"] != record["id"]:
                        raise SessionError("Pod ownership validation failed; refusing deletion")
                    if pod.get("status") != "TERMINATED":
                        self.client.terminate_pod(record["id"])
                    for attempt in range(3):
                        observed = self.client.get_pod(record["id"])
                        if observed is None or observed.get("status") == "TERMINATED":
                            break
                        if attempt < 2:
                            self.sleep(1)
                    else:
                        raise SessionError("Pod termination has not been verified")
                self.mutate(lambda j, selected_role=role: next(
                    r for r in j["pods"] if r["role"] == selected_role
                ).update(state="terminated", terminated_at=self.clock()))
            except SessionError as exc:
                errors.append(f"{role}: {exc}")
        def finish(j):
            pending = any(r["state"] in {"creating", "uncertain", "created"} for r in j["pods"])
            j["state"] = "needs_cleanup" if pending or errors else "closed"
            j["cleanup_errors"] = errors
            if j["state"] == "closed":
                j["closed_at"] = self.clock()
        return self.mutate(finish)

    def watchdog_tick(self):
        journal = self.heartbeat()
        if journal["state"] == "closed":
            return journal
        if journal["cleanup_requested"] or self.clock() >= journal["deadline"]:
            return self.cleanup()
        return journal

    def emergency_cleanup(self, snapshot: dict):
        """Verify/delete from a validated snapshot when journal writes fail.

        This does not claim persisted cleanup success. Normal cleanup must later
        record/verify the result. Existing baseline IDs and unattempted names are
        still excluded, and each DELETE is preceded by provider ownership checks.
        """
        self._validate(snapshot)
        removed = []
        for record in snapshot["pods"]:
            pod_id = record["id"]
            if pod_id is None and record["state"] in {"creating", "uncertain"}:
                candidates = [pod for pod in self.client.list_pods()
                              if pod.get("name") == record["name"]]
                if len(candidates) != 1 or not self._matches(candidates[0], record, snapshot):
                    continue
                pod_id = candidates[0]["id"]
            if pod_id is None:
                continue
            pod = self.client.get_pod(pod_id)
            if pod is not None:
                if not self._matches(pod, record, snapshot) or pod["id"] != pod_id:
                    raise SessionError("Emergency cleanup ownership check failed")
                if pod.get("status") != "TERMINATED":
                    self.client.terminate_pod(pod_id)
                observed = self.client.get_pod(pod_id)
                if observed is not None and observed.get("status") != "TERMINATED":
                    raise SessionError("Emergency termination has not yet been verified")
            removed.append(pod_id)
        return removed

    def watchdog(self, *, poll_seconds: float = 5):
        if not 1 <= poll_seconds <= 10:
            raise SessionError("Watchdog poll interval must be 1-10 seconds")
        snapshot = None
        last_error = None
        last_error_at = 0.0
        while True:
            try:
                snapshot = self.read()
                journal = self.watchdog_tick()
                if journal["state"] == "closed":
                    return journal
                last_error = None
            except (SessionError, OSError) as exc:
                # Disk or connectivity failures must not disable future attempts.
                message = str(exc) if isinstance(exc, SessionError) else "Journal filesystem failure"
                if message != last_error or self.clock() - last_error_at >= 60:
                    print(json.dumps({"watchdog_error": message}), file=sys.stderr, flush=True)
                    last_error, last_error_at = message, self.clock()
                if snapshot is not None and (
                    snapshot["cleanup_requested"] or self.clock() >= snapshot["deadline"]
                ):
                    try:
                        self.emergency_cleanup(snapshot)
                    except (SessionError, OSError):
                        pass  # Retry while preserving the original owned IDs.
            self.sleep(poll_seconds)
