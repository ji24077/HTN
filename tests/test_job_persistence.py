"""Atomic job records survive transient Windows sharing locks without hiding failure."""

import json
import threading
from pathlib import Path

import pytest

from gpushare.dashboard import runner


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(runner, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(runner, "_redact", lambda text: text)
    monkeypatch.setattr(runner.time, "sleep", lambda delay: None)
    return runner.JobManager()


def test_replace_retries_keep_old_json_complete_and_use_unique_files(manager, monkeypatch):
    job = runner.Job(id="test", kind="test", params={}, logs=["José"])
    manager._persist(job)
    path = runner.JOB_ROOT / "test.json"
    original = path.read_bytes()
    real_replace = runner.os.replace
    attempts = []

    def locked_replace(source, target):
        attempts.append(Path(source))
        assert json.loads(Path(source).read_text(encoding="utf-8"))["logs"] == ["José"]
        if len(attempts) <= 2:
            assert path.read_bytes() == original
            raise PermissionError("Windows sharing violation")
        real_replace(source, target)

    monkeypatch.setattr(runner.os, "replace", locked_replace)
    job.status = "running"
    manager._persist(job)
    assert len(attempts) == 3
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "running"
    first_temp = attempts[0]
    manager._persist(job)
    assert attempts[-1] != first_temp
    assert not list(runner.JOB_ROOT.glob("*.tmp"))


def test_permanent_replace_failure_preserves_prior_record_and_cleans_temp(manager, monkeypatch):
    job = runner.Job(id="test", kind="test", params={})
    manager._persist(job)
    path = runner.JOB_ROOT / "test.json"
    original = path.read_bytes()
    attempts = []

    def denied(source, target):
        attempts.append(source)
        raise PermissionError("permanent deny")

    monkeypatch.setattr(runner.os, "replace", denied)
    job.status = "running"
    with pytest.raises(PermissionError, match="permanent deny"):
        manager._persist(job)
    assert len(attempts) == 8
    assert path.read_bytes() == original
    assert not list(runner.JOB_ROOT.glob("*.tmp"))


@pytest.mark.parametrize("failure_stage", ["start", "work", "finish"])
def test_job_boundary_does_not_leave_running_after_persistence_failure(
    manager, monkeypatch, caplog, failure_stage
):
    job = runner.Job(id="test", kind="test", params={})
    manager._persist(job)
    real_persist = manager._persist
    calls = []

    def fail_persist(current):
        calls.append(current.status)
        if failure_stage == "start" or len(calls) > 1:
            raise PermissionError("record is locked")
        real_persist(current)

    monkeypatch.setattr(manager, "_persist", fail_persist)
    worked = []

    def work(current):
        worked.append(True)
        if failure_stage == "work":
            manager.log(current, "progress")
        return {"ok": True}

    manager._run(job, work)
    assert job.status == "failed"
    assert job.finished_at is not None
    assert "record is locked" in job.error
    assert "Could not persist final job state" in caplog.text
    assert bool(worked) == (failure_stage != "start")
    assert len(calls) <= 4


def test_work_persistence_failure_can_record_terminal_failure(manager, monkeypatch):
    job = runner.Job(id="test", kind="test", params={})
    real_persist = manager._persist
    count = 0

    def transient_failure(current):
        nonlocal count
        count += 1
        if count == 2:
            raise PermissionError("logging failed")
        real_persist(current)

    monkeypatch.setattr(manager, "_persist", transient_failure)
    manager._run(job, lambda current: manager.log(current, "progress"))
    saved = json.loads((runner.JOB_ROOT / "test.json").read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["finished_at"] is not None
    assert saved["error"] == "logging failed"


def test_failed_initial_create_does_not_leave_queued_job(manager, monkeypatch):
    def failure(job):
        raise PermissionError("cannot create record")

    monkeypatch.setattr(manager, "_persist", failure)
    with pytest.raises(PermissionError, match="cannot create record"):
        manager.create("test", {}, lambda job: {"unexpected": True})
    assert manager.list() == []


def test_workspace_jobs_overlap_only_on_disjoint_pods(manager, monkeypatch):
    class HeldThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(runner.threading, "Thread", HeldThread)
    first = manager.create(
        "relay-qwen4b-qlora",
        {"pod_id": "pod-a5000"},
        lambda _job: {"ok": True},
    )

    second = manager.create(
        "relay-prefix-cache-benchmark",
        {"pod_id": "pod-4090"},
        lambda _job: {"ok": True},
    )

    assert first.status == second.status == "queued"
    with pytest.raises(runner.JobError, match="required execution resource"):
        manager.create(
            "relay-qwen4b-qlora",
            {"pod_id": "pod-4090"},
            lambda _job: {"unexpected": True},
        )


def test_cancelling_job_keeps_its_pod_claim_and_legacy_jobs_are_global(manager):
    manager._jobs = {
        "cancelled-later": runner.Job(
            id="cancelled-later",
            kind="relay-qwen4b-qlora",
            params={"pod_id": "pod-a5000"},
            status="cancelling",
        )
    }

    conflict = manager.resource_conflict(
        "relay-prefix-cache-benchmark", {"pod_id": "pod-a5000"}
    )
    assert conflict is not None
    assert conflict["status"] == "cancelling"
    assert manager.resource_conflict(
        "relay-demo-cache-reclaim", {"pod_id": "pod-a5000"}
    ) is not None
    assert manager.resource_conflict(
        "relay-prefix-cache-benchmark", {"pod_id": "pod-4090"}
    ) is None
    assert manager.resource_conflict("generate-data", {}) is not None


def test_internal_states_include_recovery_data_and_distinguish_loaded_workers(manager):
    work_returned = threading.Event()

    def work(_job):
        work_returned.set()
        return {"artifact": {"location": "/workspace/private/adapter"}, "ok": True}

    job = manager.create(
        "training",
        {"model_id": "version-1", "pod_id": "pod-1", "command": "private"},
        work,
    )
    assert work_returned.wait(1)
    for _attempt in range(1_000):
        if job.status not in {"queued", "running", "cancelling"}:
            break
        threading.Event().wait(0.001)

    internal = manager.states()[job.id]
    assert internal["params"]["model_id"] == "version-1"
    assert internal["result"]["artifact"]["location"] == "/workspace/private/adapter"
    assert internal["worker_resident"] is True
    assert internal["finished_at"] is not None

    public = manager.list()[0]
    assert "params" not in public
    assert "result" not in public
    assert "worker_resident" not in public

    restarted = runner.JobManager()
    loaded = restarted.states()[job.id]
    assert loaded["result"]["ok"] is True
    assert loaded["worker_resident"] is False
