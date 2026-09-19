"""Atomic job records survive transient Windows sharing locks without hiding failure."""

import json
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
