"""No cloud/GPU access: exercise the complete migration ordering with fake hosts."""

import io
import json
import shutil
import tarfile

import pytest

from gpushare.agent.evaluation import evaluation_identity
from gpushare.agent.task import REQUIRED_FIELDS
from gpushare.dashboard import migration, runner


@pytest.fixture
def fake_hosts(tmp_path, monkeypatch):
    from gpushare.trainer import checkpoint

    root = tmp_path / "project"
    data = root / "data"
    data.mkdir(parents=True)
    row = {
        "sentence": "Ann, 30, joined Lab in 2020 as engineer.",
        "record": {"name": "Ann", "age": 30, "org": "Lab", "role": "engineer", "year": 2020},
    }
    for name in ("train.jsonl", "heldout.jsonl"):
        (data / name).write_text(json.dumps(row) + "\n", encoding="utf-8")
    remote_root = "/workspace/test"
    hosts = {name: tmp_path / name for name in ("source", "target")}
    events, published = [], []
    controls = {
        "fail_before": False,
        "fail_after": False,
        "fail_verify": False,
        "source_json_parse_rate": 1.0,
        "source_exact_match_rate": 1.0,
    }
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "RUN_ROOT", root / "runs")
    monkeypatch.setattr(runner, "REMOTE_ROOT", remote_root)
    monkeypatch.setattr(runner, "_write_latest", published.append)
    monkeypatch.setattr(runner.JOBS, "update", lambda *args: None)
    monkeypatch.setattr(runner.JOBS, "log", lambda *args: None)
    monkeypatch.setattr(
        runner, "_sync_project", lambda job, info: events.append((info["side"], "sync"))
    )
    monkeypatch.setattr(runner, "_setup_migration_pod", lambda *args: None)
    monkeypatch.setattr(runner, "_run", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        migration,
        "_preflight",
        lambda job, info, vendor: {"vendor": vendor, "smoke_test": "passed"},
    )

    def remote_path(info, path):
        relative = path.removeprefix(remote_root + "/")
        return hosts[info["side"]] / relative

    def push(job, info, local, remote):
        if (local / "training-manifest.json").is_file():
            events.append((info["side"], "bundle_upload"))
        target = remote_path(info, remote)
        target.mkdir(parents=True, exist_ok=True)
        if local.is_dir():
            shutil.copytree(local, target, dirs_exist_ok=True)
        else:
            shutil.copy2(local, target / local.name)

    def pull(job, info, remote, local, **kwargs):
        if remote.endswith("/source-ckpt"):
            events.append((info["side"], "bundle_download"))
        shutil.copytree(remote_path(info, remote), local, dirs_exist_ok=True)

    def remote(job, info, argv, **kwargs):
        side = info["side"]
        if argv[1] == "-c":
            if "verify_training_bundle" in argv[2]:
                events.append((side, "verify"))
                if side == "target" and controls["fail_verify"]:
                    raise runner.JobError("checkpoint hash mismatch")
            return 0
        options = dict(zip(argv[2::2], argv[3::2], strict=False))
        if argv[1] == "scripts/train.py":
            stage = "resume" if "--resume" in argv else "source_train"
            events.append((side, stage))
            out = remote_path(info, options["--out"])
            out.mkdir(parents=True)
            manifest = {
                "global_step": int(options.get("--stop-after", options["--steps"])),
                "job_id": options["--job-id"],
            }
            (out / "training-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (out / "adapter_config.json").write_text("{}", encoding="utf-8")
            return 0
        assert argv[1] == "scripts/evaluate.py"
        is_before = "target-before-resume" in options["--out"]
        stage = (
            "source_eval" if side == "source" else "target_before" if is_before else "target_after"
        )
        events.append((side, stage))
        out = remote_path(info, options["--out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "n": 1,
            "json_parse_rate": 1.0,
            "exact_match_rate": 1.0,
            "field_accuracy": dict.fromkeys(REQUIRED_FIELDS, 1.0),
            "evaluation": evaluation_identity([row], max_new_tokens=128, seq_len=256),
            "inference": migration.INFERENCE,
        }
        fail = side == "target" and controls["fail_before" if is_before else "fail_after"]
        if side == "source":
            report["json_parse_rate"] = controls["source_json_parse_rate"]
            report["exact_match_rate"] = controls["source_exact_match_rate"]
        if fail:
            # Aggregate metrics deliberately unchanged: field gating must catch it.
            report["field_accuracy"]["year"] = 0.8
        out.write_text(json.dumps(report), encoding="utf-8")
        return int(fail)

    monkeypatch.setattr(runner, "_push", push)
    monkeypatch.setattr(runner, "_pull", pull)
    monkeypatch.setattr(runner, "_remote", remote)
    monkeypatch.setattr(
        checkpoint,
        "verify_training_bundle",
        lambda path: json.loads((path / "training-manifest.json").read_text(encoding="utf-8")),
    )
    info = {"ip": "127.0.0.1", "port": 22, "key": "unused-fake-key"}

    def run():
        return migration.run_resume_migration(
            runner.Job(id="migration-test", kind="migrate-nvidia-amd", params={}),
            source={"id": "source", "vendor": "nvidia"},
            target={"id": "target", "vendor": "amd"},
            src_info={**info, "side": "source"},
            dst_info={**info, "side": "target"},
            prepare_pods=False,
        )

    return run, events, published, controls


def test_full_migration_checks_target_before_resume_and_publishes_only_at_end(fake_hosts):
    run, events, published, _ = fake_hosts
    result = run()
    assert events.index(("target", "verify")) < events.index(("target", "target_before"))
    assert events.index(("target", "target_before")) < events.index(("target", "resume"))
    assert events.index(("target", "resume")) < events.index(("target", "target_after"))
    assert result["source_step"] == 4 and result["resumed_to_step"] == 8
    assert result["migration_mode"] == "full_training_resume"
    assert len(published) == 1 and published[0]["remote_checkpoint"].endswith("resumed-ckpt")


@pytest.mark.parametrize("failure", ["fail_before", "fail_after", "fail_verify"])
def test_failed_transfer_or_quality_never_publishes_and_precheck_failure_never_resumes(
    fake_hosts, failure
):
    run, events, published, controls = fake_hosts
    controls[failure] = True
    with pytest.raises(runner.JobError):
        run()
    assert not published
    assert (("target", "resume") in events) == (failure == "fail_after")


def test_gate_rejects_different_examples_even_with_perfect_metrics():
    report = {
        "evaluation": {"dataset_sha256": "first"},
        "inference": migration.INFERENCE,
        "json_parse_rate": 1.0,
        "exact_match_rate": 1.0,
        "field_accuracy": dict.fromkeys(REQUIRED_FIELDS, 1.0),
    }
    changed = {**report, "evaluation": {"dataset_sha256": "different"}}
    with pytest.raises(runner.JobError, match="not comparable"):
        migration._gate(report, changed)


@pytest.mark.parametrize("metric", ["source_json_parse_rate", "source_exact_match_rate"])
def test_zero_quality_source_is_rejected_before_bundle_transfer_or_amd_work(fake_hosts, metric):
    run, events, published, controls = fake_hosts
    controls[metric] = 0.0
    with pytest.raises(runner.JobError, match="source baseline has zero"):
        run()
    assert ("source", "source_eval") in events
    assert not any(
        stage in {"bundle_download", "bundle_upload", "target_before", "resume"}
        for _, stage in events
    )
    assert not published


@pytest.mark.parametrize(
    "name,link", [("../escape", False), ("/absolute", False), ("linked", True)]
)
def test_transfer_archive_rejects_traversal_and_links(tmp_path, name, link):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "../escape"
        else:
            member.size = 1
        tar.addfile(member, None if link else io.BytesIO(b"x"))
    destination = tmp_path / "safe"
    destination.mkdir()
    with pytest.raises(runner.JobError, match="unsafe"):
        runner._extract_transfer_archive(archive, destination)
    assert not list(destination.iterdir())


def test_scp_preserves_local_paths_as_single_arguments_and_rejects_remote_shell_text():
    info = {"ip": "127.0.0.1", "port": 2222, "key": "C:/key dir/key"}
    local = "C:/work dir/weights.tar.gz"
    args = runner._scp_args(info, local, runner._scp_remote(info, "/workspace/run/"))
    assert args[-2] == local and args[2] == info["key"]
    with pytest.raises(runner.JobError):
        runner._scp_remote(info, "/workspace/$(command)")


def test_project_archive_allowlist_excludes_caches_secrets_and_checkpoints(tmp_path, monkeypatch):
    root = tmp_path / "project"
    for name in (
        "src/main.py",
        "scripts/train.py",
        "data/train.jsonl",
        ".cache/key",
        ".env",
        "ckpt/model.safetensors",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example", encoding="utf-8")
    captured = []
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)

    def run(job, argv, **kwargs):
        if argv[0] == "scp":
            with tarfile.open(argv[-2], "r:gz") as tar:
                captured.extend(tar.getnames())
        return 0

    monkeypatch.setattr(runner, "_run", run)
    info = {"ip": "127.0.0.1", "port": 22, "key": "unused"}
    runner._sync_project(runner.Job(id="fake", kind="fake", params={}), info)
    assert set(captured) == {"src/main.py", "scripts/train.py", "data/train.jsonl"}


def test_ssh_key_override_does_not_require_provider_default_key(tmp_path, monkeypatch):
    key = tmp_path / "existing-key"
    key.write_text("fake key for path check", encoding="utf-8")
    monkeypatch.setattr(runner, "_project_env", lambda: {"GPUSHARE_SSH_KEY": str(key)})
    monkeypatch.setattr(runner, "_runpodctl", lambda: "fake-cli")
    monkeypatch.setattr(
        runner,
        "_capture",
        lambda *a, **kw: json.dumps(
            {"ip": "127.0.0.1", "port": 2222, "ssh_key": {"path": "/not/on/this/machine"}}
        ),
    )
    assert runner._ssh_info("fake-pod")["key"] == str(key)
