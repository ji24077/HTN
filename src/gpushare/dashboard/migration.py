"""Bounded NVIDIA-to-AMD training continuation over already-authorized SSH hosts.

This module never provisions resources or sends API credentials to GPU hosts.
The caller owns rental authorization, budget enforcement and pod termination.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

from gpushare.agent.evaluation import evaluation_identity, require_comparable
from gpushare.agent.task import REQUIRED_FIELDS

PYTHON = ".migration-venv/bin/python"
INFERENCE = {"batch": 8, "dtype": "bf16", "fuse_adapter": False, "length_bucketing": False}


def _gate(before: dict, after: dict, *, tolerance: float = 0.02) -> dict:
    from gpushare.dashboard import runner as r

    try:
        require_comparable(
            after,
            before["evaluation"],
            strict_inference=True,
            inference_settings=before["inference"],
        )
        if any(set(report["field_accuracy"]) != set(REQUIRED_FIELDS) for report in (before, after)):
            raise ValueError("all five per-field scores are required")
    except (KeyError, ValueError) as exc:
        raise r.JobError(f"migration evaluation is not comparable: {exc}") from exc
    validation = r._quality(before, after, tol=tolerance)
    if validation["status"] != "ok":
        raise r.JobError(f"migration quality gate rejected regression: {validation}")
    return validation


def _preflight(job, info, vendor: str) -> dict:
    from gpushare.dashboard import runner as r

    argv = [
        PYTHON,
        "scripts/check_env.py",
        "--json",
        "--expect-vendor",
        vendor,
        "--require-torch-version",
        "2.10.0",
        "--smoke",
        "--dtype",
        "bf16",
    ]
    command = f"cd {shlex.quote(r.REMOTE_ROOT)} && {shlex.join(argv)}"
    report = json.loads(r._capture(r._ssh_args(info, command), timeout=120))
    if report.get("vendor") != vendor or report.get("smoke_test") != "passed":
        raise r.JobError(f"{vendor} GPU preflight did not pass")
    for package, expected in (("transformers", "5.17.0"), ("peft", "0.21.0")):
        if report.get(package) != expected:
            raise r.JobError(f"migration requires {package} {expected}")
    r.JOBS.log(job, f"preflight passed: {vendor}, {report['gpu']}, torch {report['torch']}")
    return report


def _verify_remote(job, info, remote: str):
    from gpushare.dashboard import runner as r

    code = (
        "import sys; from gpushare.trainer.checkpoint import verify_training_bundle; "
        "m=verify_training_bundle(sys.argv[1]); print('verified checkpoint step', m['global_step'])"
    )
    r._remote(job, info, [PYTHON, "-c", code, remote])


def run_resume_migration(
    job,
    *,
    source: dict,
    target: dict,
    src_info: dict,
    dst_info: dict,
    total_steps: int = 8,
    stop_after: int = 4,
    eval_n: int = 50,
    initial_adapter: Path | None = None,
    prepare_pods: bool = True,
    train_data: Path | None = None,
    eval_data: Path | None = None,
) -> dict:
    """Train/pause, transfer/verify, evaluate BEFORE resume, continue and gate.

    Failure leaves reports on disk, never changes latest_run, and never
    terminates someone else's pods. Resource teardown belongs to the caller.
    """
    from gpushare.dashboard import runner as r
    from gpushare.trainer.checkpoint import verify_training_bundle

    if source["vendor"] != "nvidia" or target["vendor"] != "amd":
        raise r.JobError("full resume requires an NVIDIA source and AMD target")
    if not 0 < stop_after < total_steps <= 10000 or not 1 <= eval_n <= 2000:
        raise r.JobError("require 0 < stop_after < total_steps <= 10000 and 1 <= eval_n <= 2000")
    local = r.RUN_ROOT / job.id
    inputs = local / "input"
    inputs.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, path in (
        ("train.jsonl", train_data or r.ROOT / "data/train.jsonl"),
        ("heldout.jsonl", eval_data or r.ROOT / "data/heldout.jsonl"),
    ):
        payload = path.read_bytes()
        (inputs / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    rows = [
        json.loads(line)
        for line in (inputs / "heldout.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:eval_n]
    if not rows:
        raise r.JobError("migration evaluation dataset is empty")
    identity = evaluation_identity(rows, max_new_tokens=128, seq_len=256)
    remote_run = f"{r.REMOTE_ROOT}/.runs/{job.id}"
    relative_run = f".runs/{job.id}"
    source_ckpt, resumed_ckpt = f"{remote_run}/source-ckpt", f"{remote_run}/resumed-ckpt"
    local_eval = local / "eval"
    local_eval.mkdir()

    r.JOBS.update(job, "preparing and checking both GPU backends", 5)
    preflight = {}
    for side, pod, info in (("source", source, src_info), ("target", target, dst_info)):
        r._sync_project(job, info)
        if prepare_pods:
            r._setup_migration_pod(job, info, pod["vendor"])
        preflight[side] = _preflight(job, info, pod["vendor"])
        r._push(job, info, inputs, f"{remote_run}/input")
        verify_inputs = (
            "import hashlib,json,pathlib,sys; base=pathlib.Path(sys.argv[1]); "
            "expected=json.loads(sys.argv[2]); "
            "assert all(hashlib.sha256((base/k).read_bytes()).hexdigest()==v for k,v in expected.items()), 'input hash mismatch'"
        )
        r._remote(
            job, info, [PYTHON, "-c", verify_inputs, f"{remote_run}/input", json.dumps(hashes)]
        )
        r._run(job, r._ssh_args(info, f"mkdir -p {shlex.quote(remote_run + '/eval')}"))
    (local / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")

    training = [
        "--method",
        "lora",
        "--data",
        f"{relative_run}/input/train.jsonl",
        "--job-id",
        job.id,
        "--steps",
        str(total_steps),
        "--warmup-measure",
        "0",
        "--seq-len",
        "256",
        "--dtype",
        "bf16",
        "--attention",
        "sdpa",
        "--micro-batch",
        "2",
        "--grad-accum",
        "2",
        "--lr",
        "0.0001",
        "--seed",
        "1337",
        "--sampling",
        "shuffle",
        "--loss-chunk",
        "64",
    ]
    init = []
    if initial_adapter is not None:
        if not (initial_adapter / "adapter_config.json").is_file():
            raise r.JobError("initial_adapter must be an existing local adapter directory")
        r._push(job, src_info, initial_adapter, f"{remote_run}/initial-adapter")
        init = ["--init-adapter", f"{relative_run}/initial-adapter"]
    r.JOBS.update(job, "training NVIDIA source to checkpoint boundary", 20)
    r._remote(
        job,
        src_info,
        [
            PYTHON,
            "scripts/train.py",
            *training,
            *init,
            "--stop-after",
            str(stop_after),
            "--out",
            f"{relative_run}/source-ckpt",
        ],
    )
    _verify_remote(job, src_info, source_ckpt)

    eval_common = [
        "--data",
        f"{relative_run}/input/heldout.jsonl",
        "--n",
        str(len(rows)),
        "--batch",
        "8",
        "--dtype",
        "bf16",
        "--seq-len",
        "256",
        "--max-new-tokens",
        "128",
    ]
    r._remote(
        job,
        src_info,
        [
            PYTHON,
            "scripts/evaluate.py",
            "--model",
            f"{relative_run}/source-ckpt",
            "--out",
            f"{relative_run}/eval/source.json",
            *eval_common,
        ],
    )
    r._pull(job, src_info, f"{remote_run}/eval", local_eval)
    baseline = r._json(local_eval / "source.json")
    require_comparable(baseline, identity, strict_inference=True, inference_settings=INFERENCE)
    if baseline["json_parse_rate"] <= 0 or baseline["exact_match_rate"] <= 0:
        raise r.JobError(
            "source baseline has zero valid JSON or zero correct records; "
            "a preservation comparison would be meaningless. Use a validated initial adapter."
        )

    r.JOBS.update(job, "relaying and verifying complete training bundle", 45)
    bundle = local / "source-ckpt"
    r._pull(job, src_info, source_ckpt, bundle)
    paused = verify_training_bundle(bundle)
    if paused["global_step"] != stop_after or paused["job_id"] != job.id:
        raise r.JobError("source checkpoint did not preserve requested step/job identity")
    r._push(job, dst_info, bundle, source_ckpt)
    _verify_remote(job, dst_info, source_ckpt)
    r._push(job, dst_info, local_eval / "source.json", f"{remote_run}/eval")

    def evaluate_target(model_path: str, name: str):
        code = r._remote(
            job,
            dst_info,
            [
                PYTHON,
                "scripts/evaluate.py",
                "--model",
                model_path,
                "--out",
                f"{relative_run}/eval/{name}.json",
                *eval_common,
                "--compare",
                f"{relative_run}/eval/source.json",
                "--strict-inference",
            ],
            allow_failure=True,
        )
        r._pull(job, dst_info, f"{remote_run}/eval", local_eval)
        report = r._json(local_eval / f"{name}.json")
        gate = _gate(baseline, report)
        if code:
            raise r.JobError(f"target evaluation exited with status {code}")
        return report, gate

    r.JOBS.update(job, "checking transferred weights BEFORE any AMD optimizer step", 60)
    before_resume, transfer_gate = evaluate_target(
        f"{relative_run}/source-ckpt", "target-before-resume"
    )
    # No call to the trainer is reachable until the transferred checkpoint passes.
    r.JOBS.update(job, "restoring optimizer/sampler and continuing on AMD", 75)
    r._remote(
        job,
        dst_info,
        [
            PYTHON,
            "scripts/train.py",
            *training,
            "--resume",
            f"{relative_run}/source-ckpt",
            "--out",
            f"{relative_run}/resumed-ckpt",
        ],
    )
    _verify_remote(job, dst_info, resumed_ckpt)
    r.JOBS.update(job, "validating continued job and completed training state", 90)
    after, final_gate = evaluate_target(f"{relative_run}/resumed-ckpt", "after")
    r._pull(job, dst_info, resumed_ckpt, local / "ckpt")
    completed = verify_training_bundle(local / "ckpt")
    if completed["global_step"] != total_steps or completed["job_id"] != job.id:
        raise r.JobError("resumed checkpoint has the wrong global step/job identity")
    result = {
        "migration_mode": "full_training_resume",
        "source": source,
        "target": target,
        "source_step": stop_after,
        "resumed_to_step": total_steps,
        "input_sha256": hashes,
        "preflight": preflight,
        "before": baseline,
        "target_before_resume": before_resume,
        "after": after,
        "transfer_validation": transfer_gate,
        "validation": final_gate,
        "remote_checkpoint": resumed_ckpt,
        "local_dir": str(local),
        "reproducibility": "Optimizer/sampler state restored; cross-backend floating-point trajectories need not be bit-identical.",
        "teardown": "caller must terminate authorized rental resources",
    }
    (local / "migration-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    r._write_latest(
        {
            "job_id": job.id,
            "pod_id": target["id"],
            "pod": target,
            "local_dir": str(local),
            "remote_checkpoint": resumed_ckpt,
            "migrated_from": source["id"],
            "migration_mode": "full_training_resume",
        }
    )
    return result
