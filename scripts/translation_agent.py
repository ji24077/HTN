"""Prepare source proposals or verify them on explicitly supplied GPU endpoints."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from dotenv import dotenv_values

from gpushare.portability.engine import EngineeringAgent
from gpushare.portability.memory import SkillStore
from gpushare.portability.providers import ModelClient, ModelError

ROOT = Path(__file__).resolve().parents[1]
_SECRETS = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|rpa_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{16,}"
    r"|[A-Za-z0-9]{8}\.[A-Za-z0-9]{24,})"
)
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _known_secrets() -> tuple[str, ...]:
    environment = {**dotenv_values(ROOT / ".env"), **os.environ}
    return tuple(
        value for key, value in environment.items()
        if isinstance(value, str) and len(value) >= 8
        and any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    )


def _scrub(value, secrets: tuple[str, ...]):
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return _SECRETS.sub("[REDACTED]", value)
    if isinstance(value, list):
        return [_scrub(item, secrets) for item in value]
    if isinstance(value, dict):
        return {_scrub(key, secrets): _scrub(item, secrets) for key, item in value.items()}
    return value


def _persist_result(path: Path, result: dict) -> None:
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".result-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.025 * (attempt + 1))
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _record_lessons(
    result: dict, store: SkillStore, project_id: str, execution_result_path: Path
) -> None:
    """One demo can propose lessons; only the separate fixed suite may promote."""
    run_summary_sha256 = hashlib.sha256(execution_result_path.read_bytes()).hexdigest()
    proposals = []
    result["skill_memory"] = {
        "project_id": project_id,
        "store": str(store.root.resolve()),
        "status": "no_candidates",
        "promoted": False,
        "execution_result": str(execution_result_path.resolve()),
        "execution_result_sha256": run_summary_sha256,
        "proposals": proposals,
    }
    for lesson in result.get("lessons", []):
        phase = lesson.get("phase")
        source_hash = lesson.get("evidence_source_sha256")
        proposal = {"phase": phase, "status": "rejected", "candidate_id": None}
        proposals.append(proposal)
        if (
            phase not in {"optimize", "translate", "target_optimize"}
            or not isinstance(source_hash, str)
            or not _HASH.fullmatch(source_hash)
        ):
            proposal["reason"] = "Candidate lesson lacks a valid phase or source hash"
            result["skill_memory"]["status"] = "candidate_storage_failed"
            continue
        matching_attempts = [
            attempt for attempt in result.get("attempts", [])
            if attempt.get("phase") == phase and attempt.get("source_sha256") == source_hash
        ]
        gpu_verified = any(attempt.get("gpu_verified") is True for attempt in matching_attempts)
        evidence = {
            "run_status": result["status"],
            "run_summary_sha256": run_summary_sha256,
            "run_summary_artifact": str(execution_result_path.resolve()),
            "source_sha256": result["source_sha256"],
            "skills_sha256": result["skills_sha256"],
            "candidate_source_sha256": source_hash,
            "phase": phase,
            "target": result["target"],
            "gpu_verified": gpu_verified,
            "verification_scope": "single_demo_only",
            "fixed_suite_verified": False,
            "performance_verified": result.get("performance_verified") is True,
            "attempt_statuses": [attempt.get("status") for attempt in matching_attempts],
        }
        try:
            candidate_id = store.propose(
                lesson["text"], {"kind": "project", "project_id": project_id}, evidence
            )
        except (OSError, ValueError, KeyError):
            proposal["reason"] = "Could not store candidate; check skill-store state and lesson evidence"
            result["skill_memory"]["status"] = "candidate_storage_failed"
            continue
        status = "candidate_pending_regression" if gpu_verified else "candidate_pending_gpu_and_regression"
        proposal.update(candidate_id=candidate_id, status=status, gpu_verified=gpu_verified)
        lesson.update(candidate_id=candidate_id, status=status)
    if proposals and result["skill_memory"]["status"] != "candidate_storage_failed":
        result["skill_memory"]["status"] = "candidates_pending_regression"


def _read_source(path: Path) -> str:
    with path.open("rb") as handle:
        payload = handle.read(80_001)
    if len(payload) > 80_000:
        raise ValueError("Source exceeds the first-demo size limit")
    return payload.decode("utf-8")


def _failure(exc: Exception) -> dict:
    if isinstance(exc, FileExistsError):
        detail = "Output directory already exists; choose a new --out path."
    elif isinstance(exc, ModelError):
        detail = "Coding-model request or configuration failed; check provider access and request limits."
    elif isinstance(exc, (ValueError, KeyError, TypeError)):
        detail = "Input or configuration is invalid; check the source, skill store, connection file and limits."
    elif isinstance(exc, OSError):
        detail = "Cannot read or write required local files; check paths and permissions."
    else:
        detail = "Translation agent failed; no raw exception details were exposed."
    return {"status": "failed", "error_type": type(exc).__name__, "error": detail}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--target", choices=("amd", "nvidia"), default="amd")
    parser.add_argument("--provider", choices=("baseten", "openai"), default="baseten")
    parser.add_argument("--model-budget", type=float, default=1.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--connections", type=Path, help="Explicit source_info/target_info/work_dir JSON; no Pod rental occurs here")
    parser.add_argument("--project-id", default="demo", help="Project scope for candidate engineering lessons")
    parser.add_argument("--skills-dir", type=Path, help="Default: .gpushare/skill-memory/<project-id>")
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise FileExistsError()
        source = _read_source(args.source)
        if not 1 <= args.attempts <= 5 or not 0 < args.model_budget <= 5:
            raise ValueError("Invalid repair or model spending limit")
        secrets = _known_secrets()
        store = SkillStore(
            args.skills_dir or ROOT / ".gpushare" / "skill-memory" / args.project_id,
            project_id=args.project_id,
        )
        skills = store.active_text()
        validator = None
        if args.connections:
            from gpushare.portability.validation import RemoteDemoValidator

            connections = json.loads(args.connections.read_text(encoding="utf-8"))
            validator = RemoteDemoValidator(
                connections["source_info"], connections["target_info"], connections["work_dir"],
                args.out.parent / (args.out.name + "-verification"), args.source,
                source_vendor="nvidia" if args.target == "amd" else "amd", target_vendor=args.target,
            )
        client = ModelClient(args.provider, ROOT / ".env", budget_usd=args.model_budget)
        agent = EngineeringAgent(
            client, skills, max_attempts=args.attempts,
            event=lambda item: print(json.dumps(_scrub(item, secrets)), flush=True),
        )
        result = _scrub(agent.run(source, args.out, target=args.target, validator=validator), secrets)
        execution_result_path = args.out / "execution-result.json"
        _persist_result(execution_result_path, result)
        _record_lessons(result, store, args.project_id, execution_result_path)
        result = _scrub(result, secrets)
        _persist_result(args.out / "result.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        successful = result["status"] in {"prepared_unverified", "verified_correctness"}
        return 0 if successful and result["skill_memory"]["status"] != "candidate_storage_failed" else 1
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "error": "Translation agent interrupted."}), file=sys.stderr)
        return 130
    except Exception as exc:
        print(json.dumps(_failure(exc)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
