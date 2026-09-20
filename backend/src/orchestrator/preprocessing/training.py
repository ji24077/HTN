"""Durable optimize -> migrate -> train gates for the GPUShare native training pilot.

Reuses GPUShare's kernel translator, protected-source checks, report comparator and
paired benchmark. Proposals run through the normal worker queue, never SSH or the
backend process. Original files and the full training contract remain immutable.
"""

import base64
import hashlib
from datetime import UTC, datetime
from importlib.resources import files as package_files
from typing import Literal

from pydantic import Field

from ..llm import ModelClientError
from ..shared.protocol import Model, Requirements
from ..worker.executors.native_benchmark import SIZES, summarize_pairs
from .artifacts import bundle, encoded
from .portability.comparison import compare_reports
from .portability.kernels import convert_kernel
from .portability.scripts import (
    get_native_source,
    replace_native_source,
    translate_script,
    validate_demo_edit,
)
from .training_telemetry import record


class Preparation(Model):
    rationale: str = Field(min_length=1, max_length=2000)
    optimize: bool
    source_worker_id: str = Field(min_length=1, max_length=128)
    source_vram_mib: int = Field(ge=0)
    source_vendor: Literal["nvidia", "amd"]
    target_vendor: Literal["nvidia", "amd"]
    validation_steps: int = Field(default=8, ge=1, le=32)


class KernelProposal(Model):
    native_source: str = Field(min_length=1, max_length=60000)
    reason: str = Field(min_length=1, max_length=2000)


INSTRUCTIONS = """
You are proposing one native-code change in the training preparation loop.
Return propose_training_kernel with the complete native_source and a short reason.
Only the embedded CUDA/HIP native source can change. Preserve its exported C ABI,
float32 calculation, all input/output shapes and actual GPU execution. Preserve
the Python training loop, dataset, optimizer, precision, validation and step count.
For optimization reduce native allocation/transfer/kernel overhead. For migration
use the supplied target backend and initial deterministic translation. Preserve
error handling and free GPU allocations before every call returns; no static or
global allocation caches, host-side substitute computation, filesystem or network
operations. Use feedback from prior static checks, GPU tests and paired timings.
Only worker measurements can establish success. A failed performance gate means
the original stays active. Do not claim that native preprocessing speedup is a
whole-training speedup. Do not ask to change fixed acceptance criteria.
"""


def source(files, entrypoint):
    return base64.b64decode(files[entrypoint], validate=True).decode("utf-8")


def digest_native(script):
    return hashlib.sha256(get_native_source(script).encode()).hexdigest()


def blockers(findings):
    return [item for item in findings if item["severity"] == "blocker"]


def validate_preparation(plan, files, workload):
    prep = plan.preparation
    if prep is None:
        return
    if workload != "training" or plan.requirements.runtime != "cuda":
        raise ValueError("Native training preparation requires a CUDA/ROCm training worker")
    if prep.source_vendor != prep.target_vendor and prep.source_worker_id == plan.worker_id:
        raise ValueError("Cross-vendor migration requires distinct source and target workers")
    original = source(files, plan.entrypoint)
    template = (
        package_files("orchestrator.preprocessing.portability")
        .joinpath("native_training.py")
        .read_text()
    )
    if blockers(validate_demo_edit(template, original, "baseline")):
        raise ValueError(
            "Training preparation supports the unchanged GPUShare native training harness only"
        )
    backend = "cuda" if prep.source_vendor == "nvidia" else "hip"
    if blockers(convert_kernel(get_native_source(original), backend, backend)["findings"]):
        raise ValueError("Original native kernel contains unsupported constructs")
    # The source harness is immutable; inference/inspection/resume flags cannot
    # turn the full training phase into a no-op or a different workload.
    args = plan.run_args
    expected = {"--steps", "--out", "--checkpoint", "--expect-vendor"}
    if len(args) != 8 or set(args[::2]) != expected:
        raise ValueError("Native training needs steps, out, checkpoint and expect-vendor arguments")
    options = dict(zip(args[::2], args[1::2]))
    if not options["--steps"].isdigit() or int(options["--steps"]) < prep.validation_steps:
        raise ValueError("Full training steps must include at least the bounded validation work")
    if options["--expect-vendor"] != prep.target_vendor:
        raise ValueError("Full training must require the selected target vendor")
    for flag in ("--out", "--checkpoint"):
        if not options[flag].startswith("{output_dir}/"):
            raise ValueError("Native training reports and checkpoints must use {output_dir}")


def compatible(worker, requirements, vendor):
    caps = worker["capabilities"]
    providers = (caps.get("accelerator") or {}).get("providers", [])
    return (
        caps["runtime"] == "cuda"
        and caps["vram_mib"] >= requirements.vram_mib
        and "python_program" in caps["kinds"]
        and (
            "rocm" in providers
            if vendor == "amd"
            else "cuda" in providers and "rocm" not in providers
        )
    )


async def reserve(service, job, worker_id, requirements, vendor):
    workers = await service.store.eligible(job["job_id"])
    worker = next((w for w in workers if w["id"] == worker_id), None)
    if worker is None:
        return False
    if not compatible(worker, requirements, vendor):
        raise ValueError(
            "Selected preparation worker does not report the required GPU vendor/runtime"
        )
    if await service.store.reserve(job, 1, worker_ids=[worker_id]) != [worker_id]:
        return False
    job["data"]["workers"] = [worker_id]
    return True


async def begin(service, job):
    data = job["data"]
    data["training_preparation"] = {
        "accepted_hash": job["original_hash"],
        "attempts": {"optimization": 0, "migration": 0},
        "optimization": "pending",
        "migration": "pending",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
    }
    await service.store.save(job, "training_baseline", "Preparing the original GPU reference run.")


async def dispatch_check(service, job, stage, digest):
    from .projects import ProgramPlan

    data = job["data"]
    plan = ProgramPlan.model_validate(data["program_plan"])
    prep = plan.preparation
    target = stage == "migration"
    worker_id = plan.worker_id if target else prep.source_worker_id
    vendor = prep.target_vendor if target else prep.source_vendor
    requirements = (
        plan.requirements if target else Requirements(runtime="cuda", vram_mib=prep.source_vram_mib)
    )
    if not await reserve(service, job, worker_id, requirements, vendor):
        return
    seconds = min(
        plan.probe_timeout_seconds, int((job["deadline"] - datetime.now(UTC)).total_seconds())
    )
    if seconds <= 0:
        raise ValueError("Training preparation deadline exceeded")
    args = [
        "--steps",
        str(prep.validation_steps),
        "--expect-vendor",
        vendor,
        "--out",
        "{output_dir}/report.json",
        "--checkpoint",
        "{output_dir}/checkpoint.pt",
    ]
    work = service.task(
        job,
        stage,
        worker_id,
        digest,
        mode="program",
        entrypoint=plan.entrypoint,
        args=args,
        dependencies=plan.dependencies,
        timeout_seconds=seconds,
        program={**plan.model_dump(mode="json"), "probe": True, "workload": "training"},
        preparation={"stage": stage, "vendor": vendor, "entrypoint": plan.entrypoint},
    ).model_copy(
        update={
            "requirements": requirements,
            "allow_failover": False,
            # Optional work must not spend the reserved migration/training time
            # repeating infrastructure failures within a single candidate check.
            "max_attempts": 1 if stage == "optimization" else 3,
        }
    )
    work.payload["working_directory"] = plan.working_directory
    await service.dispatch(
        job,
        "training_" + stage + "_testing",
        f"Testing {stage} on {vendor}; full training remains gated.",
        [work],
    )


async def next_stage(service, job, stage):
    state = job["data"]["training_preparation"]
    if stage in {"baseline", "optimization"}:
        await service.store.save(
            job,
            "training_" + ("optimization" if stage == "baseline" else "migration"),
            "Reviewing optimization." if stage == "baseline" else "Reviewing target migration.",
        )
    else:
        state["status"] = "ready"
        job["data"]["program_hash"] = state["accepted_hash"]
        job["data"]["validated_hash"] = state["accepted_hash"]
        await service.store.save(
            job,
            "program_ready",
            "Preparation passed; releasing the validated artifact for full training.",
        )


async def rejected(service, job, stage, evidence):
    data = job["data"]
    state = data["training_preparation"]
    state["feedback"] = evidence
    data["last_failure"] = evidence
    record(job, stage, "rejected", evidence=evidence)
    data["checks"].append(
        {
            "round": state["attempts"].get(stage, 0),
            "stage": "training_" + stage,
            "passed": False,
            "details": evidence,
            "seeds": [],
        }
    )
    if stage == "baseline" or (
        stage == "migration" and state["attempts"][stage] >= data["limits"]["adaptations"]
    ):
        state["status"] = "failed"
        if stage == "migration":
            state[stage] = "failed"
        await service.store.save(
            job, "failed", f"Required {stage} validation failed; full training was not started."
        )
    elif state["attempts"][stage] >= data["limits"]["adaptations"]:
        state[stage] = "kept_baseline"
        record(
            job,
            stage,
            "kept_baseline",
            evidence={
                "reason": "No candidate passed correctness and performance gates within the attempt limit."
            },
        )
        await next_stage(service, job, stage)
    else:
        await service.store.save(
            job,
            "training_" + stage,
            f"{stage.capitalize()} failed its gate; feeding evidence into the next proposal.",
        )


def validate_result(result, baseline, vendor, script, steps):
    if result.get("ok") is not True:
        raise ValueError(str(result.get("error", "Worker check failed")))
    report = result.get("preparation", {}).get("report", {})
    comparison = compare_reports(baseline or report, report)
    if comparison["status"] != "passed":
        raise ValueError("Numerical/identity comparison failed: " + str(comparison))
    if (
        report.get("vendor") != vendor
        or report.get("global_step") != steps
        or report.get("start_step") != 0
        or report.get("steps_executed") != steps
        or report.get("native_kernel_source_sha256") != digest_native(script)
    ):
        raise ValueError(
            "Validation report does not match the frozen candidate, vendor, or step range"
        )
    if (
        baseline
        and report.get("torch_version", "").split("+", 1)[0]
        != baseline.get("torch_version", "").split("+", 1)[0]
    ):
        raise ValueError("Source and target PyTorch base versions differ")
    return report, comparison


def performance_gate(benchmark, original, candidate, vendor):
    if (
        not isinstance(benchmark, dict)
        or benchmark.get("correctness_verified") is not True
        or benchmark.get("same_gpu_and_process") is not True
        or benchmark.get("vendor") != vendor
        or benchmark.get("source_sha256")
        != {"baseline": digest_native(original), "candidate": digest_native(candidate)}
    ):
        raise ValueError("Missing comparable paired native benchmark evidence")
    cases = benchmark.get("cases", [])
    if len(cases) != len(SIZES) or {case.get("elements") for case in cases} != set(SIZES):
        raise ValueError("Native benchmark did not cover every fixed shape")
    summaries = [
        summarize_pairs(
            case["baseline_sample_seconds"],
            case["candidate_sample_seconds"],
            correctness_verified=case.get("correctness_verified") is True,
        )
        for case in cases
    ]
    if not all(item["performance_verified"] for item in summaries):
        raise ValueError(
            "Candidate did not meet the repeated native performance gate: " + str(summaries)
        )
    return summaries


async def advance(service, job):
    from .projects import ProgramPlan, decide

    data = job["data"]
    state = data["training_preparation"]
    plan = ProgramPlan.model_validate(data["program_plan"])
    prep = plan.preparation
    stage = job["phase"].split("_")[1]
    original_files = await service.store.files(job["job_id"], job["original_hash"])
    original = source(original_files, plan.entrypoint)
    if job["phase"].endswith("_rejected"):
        await rejected(service, job, stage, {"static_checks": state.pop("static_rejection")})
        return
    if job["phase"].endswith("_testing"):
        results = await service.outcomes(job, optional=stage == "optimization")
        if results is None:
            return
        result = results[0]
        digest = job["original_hash"] if stage == "baseline" else state["candidate_hash"]
        candidate_files = await service.store.files(job["job_id"], digest)
        candidate = source(candidate_files, plan.entrypoint)
        vendor = prep.target_vendor if stage == "migration" else prep.source_vendor
        try:
            report, comparison = validate_result(
                result, state.get("baseline"), vendor, candidate, prep.validation_steps
            )
            performance = None
            if stage == "optimization":
                performance = performance_gate(
                    result.get("preparation", {}).get("benchmark"), original, candidate, vendor
                )
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            await rejected(service, job, stage, {"error": str(exc)[:10000], "result": result})
            return
        if stage == "baseline":
            state["baseline"] = report
        else:
            state["accepted_hash"] = digest
            state[stage] = "accepted"
        state.pop("feedback", None)
        state.pop("previous_candidate", None)
        data.pop("last_failure", None)
        data["checks"].append(
            {
                "stage": job["phase"],
                "passed": True,
                "seeds": [],
                "round": state["attempts"].get(stage, 0),
                "details": {"comparison": comparison, "performance": performance, "report": report},
            }
        )
        record(
            job,
            stage,
            "passed",
            evidence={
                "comparison": comparison,
                "performance": performance,
                "report": report,
                "result": result,
            },
        )
        data.setdefault("measurements", []).append(
            {
                "stage": job["phase"],
                "worker_id": data["workers"][0],
                "tasks": 1,
                "trials": 0,
                "compute_seconds": result.get("compute_seconds", 0),
                "execution_seconds": result.get("metrics", {}).get("execution_seconds", 0),
                "observed_wall_seconds": result.get("metrics", {}).get("execution_seconds", 0),
                "output_bytes": result.get("metrics", {}).get("output_bytes", 0),
            }
        )
        await next_stage(service, job, stage)
        return
    if stage == "baseline":
        await dispatch_check(service, job, stage, job["original_hash"])
        return
    if stage == "optimization" and not prep.optimize:
        state[stage] = "skipped"
        record(job, stage, "skipped", evidence={"reason": prep.rationale})
        await next_stage(service, job, stage)
        return
    if stage == "optimization":
        remaining = (job["deadline"] - datetime.now(UTC)).total_seconds()
        migration_seconds = 0
        if prep.source_worker_id != plan.worker_id:
            migration_seconds = plan.probe_timeout_seconds + (
                90 if prep.source_vendor != prep.target_vendor else 0
            )
        needed = plan.run_timeout_seconds + migration_seconds + plan.probe_timeout_seconds
        if not job["phase"].endswith("_ready"):
            needed += 90
        if remaining < needed:
            state[stage] = "kept_baseline"
            record(
                job,
                stage,
                "kept_baseline",
                evidence={"reason": "Preserving time for required migration and full training."},
            )
            await next_stage(service, job, stage)
            return
    if job["phase"].endswith("_ready"):
        await dispatch_check(service, job, stage, state["candidate_hash"])
        return
    # Optional optimization exhaustion must not contaminate migration feedback.
    if stage == "migration" and state["attempts"][stage] == 0:
        state.pop("feedback", None)
        state.pop("previous_candidate", None)
    if (
        stage == "migration"
        and prep.source_vendor == prep.target_vendor
        and prep.source_worker_id == plan.worker_id
    ):
        state[stage] = "skipped"
        record(
            job,
            stage,
            "skipped",
            evidence={"reason": "The tested source worker is the selected training target."},
        )
        await next_stage(service, job, stage)
        return
    accepted_files = await service.store.files(job["job_id"], state["accepted_hash"])
    accepted = source(accepted_files, plan.entrypoint)
    if stage == "migration" and prep.source_vendor == prep.target_vendor:
        # Moving unchanged code between same-vendor machines still needs target validation.
        state["candidate_hash"] = state["accepted_hash"]
        state["attempts"][stage] += 1
        await service.store.save(
            job,
            "training_migration_ready",
            "Moving the accepted artifact to the target worker for validation.",
        )
        return
    seed = (
        translate_script(accepted, prep.target_vendor)["source"]
        if stage == "migration"
        else accepted
    )
    try:
        proposal = await decide(
            service,
            job,
            KernelProposal,
            "propose_training_kernel",
            proposal_instructions=INSTRUCTIONS,
            allow_control=False,
            preparation_stage=stage,
            native_source=state.get("previous_candidate", get_native_source(seed)),
            preparation_feedback=state.get("feedback"),
            preparation_baseline=state["baseline"],
            source_vendor=prep.source_vendor,
            target_vendor=prep.target_vendor,
            previous_preparation_attempts=state["attempts"],
        )
    except (ModelClientError, TimeoutError, ValueError) as exc:
        state["attempts"][stage] += 1
        await rejected(
            service, job, stage, {"error": str(exc)[:2000], "error_type": type(exc).__name__}
        )
        return
    if proposal is None:
        return
    state["attempts"][stage] += 1
    state["previous_candidate"] = proposal.native_source
    candidate = replace_native_source(accepted, proposal.native_source)
    findings = validate_demo_edit(original, candidate, stage)
    backend = (
        "hip"
        if (prep.target_vendor if stage == "migration" else prep.source_vendor) == "amd"
        else "cuda"
    )
    findings += convert_kernel(proposal.native_source, backend, backend)["findings"]
    files = {**accepted_files, plan.entrypoint: encoded(candidate)}
    # The paired benchmark reads the immutable original native source, not a model-provided baseline.
    files["__dispatch_native_baseline__.py"] = encoded(original)
    digest, raw = bundle(files)
    data["versions"].append(
        {
            "round": len(data["versions"]) + 1,
            "stage": "training_" + stage,
            "digest": digest,
            "explanation": proposal.reason,
            "code": candidate,
        }
    )
    state["candidate_hash"] = digest
    if blockers(findings):
        # Retain rejected code as an artifact before recording its failed check.
        state["static_rejection"] = findings
        record(job, stage, "rejected", evidence={"static_checks": findings})
        await service.store.save(
            job,
            "training_" + stage + "_rejected",
            "Candidate rejected by static checks.",
            artifacts=[(digest, raw)],
        )
        return
    await service.store.save(
        job,
        "training_" + stage + "_ready",
        "Candidate frozen; scheduling worker validation.",
        artifacts=[(digest, raw)],
    )
