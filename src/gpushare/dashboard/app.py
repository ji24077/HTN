"""Ji — S-3. The dashboard, and the four agent actions behind it.

    make ui          # http://127.0.0.1:8080

Serves one page and a small JSON API. It reads the REAL artefacts on disk —
eval/base.json, eval/after.json, ckpt/*/meta.json — so what you see is the
experiment that ran, not a mock. Where an artefact is missing, the UI says so
rather than filling the gap with a plausible number.

STANDALONE ON PURPOSE. The handbook has this page served by Jack's hub (S-1),
which does not exist yet. Rather than block, this carries its own tiny server
in Ji's directory. When the hub lands, the page moves and this file goes away;
the JSON shapes are the only thing to keep.

THE FOUR ACTIONS ARE MANUAL, AND THE VALIDATION AFTER THEM IS NOT. You click
migrate; the agent decides; the eval re-runs and either clears the gate or does
not. An action whose validation has not run yet reports `pending` — it never
reports success on the strength of having been requested.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from gpushare.agent import calibrate
from gpushare.agent.llm import choose
from gpushare.agent.simulate import (
    SimProber,
    predict_peak_vram_gb,
    predict_t_step,
)
from gpushare.agent.specs import CHIPS, MODELS, NETS, ChipSpec
from gpushare.agent.task import MODEL_ID, PROMPT, REQUIRED_FIELDS
from gpushare.contracts import JobConfig
from gpushare.dashboard.runner import (
    JOBS,
    JobError,
    available_models,
    generate,
    generate_stream,
    latest_run,
    list_pods,
    restore_serving,
    serving,
    start_data_generation,
    start_inference_optimization,
    start_inference_server,
    start_migration,
    start_training,
    start_training_optimization,
    stop_inference_server,
)
from gpushare.dashboard.runner import (
    set_prefix as runner_set_prefix,
)
from gpushare.relay.marketplace import (
    ProviderAgent,
    ProviderRegistry,
    community_offer,
    default_mvp_offers,
    offer_from_runpod,
)
from gpushare.relay.models import (
    ApprovalKind,
    GPUOffer,
    LeasePolicy,
    QualityMetrics,
    RelayGoal,
)
from gpushare.relay.service import ApprovalRequired, RelayError, RelayService

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).parent / "static"
MODEL_KEY = "qwen2.5-0.5b"
RELAY = RelayService(skills_root=ROOT / "skills", state_path=ROOT / ".gpushare/relay-runs.json")
PROVIDERS = ProviderRegistry(ROOT / ".gpushare/provider-offers.json")


# ─────────────────────────────────────────────────────────────────────────────
# Reading what actually ran
# ─────────────────────────────────────────────────────────────────────────────
def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def experiment() -> dict[str, Any]:
    """The recorded run. Returns None-valued fields rather than invented ones."""
    latest = latest_run()
    latest_dir = Path(latest["local_dir"]) if latest and latest.get("local_dir") else None
    before_path = latest_dir / "eval/base.json" if latest_dir else ROOT / "eval/base.json"
    after_path = latest_dir / "eval/after.json" if latest_dir else ROOT / "eval/after.json"
    meta_path = latest_dir / "ckpt/meta.json" if latest_dir else ROOT / "ckpt/run/meta.json"
    before = _read(before_path) or _read(ROOT / "eval/base.json")
    after = _read(after_path) or _read(ROOT / "eval/after.json")
    meta = _read(meta_path) or _read(ROOT / "ckpt/run/meta.json")

    out: dict[str, Any] = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "fields": list(REQUIRED_FIELDS),
        "example": {
            "sentence": "Sarah Chen, 34, joined Anthropic in 2023 as a research engineer.",
            "record": {
                "name": "Sarah Chen",
                "age": 34,
                "org": "Anthropic",
                "role": "research engineer",
                "year": 2023,
            },
        },
        "before": before,
        "after": after,
        "train": meta,
        "data": {
            "train": _count(ROOT / "data/train.jsonl"),
            "heldout": _count(ROOT / "data/heldout.jsonl"),
        },
        "latest_run": latest,
    }
    out["cost_model"] = _cost_model_check(meta)
    return out


def _count(p: Path) -> int:
    try:
        return sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
    except OSError:
        return 0


def _cost_model_check(meta: dict | None) -> dict | None:
    """Predicted vs measured, shown on the page rather than kept in a commit.

    A projection without its error is a claim. This panel is what lets the
    agent's ETA numbers be read as "modelled, and here is how far off the model
    was last time" instead of as measurements.
    """
    if not meta:
        return None
    chip = _chip_by_gpu(meta.get("gpu", ""))
    if chip is None:
        return None
    model = MODELS[MODEL_KEY]
    cal = calibrate.DEFAULTS.get(chip.chip_class, calibrate.COLD_START)
    cfg = _cfg(meta, workers=1)

    pt = predict_t_step(cfg=cfg, model=model, chip=chip, cal=cal)
    pv = predict_peak_vram_gb(cfg=cfg, model=model, cal=cal)
    mt, mv = meta["t_step_median_s"], meta["peak_vram_gb"]
    return {
        "chip": chip.name,
        "t_step": {"predicted": pt, "measured": mt, "pct": abs(pt - mt) / mt * 100},
        "peak_vram_gb": {"predicted": pv, "measured": mv, "pct": abs(pv - mv) / mv * 100},
        "note": (
            "Memory holds; time does not. Inverting the model against this run "
            "returns an MFU above 1.0, which is impossible — so the time model has "
            "a structural error, not a calibration gap. Treat every projected "
            "runtime below as unreliable until a batch sweep fixes the curve."
        ),
    }


def _chip_by_gpu(name: str) -> ChipSpec | None:
    for c in CHIPS.values():
        if c.name.lower() in name.lower():
            return c
    return None


def _cfg(meta: dict, *, workers: int, **over) -> JobConfig:
    base = dict(
        job_id="j1",
        model=MODEL_KEY,
        total_steps=meta.get("steps", 500),
        seq_len=meta["seq_len"],
        dtype=meta["dtype"],
        attention=meta["attention"],
        micro_batch=meta["micro_batch"],
        grad_accum=meta["grad_accum"],
        H=190,
        workers=[f"w{i}" for i in range(workers)],
    )
    base.update(over)
    base["global_batch_tokens"] = (
        base["micro_batch"] * base["grad_accum"] * workers * base["seq_len"]
    )
    return JobConfig(**base)


# ─────────────────────────────────────────────────────────────────────────────
# The four actions
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ActionResult:
    action: str
    from_chip: str | None
    to_chip: str | None
    decision: dict[str, Any]
    reason: str
    decided_by: str
    projection: dict[str, Any]
    validation: dict[str, Any]


def _candidates_for_speed(meta: dict, chip: ChipSpec) -> list[JobConfig]:
    """Rule-generated config candidates, best-first, all doing IDENTICAL work.

    Every candidate keeps micro_batch x grad_accum constant, so the LLM cannot
    pick a 'faster' option that simply does less — JobConfig would refuse to
    construct it anyway, which is the point of the invariant living in the
    contract rather than here.
    """
    mb, ga = meta["micro_batch"], meta["grad_accum"]
    total = mb * ga
    out = []
    for m in (32, 16, 8, 4):
        if total % m:
            continue
        cfg = _cfg(
            meta, workers=1, micro_batch=m, grad_accum=total // m, dtype="bf16", attention="sdpa"
        )
        sim = SimProber(NETS["runpod-global"])
        if sim.fits(cfg, chip):
            out.append(cfg)
    return out or [_cfg(meta, workers=1)]


def _pick_target(want: str, exclude: ChipSpec | None) -> ChipSpec | None:
    """Cheapest chip in the wanted class that is not the one we are on."""
    fits = [
        c
        for c in CHIPS.values()
        if c.chip_class == want and c.trainable() and (exclude is None or c.name != exclude.name)
    ]
    return min(fits, key=lambda c: c.credits_per_hour) if fits else None


def run_action(name: str) -> ActionResult:
    meta = _read(ROOT / "ckpt/run/meta.json")
    if not meta:
        raise ValueError("no ckpt/run/meta.json — run scripts/train.py on a pod first")
    here = _chip_by_gpu(meta.get("gpu", "")) or CHIPS["RTX 4090"]
    model = MODELS[MODEL_KEY]

    if name in ("migrate-amd-nvidia", "migrate-nextgen"):
        src = CHIPS["MI300X"] if name == "migrate-amd-nvidia" else _pick_target("ampere_24gb", None)
        dst = _pick_target("ada_24gb", None)
        return _migration(name, src, dst, meta, model)

    if name == "optimize-training":
        return _optimize_training(meta, here, model)

    if name == "optimize-inference":
        return _optimize_inference(meta, here, model)

    raise ValueError(f"unknown action {name!r}")


def _migration(name, src: ChipSpec, dst: ChipSpec, meta: dict, model) -> ActionResult:
    cfg = _cfg(meta, workers=1)
    proj = {}
    for label, c in (("from", src), ("to", dst)):
        cal = calibrate.DEFAULTS.get(c.chip_class, calibrate.COLD_START)
        t = predict_t_step(cfg=cfg, model=model, chip=c, cal=cal)
        proj[label] = {
            "chip": c.name,
            "chip_class": c.chip_class,
            "rate_per_hour": c.credits_per_hour,
            "t_step_s": t,
            "cost_to_finish": t * meta.get("steps", 500) / 3600 * c.credits_per_hour,
        }

    # The number that actually decides. Cheapest per HOUR and cheapest per JOB
    # are different questions, and the second is the one the user pays.
    cheaper = proj["to"]["cost_to_finish"] < proj["from"]["cost_to_finish"]
    reason = (
        f"{dst.name} finishes the same 500 steps for "
        f"{proj['to']['cost_to_finish']:.2f} vs {proj['from']['cost_to_finish']:.2f} credits."
        if cheaper
        else f"{dst.name} costs more per job than {src.name}; migrating would not pay for itself."
    )
    return ActionResult(
        action=name,
        from_chip=src.name,
        to_chip=dst.name,
        decision={"migrate": cheaper, "waits_for_sync": True},
        reason=reason,
        decided_by="rules",
        projection=proj,
        validation=_validation_for(dst),
    )


def _optimize_training(meta: dict, chip: ChipSpec, model) -> ActionResult:
    cands = _candidates_for_speed(meta, chip)
    sim = SimProber(NETS["runpod-global"])
    probes = [sim.probe(c, chip, config_name="optimized") for c in cands]
    d = choose(cands, probes, rules_reason="Largest resident batch that still fits the card.")

    i = cands.index(d.config)
    return ActionResult(
        action="optimize-training",
        from_chip=chip.name,
        to_chip=chip.name,
        decision={
            "dtype": d.config.dtype,
            "attention": d.config.attention,
            "micro_batch": d.config.micro_batch,
            "grad_accum": d.config.grad_accum,
            "tokens_per_step": d.config.global_batch_tokens,
        },
        reason=d.reason,
        decided_by=d.decided_by,
        projection={
            "candidates": [
                {
                    "micro_batch": c.micro_batch,
                    "grad_accum": c.grad_accum,
                    "tokens_per_step": c.global_batch_tokens,
                    "t_step_s": p.t_step_median_s,
                    "peak_vram_gb": p.peak_vram_gb,
                    "chosen": k == i,
                }
                for k, (c, p) in enumerate(zip(cands, probes, strict=True))
            ],
            "measured_baseline_t_step_s": meta["t_step_median_s"],
            "PROJECTED": True,
        },
        validation=_validation_for(chip),
    )


def _optimize_inference(meta: dict, chip: ChipSpec, model) -> ActionResult:
    """Designed, not measured. The panel says so rather than showing a number."""
    kv_per_seq = 2 * model.layers * model.d_model * 512 * 2
    free = (chip.vram_gb - model.params * 2 / 1e9) * 0.9
    return ActionResult(
        action="optimize-inference",
        from_chip=chip.name,
        to_chip=chip.name,
        decision={"engine": "vLLM", "concurrency": 64, "ignore_eos": True, "max_tokens": 128},
        reason=(
            "Decode is memory-bound, so throughput comes from batching, not FLOPs. "
            "Measure at concurrency 64 with ignore_eos and a fixed max_tokens — at "
            "concurrency 1 vLLM and a plain generate loop do the same work and the "
            "comparison shows nothing."
        ),
        decided_by="rules",
        projection={
            "mem_bw_gbs": chip.mem_bw_gbs,
            "kv_bytes_per_seq_512": kv_per_seq,
            "max_concurrent_seqs": int(free * 1e9 / kv_per_seq),
            "NOT_MEASURED": True,
        },
        validation={
            "status": "not_built",
            "detail": "The inference axis has equations but no runner yet.",
        },
    )


def _validation_for(chip: ChipSpec) -> dict[str, Any]:
    """Has the eval actually been re-run on this chip?

    Only a recorded eval counts. An action that has not been validated says
    `pending`, never `ok` — being requested is not evidence.
    """
    before, after = _read(ROOT / "eval/base.json"), _read(ROOT / "eval/after.json")
    path = ROOT / f"eval/after-{chip.chip_class}.json"
    on_target = _read(path)
    if on_target is None:
        return {
            "status": "pending",
            "detail": f"no {path.relative_to(ROOT)} — run scripts/evaluate.py on {chip.name} "
            f"with --compare eval/after.json",
        }
    if after is None:
        return {"status": "pending", "detail": "no baseline eval to compare against"}
    dp = on_target["json_parse_rate"] - after["json_parse_rate"]
    de = on_target["exact_match_rate"] - after["exact_match_rate"]
    ok = dp >= -0.02 and de >= -0.02
    return {
        "status": "ok" if ok else "regressed",
        "json_parse_rate": on_target["json_parse_rate"],
        "exact_match_rate": on_target["exact_match_rate"],
        "delta_parse": dp,
        "delta_exact": de,
        "detail": (
            "within tol=0.02 — the action preserved the model"
            if ok
            else "beyond tol=0.02 — the action broke the model"
        ),
    }
    _ = before  # kept for symmetry with the CLI gate


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────
class DataRequest(BaseModel):
    total: int = Field(default=2000, ge=100, le=10_000)
    heldout: int = Field(default=200, ge=20, le=2000)
    workers: int = Field(default=8, ge=1, le=16)


class TrainRequest(BaseModel):
    pod_id: str
    steps: int = Field(default=500, ge=10, le=10_000)
    dtype: str = "bf16"
    attention: str = "sdpa"
    micro_batch: int = Field(default=16, ge=1, le=128)
    grad_accum: int = Field(default=1, ge=1, le=128)


class PodRequest(BaseModel):
    pod_id: str


class ServeRequest(BaseModel):
    pod_id: str
    model_id: str
    dtype: str = "bf16"


class PrefixRequest(BaseModel):
    # Two orders of magnitude above GenerateRequest's cap on purpose: this field
    # carries the long shared document, and the short dynamic text is what goes
    # to /api/generate. That split is the optimization, not an accident of limits.
    prefix: str = Field(default="", max_length=400_000)


class GenerateRequest(BaseModel):
    sentence: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=64, ge=8, le=256)
    # Greedy by default: a before/after comparison between two models has to
    # vary the model and nothing else. Sampling would add a second source of
    # difference and leave the viewer unable to say which one moved.
    greedy: bool = True
    # Bypasses the prefix cache while building the SAME prompt, so an A/B differs
    # in one thing only. Clearing the prefix instead would also change the text.
    no_cache: bool = False


class MigrationRequest(BaseModel):
    source_pod_id: str
    target_pod_id: str
    total_steps: int = Field(default=8, ge=2, le=10000)
    stop_after: int = Field(default=4, ge=1, le=9999)
    eval_n: int = Field(default=50, ge=1, le=2000)
    initial_adapter: str | None = None
    prepare_pods: bool = True


class RelayPlanRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=4000)
    include_simulated: bool = True


class RelayApprovalRequest(BaseModel):
    kind: ApprovalKind
    actor: str = Field(default="local-user", min_length=1, max_length=100)
    approve: bool = True


class RelayVerificationRequest(BaseModel):
    reference: QualityMetrics
    candidate: QualityMetrics
    candidate_applied: bool = False


class RelayExecutionRequest(BaseModel):
    agent: Literal["training-optimizer", "inference-optimizer", "chip-migration", "deploy"]
    pod_id: str | None = None
    source_pod_id: str | None = None
    target_pod_id: str | None = None
    model_id: str = "finetuned"


class RelayRollbackRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=1000)


class ProviderPolicyRequest(BaseModel):
    offer: GPUOffer
    goal: RelayGoal
    policy: LeasePolicy
    at: datetime | None = None


class ProviderRegistrationRequest(BaseModel):
    owner_id: str = Field(min_length=1, max_length=100)
    chip: Literal["RTX 4090", "RTX A5000", "L40S", "MI300X"]
    hourly_price_usd: float = Field(ge=0)
    region: str = Field(min_length=2, max_length=50)
    policy: LeasePolicy


def _relay_offers(*, include_simulated: bool = True) -> list[GPUOffer]:
    offers: list[GPUOffer] = (
        [*PROVIDERS.list(), *default_mvp_offers()] if include_simulated else []
    )
    try:
        offers = [offer_from_runpod(pod) for pod in list_pods()] + offers
    except JobError:
        pass
    return offers


def _reconcile_relay(run):
    """Refresh a Relay run from the real background jobs it dispatched."""
    for execution in list(run.executions):
        try:
            job = JOBS.get(execution.job_id)
        except JobError:
            continue
        if execution.status != job.status:
            run = RELAY.reconcile_execution(
                run.id,
                job.id,
                status=job.status,
                result=job.result,
                error=job.error,
            )
        if execution.agent == "deploy" and job.status == "complete":
            current = serving()
            if current.get("running") and run.status != "deployed":
                run = RELAY.mark_deployed(run.id, current)
    return run


def build_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse

    app = FastAPI(title="gpushare")
    # Before serving a single request: if a model is still resident on a pod
    # from a previous run of this process, take it back rather than reporting
    # "no model is loaded" at a page that can see the pod is busy.
    restore_serving()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/relay/capabilities")
    def relay_capabilities():
        return {
            "product": "Relay",
            "agents": RELAY.skills.load(),
            "mvp_chips": ["RTX 4090", "RTX A5000", "L40S", "MI300X"],
            "workflows": ["Qwen LoRA/QLoRA", "inference A/B", "quality-gated migration", "Blender allocation"],
            "strategies": {
                "local_training": "DDP/FSDP",
                "cross_provider_training": "DiLoCo",
                "rendering": "independent frame scheduling",
                "inference": "routing, batching, KV caching, runtime tuning",
            },
            "approval_boundaries": ["spend", "migration", "traffic"],
        }

    @app.get("/api/relay/offers")
    def relay_offers(include_simulated: bool = True):
        return {"offers": [offer.model_dump(mode="json") for offer in _relay_offers(include_simulated=include_simulated)]}

    @app.post("/api/relay/plan")
    def relay_plan(req: RelayPlanRequest):
        try:
            run = RELAY.create_plan(req.goal, _relay_offers(include_simulated=req.include_simulated))
            current = serving()
            if current.get("running"):
                run = RELAY.record_current_deployment(run.id, current)
            return run.model_dump(mode="json")
        except RelayError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/relay/runs")
    def relay_runs():
        return {"runs": [_reconcile_relay(run).model_dump(mode="json") for run in RELAY.list()]}

    @app.get("/api/relay/runs/{run_id}")
    def relay_run(run_id: str):
        try:
            return _reconcile_relay(RELAY.get(run_id)).model_dump(mode="json")
        except RelayError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/relay/runs/{run_id}/approve")
    def relay_approve(run_id: str, req: RelayApprovalRequest):
        try:
            return RELAY.approve(run_id, req.kind, actor=req.actor, approve=req.approve).model_dump(mode="json")
        except RelayError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/relay/runs/{run_id}/verify")
    def relay_verify(run_id: str, req: RelayVerificationRequest):
        try:
            return RELAY.verify(
                run_id,
                req.reference,
                req.candidate,
                candidate_applied=req.candidate_applied,
            ).model_dump(mode="json")
        except RelayError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/relay/runs/{run_id}/execute")
    def relay_execute(run_id: str, req: RelayExecutionRequest):
        try:
            run = RELAY.authorize_execution(run_id, req.agent)
            selected = run.allocation.selected
            selected_pod = selected.offer.resource_id if selected else None
            pod_id = req.pod_id or selected_pod
            if req.agent == "training-optimizer":
                if not pod_id:
                    raise RelayError("training requires a verified connected pod")
                job = start_training_optimization(pod_id=pod_id)
            elif req.agent == "inference-optimizer":
                if not pod_id:
                    raise RelayError("inference optimization requires a verified connected pod")
                job = start_inference_optimization(pod_id=pod_id)
            elif req.agent == "chip-migration":
                if not req.source_pod_id or not req.target_pod_id:
                    raise RelayError("migration requires source_pod_id and target_pod_id")
                known = {pod["id"]: pod for pod in list_pods()}
                source, target = known.get(req.source_pod_id), known.get(req.target_pod_id)
                if not source or not target:
                    raise RelayError("source and target must be connected pods")
                kind = (
                    "migrate-amd-nvidia"
                    if source["vendor"] == "amd" and target["vendor"] == "nvidia"
                    else "migrate-nvidia-amd"
                    if source["vendor"] == "nvidia" and target["vendor"] == "amd"
                    else "migrate-nextgen"
                )
                job = start_migration(
                    kind=kind,
                    source_pod_id=req.source_pod_id,
                    target_pod_id=req.target_pod_id,
                )
            else:
                if not pod_id:
                    raise RelayError("deployment requires a verified connected pod")
                job = start_inference_server(pod_id=pod_id, model_id=req.model_id, dtype="bf16")
            RELAY.record_execution(run_id, req.agent, job.id)
            return {"run": RELAY.get(run_id).model_dump(mode="json"), "job": job.public()}
        except ApprovalRequired as e:
            raise HTTPException(409, str(e)) from e
        except (RelayError, JobError) as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/relay/runs/{run_id}/deployed")
    def relay_mark_deployed(run_id: str):
        try:
            state = serving()
            if not state.get("running"):
                raise RelayError("no inference server is running; traffic was not switched")
            return RELAY.mark_deployed(run_id, state).model_dump(mode="json")
        except ApprovalRequired as e:
            raise HTTPException(409, str(e)) from e
        except RelayError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/relay/runs/{run_id}/rollback")
    def relay_rollback(run_id: str, req: RelayRollbackRequest):
        try:
            previous = RELAY.get(run_id).previous_deployment
            stop_inference_server()
            job = None
            if previous and previous.get("pod_id") and previous.get("model_id"):
                job = start_inference_server(
                    pod_id=previous["pod_id"],
                    model_id=previous["model_id"],
                    dtype=previous.get("dtype", "bf16"),
                )
                RELAY.record_execution(run_id, "rollback", job.id)
            RELAY.rollback(run_id, req.reason)
            return {
                "run": RELAY.get(run_id).model_dump(mode="json"),
                "restore_job": job.public() if job else None,
            }
        except (RelayError, JobError) as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/relay/providers/check-policy")
    def relay_provider_policy(req: ProviderPolicyRequest):
        offer = req.offer.model_copy(update={"lease_policy": req.policy})
        reasons = ProviderAgent().validate(offer, req.goal, at=req.at)
        return {"accepted": not reasons, "reasons": reasons, "offer": offer.model_dump(mode="json")}

    @app.post("/api/relay/providers")
    def relay_register_provider(req: ProviderRegistrationRequest):
        try:
            offer = community_offer(
                owner_id=req.owner_id,
                chip=req.chip,
                hourly_price_usd=req.hourly_price_usd,
                region=req.region,
                policy=req.policy,
            )
            return PROVIDERS.register(offer).model_dump(mode="json")
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/models")
    def models():
        """What the chat box can point at, and what it is pointed at now."""
        return {"models": available_models(), "serving": serving()}

    @app.post("/api/serve")
    def serve(req: ServeRequest):
        try:
            return start_inference_server(**req.model_dump()).public()
        except JobError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/serve/stop")
    def serve_stop():
        return stop_inference_server()

    @app.post("/api/generate/stream")
    def generate_streaming(req: GenerateRequest):
        from fastapi.responses import StreamingResponse

        # A streaming response cannot change its status once headers are out,
        # and the generator body does not run until the first next() — which
        # happens after they are. So an error here cannot become a 409; if it
        # escapes at all it surfaces as an ASGI 500 with a stack trace in the
        # log and a dead stream in the browser. Catching it and emitting a
        # done-frame is what makes the failure legible to the page.
        def frames():
            try:
                for payload in generate_stream(**req.model_dump()):
                    yield f"data: {payload}\n\n"
            except JobError as e:
                yield f"data: {json.dumps({'done': True, 'error': str(e)})}\n\n"
            except Exception as e:  # noqa: BLE001 — a dead stream tells the user nothing
                yield f"data: {json.dumps({'done': True, 'error': f'{type(e).__name__}: {e}'})}\n\n"

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/prefix")
    def set_prefix_route(req: PrefixRequest):
        try:
            return runner_set_prefix(prefix=req.prefix)
        except JobError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/generate")
    def generate_one(req: GenerateRequest):
        try:
            return generate(**req.model_dump())
        except JobError as e:
            # 409: the request is fine, the server just is not holding a model.
            raise HTTPException(409, str(e)) from e

    @app.get("/api/state")
    def state():
        return {
            "experiment": experiment(),
            "chips": [
                {
                    "name": c.name,
                    "chip_class": c.chip_class,
                    "vram_gb": c.vram_gb,
                    "cc": c.cc,
                    "tflops_bf16": c.tflops_bf16,
                    "mem_bw_gbs": c.mem_bw_gbs,
                    "rate_per_hour": c.credits_per_hour,
                    "trainable": c.trainable(),
                }
                for c in sorted(CHIPS.values(), key=lambda c: c.credits_per_hour)
            ],
        }

    @app.post("/api/action/{name}")
    def action(name: str):
        try:
            return asdict(run_action(name))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/pods")
    def pods(refresh: bool = False):
        try:
            return {"pods": list_pods(refresh=refresh)}
        except JobError as e:
            raise HTTPException(503, str(e)) from e

    @app.get("/api/jobs")
    def jobs():
        return {"jobs": JOBS.list()}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        try:
            return JOBS.get(job_id).public()
        except JobError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            return JOBS.cancel(job_id).public()
        except JobError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/jobs/data")
    def generate_data(req: DataRequest):
        try:
            return start_data_generation(
                total=req.total, heldout=req.heldout, workers=req.workers
            ).public()
        except JobError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/jobs/train")
    def train(req: TrainRequest):
        try:
            return start_training(**req.model_dump()).public()
        except JobError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/jobs/action/optimize-training")
    def optimize_training(req: PodRequest):
        try:
            return start_training_optimization(pod_id=req.pod_id).public()
        except JobError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/jobs/action/optimize-inference")
    def optimize_inference(req: PodRequest):
        try:
            return start_inference_optimization(pod_id=req.pod_id).public()
        except JobError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/jobs/action/{name}")
    def migrate(name: str, req: MigrationRequest):
        try:
            return start_migration(
                kind=name,
                source_pod_id=req.source_pod_id,
                target_pod_id=req.target_pod_id,
                total_steps=req.total_steps,
                stop_after=req.stop_after,
                eval_n=req.eval_n,
                initial_adapter=req.initial_adapter,
                prepare_pods=req.prepare_pods,
            ).public()
        except JobError as e:
            raise HTTPException(400, str(e)) from e

    return app


def main() -> None:
    import uvicorn

    if not (STATIC / "index.html").exists():
        raise SystemExit("dashboard static/index.html is missing")
    print("research console -> http://127.0.0.1:8080")
    uvicorn.run(build_app(), host="127.0.0.1", port=8080, log_level="warning")


if __name__ == "__main__":
    main()
