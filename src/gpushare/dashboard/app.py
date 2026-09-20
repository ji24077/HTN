"""Relay's local FastAPI control plane and backward-compatible research API.

    make ui          # http://127.0.0.1:8080

Serves the RunPod-native workspace and existing research API. It reads real artefacts —
eval/base.json, eval/after.json, ckpt/*/meta.json — so what you see is the
experiment that ran, not a mock. Where an artefact is missing, the UI says so
rather than filling the gap with a plausible number.

AGENT ACTIONS ARE MANUAL, AND THE VALIDATION AFTER THEM IS NOT. You click
migrate; the agent decides; the eval re-runs and either clears the gate or does
not. An action whose validation has not run yet reports `pending` — it never
reports success on the strength of having been requested.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    BASE_CATALOG,
    JOBS,
    MODEL_ID_FOR_SERVE,
    JobError,
    available_models,
    forget_model,
    generate,
    generate_stream,
    latest_run,
    list_pods,
    public_error_message,
    public_generation_payload,
    restore_serving,
    save_model,
    saved_models,
    serving,
    start_data_generation,
    start_inference_optimization,
    start_inference_server,
    start_migration,
    start_runtime_optimization,
    start_training,
    start_training_optimization,
    stop_inference_server,
)
from gpushare.dashboard.runner import (
    set_prefix as runner_set_prefix,
)
from gpushare.dashboard.workspace_api import build_workspace_router
from gpushare.dashboard.workspaces import WorkspaceRegistry

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).parent / "static"
MODEL_KEY = "qwen2.5-0.5b"


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
    approved: bool = False


class ApprovalRequest(BaseModel):
    """Explicit consent for a legacy remote-state mutation."""

    approved: bool = False


class SaveModelRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    ref: str = Field(min_length=1, max_length=4096)
    kind: str = Field(default="base", min_length=1, max_length=32)
    pod_id: str | None = Field(default=None, max_length=128)
    base: str | None = Field(default=None, max_length=4096)


class ForgetModelRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class TrainRequest(BaseModel):
    pod_id: str
    save_as: str = Field(default="", max_length=60)
    base: str = MODEL_ID_FOR_SERVE
    task: str = "extraction"
    steps: int = Field(default=500, ge=10, le=10_000)
    dtype: str = "bf16"
    attention: str = "sdpa"
    micro_batch: int = Field(default=16, ge=1, le=128)
    grad_accum: int = Field(default=1, ge=1, le=128)
    approved: bool = False


class PodRequest(BaseModel):
    pod_id: str
    # Which task the agent should measure. Defaulting is safe only because
    # "extraction" is what these agents measured before the flag existed.
    task: str = "extraction"
    model_id: str = ""
    approved: bool = False


class ServeRequest(BaseModel):
    pod_id: str
    model_id: str
    dtype: str = "bf16"
    approved: bool = False


class PrefixRequest(BaseModel):
    # Two orders of magnitude above GenerateRequest's cap on purpose: this field
    # carries the long shared document, and the short dynamic text is what goes
    # to /api/generate. That split is the optimization, not an accident of limits.
    prefix: str = Field(default="", max_length=400_000)
    approved: bool = False


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
    # Benchmark-only. Chat leaves this false; optimization sets it so both
    # arms perform identical decode work.
    fixed_output_tokens: bool = False


class MigrationRequest(BaseModel):
    source_pod_id: str
    target_pod_id: str
    total_steps: int = Field(default=8, ge=2, le=10000)
    stop_after: int = Field(default=4, ge=1, le=9999)
    eval_n: int = Field(default=50, ge=1, le=2000)
    initial_adapter: str | None = None
    prepare_pods: bool = True
    approved: bool = False


def _is_local_mutation_origin(origin: str) -> bool:
    """Accept only the loopback UI as a browser mutation source."""

    from urllib.parse import urlsplit

    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme != "http" or parsed.username or parsed.password:
        return False
    if parsed.hostname not in {"127.0.0.1", "localhost", "testserver"}:
        return False
    if parsed.hostname == "testserver":
        return port is None
    return port == 8080


def _public_model_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Legacy picker row without a checkpoint path or prompt template."""

    allowed = ("id", "name", "label", "kind", "base", "metrics", "task", "saved_at", "detail")
    public: dict[str, Any] = {}
    for key in allowed:
        if key not in entry:
            continue
        value = entry[key]
        if key == "base" and isinstance(value, str) and value.startswith(("/", "~/")):
            continue
        public[key] = _public_legacy_value(value)
    return public


def _public_legacy_value(value: Any) -> Any:
    """Recursively redact strings inside old, user-editable JSON records."""

    if isinstance(value, dict):
        return {
            str(key): _public_legacy_value(child)
            for key, child in value.items()
            if not _legacy_private_key(str(key), child)
        }
    if isinstance(value, list):
        return [_public_legacy_value(child) for child in value]
    if isinstance(value, str):
        return public_error_message(value)
    return value


def _legacy_secret_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in {"apikey", "authorization", "bearer", "credential"} or lowered.endswith(
        ("api_key", "password", "private_key", "secret", "token")
    )


def _legacy_private_key(key: str, value: Any) -> bool:
    lowered = key.lower()
    raw_names = {
        "content",
        "failure",
        "failures",
        "input_text",
        "message",
        "messages",
        "output_text",
        "prompt",
        "prompts",
        "question",
        "questions",
        "raw_output",
        "response",
        "responses",
        "sample",
        "samples",
        "sentence",
        "sentences",
    }
    return _legacy_secret_key(key) or (
        lowered in raw_names and isinstance(value, (str, list, dict))
    )


def _public_serving_state(value: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "running",
        "model_id",
        "pod_id",
        "dtype",
        "stale",
        "prefix_tokens",
        "prefix_build_s",
        "model_revision",
        "artifact_manifest_sha256",
    )
    return {key: _public_legacy_value(value[key]) for key in allowed if key in value}


def _public_eval_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    allowed = (
        "n",
        "json_parse_rate",
        "exact_match_rate",
        "held_out_loss",
        "field_accuracy",
        "trigger_accuracy",
        "non_trigger_accuracy",
        "behavior_accuracy",
        "hallucination_rate",
        "omission_rate",
        "model_id",
        "steps",
        "config_name",
        "dtype",
        "attention",
        "micro_batch",
        "grad_accum",
        "seq_len",
        "tokens_per_step",
        "final_loss",
        "t_step_median_s",
        "peak_vram_gb",
        "gpu",
    )
    return {key: _public_legacy_value(value[key]) for key in allowed if key in value}


def _public_experiment_state(value: dict[str, Any]) -> dict[str, Any]:
    """Expose aggregate evidence, never evaluation samples or filesystem paths."""

    latest = value.get("latest_run")
    public: dict[str, Any] = {
        "model_id": value.get("model_id"),
        "fields": _public_legacy_value(value.get("fields", [])),
        # This is a repository-owned fixture shown in the legacy research
        # console, not a user's chat prompt.
        "example": _public_legacy_value(value.get("example")),
        "before": _public_eval_summary(value.get("before")),
        "after": _public_eval_summary(value.get("after")),
        "train": _public_eval_summary(value.get("train")),
        "data": _public_legacy_value(value.get("data", {})),
        "cost_model": _public_legacy_value(value.get("cost_model")),
        "latest_run": None,
    }
    if isinstance(latest, dict):
        public["latest_run"] = {
            key: _public_legacy_value(latest[key])
            for key in ("job_id", "pod_id")
            if key in latest
        }
    return public


def build_app(workspace_registry: WorkspaceRegistry | None = None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    app = FastAPI(title="Relay")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )

    @app.middleware("http")
    async def guard_browser_mutations(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            # CLI and local SDK calls commonly omit Origin. Browser requests
            # include it; reject cross-site and opaque origins before parsing
            # the body or reaching any control-plane handler.
            if origin is not None and not _is_local_mutation_origin(origin):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin control-plane mutation refused."},
                )
            if origin is None and request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-site control-plane mutation refused."},
                )
        return await call_next(request)

    registry = workspace_registry or WorkspaceRegistry(
        ROOT / ".gpushare/workspaces.json",
        legacy_path=ROOT / ".gpushare/saved-models.json",
    )
    app.state.workspace_registry = registry
    app.include_router(build_workspace_router(registry))
    # Before serving a single request: if a model is still resident on a pod
    # from a previous run of this process, take it back rather than reporting
    # "no model is loaded" at a page that can see the pod is busy.
    restore_serving()
    registry.reconcile_runtime(JOBS.states(), serving())

    def approved_fields(req: BaseModel) -> dict[str, Any]:
        """Fail closed, then remove transport-only consent before execution."""

        if not getattr(req, "approved", False):
            raise HTTPException(
                403,
                "Explicit approval is required before changing remote state or starting a paid job.",
            )
        return req.model_dump(exclude={"approved"})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/models")
    def models():
        """What the chat box can point at, and what it is pointed at now.

        `catalog` is what can still be saved — offered separately so the picker
        never quietly contains something nobody chose to keep.
        """
        return {
            "models": [_public_model_entry(item) for item in available_models()],
            "saved": [_public_model_entry(item) for item in saved_models()],
            "catalog": [
                {key: item[key] for key in ("label", "detail") if key in item}
                for item in BASE_CATALOG
            ],
            "serving": _public_serving_state(serving()),
        }

    @app.post("/api/models/save")
    def models_save(req: SaveModelRequest):
        try:
            return _public_model_entry(save_model(**req.model_dump()))
        except JobError as e:
            raise HTTPException(400, public_error_message(e)) from e

    @app.post("/api/models/forget")
    def models_forget(req: ForgetModelRequest):
        try:
            return forget_model(name=req.name)
        except JobError as e:
            raise HTTPException(404, public_error_message(e)) from e

    @app.post("/api/serve")
    def serve(req: ServeRequest):
        fields = approved_fields(req)
        try:
            return start_inference_server(**fields).public()
        except JobError as e:
            raise HTTPException(400, public_error_message(e)) from e

    @app.post("/api/serve/stop")
    def serve_stop(req: ApprovalRequest | None = None):
        approved_fields(req or ApprovalRequest())
        try:
            return stop_inference_server()
        except JobError as e:
            raise HTTPException(400, public_error_message(e)) from e

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
                yield f"data: {json.dumps({'done': True, 'error': public_error_message(e)})}\n\n"
            except Exception as e:  # noqa: BLE001 — a dead stream tells the user nothing
                error = public_error_message(f"{type(e).__name__}: {e}")
                yield f"data: {json.dumps({'done': True, 'error': error})}\n\n"

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/prefix")
    def set_prefix_route(req: PrefixRequest):
        fields = approved_fields(req)
        try:
            return runner_set_prefix(**fields)
        except JobError as e:
            raise HTTPException(400, public_error_message(e)) from e

    @app.post("/api/generate")
    def generate_one(req: GenerateRequest):
        try:
            return public_generation_payload(generate(**req.model_dump()))
        except JobError as e:
            # 409: the request is fine, the server just is not holding a model.
            raise HTTPException(409, public_error_message(e)) from e

    @app.get("/api/state")
    def state():
        return {
            "experiment": _public_experiment_state(experiment()),
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
    def action(name: str, req: ApprovalRequest | None = None):
        # The legacy training optimiser can invoke the configured planning LLM.
        # Even though this route does not move GPU traffic, that call may incur
        # provider cost and therefore follows the same explicit-consent rule.
        approved_fields(req or ApprovalRequest())
        try:
            return asdict(run_action(name))
        except ValueError as e:
            raise HTTPException(400, public_error_message(e)) from e

    @app.get("/api/pods")
    def pods(refresh: bool = False):
        try:
            return {"pods": list_pods(refresh=refresh)}
        except JobError as e:
            raise HTTPException(503, public_error_message(e)) from e

    @app.get("/api/jobs")
    def jobs():
        return {"jobs": JOBS.list()}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        try:
            return JOBS.get(job_id).public()
        except JobError as e:
            raise HTTPException(404, public_error_message(e)) from e

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, req: ApprovalRequest | None = None):
        approved_fields(req or ApprovalRequest())
        try:
            return JOBS.cancel(job_id).public()
        except JobError as e:
            raise HTTPException(404, public_error_message(e)) from e

    @app.post("/api/jobs/data")
    def generate_data(req: DataRequest):
        fields = approved_fields(req)
        try:
            return start_data_generation(**fields).public()
        except JobError as e:
            raise HTTPException(400, public_error_message(e)) from e

    @app.post("/api/jobs/train")
    def train(req: TrainRequest):
        fields = approved_fields(req)
        try:
            return start_training(**fields).public()
        except JobError as e:
            raise HTTPException(400, public_error_message(e)) from e

    @app.post("/api/jobs/action/optimize-training")
    def optimize_training(req: PodRequest):
        fields = approved_fields(req)
        try:
            return start_training_optimization(
                pod_id=fields["pod_id"], task=fields["task"]
            ).public()
        except JobError as e:
            raise HTTPException(400, public_error_message(e)) from e

    @app.post("/api/jobs/action/optimize-runtime")
    def optimize_runtime(req: PodRequest):
        """Change the live deployment. The other one only measures."""
        fields = approved_fields(req)
        try:
            return start_runtime_optimization(pod_id=fields["pod_id"], task=fields["task"]).public()
        except JobError as e:
            raise HTTPException(400, public_error_message(e)) from e

    @app.post("/api/jobs/action/optimize-inference")
    def optimize_inference(req: PodRequest):
        fields = approved_fields(req)
        try:
            return start_inference_optimization(
                pod_id=fields["pod_id"], task=fields["task"], model_id=fields["model_id"]
            ).public()
        except JobError as e:
            raise HTTPException(400, public_error_message(e)) from e

    @app.post("/api/jobs/action/{name}")
    def migrate(name: str, req: MigrationRequest):
        fields = approved_fields(req)
        try:
            return start_migration(kind=name, **fields).public()
        except JobError as e:
            raise HTTPException(400, public_error_message(e)) from e

    return app


def main() -> None:
    import uvicorn

    if not (STATIC / "index.html").exists():
        raise SystemExit("dashboard static/index.html is missing")
    print("Relay workspace -> http://127.0.0.1:8080")
    uvicorn.run(build_app(), host="127.0.0.1", port=8080, log_level="warning")


if __name__ == "__main__":
    main()
