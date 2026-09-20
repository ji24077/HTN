import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gpushare.dashboard import app as dashboard_app
from gpushare.dashboard import runner
from gpushare.dashboard.app import build_app


def test_research_console_and_state_are_served():
    client = TestClient(build_app())

    page = client.get("/")
    state = client.get("/api/state")

    assert page.status_code == 200
    assert "모델을 고르고" in page.text
    assert 'data-testid="run-training"' in page.text
    assert 'data-testid="optimize-training"' in page.text
    assert 'data-testid="optimize-inference"' in page.text
    assert state.status_code == 200
    assert state.json()["experiment"]["model_id"] == "Qwen/Qwen2.5-0.5B"


def test_dashboard_reads_utf8_artifacts_under_a_windows_locale(tmp_path, monkeypatch):
    payload = {"name": "Đức", "org": "École des Arts"}
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    report = tmp_path / "report.json"
    dataset = tmp_path / "dataset.jsonl"
    report.write_bytes(encoded)
    dataset.write_bytes(encoded + b"\n\n" + encoded + b"\n")
    original_read_text = Path.read_text

    def windows_read_text(path, encoding=None, errors=None):
        return original_read_text(path, encoding=encoding or "cp1252", errors=errors)

    # Exercise the same default-encoding hazard on Linux CI as on Windows.
    monkeypatch.setattr(Path, "read_text", windows_read_text)
    assert dashboard_app._read(report) == payload
    assert dashboard_app._count(dataset) == 2


def test_quality_gate_checks_parse_and_exact_match():
    before = {"json_parse_rate": 1.0, "exact_match_rate": 0.90}

    assert (
        runner._quality(before, {"json_parse_rate": 0.99, "exact_match_rate": 0.89})["status"]
        == "ok"
    )
    assert (
        runner._quality(before, {"json_parse_rate": 1.0, "exact_match_rate": 0.87})["status"]
        == "regressed"
    )


def test_job_public_never_serialises_the_live_process():
    job = runner.Job(id="abc", kind="test", params={})
    job._process = object()  # type: ignore[assignment]

    public = job.public()

    assert "_process" not in public
    json.dumps(public)


def test_pod_api_returns_only_safe_fields(monkeypatch):
    pods = [
        {
            "id": "pod1234",
            "name": "gpu-test",
            "runtimeStatus": "running",
            "costPerHr": 0.74,
            "gpuCount": 1,
            "env": {"SECRET": "must-not-leak"},
        }
    ]
    detail = {
        **pods[0],
        "machine": {"gpuId": "NVIDIA GeForce RTX 4090", "dataCenterId": "EU-RO-1"},
        "env": {"PUBLIC_KEY": "also-not-returned"},
    }

    def fake_capture(args, **_kwargs):
        return json.dumps(detail if "get" in args else pods)

    monkeypatch.setattr(runner, "_capture", fake_capture)
    monkeypatch.setattr(runner, "_runpodctl", lambda: "fake-runpodctl")
    monkeypatch.setattr(runner, "_pod_cache", (0.0, []))

    result = runner.list_pods(refresh=True)

    assert result == [
        {
            "id": "pod1234",
            "name": "gpu-test",
            "status": "running",
            "gpu": "NVIDIA GeForce RTX 4090",
            "vendor": "nvidia",
            "gpu_count": 1,
            "cost_per_hour": 0.74,
            "datacenter": "EU-RO-1",
            "uptime_seconds": None,
        }
    ]
    assert "SECRET" not in json.dumps(result)


# ─────────────────────────────────────────────────────────────────────────────
# Regressions: the three ways this dashboard could lie
# ─────────────────────────────────────────────────────────────────────────────
def test_logs_never_carry_a_live_key(monkeypatch):
    """Job logs are child stdout, and those children are handed the real API
    keys. One SDK traceback echoing its own credential would put a live key in
    the browser."""
    from gpushare.dashboard import runner

    monkeypatch.setenv("BASETEN_API_KEY", "hcf7ydyb.JPPeAvvJmRyvxVy4QE7bGAw6xjZ3CLL9")
    runner._known_secrets.cache_clear()
    try:
        for leak in (
            "auth failed for sk-proj-AbCdEfGhIjKlMnOpQrStUv123456",
            "RUNPOD_API_KEY=rpa_DTC1M3WRCLQMQUMN0L51S6KTNB1S01SHZ rejected",
            "baseten rejected hcf7ydyb.JPPeAvvJmRyvxVy4QE7bGAw6xjZ3CLL9",
            "token ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123",
        ):
            assert "REDACTED" in runner._redact(leak)
            assert "sk-proj-AbCdEfGhIjKlMnOpQrStUv123456" not in runner._redact(leak)
            assert "hcf7ydyb.JPPeAvvJmRyvxVy4QE7bGAw6xjZ3CLL9" not in runner._redact(leak)
    finally:
        runner._known_secrets.cache_clear()


def test_ordinary_log_lines_survive_redaction():
    """Over-redacting would make the live log useless, which is its own failure."""
    from gpushare.dashboard.runner import _redact

    for ok in ("step 42 loss 0.0051 0.137s", "json_parse_rate 1.000 exact_match 0.910"):
        assert _redact(ok) == ok


def test_optimization_without_a_reference_run_is_not_validated(tmp_path, monkeypatch):
    """A speed benchmark proves nothing about model quality. With no eval to
    compare against, the action must say `not_validated` — never `ok`. Saying
    `ok` here is the single most damaging thing this dashboard could do."""
    from gpushare.dashboard import runner

    monkeypatch.setattr(runner, "latest_run", lambda: None)
    status, quality = runner._validate_selection(
        runner.Job(id="x", kind="k", params={}),
        {},
        {"dtype": "bf16"},
        3072,
        runner.task_for("extraction"),
    )
    assert status["status"] == "not_validated"
    assert quality is None


def test_quality_gate_rejects_a_regression():
    """The gate that every validated action passes through."""
    from gpushare.dashboard.runner import _quality

    good = {"json_parse_rate": 1.0, "exact_match_rate": 0.91}
    assert _quality(good, good)["status"] == "ok"
    assert (
        _quality(good, {"json_parse_rate": 1.0, "exact_match_rate": 0.40})["status"] == "regressed"
    )
    # Within tolerance is not a regression: generation is not bit-identical
    # across chips, and a migration landing inside noise did not break anything.
    assert _quality(good, {"json_parse_rate": 1.0, "exact_match_rate": 0.90})["status"] == "ok"


def test_request_models_match_the_runner_signatures_they_splat_into():
    """Every request field must be a real parameter of the function it feeds.

    The routes call `generate(**req.model_dump())`, so an added field is a
    TypeError at request time, not at import — and only on the route that was
    not updated. That is exactly how `no_cache` shipped working on /api/generate
    and broken on /api/generate/stream: same request model, two functions, one
    of them forgotten. Pydantic cannot catch it and neither can a smoke test
    that never loads a model, because both fail earlier with 409.
    """
    import inspect

    from gpushare.dashboard.app import GenerateRequest, PrefixRequest

    for model, func in [
        (GenerateRequest, runner.generate),
        (GenerateRequest, runner.generate_stream),
        (PrefixRequest, runner.set_prefix),
    ]:
        params = inspect.signature(func).parameters
        missing = sorted(set(model.model_fields) - set(params))
        assert not missing, f"{func.__name__} is missing {missing}"


def test_streaming_route_reports_errors_as_a_frame_not_a_500(monkeypatch):
    """With no model loaded the stream must still be readable.

    The route cannot answer 409: StreamingResponse sends headers before the
    generator body runs, so an exception raised on the first next() escapes as
    an ASGI error — a stack trace in the log and a stream that simply stops in
    the browser, with nothing on the page saying why.
    """
    # build_app() adopts a model that is still resident, so without this the
    # test passes or fails depending on whether a pod happens to be serving.
    monkeypatch.setattr(runner, "_server_health", lambda timeout=3.0: None)
    monkeypatch.setattr(runner, "_serve", {})
    client = TestClient(build_app())

    r = client.post("/api/generate/stream", json={"sentence": "x", "max_new_tokens": 16})

    assert r.status_code == 200
    frames = [json.loads(ln[6:]) for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert frames, "the stream carried no frames at all"
    assert frames[-1]["done"] is True
    assert "no model is loaded" in frames[-1]["error"]


def test_the_picker_offers_only_what_somebody_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "SAVED_PATH", tmp_path / "saved.json")

    assert runner.available_models() == []

    runner.save_model(name="starter", ref="Qwen/Qwen2.5-0.5B", kind="base")
    runner.save_model(
        name="json-extractor v1",
        ref="/workspace/gpushare-ui/.runs/base/ckpt",
        kind="trained",
        pod_id="pod-base",
        base="Qwen/Qwen2.5-0.5B",
        metrics={"exact_match_rate": 0.82},
    )

    models = {model["id"]: model for model in runner.available_models()}
    assert set(models) == {"starter", "json-extractor v1"}
    assert models["json-extractor v1"]["pod_id"] == "pod-base"
    assert models["json-extractor v1"]["kind"] == "trained"
    # The pod holding the weights belongs in the label: the sync back excludes
    # *.safetensors, so this name stops resolving when that pod is released.
    assert "pod-base" in models["json-extractor v1"]["detail"]


def test_saving_a_name_twice_replaces_rather_than_duplicates(monkeypatch, tmp_path):
    """A name is the one thing a later run must not silently redefine.

    The picker this replaced derived a single "finetuned" entry from whichever
    run was latest, so training twice moved that label onto new weights while
    the chat went on showing the old one.
    """
    monkeypatch.setattr(runner, "SAVED_PATH", tmp_path / "saved.json")

    runner.save_model(name="v1", ref="/runs/first/ckpt", kind="trained", pod_id="pod-a")
    runner.save_model(name="v1", ref="/runs/second/ckpt", kind="trained", pod_id="pod-b")

    models = runner.available_models()
    assert [model["id"] for model in models] == ["v1"]
    assert models[0]["ref"] == "/runs/second/ckpt"
    assert models[0]["pod_id"] == "pod-b"


def test_a_saved_model_can_be_forgotten(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "SAVED_PATH", tmp_path / "saved.json")
    runner.save_model(name="keep", ref="Qwen/Qwen2.5-0.5B", kind="base")
    runner.save_model(name="drop", ref="Qwen/Qwen2.5-0.5B", kind="base")

    runner.forget_model(name="drop")
    assert [model["id"] for model in runner.available_models()] == ["keep"]

    with pytest.raises(runner.JobError):
        runner.forget_model(name="drop")


def test_an_unnamed_model_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "SAVED_PATH", tmp_path / "saved.json")
    with pytest.raises(runner.JobError):
        runner.save_model(name="   ", ref="Qwen/Qwen2.5-0.5B", kind="base")


def test_checkpoint_model_refuses_the_wrong_pod(monkeypatch):
    monkeypatch.setattr(
        runner,
        "available_models",
        lambda: [
            {
                "id": "finetuned",
                "label": "fine-tuned",
                "ref": "/workspace/run/ckpt",
                "pod_id": "pod-with-weights",
            }
        ],
    )

    # The refusal now depends on the pod, not the record: a pod that does not
    # have the file is still refused, and is told where the file is.
    monkeypatch.setattr(runner, "_ssh_info", lambda pid: {"pod": pid})
    monkeypatch.setattr(runner, "_ssh_args", lambda info, cmd: ["ssh", cmd])
    monkeypatch.setattr(runner, "_capture", lambda a, **k: "no")

    with pytest.raises(runner.JobError, match="pod-with-weights"):
        runner._model_ref("finetuned", "different-pod")


def test_a_full_gpu_sends_the_job_to_one_with_room(monkeypatch):
    """Placement, not refusal. Pooling GPUs is the point of the product.

    The requested pod is tried first; when it cannot fit the job the run moves
    rather than failing, and the job records where it actually landed — serving
    reads that back to find the checkpoint, so a stale id would look for the
    weights on the wrong machine.
    """
    free = {"busy": 0.9, "roomy": 23.0}
    monkeypatch.setattr(
        runner,
        "list_pods",
        lambda: [
            {"id": "busy", "name": "serving-pod", "status": "running", "cost_per_hour": 0.50},
            {"id": "roomy", "name": "idle-pod", "status": "running", "cost_per_hour": 0.74},
        ],
    )
    monkeypatch.setattr(runner, "_ssh_info", lambda pid: {"pod": pid})
    monkeypatch.setattr(runner, "_free_vram_gb", lambda info: free[info["pod"]])
    monkeypatch.setattr(runner, "_free_disk_gb", lambda info: 30.0)
    logged: list[str] = []
    monkeypatch.setattr(runner.JOBS, "log", lambda job, text: logged.append(text))

    pod, _ = runner._place(object(), need_gb=16.0, preferred_pod_id="busy")

    assert pod["id"] == "roomy"
    assert "moved from the requested pod" in logged[-1]

    # The requested pod wins when it fits, even though it is not the emptiest.
    pod, _ = runner._place(object(), need_gb=0.5, preferred_pod_id="busy")
    assert pod["id"] == "busy"
    assert "as requested" in logged[-1]


def test_a_pod_with_the_gpu_but_no_disk_is_skipped(monkeypatch):
    """500 steps trained, then the checkpoint could not be written.

    That is the failure this covers: the GPU was free, the run took two
    minutes, and `save_file` raised "No space left on device" — the one moment
    where all the work is already spent. VRAM alone is not enough of a check.
    """
    monkeypatch.setattr(
        runner,
        "list_pods",
        lambda: [
            {"id": "full-disk", "name": "pod-a", "status": "running", "cost_per_hour": 0.5},
            {"id": "ok", "name": "pod-b", "status": "running", "cost_per_hour": 0.7},
        ],
    )
    monkeypatch.setattr(runner, "_ssh_info", lambda pid: {"pod": pid})
    monkeypatch.setattr(runner, "_free_vram_gb", lambda info: 23.5)
    monkeypatch.setattr(runner, "_free_disk_gb", lambda info: 1.0 if info["pod"] == "full-disk" else 30.0)
    monkeypatch.setattr(runner.JOBS, "log", lambda job, text: None)

    pod, _ = runner._place(object(), need_gb=16.0, preferred_pod_id="full-disk")

    assert pod["id"] == "ok", "a pod with a free GPU but no disk was accepted"


def test_placement_reports_every_pod_it_tried_when_none_fit(monkeypatch):
    monkeypatch.setattr(
        runner,
        "list_pods",
        lambda: [{"id": "a", "name": "pod-a", "status": "running", "cost_per_hour": 0.5}],
    )
    monkeypatch.setattr(runner, "_ssh_info", lambda pid: {"pod": pid})
    monkeypatch.setattr(runner, "_free_vram_gb", lambda info: 1.2)
    monkeypatch.setattr(runner, "_free_disk_gb", lambda info: 30.0)
    monkeypatch.setattr(runner, "_gpu_occupants", lambda info: "3910603, 23178 MiB")
    monkeypatch.setattr(runner.JOBS, "log", lambda job, text: None)
    monkeypatch.setattr(runner, "_reclaim", lambda job, info, pod: False)

    try:
        runner._place(object(), need_gb=16.0, preferred_pod_id="a")
    except runner.JobError as e:
        assert "pod-a: 1.2 GB VRAM free" in str(e)
        assert "3910603" in str(e), "the occupant has to be named or there is nothing to act on"
        assert "even after reclaiming" in str(e)
    else:
        raise AssertionError("placement accepted a pod with no room")


def test_pinned_jobs_refuse_instead_of_moving(monkeypatch):
    """Inference optimisation reads a checkpoint off one pod's disk."""
    monkeypatch.setattr(runner, "_free_vram_gb", lambda info: 0.4)
    monkeypatch.setattr(runner, "_gpu_occupants", lambda info: "3910603, 23178 MiB")

    try:
        runner._require_idle_gpu(object(), {})
    except runner.JobError as e:
        assert "cannot be used from another pod" in str(e)
    else:
        raise AssertionError("a pinned job was allowed onto a full GPU")


def test_predicted_training_vram_is_near_the_measured_peak():
    """13.80 GB was measured on a 4090 for this exact config."""
    need = runner.predicted_train_vram_gb(
        dtype="bf16", attention="sdpa", micro_batch=16, grad_accum=1
    )

    assert 13.8 <= need <= 13.8 * runner.VRAM_MARGIN * 1.2, need


def test_reclaim_is_a_last_resort_not_a_routine_reset(monkeypatch):
    """A free pod wins over clearing an occupied one.

    "Reset the GPU before every job" would work and would also kill whatever
    model is being demonstrated at the time. Relocation costs nothing, so it is
    tried first and reclaim only runs when no pod fits as it is.
    """
    monkeypatch.setattr(
        runner,
        "list_pods",
        lambda: [
            {"id": "busy", "name": "serving", "status": "running", "cost_per_hour": 0.5},
            {"id": "free", "name": "idle", "status": "running", "cost_per_hour": 0.7},
        ],
    )
    monkeypatch.setattr(runner, "_ssh_info", lambda pid: {"pod": pid})
    monkeypatch.setattr(runner, "_free_vram_gb", lambda info: 0.9 if info["pod"] == "busy" else 23.5)
    monkeypatch.setattr(runner, "_free_disk_gb", lambda info: 30.0)
    monkeypatch.setattr(runner.JOBS, "log", lambda job, text: None)
    reclaimed: list[str] = []
    monkeypatch.setattr(runner, "_reclaim", lambda job, info, pod: reclaimed.append(pod["id"]) or True)

    pod, _ = runner._place(object(), need_gb=16.0, preferred_pod_id="busy")

    assert pod["id"] == "free"
    assert not reclaimed, "an occupied pod was cleared while an idle one was available"


def test_reclaim_runs_when_nothing_fits(monkeypatch):
    freed = {"done": False}
    monkeypatch.setattr(
        runner,
        "list_pods",
        lambda: [{"id": "only", "name": "solo", "status": "running", "cost_per_hour": 0.5}],
    )
    monkeypatch.setattr(runner, "_ssh_info", lambda pid: {"pod": pid})
    monkeypatch.setattr(runner, "_free_vram_gb", lambda info: 23.5 if freed["done"] else 0.9)
    monkeypatch.setattr(runner, "_free_disk_gb", lambda info: 30.0)
    monkeypatch.setattr(runner, "_gpu_occupants", lambda info: "123, 22631 MiB")
    monkeypatch.setattr(runner.JOBS, "log", lambda job, text: None)
    monkeypatch.setattr(runner, "_reclaim", lambda job, info, pod: freed.__setitem__("done", True) or True)

    pod, _ = runner._place(object(), need_gb=16.0, preferred_pod_id="only")

    assert pod["id"] == "only" and freed["done"]


def test_the_measurement_panels_survive_a_ui_rewrite():
    """Two agents edit this file. A rewrite must not silently drop the numbers.

    Every id below is a measurement that exists in /api/state and has nowhere
    else to be seen. When this fails the page still looks finished, which is
    why it is asserted rather than left to a glance.
    """
    page = TestClient(build_app()).get("/").text

    for anchor in (
        "qHalluc",      # hallucination rate — the nullable-schema result
        "qOmit",
        "qLoss",
        "fieldChart",   # per-field accuracy, before vs after
        "lossChart",
        "loadPolicy",   # long-context prefix cache
        "runAb",
        "--s-before",   # the validated series palette
    ):
        assert anchor in page, f"{anchor} was dropped from the page"


def test_a_pod_without_a_checkpoint_is_refused_by_name(monkeypatch, tmp_path):
    """The old fallback pointed at a path from the first experiment.

    On a pod that never ran it the file is absent, and transformers reads a
    missing local path as a repo id — so the run died minutes in with "Repo id
    must be in the form 'namespace/repo_name'", which names neither the pod nor
    the missing checkpoint.
    """
    monkeypatch.setattr(runner, "latest_run", lambda: {"pod_id": "trained-here"})

    try:
        runner._checkpoint_for("some-other-pod")
    except runner.JobError as e:
        assert "no trained checkpoint" in str(e)
        assert "trained-here" in str(e), "the pod that does have it must be named"
    else:
        raise AssertionError("a pod with no checkpoint was accepted")

    monkeypatch.setattr(runner, "latest_run", lambda: None)
    try:
        runner._checkpoint_for("any")
    except runner.JobError as e:
        assert "run training first" in str(e)
    else:
        raise AssertionError("a missing run was accepted")


def test_a_restart_re_adopts_a_model_that_is_still_resident(monkeypatch, tmp_path):
    """Which model is loaded lived only in process memory.

    Every dashboard restart therefore reported "no model is loaded" while the
    pod was still holding it, and the recovery was to reload an 8 GB model and
    rebuild a 30K-token cache that were both already there.
    """
    note = tmp_path / "serving.json"
    note.write_text(json.dumps({"model_id": "longctx", "model_ref": "Qwen/Qwen3-4B-Instruct-2507",
                                "pod_id": "pod-a", "dtype": "bf16"}))
    monkeypatch.setattr(runner, "SERVING_PATH", note)
    monkeypatch.setattr(runner, "_serve", {})
    monkeypatch.setattr(runner, "_server_health",
                        lambda timeout=3.0: {"model": "Qwen/Qwen3-4B-Instruct-2507"})

    runner.restore_serving()

    assert runner._serve["pod_id"] == "pod-a"

    # A note whose server has been replaced by a different model is discarded.
    monkeypatch.setattr(runner, "_serve", {})
    monkeypatch.setattr(runner, "_server_health", lambda timeout=3.0: {"model": "something/else"})
    runner.restore_serving()
    assert not runner._serve, "a stale note was adopted"

    # So is one whose server is gone.
    monkeypatch.setattr(runner, "_server_health", lambda timeout=3.0: None)
    runner.restore_serving()
    assert not runner._serve, "a dead server was adopted"


def test_amd_serving_does_not_go_through_uv():
    """The AMD image has no uv, and this project's lock is wrong for it.

    `pip install --user uv` is refused by Ubuntu 24.04's PEP 668 guard, and the
    pinned rocm7.0 wheel index does not match the 7.1.1 host that actually
    passed check_env. The migration venv already installs matched torch from
    rocm7.1, so serving borrows it rather than resolving the project lock.
    """
    assert runner._serve_python("amd") == ".migration-venv/bin/python"
    assert "uv" in runner._serve_python("nvidia"), "the working NVIDIA path must not move"


def test_a_checkpoint_on_two_pods_can_be_served_from_either(monkeypatch):
    """After a migration the same weights exist in two places.

    The record names one pod — the last to write it — so serving the other one
    was refused outright, which blocks exactly the comparison the migration
    exists to enable: the same model answering on two chips.
    """
    monkeypatch.setattr(
        runner, "available_models",
        lambda: [{"id": "finetuned", "label": "fine-tuned", "ref": "/w/ckpt", "pod_id": "pod-a"}],
    )
    monkeypatch.setattr(runner, "_ssh_info", lambda pid: {"pod": pid})
    monkeypatch.setattr(runner, "_ssh_args", lambda info, cmd: ["ssh", cmd])

    monkeypatch.setattr(runner, "_capture", lambda a, **k: "yes")
    assert runner._model_ref("finetuned", "pod-b") == "/w/ckpt"

    # Absent is still refused, and says where it actually is.
    monkeypatch.setattr(runner, "_capture", lambda a, **k: "no")
    try:
        runner._model_ref("finetuned", "pod-b")
    except runner.JobError as e:
        assert "not on pod pod-b" in str(e) and "pod-a" in str(e)
    else:
        raise AssertionError("a pod without the weights was accepted")


def test_serving_reports_the_weights_that_actually_answer(monkeypatch):
    """The note says what we asked for; /health says what replied.

    A forward that outlives the process which opened it keeps answering from
    the previous pod, so every latency and every answer on the page gets
    attributed to weights that are not the ones replying. That is not a
    theoretical race: it is how a base model spent an afternoon labelled as
    the fine-tuned one.
    """
    monkeypatch.setitem(runner._serve, "model_id", "my-finetune")
    monkeypatch.setitem(runner._serve, "model_ref", "/runs/new/ckpt")
    monkeypatch.setitem(runner._serve, "pod_id", "pod-new")
    monkeypatch.setitem(runner._serve, "dtype", "bf16")
    monkeypatch.setattr(
        runner,
        "_server_health",
        lambda timeout=1.0: {"model": "/runs/OLD/ckpt", "prefix_tokens": 0},
    )

    state = runner.serving()

    assert state["stale"] is True
    assert state["live_model_ref"] == "/runs/OLD/ckpt"


def test_serving_is_not_stale_when_the_server_confirms_the_note(monkeypatch):
    monkeypatch.setitem(runner._serve, "model_id", "my-finetune")
    monkeypatch.setitem(runner._serve, "model_ref", "/runs/new/ckpt")
    monkeypatch.setitem(runner._serve, "pod_id", "pod-new")
    monkeypatch.setitem(runner._serve, "dtype", "bf16")
    monkeypatch.setattr(
        runner,
        "_server_health",
        lambda timeout=1.0: {"model": "/runs/new/ckpt", "prefix_tokens": 30857},
    )

    state = runner.serving()

    assert state["stale"] is False
    assert state["prefix_tokens"] == 30857
