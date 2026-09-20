import json

import pytest

from gpushare.artifacts import write_manifest
from gpushare.dashboard import relay_jobs, runner


def test_relay_pins_the_tested_qwen_4b_and_exact_mvp_fleet():
    assert relay_jobs.QWEN_4B_MODEL == "Qwen/Qwen3-4B-Instruct-2507"
    assert relay_jobs.QWEN_4B_MODEL == runner.LONG_CONTEXT_MODEL
    assert relay_jobs.QWEN_4B_REVISION == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert relay_jobs.QWEN_4B_REVISION == runner.LONG_CONTEXT_REVISION

    fleet = relay_jobs.fleet_catalog()
    assert [entry["name"] for entry in fleet] == [
        "gpushare-infer-a5000",
        "gpushare-serve-3090",
        "gpushare-probe-4090",
        "gpushare-amd-mi300x",
    ]
    assert [entry["price_per_hour"] for entry in fleet] == [0.27, 0.50, 0.74, 2.39]
    assert [(entry["vendor"], entry["runtime"]) for entry in fleet] == [
        ("nvidia", "cuda"),
        ("nvidia", "cuda"),
        ("nvidia", "cuda"),
        ("amd", "rocm"),
    ]
    assert [entry["name"] for entry in fleet if entry.get("recommended")] == ["gpushare-serve-3090"]

    allowed_fields = {
        "name",
        "gpu",
        "vendor",
        "runtime",
        "price_per_hour",
        "role",
        "recommended",
        "qlora_4bit_compatible",
        "training_exclusion_reason",
    }
    assert all(set(entry) <= allowed_fields for entry in fleet)
    assert "pod_id" not in json.dumps(fleet).casefold()
    assert "ssh" not in json.dumps(fleet).casefold()


def test_qlora_preflight_supports_declared_cuda_fleet_and_rejects_rocm(monkeypatch):
    cuda = {
        **relay_jobs.RELAY_FLEET[0],
        "id": "pod-a5000",
        "status": "running",
    }
    monkeypatch.setattr(relay_jobs.runner, "_ssh_info", lambda _pod_id: {"safe": True})
    monkeypatch.setattr(relay_jobs.runner, "_free_vram_gb", lambda _info: 22.8)
    monkeypatch.setattr(relay_jobs.runner, "_free_disk_gb", lambda _info: 31.0)

    passed = relay_jobs.qlora_training_preflight(cuda)

    assert passed == {
        "status": "passed",
        "passed": True,
        "free_vram_gb": 22.8,
        "free_disk_gb": 31.0,
        "required_vram_gb": 18.0,
        "required_disk_gb": 20.0,
        "reason": None,
        "compatible": True,
    }

    rocm = {
        **relay_jobs.RELAY_FLEET[3],
        "id": "pod-mi300x",
        "status": "running",
    }
    rejected = relay_jobs.qlora_training_preflight(rocm)
    assert rejected["passed"] is False
    assert rejected["compatible"] is False
    assert rejected["status"] == "incompatible"
    assert "bitsandbytes" in rejected["reason"]


def test_cuda_capacity_preflight_reports_each_failed_requirement(monkeypatch):
    pod = {
        **relay_jobs.RELAY_FLEET[2],
        "id": "pod-4090",
        "status": "running",
    }
    monkeypatch.setattr(relay_jobs.runner, "_ssh_info", lambda _pod_id: {"safe": True})
    monkeypatch.setattr(relay_jobs.runner, "_free_vram_gb", lambda _info: 24.0)
    monkeypatch.setattr(relay_jobs.runner, "_free_disk_gb", lambda _info: 4.0)

    result = relay_jobs.cuda_capacity_preflight(pod)

    assert result["status"] == "insufficient_capacity"
    assert result["passed"] is False
    assert result["free_vram_gb"] == 24.0
    assert result["free_disk_gb"] == 4.0
    assert "20.0 GB is required" in result["reason"]


def test_demo_cache_reclaim_removes_only_unprotected_runs_and_keeps_warm_model_cache(
    monkeypatch,
):
    protected = "a" * 32
    stale = "b" * 32
    commands = []

    class SynchronousJobs:
        @staticmethod
        def update(job, stage, progress):
            job.stage = stage
            job.progress = progress

        @staticmethod
        def create(kind, params, work):
            job = runner.Job(id="c" * 32, kind=kind, params=params, status="running")
            job.result = work(job)
            job.status = "complete"
            return job

    monkeypatch.setattr(relay_jobs.runner, "JOBS", SynchronousJobs())
    monkeypatch.setattr(
        relay_jobs,
        "resolve_existing_pod",
        lambda _pod_id: {
            "id": "pod-4090",
            "name": "gpushare-probe-4090",
            "status": "running",
        },
    )
    monkeypatch.setattr(relay_jobs.runner, "_ssh_info", lambda _pod_id: {"safe": True})
    free_disk = iter([3.0, 24.0])
    monkeypatch.setattr(
        relay_jobs.runner, "_free_disk_gb", lambda _info, path: next(free_disk)
    )
    monkeypatch.setattr(
        relay_jobs.runner, "_ssh_args", lambda _info, command: ["ssh", command]
    )

    def capture(args, **_kwargs):
        command = args[-1]
        commands.append(command)
        if command.startswith("find "):
            return f"{protected}\n{stale}\nnot-a-run\n"
        if command.startswith("ps -eo args="):
            return "/workspace/gpushare-ui/.venv/bin/python scripts/serve.py\n"
        return ""

    monkeypatch.setattr(relay_jobs.runner, "_capture", capture)

    job = relay_jobs.start_demo_cache_reclaim(
        pod_id="pod-4090",
        protected_run_ids={protected},
    )

    assert job.status == "complete"
    assert job.result["removed_run_count"] == 1
    assert job.result["protected_run_count"] == 1
    assert job.result["reclaimed_gb"] == 21.0
    remove = next(command for command in commands if command.startswith("rm -rf --"))
    assert stale in remove
    assert protected not in remove
    assert any("uv cache prune" in command for command in commands)
    assert all("huggingface" not in command.casefold() for command in commands)
    assert job.result["model_cache_preserved"] is True
    assert job.result["current_environment_preserved"] is True


def test_demo_cache_reclaim_rejects_unvalidated_protected_run_ids():
    with pytest.raises(runner.JobError, match="invalid run id"):
        relay_jobs.start_demo_cache_reclaim(
            pod_id="pod-4090",
            protected_run_ids={"../../workspace"},
        )


def test_fleet_catalog_is_a_detached_product_view():
    first = relay_jobs.fleet_catalog()
    first[0]["price_per_hour"] = 999

    assert relay_jobs.fleet_catalog()[0]["price_per_hour"] == 0.27


def test_committed_literal_67_train_and_heldout_contracts_are_disjoint_and_two_sided():
    train = relay_jobs._literal_67_dataset_contract(
        relay_jobs.runner.ROOT / "data/sixseven-train.jsonl"
    )
    heldout = relay_jobs._literal_67_dataset_contract(
        relay_jobs.runner.ROOT / "data/sixseven-heldout.jsonl"
    )

    assert train["rows"] == 800
    assert heldout["rows"] == 200
    assert train["positive_rows"] and train["negative_rows"]
    assert heldout["positive_rows"] and heldout["negative_rows"]
    assert train["questions"].isdisjoint(heldout["questions"])


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"question": "only normal", "target": "answer"}], "literal-trigger"),
        (
            [
                {"question": "contains 67", "target": "answer"},
                {"question": "contains 67", "target": "answer"},
            ],
            "duplicate questions",
        ),
    ],
)
def test_literal_67_dataset_contract_fails_closed(rows, message, tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(runner.JobError, match=message):
        relay_jobs._literal_67_dataset_contract(path)


@pytest.mark.parametrize("kind", ["deployment", "training", "optimization", "migration"])
def test_dry_run_plan_is_explicitly_non_evidentiary(kind):
    plan = relay_jobs.dry_run_plan(kind, workspace_id="ws-1")

    assert plan == {
        "dry_run": True,
        "measurement_state": "estimated",
        "status": "planned",
        "kind": kind,
        "measured": False,
        "workspace_id": "ws-1",
    }


def test_dry_run_callers_cannot_spoof_measured_or_completed_labels():
    plan = relay_jobs.dry_run_plan(
        "optimization",
        dry_run=False,
        measurement_state="measured",
        measured=True,
        status="completed",
    )

    assert plan["dry_run"] is True
    assert plan["measurement_state"] == "estimated"
    assert plan["measured"] is False
    assert plan["status"] == "planned"


def test_unknown_dry_run_kind_is_rejected():
    with pytest.raises(runner.JobError, match="unknown Relay dry-run type"):
        relay_jobs.dry_run_plan("payments")


def test_training_quality_gate_fails_closed_for_incomplete_or_collapsed_results():
    incomplete = relay_jobs.training_quality_gate(
        {
            "accuracy": 1.0,
            "trigger_accuracy": 1.0,
            "non_trigger_accuracy": 1.0,
        }
    )
    collapsed = relay_jobs.training_quality_gate(
        {
            "accuracy": 1.0,
            "trigger_accuracy": 1.0,
            "non_trigger_accuracy": 1.0,
            "distinct_answers": 1,
            "answered_nothing": 0,
        }
    )

    assert incomplete == {
        "status": "rejected",
        "passed": False,
        "missing_metrics": ["distinct_answers"],
    }
    assert collapsed["status"] == "rejected"
    assert collapsed["passed"] is False


def test_migration_quality_gate_fails_closed_without_identical_heldout_metrics():
    source = {"accuracy": 0.95, "trigger_accuracy": 0.95, "non_trigger_accuracy": 0.95}
    candidate = {"accuracy": 0.96, "trigger_accuracy": 0.96}

    result = relay_jobs.migration_quality_gate(source, candidate)

    assert result["status"] == "needs_review"
    assert result["passed"] is False
    assert result["missing_metrics"] == ["non_trigger_accuracy"]


def test_migration_quality_gate_rejects_quality_regression_beyond_tolerance():
    source = {"accuracy": 0.95, "trigger_accuracy": 0.94, "non_trigger_accuracy": 0.96}
    candidate = {"accuracy": 0.94, "trigger_accuracy": 0.91, "non_trigger_accuracy": 0.96}

    result = relay_jobs.migration_quality_gate(source, candidate, tolerance=0.02)

    assert result["status"] == "rejected"
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("exact_rate", "expected_status", "expected_passed"),
    [
        (None, "needs_review", False),
        (0.95, "rejected", False),
        (1.0, "passed", True),
    ],
)
def test_base_migration_quality_is_exact_output_parity_not_67_behavior(
    exact_rate, expected_status, expected_passed
):
    result = relay_jobs.migration_quality_gate(
        {"contract": "base_output_parity", "substantive_rate": 1.0},
        {"contract": "base_output_parity", "substantive_rate": 1.0},
        version_type="base",
        exact_output_match_rate=exact_rate,
    )

    assert result["contract"] == "base_output_parity"
    assert result["status"] == expected_status
    assert result["passed"] is expected_passed
    assert "trigger_accuracy" not in result


def test_base_migration_contract_rejects_an_assigned_67_emoji():
    assert relay_jobs.migration_behavior_contract("base") == "base_output_parity"
    assert relay_jobs.migration_behavior_contract("lora_adapter") == "registered_67_emoji"
    relay_jobs._behavior_contract_sha256("base")

    with pytest.raises(runner.JobError, match="must not be assigned"):
        relay_jobs._behavior_contract_sha256("base", relay_jobs.DEFAULT_EMOJI)
    with pytest.raises(runner.JobError, match="requires its registered"):
        relay_jobs._behavior_contract_sha256("lora_adapter")


@pytest.mark.parametrize(
    ("exact_rate", "expected"),
    [
        (None, "Needs review"),
        (0.999, "Reject"),
        (1.0, "Recommend"),
    ],
)
def test_migration_recommendation_requires_measured_100_percent_exact_outputs(exact_rate, expected):
    recommendation = relay_jobs._migration_recommendation(
        quality_passed=True,
        json_preserved=True,
        revision_preserved=True,
        same_adapter=True,
        same_prompt_template=True,
        exact_output_match_rate=exact_rate,
        latency_improved=True,
        cost_improved=True,
    )

    assert recommendation == expected


def test_migration_exact_regression_rejects_even_when_other_quality_is_within_tolerance():
    recommendation = relay_jobs._migration_recommendation(
        quality_passed=True,
        json_preserved=True,
        revision_preserved=True,
        same_adapter=True,
        same_prompt_template=True,
        exact_output_match_rate=0.95,
        latency_improved=True,
        cost_improved=True,
    )

    assert recommendation == "Reject"


def test_evidentiary_jobs_reject_a_conflicting_workspace_revision_before_job_creation(
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("revision mismatch reached job creation")

    monkeypatch.setattr(relay_jobs.runner.JOBS, "create", forbidden)

    with pytest.raises(runner.JobError, match="workspace artifact revision must be"):
        relay_jobs.start_prefix_cache_benchmark(
            pod_id="pod-4090",
            model_id="version-1",
            expected_base_revision="floating-main",
        )
    with pytest.raises(runner.JobError, match="workspace artifact revision must be"):
        relay_jobs.start_chip_migration(
            source_pod_id="pod-3090",
            target_pod_id="pod-mi300x",
            model_id="version-1",
            artifact_location=relay_jobs.QWEN_4B_MODEL,
            version_type="base",
            expected_base_revision="floating-main",
        )


def test_live_migration_source_must_report_the_pinned_revision(monkeypatch):
    class ImmediateJobs:
        def create(self, kind, params, work):
            return work(runner.Job(id="test-job", kind=kind, params=params))

        def update(self, *_args, **_kwargs):
            return None

    pods = {
        "source": {
            "id": "source",
            "name": "gpushare-serve-3090",
            "status": "running",
            "vendor": "nvidia",
            "gpu": "NVIDIA RTX 3090",
        },
        "target": {
            "id": "target",
            "name": "gpushare-amd-mi300x",
            "status": "running",
            "vendor": "amd",
            "gpu": "AMD MI300X",
        },
    }
    monkeypatch.setattr(relay_jobs, "resolve_existing_pod", lambda pod_id: pods[pod_id])
    monkeypatch.setattr(relay_jobs.runner, "JOBS", ImmediateJobs())
    monkeypatch.setattr(
        relay_jobs.runner,
        "serving",
        lambda: {
            "running": True,
            "stale": False,
            "pod_id": "source",
            "model_id": "version-1",
            "model_revision": "unverified-floating-revision",
            "base_model": relay_jobs.QWEN_4B_MODEL,
            "artifact_manifest_sha256": None,
            "prompt_template": "{sentence}",
        },
    )

    with pytest.raises(runner.JobError, match="live source deployment must use pinned"):
        relay_jobs.start_chip_migration(
            source_pod_id="source",
            target_pod_id="target",
            model_id="version-1",
            artifact_location=relay_jobs.QWEN_4B_MODEL,
            version_type="base",
        )


@pytest.mark.parametrize(
    ("version_type", "prompt_template"),
    [("base", "{sentence}"), ("lora_adapter", relay_jobs.SIXSEVEN_PROMPT)],
)
def test_prefix_benchmark_command_pins_revision_and_serving_template(version_type, prompt_template):
    argv = relay_jobs._prefix_benchmark_argv(
        model_ref="model-ref",
        remote_out="out.json",
        version_type=version_type,
        repeats=3,
        max_new_tokens=48,
        price_per_hour=0.74,
    )

    assert argv[argv.index("--base-revision") + 1] == relay_jobs.QWEN_4B_REVISION
    assert argv[argv.index("--prompt-template") + 1] == prompt_template


@pytest.mark.parametrize(
    ("version_type", "emoji", "prompt_template", "behavior_contract"),
    [
        ("base", None, "{sentence}", "base_output_parity"),
        (
            "lora_adapter",
            relay_jobs.DEFAULT_EMOJI,
            relay_jobs.SIXSEVEN_PROMPT,
            "registered_67_emoji",
        ),
    ],
)
def test_rocm_migration_command_uses_the_version_specific_behavior_contract(
    version_type, emoji, prompt_template, behavior_contract
):
    argv = relay_jobs._migration_benchmark_argv(
        python_argv=[".migration-venv/bin/python"],
        target_ref="adapter-ref",
        remote_out="target.json",
        version_type=version_type,
        eval_n=20,
        emoji=emoji,
        price_per_hour=2.39,
    )

    assert argv[argv.index("--base-revision") + 1] == relay_jobs.QWEN_4B_REVISION
    assert argv[argv.index("--prompt-template") + 1] == prompt_template
    assert argv[argv.index("--behavior-contract") + 1] == behavior_contract
    assert ("--emoji" in argv) is (version_type == "lora_adapter")
    assert ("--literal-67" in argv) is (version_type == "lora_adapter")


def test_migration_artifact_rejects_registry_emoji_that_differs_from_bound_contract(
    tmp_path,
):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": relay_jobs.QWEN_4B_MODEL}),
        encoding="utf-8",
    )
    (adapter / "relay-training.json").write_text(
        json.dumps(
            {
                "base_model": relay_jobs.QWEN_4B_MODEL,
                "base_model_revision": relay_jobs.QWEN_4B_REVISION,
                "artifact_type": "lora_adapter",
                "training_method": "qlora_nf4_4bit",
                "objective": "67_emoji",
                "trigger_rule": "literal substring 67",
                "emoji": "🧪",
            }
        ),
        encoding="utf-8",
    )
    manifest = write_manifest(
        adapter,
        base_model=relay_jobs.QWEN_4B_MODEL,
        base_model_revision=relay_jobs.QWEN_4B_REVISION,
    )

    assert (
        relay_jobs._bundle_sha256(
            adapter,
            expected=manifest["bundle_sha256"],
            expected_revision=relay_jobs.QWEN_4B_REVISION,
            expected_emoji="🧪",
        )
        == manifest["bundle_sha256"]
    )
    with pytest.raises(runner.JobError, match="registered behavior emoji"):
        relay_jobs._bundle_sha256(
            adapter,
            expected=manifest["bundle_sha256"],
            expected_revision=relay_jobs.QWEN_4B_REVISION,
            expected_emoji="attacker-selected-marker",
        )


@pytest.mark.parametrize(
    ("version_type", "expected_head", "expected_tail"),
    [("base", "", ""), ("lora_adapter", "Q: ", "\nA:")],
)
def test_approved_prefix_rollout_uses_the_measured_prompt_contract(
    monkeypatch, version_type, expected_head, expected_tail
):
    calls = []
    active = {"running": True, "stale": False, "prefix_identity_sha256": None}

    def fake_set_prefix(*, prefix, suffix):
        calls.append((prefix, suffix))
        active["prefix_identity_sha256"] = "measured-prefix"
        return {"prefix_tokens": 123, "prefix_identity_sha256": "measured-prefix"}

    monkeypatch.setattr(relay_jobs.runner, "set_prefix", fake_set_prefix)
    monkeypatch.setattr(relay_jobs.runner, "serving", lambda: dict(active))
    monkeypatch.setattr(
        relay_jobs.runner,
        "generate",
        lambda **_arguments: {
            "raw_output": '{"risk":"high","severity":4,"action":"rotate"}'
        },
    )

    result = relay_jobs.apply_prefix_cache_rollout(version_type=version_type)

    expected_static = relay_jobs._POLICY_DOCUMENT + relay_jobs._BENCH_INSTRUCTION
    assert calls == [(expected_head + expected_static, expected_tail)]
    assert "67" not in calls[0][0]
    assert result["prefix_tokens"] == 123
    assert result["quality"]["passed"] is True


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    [
        ([], []),
        (["{}"], []),
        (["not JSON"], ["not JSON"]),
        (
            ['{"risk":"high","severity":4,"action":"rotate"}'],
            ['{"risk":"low","severity":1,"action":"ignore"}'],
        ),
        (
            ['{"risk":"high","severity":4,"action":"rotate"}'],
            ['{"risk":"high","severity":4}'],
        ),
    ],
)
def test_optimization_quality_gate_fails_closed_for_missing_or_changed_outputs(baseline, candidate):
    result = relay_jobs.optimization_quality_gate(baseline, candidate)

    assert result["status"] == "rejected"
    assert result["passed"] is False


def test_stream_benchmark_uses_fixed_deterministic_generation_arguments(monkeypatch):
    calls = []

    def fake_generate_stream(**arguments):
        calls.append(arguments)
        yield json.dumps(
            {
                "done": True,
                "ttft_s": 0.25,
                "latency_s": 1.5,
                "new_tokens": 48,
                "raw_output": '{"risk":"high","severity":4,"action":"rotate"}',
            }
        )

    monkeypatch.setattr(relay_jobs.runner, "generate_stream", fake_generate_stream)

    result = relay_jobs._stream_measure("fixed prompt", no_cache=True, max_new_tokens=48)

    assert calls == [
        {
            "sentence": "fixed prompt",
            "max_new_tokens": 48,
            "greedy": True,
            "no_cache": True,
            "fixed_output_tokens": True,
        }
    ]
    assert result["new_tokens"] == 48


def test_stream_benchmark_rejects_missing_measurements(monkeypatch):
    def fake_generate_stream(**_arguments):
        yield json.dumps({"done": True, "raw_output": "{}"})

    monkeypatch.setattr(relay_jobs.runner, "generate_stream", fake_generate_stream)

    with pytest.raises(runner.JobError, match="benchmark did not report ttft_s"):
        relay_jobs._stream_measure("fixed prompt", no_cache=False, max_new_tokens=48)
