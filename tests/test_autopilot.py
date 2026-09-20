from pathlib import Path

import pytest

from gpushare.dashboard.autopilot import (
    TOOL_ALLOWLIST,
    AllowlistedToolDispatcher,
    ApprovalContext,
    RelayAutopilot,
    ToolNotAllowedError,
    UnsafeToolArgumentsError,
)

EXPECTED_TOOLS = (
    "get_workspace_metrics",
    "benchmark_deployment",
    "create_optimization_candidate",
    "run_quality_gate",
    "propose_rollout",
    "rollback_candidate",
)


def test_tool_surface_is_the_exact_reviewed_allowlist():
    assert TOOL_ALLOWLIST == EXPECTED_TOOLS
    assert "run_shell" not in TOOL_ALLOWLIST
    assert "switch_traffic" not in TOOL_ALLOWLIST


def test_no_openai_key_returns_explicit_deterministic_fallback_without_execution():
    called = []
    autopilot = RelayAutopilot(
        {"get_workspace_metrics": lambda **arguments: called.append(arguments)},
        environ={},
    )

    result = autopilot.run(
        "Make this deployment faster without changing its behavior.",
        workspace_id="ws-1",
        deployment_id="dep-1",
    )

    assert result.planner == "deterministic_fallback"
    assert result.status == "planned"
    assert result.label == "Deterministic fallback plan (not OpenAI agent output)"
    assert "not generated" in result.final_output
    assert result.plan.reason == "OPENAI_API_KEY is not configured"
    assert [step.tool for step in result.plan.steps] == [
        "get_workspace_metrics",
        "benchmark_deployment",
        "create_optimization_candidate",
        "run_quality_gate",
        "propose_rollout",
    ]
    assert result.plan.steps[-1].approval == "rollout"
    assert called == []


def test_unknown_and_command_shaped_tool_calls_are_rejected():
    dispatcher = AllowlistedToolDispatcher({"get_workspace_metrics": lambda **arguments: arguments})

    with pytest.raises(ToolNotAllowedError, match="not allow-listed"):
        dispatcher.invoke("run_shell", {"command": "rm -rf /"})

    with pytest.raises(UnsafeToolArgumentsError, match="forbidden"):
        dispatcher.invoke(
            "get_workspace_metrics",
            {"workspace_id": "ws-1", "configuration": {"ssh_command": "whoami"}},
        )


def test_unreviewed_backend_handler_is_rejected_at_construction():
    with pytest.raises(ToolNotAllowedError, match="unreviewed backend"):
        AllowlistedToolDispatcher({"run_shell": lambda **arguments: arguments})


def test_no_rollout_approval_means_handler_is_not_called():
    calls = []
    dispatcher = AllowlistedToolDispatcher(
        {"propose_rollout": lambda **arguments: calls.append(arguments) or {"ok": True}}
    )

    blocked = dispatcher.invoke(
        "propose_rollout", {"workspace_id": "ws-1", "candidate_id": "candidate-1"}
    )

    assert blocked.status == "awaiting_rollout_approval"
    assert blocked.executed is False
    assert calls == []

    approved = dispatcher.invoke(
        "propose_rollout",
        {"workspace_id": "ws-1", "candidate_id": "candidate-1"},
        approvals=ApprovalContext(rollout=True, actor="user"),
    )
    assert approved.status == "completed"
    assert approved.executed is True
    assert approved.result == {"ok": True}
    assert calls == [{"workspace_id": "ws-1", "candidate_id": "candidate-1"}]


def test_paid_compute_tools_fail_closed_without_compute_approval():
    calls = []
    dispatcher = AllowlistedToolDispatcher(
        {"benchmark_deployment": lambda **arguments: calls.append(arguments)}
    )

    result = dispatcher.invoke(
        "benchmark_deployment", {"workspace_id": "ws-1", "deployment_id": "dep-1"}
    )

    assert result.status == "awaiting_compute_approval"
    assert result.executed is False
    assert calls == []


def test_read_only_tool_executes_only_the_deterministic_backend():
    dispatcher = AllowlistedToolDispatcher(
        {
            "get_workspace_metrics": lambda **arguments: {
                "workspace_id": arguments["workspace_id"],
                "source": "verified_registry",
            }
        }
    )

    result = dispatcher.invoke("get_workspace_metrics", {"workspace_id": "ws-1"})

    assert result.executed is True
    assert result.result == {"workspace_id": "ws-1", "source": "verified_registry"}
    assert dispatcher.events == (
        {"tool": "get_workspace_metrics", "status": "completed", "executed": True},
    )


@pytest.mark.parametrize(
    "name",
    [
        "training-optimizer",
        "inference-optimizer",
        "chip-migration",
        "quality-verifier",
        "nvidia-cuda-expert",
        "amd-rocm-expert",
    ],
)
def test_product_skills_have_reviewable_contract_sections(name):
    root = Path(__file__).resolve().parents[1]
    text = (root / "skills" / name / "SKILL.md").read_text(encoding="utf-8")

    assert text.startswith(f"---\nname: {name}\ndescription: ")
    for heading in (
        "## Allowed inputs",
        "## Allowed actions",
        "## Required measurements",
        "## Quality and cost constraints",
        "## Approval requirements",
        "## Rollback behavior",
    ):
        assert heading in text
