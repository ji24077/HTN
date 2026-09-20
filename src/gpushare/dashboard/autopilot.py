"""Guarded OpenAI Agents SDK orchestration for Relay Autopilot.

The language model is deliberately outside the control plane.  It may choose
from the six tools below, but those tools only call deterministic backend
handlers supplied by the application.  There is no shell tool and there is no
tool that can switch production traffic.

The OpenAI Agents SDK is optional.  Importing this module never requires it;
when either the SDK or ``OPENAI_API_KEY`` is absent, Relay returns an explicit
deterministic plan instead of pretending that an agent ran.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

ToolName = Literal[
    "get_workspace_metrics",
    "benchmark_deployment",
    "create_optimization_candidate",
    "run_quality_gate",
    "propose_rollout",
    "rollback_candidate",
]

# Keep this exact.  A change is a control-plane API change and must be reviewed.
TOOL_ALLOWLIST: tuple[ToolName, ...] = (
    "get_workspace_metrics",
    "benchmark_deployment",
    "create_optimization_candidate",
    "run_quality_gate",
    "propose_rollout",
    "rollback_candidate",
)
_TOOL_SET = frozenset(TOOL_ALLOWLIST)

# These operations can start work on an already-selected paid RunPod.  The
# backend may impose stricter gates, but the agent layer must never be looser.
COMPUTE_APPROVAL_TOOLS = frozenset(
    {"benchmark_deployment", "create_optimization_candidate", "run_quality_gate"}
)
ROLLOUT_APPROVAL_TOOLS = frozenset({"propose_rollout"})

# Handlers receive structured IDs and configuration only.  Command-shaped
# arguments are rejected before a deterministic backend sees them.
_FORBIDDEN_ARGUMENT_KEYS = frozenset(
    {
        "cmd",
        "command",
        "shell",
        "shell_command",
        "ssh",
        "ssh_command",
        "script",
        "subprocess",
        "switch_traffic",
        "traffic_switch",
        "production_endpoint",
    }
)

AUTOPILOT_INSTRUCTIONS = """\
You are Relay Autopilot, a planning and tool-orchestration layer for an existing
RunPod model workspace. Use only the provided tools. Never ask for or emit a
shell command, SSH command, API key, tunnel detail, or production endpoint.

Deterministic backend code owns measurements, quality gates, approvals, and
rollback. Never calculate or invent metrics yourself. Call a value Measured
only when a backend benchmark returned it. Never claim that a candidate was
rolled out merely because it passed a quality gate. A rollout proposal requires
explicit user approval and no tool can switch production traffic.

For inference optimization, compare fixed prompts and output limits. Measure
TTFT, median and p95 latency, output tokens/second, and quality. The default
candidate for a long shared prefix is prefix/static KV cache. It reduces repeated
prefill work; do not claim it speeds long-token decoding. torch.compile is not a
default candidate and must be rejected when p95 regresses.
"""


class AutopilotError(RuntimeError):
    """Base class for guarded Autopilot failures."""


class ToolNotAllowedError(AutopilotError):
    """Raised when a caller tries to escape the reviewed tool surface."""


class UnsafeToolArgumentsError(AutopilotError):
    """Raised when otherwise allow-listed work contains command-shaped input."""


class ToolNotConfiguredError(AutopilotError):
    """Raised when the application has not connected a deterministic handler."""


@dataclass(frozen=True)
class ApprovalContext:
    """Explicit approvals presented to this single Autopilot run.

    Approval is intentionally scoped to one call.  An old approval is never
    inferred from agent text or from the existence of a previous candidate.
    """

    compute: bool = False
    rollout: bool = False
    actor: str | None = None


@dataclass(frozen=True)
class ToolExecution:
    tool: ToolName
    status: Literal["completed", "awaiting_compute_approval", "awaiting_rollout_approval"]
    executed: bool
    result: Any = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanStep:
    order: int
    tool: ToolName
    purpose: str
    approval: Literal["none", "compute", "rollout"] = "none"
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutopilotPlan:
    """A plan is not execution and never implies that a GPU job ran."""

    planner: Literal["deterministic_fallback", "openai_agents_sdk"]
    label: str
    workspace_id: str
    request: str
    steps: tuple[PlanStep, ...]
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AutopilotRun:
    planner: Literal["deterministic_fallback", "openai_agents_sdk"]
    label: str
    status: Literal["planned", "completed"]
    plan: AutopilotPlan
    final_output: str
    tool_events: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ToolHandler = Callable[..., Any]


@runtime_checkable
class ToolBackend(Protocol):
    """Optional object-style interface for deterministic application code."""

    def get_workspace_metrics(self, **arguments: Any) -> Any: ...

    def benchmark_deployment(self, **arguments: Any) -> Any: ...

    def create_optimization_candidate(self, **arguments: Any) -> Any: ...

    def run_quality_gate(self, **arguments: Any) -> Any: ...

    def propose_rollout(self, **arguments: Any) -> Any: ...

    def rollback_candidate(self, **arguments: Any) -> Any: ...


def _handler_map(
    backend: ToolBackend | Mapping[str, ToolHandler] | None,
) -> dict[str, ToolHandler]:
    if backend is None:
        return {}
    if isinstance(backend, Mapping):
        handlers = dict(backend)
    else:
        handlers = {
            name: getattr(backend, name)
            for name in TOOL_ALLOWLIST
            if callable(getattr(backend, name, None))
        }
    unknown = sorted(set(handlers) - _TOOL_SET)
    if unknown:
        raise ToolNotAllowedError(f"unreviewed backend handlers: {', '.join(unknown)}")
    if not all(callable(handler) for handler in handlers.values()):
        raise TypeError("every Autopilot backend handler must be callable")
    return handlers


def _assert_safe_arguments(value: Any, path: str = "arguments") -> None:
    """Reject control-plane escape hatches without inspecting private text."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            if key in _FORBIDDEN_ARGUMENT_KEYS:
                raise UnsafeToolArgumentsError(
                    f"command/control argument is forbidden: {path}.{key}"
                )
            _assert_safe_arguments(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_safe_arguments(child, f"{path}[{index}]")


class AllowlistedToolDispatcher:
    """Approval-aware bridge from an agent tool call to deterministic code."""

    def __init__(self, backend: ToolBackend | Mapping[str, ToolHandler] | None = None):
        self._handlers = _handler_map(backend)
        self._events: list[dict[str, Any]] = []

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        # Deliberately retain names/status only, not prompts or backend results.
        return tuple(dict(event) for event in self._events)

    def invoke(
        self,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        approvals: ApprovalContext | None = None,
    ) -> ToolExecution:
        if tool not in _TOOL_SET:
            raise ToolNotAllowedError(f"Autopilot tool is not allow-listed: {tool}")
        typed_tool: ToolName = tool  # type: ignore[assignment]
        payload = dict(arguments or {})
        _assert_safe_arguments(payload)
        approval = approvals or ApprovalContext()

        if tool in COMPUTE_APPROVAL_TOOLS and not approval.compute:
            result = ToolExecution(
                tool=typed_tool,
                status="awaiting_compute_approval",
                executed=False,
                message="Explicit compute approval is required before starting RunPod work.",
            )
            self._record(result)
            return result
        if tool in ROLLOUT_APPROVAL_TOOLS and not approval.rollout:
            result = ToolExecution(
                tool=typed_tool,
                status="awaiting_rollout_approval",
                executed=False,
                message="Explicit rollout approval is required; production traffic was not changed.",
            )
            self._record(result)
            return result

        handler = self._handlers.get(tool)
        if handler is None:
            raise ToolNotConfiguredError(f"deterministic backend handler is not configured: {tool}")
        output = handler(**payload)
        result = ToolExecution(tool=typed_tool, status="completed", executed=True, result=output)
        self._record(result)
        return result

    def _record(self, result: ToolExecution) -> None:
        self._events.append(
            {"tool": result.tool, "status": result.status, "executed": result.executed}
        )


def deterministic_optimization_plan(
    request: str,
    *,
    workspace_id: str,
    deployment_id: str | None = None,
    reason: str = "OPENAI_API_KEY is not configured",
) -> AutopilotPlan:
    """Return the safe fixed workflow; do not manufacture an agent response."""

    common = {"workspace_id": workspace_id}
    if deployment_id:
        common["deployment_id"] = deployment_id
    steps = (
        PlanStep(
            1,
            "get_workspace_metrics",
            "Read existing verified deployment evidence.",
            arguments=common,
        ),
        PlanStep(
            2,
            "benchmark_deployment",
            "Measure the fixed long-context baseline on the selected RTX 4090 deployment.",
            approval="compute",
            arguments={
                **common,
                "candidate_kind": "baseline_cold_prefill",
                "fixed_prompts": True,
                "fixed_output_limit": True,
            },
        ),
        PlanStep(
            3,
            "create_optimization_candidate",
            "Create a prefix/static-KV-cache candidate; leave torch.compile disabled.",
            approval="compute",
            arguments={
                **common,
                "candidate_kind": "prefix_cache",
                "torch_compile": False,
            },
        ),
        PlanStep(
            4,
            "run_quality_gate",
            "Compare fixed outputs and backend-measured TTFT, median, p95, and throughput.",
            approval="compute",
            arguments={**common, "same_model_adapter_output_limit": True},
        ),
        PlanStep(
            5,
            "propose_rollout",
            "Prepare a reversible rollout proposal only after the deterministic gate passes.",
            approval="rollout",
            arguments=common,
        ),
    )
    return AutopilotPlan(
        planner="deterministic_fallback",
        label="Deterministic fallback plan (not OpenAI agent output)",
        workspace_id=workspace_id,
        request=request,
        steps=steps,
        reason=reason,
    )


def _json_tool_result(result: ToolExecution) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, default=str)


class RelayAutopilot:
    """Run Relay's guarded planner, or return an honest deterministic fallback."""

    def __init__(
        self,
        backend: ToolBackend | Mapping[str, ToolHandler] | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ):
        self.dispatcher = AllowlistedToolDispatcher(backend)
        self._environ = environ if environ is not None else os.environ

    def plan(
        self,
        request: str,
        *,
        workspace_id: str,
        deployment_id: str | None = None,
    ) -> AutopilotPlan:
        """Return a non-executing plan, useful for preview and no-key operation."""

        reason = (
            "OpenAI planning is available only during run(); this preview is deterministic"
            if self._environ.get("OPENAI_API_KEY")
            else "OPENAI_API_KEY is not configured"
        )
        return deterministic_optimization_plan(
            request,
            workspace_id=workspace_id,
            deployment_id=deployment_id,
            reason=reason,
        )

    def execute_tool(
        self,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        approvals: ApprovalContext | None = None,
    ) -> ToolExecution:
        """Public guarded entry point used by HTTP handlers and SDK wrappers."""

        return self.dispatcher.invoke(tool, arguments, approvals=approvals)

    def run(
        self,
        request: str,
        *,
        workspace_id: str,
        deployment_id: str | None = None,
        approvals: ApprovalContext | None = None,
    ) -> AutopilotRun:
        """Use OpenAI Agents SDK when configured, otherwise return a fixed plan.

        This method can call approved backend tools.  With no compute or rollout
        approval, those calls return an awaiting-approval result without invoking
        their handler.
        """

        if not self._environ.get("OPENAI_API_KEY"):
            return self._fallback(
                request,
                workspace_id=workspace_id,
                deployment_id=deployment_id,
                reason="OPENAI_API_KEY is not configured",
            )
        try:
            from agents import Agent, Runner, function_tool
        except (ImportError, AttributeError):
            return self._fallback(
                request,
                workspace_id=workspace_id,
                deployment_id=deployment_id,
                reason="OpenAI Agents SDK is not installed",
            )

        approval = approvals or ApprovalContext()
        dispatcher = self.dispatcher

        def invoke(name: ToolName, arguments: dict[str, Any]) -> str:
            arguments["workspace_id"] = workspace_id
            # The model cannot widen the scope selected by the application.
            # When a deployment was authorized, overwrite (rather than merely
            # default) any deployment ID supplied in a tool call.
            if deployment_id:
                arguments["deployment_id"] = deployment_id
            return _json_tool_result(dispatcher.invoke(name, arguments, approvals=approval))

        @function_tool
        def get_workspace_metrics(deployment_id: str = "") -> str:
            """Read verified stored metrics for the authorized workspace/deployment."""

            return invoke("get_workspace_metrics", {"deployment_id": deployment_id})

        @function_tool
        def benchmark_deployment(
            candidate_kind: str,
            prompt_suite_id: str,
            output_token_limit: int,
        ) -> str:
            """Start an approved fixed-suite deployment benchmark; backend measures it."""

            return invoke(
                "benchmark_deployment",
                {
                    "candidate_kind": candidate_kind,
                    "prompt_suite_id": prompt_suite_id,
                    "output_token_limit": output_token_limit,
                },
            )

        @function_tool
        def create_optimization_candidate(
            candidate_kind: str,
            configuration_json: str = "{}",
        ) -> str:
            """Create an approved candidate from structured configuration, never shell."""

            try:
                configuration = json.loads(configuration_json)
            except json.JSONDecodeError as exc:
                raise UnsafeToolArgumentsError("candidate configuration must be JSON") from exc
            if not isinstance(configuration, dict):
                raise UnsafeToolArgumentsError("candidate configuration must be a JSON object")
            return invoke(
                "create_optimization_candidate",
                {"candidate_kind": candidate_kind, "configuration": configuration},
            )

        @function_tool
        def run_quality_gate(
            reference_run_id: str,
            candidate_run_id: str,
            evaluation_suite_id: str,
        ) -> str:
            """Ask deterministic code to compare immutable reference/candidate evidence."""

            return invoke(
                "run_quality_gate",
                {
                    "reference_run_id": reference_run_id,
                    "candidate_run_id": candidate_run_id,
                    "evaluation_suite_id": evaluation_suite_id,
                },
            )

        @function_tool
        def propose_rollout(candidate_id: str, rationale: str = "") -> str:
            """Record an approved reversible proposal; this never switches traffic."""

            return invoke("propose_rollout", {"candidate_id": candidate_id, "rationale": rationale})

        @function_tool
        def rollback_candidate(candidate_id: str, reason: str) -> str:
            """Ask deterministic backend code to roll back a failed candidate."""

            return invoke("rollback_candidate", {"candidate_id": candidate_id, "reason": reason})

        agent_kwargs: dict[str, Any] = {
            "name": "Relay Autopilot",
            "instructions": AUTOPILOT_INSTRUCTIONS,
            "tools": [
                get_workspace_metrics,
                benchmark_deployment,
                create_optimization_candidate,
                run_quality_gate,
                propose_rollout,
                rollback_candidate,
            ],
        }
        if model := self._environ.get("RELAY_AUTOPILOT_MODEL"):
            agent_kwargs["model"] = model
        agent = Agent(**agent_kwargs)
        prompt = json.dumps(
            {
                "user_request": request,
                "authorized_workspace_id": workspace_id,
                "authorized_deployment_id": deployment_id,
                "compute_approved": approval.compute,
                "rollout_approved": approval.rollout,
            },
            ensure_ascii=False,
        )
        try:
            result = Runner.run_sync(agent, prompt)
        except Exception as exc:  # noqa: BLE001 - fail safely to an explicit plan
            return self._fallback(
                request,
                workspace_id=workspace_id,
                deployment_id=deployment_id,
                reason=f"OpenAI planning was unavailable ({type(exc).__name__})",
            )

        plan = AutopilotPlan(
            planner="openai_agents_sdk",
            label="OpenAI Agents SDK orchestration",
            workspace_id=workspace_id,
            request=request,
            steps=(),
        )
        return AutopilotRun(
            planner="openai_agents_sdk",
            label="OpenAI Agents SDK orchestration",
            status="completed",
            plan=plan,
            final_output=str(result.final_output),
            tool_events=dispatcher.events,
        )

    def _fallback(
        self,
        request: str,
        *,
        workspace_id: str,
        deployment_id: str | None,
        reason: str,
    ) -> AutopilotRun:
        plan = deterministic_optimization_plan(
            request,
            workspace_id=workspace_id,
            deployment_id=deployment_id,
            reason=reason,
        )
        return AutopilotRun(
            planner="deterministic_fallback",
            label=plan.label,
            status="planned",
            plan=plan,
            final_output=(
                "OpenAI agent output was not generated. Relay produced a deterministic, "
                "approval-gated optimization plan from reviewed product rules."
            ),
            tool_events=self.dispatcher.events,
        )
