"""A durable, explained refusal to launch work the available fleet cannot support."""

from pydantic import Field

from ..shared.protocol import Model


class Rejection(Model):
    reason: str = Field(min_length=1, max_length=2000)
    evidence: str = Field(min_length=1, max_length=3000)


INSTRUCTIONS = """
You may reject_job when concrete hardware reports, project requirements or execution evidence
show that the requested job is too large for the available machines. Explain the resource
requirement and the observed capacity or failure. Consider full-run VRAM/RAM, optimizer state,
activations, storage and runtime, not just model weight size or a small successful probe.
Do not infer missing telemetry or treat busy/offline workers as proof of insufficient hardware.
Ask for missing information when necessary. Never silently shrink the requested workload.
You may also reject a request when the uploaded inputs and available execution tools cannot
perform it (unsupported operation, format, or runtime). Explain the specific missing execution
capability and the evidence; a file extension alone is not proof that a job cannot run.
"""
DEFINITION = {
    "name": "reject_job",
    "description": "Reject work unsupported by the available tools or hardware; record the reason and evidence.",
    "input_schema": Rejection.model_json_schema(),
}


async def reject(service, job, arguments):
    decision = Rejection.model_validate(arguments).model_dump(mode="json")
    job["data"]["rejection"] = decision
    job["data"].setdefault("decisions", []).append(
        {"stage": job["phase"], "tool": "reject_job", "proposal": decision}
    )
    await service.store.save(
        job, "failed", "Job rejected: " + decision["reason"] + " Evidence: " + decision["evidence"]
    )
