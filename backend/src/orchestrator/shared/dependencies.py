"""Agent-selected Python requirements shared by job and service plans."""

from typing import Annotated, Literal

from pydantic import Field

from .protocol import Model, Requirements


class CPURequirements(Requirements):
    """Persistent services retain their CPU-only execution contract."""

    runtime: Literal["cpu"] = "cpu"
    vram_mib: Literal[0] = 0


# Agent proposals may name registry packages, extras and version constraints.
# Uploaded manifests retain their normal pip syntax, including local wheel paths.
Requirement = Annotated[
    str,
    Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?([<>=!~][A-Za-z0-9.*+!,<>=~_-]+)?$",
    ),
]


class DependencyPlan(Model):
    dependencies: list[Requirement] = Field(
        default_factory=list,
        max_length=64,
        description=(
            "Additional Python distribution requirements inferred from original source, "
            "e.g. Pillow or numpy>=1.26,<3. Uploaded manifest constraints also apply. "
            "Use [] when the manifest or standard library covers the project."
        ),
    )


INSTRUCTIONS = """
Choose dependencies as part of the execution plan, including libraries needed by the validator.
The worker automatically installs uploaded requirements.txt (preferred) or static
pyproject.toml [project].dependencies, plus the plan's dependencies, into a private environment.
It selects the nearest manifest from working_directory up to the upload root.
Infer additional Python DISTRIBUTION names from original imports and package usage; for example
PIL maps to Pillow and sklearn maps to scikit-learn. Exclude standard-library and uploaded local
modules. Respect all explicit version pins and constraints. Do not replace or loosen them.
Choose versions only when source or supplied metadata establishes a compatibility requirement;
otherwise an unversioned distribution name lets the resolver choose a compatible release.
Explain inferred packages in the plan summary. Ask only when the package identity or necessary
version cannot be inferred reliably. Never guess private package URLs or request credentials.
Dependencies must be registry package requirements, not shell commands, paths, URLs or pip flags.
Dependency setup is authorized by submission and counts toward execution/startup time and budget;
allow enough time for installation, including probes. Installation logs are execution logs.
Review actual installation/probe failures before revising a plan. Do not claim setup succeeded
until the worker reports it. After validation, keep the dependency plan frozen with the code.
Worker capabilities report the available runtime (CPU, CUDA, or MPS) and python.pytorch
with the bundled PyTorch version. Keep that exact build and its compatibility constraints.
GPU-capable builds also run on CPU, but GPU execution requires a reported usable device
and source that places its model and tensors on that device. Prefer compatible GPUs for
automatic device selection; CPU is the fallback when none is available. Explicit GPU-only
requests must not be silently downgraded. Additional Python libraries can be installed.
Do not choose Blender or non-Python runtimes. OS packages and model weights are not
installed by this dependency setup step.
"""
