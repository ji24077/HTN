"""Standard-library dependency bootstrap, copied into the supervised project process.

Installers share the launcher's process group and stripped environment. The worker
owns cancellation/deadlines, and its workspace cleanup removes the environment.
"""

import importlib.metadata
import json
import os
import site
import subprocess
import time
import tomllib
import venv
from pathlib import Path


def manifest(root, working):
    """Use the nearest manifest, walking from the working directory to the upload root."""
    root, working = root.resolve(), working.resolve()
    if not working.is_relative_to(root):
        raise ValueError("Invalid dependency working directory")
    while True:
        requirements = working / "requirements.txt"
        if requirements.is_file():
            return requirements, ["-r", str(requirements)]
        project = working / "pyproject.toml"
        if project.is_file():
            document = tomllib.loads(project.read_text(encoding="utf-8"))
            metadata = document.get("project", {})
            if "dependencies" in metadata.get("dynamic", []):
                raise ValueError("Dynamic project dependencies need an uploaded requirements.txt")
            dependencies = metadata.get("dependencies", [])
            if not isinstance(dependencies, list) or any(
                not isinstance(item, str) or not item.strip() for item in dependencies
            ):
                raise ValueError(
                    "pyproject.toml project.dependencies must be a list of requirements"
                )
            if dependencies:
                return project, dependencies
            if "project" in document:
                return None
        if working == root:
            return None
        working = working.parent


def prepare(root, working, request):
    """Re-exec this launcher in a private venv after installing declared dependencies."""
    if os.environ.get("DISPATCH_DEPENDENCIES_READY") == "1":
        return
    root = root.resolve()
    selected = manifest(root, working)
    additional = request.get("dependencies", [])
    if (
        not isinstance(additional, list)
        or len(additional) > 64
        or any(not isinstance(item, str) or not item or len(item) > 256 for item in additional)
    ):
        raise ValueError("Invalid planned Python dependencies")
    if selected is None and not additional:
        return
    source, arguments = selected if selected else (None, [])
    label = str(source.relative_to(root)) if source else "the agent's dependency plan"
    if source and additional:
        label += " and the agent's dependency plan"
    # Resolve manifest constraints and inferred requirements together. A conflict
    # must fail rather than silently replacing an uploaded version pin.
    if source and source.name == "requirements.txt":
        arguments = [*arguments, "--", *additional]
    else:
        arguments = ["--", *arguments, *additional]
    started = time.perf_counter()
    print(f"Installing Python dependencies from {label}…", flush=True)
    if additional:
        print("Agent-selected packages: " + ", ".join(additional), flush=True)
    environment = root / "__dispatch_environment__"
    try:
        builder = venv.EnvBuilder(
            with_pip=True, system_site_packages=True, symlinks=os.name != "nt"
        )
        builder.create(environment)
        context = builder.ensure_directories(environment)
        # A venv's system-site-packages points at the base Python, not at the
        # worker's own venv. Keep its CUDA/PyTorch/Pillow installations visible,
        # behind this attempt's packages, without installing into that environment.
        Path(context.lib_path, "__dispatch_worker__.pth").write_text(
            "\n".join(site.getsitepackages()) + "\n", encoding="utf-8"
        )
        env = {
            **os.environ,
            "VIRTUAL_ENV": str(environment),
            "PATH": context.bin_path + os.pathsep + os.environ.get("PATH", os.defpath),
        }
        # Keep this worker's exact PyTorch build (CUDA, MPS-capable, or CPU).
        # Additional packages must not silently replace the serving runtime.
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_version = None
        runtime_arguments = []
        if torch_version:
            constraints = root / "__dispatch_constraints__.txt"
            constraints.write_text(f"torch=={torch_version}\n", encoding="utf-8")
            runtime_arguments = ["--constraint", str(constraints)]
            build = torch_version.partition("+")[2]
            if build == "cpu" or (build.startswith("cu") and build[2:].isdigit()):
                runtime_arguments += [
                    "--extra-index-url",
                    f"https://download.pytorch.org/whl/{build}",
                ]
        subprocess.run(
            [
                context.env_exec_cmd,
                "-I",
                "-m",
                "pip",
                "--isolated",
                "--disable-pip-version-check",
                "install",
                "--no-input",
                "--no-cache-dir",
                *runtime_arguments,
                *arguments,
            ],
            cwd=source.parent if source else working,
            env=env,
            stdin=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError) and exc.output:
            output = (
                exc.output.decode(errors="replace") if isinstance(exc.output, bytes) else exc.output
            )
            print(output[-8000:], flush=True)
        raise RuntimeError(f"Dependency setup failed for {label}; see installation logs") from exc
    elapsed = time.perf_counter() - started
    print(f"Python dependencies ready ({elapsed:.1f}s). Starting project.\n", flush=True)
    # stdin has already been consumed. Preserve the credential-free request for
    # the replacement launcher; exec keeps the PID/process group and owner watchdog.
    request["__dispatch_dependency_seconds"] = elapsed
    (root / "__dispatch_request__.json").write_text(json.dumps(request), encoding="utf-8")
    env["DISPATCH_DEPENDENCIES_READY"] = "1"
    os.execve(
        context.env_exec_cmd,
        [context.env_exec_cmd, "-I", str(root / "__dispatch_launcher__.py")],
        env,
    )
