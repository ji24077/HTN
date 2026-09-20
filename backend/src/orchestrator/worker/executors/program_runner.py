"""Trusted program driver copied into each isolated execution workspace.

Uses only the standard library. The uploaded validator owns domain checks such
as loading a checkpoint or decoding an image and checking its dimensions.
"""

import json
import math
import os
import runpy
import stat
import sys


def output_path(directory, name):
    if (
        not name
        or "\\" in name
        or ":" in name
        or any(p in {"", ".", ".."} for p in name.split("/"))
    ):
        raise ValueError("Invalid output path")
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Output escapes its directory")
    for parent in (path, *path.parents):
        if parent == directory.parent:
            break
        if parent.is_symlink():
            raise ValueError("Output symlinks are not allowed")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Output must be a regular file, not a link or device")
    if info.st_size == 0:
        raise ValueError("Outputs must be nonempty")
    return path


def run_script(root, working, entrypoint, arguments, output):
    path = root / entrypoint
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Invalid uploaded entrypoint")
    os.chdir(working)
    sys.path.insert(0, str(path.parent))
    sys.argv = [
        str(path),
        *[argument.replace("{output_dir}", str(output)) for argument in arguments],
    ]
    try:
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise ValueError(f"{entrypoint} exited with status {exc.code}") from exc


def run(request, root):
    plan = request["program"]
    output = root / "__dispatch_outputs__"
    working = (root / request.get("working_directory", ".")).resolve()
    run_script(root, working, request["entrypoint"], request["args"], output)
    if plan["probe"]:
        return {"ok": True, "validation": {"probe_only": True}}

    # A separate uploaded validator checks the actual full-run outputs. Exiting
    # successfully from training or rendering alone is not accepted as success.
    run_script(root, working, plan["validator"], plan["validation_args"], output)
    declared = plan["outputs"]
    for item in declared:
        output_path(output, item["path"])
    metrics = {}
    if plan["metrics"]:
        path = output_path(output, "metrics.json")
        if path.stat().st_size > 48000:
            raise ValueError("Validation metrics exceed 48 KiB")
        metrics = json.loads(path.read_text())
        if not isinstance(metrics, dict):
            raise ValueError("Validation metrics must be an object")
        # Return only the contracted measurements, never an unbounded report.
        metrics = {rule["name"]: metrics.get(rule["name"]) for rule in plan["metrics"]}
        for rule in plan["metrics"]:
            value = metrics[rule["name"]]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Missing or nonfinite metric: {rule['name']}")
            if rule["minimum"] is not None and value < rule["minimum"]:
                raise ValueError(f"Metric {rule['name']} is below the accepted minimum")
            if rule["maximum"] is not None and value > rule["maximum"]:
                raise ValueError(f"Metric {rule['name']} exceeds the accepted maximum")
    return {
        "ok": True,
        "validation": {
            "validator": plan["validator"],
            "metrics": metrics,
            "output_count": len(declared),
        },
    }


def collect_outputs(directory, names=None):
    """Called by the parent only after the project's process group has exited."""
    if not directory.exists():
        return []
    if directory.is_symlink():
        raise ValueError("Output directory cannot be a symlink")
    if names is None:
        names = []
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("Output symlinks are not allowed")
            if not path.is_dir():
                names.append(path.relative_to(directory).as_posix())
            if len(names) > 100:
                raise ValueError("Execution produced more than 100 output files")
    if len(names) > 100 or len(set(names)) != len(names):
        raise ValueError("Output list must have at most 100 unique files")
    files = [(name, output_path(directory, name)) for name in names]
    return files
