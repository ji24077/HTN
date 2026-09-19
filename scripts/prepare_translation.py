"""Prepare an auditable native-source translation without a model or execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from gpushare.portability.scripts import translate_script, validate_demo_edit


def prepare(source_path: Path, target: str, output: Path) -> dict:
    """Create a fresh review folder; never overwrite previous work or execute input."""
    source = source_path.read_bytes().decode("utf-8")
    translated = translate_script(source, target)
    findings = translated["findings"] + validate_demo_edit(source, translated["source"], "translation")
    blocked = any(item["severity"] == "blocker" for item in findings)
    result = {
        "schema_version": 1,
        "status": "blocked" if blocked else "prepared_unverified",
        "gpu_verified": False,
        "code_executed": False,
        "model_used": False,
        "changed": translated["changed"],
        "source_backend": translated["source_backend"],
        "target_backend": translated["target_backend"],
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "translated_sha256": hashlib.sha256(translated["source"].encode("utf-8")).hexdigest(),
        "mappings": translated["mappings"],
        "findings": findings,
        "files": {"original": "original.py", "translated": "translated.py"},
        "next_validation": ["target compilation", "actual native GPU execution", "fixed numerical checks", "training/checkpoint checks", "same-device timing"],
        "note": "A static draft is not evidence of successful GPU translation, correctness, or speedup.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "original.py").write_text(source, encoding="utf-8", newline="")
    (output / "translated.py").write_text(translated["source"], encoding="utf-8", newline="")
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path, help="Python script with one CUDA_SOURCE or HIP_SOURCE literal")
    parser.add_argument("--target", choices=("amd", "nvidia"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory; existing paths are refused")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.script, args.target, args.out)
    except (OSError, UnicodeError, ValueError) as exc:
        print(json.dumps({"status": "error", "gpu_verified": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 1 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
