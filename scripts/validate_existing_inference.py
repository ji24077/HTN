"""Run the unchanged evaluator, then validate candidates through its own functions.

No rentals, downloads, credentials, model changes or edited reference code.
Use a pinned, populated HF cache with offline mode. Exit 2 rejects a candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gpushare.agent.evaluation import evaluation_identity
from gpushare.agent.latency import file_digest, model_identity
from gpushare.agent.prediction_validation import compare_predictions

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts/evaluate.py"


def write(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_inputs(args):
    from huggingface_hub import try_to_load_from_cache

    identity = model_identity(args.base, args.model)
    if identity["sha256"] != args.expected_model_sha256:
        raise ValueError("frozen model files changed")
    config = json.loads((args.model / "adapter_config.json").read_text(encoding="utf-8"))
    reference = config["base_model_name_or_path"]
    # The original loader resolves this ID from the cache. Check those exact
    # bytes as well as the explicitly named snapshot before executing it.
    for name, expected in identity["files"].items():
        if name.startswith("base/"):
            cached = try_to_load_from_cache(reference, name.removeprefix("base/"), revision="main")
            if not isinstance(cached, str) or file_digest(Path(cached)) != expected:
                raise ValueError("the existing loader's cached base differs from the frozen snapshot")
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()][:args.n]
    if len(rows) != args.n:
        raise ValueError("the complete declared dataset is required")
    return identity, rows


def candidate(args, rows):
    import evaluate
    import torch
    from transformers.generation.configuration_utils import CompileConfig

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one actual GPU required")
    torch.set_num_threads(4)
    model = evaluate.load_model(str(args.model), torch.bfloat16).cuda().eval()
    tok = evaluate.load_tokenizer(str(args.model), padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    for name, parameter in model.named_parameters():
        if parameter.device.type != "cuda":
            raise RuntimeError("model was not entirely placed on the GPU")
        expected = torch.float32 if "lora_" in name else torch.bfloat16
        if parameter.is_floating_point() and parameter.dtype != expected:
            raise RuntimeError("the existing loader changed the precision policy")
    conf = model.generation_config
    conf.cache_implementation = "static"
    conf.disable_compile = args.candidate_engine == "static"
    if not conf.disable_compile:
        import torch._inductor.config as inductor_config

        inductor_config.emulate_precision_casts = True
        conf.compile_config = CompileConfig(fullgraph=True, mode="reduce-overhead")
    torch.cuda.synchronize()
    started = time.perf_counter()
    # Existing evaluator, including its tokenizer, greedy generation, complete
    # indexed outputs, ground-truth scoring and held-out loss calculation.
    result = evaluate.run_eval(model, tok, rows, batch=args.batch, seq_len=192,
                               max_new_tokens=128, length_bucketing=False)
    torch.cuda.synchronize()
    report = evaluate._serialise(result)
    report.update(evaluation=evaluation_identity(rows, max_new_tokens=128, seq_len=192),
                  runtime={"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
                           "cuda": torch.version.cuda, "hip": torch.version.hip,
                           "batch": args.batch, "dtype": "bf16", "seconds": time.perf_counter() - started},
                  inference={"batch": args.batch, "dtype": "bf16", "fuse_adapter": False,
                             "length_bucketing": False, "engine": args.candidate_engine},
                  evaluator_sha256=file_digest(EVALUATOR),
                  timing_note="Includes quality evaluation, compilation and held-out loss; not resident response latency.")
    write(args.out, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--engines", nargs="+", choices=("static", "compile-strict"), default=["static", "compile-strict"])
    parser.add_argument("--candidate-engine", choices=("static", "compile-strict"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("model", "base", "data", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if not 1 <= args.n <= 1000 or not 1 <= args.batch <= 32:
        parser.error("use 1..1000 cases and batch 1..32")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONIOENCODING="utf-8")
    identity, rows = verify_inputs(args)
    if args.candidate_engine:
        candidate(args, rows)
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    code_hash = file_digest(EVALUATOR)
    result = {"status": "running", "n": len(rows), "dataset_file_sha256": file_digest(args.data),
              "model_identity": identity, "reference_evaluator_sha256": code_hash,
              "reference_code_unchanged": True, "candidates": {}, "commands": []}
    report_path = args.out / "validation.json"

    def run(command, label):
        result["commands"].append({"label": label, "argv": command[1:]})
        write(report_path, result)
        print(json.dumps({"phase": label, "n": len(rows)}), flush=True)
        with (args.out / (label + ".log")).open("wb") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, timeout=1800)
        if file_digest(EVALUATOR) != code_hash:
            result["reference_code_unchanged"] = False
            raise RuntimeError("existing evaluator code changed during validation")
        return completed.returncode

    try:
        reference_path = args.out / "reference.json"
        reference_command = [sys.executable, str(EVALUATOR), "--model", str(args.model), "--data", str(args.data),
                             "--out", str(reference_path), "--n", str(args.n), "--batch", str(args.batch),
                             "--dtype", "bf16", "--max-new-tokens", "128", "--seq-len", "192", "--show", "0"]
        if run(reference_command, "unchanged-evaluator") != 0:
            raise RuntimeError("unchanged evaluator failed; inspect its retained log")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        result["reference"] = {k: reference[k] for k in ("n", "json_parse_rate", "exact_match_rate", "runtime")}
        for engine in dict.fromkeys(args.engines):
            candidate_path = args.out / (engine + ".json")
            command = [sys.executable, str(Path(__file__).resolve()), "--model", str(args.model), "--base", str(args.base),
                       "--expected-model-sha256", args.expected_model_sha256, "--data", str(args.data),
                       "--out", str(candidate_path), "--n", str(args.n), "--batch", str(args.batch), "--candidate-engine", engine]
            exit_code = run(command, engine)
            if exit_code:
                result["candidates"][engine] = {"passed": False, "exit_code": exit_code, "error": "candidate execution failed; see retained log"}
            else:
                measured = json.loads(candidate_path.read_text(encoding="utf-8"))
                result["candidates"][engine] = compare_predictions(reference, measured, rows)
            write(report_path, result)
        result["status"] = "passed" if all(c["passed"] for c in result["candidates"].values()) else "rejected"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        write(report_path, result)
    print(json.dumps({"status": result["status"], "out": str(report_path)}), flush=True)
    return 0 if result["status"] == "passed" else 2 if result["status"] == "rejected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
