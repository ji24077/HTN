"""Measure resident sentence-to-JSON latency with frozen inputs and exact parity.

Runs one GPU, serial requests, BF16, greedy decoding. No CPU fallback, account
credentials, rentals or network model downloads. Results include unsuccessful
optimization trials. Compilation/model loading are reported separately.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

from gpushare.agent.latency import (
    benchmark_exit_code,
    compare,
    contract,
    model_identity,
    read_cases,
    summarize,
)
from gpushare.agent.task import PROMPT, parse_output


def save(path: Path, report: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def request(torch, model, tokenizer, case, config):
    torch.cuda.synchronize()
    start = time.perf_counter()
    inputs = tokenizer(PROMPT.format(sentence=case["sentence"]), return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    output = model.generate(**inputs, generation_config=config)
    torch.cuda.synchronize()
    tokens = output[0, inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(tokens, skip_special_tokens=True)
    parsed = parse_output(raw)
    elapsed = time.perf_counter() - start
    return {"case_id": case["id"], "raw_output": raw,
            "parsed": parsed.model_dump() if parsed is not None else None,
            "new_tokens": int(tokens.numel()), "latency_s": elapsed}


def benchmark(args):
    # Hash all model bytes before loading; every GPU receives exactly these files.
    cases = read_cases(args.cases)
    identity = model_identity(args.base, args.adapter)
    if identity["sha256"] != args.expected_model_sha256:
        raise ValueError("model bytes do not match the frozen local checkpoint")
    task = contract(cases, identity, max_new=args.max_new, rounds=args.rounds, warmups=args.warmups)
    if args.out.exists():
        raise ValueError("--out must be a new path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "status": "starting", "contract": task,
              "model_identity": identity, "cases": cases, "samples": [], "warmup_samples": [],
              "gpu_verified": False, "setup": {}, "errors": [], "optimization": args.optimization}
    save(args.out, report)
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from transformers.generation.configuration_utils import CompileConfig

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one visible GPU is required; CPU fallback is forbidden")
        torch.cuda.set_device(0)
        actual_vendor = "amd" if torch.version.hip else "nvidia"
        if actual_vendor != args.expect_vendor:
            raise RuntimeError("unexpected GPU vendor")
        name = torch.cuda.get_device_name(0)
        if args.expect_gpu.lower() not in name.lower():
            raise RuntimeError(f"unexpected GPU model: {name}")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.optimization == "compile-strict":
            # Retain eager intermediate BF16 rounding across compiler fusions.
            # This does not change the stored or execution precision policy.
            import torch._inductor.config as inductor_config

            inductor_config.emulate_precision_casts = True
        props = torch.cuda.get_device_properties(0)
        gpu_uuid = str(getattr(props, "uuid", None) or "")
        if not gpu_uuid:
            gpu_uuid = f"runpod:{args.pod_id}:device-0"
        report["environment"] = {"gpu": name, "gpu_uuid": gpu_uuid, "vendor": actual_vendor,
                                 "torch": str(torch.__version__), "cuda": torch.version.cuda,
                                 "hip": torch.version.hip, "transformers": version("transformers"),
                                 "peft": version("peft"), "python": platform.python_version(),
                                 "cpu_count": os.cpu_count(), "torch_threads": 4,
                                 "visible_devices": 1, "vram_bytes": props.total_memory}
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                  attn_implementation="sdpa", local_files_only=True)
        model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True,
                                         autocast_adapter_dtype=True).to("cuda").eval()
        if {p.device.type for p in model.parameters()} != {"cuda"}:
            raise RuntimeError("model parameters are not entirely on the selected GPU")
        # Keep the existing evaluation policy: BF16 base and FP32 adapter weights.
        for parameter_name, parameter in model.named_parameters():
            expected = torch.float32 if "lora_" in parameter_name else torch.bfloat16
            if parameter.is_floating_point() and parameter.dtype != expected:
                raise RuntimeError(f"unexpected parameter precision: {parameter_name}")
        report["environment"]["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        torch.cuda.synchronize()
        report["setup"]["model_load_s"] = time.perf_counter() - started
        report["gpu_verified"] = True
        report["status"] = "warming"
        configurations = {}
        for label in ("baseline", "optimized"):
            conf = GenerationConfig.from_model_config(model.config)
            conf.max_new_tokens = args.max_new
            conf.do_sample = False
            conf.use_cache = True
            conf.pad_token_id = tokenizer.pad_token_id
            conf.cache_implementation = "dynamic" if label == "baseline" else "static"
            conf.disable_compile = label == "baseline" or args.optimization == "static"
            if not conf.disable_compile:
                conf.compile_config = CompileConfig(fullgraph=True, mode="reduce-overhead")
            configurations[label] = conf
        save(args.out, report)
        active = ["baseline", "optimized"]
        with torch.inference_mode():
            # Each exact prompt/engine is warmed; compile/warmup time is never hidden.
            for label in list(active):
                started = time.perf_counter()
                try:
                    for case in cases:
                        for index in range(args.warmups):
                            trial = request(torch, model, tokenizer, case, configurations[label])
                            report["warmup_samples"].append({**trial, "engine": label, "round": index})
                        print(json.dumps({"phase": "warmup", "engine": label, "case": case["id"]}), flush=True)
                        save(args.out, report)
                except Exception as exc:
                    if label == "baseline":
                        raise
                    report["errors"].append({"engine": label, "phase": "warmup", "type": type(exc).__name__,
                                             "detail": str(exc)[-2000:]})
                    active.remove(label)
                report["setup"][f"{label}_warmup_s"] = time.perf_counter() - started
            # An independent profiler sample is outside latency measurements.
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    request(torch, model, tokenizer, cases[0], configurations["baseline"])
                device_events = [e for e in profiler.events() if str(e.device_type).endswith("CUDA")]
                report["profiler"] = {"device_events": len(device_events),
                                       "device_time_us": sum(e.device_time_total for e in device_events)}
                profiler.export_chrome_trace(str(args.out.with_suffix(".trace.json")))
            except Exception as exc:
                report["profiler"] = {"error_type": type(exc).__name__, "device_events": 0}
            report["status"] = "measuring"
            for round_index in range(args.rounds):
                # Alternating engine order and rotating case order reduce ordering bias.
                ordered = cases[round_index % len(cases):] + cases[:round_index % len(cases)]
                for case_index, case in enumerate(ordered):
                    order = active if (round_index + case_index) % 2 == 0 else list(reversed(active))
                    for label in list(order):
                        torch.cuda.reset_peak_memory_stats()
                        try:
                            trial = request(torch, model, tokenizer, case, configurations[label])
                            trial.update(engine=label, round=round_index,
                                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                         peak_reserved_bytes=torch.cuda.max_memory_reserved())
                            report["samples"].append(trial)
                        except Exception as exc:
                            if label == "baseline":
                                raise
                            report["errors"].append({"engine": label, "phase": "measure", "type": type(exc).__name__,
                                                     "detail": str(exc)[-2000:]})
                            active.remove(label)
                        save(args.out, report)
                print(json.dumps({"phase": "measure", "round": round_index + 1, "requests": len(report["samples"])}), flush=True)
        report["summaries"] = {label: summarize(report, label) for label in active}
        try:
            from torch._dynamo.utils import counters
            report["compilation"] = {"requested": args.optimization != "static",
                                      "emulate_precision_casts": args.optimization == "compile-strict",
                                      "unique_graphs": counters["stats"]["unique_graphs"],
                                      "calls_captured": counters["stats"]["calls_captured"]}
        except (ImportError, KeyError):
            report["compilation"] = {"requested": args.optimization != "static", "verified": False}
        if "optimized" in active:
            report["comparison"] = compare(report, report, before_engine="baseline", after_engine="optimized")
        report["status"] = "measured" if "optimized" in active else "baseline_only"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"phase": "run", "type": type(exc).__name__, "detail": str(exc)[-2000:]})
    finally:
        save(args.out, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect-vendor", choices=("nvidia", "amd"), required=True)
    parser.add_argument("--expect-gpu", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--pod-id", required=True, help="owned pod identity when the backend exposes no device UUID")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--optimization", choices=("compile-strict", "compile", "static"), default="compile-strict")
    parser.add_argument("--measure-only", action="store_true",
                        help="explicitly allow exit zero for completed but rejected measurements; never means a candidate passed")
    args = parser.parse_args()
    result = benchmark(args)
    print(json.dumps({"status": result["status"], "out": str(args.out),
                      "speedup_verified": result.get("comparison", {}).get("speedup_verified", False)}), flush=True)
    return benchmark_exit_code(result, measure_only=args.measure_only)


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        raise SystemExit(main())
