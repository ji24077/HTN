"""Run the benchmark's retention path using CPU tensors and a fake generator."""

import importlib.util
import json
from pathlib import Path

import pytest

from gpushare.dashboard import runner


@pytest.mark.parametrize("change_candidate", [False, True])
def test_benchmark_retains_outputs_and_strictly_checks_batching(monkeypatch, change_candidate):
    torch = pytest.importorskip("torch")
    path = Path(__file__).parents[1] / "scripts/benchmark_inference.py"
    spec = importlib.util.spec_from_file_location("inference_benchmark_safety", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    rows = [{"sentence": f"Person {i}", "record": {"name": f"Person {i}", "age": 30,
             "org": "Lab", "role": "engineer", "year": 2020}} for i in (0, 1)]

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, prompts, **kwargs):
            return {"input_ids": torch.tensor([[int("Person 1" in p)] for p in prompts])}

        def decode(self, tokens, **kwargs):
            index, changed = tokens.tolist()
            record = dict(rows[index]["record"])
            if changed:
                record["year"] = 2021
            return json.dumps(record)

    class Model:
        def generate(self, input_ids, **kwargs):
            changed = int(change_candidate and len(input_ids) > 1)
            output = torch.tensor([[int(row[0]), changed] for row in input_ids])
            return torch.cat([input_ids, output], dim=1)

    before, _ = benchmark.run(Model(), Tokenizer(), rows, batch=1, max_new=2)
    after, _ = benchmark.run(Model(), Tokenizer(), rows, batch=2, max_new=2)
    assert [sample["source_index"] for sample in after["samples"]] == [0, 1]
    assert before["generated_tokens"] == after["generated_tokens"] == 4
    assert before["evaluation"] == after["evaluation"]
    gate = runner._quality(before, after)
    assert gate["status"] == ("regressed" if change_candidate else "ok")
    assert gate["output_preservation_verified"] is not change_candidate
