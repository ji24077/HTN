"""The only tests worth writing at hour 0. Run them before every push."""

import pytest
from pydantic import ValidationError

from gpushare.contracts import (
    JobConfig,
    ShareRules,
    TrainSync,
    WorkerRegister,
    classify_chip,
    parse_event,
)


def test_event_roundtrip():
    ev = WorkerRegister(
        worker_id="w1", owner="junho", vendor="nvidia", gpu="RTX 3090",
        vram_gb=24, cc=8.6, chip_class="ampere_24gb",
        share=ShareRules(not_gaming=True), credits_per_hour=1.0,
    )
    back = parse_event(ev.model_dump_json())
    assert back == ev


def test_typo_in_chip_class_fails_loudly():
    """This is the whole point. A typo used to silently break per-chip learning."""
    with pytest.raises(ValidationError):
        WorkerRegister(
            worker_id="w1", owner="junho", vendor="nvidia", gpu="RTX 3090",
            vram_gb=24, cc=8.6, chip_class="ampere24gb",      # missing underscore
            share=ShareRules(), credits_per_hour=1.0,
        )


def test_fixed_work_invariant_enforced():
    """Ji cannot hand Ethan a config that does less work and calls it faster."""
    ok = dict(job_id="j1", model="nanogpt-124m", total_steps=8000, seq_len=1024,
              dtype="bf16", attention="sdpa", micro_batch=16, grad_accum=1,
              global_batch_tokens=32768, H=190, workers=["w1", "w2"])
    JobConfig(**ok)                                   # 16*1*2*1024 == 32768

    with pytest.raises(ValidationError):
        JobConfig(**{**ok, "micro_batch": 32})        # batch up, accum not down


def test_H_is_clamped():
    with pytest.raises(ValidationError):
        JobConfig(job_id="j1", model="m", total_steps=10, seq_len=1024,
                  dtype="bf16", attention="sdpa", micro_batch=16, grad_accum=1,
                  global_batch_tokens=32768, H=2000, workers=["w1", "w2"])


def test_junk_lines_are_skipped_not_crashed():
    """torch warnings will leak onto stdout. The daemon must survive them."""
    assert parse_event("UserWarning: something something") is None
    assert parse_event("") is None


def test_classify_is_shared():
    assert classify_chip(gpu_name="RTX 3090", cc=8.6, vram_gb=24, is_amd=False) == "ampere_24gb"
    assert classify_chip(gpu_name="RTX 4090", cc=8.9, vram_gb=24, is_amd=False) == "ada_24gb"
    assert classify_chip(gpu_name="MI250", cc=0.0, vram_gb=128, is_amd=True) == "cdna_amd"
    assert classify_chip(gpu_name="GTX 970", cc=5.2, vram_gb=4, is_amd=False) == "default"


def test_participants_shrink_is_representable():
    """Demo beat 4: the sync after a reclaim has a shorter participant list."""
    s = TrainSync(round=7, participants=["w2"], bytes=40_000_000,
                  duration_s=2.1, global_loss=1.79)
    assert len(s.participants) == 1
