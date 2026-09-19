"""Ji — the four axes, split so each can be varied on its own.

DELIBERATELY NOT IN contracts.py. That file is frozen and shared; these shapes
are the agent's internal cost model and neither Jack nor Ethan needs them.
`JobConfig.model` stays a plain string — MODELS maps it back to a spec here.

    ModelSpec   params/layers/d_model     -> sync bytes, memory, FLOPs
    DataSpec    corpus size, seq_len      -> total_steps
    ChipSpec    peak TFLOPS, VRAM, price  -> mostly from the RunPod catalog
    NetSpec     bandwidth, latency        -> T_sync, and therefore H
"""

from __future__ import annotations

from dataclasses import dataclass

from gpushare.contracts import Attention, ChipClass, DType

# Bytes per element. fp32 masters and optimizer state are always fp32 regardless.
DTYPE_BYTES: dict[DType, int] = {"bf16": 2, "fp16": 2, "fp32": 4}


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelSpec:
    """A decoder-only transformer, described by the numbers the cost model needs."""

    name: str
    params: int  # total, including embeddings
    layers: int
    d_model: int
    n_heads: int
    vocab: int
    n_kv_heads: int | None = None  # None == MHA. Fewer KV heads is GQA.

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    def param_flops_per_token(self) -> float:
        """Every matmul that scales with parameter count. fwd + bwd."""
        return 6 * self.params

    def attn_flops_per_token(self, seq_len: int) -> float:
        """Attention. Scales with seq_len INSTEAD of with parameters, which is why
        long sequences change which lever matters — and why sdpa, which only
        touches this term, is worth more at 2048 than at 512."""
        return 12 * self.layers * self.d_model * seq_len

    def flops_per_token(self, seq_len: int) -> float:
        """Forward + backward FLOPs per token (the nanoGPT / PaLM convention)."""
        return self.param_flops_per_token() + self.attn_flops_per_token(seq_len)

    def sync_bytes(self, dtype: DType) -> int:
        """Pseudo-gradient on the wire, once per DiLoCo round.

        This number divided by the link bandwidth IS T_sync, and T_sync is what
        compute_H() keys on. Getting the model bigger moves H, which is the whole
        reason H is a measured lever rather than a constant.
        """
        return self.params * DTYPE_BYTES[dtype]

    def static_bytes(self, dtype: DType) -> int:
        """Weights + grads + AdamW state. Independent of batch size.

        Mixed precision keeps an fp32 master copy and fp32 moments, so bf16 saves
        far less memory than "half the dtype" suggests. Worth modelling honestly:
        it's why the batch-size search has less headroom than people expect.
        """
        b = DTYPE_BYTES[dtype]
        weights = self.params * b
        grads = self.params * b
        adam = self.params * 4 * 2  # fp32 exp_avg + exp_avg_sq
        master = self.params * 4 if dtype != "fp32" else 0
        return weights + grads + adam + master

    def activation_bytes(
        self, *, micro_batch: int, seq_len: int, dtype: DType, attention: Attention
    ) -> int:
        """Saved activations. This is the term batch-size search actually spends.

        Split in two on purpose:
          linear  — scales with tokens. Unavoidable.
          attn    — scales with seq_len SQUARED, and only when attention is eager.
                    sdpa/flash never materialises the seq×seq matrix, so this term
                    disappears entirely. That is the whole memory case for lever 2.
        """
        b = DTYPE_BYTES[dtype]
        linear = micro_batch * seq_len * self.layers * self.d_model * _ACT_PER_TOKEN * b

        # Logits: micro_batch x seq_len x vocab, plus an fp32 copy because
        # cross-entropy upcasts. Negligible for GPT-2's 50k vocab on a 124M
        # model; DOMINANT for Qwen2.5-0.5B, where a 152k vocab against 896
        # hidden means the logits outweigh the weights. Omitting it is why the
        # first version of this model under-predicted peak memory.
        logits = micro_batch * seq_len * self.vocab * (b + 4)

        if attention == "sdpa":
            return linear + logits
        # Eager stages a seq x seq score matrix per QUERY head. GQA shrinks the
        # KV cache, not this term — the scores still exist per query head.
        attn = micro_batch * self.n_heads * seq_len * seq_len * self.layers * b
        return linear + logits + attn

    def min_vram_gb(
        self, *, dtype: DType, micro_batch: int, seq_len: int, attention: Attention
    ) -> float:
        """Smallest card this fits on at all. Feeds pick_chips()."""
        total = self.static_bytes(dtype) + self.activation_bytes(
            micro_batch=micro_batch, seq_len=seq_len, dtype=dtype, attention=attention
        )
        return total / 1e9


# Activations retained per token per layer, in multiples of d_model. A rough
# constant for a standard block (norms, qkv, proj, mlp up/down). The per-chip
# `act_overhead` in calibrate.py absorbs whatever this misses.
_ACT_PER_TOKEN = 12


MODELS: dict[str, ModelSpec] = {
    "nanogpt-124m": ModelSpec(
        name="nanogpt-124m", params=124_000_000, layers=12, d_model=768, n_heads=12, vocab=50_257
    ),
    "nanogpt-350m": ModelSpec(
        name="nanogpt-350m", params=350_000_000, layers=24, d_model=1024, n_heads=16, vocab=50_257
    ),
    # Cuts sync cost ~4x, but the simulator says it still pins H at the ceiling
    # on a 100 Mbps link — shrinking the model is not enough to escape it. Kept
    # as the cheap config for smoke tests, not as the fix for a slow network.
    # The actual workload: full fine-tune, JSON extraction. Real config from
    # Qwen/Qwen2.5-0.5B — 24 layers, 896 hidden, 14 query heads over 2 KV
    # heads (GQA), tied embeddings. The 151,936 vocab is what makes the logits
    # term above matter.
    "qwen2.5-0.5b": ModelSpec(
        name="qwen2.5-0.5b",
        params=494_000_000,
        layers=24,
        d_model=896,
        n_heads=14,
        n_kv_heads=2,
        vocab=151_936,
    ),
    "nanogpt-30m": ModelSpec(
        name="nanogpt-30m", params=30_000_000, layers=6, d_model=384, n_heads=6, vocab=50_257
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataSpec:
    name: str
    tokens_total: int
    seq_len: int

    def total_steps(self, global_batch_tokens: int, epochs: float = 1.0) -> int:
        """How many optimizer steps one pass over the corpus takes.

        total_steps is an INPUT to JobConfig, so deriving it here is what makes
        "change the dataset, everything downstream follows" true rather than a
        number somebody typed.
        """
        return max(1, int(self.tokens_total * epochs / global_batch_tokens))


DATASETS: dict[str, DataSpec] = {
    "tinyshakespeare": DataSpec("tinyshakespeare", tokens_total=1_100_000, seq_len=1024),
    "openwebtext-1b": DataSpec("openwebtext-1b", tokens_total=1_000_000_000, seq_len=1024),
    "fineweb-10b": DataSpec("fineweb-10b", tokens_total=10_000_000_000, seq_len=1024),
}


# ─────────────────────────────────────────────────────────────────────────────
# Chip
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ChipSpec:
    """One rentable GPU. vram/price/availability come from the RunPod catalog;
    tflops and memory bandwidth come from the table below."""

    chip_class: ChipClass
    name: str
    vram_gb: float
    cc: float
    tflops_bf16: float  # dense, fp32-accumulate — what training actually gets
    mem_bw_gbs: float  # decode is memory-bound, so inference needs this
    credits_per_hour: float

    def tflops_for(self, dtype: DType) -> float:
        """Peak for this dtype.

        These are NOMINAL peaks and they only have to be CONSISTENT, not exact —
        the fitted MFU in calibrate.py multiplies through them, so a 10% error in
        the table shows up as a 10% shift in MFU and cancels. What would actually
        break the model is using sparse or fp16-accumulate numbers for one chip
        and dense fp32-accumulate for another.
        """
        if dtype == "fp32":
            return self.tflops_bf16 / 2  # TF32 path; true fp32 is far worse
        return self.tflops_bf16

    def trainable(self) -> bool:
        """cc < 7.0 has no fp16/bf16 tensor cores. Same rule as contracts.is_trainable."""
        return self.cc >= 7.0


CHIPS: dict[str, ChipSpec] = {
    "RTX 4090": ChipSpec("ada_24gb", "RTX 4090", 24, 8.9, 82.6, 1008, 0.74),
    "RTX 3090": ChipSpec("ampere_24gb", "RTX 3090", 24, 8.6, 35.6, 936, 0.50),
    "RTX A5000": ChipSpec("ampere_24gb", "RTX A5000", 24, 8.6, 27.8, 768, 0.27),
    "A40": ChipSpec("ampere_24gb", "A40", 48, 8.6, 37.4, 696, 0.49),
    "RTX A6000": ChipSpec("ampere_24gb", "RTX A6000", 48, 8.6, 38.7, 768, 0.53),
    "L4": ChipSpec("ada_24gb", "L4", 24, 8.9, 30.3, 300, 0.49),
    # No bf16 on Volta — the chip agent has to fall back to fp16 + GradScaler.
    "Tesla V100": ChipSpec("turing_16gb", "Tesla V100", 16, 7.0, 31.4, 900, 0.19),
    "MI300X": ChipSpec("cdna_amd", "MI300X", 192, 9.4, 653.0, 5300, 2.39),
}


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NetSpec:
    name: str
    bandwidth_mbps: float
    latency_ms: float

    def bytes_per_s(self) -> float:
        return self.bandwidth_mbps * 1e6 / 8


NETS: dict[str, NetSpec] = {
    # Measured ceiling, not an estimate: RunPod caps pod-to-pod global networking
    # at 100 Mbps regardless of region. That is home-broadband speed, which is
    # exactly the regime DiLoCo exists for — and it means H moves for real here.
    "runpod-global": NetSpec("runpod-global", bandwidth_mbps=100, latency_ms=30),
    "tailscale-direct": NetSpec("tailscale-direct", bandwidth_mbps=200, latency_ms=20),
    "tailscale-derp": NetSpec("tailscale-derp", bandwidth_mbps=20, latency_ms=90),
    "lan-wired": NetSpec("lan-wired", bandwidth_mbps=1000, latency_ms=1),
}
