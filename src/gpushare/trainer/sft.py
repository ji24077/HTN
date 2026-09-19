"""Shared SFT math: answer-only loss without a batch x sequence x vocab tensor.

Specific to the Qwen2 causal model used by this project. The decoder still
attends to the complete prompt. Only the output projection is restricted to
positions whose next token is supervised. Checkpointed projection chunks bound
the large vocabulary's training activation memory without changing the loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def supervised_tokens(labels: torch.Tensor) -> int:
    return int((labels[:, 1:] != -100).sum())


def prepare_batch(ids: torch.Tensor, labels: torch.Tensor, *, trim: bool = True):
    positions = torch.arange(labels.shape[1], device=labels.device)
    # EOS can also be the padding id. Labels, not token ids, distinguish them.
    ends = torch.where(labels != -100, positions + 1, 0).max(dim=1).values
    if bool((ends == 0).any()):
        raise ValueError("each example must contain supervised answer tokens")
    width = int(ends.max()) if trim else ids.shape[1]
    mask = positions[:width].unsqueeze(0) < ends.unsqueeze(1)
    return ids[:, :width], labels[:, :width], mask.long()


def target_loss_sum(model, ids, labels, attention_mask, *, chunk_size: int = 64):
    """Exact sum of next-token CE over answer tokens; normalize per global batch.

    A sum (rather than the mean of micro-batch means) makes accumulation
    invariant to different answer lengths and micro-batch partitions.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if base.config.model_type != "qwen2":
        raise ValueError("answer-only projection currently supports Qwen2 models")
    hidden = base.model(
        input_ids=ids, attention_mask=attention_mask, use_cache=False
    ).last_hidden_state
    targets = labels[:, 1:]
    valid = targets != -100
    hidden = hidden[:, :-1][valid]
    targets = targets[valid]
    if targets.numel() == 0:
        raise ValueError("batch has no supervised next tokens")

    def project_and_loss(h, y):
        return F.cross_entropy(base.lm_head(h).float(), y, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(targets), chunk_size):
        h, y = hidden[start : start + chunk_size], targets[start : start + chunk_size]
        if torch.is_grad_enabled():
            total = total + checkpoint(project_and_loss, h, y, use_reentrant=False)
        else:
            total = total + project_and_loss(h, y)
    return total


def standard_loss_sum(model, ids, labels, attention_mask):
    return model(
        input_ids=ids, labels=labels, attention_mask=attention_mask, use_cache=False
    ).loss * supervised_tokens(labels)
