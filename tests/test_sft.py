"""Compare optimized math with the actual Transformers Qwen2 implementation."""

import copy
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from gpushare.trainer.sft import (  # noqa: E402
    prepare_batch,
    standard_loss_sum,
    supervised_tokens,
    target_loss_sum,
)


def tiny_model():
    torch.manual_seed(7)
    config = transformers.Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    return transformers.Qwen2ForCausalLM(config)


def batch():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0], [11, 12, 13, 14, 15, 16, 17, 0]])
    labels = torch.tensor(
        [[-100, -100, -100, 4, 5, 6, -100, -100], [-100, 12, 13, 14, 15, 16, 17, -100]]
    )
    return prepare_batch(ids, labels)


@pytest.mark.parametrize(
    ("lora", "nonzero_adapter"),
    [(False, False), (True, False), (True, True)],
    ids=["full", "fresh-lora", "nonzero-lora"],
)
@pytest.mark.parametrize("decoder_checkpointing", [False, True])
def test_target_projection_matches_reference_loss_and_every_gradient(
    lora, nonzero_adapter, decoder_checkpointing
):
    reference = tiny_model()
    if lora:
        peft = pytest.importorskip("peft")
        reference = peft.get_peft_model(
            reference,
            peft.LoraConfig(
                task_type="CAUSAL_LM",
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "v_proj"],
            ),
        )
        if nonzero_adapter:
            # Fresh LoRA B matrices are zero, so their A gradients vanish.
            # Exercise continued training with both factors active, initializing
            # before deepcopy so the reference and candidate remain identical.
            adapter_rng = torch.Generator().manual_seed(41)
            with torch.no_grad():
                for name, parameter in reference.named_parameters():
                    if "lora_A" in name or "lora_B" in name:
                        parameter.normal_(std=0.03, generator=adapter_rng)
    candidate = copy.deepcopy(reference)
    if decoder_checkpointing:
        candidate.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    ids, labels, mask = batch()
    expected = standard_loss_sum(reference, ids, labels, mask)
    actual = target_loss_sum(candidate, ids, labels, mask, chunk_size=2)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
    expected.backward()
    actual.backward()
    for (name, left), (_, right) in zip(
        reference.named_parameters(), candidate.named_parameters(), strict=True
    ):
        if left.requires_grad:
            assert left.grad is not None and right.grad is not None, name
            if nonzero_adapter:
                assert bool(left.grad.abs().sum() > 0), name
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=3e-6, msg=name)


def test_accumulation_matches_whole_batch_with_unequal_answer_lengths():
    reference = tiny_model()
    candidate = copy.deepcopy(reference)
    ids, labels, mask = batch()
    count = supervised_tokens(labels)
    (standard_loss_sum(reference, ids, labels, mask) / count).backward()
    for row in range(2):
        (
            target_loss_sum(
                candidate,
                ids[row : row + 1],
                labels[row : row + 1],
                mask[row : row + 1],
                chunk_size=3,
            )
            / count
        ).backward()
    for (name, left), (_, right) in zip(
        reference.named_parameters(), candidate.named_parameters(), strict=True
    ):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=3e-6, msg=name)


def test_padding_trim_preserves_loss_even_when_eos_is_pad():
    model = tiny_model().eval()
    ids, labels, _ = batch()
    ids[0, 5] = 0  # this is a supervised EOS, not padding
    labels[0, 5] = 0
    padded_ids = torch.nn.functional.pad(ids, (0, 5))
    padded_labels = torch.nn.functional.pad(labels, (0, 5), value=-100)
    trimmed = prepare_batch(padded_ids, padded_labels)
    full = prepare_batch(padded_ids, padded_labels, trim=False)
    assert trimmed[2][0, 5] == 1
    assert trimmed[0].shape[1] == 7
    with torch.no_grad():
        torch.testing.assert_close(
            target_loss_sum(model, *trimmed), standard_loss_sum(model, *full)
        )


def test_no_answer_tokens_is_rejected():
    ids, labels, mask = batch()
    with pytest.raises(ValueError, match="no supervised"):
        target_loss_sum(tiny_model(), ids, torch.full_like(labels, -100), mask)


def test_full_checkpoint_roundtrip_with_tied_embeddings(tmp_path):
    model = tiny_model().eval()
    model.save_pretrained(tmp_path)
    loaded = transformers.Qwen2ForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            target_loss_sum(model, *batch()), target_loss_sum(loaded, *batch())
        )


def test_encoder_refuses_to_silently_drop_long_examples():
    path = Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("train_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Tokenizer:
        eos_token = "!"
        pad_token_id = 0

        def __call__(self, value, **kwargs):
            return type("Tokens", (), {"input_ids": [1] * len(value)})()

    with pytest.raises(SystemExit, match="rather than silently"):
        module.encode(Tokenizer(), [("short", "ok"), ("much too long", "answer")], 10)
