import copy

import pytest

from gpushare.agent.evaluation import evaluation_identity, require_comparable


@pytest.fixture
def experiment():
    identity = evaluation_identity(
        [{"sentence": "A", "record": {"age": 21}}], max_new_tokens=128, seq_len=256
    )
    return identity, {
        "evaluation": identity,
        "inference": {"batch": 4, "dtype": "bf16", "length_bucketing": False},
        "runtime": {"batch": 4, "dtype": "bf16", "gpu": "GPU A", "seconds": 5},
    }


def test_default_comparison_allows_inference_changes_for_migration(experiment):
    identity, previous = experiment
    require_comparable(
        previous,
        identity,
        inference_settings={"batch": 16, "dtype": "fp32", "length_bucketing": True},
    )


@pytest.mark.parametrize(
    "setting,value", [("batch", 8), ("dtype", "fp32"), ("length_bucketing", True)]
)
def test_strict_mode_rejects_changed_inference_controls(experiment, setting, value):
    identity, previous = experiment
    current = {**previous["inference"], setting: value}
    with pytest.raises(ValueError, match="inference settings mismatch"):
        require_comparable(previous, identity, strict_inference=True, inference_settings=current)


def test_strict_mode_compares_controls_without_runtime_measurements(experiment):
    identity, previous = experiment
    current = copy.deepcopy(previous["inference"])
    require_comparable(previous, identity, strict_inference=True, inference_settings=current)
    previous["runtime"] = {"gpu": "GPU B", "seconds": 17, "peak_allocated_gb": 9}
    require_comparable(previous, identity, strict_inference=True, inference_settings=current)


def test_strict_mode_accepts_legacy_recorded_batch_and_dtype(experiment):
    identity, previous = experiment
    del previous["inference"]
    require_comparable(
        previous,
        identity,
        strict_inference=True,
        inference_settings={"batch": 4, "dtype": "bf16"},
    )
    with pytest.raises(ValueError, match="inference settings mismatch"):
        require_comparable(
            previous,
            identity,
            strict_inference=True,
            inference_settings={"batch": 4, "dtype": "bf16", "length_bucketing": False},
        )


@pytest.mark.parametrize("settings", [None, {}, {"batch": 4}, {"batch": 4, "dtype": None}])
def test_strict_mode_requires_known_current_settings(experiment, settings):
    identity, previous = experiment
    with pytest.raises(ValueError, match="requires"):
        require_comparable(previous, identity, strict_inference=True, inference_settings=settings)


def test_strict_mode_does_not_fall_back_when_explicit_settings_are_missing(experiment):
    identity, previous = experiment
    previous["inference"] = {"dtype": "bf16"}
    with pytest.raises(ValueError, match="inference settings mismatch"):
        require_comparable(
            previous,
            identity,
            strict_inference=True,
            inference_settings={"batch": 4, "dtype": "bf16"},
        )
