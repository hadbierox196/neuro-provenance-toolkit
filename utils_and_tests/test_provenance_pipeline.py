"""Tests for provenance_pipeline: TaintedArray, propagation rules, and the guardrail.

Includes a property-style test (`test_taint_never_heals`) that taint
can only stay the same or increase across an arbitrary sequence of the
module's own operations -- checked over many random operation
sequences rather than one hand-picked example.
"""
from __future__ import annotations

import numpy as np
import pytest

from provenance_pipeline import (
    TaintedArray,
    TaintLevel,
    TaintPolicy,
    TaintViolationError,
    assert_clean,
    filter_clean_epochs,
)
from provenance_pipeline.propagation import (
    apply_average_reference,
    apply_temporal_filter,
    wrap_opaque_transform,
)


def test_new_array_is_clean() -> None:
    ta = TaintedArray(np.zeros((4, 100)))
    assert ta.is_clean()
    assert ta.max_taint() == TaintLevel.CLEAN
    assert ta.contaminated_fraction() == 0.0


def test_mark_sets_exactly_the_requested_region() -> None:
    ta = TaintedArray(np.zeros((4, 100)))
    mask = np.zeros((4, 100), dtype=bool)
    mask[1, 10:20] = True
    dirty = ta.mark(mask, TaintLevel.CONTAMINATED, "test")
    assert dirty.taint[1, 10:20].min() == TaintLevel.CONTAMINATED
    assert dirty.taint[0].max() == TaintLevel.CLEAN
    assert dirty.taint[1, :10].max() == TaintLevel.CLEAN


def test_ufunc_preserves_taint(artifact_signal: tuple[np.ndarray, np.ndarray]) -> None:
    data, bad_mask = artifact_signal
    ta = TaintedArray(data).mark(bad_mask, TaintLevel.CONTAMINATED, "artifact")
    scaled = ta * 2.0 + 1.0
    assert scaled.max_taint() == TaintLevel.CONTAMINATED
    assert np.array_equal(scaled.taint > 0, ta.taint > 0)


def test_two_array_op_takes_max_taint() -> None:
    a = TaintedArray(np.ones((2, 5)))
    b = TaintedArray(np.ones((2, 5)))
    mask = np.zeros((2, 5), dtype=bool)
    mask[0, 0] = True
    a = a.mark(mask, TaintLevel.SUSPECT, "test")
    combined = a + b
    assert combined.taint[0, 0] == TaintLevel.SUSPECT
    assert combined.taint[1, 1] == TaintLevel.CLEAN


def test_reduction_upgrades_to_worst_case(artifact_signal: tuple[np.ndarray, np.ndarray]) -> None:
    data, bad_mask = artifact_signal
    ta = TaintedArray(data).mark(bad_mask, TaintLevel.CONTAMINATED, "artifact")
    channel_mean = np.mean(ta, axis=1)
    bad_channel = int(np.flatnonzero(bad_mask.any(axis=1))[0])
    assert channel_mean.taint[bad_channel] == TaintLevel.CONTAMINATED
    for ch in range(data.shape[0]):
        if ch != bad_channel:
            assert channel_mean.taint[ch] == TaintLevel.CLEAN


def test_temporal_filter_smears_within_kernel_width() -> None:
    ta = TaintedArray(np.random.default_rng(0).standard_normal((2, 200)))
    mask = np.zeros((2, 200), dtype=bool)
    mask[0, 100] = True
    ta = ta.mark(mask, TaintLevel.CONTAMINATED, "glitch")

    def identity_filter(x: np.ndarray) -> np.ndarray:
        return x.copy()

    filtered = apply_temporal_filter(ta, identity_filter, kernel_width=9)
    n_tainted = int((filtered.taint[0] > 0).sum())
    assert n_tainted == 9
    assert filtered.taint[1].max() == TaintLevel.CLEAN


def test_average_reference_spreads_taint_across_channels() -> None:
    ta = TaintedArray(np.random.default_rng(0).standard_normal((4, 50)))
    mask = np.zeros((4, 50), dtype=bool)
    mask[2, 10] = True
    ta = ta.mark(mask, TaintLevel.CONTAMINATED, "bad channel")
    referenced = apply_average_reference(ta)
    assert referenced.taint[:, 10].min() == TaintLevel.CONTAMINATED
    assert referenced.taint[:, 0].max() == TaintLevel.CLEAN


def test_opaque_transform_conservative_policy_taints_everything() -> None:
    ta = TaintedArray(np.random.default_rng(0).standard_normal((3, 50)))
    mask = np.zeros((3, 50), dtype=bool)
    mask[0, 0] = True
    ta = ta.mark(mask, TaintLevel.CONTAMINATED, "test")
    black_box = wrap_opaque_transform(lambda x: x @ np.random.default_rng(1).standard_normal((50, 10)))
    out = black_box(ta)
    assert out.is_clean() is False
    assert (out.taint == TaintLevel.CONTAMINATED).all()


def test_assert_clean_raises_on_violation() -> None:
    ta = TaintedArray(np.zeros((2, 10)))
    mask = np.zeros((2, 10), dtype=bool)
    mask[0, 0] = True
    dirty = ta.mark(mask, TaintLevel.CONTAMINATED, "test")
    with pytest.raises(TaintViolationError):
        assert_clean(dirty, TaintPolicy(max_taint_level=TaintLevel.CLEAN))
    assert_clean(ta, TaintPolicy(max_taint_level=TaintLevel.CLEAN))  # should not raise


def test_filter_clean_epochs_removes_exactly_the_bad_ones() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((10, 4, 20))
    taint = np.zeros_like(data, dtype=np.uint8)
    taint[3, 0, :] = TaintLevel.CONTAMINATED
    taint[7, 1, :] = TaintLevel.CONTAMINATED
    ta = TaintedArray(data, taint, stage="EPOCHED")

    kept, keep_mask = filter_clean_epochs(ta, TaintPolicy(max_taint_level=TaintLevel.CLEAN))
    assert np.flatnonzero(~keep_mask).tolist() == [3, 7]
    assert kept.data.shape[0] == 8


def test_taint_never_heals() -> None:
    """Property test: across random operation sequences, taint per-sample never decreases."""
    n_trials = 30

    def random_op(ta: TaintedArray, trial_rng: np.random.Generator) -> TaintedArray:
        choice = trial_rng.integers(0, 4)
        if choice == 0:
            return ta * trial_rng.uniform(0.5, 2.0)
        if choice == 1:
            return ta + trial_rng.standard_normal()
        if choice == 2:
            return apply_average_reference(ta)
        mask = trial_rng.random(ta.shape) < 0.05
        return ta.mark(mask, TaintLevel.SUSPECT, "random spot check")

    for trial in range(n_trials):
        trial_rng = np.random.default_rng(trial)
        data = trial_rng.standard_normal((4, 60))
        taint = (trial_rng.random((4, 60)) < 0.1).astype(np.uint8) * TaintLevel.CONTAMINATED
        ta = TaintedArray(data, taint)

        for _ in range(5):
            previous_taint = ta.taint.copy()
            ta = random_op(ta, trial_rng)
            assert (ta.taint >= previous_taint).all(), (
                f"trial {trial}: taint decreased somewhere -- provenance must never heal"
            )
