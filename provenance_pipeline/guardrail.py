"""Guardrail: refuse to let contaminated data reach a model silently.

`assert_clean` and `filter_clean_epochs` are the two entry points a
downstream consumer (e.g. classifier_benchmark) calls right before
handing data to a model, turning "hopefully the pipeline caught it"
into an explicit, auditable check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from provenance_pipeline.tainted_array import TaintedArray, TaintLevel


class TaintViolationError(Exception):
    """Raised when a TaintedArray fails a TaintPolicy check."""


@dataclass(frozen=True)
class TaintPolicy:
    """Acceptance criteria for data about to enter a model or analysis.

    Parameters
    ----------
    max_taint_level : TaintLevel
        Highest taint level tolerated anywhere in the array.
    max_contaminated_fraction : float
        Highest fraction of CONTAMINATED samples tolerated before the
        whole array is rejected outright, even if `max_taint_level`
        would otherwise let SUSPECT/borderline arrays through.
    """

    max_taint_level: TaintLevel = TaintLevel.CLEAN
    max_contaminated_fraction: float = 0.0


def assert_clean(ta: TaintedArray, policy: TaintPolicy = TaintPolicy()) -> None:
    """Raise TaintViolationError if `ta` violates `policy`."""
    if ta.max_taint() > policy.max_taint_level:
        raise TaintViolationError(
            f"stage={ta.stage!r}: max taint {ta.max_taint().name} exceeds "
            f"policy limit {policy.max_taint_level.name}"
        )
    frac = ta.contaminated_fraction()
    if frac > policy.max_contaminated_fraction:
        raise TaintViolationError(
            f"stage={ta.stage!r}: {frac:.2%} contaminated samples exceeds "
            f"policy limit {policy.max_contaminated_fraction:.2%}"
        )


def filter_clean_epochs(
    ta: TaintedArray, policy: TaintPolicy = TaintPolicy(), epoch_axis: int = 0
) -> tuple[TaintedArray, np.ndarray]:
    """Keep only epochs that satisfy `policy`; return (kept, keep_mask).

    `ta.data` / `ta.taint` are expected to be epoched, i.e. `epoch_axis`
    indexes trials. Per-epoch severity is the worst taint level and the
    contaminated-sample fraction within that epoch.
    """
    other_axes = tuple(i for i in range(ta.taint.ndim) if i != epoch_axis)
    per_epoch_max = np.max(ta.taint, axis=other_axes)
    per_epoch_frac = np.mean(ta.taint >= TaintLevel.CONTAMINATED, axis=other_axes)
    keep_mask = (per_epoch_max <= policy.max_taint_level) & (
        per_epoch_frac <= policy.max_contaminated_fraction
    )
    keep_idx = np.flatnonzero(keep_mask)
    kept_data = np.take(ta.data, keep_idx, axis=epoch_axis)
    kept_taint = np.take(ta.taint, keep_idx, axis=epoch_axis)
    n_rejected = int((~keep_mask).sum())
    kept = ta._advance(
        kept_data,
        kept_taint,
        ta.stage,
        f"guardrail:filter_clean_epochs(kept={int(keep_mask.sum())}, rejected={n_rejected})",
    )
    return kept, keep_mask


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    n_epochs, n_channels, n_times = 20, 8, 100
    data = rng.standard_normal((n_epochs, n_channels, n_times))
    taint = np.zeros_like(data, dtype=np.uint8)

    contaminated_epochs = [3, 7, 15]
    for e in contaminated_epochs:
        taint[e, rng.integers(0, n_channels), :] = TaintLevel.CONTAMINATED

    epochs = TaintedArray(data, taint, stage="EPOCHED")
    print(epochs)

    policy = TaintPolicy(max_taint_level=TaintLevel.SUSPECT, max_contaminated_fraction=0.0)
    try:
        assert_clean(epochs, policy)
    except TaintViolationError as exc:
        print(f"assert_clean correctly rejected the batch: {exc}")

    clean_epochs, keep_mask = filter_clean_epochs(epochs, policy)
    rejected = np.flatnonzero(~keep_mask).tolist()
    print(f"kept {int(keep_mask.sum())}/{n_epochs} epochs; rejected indices: {rejected}")
    assert rejected == contaminated_epochs
    assert_clean(clean_epochs, policy)
    print("OK: guardrail rejects exactly the contaminated epochs and lets the rest through clean.")
