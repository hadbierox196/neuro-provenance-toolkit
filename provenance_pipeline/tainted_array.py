"""Core TaintedArray wrapper: NumPy-native contamination tracking.

TaintedArray pairs a NumPy array with a same-shaped provenance mask so
that contamination introduced at any pipeline stage cannot be silently
lost by downstream arithmetic, reductions, or opaque transforms. It
implements NumPy's ``__array_ufunc__`` / ``__array_function__`` dispatch
protocols (via ``numpy.lib.mixins.NDArrayOperatorsMixin``) so ordinary
NumPy code (``a + b``, ``np.mean(a)``, ...) propagates taint
automatically, without callers having to remember to carry the mask by
hand.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class TaintLevel(IntEnum):
    """Ordered contamination severity. Higher always wins on combination."""

    CLEAN = 0
    SUSPECT = 1
    CONTAMINATED = 2


@dataclass(frozen=True)
class ProvenanceRecord:
    """One entry in a TaintedArray's audit trail."""

    stage: str
    operation: str
    taint_in: int
    taint_out: int


HANDLED_FUNCTIONS: dict[Callable[..., Any], Callable[..., Any]] = {}


def _implements(np_function: Callable[..., Any]) -> Callable[..., Any]:
    """Register a TaintedArray-aware override for a NumPy free function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class TaintedArray(np.lib.mixins.NDArrayOperatorsMixin):
    """A NumPy array shadowed by a same-shaped provenance/taint mask.

    Parameters
    ----------
    data : np.ndarray
        The underlying numeric payload (e.g. channels x times, or
        epochs x channels x times).
    taint : np.ndarray, optional
        Integer array (values from `TaintLevel`) with the same shape as
        `data`. Defaults to all-CLEAN.
    stage : str, default "RAW"
        Pipeline stage tag, e.g. "RAW", "FILTERED", "ICA_CLEANED",
        "EPOCHED".
    provenance : list[ProvenanceRecord], optional
        Audit trail accumulated so far.
    """

    def __init__(
        self,
        data: np.ndarray,
        taint: np.ndarray | None = None,
        stage: str = "RAW",
        provenance: list[ProvenanceRecord] | None = None,
    ) -> None:
        self.data = np.asarray(data)
        if taint is None:
            taint = np.zeros(self.data.shape, dtype=np.uint8)
        taint = np.asarray(taint, dtype=np.uint8)
        if taint.shape != self.data.shape:
            raise ValueError(
                f"taint shape {taint.shape} must match data shape {self.data.shape}"
            )
        self.taint = taint
        self.stage = stage
        self.provenance: list[ProvenanceRecord] = list(provenance) if provenance else []

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def max_taint(self) -> TaintLevel:
        return TaintLevel(int(self.taint.max())) if self.taint.size else TaintLevel.CLEAN

    def is_clean(self) -> bool:
        return self.max_taint() == TaintLevel.CLEAN

    def contaminated_fraction(self) -> float:
        if self.taint.size == 0:
            return 0.0
        return float(np.mean(self.taint >= TaintLevel.CONTAMINATED))

    def mark(self, mask: np.ndarray, level: TaintLevel, operation: str) -> TaintedArray:
        """Return a copy with `level` taint applied wherever `mask` is True."""
        mask = np.broadcast_to(mask, self.data.shape)
        new_taint = self.taint.copy()
        new_taint[mask] = np.maximum(new_taint[mask], np.uint8(level))
        return self._advance(self.data, new_taint, self.stage, operation)

    def advance_stage(self, stage: str) -> TaintedArray:
        """Return a copy tagged with a new pipeline stage (e.g. after epoching)."""
        return self._advance(self.data, self.taint, stage, f"stage->{stage}")

    def _advance(
        self, data: np.ndarray, taint: np.ndarray, stage: str, operation: str
    ) -> TaintedArray:
        record = ProvenanceRecord(
            stage=stage,
            operation=operation,
            taint_in=int(self.taint.max()) if self.taint.size else 0,
            taint_out=int(taint.max()) if taint.size else 0,
        )
        return TaintedArray(data, taint, stage, self.provenance + [record])

    def __repr__(self) -> str:
        return (
            f"TaintedArray(shape={self.data.shape}, stage={self.stage!r}, "
            f"max_taint={self.max_taint().name}, "
            f"contaminated_frac={self.contaminated_fraction():.2%})"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: Any) -> TaintedArray:
        return TaintedArray(self.data[key], self.taint[key], self.stage, self.provenance)

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        warnings.warn(
            "Converting TaintedArray to a plain ndarray drops its taint mask; "
            "prefer `.data` with an explicit taint check, or wrap_opaque_transform().",
            stacklevel=2,
        )
        return self.data.astype(dtype) if dtype is not None else self.data

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        if method != "__call__":
            return NotImplemented
        raw_inputs = []
        taint_masks = []
        for inp in inputs:
            if isinstance(inp, TaintedArray):
                raw_inputs.append(inp.data)
                taint_masks.append(inp.taint)
            else:
                raw_inputs.append(inp)
        out_data = getattr(ufunc, method)(*raw_inputs, **kwargs)
        if isinstance(out_data, tuple):
            return NotImplemented
        combined = taint_masks[0]
        for t in taint_masks[1:]:
            combined = np.maximum(combined, t)
        combined = np.broadcast_to(combined, out_data.shape).astype(np.uint8).copy()
        record = ProvenanceRecord(
            stage=self.stage,
            operation=f"ufunc:{ufunc.__name__}",
            taint_in=int(max(t.max() for t in taint_masks)) if taint_masks else 0,
            taint_out=int(combined.max()) if combined.size else 0,
        )
        return TaintedArray(out_data, combined, self.stage, self.provenance + [record])

    def __array_function__(
        self, func: Callable[..., Any], types: tuple, args: tuple, kwargs: dict
    ) -> Any:
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def _as_tainted(x: TaintedArray | np.ndarray) -> TaintedArray:
    return x if isinstance(x, TaintedArray) else TaintedArray(np.asarray(x))


@_implements(np.mean)
def _mean(
    a: TaintedArray, axis: int | tuple[int, ...] | None = None, **kwargs: Any
) -> TaintedArray:
    a = _as_tainted(a)
    out_data = np.atleast_1d(np.mean(a.data, axis=axis, **kwargs))
    out_taint = np.atleast_1d(np.max(a.taint, axis=axis)).astype(np.uint8)
    return a._advance(out_data, out_taint, a.stage, "np.mean")


@_implements(np.sum)
def _sum(
    a: TaintedArray, axis: int | tuple[int, ...] | None = None, **kwargs: Any
) -> TaintedArray:
    a = _as_tainted(a)
    out_data = np.atleast_1d(np.sum(a.data, axis=axis, **kwargs))
    out_taint = np.atleast_1d(np.max(a.taint, axis=axis)).astype(np.uint8)
    return a._advance(out_data, out_taint, a.stage, "np.sum")


@_implements(np.std)
def _std(
    a: TaintedArray, axis: int | tuple[int, ...] | None = None, **kwargs: Any
) -> TaintedArray:
    a = _as_tainted(a)
    out_data = np.atleast_1d(np.std(a.data, axis=axis, **kwargs))
    out_taint = np.atleast_1d(np.max(a.taint, axis=axis)).astype(np.uint8)
    return a._advance(out_data, out_taint, a.stage, "np.std")


@_implements(np.concatenate)
def _concatenate(arrays: Any, axis: int = 0, **kwargs: Any) -> TaintedArray:
    tainted = [_as_tainted(a) for a in arrays]
    out_data = np.concatenate([a.data for a in tainted], axis=axis, **kwargs)
    out_taint = np.concatenate([a.taint for a in tainted], axis=axis, **kwargs)
    return TaintedArray(out_data, out_taint, tainted[0].stage, tainted[0].provenance)


@_implements(np.stack)
def _stack(arrays: Any, axis: int = 0, **kwargs: Any) -> TaintedArray:
    tainted = [_as_tainted(a) for a in arrays]
    out_data = np.stack([a.data for a in tainted], axis=axis, **kwargs)
    out_taint = np.stack([a.taint for a in tainted], axis=axis, **kwargs)
    return TaintedArray(out_data, out_taint, tainted[0].stage, tainted[0].provenance)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    clean = TaintedArray(rng.standard_normal((4, 100)), stage="RAW")
    print("clean:", clean)

    bad_mask = np.zeros((4, 100), dtype=bool)
    bad_mask[2, 40:60] = True  # channel 2, samples 40-59 contaminated
    dirty = clean.mark(bad_mask, TaintLevel.CONTAMINATED, "manual bad-segment annotation")
    print("dirty:", dirty)

    scaled = dirty * 2.0 + 1.0  # ordinary NumPy arithmetic
    print("scaled (taint preserved through ufuncs):", scaled)
    assert scaled.max_taint() == TaintLevel.CONTAMINATED

    channel_mean = np.mean(dirty, axis=1)
    print("per-channel mean taint:", channel_mean.taint)
    assert channel_mean.taint[2] == TaintLevel.CONTAMINATED
    assert channel_mean.taint[0] == TaintLevel.CLEAN
    print("OK: taint survives arithmetic and reduction, isolated to channel 2.")
