"""Transform-specific taint propagation rules.

Each function here wraps a real signal-processing or model step (a
temporal filter, an average reference, an arbitrary black-box callable)
and returns a new TaintedArray whose mask reflects how that specific
kind of transform can spread contamination -- not just the generic
elementwise rule that TaintedArray's NumPy-protocol hooks already give
you for free.
"""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from scipy.ndimage import maximum_filter1d

from provenance_pipeline.tainted_array import ProvenanceRecord, TaintedArray, TaintLevel


def apply_temporal_filter(
    ta: TaintedArray,
    filter_fn: Callable[[np.ndarray], np.ndarray],
    kernel_width: int,
    stage: str = "FILTERED",
    op_name: str = "temporal_filter",
) -> TaintedArray:
    """Apply a temporal filter and smear taint within +/- kernel_width//2 samples.

    A FIR/IIR filter's output at time t depends on a window of input
    samples around t, so a single contaminated input sample can taint a
    whole neighborhood of output samples even when the filter itself is
    numerically well-behaved.
    """
    if kernel_width < 1:
        raise ValueError("kernel_width must be >= 1")
    out_data = filter_fn(ta.data)
    if out_data.shape != ta.data.shape:
        raise ValueError("filter_fn must preserve array shape for taint alignment")
    smeared = maximum_filter1d(ta.taint, size=kernel_width, axis=-1, mode="nearest")
    return ta._advance(
        out_data, smeared.astype(np.uint8), stage, f"{op_name}(kernel={kernel_width})"
    )


def apply_average_reference(
    ta: TaintedArray,
    channel_axis: int = 0,
    stage: str = "FILTERED",
    op_name: str = "average_reference",
) -> TaintedArray:
    """Subtract the across-channel mean, the way MNE's average reference does.

    Because the reference is computed from *all* channels, a single
    contaminated channel contaminates the shared reference, and
    subtracting that reference then taints every other channel too.
    """
    ref = np.mean(ta.data, axis=channel_axis, keepdims=True)
    out_data = ta.data - ref
    ref_taint = np.max(ta.taint, axis=channel_axis, keepdims=True)
    out_taint = np.maximum(ta.taint, ref_taint)
    return ta._advance(out_data, out_taint.astype(np.uint8), stage, op_name)


def wrap_opaque_transform(
    fn: Callable[[np.ndarray], np.ndarray],
    policy: Literal["conservative", "channelwise"] = "conservative",
    channel_axis: int = 0,
    stage: str = "MODEL_OUTPUT",
    op_name: str = "opaque_transform",
) -> Callable[[TaintedArray], TaintedArray]:
    """Wrap an arbitrary black-box array->array callable so taint still propagates.

    `fn` (e.g. a PyTorch model's forward pass) is called on the raw
    ndarray with no knowledge of taint whatsoever. Because we cannot see
    inside `fn`, `policy="conservative"` assumes the worst: if *any*
    input sample anywhere is tainted, the entire output is marked with
    that same taint level, since the model could in principle route
    information from any input to any output (attention, global
    pooling, dense layers, ...). `policy="channelwise"` is a looser
    option for callers who know `fn` only mixes information within a
    channel, and propagates taint only along matching channel indices.
    """

    def wrapped(ta: TaintedArray) -> TaintedArray:
        out_data = np.asarray(fn(ta.data))
        if policy == "conservative":
            worst = ta.max_taint()
            out_taint = np.full(out_data.shape, fill_value=int(worst), dtype=np.uint8)
        elif policy == "channelwise":
            if out_data.shape[channel_axis] != ta.data.shape[channel_axis]:
                raise ValueError("channelwise policy requires fn to preserve the channel axis")
            reduce_axes = tuple(i for i in range(ta.taint.ndim) if i != channel_axis)
            per_channel_worst = np.max(ta.taint, axis=reduce_axes)
            shape = [1] * out_data.ndim
            shape[channel_axis] = -1
            out_taint = np.broadcast_to(
                per_channel_worst.reshape(shape), out_data.shape
            ).astype(np.uint8).copy()
        else:
            raise ValueError(f"unknown policy: {policy!r}")
        record = ProvenanceRecord(
            stage=stage,
            operation=f"{op_name}[{policy}]",
            taint_in=int(ta.taint.max()) if ta.taint.size else 0,
            taint_out=int(out_taint.max()) if out_taint.size else 0,
        )
        return TaintedArray(out_data, out_taint, stage, ta.provenance + [record])

    return wrapped


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    raw = TaintedArray(rng.standard_normal((3, 200)), stage="RAW")
    bad = np.zeros((3, 200), dtype=bool)
    bad[1, 100] = True  # a single bad sample on channel 1
    raw = raw.mark(bad, TaintLevel.CONTAMINATED, "single-sample glitch")

    def moving_average_stub(x: np.ndarray) -> np.ndarray:
        kernel = np.ones(9) / 9
        return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), -1, x)

    filtered = apply_temporal_filter(raw, moving_average_stub, kernel_width=9)
    n_tainted = int((filtered.taint[1] > 0).sum())
    print(f"filtering smeared 1 contaminated sample to {n_tainted} tainted samples on channel 1")
    assert filtered.taint[0].max() == TaintLevel.CLEAN

    referenced = apply_average_reference(filtered)
    print("post-reference max taint per channel:", referenced.taint.max(axis=1))
    assert referenced.taint[0].max() == TaintLevel.CONTAMINATED, "avg-ref should spread taint to channel 0"

    black_box = wrap_opaque_transform(lambda x: x @ rng.standard_normal((200, 50)), policy="conservative")
    model_out = black_box(referenced)
    print("opaque-model output:", model_out)
    assert model_out.max_taint() == TaintLevel.CONTAMINATED, "contamination must survive an opaque transform"
    print("OK: a single bad sample survives filtering, referencing, and an opaque model, still flagged.")
