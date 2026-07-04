"""Generic synthetic data generators shared by the test suite's fixtures.

Deliberately generic (multichannel time series, optionally epoched,
optionally with an injected artifact) rather than a copy of any one
module's own domain-specific generator -- those already exist and are
tested where they live (e.g. `encoding_models.simulate_lnp`,
`dim_reduction_viz.simulate_population_trials`). This module is a
single place for the *test suite* to get arrays of a known shape and
known properties without every test file rolling its own.
"""
from __future__ import annotations

import numpy as np


def synthetic_multichannel_signal(
    n_channels: int = 8,
    n_times: int = 1000,
    seed: int | None = 0,
    artifact_channel: int | None = None,
    artifact_span: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian noise + shared 10 Hz oscillation, with an optional injected artifact.

    Returns (data, bad_mask) where `bad_mask` is a boolean array the
    same shape as `data`, True over the injected artifact region (all
    False if no artifact was requested).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_times)
    data = 0.5 * rng.standard_normal((n_channels, n_times))
    data += np.sin(2 * np.pi * t / 25.0)

    bad_mask = np.zeros_like(data, dtype=bool)
    if artifact_channel is not None:
        lo, hi = artifact_span if artifact_span is not None else (n_times // 2, n_times // 2 + 20)
        data[artifact_channel, lo:hi] += 10.0
        bad_mask[artifact_channel, lo:hi] = True
    return data, bad_mask


def synthetic_two_class_epochs(
    n_epochs: int = 40,
    n_channels: int = 8,
    n_times: int = 100,
    separation: float = 2.0,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Epoched two-class data with a controllable, class-dependent *power* signal.

    The class difference is an oscillation added to half the channels
    (variance-based), not a constant offset -- a DC shift wouldn't
    change any channel's variance, so power/covariance-based methods
    like CSP would be structurally blind to it. `separation` scales the
    oscillation amplitude; 0 gives pure noise (chance-level for any
    classifier), larger values give an easier task.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_epochs)
    X = rng.standard_normal((n_epochs, n_channels, n_times))
    half = n_channels // 2
    t = np.linspace(0.0, 1.0, n_times)
    oscillation = np.sin(2 * np.pi * 10 * t)
    for i in range(n_epochs):
        chans = slice(0, half) if y[i] == 0 else slice(half, n_channels)
        X[i, chans, :] += separation * oscillation
    return X.astype(np.float64), y.astype(np.int64)


if __name__ == "__main__":
    data, bad_mask = synthetic_multichannel_signal(artifact_channel=2, artifact_span=(100, 130))
    print(f"multichannel signal: {data.shape}, artifact samples flagged: {int(bad_mask.sum())}")
    assert bad_mask.sum() == 30
    assert not bad_mask[0].any(), "only the requested channel should be flagged"

    X, y = synthetic_two_class_epochs(n_epochs=40, separation=3.0, seed=0)
    print(f"two-class epochs: {X.shape}, label balance: {np.bincount(y)}")
    half = X.shape[1] // 2
    class0_power = X[y == 0, :half, :].var()
    class1_power = X[y == 1, :half, :].var()
    print(f"variance in the first half of channels: class0={class0_power:.2f}, class1={class1_power:.2f}")
    assert class0_power > class1_power, "class 0 should show elevated power in its own half of channels"
    print("OK: synthetic generators produce arrays of the documented shape with the intended structure.")
