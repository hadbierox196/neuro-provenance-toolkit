"""Linear-Nonlinear-Poisson forward model: ground-truth kernel -> simulated spikes.

Generates a stimulus, a known temporal kernel, and Poisson spike counts
from the canonical exponential-nonlinearity LNP model -- the same
generative model `glm_fit.fit_poisson_glm` assumes, so kernel recovery
can be checked against ground truth rather than just "the fit
converged".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_biphasic_kernel(
    length: int = 25,
    peak_lag: float = 4.0,
    undershoot_lag: float = 10.0,
    peak_width: float = 2.5,
    undershoot_width: float = 4.0,
    undershoot_amplitude: float = 0.4,
) -> np.ndarray:
    """A biphasic temporal receptive field: positive lobe then a negative undershoot."""
    t = np.arange(length)
    pos_lobe = np.exp(-0.5 * ((t - peak_lag) / peak_width) ** 2)
    neg_lobe = np.exp(-0.5 * ((t - undershoot_lag) / undershoot_width) ** 2)
    kernel = pos_lobe - undershoot_amplitude * neg_lobe
    return kernel / np.linalg.norm(kernel)


def build_design_matrix(stimulus: np.ndarray, kernel_length: int) -> np.ndarray:
    """Row n holds [stimulus[n], stimulus[n-1], ..., stimulus[n-kernel_length+1]].

    Zero-padded at the start, so every row of the stimulus has a full
    history even near time 0.
    """
    n = len(stimulus)
    padded = np.concatenate([np.zeros(kernel_length - 1), stimulus])
    cols = [padded[kernel_length - 1 - k : kernel_length - 1 - k + n] for k in range(kernel_length)]
    return np.column_stack(cols)


@dataclass(frozen=True)
class LNPSimulation:
    """Ground truth + simulated output of one LNP run."""

    stimulus: np.ndarray
    kernel: np.ndarray
    bias: float
    rate: np.ndarray
    spike_counts: np.ndarray


def simulate_lnp(
    n_samples: int = 20000,
    kernel: np.ndarray | None = None,
    bias: float = -2.0,
    seed: int | None = None,
) -> LNPSimulation:
    """Simulate an LNP neuron: white-noise stimulus -> linear filter -> exp -> Poisson spikes.

    `bias` sets the baseline log-rate; with a unit-norm kernel and a
    standard-normal stimulus the linear term has variance ~1, so
    bias=-2.0 gives a mean rate low enough to avoid saturation while
    still producing thousands of spikes over 20000 samples.
    """
    rng = np.random.default_rng(seed)
    if kernel is None:
        kernel = make_biphasic_kernel()
    stimulus = rng.standard_normal(n_samples)
    X = build_design_matrix(stimulus, len(kernel))
    linear = X @ kernel + bias
    rate = np.exp(np.clip(linear, -20, 20))
    spike_counts = rng.poisson(rate)
    return LNPSimulation(stimulus=stimulus, kernel=kernel, bias=bias, rate=rate, spike_counts=spike_counts)


if __name__ == "__main__":
    sim = simulate_lnp(n_samples=20000, seed=0)
    print(f"stimulus: {sim.stimulus.shape}, kernel length: {len(sim.kernel)}")
    print(f"mean rate: {sim.rate.mean():.3f} spikes/bin, total spikes: {int(sim.spike_counts.sum())}")
    print(f"spike count histogram (capped at 4+): {np.bincount(np.minimum(sim.spike_counts, 4))}")
    assert sim.spike_counts.sum() > 1000, "need enough spikes for reliable downstream fitting"
    print("OK: LNP forward model produces a reasonable spike count from a known kernel.")
