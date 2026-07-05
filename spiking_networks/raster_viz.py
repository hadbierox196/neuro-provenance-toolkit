"""Visualization helpers for spiking decision-network output.

Pure NumPy + Matplotlib -- no Brian2 or PyTorch dependency -- so these
work on spikes from either `lif_decision_network` or
`surrogate_gradient_snn`, or on any (times, indices) spike data.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_raster(
    spike_t_a: np.ndarray,
    spike_i_a: np.ndarray,
    spike_t_b: np.ndarray,
    spike_i_b: np.ndarray,
    ax: Axes | None = None,
    title: str = "Spike raster",
) -> Axes:
    """Scatter raster with pool A below pool B, offset so both are visible."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(spike_t_a, spike_i_a, s=4, color="#1f77b4", label="pool A")
    n_a = int(spike_i_a.max()) + 1 if spike_i_a.size else 0
    ax.scatter(spike_t_b, spike_i_b + n_a + 2, s=4, color="#d62728", label="pool B")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron index")
    ax.set_title(title)
    ax.legend(loc="upper right", markerscale=3, fontsize=8)
    return ax


def plot_population_rates(
    rate_t_ms: np.ndarray,
    rate_a_hz: np.ndarray,
    rate_b_hz: np.ndarray,
    decision_threshold_hz: float | None = None,
    decision_time_ms: float | None = None,
    ax: Axes | None = None,
    title: str = "Population firing rates",
) -> Axes:
    """Smoothed population rate traces for both pools, with optional decision markers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    ax.plot(rate_t_ms, rate_a_hz, color="#1f77b4", label="pool A")
    ax.plot(rate_t_ms, rate_b_hz, color="#d62728", label="pool B")
    if decision_threshold_hz is not None:
        ax.axhline(
            decision_threshold_hz, color="gray", linestyle="--", linewidth=1, label="decision threshold"
        )
    if decision_time_ms is not None:
        ax.axvline(decision_time_ms, color="black", linestyle=":", linewidth=1, label="decision")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("rate (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_psychometric(
    coherences: np.ndarray,
    accuracies: np.ndarray,
    ax: Axes | None = None,
    title: str = "Psychometric function",
) -> Axes:
    """Accuracy (P correct) vs. coherence."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(coherences, accuracies, marker="o", color="#2ca02c")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("coherence")
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    return ax


def plot_chronometric(
    coherences: np.ndarray,
    mean_decision_times_ms: np.ndarray,
    ax: Axes | None = None,
    title: str = "Chronometric function",
) -> Axes:
    """Mean decision time vs. coherence."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(coherences, mean_decision_times_ms, marker="o", color="#9467bd")
    ax.set_xlabel("coherence")
    ax.set_ylabel("mean decision time (ms)")
    ax.set_title(title)
    return ax


def summary_figure(
    spike_t_a: np.ndarray,
    spike_i_a: np.ndarray,
    spike_t_b: np.ndarray,
    spike_i_b: np.ndarray,
    rate_t_ms: np.ndarray,
    rate_a_hz: np.ndarray,
    rate_b_hz: np.ndarray,
    coherences: np.ndarray,
    accuracies: np.ndarray,
    mean_decision_times_ms: np.ndarray,
    decision_threshold_hz: float | None = None,
    decision_time_ms: float | None = None,
) -> Figure:
    """Combine raster + rates + psychometric + chronometric into one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_raster(spike_t_a, spike_i_a, spike_t_b, spike_i_b, ax=axes[0, 0])
    plot_population_rates(
        rate_t_ms, rate_a_hz, rate_b_hz, decision_threshold_hz, decision_time_ms, ax=axes[0, 1]
    )
    plot_psychometric(coherences, accuracies, ax=axes[1, 0])
    plot_chronometric(coherences, mean_decision_times_ms, ax=axes[1, 1])
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_per_pool, duration_ms = 30, 500.0

    def _poisson_spikes(rate_hz: float, n: int, duration_ms: float) -> tuple[np.ndarray, np.ndarray]:
        expected = rate_hz * (duration_ms / 1000.0) * n
        n_spikes = rng.poisson(expected)
        return rng.uniform(0, duration_ms, n_spikes), rng.integers(0, n, n_spikes)

    t_a, i_a = _poisson_spikes(45.0, n_per_pool, duration_ms)
    t_b, i_b = _poisson_spikes(15.0, n_per_pool, duration_ms)

    rate_t = np.linspace(0, duration_ms, 200)
    rate_a = 45.0 + 5 * rng.standard_normal(200)
    rate_b = 15.0 + 5 * rng.standard_normal(200)

    coherences = np.array([0.0, 0.15, 0.3, 0.5, 0.75, 1.0])
    accuracies = np.array([0.5, 0.65, 0.78, 0.9, 0.95, 0.97])
    rts = np.array([650, 500, 420, 350, 300, 280])

    fig = summary_figure(
        t_a, i_a, t_b, i_b, rate_t, rate_a, rate_b,
        coherences, accuracies, rts,
        decision_threshold_hz=55.0, decision_time_ms=320.0,
    )
    out_path = "/tmp/raster_viz_demo.png"
    fig.savefig(out_path, dpi=120)
    print(f"saved summary figure with synthetic demo data to {out_path}")
    assert len(t_a) > 0 and len(t_b) > 0
    print("OK: all four plot panels render without error on synthetic spike/behavioral data.")
