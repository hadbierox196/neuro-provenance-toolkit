"""Spike-triggered average: the classical (pre-GLM) receptive field estimator.

STA is the spike-count-weighted mean stimulus history preceding each
bin. For a white-noise stimulus, STA is proportional to the true
kernel's *direction* even when the nonlinearity is unknown -- but not
necessarily its exact scale, since STA has no notion of the
nonlinearity's shape (Chichilnisky, 2001). It is therefore compared to
ground truth by normalized (cosine) similarity here, while
`fit_poisson_glm` -- which assumes the correct exponential nonlinearity
-- is expected to also recover the exact scale.
"""
from __future__ import annotations

import numpy as np

from encoding_models.lnp_simulation import build_design_matrix


def spike_triggered_average(stimulus: np.ndarray, spike_counts: np.ndarray, kernel_length: int) -> np.ndarray:
    """Spike-count-weighted average of the stimulus history preceding each bin."""
    X = build_design_matrix(stimulus, kernel_length)
    total_spikes = spike_counts.sum()
    if total_spikes == 0:
        raise ValueError("no spikes to average")
    return (X * spike_counts[:, None]).sum(axis=0) / total_spikes


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity: a shape-only comparison, insensitive to overall scale."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def plot_kernel_comparison(
    kernel_true: np.ndarray, kernel_glm: np.ndarray | None = None, kernel_sta: np.ndarray | None = None
):
    """Overlay the ground-truth kernel against GLM and/or STA estimates."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    lags = np.arange(len(kernel_true))
    ax.plot(lags, kernel_true, label="ground truth", color="black", linewidth=2)
    if kernel_glm is not None:
        ax.plot(lags, kernel_glm, label="GLM fit", color="#1f77b4", linestyle="--")
    if kernel_sta is not None:
        ax.plot(lags, kernel_sta / np.linalg.norm(kernel_sta), label="STA (unit-normalized)",
                 color="#d62728", linestyle=":")
    ax.set_xlabel("lag (bins)")
    ax.set_ylabel("kernel weight")
    ax.set_title("Receptive field kernel: ground truth vs. estimates")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_psth_comparison(spike_counts: np.ndarray, predicted_rate: np.ndarray, n_bins: int = 200):
    """Bin observed spike counts and overlay against a model-predicted rate trace."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3))
    n = len(spike_counts)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    binned_obs = np.array([spike_counts[edges[i]:edges[i + 1]].mean() for i in range(n_bins)])
    binned_pred = np.array([predicted_rate[edges[i]:edges[i + 1]].mean() for i in range(n_bins)])
    x = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(x, binned_obs, label="observed (binned)", color="black", alpha=0.7)
    ax.plot(x, binned_pred, label="GLM predicted rate", color="#1f77b4")
    ax.set_xlabel("time bin")
    ax.set_ylabel("rate (spikes/bin)")
    ax.set_title("Predicted vs. observed firing rate (PSTH-style)")
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from encoding_models.glm_fit import fit_poisson_glm
    from encoding_models.lnp_simulation import build_design_matrix, simulate_lnp

    sim = simulate_lnp(n_samples=20000, seed=0)
    sta = spike_triggered_average(sim.stimulus, sim.spike_counts, len(sim.kernel))
    sta_normalized = sta / np.linalg.norm(sta)

    shape_similarity = normalized_correlation(sta_normalized, sim.kernel)
    print(f"STA vs ground-truth kernel, shape similarity (cosine): {shape_similarity:.3f}")
    print(f"STA raw norm: {np.linalg.norm(sta):.4f} vs. kernel norm: {np.linalg.norm(sim.kernel):.4f}")
    print("  (STA recovers shape, not the nonlinearity-dependent scale -- the mismatch above is expected)")
    assert shape_similarity > 0.85, "STA should recover the correct kernel shape on white-noise stimuli"
    print("OK: STA baseline recovers the true kernel's shape from white-noise stimulus/spike data.")

    kernel_hat, bias_hat, _ = fit_poisson_glm(sim.stimulus, sim.spike_counts, len(sim.kernel), l2_lambda=2.0)
    fig_kernel = plot_kernel_comparison(sim.kernel, kernel_glm=kernel_hat, kernel_sta=sta)
    fig_kernel.savefig("/tmp/kernel_comparison.png", dpi=120)
    print("saved kernel comparison plot to /tmp/kernel_comparison.png")

    X = build_design_matrix(sim.stimulus, len(sim.kernel))
    predicted_rate = np.exp(np.clip(X @ kernel_hat + bias_hat, -20, 20))
    fig_psth = plot_psth_comparison(sim.spike_counts, predicted_rate, n_bins=200)
    fig_psth.savefig("/tmp/psth_comparison.png", dpi=120)
    print("saved predicted-vs-observed rate plot to /tmp/psth_comparison.png")
