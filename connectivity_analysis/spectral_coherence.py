"""Magnitude-squared coherence between channel pairs via Welch's method.

Uses `scipy.signal.coherence` directly -- Welch cross-spectral
estimation is a numerical primitive, not a modeling decision, so
there's nothing to gain by reimplementing FFT-based spectral averaging
by hand. The judgment calls (segment length, band definition, how to
summarize a full spectrum into a heatmap) are what's implemented here.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import coherence


def pairwise_coherence(
    data: np.ndarray, fs: float, nperseg: int = 256
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Coherence spectrum for every channel pair.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    dict mapping (i, j) with i < j to (freqs, coherence_values).
    """
    n_channels = data.shape[0]
    results: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            freqs, cxy = coherence(data[i], data[j], fs=fs, nperseg=nperseg)
            results[(i, j)] = (freqs, cxy)
    return results


def band_coherence_matrix(
    data: np.ndarray, fs: float, fmin: float, fmax: float, nperseg: int = 256
) -> np.ndarray:
    """Symmetric (n_channels, n_channels) matrix of mean coherence within [fmin, fmax] Hz.

    Diagonal is set to 1.0 (a channel is perfectly coherent with itself).
    """
    n_channels = data.shape[0]
    matrix = np.eye(n_channels)
    for (i, j), (freqs, cxy) in pairwise_coherence(data, fs, nperseg).items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        value = float(cxy[band_mask].mean())
        matrix[i, j] = matrix[j, i] = value
    return matrix


def plot_matrix_heatmap(
    matrix: np.ndarray,
    title: str = "Connectivity matrix",
    labels: list[str] | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Render an (n_channels, n_channels) matrix as an annotated heatmap.

    Works for coherence matrices, Granger F-stat matrices, or Granger
    p-value matrices alike -- any square channel-by-channel array.
    """
    import matplotlib.pyplot as plt

    n = matrix.shape[0]
    labels = labels if labels is not None else [f"ch{i}" for i in range(n)]
    fig, ax = plt.subplots(figsize=(1.2 * n + 2, 1.2 * n + 1))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            text = "-" if np.isnan(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fs, n_times = 250.0, 5000
    t = np.arange(n_times) / fs

    shared = np.sin(2 * np.pi * 10 * t)
    coupled_a = shared + 0.5 * rng.standard_normal(n_times)
    coupled_b = shared + 0.5 * rng.standard_normal(n_times)
    independent_a = rng.standard_normal(n_times)
    independent_b = rng.standard_normal(n_times)
    data = np.stack([coupled_a, coupled_b, independent_a, independent_b])

    freqs, cxy_coupled = coherence(data[0], data[1], fs=fs, nperseg=256)
    freqs, cxy_independent = coherence(data[2], data[3], fs=fs, nperseg=256)
    idx_10hz = int(np.argmin(np.abs(freqs - 10)))
    print(f"coherence at 10 Hz: shared-oscillation pair={cxy_coupled[idx_10hz]:.3f}, "
          f"independent-noise pair={cxy_independent[idx_10hz]:.3f}")
    assert cxy_coupled[idx_10hz] > 0.8, "channels sharing a 10 Hz source should show high coherence there"
    assert cxy_independent[idx_10hz] < 0.3, "independent noise channels should show low coherence"

    matrix = band_coherence_matrix(data, fs, fmin=8.0, fmax=12.0)
    print("alpha-band (8-12 Hz) coherence matrix:")
    print(np.round(matrix, 2))
    assert matrix[0, 1] > matrix[0, 2], "shared-source pair should out-cohere unrelated pairs in-band"

    fig = plot_matrix_heatmap(
        matrix,
        title="Alpha-band (8-12 Hz) coherence",
        labels=["coupled_a", "coupled_b", "indep_a", "indep_b"],
    )
    out_path = "/tmp/coherence_heatmap.png"
    fig.savefig(out_path, dpi=120)
    print(f"saved heatmap to {out_path}")
    print("OK: coherence correctly separates a shared-oscillation pair from independent noise.")
