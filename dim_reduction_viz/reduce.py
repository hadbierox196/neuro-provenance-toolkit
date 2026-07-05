"""Dimensionality reduction: PCA via SVD (no sklearn) and optional UMAP.

PCA is implemented directly from the SVD of the centered data rather
than calling a library `.fit()` -- there is no simpler "from scratch"
option that isn't just this. UMAP is a genuinely complex nonlinear
algorithm with no reasonable from-scratch reimplementation, so it's
used via the `umap-learn` package when available, and skipped with a
clear message (not a crash) when it isn't.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAResult:
    scores: np.ndarray  # (n_samples, n_components)
    components: np.ndarray  # (n_components, n_features)
    explained_variance_ratio: np.ndarray  # (n_components,)
    mean: np.ndarray  # (n_features,)


def pca_via_svd(data: np.ndarray, n_components: int | None = None) -> PCAResult:
    """PCA of `data` (n_samples, n_features) via SVD of the centered matrix."""
    mean = data.mean(axis=0)
    centered = data - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = len(s) if n_components is None else n_components

    explained_variance = (s**2) / (data.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    scores = u[:, :n_components] * s[:n_components]
    components = vt[:n_components]
    return PCAResult(
        scores=scores,
        components=components,
        explained_variance_ratio=explained_variance_ratio[:n_components],
        mean=mean,
    )


def umap_embedding(
    data: np.ndarray, n_components: int = 2, seed: int | None = None, **kwargs
) -> np.ndarray | None:
    """UMAP embedding if `umap-learn` is installed, else None (caller should fall back to PCA)."""
    try:
        import umap
    except ImportError:
        print("[umap_embedding] umap-learn is not installed; skipping UMAP, use PCA instead.")
        return None
    reducer = umap.UMAP(n_components=n_components, random_state=seed, **kwargs)
    return reducer.fit_transform(data)


def latent_recovery_r2(true_latent: np.ndarray, pca_scores: np.ndarray, n_components: int = 2) -> float:
    """R^2 of the best linear reconstruction of `true_latent` from the top PCA scores.

    PCA components are only identifiable up to rotation/reflection, so
    "recovered the latent" is checked via linear reconstructability,
    not a direct elementwise comparison.
    """
    x = pca_scores[:, :n_components]
    x_design = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(x_design, true_latent, rcond=None)
    predicted = x_design @ beta
    ss_res = np.sum((true_latent - predicted) ** 2)
    ss_tot = np.sum((true_latent - true_latent.mean(axis=0)) ** 2)
    return float(1 - ss_res / ss_tot)


if __name__ == "__main__":
    from dim_reduction_viz.population_simulation import simulate_population_trials

    trials = simulate_population_trials(n_trials=30, n_times=200, n_neurons=50, seed=0)
    flat_activity = trials.activity.reshape(-1, trials.activity.shape[-1])
    flat_latent = trials.latent.reshape(-1, 2)

    result = pca_via_svd(flat_activity, n_components=5)
    print(f"explained variance ratio (top 5 PCs): {np.round(result.explained_variance_ratio, 3)}")
    top2_variance = result.explained_variance_ratio[:2].sum()
    print(f"variance explained by top 2 PCs: {top2_variance:.3f}")
    assert top2_variance > 0.7, "top 2 PCs should capture most variance given a 2D latent"

    r2 = latent_recovery_r2(flat_latent, result.scores, n_components=2)
    print(f"R^2 of linear reconstruction of true latent from top-2 PC scores: {r2:.3f}")
    assert r2 > 0.85, "top-2 PCA scores should linearly reconstruct the true 2D latent well"

    embedding = umap_embedding(flat_activity, n_components=2, seed=0)
    if embedding is None:
        print("UMAP unavailable here; PCA result above stands on its own.")
    else:
        print(f"UMAP embedding shape: {embedding.shape}")
    print("OK: PCA recovers the low-dimensional latent structure from noisy population activity.")
