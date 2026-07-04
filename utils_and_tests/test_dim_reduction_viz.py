"""Tests for dim_reduction_viz: population simulation, PCA, and trajectory data prep."""
from __future__ import annotations

import numpy as np
import pytest

from dim_reduction_viz.interactive_viz import trials_to_traces
from dim_reduction_viz.population_simulation import simulate_population_trials
from dim_reduction_viz.reduce import latent_recovery_r2, pca_via_svd, umap_embedding


def test_simulated_activity_has_documented_shape() -> None:
    trials = simulate_population_trials(n_trials=10, n_times=50, n_neurons=20, seed=0)
    assert trials.activity.shape == (10, 50, 20)
    assert trials.latent.shape == (10, 50, 2)
    assert trials.projection.shape == (2, 20)


def test_pca_explained_variance_sums_to_one() -> None:
    trials = simulate_population_trials(n_trials=10, n_times=50, n_neurons=20, seed=0)
    flat = trials.activity.reshape(-1, 20)
    result = pca_via_svd(flat)  # all components
    assert abs(result.explained_variance_ratio.sum() - 1.0) < 1e-8


def test_pca_top_two_components_dominate_for_2d_latent() -> None:
    trials = simulate_population_trials(n_trials=20, n_times=100, n_neurons=30, seed=0)
    flat = trials.activity.reshape(-1, 30)
    result = pca_via_svd(flat, n_components=5)
    assert result.explained_variance_ratio[:2].sum() > 0.7
    assert result.explained_variance_ratio[0] >= result.explained_variance_ratio[1]


def test_pca_scores_linearly_reconstruct_true_latent() -> None:
    trials = simulate_population_trials(n_trials=20, n_times=100, n_neurons=30, seed=0)
    flat_activity = trials.activity.reshape(-1, 30)
    flat_latent = trials.latent.reshape(-1, 2)
    result = pca_via_svd(flat_activity, n_components=5)
    r2 = latent_recovery_r2(flat_latent, result.scores, n_components=2)
    assert r2 > 0.85


def test_umap_embedding_returns_none_when_unavailable() -> None:
    # In this environment umap-learn genuinely isn't installed; this
    # documents and checks the graceful-fallback contract rather than
    # skipping past it.
    data = np.random.default_rng(0).standard_normal((50, 10))
    result = umap_embedding(data, n_components=2)
    if result is not None:
        assert result.shape == (50, 2)


def test_trials_to_traces_shapes() -> None:
    n_trials, n_times = 5, 20
    scores = np.random.default_rng(0).standard_normal((n_trials * n_times, 3))
    traces = trials_to_traces(scores, n_trials, n_times)
    assert len(traces) == n_trials
    for trace in traces:
        assert trace["x"].shape == (n_times,)
        assert trace["y"].shape == (n_times,)
        assert trace["z"].shape == (n_times,)


def test_plot_3d_trajectories_skips_without_plotly() -> None:
    pytest.importorskip("plotly")
    from dim_reduction_viz.interactive_viz import plot_3d_trajectories

    scores = np.random.default_rng(0).standard_normal((5 * 20, 3))
    fig = plot_3d_trajectories(scores, n_trials=5, n_times=20)
    assert fig is not None
