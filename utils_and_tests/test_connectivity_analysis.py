"""Tests for connectivity_analysis: coherence and Granger causality against known ground truth."""
from __future__ import annotations

import numpy as np

from connectivity_analysis.granger_causality import (
    granger_causality_fstat,
    granger_causality_matrix,
    simulate_coupled_ar,
)
from connectivity_analysis.spectral_coherence import band_coherence_matrix, pairwise_coherence


def test_coherence_high_for_shared_oscillation() -> None:
    rng = np.random.default_rng(0)
    fs, n_times = 250.0, 5000
    t = np.arange(n_times) / fs
    shared = np.sin(2 * np.pi * 10 * t)
    a = shared + 0.5 * rng.standard_normal(n_times)
    b = shared + 0.5 * rng.standard_normal(n_times)
    data = np.stack([a, b])
    matrix = band_coherence_matrix(data, fs, fmin=8.0, fmax=12.0)
    assert matrix[0, 1] > 0.7


def test_coherence_low_for_independent_noise() -> None:
    rng = np.random.default_rng(0)
    fs, n_times = 250.0, 5000
    data = rng.standard_normal((2, n_times))
    matrix = band_coherence_matrix(data, fs, fmin=8.0, fmax=12.0)
    assert matrix[0, 1] < 0.3


def test_pairwise_coherence_covers_all_pairs() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 2000))
    results = pairwise_coherence(data, fs=250.0)
    expected_pairs = {(i, j) for i in range(4) for j in range(i + 1, 4)}
    assert set(results.keys()) == expected_pairs


def test_granger_recovers_true_causal_direction() -> None:
    x, y = simulate_coupled_ar(n_samples=2000, coupling=0.6, seed=0)
    _, p_xy = granger_causality_fstat(y, x, lag=2)  # does x cause y? (true)
    _, p_yx = granger_causality_fstat(x, y, lag=2)  # does y cause x? (false)
    assert p_xy < 0.001
    assert p_yx > 0.05


def test_granger_matrix_agrees_with_pairwise() -> None:
    x, y = simulate_coupled_ar(n_samples=1500, coupling=0.6, seed=1)
    rng = np.random.default_rng(2)
    independent = rng.standard_normal(1500)
    data = np.stack([x, y, independent])
    f_matrix, p_matrix = granger_causality_matrix(data, lag=2)

    f_direct, p_direct = granger_causality_fstat(y, x, lag=2)
    assert abs(p_matrix[1, 0] - p_direct) < 1e-9
    assert abs(f_matrix[1, 0] - f_direct) < 1e-9
    assert np.isnan(f_matrix[0, 0])  # diagonal undefined by construction


def test_no_coupling_gives_no_significant_direction() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(2000)
    y = rng.standard_normal(2000)
    _, p_xy = granger_causality_fstat(y, x, lag=2)
    _, p_yx = granger_causality_fstat(x, y, lag=2)
    assert p_xy > 0.01
    assert p_yx > 0.01


def test_plot_matrix_heatmap_runs() -> None:
    from connectivity_analysis.spectral_coherence import plot_matrix_heatmap

    matrix = np.array([[1.0, 0.7, 0.1], [0.7, 1.0, 0.2], [0.1, 0.2, 1.0]])
    fig = plot_matrix_heatmap(matrix, labels=["a", "b", "c"])
    assert fig is not None


def test_plot_matrix_heatmap_handles_nan_diagonal() -> None:
    from connectivity_analysis.spectral_coherence import plot_matrix_heatmap

    matrix = np.array([[np.nan, 0.5], [0.3, np.nan]])
    fig = plot_matrix_heatmap(np.nan_to_num(matrix, nan=0.0))
    assert fig is not None
