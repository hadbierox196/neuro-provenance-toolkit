"""Tests for spiking_networks: plotting helpers and task generation run everywhere;
Brian2 simulation and PyTorch training are skipped when those packages are absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from spiking_networks.raster_viz import plot_population_rates, plot_psychometric, plot_raster
from spiking_networks.surrogate_gradient_snn import generate_task_batch


def test_generate_task_batch_shape_and_labels() -> None:
    coherences = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    spikes, labels = generate_task_batch(coherences, n_per_pool=10, n_steps=50, seed=0)
    assert spikes.shape == (5, 50, 20)
    assert set(np.unique(spikes)) <= {0.0, 1.0}
    assert labels.tolist() == [0, 0, 0, 1, 1]


def test_generate_task_batch_encodes_coherence_as_rate_difference() -> None:
    spikes, _ = generate_task_batch(np.array([1.0]), n_per_pool=20, n_steps=200, seed=0)
    rate_a = spikes[0, :, :20].mean()
    rate_b = spikes[0, :, 20:].mean()
    assert rate_a > rate_b


def test_generate_task_batch_no_saturation() -> None:
    spikes, _ = generate_task_batch(np.array([1.0, -1.0]), n_per_pool=20, n_steps=200, seed=0)
    assert spikes.mean() < 0.9, "input encoding should not saturate at the spike-probability ceiling"


def test_plot_raster_runs_on_synthetic_spikes() -> None:
    rng = np.random.default_rng(0)
    t_a, i_a = rng.uniform(0, 500, 100), rng.integers(0, 20, 100)
    t_b, i_b = rng.uniform(0, 500, 80), rng.integers(0, 20, 80)
    ax = plot_raster(t_a, i_a, t_b, i_b)
    assert ax is not None


def test_plot_population_rates_runs() -> None:
    t = np.linspace(0, 500, 100)
    rate_a = 40 + 5 * np.random.default_rng(0).standard_normal(100)
    rate_b = 20 + 5 * np.random.default_rng(1).standard_normal(100)
    ax = plot_population_rates(t, rate_a, rate_b, decision_threshold_hz=55.0, decision_time_ms=200.0)
    assert ax is not None


def test_plot_psychometric_runs() -> None:
    coherences = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    accuracies = np.array([0.5, 0.65, 0.8, 0.9, 0.97])
    ax = plot_psychometric(coherences, accuracies)
    assert ax is not None


def test_brian2_decision_trial_runs_and_is_coherence_sensitive() -> None:
    pytest.importorskip("brian2")
    from spiking_networks.lif_decision_network import run_coherence_sweep

    sweep = run_coherence_sweep([0.0, 1.0], n_trials_per_level=5, seed0=0)
    acc_low = sweep[0.0]["winners"].count("A") / 5
    acc_high = sweep[1.0]["winners"].count("A") / 5
    assert acc_high >= acc_low


def test_surrogate_training_reduces_loss() -> None:
    pytest.importorskip("torch")
    from spiking_networks.surrogate_gradient_snn import train

    _, losses = train(n_epochs=50, seed=0)
    assert losses[-1] < losses[0]
