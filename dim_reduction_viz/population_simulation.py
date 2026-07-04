"""Simulate neural population activity driven by a low-dimensional latent.

A 2D rotational latent (a limit cycle, in the spirit of the rotational
dynamics reported in motor cortex population recordings) is projected
into a high-dimensional noisy "population" space -- giving
`reduce.py` a known ground-truth low-dimensional structure to recover,
rather than an arbitrary blob of numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PopulationTrials:
    """Simulated trial-structured population activity and its ground-truth latent."""

    activity: np.ndarray  # (n_trials, n_times, n_neurons)
    latent: np.ndarray  # (n_trials, n_times, 2)
    projection: np.ndarray  # (2, n_neurons)


def simulate_population_trials(
    n_trials: int = 30,
    n_times: int = 200,
    n_neurons: int = 50,
    noise_std: float = 0.3,
    seed: int | None = None,
) -> PopulationTrials:
    """Each trial traces one lap of a 2D limit cycle from a random starting phase.

    All trials share the same random projection into neuron space, so
    the underlying 2D structure is consistent across trials -- only the
    starting phase and noise differ, as in repeated behavioral trials.
    """
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((2, n_neurons))
    projection /= np.linalg.norm(projection, axis=0, keepdims=True)

    t = np.linspace(0.0, 2 * np.pi, n_times, endpoint=False)
    activity = np.empty((n_trials, n_times, n_neurons))
    latent = np.empty((n_trials, n_times, 2))
    for trial in range(n_trials):
        phase0 = rng.uniform(0, 2 * np.pi)
        trial_latent = np.stack([np.cos(t + phase0), np.sin(t + phase0)], axis=1)
        trial_activity = trial_latent @ projection + noise_std * rng.standard_normal((n_times, n_neurons))
        latent[trial] = trial_latent
        activity[trial] = trial_activity
    return PopulationTrials(activity=activity, latent=latent, projection=projection)


if __name__ == "__main__":
    trials = simulate_population_trials(n_trials=30, n_times=200, n_neurons=50, seed=0)
    print(f"activity shape: {trials.activity.shape} (trials, times, neurons)")
    print(f"latent shape: {trials.latent.shape}, projection shape: {trials.projection.shape}")

    flat_activity = trials.activity.reshape(-1, trials.activity.shape[-1])
    signal_var = np.var(trials.latent.reshape(-1, 2) @ trials.projection, axis=0).sum()
    total_var = np.var(flat_activity, axis=0).sum()
    print(f"fraction of total population variance driven by the 2D latent: {signal_var / total_var:.2f}")
    assert signal_var / total_var > 0.5, "latent signal should dominate over noise for PCA to recover it"
    print("OK: simulated population activity is latent-dominated, ready for dimensionality reduction.")
