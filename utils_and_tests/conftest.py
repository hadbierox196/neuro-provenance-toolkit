"""Shared pytest fixtures: seeded synthetic data at shapes the test suite needs."""
from __future__ import annotations

import numpy as np
import pytest

from utils_and_tests.synthetic_data import synthetic_multichannel_signal, synthetic_two_class_epochs


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh, seeded RNG -- depend on this instead of calling np.random directly."""
    return np.random.default_rng(0)


@pytest.fixture
def clean_signal() -> tuple[np.ndarray, np.ndarray]:
    """(data, bad_mask) with no injected artifact; bad_mask is all-False."""
    return synthetic_multichannel_signal(n_channels=8, n_times=500, seed=0)


@pytest.fixture
def artifact_signal() -> tuple[np.ndarray, np.ndarray]:
    """(data, bad_mask) with one channel carrying an injected artifact."""
    return synthetic_multichannel_signal(
        n_channels=8, n_times=500, seed=1, artifact_channel=3, artifact_span=(200, 230)
    )


@pytest.fixture
def two_class_epochs() -> tuple[np.ndarray, np.ndarray]:
    """(X, y) epoched data with a clearly learnable class-dependent signal."""
    return synthetic_two_class_epochs(n_epochs=40, n_channels=8, n_times=100, separation=3.0, seed=0)
