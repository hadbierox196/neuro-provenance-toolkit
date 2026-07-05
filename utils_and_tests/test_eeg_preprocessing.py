"""Tests for eeg_preprocessing.

Most functions here operate on `mne.io.Raw` objects and are skipped
per-test via `pytest.importorskip("mne")` when MNE isn't installed.
`plot_psd_comparison` is the exception -- it takes plain NumPy arrays,
not a Raw object, so it's tested unconditionally.
"""
from __future__ import annotations

import numpy as np
import pytest

from eeg_preprocessing.pipeline import plot_psd_comparison


def test_plot_psd_comparison_runs_without_mne() -> None:
    freqs = np.linspace(0, 60, 200)
    db_before = -20 - 0.1 * freqs
    db_after = -20 - 0.1 * freqs - 2
    fig = plot_psd_comparison(freqs, db_before, db_after, notch_freqs=(60.0,))
    assert fig is not None


def test_synthetic_raw_has_expected_channels() -> None:
    pytest.importorskip("mne")
    from eeg_preprocessing.pipeline import _make_synthetic_raw

    raw = _make_synthetic_raw(seed=0)
    assert len(raw.ch_names) == 17
    assert raw.get_channel_types()[-1] == "eog"


def test_pipeline_flags_the_injected_blink() -> None:
    pytest.importorskip("mne")
    from eeg_preprocessing.pipeline import PreprocessConfig, _make_synthetic_raw, run_preprocessing

    raw = _make_synthetic_raw(seed=0)
    cfg = PreprocessConfig(n_ica_components=10)
    cleaned, report, ica = run_preprocessing(raw, cfg)
    assert len(report.excluded_ica_components) >= 1
    assert cleaned.get_data().shape == raw.get_data().shape


def test_notch_filter_reduces_line_noise() -> None:
    pytest.importorskip("mne")
    from eeg_preprocessing.pipeline import (
        PreprocessConfig,
        _make_synthetic_raw,
        compare_psd,
        run_preprocessing,
    )

    raw = _make_synthetic_raw(seed=0)
    cfg = PreprocessConfig(notch_freqs=(60.0,), n_ica_components=10)
    cleaned, _, _ = run_preprocessing(raw, cfg)
    freqs, db_before, db_after = compare_psd(raw, cleaned)
    line_idx = int((abs(freqs - 60)).argmin())
    assert db_after[line_idx] < db_before[line_idx]


def test_ica_utils_returns_consistent_shapes() -> None:
    pytest.importorskip("mne")
    from eeg_preprocessing.ica_utils import fit_ica_and_find_artifacts
    from eeg_preprocessing.pipeline import _make_synthetic_raw

    raw = _make_synthetic_raw(seed=1)
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design="firwin", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    cleaned, ica, excluded, scores = fit_ica_and_find_artifacts(raw, n_components=10)
    assert cleaned.get_data().shape == raw.get_data().shape
    assert len(scores) == len(excluded) or len(excluded) == 0
