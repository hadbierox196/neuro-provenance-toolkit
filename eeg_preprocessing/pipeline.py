"""Filter -> reference -> ICA orchestration for continuous EEG.

Wraps the three canonical MNE-Python preprocessing steps -- band/notch
filtering, re-referencing, and ICA-based artifact removal -- behind one
`PreprocessConfig` + `run_preprocessing()` call, and returns a
`PreprocessReport` describing what happened (filter settings, which ICA
components were rejected and why) rather than silently mutating data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eeg_preprocessing.ica_utils import fit_ica_and_find_artifacts


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings for one preprocessing run."""

    l_freq: float = 1.0
    h_freq: float = 40.0
    notch_freqs: tuple[float, ...] = (60.0,)
    reference: str = "average"
    n_ica_components: int = 10
    ica_random_state: int = 97
    eog_ch_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PreprocessReport:
    """What `run_preprocessing` actually did, for logging/reproducibility."""

    l_freq: float
    h_freq: float
    notch_freqs: tuple[float, ...]
    reference: str
    n_ica_components: int
    excluded_ica_components: list[int]
    eog_scores: list[float]


def run_preprocessing(raw, cfg: PreprocessConfig = PreprocessConfig()):
    """Filter, re-reference, and ICA-clean `raw`. Returns (cleaned_raw, report, ica)."""
    raw = raw.copy()
    raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, fir_design="firwin", verbose=False)
    if cfg.notch_freqs:
        raw.notch_filter(freqs=list(cfg.notch_freqs), verbose=False)
    raw.set_eeg_reference(cfg.reference, projection=False, verbose=False)

    cleaned, ica, excluded, eog_scores = fit_ica_and_find_artifacts(
        raw,
        n_components=cfg.n_ica_components,
        random_state=cfg.ica_random_state,
        eog_ch_names=cfg.eog_ch_names,
    )

    report = PreprocessReport(
        l_freq=cfg.l_freq,
        h_freq=cfg.h_freq,
        notch_freqs=cfg.notch_freqs,
        reference=cfg.reference,
        n_ica_components=cfg.n_ica_components,
        excluded_ica_components=excluded,
        eog_scores=eog_scores,
    )
    return cleaned, report, ica


def compare_psd(raw_before, raw_after, fmax: float = 60.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Channel-averaged PSD (dB) before vs. after, for a before/after comparison plot."""
    spec_before = raw_before.compute_psd(fmax=fmax, verbose=False)
    spec_after = raw_after.compute_psd(fmax=fmax, verbose=False)
    psd_before, freqs = spec_before.get_data(return_freqs=True)
    psd_after, _ = spec_after.get_data(return_freqs=True)
    db_before = 10 * np.log10(psd_before.mean(axis=0))
    db_after = 10 * np.log10(psd_after.mean(axis=0))
    return freqs, db_before, db_after


def plot_psd_comparison(freqs: np.ndarray, db_before: np.ndarray, db_after: np.ndarray, notch_freqs=()):
    """Plot channel-averaged PSD before vs. after preprocessing.

    Takes the plain arrays `compare_psd` returns rather than Raw
    objects, so this function itself needs only matplotlib/NumPy -- it
    can be exercised without MNE installed, even though `compare_psd`
    cannot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freqs, db_before, label="before", color="#d62728", alpha=0.8)
    ax.plot(freqs, db_after, label="after", color="#1f77b4", alpha=0.8)
    for f in notch_freqs:
        ax.axvline(f, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (dB)")
    ax.set_title("Channel-averaged PSD: before vs. after preprocessing")
    ax.legend()
    fig.tight_layout()
    return fig


def _make_synthetic_raw(seed: int = 0):
    """Synthetic 16-channel EEG + 1 EOG Raw with an injected eye-blink artifact.

    Not real data -- built so the pipeline has a genuine artifact
    (frontally-dominant, EOG-correlated) to find and remove, and a
    known-clean background (alpha rhythm + noise + 60 Hz line noise) to
    verify the filter/notch steps against.
    """
    import mne

    rng = np.random.default_rng(seed)
    sfreq = 250.0
    n_times = int(sfreq * 20)
    ch_names = [f"EEG{i:03d}" for i in range(16)] + ["EOG001"]
    ch_types = ["eeg"] * 16 + ["eog"]
    info = mne.create_info(ch_names, sfreq, ch_types)

    t = np.arange(n_times) / sfreq
    data = 2e-5 * rng.standard_normal((17, n_times))
    data[:16, :] += 3e-5 * np.sin(2 * np.pi * 10 * t)  # alpha rhythm
    data[:16, :] += 4e-5 * np.sin(2 * np.pi * 60 * t)  # powerline noise

    width = int(0.2 * sfreq)
    blink_centers = rng.choice(np.arange(width, n_times - width), size=8, replace=False)
    blink = np.zeros(n_times)
    for center in blink_centers:
        lo, hi = center - width // 2, center + width // 2
        blink[lo:hi] += 15e-5 * np.hanning(hi - lo)
    data[16, :] = blink
    frontal_gain = np.linspace(1.0, 0.2, 16)
    data[:16, :] += np.outer(frontal_gain, blink)

    return mne.io.RawArray(data, info, verbose=False)


if __name__ == "__main__":
    raw = _make_synthetic_raw(seed=0)
    print(f"synthetic raw: {len(raw.ch_names)} channels, {raw.times[-1]:.1f} s @ {raw.info['sfreq']:.0f} Hz")

    cfg = PreprocessConfig(l_freq=1.0, h_freq=40.0, notch_freqs=(60.0,), n_ica_components=10)
    cleaned, report, ica = run_preprocessing(raw, cfg)
    print(f"filter: {report.l_freq}-{report.h_freq} Hz, notch: {report.notch_freqs}")
    print(f"ICA excluded components (EOG-correlated): {report.excluded_ica_components}")
    print(f"EOG correlation scores: {[f'{s:.2f}' for s in report.eog_scores]}")

    freqs, db_before, db_after = compare_psd(raw, cleaned)
    line_idx = int(np.argmin(np.abs(freqs - 60)))
    print(f"PSD at 60 Hz: before={db_before[line_idx]:.1f} dB, after={db_after[line_idx]:.1f} dB")

    fig = plot_psd_comparison(freqs, db_before, db_after, notch_freqs=report.notch_freqs)
    out_path = "/tmp/psd_comparison.png"
    fig.savefig(out_path, dpi=120)
    print(f"saved before/after PSD comparison to {out_path}")

    assert len(report.excluded_ica_components) >= 1, "should catch the injected blink artifact"
    assert db_after[line_idx] < db_before[line_idx], "notch filter should reduce 60 Hz power"
    print("OK: pipeline filters, re-references, notches 60 Hz, and ICA-removes the blink artifact.")
