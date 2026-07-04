"""ICA fitting and automatic EOG-correlated artifact-component detection.

Fits ICA on a 1 Hz-highpassed copy of the data (standard MNE guidance:
a mild highpass markedly improves ICA convergence) but applies the
resulting unmixing to the original, non-highpassed Raw -- so the
returned cleaned signal keeps its original low-frequency content aside
from whatever components were flagged as artifactual.
"""
from __future__ import annotations

import numpy as np


def fit_ica_and_find_artifacts(
    raw,
    n_components: int = 10,
    random_state: int = 97,
    eog_ch_names: tuple[str, ...] | None = None,
    eog_threshold: float = 3.0,
):
    """Fit ICA and auto-exclude components correlated with an EOG channel.

    Parameters
    ----------
    raw : mne.io.Raw
        Filtered, re-referenced continuous data.
    n_components : int
        Number of ICA components. Keep comfortably below the channel
        count -- average referencing makes the data rank-deficient by 1.
    eog_ch_names : tuple[str, ...], optional
        Explicit EOG channel name(s). If None, uses any channel typed
        "eog" in `raw.info`; if there is none, no automatic exclusion
        is attempted (an empty exclude list is returned).

    Returns
    -------
    cleaned : mne.io.Raw
    ica : mne.preprocessing.ICA
    excluded_indices : list[int]
    eog_scores : list[float]
    """
    import mne
    from mne.preprocessing import ICA

    ica = ICA(n_components=n_components, random_state=random_state, method="infomax", fit_params=dict(extended=True))
    raw_for_fit = raw.copy().filter(l_freq=1.0, h_freq=None, fir_design="firwin", verbose=False)
    ica.fit(raw_for_fit)

    eog_picks = mne.pick_types(raw.info, eeg=False, eog=True)
    excluded_indices: list[int] = []
    eog_scores: list[float] = []
    if len(eog_picks) > 0 or eog_ch_names:
        ch_name_arg = list(eog_ch_names) if eog_ch_names else None
        found_indices, scores = ica.find_bads_eog(raw, ch_name=ch_name_arg, threshold=eog_threshold, verbose=False)
        excluded_indices = list(found_indices)
        eog_scores = np.asarray(scores).ravel().tolist()

    ica.exclude = excluded_indices
    cleaned = raw.copy()
    ica.apply(cleaned, verbose=False)
    return cleaned, ica, excluded_indices, eog_scores


if __name__ == "__main__":
    from eeg_preprocessing.pipeline import _make_synthetic_raw

    raw = _make_synthetic_raw(seed=1)
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design="firwin", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    cleaned, ica, excluded, scores = fit_ica_and_find_artifacts(raw, n_components=10)
    print(f"fit ICA: {ica.n_components_} components")
    print(f"excluded as EOG-correlated: {excluded}")
    print(f"EOG correlation scores: {[f'{s:.2f}' for s in scores]}")
    assert len(excluded) >= 1, "should catch the injected blink artifact"
    print("OK: at least one EOG-correlated component was detected and removed.")
