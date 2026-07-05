"""End-to-end motor-imagery benchmark: data -> guardrail -> CSP -> PyTorch MLP -> metrics.

Ties `provenance_pipeline` in as a pre-model guardrail: epochs are
wrapped as a TaintedArray, a few are marked contaminated, and
`filter_clean_epochs` removes them before CSP or the classifier ever
see them -- the exact scenario `provenance_pipeline` was built for.

The `torch` import is deferred into `train_one_fold` so that data
loading, the guardrail wiring, and the metric utilities below remain
usable in environments without PyTorch installed.
"""
from __future__ import annotations

import numpy as np

from classifier_benchmark.csp_features import CSP
from provenance_pipeline import TaintedArray, TaintLevel, TaintPolicy, filter_clean_epochs


def load_motor_imagery_epochs(
    use_real_data: bool = True, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Load PhysioNet EEGBCI motor-imagery epochs; fall back to synthetic data.

    Returns (X, y) with X shaped (n_epochs, n_channels, n_times) and y a
    binary label array (0=hands, 1=feet). The real-data path needs
    network access to fetch PhysioNet via MNE on first call and is
    skipped automatically -- falling back to a synthetic dataset with a
    comparable discriminative structure -- if MNE is missing, the
    download fails, or `use_real_data` is False.
    """
    if use_real_data:
        try:
            return _load_physionet_eegbci()
        except Exception as exc:  # network, missing package, subject unavailable, ...
            print(
                f"[load_motor_imagery_epochs] real-data path unavailable ({exc!r}); "
                "using synthetic fallback"
            )
    return _synthetic_motor_imagery_epochs(seed=seed)


def _load_physionet_eegbci(subject: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fetch and epoch PhysioNet EEGBCI imagined hands-vs-feet motor imagery.

    Follows the standard MNE recipe for this dataset: relabel the T1/T2
    annotations to hands/feet, band-pass to the mu/beta band (7-30 Hz),
    average reference, and epoch 1-3 s after cue onset (skipping the
    onset-evoked response).
    """
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    runs = [6, 10, 14]  # imagined hands vs feet, per PhysioNet run documentation
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)
    raw.annotations.rename({"T1": "hands", "T2": "feet"})
    raw.set_eeg_reference(projection=True)
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    events, event_id = mne.events_from_annotations(raw, event_id={"hands": 1, "feet": 2})
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=1.0,
        tmax=3.0,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    X = epochs.get_data(copy=True)
    y = (epochs.events[:, -1] == event_id["feet"]).astype(np.int64)
    return X, y


def _synthetic_motor_imagery_epochs(
    n_epochs: int = 200, n_channels: int = 16, n_times: int = 200, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Two-class synthetic epochs with a spatially-localized 10 Hz oscillation.

    Mimics sensorimotor mu-rhythm ERD/ERS: class 0 carries the
    discriminative oscillation on the first half of channels, class 1
    on the second half, on top of shared background noise -- enough
    structure for CSP to find real spatial filters, not chance-level
    ones.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_epochs)
    X = rng.standard_normal((n_epochs, n_channels, n_times)).astype(np.float64) * 0.5
    t = np.linspace(0.0, 1.0, n_times)
    source = np.sin(2 * np.pi * 10 * t)
    half = n_channels // 2
    for i in range(n_epochs):
        amp = rng.uniform(1.5, 2.5)
        chans = slice(0, half) if y[i] == 0 else slice(half, n_channels)
        X[i, chans, :] += amp * source
    return X, y.astype(np.int64)


def inject_synthetic_contamination(
    X: np.ndarray, seed: int = 1, n_bad_epochs: int = 6
) -> tuple[TaintedArray, np.ndarray]:
    """Wrap epochs as a TaintedArray with a few epochs marked CONTAMINATED."""
    rng = np.random.default_rng(seed)
    ta = TaintedArray(X, stage="EPOCHED")
    bad_epochs = np.sort(rng.choice(X.shape[0], size=n_bad_epochs, replace=False))
    mask = np.zeros(X.shape, dtype=bool)
    for e in bad_epochs:
        bad_channel = rng.integers(0, X.shape[1])
        mask[e, bad_channel, :] = True
    tainted = ta.mark(mask, TaintLevel.CONTAMINATED, "synthetic artifact injection")
    return tainted, bad_epochs


def roc_auc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based binary ROC-AUC (Mann-Whitney U statistic), no sklearn."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(np.sum(y_true == 1))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[y_true == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def k_fold_indices(n: int, k: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Manual k-fold split into (train_idx, val_idx) pairs, no sklearn."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2x2 confusion matrix, rows=true, cols=predicted, no sklearn."""
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    return cm


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_components: int = 4,
    hidden_dim: int = 16,
    n_epochs_torch: int = 60,
    lr: float = 1e-2,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit CSP + a PyTorch MLP on one fold; return (accuracy, auc, y_val, val_preds)."""
    import torch
    from torch import nn

    from classifier_benchmark.model import CSPClassifier

    csp = CSP(n_components=n_components)
    feats_train = csp.fit_transform(X_train, y_train)
    feats_val = csp.transform(X_val)

    mean, std = feats_train.mean(axis=0), feats_train.std(axis=0) + 1e-8
    feats_train = (feats_train - mean) / std
    feats_val = (feats_val - mean) / std

    model = CSPClassifier(n_features=feats_train.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(feats_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    model.train()
    for _ in range(n_epochs_torch):
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(feats_val, dtype=torch.float32))
        val_probs = torch.softmax(val_logits, dim=1)[:, 1].numpy()
        val_preds = val_logits.argmax(dim=1).numpy()

    accuracy = float((val_preds == y_val).mean())
    auc = roc_auc_binary(y_val, val_probs)
    return accuracy, auc, y_val, val_preds


if __name__ == "__main__":
    X, y = load_motor_imagery_epochs(use_real_data=True, seed=0)
    print(f"loaded epochs: X={X.shape}, class balance={np.bincount(y)}")

    tainted, bad_epoch_idx = inject_synthetic_contamination(X, n_bad_epochs=6)
    print(f"injected contamination into epochs: {bad_epoch_idx.tolist()}")

    policy = TaintPolicy(max_taint_level=TaintLevel.CLEAN, max_contaminated_fraction=0.0)
    clean, keep_mask = filter_clean_epochs(tainted, policy)
    rejected = np.flatnonzero(~keep_mask)
    rejected_ok = np.array_equal(rejected, bad_epoch_idx)
    print(
        f"guardrail kept {int(keep_mask.sum())}/{len(X)} epochs "
        f"(rejected exactly the injected ones: {rejected_ok})"
    )
    assert rejected_ok

    X_clean, y_clean = clean.data, y[keep_mask]

    splits = k_fold_indices(len(X_clean), k=5, seed=0)
    accuracies, aucs = [], []
    for fold, (train_idx, val_idx) in enumerate(splits):
        acc, auc, _y_true_fold, _y_pred_fold = train_one_fold(
            X_clean[train_idx], y_clean[train_idx], X_clean[val_idx], y_clean[val_idx]
        )
        accuracies.append(acc)
        aucs.append(auc)
        print(f"fold {fold}: accuracy={acc:.3f} auc={auc:.3f}")
    print(f"mean accuracy={np.mean(accuracies):.3f} +/- {np.std(accuracies):.3f}")
    print(f"mean AUC={np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X_clean))
    n_train = int(0.8 * len(X_clean))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    _acc_test, _auc_test, y_test_true, y_test_pred = train_one_fold(
        X_clean[train_idx], y_clean[train_idx], X_clean[test_idx], y_clean[test_idx]
    )
    cm = confusion_matrix_binary(y_test_true, y_test_pred)
    print("held-out confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    assert np.mean(accuracies) > 0.6, "sanity check: should beat chance on separable synthetic classes"
    print("OK: guardrail removed contaminated epochs before any model saw them; classifier beats chance.")
