# neuro-provenance-toolkit

Eight small, independently-runnable modules covering signal processing, spiking
network simulation, encoding models, connectivity, population geometry, and a
classifier — all wired to one idea that ties the repo together:
**contamination shouldn't be able to hide.**

## The centerpiece: `provenance_pipeline`

Every real EEG pipeline eventually asks "did a bad channel leak into this
result?" — and usually can't answer with certainty once the data has passed
through several transforms, especially an opaque model. `provenance_pipeline`
wraps NumPy arrays in a `TaintedArray` that carries a same-shaped
contamination mask through arithmetic, reductions, temporal filtering,
cross-channel referencing, and even a black-box model, via NumPy's
`__array_ufunc__` / `__array_function__` dispatch protocols. `classifier_benchmark`
wires it in as a guardrail: contaminated epochs are identified and removed
*before* they reach the model, with an audit trail of what was rejected and why.

## Modules

| Module | What it does |
|---|---|
| `provenance_pipeline` | Taint-tracking `TaintedArray`; propagation rules; guardrail gate |
| `classifier_benchmark` | CSP + PyTorch MLP motor-imagery classifier, gated by the guardrail |
| `spiking_networks` | A hand-tuned Brian2 2AFC decision circuit *and* a PyTorch surrogate-gradient version of the same task |
| `eeg_preprocessing` | MNE filter → reference → ICA pipeline with a structured report, not silent mutation |
| `encoding_models` | LNP simulation + a from-scratch, gradient-checked Poisson GLM + STA baseline |
| `connectivity_analysis` | Welch coherence + a from-scratch VAR/Granger causality F-test |
| `dim_reduction_viz` | SVD-based PCA + optional UMAP on simulated population trajectories, interactive Plotly 3D plot |
| `utils_and_tests` | Shared types, synthetic data generators, and the pytest suite (incl. a property test that taint is monotonic) |

Each module is self-contained: `python -m <module>.<file>` runs that file's
own demo. Nothing requires a GPU or an internet connection except
`eeg_preprocessing`'s and `classifier_benchmark`'s optional real-data paths
(PhysioNet EEGBCI via MNE), which fall back to synthetic data automatically
when a network or the dataset isn't available.

## Design choices worth knowing about

- **Hand-rolled math over library calls, where the point is the math.**
  PCA is SVD, not `sklearn.decomposition.PCA`. CSP is a generalized
  eigendecomposition, not `mne.decoding.CSP`. The Poisson GLM is a custom
  negative log-likelihood fit with `scipy.optimize`, gradient-checked against
  finite differences, not `statsmodels`. Granger causality is a VAR(p)
  least-squares fit and an F-test, not a library call. Numerical *primitives*
  (SVD, generalized eigh, Welch's method, L-BFGS-B) are used directly —
  there's nothing to gain by reimplementing FFT-based spectral estimation
  by hand, unlike the modeling decisions above.
- **Lazy imports everywhere a heavy dependency is optional.** Every
  package's `__init__.py` resolves its public names through one shared
  `utils_and_tests.shared_types.lazy_getattr` factory (PEP 562), so a
  submodule is only imported when one of its names is actually used —
  `import classifier_benchmark` doesn't require torch, and
  `import spiking_networks` doesn't require Brian2, until you call
  something that needs them.
- **Graceful degradation, not silent failure.** Real-dataset fetches
  (PhysioNet), optional visualization libraries (Plotly, UMAP), and any
  missing heavy dependency all fail with a clear printed message and a
  tested fallback path — never a crash, never a silent wrong answer.

## Quickstart

```bash
pip install -e ".[dev,viz]"
pytest utils_and_tests -v          # heavy-dependency tests skip cleanly if e.g. torch/brian2/mne aren't installed
python -m provenance_pipeline.guardrail    # or any other <module>.<file>
```

## Repo layout

```
provenance_pipeline/      classifier_benchmark/     spiking_networks/
eeg_preprocessing/         encoding_models/          connectivity_analysis/
dim_reduction_viz/         utils_and_tests/
.github/workflows/ci.yml   pyproject.toml            LICENSE
```
