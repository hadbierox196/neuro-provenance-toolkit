"""PCA/UMAP dimensionality reduction and interactive Plotly visualization for population data."""
from __future__ import annotations

from utils_and_tests.shared_types import lazy_getattr

__all__ = [
    "PopulationTrials",
    "simulate_population_trials",
    "PCAResult",
    "pca_via_svd",
    "umap_embedding",
    "latent_recovery_r2",
    "trials_to_traces",
    "plot_3d_trajectories",
]

_EXPORTS = {
    "PopulationTrials": "dim_reduction_viz.population_simulation",
    "simulate_population_trials": "dim_reduction_viz.population_simulation",
    "PCAResult": "dim_reduction_viz.reduce",
    "pca_via_svd": "dim_reduction_viz.reduce",
    "umap_embedding": "dim_reduction_viz.reduce",
    "latent_recovery_r2": "dim_reduction_viz.reduce",
    "trials_to_traces": "dim_reduction_viz.interactive_viz",
    "plot_3d_trajectories": "dim_reduction_viz.interactive_viz",
}

__getattr__ = lazy_getattr(_EXPORTS, __name__)
