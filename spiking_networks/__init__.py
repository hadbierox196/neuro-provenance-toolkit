"""Spiking decision networks: a hand-tuned Brian2 circuit and a learned PyTorch one."""
from __future__ import annotations

from utils_and_tests.shared_types import lazy_getattr

__all__ = [
    "DecisionNetworkParams",
    "DecisionResult",
    "run_decision_trial",
    "run_coherence_sweep",
    "generate_task_batch",
    "train",
    "evaluate",
    "plot_raster",
    "plot_population_rates",
    "plot_psychometric",
    "plot_chronometric",
    "summary_figure",
]

_EXPORTS = {
    "DecisionNetworkParams": "spiking_networks.lif_decision_network",
    "DecisionResult": "spiking_networks.lif_decision_network",
    "run_decision_trial": "spiking_networks.lif_decision_network",
    "run_coherence_sweep": "spiking_networks.lif_decision_network",
    "generate_task_batch": "spiking_networks.surrogate_gradient_snn",
    "train": "spiking_networks.surrogate_gradient_snn",
    "evaluate": "spiking_networks.surrogate_gradient_snn",
    "plot_raster": "spiking_networks.raster_viz",
    "plot_population_rates": "spiking_networks.raster_viz",
    "plot_psychometric": "spiking_networks.raster_viz",
    "plot_chronometric": "spiking_networks.raster_viz",
    "summary_figure": "spiking_networks.raster_viz",
}

__getattr__ = lazy_getattr(_EXPORTS, __name__)
