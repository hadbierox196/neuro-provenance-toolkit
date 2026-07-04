"""Interactive Plotly visualization of population trajectories.

Takes PCA or UMAP output (plain NumPy arrays from `reduce.py`) and
renders an interactive 3D trajectory, one line per trial, exported to a
standalone HTML file. Data preparation is plain NumPy and is exercised
in the demo below independent of whether Plotly is actually installed;
only the final render call needs it.
"""
from __future__ import annotations

import numpy as np


def trials_to_traces(
    scores: np.ndarray, n_trials: int, n_times: int
) -> list[dict]:
    """Reshape flat (n_trials*n_times, n_components) scores into one dict per trial.

    Kept separate from the actual Plotly call so this reshaping logic
    is testable without Plotly installed.
    """
    reshaped = scores.reshape(n_trials, n_times, -1)
    traces = []
    for trial in range(n_trials):
        traces.append(
            {
                "trial": trial,
                "x": reshaped[trial, :, 0],
                "y": reshaped[trial, :, 1],
                "z": reshaped[trial, :, 2] if reshaped.shape[2] > 2 else np.zeros(n_times),
                "t": np.arange(n_times),
            }
        )
    return traces


def plot_3d_trajectories(
    scores: np.ndarray,
    n_trials: int,
    n_times: int,
    title: str = "Population trajectory",
    axis_labels: tuple[str, str, str] = ("PC1", "PC2", "PC3"),
):
    """Build an interactive Plotly figure: one 3D line+marker trace per trial."""
    import plotly.graph_objects as go

    traces = trials_to_traces(scores, n_trials, n_times)
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(
            go.Scatter3d(
                x=trace["x"],
                y=trace["y"],
                z=trace["z"],
                mode="lines+markers",
                marker=dict(size=2, color=trace["t"], colorscale="Viridis", showscale=False),
                line=dict(width=2),
                name=f"trial {trace['trial']}",
                showlegend=False,
            )
        )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=axis_labels[0], yaxis_title=axis_labels[1], zaxis_title=axis_labels[2]),
    )
    return fig


if __name__ == "__main__":
    from dim_reduction_viz.population_simulation import simulate_population_trials
    from dim_reduction_viz.reduce import pca_via_svd

    n_trials, n_times = 30, 200
    trials = simulate_population_trials(n_trials=n_trials, n_times=n_times, n_neurons=50, seed=0)
    flat_activity = trials.activity.reshape(-1, trials.activity.shape[-1])
    result = pca_via_svd(flat_activity, n_components=3)

    print("testing trace reshaping (pure NumPy, no plotly needed for this part)...")
    traces = trials_to_traces(result.scores, n_trials, n_times)
    assert len(traces) == n_trials
    assert traces[0]["x"].shape == (n_times,)
    print(f"reshaped {n_trials} trials x {n_times} timepoints into per-trial (x, y, z) traces.")

    try:
        fig = plot_3d_trajectories(result.scores, n_trials, n_times, title="Population PCA trajectory")
        out_path = "/tmp/population_trajectory.html"
        fig.write_html(out_path)
        print(f"saved interactive figure to {out_path}")
    except ImportError as exc:
        print(f"plotly not installed here ({exc}); trace data above is what would be plotted.")
    print("OK: trajectory data prepared; render step uses Plotly directly if available.")
