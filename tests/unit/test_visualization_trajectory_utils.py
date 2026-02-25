"""Unit tests for trajectory visualization routing and guard contracts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chatspatial.models.data import VisualizationParameters
from chatspatial.tools.visualization import trajectory as viz_traj
from chatspatial.utils.exceptions import (
    DataCompatibilityError,
    DataNotFoundError,
    ParameterError,
)


class DummyCtx:
    def __init__(self):
        self.infos: list[str] = []

    async def info(self, msg: str):
        self.infos.append(msg)


def _install_fake_cellrank(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cr = type("FakeCellRank", (), {})()
    fake_cr.pl = type("FakeCRPlt", (), {})()
    monkeypatch.setitem(__import__("sys").modules, "cellrank", fake_cr)


@pytest.mark.asyncio
async def test_create_trajectory_visualization_rejects_unsupported_subtype(
    minimal_spatial_adata,
):
    with pytest.raises(ParameterError, match="Unsupported subtype for trajectory"):
        await viz_traj.create_trajectory_visualization(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="trajectory", subtype="bad"),
        )


@pytest.mark.asyncio
async def test_create_trajectory_visualization_routes_pseudotime(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    sentinel = object()

    async def _fake(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(viz_traj, "_create_trajectory_pseudotime_plot", _fake)
    out = await viz_traj.create_trajectory_visualization(
        minimal_spatial_adata,
        VisualizationParameters(plot_type="trajectory", subtype="pseudotime"),
        context=DummyCtx(),
    )
    assert out is sentinel


@pytest.mark.asyncio
async def test_pseudotime_plot_requires_pseudotime_column(minimal_spatial_adata):
    with pytest.raises(DataNotFoundError, match="No pseudotime found"):
        await viz_traj._create_trajectory_pseudotime_plot(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="trajectory", subtype="pseudotime"),
            context=DummyCtx(),
        )


@pytest.mark.asyncio
async def test_pseudotime_plot_requires_valid_basis(minimal_spatial_adata, monkeypatch):
    adata = minimal_spatial_adata.copy()
    adata.obs["dpt_pseudotime"] = np.linspace(0, 1, adata.n_obs)
    monkeypatch.setattr(viz_traj, "infer_basis", lambda *_args, **_kwargs: None)
    with pytest.raises(DataCompatibilityError, match="No valid embedding basis found"):
        await viz_traj._create_trajectory_pseudotime_plot(
            adata,
            VisualizationParameters(plot_type="trajectory", subtype="pseudotime"),
            context=DummyCtx(),
        )


@pytest.mark.asyncio
async def test_cellrank_circular_requires_fate_probabilities(minimal_spatial_adata, monkeypatch):
    monkeypatch.setattr(viz_traj, "require", lambda *_args, **_kwargs: None)
    _install_fake_cellrank(monkeypatch)
    with pytest.raises(DataNotFoundError, match="CellRank fate probabilities not found"):
        await viz_traj._create_cellrank_circular_projection(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="trajectory", subtype="circular"),
            context=DummyCtx(),
        )


@pytest.mark.asyncio
async def test_cellrank_fate_map_requires_cluster_key_when_no_categorical(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obsm["lineages_fwd"] = np.ones((adata.n_obs, 2))
    # remove categorical fallback
    adata.obs = pd.DataFrame(index=adata.obs.index)
    monkeypatch.setattr(viz_traj, "require", lambda *_args, **_kwargs: None)
    _install_fake_cellrank(monkeypatch)

    with pytest.raises(ParameterError, match="cluster_key is required for fate map"):
        await viz_traj._create_cellrank_fate_map(
            adata,
            VisualizationParameters(plot_type="trajectory", subtype="fate_map"),
            context=DummyCtx(),
        )


@pytest.mark.asyncio
async def test_cellrank_gene_trends_requires_time_key(minimal_spatial_adata, monkeypatch):
    adata = minimal_spatial_adata.copy()
    adata.obsm["lineages_fwd"] = np.ones((adata.n_obs, 2))
    monkeypatch.setattr(viz_traj, "require", lambda *_args, **_kwargs: None)
    _install_fake_cellrank(monkeypatch)

    with pytest.raises(DataNotFoundError, match="No pseudotime found"):
        await viz_traj._create_cellrank_gene_trends(
            adata,
            VisualizationParameters(plot_type="trajectory", subtype="gene_trends"),
            context=DummyCtx(),
        )


@pytest.mark.asyncio
async def test_palantir_results_requires_pseudotime(minimal_spatial_adata):
    with pytest.raises(DataNotFoundError, match="Palantir results not found"):
        await viz_traj._create_palantir_results(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="trajectory", subtype="palantir"),
            context=DummyCtx(),
        )


@pytest.mark.asyncio
async def test_palantir_results_nonempty_fate_probs_branch(minimal_spatial_adata, monkeypatch):
    adata = minimal_spatial_adata.copy()
    adata.obs["palantir_pseudotime"] = np.linspace(0, 1, adata.n_obs)
    adata.obs["palantir_entropy"] = np.linspace(1, 0, adata.n_obs)
    adata.obsm["palantir_fate_probs"] = pd.DataFrame(
        {"f1": np.linspace(0.6, 0.3, adata.n_obs), "f2": np.linspace(0.4, 0.7, adata.n_obs)},
        index=adata.obs_names,
    )

    monkeypatch.setattr(viz_traj, "infer_basis", lambda *_args, **_kwargs: "spatial")

    calls: list[str] = []

    def _embedding(*_args, **kwargs):
        calls.append(str(kwargs.get("color")))
        return None

    monkeypatch.setattr(viz_traj.sc.pl, "embedding", _embedding)

    fig = await viz_traj._create_palantir_results(
        adata,
        VisualizationParameters(plot_type="trajectory", subtype="palantir"),
        context=DummyCtx(),
    )
    assert fig is not None
    assert "palantir_pseudotime" in calls
    assert "palantir_entropy" in calls
    assert "_dominant_fate" in calls
    assert "_dominant_fate" not in adata.obs.columns
    plt.close("all")
