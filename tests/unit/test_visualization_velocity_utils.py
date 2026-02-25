"""Unit tests for RNA velocity visualization contracts."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chatspatial.models.data import VisualizationParameters
from chatspatial.tools.visualization import velocity as viz_vel
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


@pytest.mark.asyncio
async def test_create_rna_velocity_visualization_rejects_unsupported_subtype(
    minimal_spatial_adata,
):
    with pytest.raises(ParameterError, match="Unsupported subtype for rna_velocity"):
        await viz_vel.create_rna_velocity_visualization(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="velocity", subtype="unknown"),
        )


@pytest.mark.asyncio
async def test_create_rna_velocity_visualization_routes_stream_by_default(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    sentinel = object()

    async def _fake_stream(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(viz_vel, "_create_velocity_stream_plot", _fake_stream)
    out = await viz_vel.create_rna_velocity_visualization(
        minimal_spatial_adata,
        VisualizationParameters(plot_type="velocity", subtype="stream"),
        context=DummyCtx(),
    )
    assert out is sentinel


@pytest.mark.asyncio
async def test_stream_requires_velocity_graph(minimal_spatial_adata, monkeypatch):
    monkeypatch.setattr(viz_vel, "require", lambda *_args, **_kwargs: None)
    with pytest.raises(DataNotFoundError, match="RNA velocity not computed"):
        await viz_vel._create_velocity_stream_plot(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="velocity", subtype="stream"),
        )


@pytest.mark.asyncio
async def test_stream_requires_valid_basis(minimal_spatial_adata, monkeypatch):
    adata = minimal_spatial_adata.copy()
    adata.uns["velocity_graph"] = np.eye(adata.n_obs)
    monkeypatch.setattr(viz_vel, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(viz_vel, "infer_basis", lambda *_args, **_kwargs: None)

    with pytest.raises(DataCompatibilityError, match="No valid embedding basis found"):
        await viz_vel._create_velocity_stream_plot(
            adata,
            VisualizationParameters(plot_type="velocity", subtype="stream"),
        )


@pytest.mark.asyncio
async def test_phase_requires_required_layers(minimal_spatial_adata, monkeypatch):
    monkeypatch.setattr(viz_vel, "require", lambda *_args, **_kwargs: None)
    with pytest.raises(DataNotFoundError, match="Missing layers for phase plot"):
        await viz_vel._create_velocity_phase_plot(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="velocity", subtype="phase"),
            context=DummyCtx(),
        )


@pytest.mark.asyncio
async def test_proportions_requires_velocity_layers(minimal_spatial_adata, monkeypatch):
    monkeypatch.setattr(viz_vel, "require", lambda *_args, **_kwargs: None)
    with pytest.raises(DataNotFoundError, match="Spliced and unspliced layers are required"):
        await viz_vel._create_velocity_proportions_plot(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="velocity", subtype="proportions"),
            context=DummyCtx(),
        )


@pytest.mark.asyncio
async def test_heatmap_requires_time_or_velocity_graph(minimal_spatial_adata, monkeypatch):
    adata = minimal_spatial_adata.copy()
    monkeypatch.setattr(viz_vel, "require", lambda *_args, **_kwargs: None)
    with pytest.raises(DataNotFoundError, match="No time ordering available"):
        await viz_vel._create_velocity_heatmap(
            adata,
            VisualizationParameters(plot_type="velocity", subtype="heatmap"),
            context=DummyCtx(),
        )


@pytest.mark.asyncio
async def test_heatmap_computes_velocity_pseudotime_when_graph_exists(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.uns["velocity_graph"] = np.eye(adata.n_obs)
    called: dict[str, bool] = {}

    monkeypatch.setattr(viz_vel, "require", lambda *_args, **_kwargs: None)

    fake_scv = ModuleType("scvelo")

    def _vp(adata_obj):
        adata_obj.obs["velocity_pseudotime"] = np.linspace(0, 1, adata_obj.n_obs)
        called["vp"] = True

    def _heatmap(*_args, **_kwargs):
        plt.figure()
        return None

    fake_scv.tl = SimpleNamespace(velocity_pseudotime=_vp)
    fake_scv.pl = SimpleNamespace(heatmap=_heatmap)
    monkeypatch.setitem(__import__("sys").modules, "scvelo", fake_scv)

    fig = await viz_vel._create_velocity_heatmap(
        adata,
        VisualizationParameters(plot_type="velocity", subtype="heatmap"),
        context=DummyCtx(),
    )
    assert fig is not None
    assert called["vp"] is True


@pytest.mark.asyncio
async def test_paga_requires_cluster_key(minimal_spatial_adata):
    with pytest.raises(ParameterError, match="cluster_key is required for PAGA plot"):
        await viz_vel._create_velocity_paga_plot(
            minimal_spatial_adata,
            VisualizationParameters(
                plot_type="velocity", subtype="paga", cluster_key="missing_key"
            ),
            context=DummyCtx(),
        )
