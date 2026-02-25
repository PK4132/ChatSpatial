"""Unit tests for feature visualization contracts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chatspatial.models.data import VisualizationParameters
from chatspatial.tools.visualization import feature as viz_feature
from chatspatial.utils.exceptions import DataNotFoundError, ParameterError


class DummyCtx:
    def __init__(self):
        self.infos: list[str] = []

    async def info(self, msg: str):
        self.infos.append(msg)

    async def warning(self, msg: str):
        self.infos.append(msg)


def test_parse_lr_pairs_from_features_contract():
    regular, lr_pairs = viz_feature._parse_lr_pairs_from_features(
        ["geneA", "CCL5^CCR5", "CD3D_CD3E", "_temp"]
    )
    assert "geneA" in regular
    assert "_temp" in regular
    assert ("CCL5", "CCR5") in lr_pairs
    assert ("CD3D", "CD3E") in lr_pairs


@pytest.mark.asyncio
async def test_create_feature_visualization_invalid_basis(minimal_spatial_adata):
    with pytest.raises(ParameterError, match="Invalid basis"):
        await viz_feature.create_feature_visualization(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="feature", basis="bad", feature="gene_0"),
        )


@pytest.mark.asyncio
async def test_create_feature_visualization_requires_features_when_no_cluster(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    monkeypatch.setattr(viz_feature, "get_cluster_key", lambda *_args, **_kwargs: None)
    with pytest.raises(ParameterError, match="No features specified"):
        await viz_feature.create_feature_visualization(
            adata,
            VisualizationParameters(plot_type="feature", basis="spatial", feature=None),
        )


@pytest.mark.asyncio
async def test_create_feature_visualization_umap_fallback_computation(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx()

    def _ensure_umap(a):
        a.obsm["X_umap"] = np.ones((a.n_obs, 2))
        return True

    monkeypatch.setattr(viz_feature, "ensure_umap", _ensure_umap)

    async def _validated(*_args, **_kwargs):
        return ["gene_0"]

    monkeypatch.setattr(viz_feature, "get_validated_features", _validated)

    async def _single(*_args, **_kwargs):
        return plt.figure()

    monkeypatch.setattr(viz_feature, "_create_single_feature_plot", _single)

    fig = await viz_feature.create_feature_visualization(
        adata,
        VisualizationParameters(plot_type="feature", basis="umap", feature="gene_0"),
        context=ctx,
    )
    assert fig is not None
    assert any("Computed UMAP embedding" in m for m in ctx.infos)


@pytest.mark.asyncio
async def test_create_feature_visualization_routes_lr_pairs(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    sentinel = object()

    async def _lr(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(viz_feature, "_create_lr_pairs_visualization", _lr)

    out = await viz_feature.create_feature_visualization(
        adata,
        VisualizationParameters(
            plot_type="feature",
            basis="spatial",
            feature=["CCL5^CCR5"],
        ),
        context=DummyCtx(),
    )
    assert out is sentinel


@pytest.mark.asyncio
async def test_create_single_feature_plot_feature_not_found(minimal_spatial_adata):
    with pytest.raises(DataNotFoundError, match="Feature 'missing' not found"):
        await viz_feature._create_single_feature_plot(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="feature", basis="spatial", feature="missing"),
            "missing",
            "spatial",
            minimal_spatial_adata.obsm["spatial"],
        )


@pytest.mark.asyncio
async def test_create_single_feature_plot_gene_branch(minimal_spatial_adata):
    fig = await viz_feature._create_single_feature_plot(
        minimal_spatial_adata,
        VisualizationParameters(plot_type="feature", basis="spatial", feature="gene_0"),
        "gene_0",
        "spatial",
        minimal_spatial_adata.obsm["spatial"],
    )
    assert fig is not None
    plt.close(fig)


@pytest.mark.asyncio
async def test_create_single_feature_plot_categorical_obs_branch(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obs["cluster"] = pd.Categorical(["A"] * (adata.n_obs // 2) + ["B"] * (adata.n_obs - adata.n_obs // 2))

    fig = await viz_feature._create_single_feature_plot(
        adata,
        VisualizationParameters(plot_type="feature", basis="spatial", feature="cluster"),
        "cluster",
        "spatial",
        adata.obsm["spatial"],
    )
    assert fig is not None
    plt.close(fig)


@pytest.mark.asyncio
async def test_create_multi_feature_plot_cleans_temp_column(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    fig = await viz_feature._create_multi_feature_plot(
        adata,
        VisualizationParameters(plot_type="feature", basis="spatial", feature=["gene_0", "gene_1"]),
        context=None,
        features=["gene_0", "gene_1"],
        basis="spatial",
        coords=adata.obsm["spatial"],
    )
    assert fig is not None
    assert "_feature_viz_temp_99" not in adata.obs.columns
    plt.close(fig)
