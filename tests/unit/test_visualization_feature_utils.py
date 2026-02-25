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


@pytest.mark.asyncio
async def test_create_feature_visualization_requires_spatial_coordinates(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    del adata.obsm["spatial"]

    with pytest.raises(DataNotFoundError, match="Spatial coordinates not found"):
        await viz_feature.create_feature_visualization(
            adata,
            VisualizationParameters(plot_type="feature", basis="spatial", feature="gene_0"),
        )


@pytest.mark.asyncio
async def test_create_feature_visualization_umap_missing_when_not_computable(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obsm.pop("X_umap", None)
    monkeypatch.setattr(viz_feature, "ensure_umap", lambda _adata: False)

    with pytest.raises(DataNotFoundError, match="UMAP embedding not found"):
        await viz_feature.create_feature_visualization(
            adata,
            VisualizationParameters(plot_type="feature", basis="umap", feature="gene_0"),
            context=DummyCtx(),
        )


@pytest.mark.asyncio
async def test_create_feature_visualization_requires_pca_embedding(minimal_spatial_adata):
    with pytest.raises(DataNotFoundError, match="PCA embedding not found"):
        await viz_feature.create_feature_visualization(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="feature", basis="pca", feature="gene_0"),
            context=DummyCtx(),
        )


@pytest.mark.asyncio
async def test_create_feature_visualization_routes_multi_feature_path(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    sentinel = object()

    async def _validated(*_args, **_kwargs):
        return ["gene_0", "gene_1"]

    async def _multi(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(viz_feature, "get_validated_features", _validated)
    monkeypatch.setattr(viz_feature, "_create_multi_feature_plot", _multi)

    out = await viz_feature.create_feature_visualization(
        minimal_spatial_adata.copy(),
        VisualizationParameters(
            plot_type="feature",
            basis="spatial",
            feature=["gene_0", "gene_1"],
        ),
        context=DummyCtx(),
    )
    assert out is sentinel


@pytest.mark.asyncio
async def test_create_single_feature_plot_numeric_obs_branch_hides_axes(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    adata.obs["score"] = np.linspace(0, 1, adata.n_obs)
    adata.obsm["X_pca"] = np.column_stack([np.arange(adata.n_obs), np.arange(adata.n_obs)])

    fig = await viz_feature._create_single_feature_plot(
        adata,
        VisualizationParameters(
            plot_type="feature",
            basis="pca",
            feature="score",
            show_colorbar=False,
            show_axes=False,
        ),
        "score",
        "pca",
        adata.obsm["X_pca"],
    )
    assert fig is not None
    assert fig.axes[0].get_title() == "score (pca)"
    assert not fig.axes[0].axison
    plt.close(fig)


@pytest.mark.asyncio
async def test_create_multi_feature_plot_umap_handles_gene_and_numeric_obs(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    adata.obsm["X_umap"] = np.column_stack([np.arange(adata.n_obs), np.arange(adata.n_obs)])
    adata.obs["score"] = np.linspace(0, 10, adata.n_obs)

    fig = await viz_feature._create_multi_feature_plot(
        adata,
        VisualizationParameters(
            plot_type="feature",
            basis="umap",
            feature=["gene_0", "score"],
            show_colorbar=False,
            show_axes=False,
            color_scale="log",
        ),
        context=None,
        features=["gene_0", "score"],
        basis="umap",
        coords=adata.obsm["X_umap"],
    )

    assert fig is not None
    assert "_feature_viz_temp_99" not in adata.obs.columns
    plt.close(fig)


@pytest.mark.asyncio
async def test_create_lr_pairs_visualization_requires_available_pairs(
    minimal_spatial_adata,
):
    with pytest.raises(DataNotFoundError, match="None of the specified LR pairs found"):
        await viz_feature._create_lr_pairs_visualization(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="feature", basis="spatial"),
            context=DummyCtx(),
            lr_pairs=[("LIG", "REC")],
            basis="spatial",
            coords=minimal_spatial_adata.obsm["spatial"],
        )


@pytest.mark.asyncio
async def test_create_lr_pairs_visualization_limits_pairs_and_reports_titles(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    adata.obsm["X_umap"] = np.column_stack([np.arange(adata.n_obs), np.arange(adata.n_obs)])
    ctx = DummyCtx()
    lr_pairs = [
        ("gene_0", "gene_1"),
        ("gene_2", "gene_3"),
        ("gene_4", "gene_5"),
        ("gene_6", "gene_7"),
        ("gene_8", "gene_9"),
    ]

    fig = await viz_feature._create_lr_pairs_visualization(
        adata,
        VisualizationParameters(
            plot_type="feature",
            basis="umap",
            show_colorbar=False,
            show_correlation_stats=False,
            correlation_method="kendall",
            color_scale="sqrt",
        ),
        context=ctx,
        lr_pairs=lr_pairs,
        basis="umap",
        coords=adata.obsm["X_umap"],
    )

    titles = [ax.get_title() for ax in fig.axes]
    assert any("Too many LR pairs" in msg for msg in ctx.infos)
    assert any("Visualizing 4 LR pairs" in msg for msg in ctx.infos)
    assert any(" vs " in title for title in titles)
    assert "_lr_viz_temp_99" not in adata.obs.columns
    plt.close(fig)
