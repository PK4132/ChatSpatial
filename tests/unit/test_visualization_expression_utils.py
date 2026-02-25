"""Unit tests for expression visualization routing and contracts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chatspatial.models.data import VisualizationParameters
from chatspatial.tools.visualization import expression as expr
from chatspatial.utils.exceptions import ParameterError


class DummyCtx:
    def __init__(self):
        self.infos: list[str] = []

    async def info(self, msg: str):
        self.infos.append(msg)


@pytest.mark.asyncio
async def test_create_expression_visualization_routes_heatmap_by_subtype(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    sentinel = object()

    async def _fake_heatmap(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(expr, "_create_heatmap", _fake_heatmap)
    out = await expr.create_expression_visualization(
        minimal_spatial_adata,
        VisualizationParameters(plot_type="expression", subtype="heatmap"),
    )
    assert out is sentinel


@pytest.mark.asyncio
async def test_create_expression_visualization_rejects_invalid_subtype(
    minimal_spatial_adata,
):
    with pytest.raises(ParameterError, match="Invalid expression subtype"):
        await expr.create_expression_visualization(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="expression", subtype="bad"),
        )


@pytest.mark.asyncio
async def test_heatmap_requires_cluster_key(minimal_spatial_adata):
    with pytest.raises(ParameterError, match="Heatmap requires cluster_key"):
        await expr._create_heatmap(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="expression", subtype="heatmap"),
            context=None,
        )


@pytest.mark.asyncio
async def test_heatmap_requires_valid_features(minimal_spatial_adata, monkeypatch):
    adata = minimal_spatial_adata.copy()
    adata.obs["leiden"] = ["0"] * adata.n_obs
    monkeypatch.setattr(expr, "validate_obs_column", lambda *_args, **_kwargs: None)

    async def _empty_features(*_args, **_kwargs):
        return []

    monkeypatch.setattr(expr, "get_validated_features", _empty_features)

    with pytest.raises(ParameterError, match="No valid gene features"):
        await expr._create_heatmap(
            adata,
            VisualizationParameters(
                plot_type="expression", subtype="heatmap", cluster_key="leiden"
            ),
            context=None,
        )


@pytest.mark.asyncio
async def test_dotplot_passes_optional_kwargs(minimal_spatial_adata, monkeypatch):
    adata = minimal_spatial_adata.copy()
    adata.obs["leiden"] = ["0"] * adata.n_obs
    called: dict[str, object] = {}

    monkeypatch.setattr(expr, "validate_obs_column", lambda *_args, **_kwargs: None)

    async def _features(*_args, **_kwargs):
        return ["gene_0", "gene_1"]

    monkeypatch.setattr(expr, "get_validated_features", _features)

    def _dotplot(**kwargs):
        called.update(kwargs)
        plt.figure()
        return None

    monkeypatch.setattr(expr.sc.pl, "dotplot", _dotplot)

    params = VisualizationParameters(
        plot_type="expression",
        subtype="dotplot",
        cluster_key="leiden",
        dotplot_dendrogram=True,
        dotplot_swap_axes=True,
        dotplot_standard_scale="var",
        dotplot_dot_min=0.1,
        dotplot_dot_max=0.9,
        dotplot_smallest_dot=2.0,
    )

    fig = await expr._create_dotplot(adata, params, context=None)
    assert fig is not None
    assert called["groupby"] == "leiden"
    assert called["dendrogram"] is True
    assert called["swap_axes"] is True
    assert called["standard_scale"] == "var"
    assert called["dot_min"] == pytest.approx(0.1)
    assert called["dot_max"] == pytest.approx(0.9)
    assert called["smallest_dot"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_correlation_uses_requested_method_and_logs(minimal_spatial_adata, monkeypatch):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx()
    captured: dict[str, object] = {}

    async def _features(*_args, **_kwargs):
        return ["gene_0", "gene_1", "gene_2"]

    monkeypatch.setattr(expr, "get_validated_features", _features)
    monkeypatch.setattr(
        expr, "get_genes_expression", lambda *_args, **_kwargs: np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    )

    class _Grid:
        def __init__(self):
            self.fig = plt.figure()

    def _clustermap(corr_df, **kwargs):
        captured["corr"] = corr_df
        captured["kwargs"] = kwargs
        return _Grid()

    monkeypatch.setattr(expr.sns, "clustermap", _clustermap)

    fig = await expr._create_correlation(
        adata,
        VisualizationParameters(
            plot_type="expression",
            subtype="correlation",
            correlation_method="spearman",
        ),
        context=ctx,
    )

    assert fig is not None
    assert captured["kwargs"]["fmt"] == ".2f"
    assert any("gene correlation visualization" in m.lower() for m in ctx.infos)
