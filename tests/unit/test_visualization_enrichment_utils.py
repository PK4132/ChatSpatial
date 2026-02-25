"""Unit tests for enrichment visualization utility contracts."""

from __future__ import annotations

from types import ModuleType

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from chatspatial.models.data import VisualizationParameters
from chatspatial.tools.visualization import enrichment as viz_enrich
from chatspatial.utils.exceptions import DataNotFoundError, ParameterError, ProcessingError


class DummyCtx:
    def __init__(self):
        self.infos: list[str] = []

    async def info(self, msg: str):
        self.infos.append(msg)


def test_get_score_columns_fallback_suffix_search(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obs["Wnt_score"] = 0.1
    adata.obs["NotScore"] = 0.2
    assert viz_enrich._get_score_columns(adata) == ["Wnt_score"]


def test_resolve_score_column_validates_and_defaults(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obs["A_score"] = 0.2
    cols = ["A_score"]
    assert viz_enrich._resolve_score_column(adata, None, cols) == "A_score"
    assert viz_enrich._resolve_score_column(adata, "A", cols) == "A_score"

    with pytest.raises(DataNotFoundError, match="Score column 'missing' not found"):
        viz_enrich._resolve_score_column(adata, "missing", cols)


@pytest.mark.asyncio
async def test_create_enrichment_visualization_routes_violin(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["A_score"] = 0.3
    sentinel = object()

    def _fake_violin(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(viz_enrich, "_create_enrichment_violin", _fake_violin)
    out = await viz_enrich._create_enrichment_visualization(
        adata,
        VisualizationParameters(plot_type="enrichment", subtype="violin"),
        context=DummyCtx(),
    )
    assert out is sentinel


@pytest.mark.asyncio
async def test_create_enrichment_visualization_requires_scores(minimal_spatial_adata):
    with pytest.raises(DataNotFoundError, match="No enrichment scores found"):
        await viz_enrich._create_enrichment_visualization(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="enrichment", subtype="violin"),
            context=DummyCtx(),
        )


@pytest.mark.asyncio
async def test_create_pathway_enrichment_visualization_routes_spatial(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    sentinel = object()

    async def _fake_router(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(viz_enrich, "_create_enrichment_visualization", _fake_router)
    out = await viz_enrich.create_pathway_enrichment_visualization(
        minimal_spatial_adata,
        VisualizationParameters(plot_type="enrichment", subtype="spatial_score"),
        context=DummyCtx(),
    )
    assert out is sentinel


@pytest.mark.asyncio
async def test_create_pathway_enrichment_visualization_requires_results_key(
    minimal_spatial_adata,
):
    with pytest.raises(DataNotFoundError, match="GSEA results not found"):
        await viz_enrich.create_pathway_enrichment_visualization(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="enrichment", subtype="barplot"),
            context=DummyCtx(),
        )


def test_gsea_results_to_dataframe_and_term_resolution():
    df = viz_enrich._gsea_results_to_dataframe(
        {"PathA": {"NES": 1.2, "pval": 0.01}, "PathB": {"NES": -1.1, "pval": 0.2}}
    )
    assert set(df["Term"]) == {"PathA", "PathB"}

    df2 = pd.DataFrame({"pathway": ["A"], "pval": [0.1]})
    viz_enrich._ensure_term_column(df2)
    assert "Term" in df2.columns


def test_find_pvalue_column_priority():
    df = pd.DataFrame({"Adjusted P-value": [0.1], "pval": [0.2]})
    assert viz_enrich._find_pvalue_column(df) == "Adjusted P-value"
    df2 = pd.DataFrame({"fdr": [0.1]})
    assert viz_enrich._find_pvalue_column(df2) == "fdr"


def test_ensure_term_column_raises_when_missing():
    df = pd.DataFrame({"x": [1], "y": [2]})
    with pytest.raises(DataNotFoundError, match="No pathway/term column found"):
        viz_enrich._ensure_term_column(df)


def test_create_enrichmap_single_score_requires_feature(minimal_spatial_adata):
    with pytest.raises(DataNotFoundError, match="Feature parameter required"):
        viz_enrich._create_enrichmap_single_score(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="enrichment", subtype="spatial_score"),
            "sample_1",
            em=object(),
            context=None,
        )


def test_create_enrichmap_spatial_requires_dependency(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    modules = __import__("sys").modules
    if "enrichmap" in modules:
        monkeypatch.delitem(modules, "enrichmap", raising=False)

    # Force import to fail regardless of environment state
    import builtins

    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "enrichmap":
            raise ImportError("missing enrichmap")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ProcessingError, match="requires EnrichMap"):
        viz_enrich._create_enrichmap_spatial(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="enrichment", subtype="spatial_score"),
            score_cols=["A_score"],
            context=None,
        )


def test_create_gsea_barplot_wraps_gseapy_errors(monkeypatch: pytest.MonkeyPatch):
    fake_gp = ModuleType("gseapy")

    def _boom(**_kwargs):
        raise RuntimeError("bad data")

    fake_gp.barplot = _boom
    monkeypatch.setitem(__import__("sys").modules, "gseapy", fake_gp)

    with pytest.raises(ProcessingError, match="gseapy.barplot failed"):
        viz_enrich._create_gsea_barplot(
            {"PathA": {"Adjusted P-value": 0.1}},
            VisualizationParameters(plot_type="enrichment", subtype="barplot"),
        )

    plt.close("all")
