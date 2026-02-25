"""Unit tests for deconvolution visualization retrieval and routing."""

from __future__ import annotations

import numpy as np
import pytest

from chatspatial.models.data import VisualizationParameters
from chatspatial.tools.visualization import deconvolution as viz_deconv
from chatspatial.utils.exceptions import DataNotFoundError, ParameterError


class DummyCtx:
    def __init__(self):
        self.infos: list[str] = []
        self.warnings: list[str] = []

    async def info(self, msg: str):
        self.infos.append(msg)

    async def warning(self, msg: str):
        self.warnings.append(msg)


def _add_deconv_metadata(
    adata,
    method: str,
    *,
    proportions_key: str,
    cell_types: list[str] | None = None,
    dominant_type_key: str | None = None,
) -> None:
    adata.uns[f"deconvolution_{method}_metadata"] = {
        "parameters": {
            "proportions_key": proportions_key,
            "cell_types": cell_types,
            "dominant_type_key": dominant_type_key,
        }
    }


def test_get_available_methods_prefers_metadata_then_fallback(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.uns["deconvolution_a_metadata"] = {"parameters": {}}
    adata.uns["deconvolution_b_metadata"] = {"parameters": {}}
    adata.obsm["deconvolution_legacy"] = np.zeros((adata.n_obs, 2))

    methods = viz_deconv._get_available_methods(adata)
    assert set(methods) == {"a", "b"}

    adata2 = minimal_spatial_adata.copy()
    adata2.obsm["deconvolution_rctd"] = np.zeros((adata2.n_obs, 2))
    assert viz_deconv._get_available_methods(adata2) == ["rctd"]


@pytest.mark.asyncio
async def test_get_deconvolution_data_requires_existing_results(minimal_spatial_adata):
    with pytest.raises(DataNotFoundError, match="No deconvolution results found"):
        await viz_deconv.get_deconvolution_data(minimal_spatial_adata)


@pytest.mark.asyncio
async def test_get_deconvolution_data_requires_method_when_multiple(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.uns["deconvolution_m1_metadata"] = {"parameters": {}}
    adata.uns["deconvolution_m2_metadata"] = {"parameters": {}}

    with pytest.raises(ParameterError, match="Multiple deconvolution results"):
        await viz_deconv.get_deconvolution_data(adata)


@pytest.mark.asyncio
async def test_get_deconvolution_data_validates_explicit_method(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.uns["deconvolution_m1_metadata"] = {"parameters": {}}

    with pytest.raises(DataNotFoundError, match="Deconvolution 'missing' not found"):
        await viz_deconv.get_deconvolution_data(adata, method="missing")


@pytest.mark.asyncio
async def test_get_deconvolution_data_auto_select_and_context_info(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obsm["deconvolution_rctd"] = np.array([[0.8, 0.2]] * adata.n_obs)
    adata.uns["deconvolution_rctd_cell_types"] = ["T", "B"]
    adata.obs["dominant_celltype_rctd"] = ["T"] * adata.n_obs
    ctx = DummyCtx()

    out = await viz_deconv.get_deconvolution_data(adata, method=None, context=ctx)

    assert out.method == "rctd"
    assert out.proportions_key == "deconvolution_rctd"
    assert out.cell_types == ["T", "B"]
    assert out.dominant_type_key == "dominant_celltype_rctd"
    assert any("Auto-selected deconvolution method: rctd" in m for m in ctx.infos)


@pytest.mark.asyncio
async def test_get_deconvolution_data_reads_metadata_keys(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obsm["custom_props"] = np.array([[0.6, 0.4]] * adata.n_obs)
    adata.obs["custom_dom"] = ["A"] * adata.n_obs
    _add_deconv_metadata(
        adata,
        "mock",
        proportions_key="custom_props",
        cell_types=["A", "B"],
        dominant_type_key="custom_dom",
    )

    out = await viz_deconv.get_deconvolution_data(adata, method="mock")
    assert out.proportions_key == "custom_props"
    assert out.cell_types == ["A", "B"]
    assert out.dominant_type_key == "custom_dom"
    assert list(out.proportions.columns) == ["A", "B"]


@pytest.mark.asyncio
async def test_get_deconvolution_data_fallbacks_to_generic_cell_types_with_warning(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    adata.obsm["deconvolution_fallback"] = np.array([[0.5, 0.3, 0.2]] * adata.n_obs)
    adata.uns["deconvolution_fallback_metadata"] = {"parameters": {}}
    ctx = DummyCtx()

    out = await viz_deconv.get_deconvolution_data(adata, method="fallback", context=ctx)

    assert out.cell_types == ["CellType_0", "CellType_1", "CellType_2"]
    assert out.dominant_type_key is None
    assert any("Cell type names not found" in m for m in ctx.warnings)


@pytest.mark.asyncio
async def test_create_deconvolution_visualization_routes_aliases(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    params = VisualizationParameters(plot_type="deconvolution", subtype="dominant")

    sentinel = object()
    calls: dict[str, bool] = {}

    async def fake_dominant(*_args, **_kwargs):
        calls["dominant"] = True
        return sentinel

    monkeypatch.setattr(viz_deconv, "_create_dominant_celltype_map", fake_dominant)

    out = await viz_deconv.create_deconvolution_visualization(adata, params)
    assert out is sentinel
    assert calls["dominant"] is True


@pytest.mark.asyncio
async def test_create_deconvolution_visualization_unknown_subtype_error(
    minimal_spatial_adata,
):
    with pytest.raises(ParameterError, match="Unknown deconvolution visualization type"):
        await viz_deconv.create_deconvolution_visualization(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="deconvolution", subtype="mystery"),
        )
