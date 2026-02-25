"""Integration contracts for deconvolution module entrypoints."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import chatspatial.tools.deconvolution as deconv_module
from chatspatial.models.analysis import DeconvolutionResult
from chatspatial.models.data import DeconvolutionParameters
from chatspatial.tools.deconvolution import (
    _store_results,
    deconvolve_spatial_data,
)
from chatspatial.tools.deconvolution.base import PreparedDeconvolutionData
from chatspatial.utils.exceptions import DataError, ParameterError


class DummyCtx:
    def __init__(self, datasets: dict[str, object]):
        self.datasets = datasets
        self.updated: dict[str, object] = {}

    async def get_adata(self, data_id: str):
        return self.datasets[data_id]

    async def set_adata(self, data_id: str, adata):
        self.updated[data_id] = adata


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deconvolve_spatial_data_requires_nonempty_data_id():
    params = DeconvolutionParameters(
        method="flashdeconv",
        reference_data_id="ref",
        cell_type_key="cell_type",
    )
    with pytest.raises(ParameterError, match="Dataset ID cannot be empty"):
        await deconvolve_spatial_data("", DummyCtx({}), params)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deconvolve_spatial_data_requires_reference_data_id(minimal_spatial_adata):
    spatial = minimal_spatial_adata.copy()
    spatial.obs["cell_type"] = ["A"] * 30 + ["B"] * 30
    ctx = DummyCtx({"d1": spatial})

    params = DeconvolutionParameters(
        method="flashdeconv",
        reference_data_id=None,
        cell_type_key="cell_type",
    )
    with pytest.raises(ParameterError, match="requires reference_data_id"):
        await deconvolve_spatial_data("d1", ctx, params)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deconvolve_spatial_data_rejects_empty_spatial_dataset(minimal_spatial_adata):
    spatial = minimal_spatial_adata[:0, :].copy()
    reference = minimal_spatial_adata.copy()
    reference.obs["cell_type"] = ["A"] * 30 + ["B"] * 30
    ctx = DummyCtx({"d1": spatial, "ref": reference})

    params = DeconvolutionParameters(
        method="flashdeconv",
        reference_data_id="ref",
        cell_type_key="cell_type",
    )
    with pytest.raises(DataError, match="contains no observations"):
        await deconvolve_spatial_data("d1", ctx, params)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deconvolve_spatial_data_dispatch_contract_with_mocks(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    spatial = minimal_spatial_adata.copy()
    reference = minimal_spatial_adata.copy()
    reference.obs["cell_type"] = ["A"] * 30 + ["B"] * 30
    ctx = DummyCtx({"d1": spatial, "ref": reference})

    monkeypatch.setattr(deconv_module, "_check_method_availability", lambda *a, **k: None)

    prepared = PreparedDeconvolutionData(
        spatial=spatial,
        reference=reference,
        cell_type_key="cell_type",
        cell_types=["A", "B"],
        common_genes=list(spatial.var_names),
        spatial_coords=spatial.obsm["spatial"],
        ctx=ctx,
    )
    calls: dict[str, object] = {}

    async def fake_prepare_deconvolution(**kwargs):
        calls["prepared"] = True
        return prepared

    def fake_dispatch(data, params, config):
        calls["method"] = params.method
        props = pd.DataFrame(
            {"A": [0.8] * data.spatial.n_obs, "B": [0.2] * data.spatial.n_obs},
            index=data.spatial.obs_names,
        )
        return props, {"n_spots": data.spatial.n_obs, "genes_used": data.n_genes}

    async def fake_store(spatial_adata, proportions, stats, method, data_id, ctx):
        calls["stored"] = (method, data_id, len(proportions))
        return DeconvolutionResult(
            data_id=data_id,
            method=method,
            dominant_type_key=f"dominant_celltype_{method}",
            n_cell_types=2,
            cell_types=["A", "B"],
            proportions_key=f"deconvolution_{method}",
            n_spots=stats["n_spots"],
            genes_used=stats["genes_used"],
        )

    monkeypatch.setattr(deconv_module, "prepare_deconvolution", fake_prepare_deconvolution)
    monkeypatch.setattr(deconv_module, "_dispatch_method", fake_dispatch)
    monkeypatch.setattr(deconv_module, "_store_results", fake_store)

    params = DeconvolutionParameters(
        method="flashdeconv",
        reference_data_id="ref",
        cell_type_key="cell_type",
    )
    result = await deconvolve_spatial_data("d1", ctx, params)

    assert isinstance(result, DeconvolutionResult)
    assert result.method == "flashdeconv"
    assert calls["prepared"] is True
    assert calls["method"] == "flashdeconv"
    assert calls["stored"] == ("flashdeconv", "d1", spatial.n_obs)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_results_persists_expected_keys_and_calls_set_adata(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx({"d1": adata})

    monkeypatch.setattr(deconv_module, "store_analysis_metadata", lambda *a, **k: None)
    monkeypatch.setattr(deconv_module, "export_analysis_result", lambda *a, **k: None)

    proportions = pd.DataFrame(
        {"T": np.full(adata.n_obs, 0.6), "B": np.full(adata.n_obs, 0.4)},
        index=adata.obs_names,
    )
    stats = {"n_spots": adata.n_obs, "genes_used": adata.n_vars}

    result = await _store_results(
        spatial_adata=adata,
        proportions=proportions,
        stats=stats,
        method="flashdeconv",
        data_id="d1",
        ctx=ctx,
    )

    assert isinstance(result, DeconvolutionResult)
    assert result.proportions_key == "deconvolution_flashdeconv"
    assert "deconvolution_flashdeconv" in adata.obsm
    assert "dominant_celltype_flashdeconv" in adata.obs
    assert "d1" in ctx.updated
