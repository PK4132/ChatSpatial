"""Unit tests for deconvolution base utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chatspatial.tools.deconvolution.base import (
    MethodConfig,
    _prepare_counts,
    create_deconvolution_stats,
)


class DummyCtx:
    async def warning(self, msg: str):
        return None


def test_method_config_extract_kwargs_maps_and_adds_gpu_flag():
    class Params:
        flashdeconv_sketch_dim = 256
        flashdeconv_lambda_spatial = 1234.0
        use_gpu = True

    cfg = MethodConfig(
        module_name="flashdeconv",
        dependencies=("flashdeconv",),
        supports_gpu=True,
        param_mapping=(
            ("flashdeconv_sketch_dim", "sketch_dim"),
            ("flashdeconv_lambda_spatial", "lambda_spatial"),
        ),
    )

    kwargs = cfg.extract_kwargs(Params())
    assert kwargs["sketch_dim"] == 256
    assert kwargs["lambda_spatial"] == 1234.0
    assert kwargs["use_gpu"] is True


def test_create_deconvolution_stats_has_consistent_summary_fields():
    proportions = pd.DataFrame(
        {
            "T": [0.9, 0.2, 0.1],
            "B": [0.1, 0.8, 0.9],
        },
        index=["spot_1", "spot_2", "spot_3"],
    )
    stats = create_deconvolution_stats(
        proportions=proportions,
        common_genes=["g1", "g2", "g3"],
        method="flashdeconv",
        device="CPU",
    )

    assert stats["method"] == "flashdeconv"
    assert stats["n_spots"] == 3
    assert stats["n_cell_types"] == 2
    assert stats["genes_used"] == 3
    assert sum(stats["dominant_types"].values()) == 3


@pytest.mark.asyncio
async def test_prepare_counts_prefers_raw_and_preserves_obsm(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    raw = adata.copy()
    raw.var["raw_marker"] = "keep"
    adata.raw = raw
    adata.obsm["extra"] = np.ones((adata.n_obs, 2))

    out = await _prepare_counts(adata, "Spatial", DummyCtx(), require_int_dtype=False)

    assert out.n_obs == adata.n_obs
    assert "extra" in out.obsm
    assert "raw_marker" in out.var.columns


@pytest.mark.asyncio
async def test_prepare_counts_converts_integer_like_data_to_int32(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.X = np.rint(np.asarray(adata.X)).astype(np.float64)

    out = await _prepare_counts(adata, "Spatial", DummyCtx(), require_int_dtype=True)
    assert out.X.dtype == np.int32

