"""Unit tests for condition_comparison low-level utilities."""

from __future__ import annotations

import numpy as np
import pytest

from chatspatial.tools.condition_comparison import _create_pseudobulk
from chatspatial.utils.exceptions import DataError


def test_create_pseudobulk_aggregates_per_sample(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obs["condition"] = ["treated"] * 30 + ["control"] * 30
    adata.obs["sample"] = ["s1"] * 15 + ["s2"] * 15 + ["s3"] * 15 + ["s4"] * 15

    counts_df, metadata_df, cell_counts = _create_pseudobulk(
        adata=adata,
        raw_X=adata.X,
        var_names=adata.var_names,
        sample_key="sample",
        condition_key="condition",
        min_cells_per_sample=10,
    )

    assert counts_df.shape[0] == 4
    assert list(metadata_df.columns) == ["condition"]
    assert metadata_df.loc["s1", "condition"] == "treated"
    assert metadata_df.loc["s4", "condition"] == "control"
    assert cell_counts == {"s1": 15, "s2": 15, "s3": 15, "s4": 15}


def test_create_pseudobulk_raises_when_no_sample_meets_min_cells(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obs["condition"] = ["treated"] * 30 + ["control"] * 30
    adata.obs["sample"] = [f"s{i}" for i in range(60)]  # one cell per sample

    with pytest.raises(DataError, match="No samples have >="):
        _create_pseudobulk(
            adata=adata,
            raw_X=adata.X,
            var_names=adata.var_names,
            sample_key="sample",
            condition_key="condition",
            min_cells_per_sample=2,
        )


def test_create_pseudobulk_cell_type_filter_only_keeps_selected_type(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obs["condition"] = ["treated"] * 30 + ["control"] * 30
    adata.obs["sample"] = ["s1"] * 15 + ["s2"] * 15 + ["s3"] * 15 + ["s4"] * 15
    adata.obs["cell_type"] = ["T"] * 20 + ["B"] * 40

    counts_df, metadata_df, cell_counts = _create_pseudobulk(
        adata=adata,
        raw_X=adata.X,
        var_names=adata.var_names,
        sample_key="sample",
        condition_key="condition",
        cell_type="T",
        cell_type_key="cell_type",
        min_cells_per_sample=5,
    )

    # Only samples containing enough T cells survive.
    assert counts_df.shape[0] >= 1
    assert set(metadata_df["condition"]).issubset({"treated", "control"})
    assert all(v >= 5 for v in cell_counts.values())

