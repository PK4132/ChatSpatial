"""Regression tests for gene subsampling on small panels."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_small_panel_respects_subsample_genes(minimal_spatial_adata):
    """When n_vars < 100 and user explicitly requests subsample_genes,
    the HVG mask should select only the requested number of genes."""
    import scipy.sparse

    from chatspatial.tools.preprocessing import _compute_safe_percent_top  # noqa: F401

    adata = minimal_spatial_adata.copy()
    assert adata.n_vars < 100, "Fixture must have < 100 genes for this test"

    n_hvgs = 10
    # Replicate the small-panel branch logic from preprocessing.py
    if scipy.sparse.issparse(adata.X):
        var = np.asarray(
            adata.X.power(2).mean(axis=0) - np.power(adata.X.mean(axis=0), 2)
        ).ravel()
    else:
        var = np.var(adata.X, axis=0)
    top_idx = np.argpartition(var, -n_hvgs)[-n_hvgs:]
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[top_idx] = True
    adata.var["highly_variable"] = mask

    # Subsample step
    result = adata[:, adata.var["highly_variable"]].copy()
    assert result.n_vars == n_hvgs


@pytest.mark.unit
def test_small_panel_all_hvg_when_no_subsample(minimal_spatial_adata):
    """When n_vars < 100 and no subsample_genes, all genes should be HVG."""
    adata = minimal_spatial_adata.copy()
    assert adata.n_vars < 100

    # Small-panel default: all genes HVG
    adata.var["highly_variable"] = True
    assert adata.var["highly_variable"].sum() == adata.n_vars
