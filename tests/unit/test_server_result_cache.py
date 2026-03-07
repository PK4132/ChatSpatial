"""Regression tests for server-layer result caching contracts."""

from __future__ import annotations

import pytest

from chatspatial.spatial_mcp_adapter import DefaultSpatialDataManager


@pytest.fixture
def dm(minimal_spatial_adata):
    """DataManager with one dataset pre-loaded."""
    mgr = DefaultSpatialDataManager()
    mgr.data_store["d1"] = {"adata": minimal_spatial_adata, "results": {}}
    mgr.data_store["d2"] = {"adata": minimal_spatial_adata.copy(), "results": {}}
    return mgr


@pytest.mark.asyncio
async def test_parametric_key_prevents_overwrite(dm):
    """Two comparisons on the same dataset must coexist, not overwrite."""
    await dm.save_result("d1", "condition_comparison_A_vs_B", {"comparison": "A_vs_B"})
    await dm.save_result("d1", "condition_comparison_C_vs_D", {"comparison": "C_vs_D"})

    r1 = await dm.get_result("d1", "condition_comparison_A_vs_B")
    r2 = await dm.get_result("d1", "condition_comparison_C_vs_D")

    assert r1["comparison"] == "A_vs_B"
    assert r2["comparison"] == "C_vs_D"


@pytest.mark.asyncio
async def test_single_key_overwrites(dm):
    """Sanity: same key DOES overwrite (expected for idempotent operations)."""
    await dm.save_result("d1", "preprocessing", {"v": 1})
    await dm.save_result("d1", "preprocessing", {"v": 2})

    result = await dm.get_result("d1", "preprocessing")
    assert result["v"] == 2


@pytest.mark.asyncio
async def test_registration_saved_to_both_datasets(dm):
    """Registration result must be retrievable from both source and target."""
    reg_result = {"method": "paste", "source_id": "d1", "target_id": "d2"}

    # Simulate what server.py now does
    await dm.save_result("d1", "registration", reg_result)
    await dm.save_result("d2", "registration", reg_result)

    r1 = await dm.get_result("d1", "registration")
    r2 = await dm.get_result("d2", "registration")
    assert r1 == r2 == reg_result


@pytest.mark.asyncio
async def test_method_keyed_tools_coexist(dm):
    """Different methods for the same tool on the same dataset must coexist."""
    await dm.save_result("d1", "spatial_genes_flashs", {"method": "flashs", "n": 10})
    await dm.save_result("d1", "spatial_genes_sparkx", {"method": "sparkx", "n": 8})
    await dm.save_result("d1", "velocity_scvelo_stochastic", {"method": "scvelo"})
    await dm.save_result("d1", "velocity_velovi", {"method": "velovi"})

    assert (await dm.get_result("d1", "spatial_genes_flashs"))["method"] == "flashs"
    assert (await dm.get_result("d1", "spatial_genes_sparkx"))["method"] == "sparkx"
    assert (await dm.get_result("d1", "velocity_scvelo_stochastic"))[
        "method"
    ] == "scvelo"
    assert (await dm.get_result("d1", "velocity_velovi"))["method"] == "velovi"


@pytest.mark.asyncio
async def test_analysis_type_keyed_tools_coexist(dm):
    """Spatial statistics keyed by analysis_type must coexist."""
    await dm.save_result("d1", "spatial_statistics_moran", {"type": "moran", "i": 0.3})
    await dm.save_result("d1", "spatial_statistics_getis_ord", {"type": "getis_ord"})

    r_moran = await dm.get_result("d1", "spatial_statistics_moran")
    r_getis = await dm.get_result("d1", "spatial_statistics_getis_ord")

    assert r_moran["type"] == "moran"
    assert r_getis["type"] == "getis_ord"


@pytest.mark.asyncio
async def test_de_keys_coexist_across_methods_and_comparisons(dm):
    """DE results with different methods/comparisons must coexist."""
    await dm.save_result("d1", "de_wilcoxon_all_groups", {"m": "wilcoxon", "c": "all"})
    await dm.save_result("d1", "de_wilcoxon_T_vs_B", {"m": "wilcoxon", "c": "T_vs_B"})
    await dm.save_result("d1", "de_pydeseq2_T_vs_B", {"m": "pydeseq2", "c": "T_vs_B"})

    r1 = await dm.get_result("d1", "de_wilcoxon_all_groups")
    r2 = await dm.get_result("d1", "de_wilcoxon_T_vs_B")
    r3 = await dm.get_result("d1", "de_pydeseq2_T_vs_B")

    assert r1["c"] == "all"
    assert r2["c"] == "T_vs_B" and r2["m"] == "wilcoxon"
    assert r3["c"] == "T_vs_B" and r3["m"] == "pydeseq2"


@pytest.mark.asyncio
async def test_velocity_trajectory_deconv_cnv_coexist(dm):
    """Velocity, trajectory, deconvolution, CNV with different params must coexist."""
    # Velocity: scvelo stochastic vs dynamical
    await dm.save_result("d1", "velocity_scvelo_stochastic", {"mode": "stochastic"})
    await dm.save_result("d1", "velocity_scvelo_dynamical", {"mode": "dynamical"})
    # Trajectory: cellrank with different spatial weights
    await dm.save_result("d1", "trajectory_cellrank_sw0_50", {"sw": 0.5})
    await dm.save_result("d1", "trajectory_cellrank_sw0_00", {"sw": 0.0})
    # Deconvolution: same method, different references
    await dm.save_result("d1", "deconvolution_flashdeconv_ref_a", {"ref": "a"})
    await dm.save_result("d1", "deconvolution_flashdeconv_ref_b", {"ref": "b"})
    # CNV: different reference categories
    await dm.save_result("d1", "cnv_infercnvpy_immune", {"cats": "immune"})
    await dm.save_result("d1", "cnv_infercnvpy_stromal", {"cats": "stromal"})

    assert (await dm.get_result("d1", "velocity_scvelo_stochastic"))[
        "mode"
    ] == "stochastic"
    assert (await dm.get_result("d1", "velocity_scvelo_dynamical"))[
        "mode"
    ] == "dynamical"
    assert (await dm.get_result("d1", "trajectory_cellrank_sw0_50"))["sw"] == 0.5
    assert (await dm.get_result("d1", "trajectory_cellrank_sw0_00"))["sw"] == 0.0
    assert (await dm.get_result("d1", "deconvolution_flashdeconv_ref_a"))["ref"] == "a"
    assert (await dm.get_result("d1", "deconvolution_flashdeconv_ref_b"))["ref"] == "b"
    assert (await dm.get_result("d1", "cnv_infercnvpy_immune"))["cats"] == "immune"
    assert (await dm.get_result("d1", "cnv_infercnvpy_stromal"))["cats"] == "stromal"
