"""Contract tests for annotation and spatial domain parametric key generation."""

from __future__ import annotations

import pytest

from chatspatial.tools.annotation import _build_annotation_suffix
from chatspatial.tools.spatial_domains import _build_domain_suffix


# ---------------------------------------------------------------------------
# _build_annotation_suffix
# ---------------------------------------------------------------------------


class TestBuildAnnotationSuffix:
    def test_non_reference_method_uses_method_only(self):
        assert _build_annotation_suffix("sctype", None) == "sctype"
        assert _build_annotation_suffix("cellassign", None) == "cellassign"
        assert _build_annotation_suffix("mllmcelltype", None) == "mllmcelltype"

    def test_reference_method_without_ref_uses_method_only(self):
        """tangram/scanvi/singler without reference_data_id fall back to method."""
        assert _build_annotation_suffix("tangram", None) == "tangram"
        assert _build_annotation_suffix("scanvi", None) == "scanvi"
        assert _build_annotation_suffix("singler", None) == "singler"

    def test_reference_method_with_ref_encodes_ref_id(self):
        assert _build_annotation_suffix("tangram", "ref_pbmc") == "tangram_ref_pbmc"
        assert _build_annotation_suffix("scanvi", "atlas_v2") == "scanvi_atlas_v2"
        assert _build_annotation_suffix("singler", "hpca") == "singler_hpca"

    def test_different_refs_produce_different_suffixes(self):
        s1 = _build_annotation_suffix("scanvi", "refA")
        s2 = _build_annotation_suffix("scanvi", "refB")
        assert s1 != s2

    def test_non_reference_method_ignores_ref_id(self):
        """sctype with a ref_id (shouldn't happen, but be defensive)."""
        assert _build_annotation_suffix("sctype", "some_ref") == "sctype"


# ---------------------------------------------------------------------------
# _build_domain_suffix
# ---------------------------------------------------------------------------


class TestBuildDomainSuffix:
    def test_leiden_encodes_resolution(self):
        assert _build_domain_suffix("leiden", 0.5, 7) == "leiden_res0_5"
        assert _build_domain_suffix("leiden", 1.0, 7) == "leiden_res1_0"
        assert _build_domain_suffix("leiden", 0.3, 7) == "leiden_res0_3"

    def test_louvain_encodes_resolution(self):
        assert _build_domain_suffix("louvain", 0.8, 5) == "louvain_res0_8"

    def test_non_clustering_encodes_n_domains(self):
        assert _build_domain_suffix("spagcn", 0.5, 7) == "spagcn_n7"
        assert _build_domain_suffix("stagate", 0.5, 10) == "stagate_n10"
        assert _build_domain_suffix("graphst", 0.5, 5) == "graphst_n5"
        assert _build_domain_suffix("banksy", 0.5, 12) == "banksy_n12"

    def test_different_resolutions_produce_different_suffixes(self):
        s1 = _build_domain_suffix("leiden", 0.5, 7)
        s2 = _build_domain_suffix("leiden", 1.0, 7)
        assert s1 != s2

    def test_different_n_domains_produce_different_suffixes(self):
        s1 = _build_domain_suffix("spagcn", 0.5, 5)
        s2 = _build_domain_suffix("spagcn", 0.5, 10)
        assert s1 != s2

    def test_different_methods_same_params_produce_different_suffixes(self):
        s1 = _build_domain_suffix("leiden", 0.5, 7)
        s2 = _build_domain_suffix("louvain", 0.5, 7)
        assert s1 != s2


# ---------------------------------------------------------------------------
# Coexistence in data manager
# ---------------------------------------------------------------------------


@pytest.fixture
def dm(minimal_spatial_adata):
    from chatspatial.spatial_mcp_adapter import DefaultSpatialDataManager

    mgr = DefaultSpatialDataManager()
    mgr.data_store["d1"] = {"adata": minimal_spatial_adata, "results": {}}
    return mgr


@pytest.mark.asyncio
async def test_annotation_refs_coexist_in_cache(dm):
    """scanvi + refA and scanvi + refB must coexist."""
    await dm.save_result("d1", "annotation_scanvi_refA", {"ref": "A"})
    await dm.save_result("d1", "annotation_scanvi_refB", {"ref": "B"})

    r1 = await dm.get_result("d1", "annotation_scanvi_refA")
    r2 = await dm.get_result("d1", "annotation_scanvi_refB")
    assert r1["ref"] == "A"
    assert r2["ref"] == "B"


@pytest.mark.asyncio
async def test_domain_resolutions_coexist_in_cache(dm):
    """leiden res0.5 and leiden res1.0 must coexist."""
    await dm.save_result("d1", "spatial_domains_leiden_res0_5", {"res": 0.5})
    await dm.save_result("d1", "spatial_domains_leiden_res1_0", {"res": 1.0})

    r1 = await dm.get_result("d1", "spatial_domains_leiden_res0_5")
    r2 = await dm.get_result("d1", "spatial_domains_leiden_res1_0")
    assert r1["res"] == 0.5
    assert r2["res"] == 1.0
