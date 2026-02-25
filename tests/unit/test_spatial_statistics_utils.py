"""Unit tests for lightweight spatial_statistics helper contracts."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chatspatial.models.data import SpatialStatisticsParameters
from chatspatial.tools import spatial_statistics as ss
from chatspatial.tools.spatial_statistics import (
    _build_results_keys,
    _extract_result_summary,
)
from chatspatial.utils.exceptions import (
    DataNotFoundError,
    ParameterError,
    ProcessingError,
)


class DummyCtx:
    def __init__(self, adata=None, error: Exception | None = None):
        self.adata = adata
        self.error = error

    async def get_adata(self, _data_id: str):
        if self.error is not None:
            raise self.error
        return self.adata

    async def info(self, _msg: str):
        return None


def test_build_results_keys_neighborhood_uses_cluster_dynamic_uns_key():
    keys = _build_results_keys("neighborhood", genes=None, cluster_key="leiden")
    assert keys["uns"] == ["leiden_nhood_enrichment"]
    assert keys["obs"] == []


def test_build_results_keys_local_moran_expands_gene_specific_obs_keys():
    keys = _build_results_keys("local_moran", genes=["gene_a", "gene_b"])
    assert "gene_a_local_morans" in keys["obs"]
    assert "gene_a_lisa_cluster" in keys["obs"]
    assert "gene_b_lisa_pvalue" in keys["obs"]


def test_extract_result_summary_moran_compacts_expected_fields():
    summary = _extract_result_summary(
        {
            "n_genes_analyzed": 12,
            "n_significant": 3,
            "top_highest_autocorrelation": ["g1", "g2", "g3"],
            "mean_morans_i": 0.31,
            "analysis_key": "moranI",
        },
        "moran",
    )
    assert summary["n_features_analyzed"] == 12
    assert summary["n_significant"] == 3
    assert summary["top_features"] == ["g1", "g2", "g3"]
    assert summary["results_key"] == "moranI"


def test_extract_result_summary_local_join_count_aggregates_per_category_stats():
    summary = _extract_result_summary(
        {
            "n_categories": 2,
            "categories": ["A", "B"],
            "per_category_stats": {
                "A": {"n_significant": 4, "n_hotspots": 2},
                "B": {"n_significant": 1, "n_hotspots": 1},
            },
        },
        "local_join_count",
    )
    assert summary["n_features_analyzed"] == 2
    assert summary["n_significant"] == 5
    assert summary["top_features"] == ["A", "B"]
    assert summary["summary_metrics"]["total_significant_clusters"] == 5


def test_build_results_keys_unknown_analysis_returns_empty_key_structure():
    keys = _build_results_keys("unknown_analysis", genes=["g1"], cluster_key="x")
    assert keys == {"obs": [], "var": [], "obsm": [], "uns": []}


def test_extract_result_summary_join_count_and_network_properties_branches():
    join_summary = _extract_result_summary(
        {"bb": 10, "ww": 20, "bw": 5, "J": 35, "p_value": 0.03},
        "join_count",
    )
    assert join_summary["n_features_analyzed"] == 2
    assert join_summary["n_significant"] == 1
    assert join_summary["summary_metrics"]["total_joins"] == 35

    net_summary = _extract_result_summary(
        {"analysis_key": "network_properties", "density": 0.12, "n_cells": 60},
        "network_properties",
    )
    assert net_summary["results_key"] == "network_properties"
    assert net_summary["summary_metrics"] == {"density": 0.12}


def test_get_optimal_n_jobs_respects_requested_and_auto_rules(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    assert ss._get_optimal_n_jobs(500, requested_n_jobs=-1) == 8
    assert ss._get_optimal_n_jobs(500, requested_n_jobs=3) == 3
    assert ss._get_optimal_n_jobs(500, requested_n_jobs=None) == 1
    assert ss._get_optimal_n_jobs(3000, requested_n_jobs=None) == 2
    assert ss._get_optimal_n_jobs(9000, requested_n_jobs=None) == 4


def test_dispatch_analysis_routes_gene_cluster_and_hybrid(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    params = SpatialStatisticsParameters()
    ctx = DummyCtx(minimal_spatial_adata)

    monkeypatch.setattr(
        ss,
        "_test_gene_handler",
        lambda adata, p, c: {"route": "gene", "n_obs": adata.n_obs, "n_neighbors": p.n_neighbors},
        raising=False,
    )
    monkeypatch.setitem(
        ss._ANALYSIS_REGISTRY,
        "_test_gene",
        {"handler": "_test_gene_handler", "signature": "gene", "metadata_keys": {}},
    )
    out_gene = ss._dispatch_analysis("_test_gene", minimal_spatial_adata, params, None, ctx)
    assert out_gene["route"] == "gene"

    monkeypatch.setattr(
        ss,
        "_test_cluster_handler",
        lambda adata, ck, c: {"route": "cluster", "cluster_key": ck, "n_obs": adata.n_obs},
        raising=False,
    )
    monkeypatch.setitem(
        ss._ANALYSIS_REGISTRY,
        "_test_cluster",
        {"handler": "_test_cluster_handler", "signature": "cluster", "metadata_keys": {}},
    )
    out_cluster = ss._dispatch_analysis(
        "_test_cluster", minimal_spatial_adata, params, "leiden", ctx
    )
    assert out_cluster == {"route": "cluster", "cluster_key": "leiden", "n_obs": 60}

    monkeypatch.setattr(
        ss,
        "_test_hybrid_handler",
        lambda adata, ck, p, c: {"route": "hybrid", "cluster_key": ck, "n_neighbors": p.n_neighbors},
        raising=False,
    )
    monkeypatch.setitem(
        ss._ANALYSIS_REGISTRY,
        "_test_hybrid",
        {"handler": "_test_hybrid_handler", "signature": "hybrid", "metadata_keys": {}},
    )
    out_hybrid = ss._dispatch_analysis(
        "_test_hybrid", minimal_spatial_adata, params, "leiden", ctx
    )
    assert out_hybrid["route"] == "hybrid"
    assert out_hybrid["cluster_key"] == "leiden"


@pytest.mark.asyncio
async def test_analyze_spatial_statistics_rejects_invalid_analysis_type():
    params = SpatialStatisticsParameters().model_copy(update={"analysis_type": "invalid"})
    with pytest.raises(ParameterError, match="Unsupported analysis type"):
        await ss.analyze_spatial_statistics("d1", DummyCtx(None), params)


@pytest.mark.asyncio
async def test_analyze_spatial_statistics_rejects_non_positive_neighbors():
    params = SpatialStatisticsParameters().model_copy(update={"n_neighbors": 0})
    with pytest.raises(ParameterError, match="n_neighbors must be positive"):
        await ss.analyze_spatial_statistics("d1", DummyCtx(None), params)


@pytest.mark.asyncio
async def test_analyze_spatial_statistics_requires_cluster_key_for_cluster_analysis(
    minimal_spatial_adata,
):
    params = SpatialStatisticsParameters(analysis_type="neighborhood", cluster_key=None)
    with pytest.raises(ParameterError, match="cluster_key is required"):
        await ss.analyze_spatial_statistics("d1", DummyCtx(minimal_spatial_adata.copy()), params)


@pytest.mark.asyncio
async def test_analyze_spatial_statistics_passes_through_data_not_found():
    params = SpatialStatisticsParameters(analysis_type="moran")
    with pytest.raises(DataNotFoundError, match="missing dataset"):
        await ss.analyze_spatial_statistics(
            "missing",
            DummyCtx(None, error=DataNotFoundError("missing dataset")),
            params,
        )


@pytest.mark.asyncio
async def test_analyze_spatial_statistics_moran_success_path_updates_metadata_and_summary(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    params = SpatialStatisticsParameters(analysis_type="moran", genes=["gene_0"], n_neighbors=7)
    captured: dict[str, object] = {}

    monkeypatch.setattr(ss, "ensure_spatial_neighbors", lambda _a, n_neighs: captured.setdefault("n_neighs", n_neighs))
    monkeypatch.setattr(
        ss,
        "_dispatch_analysis",
        lambda *_args, **_kwargs: {
            "n_genes_analyzed": 1,
            "n_significant": 1,
            "top_highest_autocorrelation": ["gene_0"],
            "mean_morans_i": 0.42,
            "analysis_key": "moranI",
        },
    )

    def _store(_adata, *, analysis_name, method, parameters, results_keys, statistics, **_kwargs):
        captured["analysis_name"] = analysis_name
        captured["method"] = method
        captured["parameters"] = parameters
        captured["results_keys"] = results_keys
        captured["statistics"] = statistics

    monkeypatch.setattr(ss, "store_analysis_metadata", _store)
    monkeypatch.setattr(ss, "export_analysis_result", lambda *_args, **_kwargs: [])

    out = await ss.analyze_spatial_statistics("d1", DummyCtx(adata), params)
    assert out.analysis_type == "moran"
    assert out.n_features_analyzed == 1
    assert out.n_significant == 1
    assert out.top_features == ["gene_0"]
    assert out.summary_metrics["mean_morans_i"] == 0.42
    assert out.results_key == "moranI"
    assert captured["n_neighs"] == 7
    assert captured["analysis_name"] == "spatial_stats_moran"
    assert captured["method"] == "moran"
    assert captured["parameters"] == {"n_neighbors": 7, "genes": ["gene_0"], "n_perms": 10}
    assert captured["results_keys"] == {"obs": [], "var": [], "obsm": [], "uns": ["moranI"]}
    assert captured["statistics"] == {"n_cells": 60, "n_significant": 1}


@pytest.mark.asyncio
async def test_analyze_spatial_statistics_accepts_result_with_dict_method(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["leiden"] = pd.Categorical(["0"] * 30 + ["1"] * 30)
    params = SpatialStatisticsParameters(analysis_type="neighborhood", cluster_key="leiden")
    monkeypatch.setattr(ss, "ensure_spatial_neighbors", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ss, "store_analysis_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ss, "export_analysis_result", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        ss,
        "_dispatch_analysis",
        lambda *_args, **_kwargs: SimpleNamespace(
            dict=lambda: {
                "n_clusters": 2,
                "max_enrichment": 1.2,
                "min_enrichment": -0.7,
                "analysis_key": "leiden_nhood_enrichment",
            }
        ),
    )

    out = await ss.analyze_spatial_statistics("d1", DummyCtx(adata), params)
    assert out.analysis_type == "neighborhood"
    assert out.n_features_analyzed == 2
    assert out.summary_metrics == {"max_enrichment": 1.2, "min_enrichment": -0.7}
    assert out.results_key == "leiden_nhood_enrichment"


@pytest.mark.asyncio
async def test_analyze_spatial_statistics_wraps_invalid_result_format(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    params = SpatialStatisticsParameters(analysis_type="moran")
    monkeypatch.setattr(ss, "ensure_spatial_neighbors", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ss, "_dispatch_analysis", lambda *_args, **_kwargs: object())

    with pytest.raises(ProcessingError, match="Invalid result format"):
        await ss.analyze_spatial_statistics("d1", DummyCtx(adata), params)


def test_analyze_join_count_preserves_parameter_error_for_non_binary_clusters(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cluster3"] = pd.Categorical(["a"] * 20 + ["b"] * 20 + ["c"] * 20)

    monkeypatch.setattr(ss, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ss, "require_spatial_coords", lambda a: a.obsm["spatial"])

    fake_esda_join_counts = ModuleType("esda.join_counts")
    fake_esda_join_counts.Join_Counts = object
    fake_libpysal_weights = ModuleType("libpysal.weights")
    fake_libpysal_weights.KNN = SimpleNamespace(from_array=lambda *_args, **_kwargs: object())
    monkeypatch.setitem(__import__("sys").modules, "esda.join_counts", fake_esda_join_counts)
    monkeypatch.setitem(__import__("sys").modules, "libpysal.weights", fake_libpysal_weights)

    with pytest.raises(ParameterError, match="requires binary data"):
        ss._analyze_join_count(
            adata,
            "cluster3",
            SpatialStatisticsParameters(analysis_type="join_count", n_neighbors=5),
            DummyCtx(adata),
        )


def test_analyze_neighborhood_enrichment_uses_nan_safe_extrema(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["leiden"] = pd.Categorical(["0"] * 30 + ["1"] * 30)

    def _fake_nhood(a, cluster_key):
        del cluster_key
        a.uns["leiden_nhood_enrichment"] = {
            "zscore": np.array([[np.nan, 1.2], [-0.8, np.nan]])
        }

    monkeypatch.setattr(ss.sq.gr, "nhood_enrichment", _fake_nhood)

    out = ss._analyze_neighborhood_enrichment(adata, "leiden", DummyCtx(adata))
    assert out["n_clusters"] == 2
    assert out["max_enrichment"] == 1.2
    assert out["min_enrichment"] == -0.8
    assert out["analysis_key"] == "leiden_nhood_enrichment"


def test_analyze_co_occurrence_uses_interval_from_params_and_returns_distance_range(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["leiden"] = pd.Categorical(["0"] * 30 + ["1"] * 30)

    captured: dict[str, object] = {}

    def _fake_co_occurrence(a, cluster_key, interval):
        captured["interval"] = interval
        captured["cluster_key"] = cluster_key
        a.uns["leiden_co_occurrence"] = {
            "occ": np.zeros((2, 2, 4)),
            "interval": np.array([0.0, 5.0, 10.0, 15.0]),
        }

    monkeypatch.setattr(ss.sq.gr, "co_occurrence", _fake_co_occurrence)

    params = SpatialStatisticsParameters(
        analysis_type="co_occurrence",
        cluster_key="leiden",
        co_occurrence_interval=42,
    )
    out = ss._analyze_co_occurrence(adata, "leiden", params, DummyCtx(adata))

    assert captured == {"interval": 42, "cluster_key": "leiden"}
    assert out["n_clusters"] == 2
    assert out["n_intervals"] == 4
    assert out["distance_range"] == (0.0, 15.0)


def test_analyze_co_occurrence_uses_default_interval_when_not_set(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["leiden"] = pd.Categorical(["0"] * 30 + ["1"] * 30)

    captured: dict[str, object] = {}

    def _fake_co_occurrence(a, cluster_key, interval):
        del cluster_key
        captured["interval"] = interval
        a.uns["leiden_co_occurrence"] = {"occ": np.zeros((2, 2, 3))}

    monkeypatch.setattr(ss.sq.gr, "co_occurrence", _fake_co_occurrence)

    params = SpatialStatisticsParameters(
        analysis_type="co_occurrence",
        cluster_key="leiden",
        co_occurrence_interval=None,
    )
    out = ss._analyze_co_occurrence(adata, "leiden", params, DummyCtx(adata))

    assert captured["interval"] == 50
    assert out["n_intervals"] == 50


def test_analyze_gearys_c_raises_when_results_missing(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()

    monkeypatch.setattr(ss, "select_genes_for_analysis", lambda *_a, **_k: ["gene_0", "gene_1"])
    monkeypatch.setattr(ss.sq.gr, "spatial_autocorr", lambda *_a, **_k: None)

    with pytest.raises(ProcessingError, match="Geary's C computation did not produce results"):
        ss._analyze_gearys_c(
            adata,
            SpatialStatisticsParameters(analysis_type="geary", genes=["gene_0"]),
            DummyCtx(adata),
        )


@pytest.mark.asyncio
async def test_analyze_spatial_statistics_wraps_unexpected_get_adata_errors():
    params = SpatialStatisticsParameters(analysis_type="moran")
    with pytest.raises(ProcessingError, match="Error in moran analysis"):
        await ss.analyze_spatial_statistics(
            "d-bad",
            DummyCtx(None, error=RuntimeError("boom")),
            params,
        )


def test_analyze_getis_ord_fdr_branch_adds_corrected_pvalues(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()

    monkeypatch.setattr(ss, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ss,
        "select_genes_for_analysis",
        lambda *_args, **_kwargs: ["gene_0", "gene_1"],
    )

    class _FakeGLocal:
        def __init__(self, y, w, transform="R", star=True):
            del w, transform, star
            y = np.asarray(y)
            self.Zs = np.where(y >= np.median(y), 2.5, -2.5)
            self.p_sim = np.full(y.shape[0], 0.01, dtype=float)

    fake_esda_getisord = ModuleType("esda.getisord")
    fake_esda_getisord.G_Local = _FakeGLocal

    class _KNNObj:
        def __init__(self):
            self.transform = None

    fake_pysal_lib = ModuleType("pysal.lib")
    fake_pysal_lib.weights = SimpleNamespace(
        KNN=SimpleNamespace(from_array=lambda *_args, **_kwargs: _KNNObj())
    )

    monkeypatch.setitem(__import__("sys").modules, "esda.getisord", fake_esda_getisord)
    monkeypatch.setitem(__import__("sys").modules, "pysal.lib", fake_pysal_lib)

    params = SpatialStatisticsParameters(
        analysis_type="getis_ord",
        genes=["gene_0", "gene_1"],
        n_neighbors=5,
        getis_ord_correction="fdr_bh",
        getis_ord_alpha=0.05,
    )
    out = ss._analyze_getis_ord(adata, params, DummyCtx(adata))

    assert out["n_genes_analyzed"] == 2
    assert "gene_0" in out["results"]
    assert "n_significant_corrected" in out["results"]["gene_0"]
    assert "gene_0_getis_ord_p_corrected" in adata.obs.columns
    assert "gene_1_getis_ord_p_corrected" in adata.obs.columns



def test_analyze_bivariate_moran_computes_and_persists_results(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    from scipy.sparse import csr_matrix

    adata = minimal_spatial_adata.copy()
    monkeypatch.setattr(ss, "require", lambda *_args, **_kwargs: None)

    n = adata.n_obs
    dense_w = np.ones((n, n), dtype=float)
    np.fill_diagonal(dense_w, 0.0)

    fake_libpysal_weights = ModuleType("libpysal.weights")
    fake_libpysal_weights.KNN = SimpleNamespace(
        from_array=lambda *_args, **_kwargs: SimpleNamespace(
            sparse=csr_matrix(dense_w), transform=None
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "libpysal.weights", fake_libpysal_weights)

    params = SpatialStatisticsParameters(
        analysis_type="bivariate_moran",
        gene_pairs=[("gene_0", "gene_1"), ("gene_0", "not_exist")],
        n_neighbors=4,
    )
    out = ss._analyze_bivariate_moran(adata, params, DummyCtx(adata))

    assert out["n_pairs_analyzed"] == 1
    assert "gene_0_vs_gene_1" in out["bivariate_morans_i"]
    assert "bivariate_moran" in adata.uns



def test_analyze_network_properties_disconnected_component_branch(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    from scipy.sparse import csr_matrix

    adata = minimal_spatial_adata.copy()
    n = adata.n_obs

    # Two disconnected components: first 30, last 30
    conn = np.zeros((n, n), dtype=float)
    for i in range(29):
        conn[i, i + 1] = 1.0
        conn[i + 1, i] = 1.0
    for i in range(30, 59):
        conn[i, i + 1] = 1.0
        conn[i + 1, i] = 1.0
    adata.obsp["spatial_connectivities"] = csr_matrix(conn)

    monkeypatch.setattr(ss, "require", lambda *_args, **_kwargs: None)

    out = ss._analyze_network_properties(
        adata,
        cluster_key="group",
        params=SpatialStatisticsParameters(analysis_type="network_properties", n_neighbors=4),
        ctx=DummyCtx(adata),
    )

    assert out["is_connected"] is False
    assert out["n_components"] == 2
    assert out["largest_component_size"] == 30
    assert "network_properties" in adata.uns



def test_analyze_spatial_centrality_handles_missing_node_keys(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    from scipy.sparse import eye

    adata = minimal_spatial_adata.copy()
    adata.obs["group"] = pd.Categorical(["a"] * 30 + ["b"] * 30)
    adata.obsp["spatial_connectivities"] = eye(adata.n_obs, format="csr")

    monkeypatch.setattr(ss, "require", lambda *_args, **_kwargs: None)

    fake_nx = ModuleType("networkx")
    fake_nx.from_scipy_sparse_array = lambda _arr: object()
    fake_nx.degree_centrality = lambda _g: {0: 1.0}  # missing all other nodes
    fake_nx.closeness_centrality = lambda _g: {0: 0.5}
    fake_nx.betweenness_centrality = lambda _g: {0: 0.2}

    monkeypatch.setitem(__import__("sys").modules, "networkx", fake_nx)

    out = ss._analyze_spatial_centrality(
        adata,
        cluster_key="group",
        params=SpatialStatisticsParameters(analysis_type="spatial_centrality", n_neighbors=4),
        ctx=DummyCtx(adata),
    )

    assert out["centrality_computed"] is True
    assert "degree_centrality" in adata.obs.columns
    assert adata.obs["degree_centrality"].shape[0] == adata.n_obs
    # Only node 0 from fake centrality has non-zero; others should be safely backfilled with 0
    assert float(adata.obs["degree_centrality"].iloc[0]) == 1.0
    assert float(adata.obs["degree_centrality"].iloc[1]) == 0.0
