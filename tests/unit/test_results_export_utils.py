"""Unit contracts for metadata-driven result export utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chatspatial.utils import results_export as re


def _patch_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(re.Path, "home", classmethod(lambda _cls: tmp_path))


def test_export_analysis_result_returns_empty_without_metadata(
    minimal_spatial_adata, monkeypatch, tmp_path: Path
):
    _patch_home(monkeypatch, tmp_path)
    adata = minimal_spatial_adata.copy()
    assert re.export_analysis_result(adata, "d1", "spatial_genes_flashs") == []


def test_export_analysis_result_writes_csv_and_index_with_sanitized_keys(
    minimal_spatial_adata, monkeypatch, tmp_path: Path
):
    _patch_home(monkeypatch, tmp_path)
    monkeypatch.setenv("CHATSPATIAL_EXPORT_RESULTS", "1")
    adata = minimal_spatial_adata.copy()
    adata.var["flashs_qval"] = np.linspace(0.01, 0.9, adata.n_vars)
    complex_key = r"metrics/with\slashes"
    adata.uns[complex_key] = {"a": 1, "b": 2}
    adata.uns["spatial_genes_flashs_metadata"] = {
        "method": "flashs",
        "parameters": {"registry_path": Path("data/datasets.local.json"), "n": 5},
        "statistics": {"n_significant": 3},
        "results_keys": {"uns": [complex_key], "var": ["flashs_qval"]},
    }

    exported = re.export_analysis_result(adata, "d_flashs", "spatial_genes_flashs")
    exported_names = {p.name for p in exported}

    assert "flashs_metrics_with_slashes.csv" in exported_names
    assert "flashs_flashs_qval.csv" in exported_names
    assert all(p.exists() for p in exported)

    index_path = tmp_path / ".chatspatial" / "results" / "d_flashs" / "_index.json"
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    entry = index["analyses"]["spatial_genes_flashs"]
    assert entry["method"] == "flashs"
    assert entry["parameters"]["registry_path"] == "data/datasets.local.json"
    assert set(entry["files"]) == exported_names

    listed = re.list_exported_results("d_flashs")
    assert set(listed["spatial_genes_flashs"]) == exported_names
    assert re.get_result_path(
        "d_flashs", "spatial_genes_flashs", "flashs_flashs_qval.csv"
    ).exists()


def test_extract_squidpy_co_occurrence_preserves_per_interval_structure(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    adata.obs["domain"] = pd.Categorical(["A", "B", "A"] * (adata.n_obs // 3))
    occ = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    intervals = np.array([0.0, 1.0, 2.0, 3.0])
    data = {"occ": occ, "interval": intervals}

    df = re._extract_squidpy_spatial_result(adata, "domain_co_occurrence", data)

    assert df is not None
    assert df.index.name == "domain"
    # Per-interval columns with distance labels
    assert any("d0.0-1.0" in c for c in df.columns)
    assert any("d2.0-3.0" in c for c in df.columns)
    # Summary mean still present
    assert any(c.startswith("occ_mean_") for c in df.columns)
    # 3 intervals × 2 col_labels + 2 mean cols = 8 columns
    assert len(df.columns) == 3 * 2 + 2


def test_extract_squidpy_co_occurrence_fallback_without_intervals(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    adata.obs["domain"] = pd.Categorical(["A", "B", "A"] * (adata.n_obs // 3))
    occ = np.arange(2 * 2 * 2, dtype=float).reshape(2, 2, 2)
    data = {"occ": occ}  # No interval key

    df = re._extract_squidpy_spatial_result(adata, "domain_co_occurrence", data)

    assert df is not None
    # Fallback naming: int0, int1
    assert any("int0" in c for c in df.columns)
    assert any("int1" in c for c in df.columns)
    assert any(c.startswith("occ_mean_") for c in df.columns)


def test_extract_from_obsm_supports_lineage_like_object():
    class _FakeAdata:
        def __init__(self, n_obs: int):
            self.obsm = {}
            self.obs_names = [f"cell_{i}" for i in range(n_obs)]
            self.uns = {}

    class _FakeLineage:
        def __init__(self, n_obs: int):
            self.names = ["fate_a", "fate_b"]
            self.X = np.zeros((n_obs, 2), dtype=float)

    adata = _FakeAdata(n_obs=8)
    adata.obsm["fate_probs"] = _FakeLineage(8)
    df = re._extract_from_obsm(adata, "fate_probs")

    assert list(df.columns) == ["fate_a", "fate_b"]
    assert len(df) == 8


def test_export_analysis_result_continues_when_one_key_extraction_fails(
    minimal_spatial_adata, monkeypatch, tmp_path: Path
):
    _patch_home(monkeypatch, tmp_path)
    monkeypatch.setenv("CHATSPATIAL_EXPORT_RESULTS", "1")
    adata = minimal_spatial_adata.copy()
    adata.uns["demo_metadata"] = {
        "method": "demo",
        "results_keys": {"var": ["good", "bad"]},
        "parameters": {},
        "statistics": {},
    }

    def _fake_extract(_adata, _location, key, _analysis_name):
        if key == "bad":
            raise RuntimeError("boom")
        return pd.DataFrame({"value": [1.0]}, index=["row_0"])

    monkeypatch.setattr(re, "_extract_as_dataframe", _fake_extract)

    exported = re.export_analysis_result(adata, "d2", "demo")
    assert len(exported) == 1
    assert exported[0].name == "demo_good.csv"
