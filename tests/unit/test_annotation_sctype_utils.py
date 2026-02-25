"""Unit tests for lightweight sc-type helper utilities in annotation module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chatspatial.models.data import AnnotationParameters
from chatspatial.tools import annotation as ann
from chatspatial.utils.exceptions import DataError


class DummyCtx:
    async def warning(self, _msg: str):
        return None


def test_softmax_is_stable_and_normalized():
    arr = np.array([1000.0, 1001.0, 999.0], dtype=float)
    out = ann._softmax(arr)
    assert np.isfinite(out).all()
    assert np.isclose(out.sum(), 1.0)
    assert out.argmax() == 1


def test_assign_sctype_celltypes_and_unknown_handling():
    scores = pd.DataFrame(
        {
            "cell_1": [2.0, 0.1],
            "cell_2": [-1.0, -0.2],
        },
        index=["T", "B"],
    )

    types, conf = ann._assign_sctype_celltypes(scores, DummyCtx())
    assert types[0] == "T"
    assert conf[0] > 0
    assert types[1] == "Unknown"
    assert conf[1] == 0.0


def test_assign_sctype_celltypes_rejects_empty_scores():
    with pytest.raises(DataError, match="Scores DataFrame is empty or None"):
        ann._assign_sctype_celltypes(pd.DataFrame(), DummyCtx())


def test_calculate_sctype_stats_counts_labels():
    out = ann._calculate_sctype_stats(["T", "B", "T", "Unknown"])
    assert out == {"T": 2, "B": 1, "Unknown": 1}


def test_get_sctype_cache_key_changes_with_params(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    p1 = AnnotationParameters(method="sctype", sctype_tissue="Liver")
    p2 = AnnotationParameters(method="sctype", sctype_tissue="Brain")
    k1 = ann._get_sctype_cache_key(adata, p1)
    k2 = ann._get_sctype_cache_key(adata, p2)
    assert k1 != k2


def test_load_cached_sctype_results_reads_json(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ann, "_SCTYPE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ann, "_SCTYPE_CACHE", {})
    cache_key = "abc"
    payload = {
        "cell_types": ["T", "B"],
        "counts": {"T": 1, "B": 1},
        "confidence_by_celltype": {"T": 0.7, "B": 0.6},
        "mapping_score": None,
    }
    (tmp_path / f"{cache_key}.json").write_text(json.dumps(payload), encoding="utf-8")

    out = ann._load_cached_sctype_results(cache_key, DummyCtx())
    assert out is not None
    assert out[0] == ["T", "B"]
    assert out[1]["T"] == 1


@pytest.mark.asyncio
async def test_cache_sctype_results_writes_json(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ann, "_SCTYPE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ann, "_SCTYPE_CACHE", {})
    cache_key = "k1"
    results = (["T"], {"T": 1}, {"T": 0.9}, 0.5)

    await ann._cache_sctype_results(cache_key, results, DummyCtx())

    f = tmp_path / f"{cache_key}.json"
    assert f.exists()
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data["cell_types"] == ["T"]
    assert data["mapping_score"] == 0.5


@pytest.mark.asyncio
async def test_annotate_with_sctype_cache_hit_short_circuits_pipeline(
    minimal_spatial_adata,
    monkeypatch: pytest.MonkeyPatch,
):
    adata = minimal_spatial_adata.copy()
    params = AnnotationParameters(method="sctype", sctype_tissue="Brain", sctype_use_cache=True)

    cached_cell_types = ["T"] * (adata.n_obs // 2) + ["B"] * (adata.n_obs - adata.n_obs // 2)
    cached_counts = {"T": adata.n_obs // 2, "B": adata.n_obs - adata.n_obs // 2}

    monkeypatch.setattr(ann, "validate_r_environment", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(ann, "_get_sctype_cache_key", lambda *_args, **_kwargs: "k1")
    monkeypatch.setattr(
        ann,
        "_load_cached_sctype_results",
        lambda *_args, **_kwargs: (cached_cell_types, cached_counts, {"T": 0.8, "B": 0.7}, None),
    )

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("pipeline should not run on cache hit")

    monkeypatch.setattr(ann, "_load_sctype_functions", _should_not_run)
    monkeypatch.setattr(ann, "_prepare_sctype_genesets", _should_not_run)
    monkeypatch.setattr(ann, "_run_sctype_scoring", _should_not_run)

    out = await ann._annotate_with_sctype(
        adata,
        params,
        DummyCtx(),
        output_key="cell_type_sctype",
        confidence_key="confidence_sctype",
    )

    assert out.cell_types == cached_cell_types
    assert out.counts == cached_counts
    assert adata.obs["cell_type_sctype"].dtype.name == "category"


@pytest.mark.asyncio
async def test_annotate_with_sctype_cache_miss_preserves_cell_type_order_and_caches(
    minimal_spatial_adata,
    monkeypatch: pytest.MonkeyPatch,
):
    adata = minimal_spatial_adata.copy()
    params = AnnotationParameters(method="sctype", sctype_tissue="Brain", sctype_use_cache=True)

    captured: dict[str, object] = {}

    per_cell_types = ["B", "T", "B", "Unknown"] * (adata.n_obs // 4)
    per_cell_conf = [0.9, 0.8, 0.7, 0.0] * (adata.n_obs // 4)

    monkeypatch.setattr(ann, "validate_r_environment", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(ann, "_get_sctype_cache_key", lambda *_args, **_kwargs: "k2")
    monkeypatch.setattr(ann, "_load_cached_sctype_results", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ann, "_load_sctype_functions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ann, "_prepare_sctype_genesets", lambda *_args, **_kwargs: "GS")
    monkeypatch.setattr(
        ann,
        "_run_sctype_scoring",
        lambda *_args, **_kwargs: pd.DataFrame({"c1": [1.0], "c2": [1.0]}, index=["T"]),
    )
    monkeypatch.setattr(
        ann,
        "_assign_sctype_celltypes",
        lambda *_args, **_kwargs: (per_cell_types, per_cell_conf),
    )

    async def _fake_cache(cache_key, results, _ctx):
        captured["cache_key"] = cache_key
        captured["results"] = results

    monkeypatch.setattr(ann, "_cache_sctype_results", _fake_cache)

    out = await ann._annotate_with_sctype(
        adata,
        params,
        DummyCtx(),
        output_key="cell_type_sctype",
        confidence_key="confidence_sctype",
    )

    assert out.cell_types == ["B", "T", "Unknown"]
    assert out.counts == {"B": 30, "T": 15, "Unknown": 15}
    assert captured["cache_key"] == "k2"
    assert captured["results"][0] == ["B", "T", "Unknown"]


def test_prepare_sctype_genesets_requires_tissue_without_custom_markers():
    with pytest.raises(
        ann.ParameterError,
        match="sctype_tissue is required when not using custom markers",
    ):
        ann._prepare_sctype_genesets(
            AnnotationParameters(method="sctype", sctype_tissue=None, sctype_custom_markers=None),
            DummyCtx(),
        )


def test_convert_custom_markers_rejects_empty_dict():
    with pytest.raises(DataError, match="Custom markers dictionary is empty"):
        ann._convert_custom_markers_to_gs({}, DummyCtx())


def test_convert_custom_markers_rejects_without_positive_markers():
    bad = {"T": {"negative": ["MALAT1"]}}
    with pytest.raises(DataError, match="No valid cell types found"):
        ann._convert_custom_markers_to_gs(bad, DummyCtx())


def test_convert_custom_markers_normalizes_and_filters(monkeypatch: pytest.MonkeyPatch):
    class _Conv:
        def __add__(self, _other):
            return self

    class _Lock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _LCtx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _R:
        def __getitem__(self, name: str):
            if name == "list":
                return lambda **kwargs: kwargs
            raise KeyError(name)

    robjects = type("RObj", (), {"default_converter": _Conv(), "StrVector": lambda self, xs: list(xs), "r": _R()})()
    pandas2ri = type("P2", (), {"converter": _Conv()})()
    openrlib = type("OL", (), {"rlock": _Lock()})()

    monkeypatch.setattr(
        ann,
        "validate_r_environment",
        lambda _ctx: (robjects, pandas2ri, None, None, lambda _c: _LCtx(), None, openrlib, None),
    )

    markers = {
        "T": {"positive": [" cd3d ", "", None, "CD3E"], "negative": [" malat1 "]},
        "B": {"positive": ["MS4A1"], "negative": []},
        "ignored": ["not-a-dict"],
    }

    out = ann._convert_custom_markers_to_gs(markers, DummyCtx())

    assert set(out.keys()) == {"gs_positive", "gs_negative"}
    assert out["gs_positive"]["T"] == ["CD3D", "CD3E"]
    assert out["gs_negative"]["T"] == ["MALAT1"]
    assert out["gs_positive"]["B"] == ["MS4A1"]
