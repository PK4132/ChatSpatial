"""Unit tests for spatial_genes core contracts with lightweight dependency stubs."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chatspatial.models.data import SpatialVariableGenesParameters
from chatspatial.tools import spatial_genes as sg
from chatspatial.utils.exceptions import DataError, DataNotFoundError


class DummyCtx:
    def __init__(self):
        self.warnings: list[str] = []

    async def warning(self, msg: str):
        self.warnings.append(msg)


def _install_fake_rpy2(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Lock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_ro = ModuleType("rpy2.robjects")
    fake_ro.r = {}
    fake_ro.default_converter = object()

    fake_conversion = ModuleType("rpy2.robjects.conversion")

    class _LocalConverter:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_conversion.localconverter = _LocalConverter
    fake_conversion.default_converter = object()

    fake_packages = ModuleType("rpy2.robjects.packages")
    fake_packages.importr = lambda *_args, **_kwargs: None

    fake_rinterface_lib = ModuleType("rpy2.rinterface_lib")
    fake_rinterface_lib.openrlib = SimpleNamespace(rlock=_Lock())

    monkeypatch.setitem(__import__("sys").modules, "rpy2", ModuleType("rpy2"))
    monkeypatch.setitem(__import__("sys").modules, "rpy2.robjects", fake_ro)
    monkeypatch.setitem(
        __import__("sys").modules, "rpy2.robjects.conversion", fake_conversion
    )
    monkeypatch.setitem(
        __import__("sys").modules, "rpy2.robjects.packages", fake_packages
    )
    monkeypatch.setitem(
        __import__("sys").modules, "rpy2.rinterface_lib", fake_rinterface_lib
    )


@pytest.mark.asyncio
async def test_spatialde_success_stores_var_outputs_and_metadata(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    captured: dict[str, object] = {}

    fake_naivede = ModuleType("NaiveDE")
    fake_naivede.stabilize = lambda x: x
    fake_naivede.regress_out = lambda _tc, expr_t, _formula: expr_t

    fake_spatialde = ModuleType("SpatialDE")
    fake_spatialde.run = lambda _coords, _expr: pd.DataFrame(
        {"g": ["gene_0", "gene_1"], "pval": [0.001, 0.02], "l": [1.2, 0.8]}
    )
    fake_spatialde_util = ModuleType("SpatialDE.util")
    fake_spatialde_util.qvalue = lambda pvals, pi0=None: np.array([0.01, 0.04])

    monkeypatch.setitem(__import__("sys").modules, "NaiveDE", fake_naivede)
    monkeypatch.setitem(__import__("sys").modules, "SpatialDE", fake_spatialde)
    monkeypatch.setitem(__import__("sys").modules, "SpatialDE.util", fake_spatialde_util)

    monkeypatch.setattr(sg, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "chatspatial.utils.compat.ensure_spatialde_compat",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        sg,
        "get_raw_data_source",
        lambda _adata, prefer_complete_genes=True: SimpleNamespace(
            X=_adata.X,
            var_names=_adata.var_names,
            source="raw",
        ),
    )
    monkeypatch.setattr(
        "chatspatial.utils.adata_utils.store_analysis_metadata",
        lambda _adata, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "chatspatial.utils.results_export.export_analysis_result",
        lambda *_args, **_kwargs: [],
    )

    out = await sg._identify_spatial_genes_spatialde(
        "d1",
        adata,
        SpatialVariableGenesParameters(method="spatialde", spatial_key="spatial"),
        DummyCtx(),
    )
    assert out.method == "spatialde"
    assert out.n_significant_genes == 2
    assert "spatialde_pval" in adata.var.columns
    assert "spatialde_qval" in adata.var.columns
    assert captured["analysis_name"] == "spatial_genes_spatialde"
    assert captured["results_keys"]["var"] == [
        "spatialde_pval",
        "spatialde_qval",
        "spatialde_l",
    ]


@pytest.mark.asyncio
async def test_flashs_success_stores_var_outputs_and_metadata(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    captured: dict[str, object] = {}

    class _FakeResult:
        def __init__(self, genes: list[str]):
            n = len(genes)
            self.gene_names = genes
            self.pvalues = np.linspace(0.001, 0.8, n)
            self.qvalues = np.linspace(0.01, 0.9, n)
            self.statistics = np.linspace(1.0, 2.0, n)
            self.effect_size = np.linspace(0.5, 1.5, n)
            self.pvalues_binary = np.linspace(0.002, 0.7, n)
            self.pvalues_rank = np.linspace(0.003, 0.6, n)
            self.n_expressed = np.arange(10, 10 + n)
            self.tested_mask = np.array([True] * (n - 1) + [False], dtype=bool)
            self.n_tested = n - 1
            self.n_significant = 1

    class _FakeFlashS:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_test(self, coords, X, gene_names):
            del coords, X
            return _FakeResult(gene_names)

    fake_flashs = ModuleType("flashs")
    fake_flashs.FlashS = _FakeFlashS
    monkeypatch.setitem(__import__("sys").modules, "flashs", fake_flashs)

    monkeypatch.setattr(sg, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sg,
        "get_raw_data_source",
        lambda _adata, prefer_complete_genes=False: SimpleNamespace(
            X=_adata.X,
            var_names=_adata.var_names,
            source="current",
        ),
    )
    monkeypatch.setattr(
        "chatspatial.utils.adata_utils.store_analysis_metadata",
        lambda _adata, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "chatspatial.utils.results_export.export_analysis_result",
        lambda *_args, **_kwargs: [],
    )

    out = await sg._identify_spatial_genes_flashs(
        "d_flashs",
        adata,
        SpatialVariableGenesParameters(method="flashs", spatial_key="spatial"),
        DummyCtx(),
    )

    assert out.method == "flashs"
    assert out.n_genes_analyzed == adata.n_vars - 1
    assert "flashs_pval" in adata.var.columns
    assert "flashs_qval" in adata.var.columns
    assert "flashs_statistic" in adata.var.columns
    assert "flashs_effect_size" in adata.var.columns
    assert "flashs_pval_binary" in adata.var.columns
    assert "flashs_pval_rank" in adata.var.columns
    assert "flashs_n_expressed" in adata.var.columns
    assert "flashs_tested" in adata.var.columns
    assert captured["analysis_name"] == "spatial_genes_flashs"
    assert captured["method"] == "flashs"
    assert "flashs_qval" in captured["results_keys"]["var"]


@pytest.mark.asyncio
async def test_flashs_missing_dependency_raises_import_error(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()

    def _raise_import(*_args, **_kwargs):
        raise ImportError("flashs dependency missing")

    monkeypatch.setattr(sg, "require", _raise_import)

    with pytest.raises(ImportError, match="flashs dependency missing"):
        await sg._identify_spatial_genes_flashs(
            "d_missing",
            adata,
            SpatialVariableGenesParameters(method="flashs", spatial_key="spatial"),
            DummyCtx(),
        )


@pytest.mark.asyncio
async def test_sparkx_requires_hvg_column_when_test_only_hvg_enabled(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    _install_fake_rpy2(monkeypatch)
    monkeypatch.setattr(sg, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sg,
        "get_raw_data_source",
        lambda _adata, prefer_complete_genes=True: SimpleNamespace(
            X=_adata.X,
            var_names=_adata.var_names,
            source="raw",
        ),
    )

    with pytest.raises(DataError, match="Highly variable genes marker"):
        await sg._identify_spatial_genes_sparkx(
            "d1",
            adata,
            SpatialVariableGenesParameters(method="sparkx", test_only_hvg=True),
            DummyCtx(),
        )


@pytest.mark.asyncio
async def test_sparkx_raises_when_no_hvgs_found(minimal_spatial_adata, monkeypatch):
    adata = minimal_spatial_adata.copy()
    adata.var["highly_variable"] = False
    _install_fake_rpy2(monkeypatch)
    monkeypatch.setattr(sg, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sg,
        "get_raw_data_source",
        lambda _adata, prefer_complete_genes=True: SimpleNamespace(
            X=_adata.X,
            var_names=_adata.var_names,
            source="raw",
        ),
    )

    with pytest.raises(DataNotFoundError, match="No HVGs found"):
        await sg._identify_spatial_genes_sparkx(
            "d1",
            adata,
            SpatialVariableGenesParameters(method="sparkx", test_only_hvg=True),
            DummyCtx(),
        )


@pytest.mark.asyncio
async def test_sparkx_missing_r_package_raises_informative_import_error(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import sys

    adata = minimal_spatial_adata.copy()
    _install_fake_rpy2(monkeypatch)
    monkeypatch.setattr(sg, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sg,
        "get_raw_data_source",
        lambda _adata, prefer_complete_genes=True: SimpleNamespace(
            X=_adata.X,
            var_names=_adata.var_names,
            source="raw",
        ),
    )

    fake_packages = sys.modules["rpy2.robjects.packages"]

    def _raise_importr(_name):
        raise RuntimeError("package not found")

    fake_packages.importr = _raise_importr

    with pytest.raises(ImportError, match="SPARK not installed in R"):
        await sg._identify_spatial_genes_sparkx(
            "d2",
            adata,
            SpatialVariableGenesParameters(method="sparkx", spatial_key="spatial", test_only_hvg=False),
            DummyCtx(),
        )


@pytest.mark.asyncio
async def test_spatialde_warns_for_large_gene_set_runtime(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import anndata as ad

    base = minimal_spatial_adata.copy()
    ctx = DummyCtx()

    X_big = np.tile(np.asarray(base.X), (1, 260)).astype(np.float32)
    adata = ad.AnnData(X_big)
    adata.obs_names = base.obs_names.copy()
    adata.var_names = [f"gene_{i}" for i in range(adata.n_vars)]
    adata.obsm["spatial"] = np.asarray(base.obsm["spatial"]).copy()

    fake_naivede = ModuleType("NaiveDE")
    fake_naivede.stabilize = lambda x: x
    fake_naivede.regress_out = lambda _tc, expr_t, _formula: expr_t

    fake_spatialde = ModuleType("SpatialDE")
    fake_spatialde.run = lambda _coords, _expr: pd.DataFrame(
        {"g": ["gene_0", "gene_1"], "pval": [0.001, 0.02], "l": [1.0, 0.5]}
    )
    fake_spatialde_util = ModuleType("SpatialDE.util")
    fake_spatialde_util.qvalue = lambda pvals, pi0=None: np.array([0.01, 0.04])

    monkeypatch.setitem(__import__("sys").modules, "NaiveDE", fake_naivede)
    monkeypatch.setitem(__import__("sys").modules, "SpatialDE", fake_spatialde)
    monkeypatch.setitem(__import__("sys").modules, "SpatialDE.util", fake_spatialde_util)

    monkeypatch.setattr(sg, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "chatspatial.utils.compat.ensure_spatialde_compat",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        sg,
        "get_raw_data_source",
        lambda _adata, prefer_complete_genes=True: SimpleNamespace(
            X=_adata.X,
            var_names=_adata.var_names,
            source="raw",
        ),
    )
    monkeypatch.setattr(
        "chatspatial.utils.adata_utils.store_analysis_metadata",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "chatspatial.utils.results_export.export_analysis_result",
        lambda *_args, **_kwargs: [],
    )

    out = await sg._identify_spatial_genes_spatialde(
        "d3",
        adata,
        SpatialVariableGenesParameters(method="spatialde", spatial_key="spatial"),
        ctx,
    )

    assert out.method == "spatialde"
    assert any("may take" in w for w in ctx.warnings)
