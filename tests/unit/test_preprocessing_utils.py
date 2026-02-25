"""Unit tests for preprocessing helper and preprocess_data contracts."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from anndata import AnnData

from chatspatial.models.data import PreprocessingParameters
from chatspatial.tools import preprocessing as preprocessing_mod
from chatspatial.tools.preprocessing import _compute_safe_percent_top, preprocess_data
from chatspatial.utils.exceptions import DataError, DependencyError, ParameterError, ProcessingError


class DummyCtx:
    def __init__(self, adata: AnnData):
        self._adata = adata
        self.saved_adata: AnnData | None = None
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.config_logs: list[tuple[str, dict]] = []

    async def get_adata(self, _data_id: str) -> AnnData:
        return self._adata

    async def set_adata(self, _data_id: str, adata: AnnData) -> None:
        self.saved_adata = adata

    async def warning(self, msg: str) -> None:
        self.warnings.append(msg)

    async def info(self, msg: str) -> None:
        self.infos.append(msg)

    def log_config(self, title: str, config: dict) -> None:
        self.config_logs.append((title, config))


def _make_adata(n_obs: int = 30, n_vars: int = 120) -> AnnData:
    rng = np.random.default_rng(7)
    X = rng.poisson(5, size=(n_obs, n_vars)).astype(np.float32)
    adata = AnnData(X=X)
    # Include mito/ribo-like names to exercise annotation columns.
    var_names = [f"gene_{i}" for i in range(n_vars)]
    if n_vars > 2:
        var_names[0] = "MT-ND1"
        var_names[1] = "RPS3"
    adata.var_names = var_names
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    return adata


def _install_lightweight_preprocess_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_ensure_unique(_adata, _ctx, _label):
        return None

    def _identity_standardize(adata, copy=False):
        return adata.copy() if copy else adata

    def _sample_values(adata):
        arr = adata.X
        flat = np.asarray(arr).reshape(-1)
        return flat[: min(len(flat), 128)]

    def _calc_qc_metrics(adata, qc_vars=None, percent_top=None, inplace=True):
        del qc_vars, percent_top, inplace
        counts = np.asarray(adata.X)
        adata.obs["n_genes_by_counts"] = (counts > 0).sum(axis=1)
        adata.obs["total_counts"] = counts.sum(axis=1)
        if "mt" in adata.var.columns:
            mt_mask = adata.var["mt"].to_numpy()
            mt_counts = counts[:, mt_mask].sum(axis=1)
            total = np.clip(counts.sum(axis=1), 1e-9, None)
            adata.obs["pct_counts_mt"] = mt_counts / total * 100.0

    def _hvg(adata, n_top_genes=2000):
        flags = np.zeros(adata.n_vars, dtype=bool)
        flags[: min(max(int(n_top_genes), 0), adata.n_vars)] = True
        adata.var["highly_variable"] = flags

    def _no_op(*args, **kwargs):
        del args, kwargs
        return None

    monkeypatch.setattr(preprocessing_mod, "ensure_unique_var_names_async", _noop_ensure_unique)
    monkeypatch.setattr(preprocessing_mod, "standardize_adata", _identity_standardize)
    monkeypatch.setattr(preprocessing_mod, "sample_expression_values", _sample_values)
    monkeypatch.setattr(preprocessing_mod.sc.pp, "calculate_qc_metrics", _calc_qc_metrics)
    monkeypatch.setattr(preprocessing_mod.sc.pp, "filter_genes", _no_op)
    monkeypatch.setattr(preprocessing_mod.sc.pp, "filter_cells", _no_op)
    monkeypatch.setattr(preprocessing_mod.sc.pp, "subsample", _no_op)
    monkeypatch.setattr(preprocessing_mod.sc.pp, "normalize_total", _no_op)
    monkeypatch.setattr(preprocessing_mod.sc.pp, "log1p", _no_op)
    monkeypatch.setattr(preprocessing_mod.sc.pp, "highly_variable_genes", _hvg)
    monkeypatch.setattr(preprocessing_mod.sc.pp, "scale", _no_op)


@pytest.mark.asyncio
async def test_preprocess_data_success_persists_core_artifacts(monkeypatch: pytest.MonkeyPatch):
    _install_lightweight_preprocess_mocks(monkeypatch)

    adata = _make_adata(n_obs=24, n_vars=120)
    ctx = DummyCtx(adata)
    params = PreprocessingParameters(
        normalization="log",
        n_hvgs=20,
        subsample_genes=12,
        filter_genes_min_cells=1,
        filter_cells_min_genes=1,
        remove_mito_genes=False,
        remove_ribo_genes=False,
        filter_mito_pct=None,
        scale=False,
    )

    result = await preprocess_data("d1", ctx, params)

    assert result.data_id == "d1"
    assert result.n_cells == 24
    assert result.n_genes == 12
    assert result.n_hvgs == 12

    assert ctx.saved_adata is not None
    assert "counts" in ctx.saved_adata.layers
    assert ctx.saved_adata.layers["counts"].shape == ctx.saved_adata.X.shape
    assert ctx.saved_adata.raw is not None
    assert ctx.saved_adata.uns["preprocessing"]["completed"] is True


@pytest.mark.asyncio
async def test_preprocess_data_warns_when_hvgs_too_low(monkeypatch: pytest.MonkeyPatch):
    _install_lightweight_preprocess_mocks(monkeypatch)

    adata = _make_adata(n_obs=20, n_vars=120)
    ctx = DummyCtx(adata)
    params = PreprocessingParameters(
        normalization="log",
        n_hvgs=30,
        filter_genes_min_cells=1,
        filter_cells_min_genes=1,
        remove_mito_genes=False,
        filter_mito_pct=None,
    )

    await preprocess_data("d2", ctx, params)

    assert any("recommended minimum of 500" in w for w in ctx.warnings)


@pytest.mark.asyncio
async def test_preprocess_data_rejects_none_normalization_for_raw_counts(monkeypatch: pytest.MonkeyPatch):
    _install_lightweight_preprocess_mocks(monkeypatch)

    def _raw_like_sample(_adata):
        return np.array([0.0, 10.0, 200.0, 50.0])

    monkeypatch.setattr(preprocessing_mod, "sample_expression_values", _raw_like_sample)

    adata = _make_adata(n_obs=10, n_vars=120)
    ctx = DummyCtx(adata)
    params = PreprocessingParameters(normalization="none", filter_mito_pct=None)

    with pytest.raises(DataError, match="Cannot perform HVG selection on raw counts"):
        await preprocess_data("d3", ctx, params)


@pytest.mark.asyncio
async def test_preprocess_data_unknown_normalization_raises(monkeypatch: pytest.MonkeyPatch):
    _install_lightweight_preprocess_mocks(monkeypatch)

    adata = _make_adata(n_obs=10, n_vars=120)
    ctx = DummyCtx(adata)
    # bypass Literal validation to test runtime defensive branch
    params = PreprocessingParameters.model_construct(normalization="invalid", filter_mito_pct=None)

    with pytest.raises(ParameterError, match="Unknown normalization method"):
        await preprocess_data("d4", ctx, params)


@pytest.mark.asyncio
async def test_preprocess_data_gene_subsample_requires_nonempty_hvgs(monkeypatch: pytest.MonkeyPatch):
    _install_lightweight_preprocess_mocks(monkeypatch)

    def _all_false_hvg(adata, n_top_genes=2000):
        del n_top_genes
        adata.var["highly_variable"] = False

    monkeypatch.setattr(preprocessing_mod.sc.pp, "highly_variable_genes", _all_false_hvg)

    adata = _make_adata(n_obs=16, n_vars=120)
    ctx = DummyCtx(adata)
    params = PreprocessingParameters(
        normalization="log",
        subsample_genes=20,
        n_hvgs=20,
        filter_genes_min_cells=1,
        filter_cells_min_genes=1,
        filter_mito_pct=None,
    )

    with pytest.raises(DataError, match="no genes were marked as highly variable"):
        await preprocess_data("d5", ctx, params)


@pytest.mark.asyncio
async def test_preprocess_data_pearson_residuals_requires_scanpy_support(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_lightweight_preprocess_mocks(monkeypatch)
    monkeypatch.setattr(
        preprocessing_mod.sc,
        "experimental",
        SimpleNamespace(pp=SimpleNamespace()),
        raising=False,
    )

    adata = _make_adata(n_obs=12, n_vars=120)
    ctx = DummyCtx(adata)
    params = PreprocessingParameters(normalization="pearson_residuals", filter_mito_pct=None)

    with pytest.raises(DependencyError, match="Pearson residuals normalization not available"):
        await preprocess_data("d6", ctx, params)


@pytest.mark.asyncio
async def test_preprocess_data_pearson_residuals_rejects_non_integer_input(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_lightweight_preprocess_mocks(monkeypatch)

    monkeypatch.setattr(
        preprocessing_mod.sc,
        "experimental",
        SimpleNamespace(pp=SimpleNamespace(normalize_pearson_residuals=lambda _adata: None)),
        raising=False,
    )
    monkeypatch.setattr(
        preprocessing_mod,
        "sample_expression_values",
        lambda _adata: np.array([0.1, 1.5, 2.0]),
    )

    adata = _make_adata(n_obs=12, n_vars=120)
    ctx = DummyCtx(adata)
    params = PreprocessingParameters(normalization="pearson_residuals", filter_mito_pct=None)

    with pytest.raises(DataError, match="requires raw count data"):
        await preprocess_data("d7", ctx, params)


@pytest.mark.asyncio
async def test_preprocess_data_scvi_success_writes_latent_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_lightweight_preprocess_mocks(monkeypatch)
    monkeypatch.setattr(preprocessing_mod, "require", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        preprocessing_mod,
        "sample_expression_values",
        lambda _adata: np.array([0.0, 1.0, 3.0]),
    )

    class _FakeSCVI:
        @staticmethod
        def setup_anndata(adata, layer=None, batch_key=None):
            del adata, layer, batch_key
            return None

        def __init__(self, adata, **kwargs):
            self._adata = adata
            self._kwargs = kwargs

        def train(self, **kwargs):
            del kwargs
            return None

        def get_latent_representation(self):
            return np.ones((self._adata.n_obs, 2), dtype=float)

        def get_normalized_expression(self, library_size=1e4):
            del library_size
            return np.full((self._adata.n_obs, self._adata.n_vars), 2.0, dtype=float)

    fake_scvi = SimpleNamespace(model=SimpleNamespace(SCVI=_FakeSCVI))
    monkeypatch.setitem(sys.modules, "scvi", fake_scvi)

    adata = _make_adata(n_obs=14, n_vars=120)
    ctx = DummyCtx(adata)
    params = PreprocessingParameters(
        normalization="scvi",
        n_hvgs=30,
        filter_genes_min_cells=1,
        filter_cells_min_genes=1,
        filter_mito_pct=None,
        remove_mito_genes=False,
    )

    result = await preprocess_data("d8", ctx, params)

    assert result.n_cells == 14
    assert ctx.saved_adata is not None
    assert "X_scvi" in ctx.saved_adata.obsm
    assert ctx.saved_adata.obsm["X_scvi"].shape == (14, 2)
    assert ctx.saved_adata.uns["scvi"]["training_completed"] is True


@pytest.mark.asyncio
async def test_preprocess_data_scvi_failure_is_wrapped_as_processing_error(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_lightweight_preprocess_mocks(monkeypatch)
    monkeypatch.setattr(preprocessing_mod, "require", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        preprocessing_mod,
        "sample_expression_values",
        lambda _adata: np.array([0.0, 1.0, 3.0]),
    )

    class _FakeSCVI:
        @staticmethod
        def setup_anndata(adata, layer=None, batch_key=None):
            del adata, layer, batch_key
            return None

        def __init__(self, adata, **kwargs):
            del adata, kwargs

        def train(self, **kwargs):
            del kwargs
            raise RuntimeError("boom")

    fake_scvi = SimpleNamespace(model=SimpleNamespace(SCVI=_FakeSCVI))
    monkeypatch.setitem(sys.modules, "scvi", fake_scvi)

    adata = _make_adata(n_obs=14, n_vars=120)
    ctx = DummyCtx(adata)
    params = PreprocessingParameters(normalization="scvi", filter_mito_pct=None)

    with pytest.raises(ProcessingError, match="scVI normalization failed: boom"):
        await preprocess_data("d9", ctx, params)


def test_compute_safe_percent_top_small_gene_set():
    values = _compute_safe_percent_top(10)
    assert values is not None
    assert all(v < 10 for v in values)
    assert values[-1] == 9


def test_compute_safe_percent_top_standard_gene_set():
    values = _compute_safe_percent_top(1000)
    assert values == [50, 100, 200, 500, 999]


def test_compute_safe_percent_top_degenerate_case():
    assert _compute_safe_percent_top(1) is None


@pytest.mark.asyncio
async def test_preprocess_data_sct_missing_dependency_raises_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_lightweight_preprocess_mocks(monkeypatch)

    def _raise_import_error(_pkg, _ctx):
        raise ImportError("r package missing")

    monkeypatch.setattr(preprocessing_mod, "validate_r_package", _raise_import_error)

    adata = _make_adata(n_obs=12, n_vars=120)
    ctx = DummyCtx(adata)
    params = PreprocessingParameters(normalization="sct", filter_mito_pct=None)

    with pytest.raises(DependencyError, match="SCTransform requires R and the sctransform package"):
        await preprocess_data("d10", ctx, params)


@pytest.mark.asyncio
async def test_preprocess_data_sct_rejects_non_integer_input(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_lightweight_preprocess_mocks(monkeypatch)
    monkeypatch.setattr(preprocessing_mod, "validate_r_package", lambda *_a, **_k: None)
    monkeypatch.setattr(
        preprocessing_mod,
        "sample_expression_values",
        lambda _adata: np.array([0.1, 1.2, 3.0]),
    )

    adata = _make_adata(n_obs=12, n_vars=120)
    ctx = DummyCtx(adata)
    params = PreprocessingParameters(normalization="sct", filter_mito_pct=None)

    with pytest.raises(DataError, match="SCTransform requires raw count data"):
        await preprocess_data("d11", ctx, params)


@pytest.mark.asyncio
async def test_preprocess_data_sct_import_failure_wrapped_as_processing_error(
    monkeypatch: pytest.MonkeyPatch,
):
    import builtins

    _install_lightweight_preprocess_mocks(monkeypatch)
    monkeypatch.setattr(preprocessing_mod, "validate_r_package", lambda *_a, **_k: None)
    monkeypatch.setattr(
        preprocessing_mod,
        "sample_expression_values",
        lambda _adata: np.array([0.0, 1.0, 2.0]),
    )

    real_import = builtins.__import__

    def _wrapped_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("rpy2"):
            raise ModuleNotFoundError("forced missing rpy2")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _wrapped_import)

    adata = _make_adata(n_obs=12, n_vars=120)
    ctx = DummyCtx(adata)
    params = PreprocessingParameters(normalization="sct", filter_mito_pct=None)

    with pytest.raises(ProcessingError, match="SCTransform failed"):
        await preprocess_data("d12", ctx, params)


@pytest.mark.asyncio
async def test_preprocess_data_standardize_failure_warns_and_continues(monkeypatch: pytest.MonkeyPatch):
    _install_lightweight_preprocess_mocks(monkeypatch)

    def _raise_standardize(_adata, copy=False):
        del copy
        raise RuntimeError("std boom")

    monkeypatch.setattr(preprocessing_mod, "standardize_adata", _raise_standardize)

    adata = _make_adata(n_obs=12, n_vars=120)
    ctx = DummyCtx(adata)

    result = await preprocess_data(
        "d13",
        ctx,
        PreprocessingParameters(normalization="log", filter_mito_pct=None),
    )

    assert result.n_cells == 12
    assert any("Data standardization failed" in w for w in ctx.warnings)


@pytest.mark.asyncio
async def test_preprocess_data_log_normalization_rejects_negative_values(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_lightweight_preprocess_mocks(monkeypatch)
    monkeypatch.setattr(
        preprocessing_mod,
        "sample_expression_values",
        lambda _adata: np.array([1.0, -0.2, 3.0]),
    )

    adata = _make_adata(n_obs=10, n_vars=120)
    ctx = DummyCtx(adata)

    with pytest.raises(DataError, match="requires non-negative data"):
        await preprocess_data(
            "d14",
            ctx,
            PreprocessingParameters(normalization="log", filter_mito_pct=None),
        )


@pytest.mark.asyncio
async def test_preprocess_data_scrublet_warns_when_too_few_cells(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_lightweight_preprocess_mocks(monkeypatch)

    adata = _make_adata(n_obs=40, n_vars=120)
    ctx = DummyCtx(adata)

    await preprocess_data(
        "d15",
        ctx,
        PreprocessingParameters(
            normalization="log",
            filter_mito_pct=None,
            use_scrublet=True,
        ),
    )

    assert any("Scrublet requires at least 100 cells" in w for w in ctx.warnings)
