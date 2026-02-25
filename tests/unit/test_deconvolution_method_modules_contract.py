"""Unit contracts for deconvolution method submodules (lightweight, mocked)."""

from __future__ import annotations

import builtins
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chatspatial.tools.deconvolution import cell2location as c2l_module
from chatspatial.tools.deconvolution import destvi as destvi_module
from chatspatial.tools.deconvolution import flashdeconv as flash_module
from chatspatial.tools.deconvolution import stereoscope as stereo_module
from chatspatial.tools.deconvolution import tangram as tangram_module
from chatspatial.tools.deconvolution.base import PreparedDeconvolutionData
from chatspatial.utils.exceptions import DataError, DependencyError, ProcessingError


class DummyCtx:
    def __init__(self):
        self.warnings: list[str] = []

    async def warning(self, msg: str):
        self.warnings.append(msg)


def _prepared_data(minimal_spatial_adata, *, n_types: int = 2) -> PreparedDeconvolutionData:
    spatial = minimal_spatial_adata.copy()
    reference = minimal_spatial_adata.copy()
    labels = ["A", "B"] if n_types == 2 else ["A"]
    reference.obs["cell_type"] = [labels[i % len(labels)] for i in range(reference.n_obs)]
    return PreparedDeconvolutionData(
        spatial=spatial,
        reference=reference,
        cell_type_key="cell_type",
        cell_types=labels,
        common_genes=list(spatial.var_names),
        spatial_coords=spatial.obsm["spatial"],
        ctx=DummyCtx(),
    )


def test_flashdeconv_dependency_error(minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch):
    data = _prepared_data(minimal_spatial_adata)
    monkeypatch.setattr(flash_module, "is_available", lambda *_: False)
    with pytest.raises(DependencyError, match="FlashDeconv is not available"):
        flash_module.deconvolve(data)


def test_flashdeconv_success_with_fake_backend(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    data = _prepared_data(minimal_spatial_adata)
    monkeypatch.setattr(flash_module, "is_available", lambda *_: True)

    def _fake_run(adata_st, _reference, **_kwargs):
        adata_st.obsm["flashdeconv"] = np.tile(np.array([0.7, 0.3]), (adata_st.n_obs, 1))

    fake_mod = ModuleType("flashdeconv")
    fake_mod.tl = SimpleNamespace(deconvolve=_fake_run)
    monkeypatch.setitem(__import__("sys").modules, "flashdeconv", fake_mod)

    proportions, stats = flash_module.deconvolve(data)
    assert proportions.shape == (data.n_spots, 2)
    assert list(proportions.columns) == data.cell_types
    assert stats["method"] == "FlashDeconv"


def test_tangram_dependency_error_is_wrapped(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    data = _prepared_data(minimal_spatial_adata)
    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "tangram":
            raise ImportError("missing tangram")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(DependencyError, match="tangram-sc is required"):
        tangram_module.deconvolve(data)


def test_tangram_success_with_fake_module(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    data = _prepared_data(minimal_spatial_adata)

    class _Map:
        def __init__(self, X):
            self.X = X

    def _pp_adatas(_ref, _spatial, genes):
        assert len(genes) > 0

    def _map_cells_to_space(ref_data, spatial_data, **_kwargs):
        # mapping matrix: n_ref_cells x n_spots
        return _Map(np.ones((ref_data.n_obs, spatial_data.n_obs), dtype=float))

    fake_mod = ModuleType("tangram")
    fake_mod.pp_adatas = _pp_adatas
    fake_mod.map_cells_to_space = _map_cells_to_space
    monkeypatch.setitem(__import__("sys").modules, "tangram", fake_mod)

    proportions, stats = tangram_module.deconvolve(data, mode="clusters", n_epochs=5)
    assert proportions.shape[0] == data.n_spots
    assert set(proportions.columns) == {"A", "B"}
    assert np.allclose(proportions.sum(axis=1).values, 1.0)
    assert stats["method"] == "Tangram"


def test_destvi_dependency_error(minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch):
    data = _prepared_data(minimal_spatial_adata)
    monkeypatch.setattr(destvi_module, "is_available", lambda *_: False)
    with pytest.raises(DependencyError, match="scvi-tools is required"):
        destvi_module.deconvolve(data)


def test_destvi_validates_minimum_cell_types(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    data = _prepared_data(minimal_spatial_adata, n_types=1)
    monkeypatch.setattr(destvi_module, "is_available", lambda *_: True)
    monkeypatch.setitem(__import__("sys").modules, "scvi", ModuleType("scvi"))

    with pytest.raises(DataError, match="at least 2 cell types"):
        destvi_module.deconvolve(data)


def test_destvi_success_with_fake_scvi(minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch):
    data = _prepared_data(minimal_spatial_adata)
    monkeypatch.setattr(destvi_module, "is_available", lambda *_: True)

    class FakeCondSCVI:
        @staticmethod
        def setup_anndata(*_args, **_kwargs):
            return None

        def __init__(self, _ref, **_kwargs):
            self.history = None

        def train(self, **_kwargs):
            return None

    class FakeDestVI:
        @staticmethod
        def setup_anndata(*_args, **_kwargs):
            return None

        @classmethod
        def from_rna_model(cls, spatial_data, _cond_model, **_kwargs):
            inst = cls()
            inst._n = spatial_data.n_obs
            return inst

        def train(self, **_kwargs):
            return None

        def get_proportions(self):
            return pd.DataFrame(
                np.tile([0.55, 0.45], (self._n, 1)),
                columns=["A", "B"],
            )

    fake_scvi = ModuleType("scvi")
    fake_scvi.model = SimpleNamespace(CondSCVI=FakeCondSCVI, DestVI=FakeDestVI)
    monkeypatch.setitem(__import__("sys").modules, "scvi", fake_scvi)

    proportions, stats = destvi_module.deconvolve(data, n_epochs=20, use_gpu=False)
    assert proportions.shape == (data.n_spots, 2)
    assert stats["method"] == "DestVI"
    assert stats["n_cell_types"] == 2


def test_stereoscope_success_with_fake_scvi_external(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    data = _prepared_data(minimal_spatial_adata)
    data.reference.obs["cell_type"] = data.reference.obs["cell_type"].astype(str)

    class FakeRNAStereoscope:
        @staticmethod
        def setup_anndata(*_args, **_kwargs):
            return None

        def __init__(self, _ref):
            return None

        def train(self, **_kwargs):
            return None

    class FakeSpatialStereoscope:
        @staticmethod
        def setup_anndata(*_args, **_kwargs):
            return None

        @classmethod
        def from_rna_model(cls, spatial_data, _rna_model):
            inst = cls()
            inst._n = spatial_data.n_obs
            return inst

        def train(self, **_kwargs):
            return None

        def get_proportions(self):
            return np.tile(np.array([0.8, 0.2]), (self._n, 1))

    scvi_mod = ModuleType("scvi")
    external_mod = ModuleType("scvi.external")
    external_mod.RNAStereoscope = FakeRNAStereoscope
    external_mod.SpatialStereoscope = FakeSpatialStereoscope
    monkeypatch.setitem(__import__("sys").modules, "scvi", scvi_mod)
    monkeypatch.setitem(__import__("sys").modules, "scvi.external", external_mod)

    proportions, stats = stereo_module.deconvolve(data, n_epochs=100, use_gpu=False)
    assert proportions.shape == (data.n_spots, 2)
    assert stats["method"] == "Stereoscope"
    assert stats["rna_epochs"] == 50
    assert stats["spatial_epochs"] == 50


@pytest.mark.asyncio
async def test_cell2location_apply_gene_filtering_unavailable_returns_copy(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx()
    monkeypatch.setattr(c2l_module, "is_available", lambda *_: False)

    out = await c2l_module.apply_gene_filtering(adata, ctx)
    assert out is not adata
    assert out.n_obs == adata.n_obs
    assert out.n_vars == adata.n_vars
    assert any("Skipping gene filtering" in m for m in ctx.warnings)


def test_cell2location_build_train_kwargs_variants():
    aggressive = c2l_module._build_train_kwargs(
        epochs=10,
        lr=0.01,
        train_size=0.8,
        accelerator="gpu",
        early_stopping=True,
        early_stopping_patience=5,
        validation_size=0.2,
        use_aggressive=True,
    )
    assert aggressive["accelerator"] == "gpu"
    assert aggressive["check_val_every_n_epoch"] == 1
    assert aggressive["train_size"] == pytest.approx(0.8)

    standard = c2l_module._build_train_kwargs(
        epochs=10,
        lr=0.01,
        train_size=0.7,
        accelerator="cpu",
        early_stopping=False,
        early_stopping_patience=5,
        validation_size=0.1,
        use_aggressive=False,
    )
    assert standard["max_epochs"] == 10
    assert standard["train_size"] == pytest.approx(0.7)
    assert "accelerator" not in standard


def test_cell2location_extract_reference_signatures_and_abundance(minimal_spatial_adata):
    ref = minimal_spatial_adata.copy()
    ref.uns["mod"] = {"factor_names": ["A", "B"]}
    ref.var["means_per_cluster_mu_fg_A"] = np.ones(ref.n_vars)
    ref.var["means_per_cluster_mu_fg_B"] = np.full(ref.n_vars, 2.0)

    sig = c2l_module._extract_reference_signatures(ref)
    assert list(sig.columns) == ["A", "B"]
    assert sig.shape[0] == ref.n_vars

    sp = minimal_spatial_adata.copy()
    sp.obsm["q05_cell_abundance_w_sf"] = pd.DataFrame(
        np.tile([0.6, 0.4], (sp.n_obs, 1)),
        index=sp.obs_names,
    )
    abundance = c2l_module._extract_cell_abundance(sp)
    assert abundance.shape == (sp.n_obs, 2)


def test_cell2location_extract_cell_abundance_missing_key_raises(minimal_spatial_adata):
    with pytest.raises(ProcessingError, match="did not produce expected output"):
        c2l_module._extract_cell_abundance(minimal_spatial_adata.copy())
