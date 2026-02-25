"""Unit tests for spatial_registration routing and MCP wrapper contracts."""

from __future__ import annotations

import numpy as np
import pytest

from chatspatial.models.data import RegistrationParameters
from chatspatial.tools import spatial_registration as reg
from chatspatial.utils.exceptions import ParameterError, ProcessingError


class DummyCtx:
    def __init__(self, datasets: dict[str, object]):
        self.datasets = datasets

    async def get_adata(self, data_id: str):
        return self.datasets[data_id]


def test_validate_spatial_coords_raises_for_missing_spatial(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    del adata.obsm["spatial"]
    with pytest.raises(ParameterError, match="missing spatial coordinates"):
        reg._validate_spatial_coords([adata])


def test_register_slices_requires_at_least_two_slices(minimal_spatial_adata):
    with pytest.raises(ParameterError, match="at least 2 slices"):
        reg.register_slices([minimal_spatial_adata.copy()], RegistrationParameters())


def test_register_slices_dispatches_to_paste_and_stalign(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    ad1 = minimal_spatial_adata.copy()
    ad2 = minimal_spatial_adata.copy()
    calls: list[str] = []

    def _fake_paste(adata_list, params, spatial_key="spatial"):
        calls.append(f"paste:{spatial_key}")
        return adata_list

    def _fake_stalign(adata_list, params, spatial_key="spatial"):
        calls.append(f"stalign:{spatial_key}")
        return adata_list

    monkeypatch.setattr(reg, "_register_paste", _fake_paste)
    monkeypatch.setattr(reg, "_register_stalign", _fake_stalign)

    out1 = reg.register_slices([ad1, ad2], RegistrationParameters(method="paste"))
    out2 = reg.register_slices([ad1, ad2], RegistrationParameters(method="stalign"))
    assert out1[0] is ad1
    assert out2[0] is ad1
    assert calls == ["paste:spatial", "stalign:spatial"]


@pytest.mark.asyncio
async def test_register_spatial_slices_mcp_happy_path_records_metadata(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    src = minimal_spatial_adata.copy()
    tgt = minimal_spatial_adata.copy()
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(reg, "require", lambda *_args, **_kwargs: None)

    def _fake_register_slices(adata_list, params):
        for i, adata in enumerate(adata_list):
            adata.obsm["spatial_registered"] = adata.obsm["spatial"] + i
        return adata_list

    monkeypatch.setattr(reg, "register_slices", _fake_register_slices)
    monkeypatch.setattr(
        reg,
        "store_analysis_metadata",
        lambda _adata, **kwargs: captured.append(kwargs),
    )
    monkeypatch.setattr(reg, "export_analysis_result", lambda *_args, **_kwargs: [])

    out = await reg.register_spatial_slices_mcp(
        "src",
        "tgt",
        DummyCtx({"src": src, "tgt": tgt}),
        method="paste",
    )
    assert out["registration_completed"] is True
    assert out["spatial_key_registered"] == "spatial_registered"
    assert len(captured) == 2
    assert captured[0]["analysis_name"] == "registration_paste"
    assert captured[0]["results_keys"] == {"obsm": ["spatial_registered"]}


@pytest.mark.asyncio
async def test_register_spatial_slices_mcp_wraps_runtime_errors(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    src = minimal_spatial_adata.copy()
    tgt = minimal_spatial_adata.copy()
    monkeypatch.setattr(reg, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        reg,
        "register_slices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(ProcessingError, match="Registration failed: boom"):
        await reg.register_spatial_slices_mcp(
            "src",
            "tgt",
            DummyCtx({"src": src, "tgt": tgt}),
            method="paste",
        )


def test_get_common_genes_handles_duplicate_gene_names(minimal_spatial_adata):
    ad1 = minimal_spatial_adata[:, :4].copy()
    ad2 = minimal_spatial_adata[:, 2:6].copy()

    ad1.var_names = ["g0", "g0", "g2", "g3"]
    ad2.var_names = ["g2", "g3", "g4", "g4"]

    common = reg._get_common_genes([ad1, ad2])

    assert set(common) == {"g2", "g3"}


def test_transform_coordinates_handles_zero_rows_without_nan():
    transport = np.array([[0.0, 0.0], [0.2, 0.8]], dtype=float)
    ref = np.array([[1.0, 1.0], [3.0, 5.0]], dtype=float)

    out = reg._transform_coordinates(transport, ref)

    assert out.shape == (2, 2)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[1], np.array([2.6, 4.2]))


def test_register_slices_unknown_method_raises_parameter_error(minimal_spatial_adata):
    ad1 = minimal_spatial_adata.copy()
    ad2 = minimal_spatial_adata.copy()
    params = RegistrationParameters(method="paste").model_copy(update={"method": "unknown"})

    with pytest.raises(ParameterError, match="Unknown method"):
        reg.register_slices([ad1, ad2], params)


def test_register_stalign_rejects_non_pairwise_input(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import sys
    import types

    ad1 = minimal_spatial_adata.copy()
    ad2 = minimal_spatial_adata.copy()
    ad3 = minimal_spatial_adata.copy()

    fake_st = types.ModuleType("STalign.STalign")
    pkg = types.ModuleType("STalign")
    pkg.STalign = fake_st
    monkeypatch.setitem(sys.modules, "STalign", pkg)
    monkeypatch.setitem(sys.modules, "STalign.STalign", fake_st)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(float32="float32", tensor=lambda x, dtype=None: x),
    )

    with pytest.raises(ParameterError, match="only supports pairwise registration"):
        reg._register_stalign([ad1, ad2, ad3], RegistrationParameters(method="stalign"))


def test_register_stalign_invalid_transform_payload_is_wrapped(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import sys
    import types

    ad1 = minimal_spatial_adata.copy()
    ad2 = minimal_spatial_adata.copy()

    monkeypatch.setattr(
        reg,
        "_prepare_stalign_image",
        lambda *_a, **_k: ([0, 1], "img"),
    )
    monkeypatch.setattr(reg, "get_device", lambda prefer_gpu=False: "cpu")

    fake_torch = types.SimpleNamespace(float32="float32", tensor=lambda x, dtype=None: np.asarray(x), Tensor=np.ndarray)

    fake_st = types.ModuleType("STalign.STalign")
    fake_st.LDDMM = lambda **_kwargs: {"A": None, "v": None, "xv": None}
    fake_st.transform_points_source_to_target = lambda xv, v, A, points: points

    pkg = types.ModuleType("STalign")
    pkg.STalign = fake_st

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "STalign", pkg)
    monkeypatch.setitem(sys.modules, "STalign.STalign", fake_st)

    with pytest.raises(ProcessingError, match="STalign registration failed"):
        reg._register_stalign([ad1, ad2], RegistrationParameters(method="stalign"))
