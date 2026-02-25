"""Unit contracts for data loader helpers and generic loading flow."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from chatspatial.utils import data_loader as dl
from chatspatial.utils.exceptions import DataCompatibilityError, ParameterError, ProcessingError


@pytest.mark.asyncio
async def test_load_spatial_data_rejects_missing_path():
    with pytest.raises(FileNotFoundError):
        await dl.load_spatial_data("/path/does/not/exist.h5ad", "generic")


@pytest.mark.asyncio
async def test_load_spatial_data_generic_wraps_read_error(tmp_path: Path, monkeypatch):
    path = tmp_path / "demo.h5ad"
    path.write_text("placeholder")

    fake_scanpy = ModuleType("scanpy")
    fake_scanpy.read_h5ad = lambda _p: (_ for _ in ()).throw(RuntimeError("bad file"))
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)

    with pytest.raises(ProcessingError, match="Error loading generic data"):
        await dl.load_spatial_data(str(path), "generic")


@pytest.mark.asyncio
async def test_load_spatial_data_generic_sets_raw_and_counts_layer(
    minimal_spatial_adata, tmp_path: Path, monkeypatch
):
    adata = minimal_spatial_adata.copy()
    path = tmp_path / "sample.h5ad"
    path.write_text("placeholder")

    fake_scanpy = ModuleType("scanpy")
    fake_scanpy.read_h5ad = lambda _p: adata
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)

    result = await dl.load_spatial_data(str(path), "generic")
    out = result["adata"]

    assert result["name"] == "sample"
    assert out.raw is not None
    assert "counts" in out.layers
    assert result["spatial_coordinates_available"] is True


@pytest.mark.asyncio
async def test_load_spatial_data_rejects_unsupported_platform(tmp_path: Path, monkeypatch):
    path = tmp_path / "sample.h5ad"
    path.write_text("placeholder")
    monkeypatch.setitem(sys.modules, "scanpy", ModuleType("scanpy"))

    with pytest.raises(ParameterError, match="Unsupported platform type"):
        await dl.load_spatial_data(str(path), "unsupported")


def test_find_spatial_folder_requires_expected_files(tmp_path: Path):
    h5 = tmp_path / "x.h5"
    h5.write_text("x")
    spatial_dir = tmp_path / "spatial"
    spatial_dir.mkdir()

    assert dl._find_spatial_folder(str(h5)) is None

    (spatial_dir / "tissue_positions_list.csv").write_text("barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n")
    (spatial_dir / "scalefactors_json.json").write_text("{}")
    found = dl._find_spatial_folder(str(h5))
    assert found is not None
    assert Path(found).name == "spatial"


def test_add_spatial_info_handles_barcode_suffix_alignment(
    minimal_spatial_adata, tmp_path: Path, monkeypatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs_names = [f"bc{i}-1" for i in range(adata.n_obs)]

    spatial_dir = tmp_path / "sample_a" / "spatial"
    spatial_dir.mkdir(parents=True)

    positions = pd.DataFrame(
        {
            "barcode": [f"bc{i}" for i in range(adata.n_obs)],
            "in_tissue": [1] * adata.n_obs,
            "array_row": list(range(adata.n_obs)),
            "array_col": list(range(adata.n_obs)),
            "pxl_row_in_fullres": np.linspace(0, 10, adata.n_obs),
            "pxl_col_in_fullres": np.linspace(5, 15, adata.n_obs),
        }
    )
    positions.to_csv(spatial_dir / "tissue_positions_list.csv", index=False)
    (spatial_dir / "scalefactors_json.json").write_text(json.dumps({"spot_diameter_fullres": 10}))

    monkeypatch.setattr(dl, "is_available", lambda _name: False)

    out = dl._add_spatial_info_to_adata(adata, str(spatial_dir))
    assert "spatial" in out.obsm
    assert out.obsm["spatial"].shape[1] == 2
    assert "spatial" in out.uns and "sample_a" in out.uns["spatial"]


def test_add_spatial_info_rejects_invalid_positions_format(
    minimal_spatial_adata, tmp_path: Path, monkeypatch
):
    adata = minimal_spatial_adata.copy()
    spatial_dir = tmp_path / "sample_b" / "spatial"
    spatial_dir.mkdir(parents=True)
    (spatial_dir / "tissue_positions_list.csv").write_text("a,b,c\n1,2,3\n")
    (spatial_dir / "scalefactors_json.json").write_text("{}")
    monkeypatch.setattr(dl, "is_available", lambda _name: False)

    with pytest.raises(DataCompatibilityError, match="Unexpected tissue positions format"):
        dl._add_spatial_info_to_adata(adata, str(spatial_dir))

