"""Unit tests for persistence utilities."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from chatspatial.utils import persistence


def _make_adata(n_obs: int = 8, n_vars: int = 5):
    X = np.arange(n_obs * n_vars, dtype=np.float32).reshape(n_obs, n_vars)
    adata = ad.AnnData(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    return adata


def test_get_active_path_uses_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    p = persistence.get_active_path("demo")
    assert p == tmp_path / ".chatspatial" / "active" / "demo.h5ad"


def test_export_and_load_roundtrip_with_custom_path(tmp_path: Path):
    adata = _make_adata()
    out = tmp_path / "exported" / "roundtrip.h5ad"

    exported = persistence.export_adata("d1", adata, out)
    loaded = persistence.load_adata_from_active("d1", out)

    assert exported == out
    assert exported.exists()
    assert loaded.shape == adata.shape


def test_load_adata_missing_file_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        persistence.load_adata_from_active("missing", tmp_path / "missing.h5ad")


def test_export_uses_active_path_when_custom_path_not_provided(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    adata = _make_adata()

    exported = persistence.export_adata("active_ds", adata)

    assert exported == tmp_path / ".chatspatial" / "active" / "active_ds.h5ad"
    assert exported.exists()


def test_export_wraps_underlying_write_errors(tmp_path: Path):
    class _BrokenAnnData:
        def write_h5ad(self, *_args, **_kwargs):
            raise RuntimeError("disk full")

    with pytest.raises(IOError, match="Failed to export data"):
        persistence.export_adata("broken", _BrokenAnnData(), tmp_path / "x" / "broken.h5ad")


def test_load_adata_uses_active_path_when_custom_path_not_provided(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    adata = _make_adata()
    persistence.export_adata("active_load", adata)

    loaded = persistence.load_adata_from_active("active_load")

    assert loaded.shape == adata.shape


def test_load_adata_wraps_underlying_read_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    corrupt_path = tmp_path / "broken.h5ad"
    corrupt_path.write_bytes(b"corrupt")

    import anndata

    monkeypatch.setattr(
        anndata,
        "read_h5ad",
        lambda _path: (_ for _ in ()).throw(RuntimeError("bad format")),
    )

    with pytest.raises(IOError, match="Failed to load data"):
        persistence.load_adata_from_active("unused", corrupt_path)
