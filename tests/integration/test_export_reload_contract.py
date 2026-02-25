"""Integration contract tests for export_data and reload_data tools."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import pytest

from chatspatial.server import data_manager, export_data, reload_data
from chatspatial.utils.exceptions import DataNotFoundError
from tests.fixtures.helpers import load_generic_dataset


def _extract_export_path(msg: str) -> Path:
    # "Dataset 'data_1' exported to: /path/to/file.h5ad"
    return Path(msg.split(" exported to: ", 1)[1].strip())


class DummyContext:
    """Minimal async context mock to verify info/error logging behavior."""

    def __init__(self):
        self.info_logs: list[str] = []
        self.error_logs: list[str] = []

    async def info(self, msg: str):
        self.info_logs.append(msg)

    async def error(self, msg: str):
        self.error_logs.append(msg)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_export_data_writes_file_and_returns_path(
    spatial_dataset_path, tmp_path, reset_data_manager
):
    dataset = await load_generic_dataset(spatial_dataset_path, name="export_contract")

    target = tmp_path / "exports" / "dataset_export.h5ad"
    msg = await export_data(dataset.id, path=str(target))

    out = _extract_export_path(msg)
    assert out.exists()
    assert out == target.resolve()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reload_data_applies_external_changes(
    spatial_dataset_path, tmp_path, reset_data_manager
):
    dataset = await load_generic_dataset(spatial_dataset_path, name="reload_contract")

    target = tmp_path / "exports" / "dataset_reload.h5ad"
    await export_data(dataset.id, path=str(target))

    # Simulate external script modification: subset genes and write back
    adata = ad.read_h5ad(target)
    adata_mod = adata[:, :7].copy()
    adata_mod.write_h5ad(target)

    msg = await reload_data(dataset.id, path=str(target))
    assert "reloaded" in msg
    assert "× 7 genes" in msg

    stored = await data_manager.get_dataset(dataset.id)
    assert stored["adata"].n_vars == 7


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reload_data_emits_context_info_logs_on_success(
    spatial_dataset_path, tmp_path, reset_data_manager
):
    dataset = await load_generic_dataset(spatial_dataset_path, name="reload_ctx_success")

    target = tmp_path / "exports" / "dataset_reload_ctx.h5ad"
    await export_data(dataset.id, path=str(target))

    adata = ad.read_h5ad(target)
    adata_mod = adata[:, :5].copy()
    adata_mod.write_h5ad(target)

    ctx = DummyContext()
    msg = await reload_data(dataset.id, path=str(target), context=ctx)

    assert "reloaded" in msg
    assert len(ctx.info_logs) == 2
    assert "Reloading dataset" in ctx.info_logs[0]
    assert "reloaded successfully" in ctx.info_logs[1]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reload_data_missing_file_raises_file_not_found(
    spatial_dataset_path, tmp_path, reset_data_manager
):
    dataset = await load_generic_dataset(spatial_dataset_path, name="reload_missing")

    missing = tmp_path / "does_not_exist.h5ad"
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        await reload_data(dataset.id, path=str(missing))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_export_data_missing_dataset_raises_not_found(reset_data_manager):
    with pytest.raises(DataNotFoundError, match="Dataset absent_data not found"):
        await export_data("absent_data")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_export_data_emits_context_info_logs(
    spatial_dataset_path, tmp_path, reset_data_manager
):
    dataset = await load_generic_dataset(spatial_dataset_path, name="export_ctx")
    target = tmp_path / "exports" / "ctx_export.h5ad"
    ctx = DummyContext()

    msg = await export_data(dataset.id, path=str(target), context=ctx)

    assert "exported to:" in msg
    assert len(ctx.info_logs) == 2
    assert "Exporting dataset" in ctx.info_logs[0]
    assert "Dataset exported to:" in ctx.info_logs[1]
    assert ctx.error_logs == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_export_data_logs_context_error_when_persistence_fails(
    reset_data_manager, monkeypatch: pytest.MonkeyPatch
):
    class BoomError(RuntimeError):
        pass

    async def fake_get_dataset(data_id: str):
        class FakeAdata:
            n_obs = 1
            n_vars = 1

        return {"adata": FakeAdata()}

    def fake_export_adata(data_id, adata, path):
        raise BoomError("boom")

    monkeypatch.setattr(data_manager, "get_dataset", fake_get_dataset)
    monkeypatch.setitem(
        sys.modules,
        "chatspatial.utils.persistence",
        SimpleNamespace(export_adata=fake_export_adata),
    )

    ctx = DummyContext()
    with pytest.raises(BoomError, match="boom"):
        await export_data("d_boom", context=ctx)

    assert any("Failed to export dataset: boom" in line for line in ctx.error_logs)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reload_data_logs_context_error_for_missing_file(
    spatial_dataset_path, tmp_path, reset_data_manager
):
    dataset = await load_generic_dataset(spatial_dataset_path, name="reload_ctx_missing")
    ctx = DummyContext()
    missing = tmp_path / "missing_reload_file.h5ad"

    with pytest.raises(FileNotFoundError, match="Data file not found"):
        await reload_data(dataset.id, path=str(missing), context=ctx)

    assert any("Data file not found" in line for line in ctx.error_logs)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reload_data_logs_context_error_for_generic_failure(
    reset_data_manager, monkeypatch: pytest.MonkeyPatch
):
    def fake_load_adata_from_active(data_id, path):
        raise RuntimeError("unexpected reload failure")

    monkeypatch.setitem(
        sys.modules,
        "chatspatial.utils.persistence",
        SimpleNamespace(load_adata_from_active=fake_load_adata_from_active),
    )

    ctx = DummyContext()
    with pytest.raises(RuntimeError, match="unexpected reload failure"):
        await reload_data("d_fail", path="/tmp/not_used.h5ad", context=ctx)

    assert any(
        "Failed to reload dataset: unexpected reload failure" in line
        for line in ctx.error_logs
    )
