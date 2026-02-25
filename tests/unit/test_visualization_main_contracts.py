"""Contract tests for visualization main entrypoint behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chatspatial.tools.visualization import main as viz_main
from chatspatial.utils.exceptions import DataNotFoundError, ParameterError, ProcessingError


class _Ctx:
    def __init__(self, adata):
        self._adata = adata

    async def get_adata(self, _data_id: str):
        return self._adata


def _params(**overrides):
    base = {
        "plot_type": "feature",
        "subtype": None,
        "dpi": 120,
        "output_path": None,
        "output_format": "png",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


async def test_visualize_data_rejects_unknown_plot_type():
    with pytest.raises(ParameterError, match="Invalid plot_type"):
        await viz_main.visualize_data("d1", _Ctx(None), _params(plot_type="unknown"))


async def test_visualize_data_rejects_dataset_with_too_few_cells():
    adata = SimpleNamespace(n_obs=4, n_vars=8)
    with pytest.raises(DataNotFoundError, match="too few cells"):
        await viz_main.visualize_data("d1", _Ctx(adata), _params())


async def test_visualize_data_rejects_dataset_with_too_few_genes():
    adata = SimpleNamespace(n_obs=8, n_vars=4)
    with pytest.raises(DataNotFoundError, match="too few genes"):
        await viz_main.visualize_data("d1", _Ctx(adata), _params())


async def test_visualize_data_wraps_unexpected_handler_errors(monkeypatch):
    adata = SimpleNamespace(n_obs=8, n_vars=8)
    close_calls: list[str] = []

    async def _boom(_adata, _params, _ctx):
        raise RuntimeError("handler failed")

    monkeypatch.setattr(viz_main.sc.settings, "set_figure_params", lambda **_kwargs: None)
    monkeypatch.setattr(viz_main, "PLOT_HANDLERS", {"feature": _boom})
    monkeypatch.setattr(viz_main.plt, "close", lambda arg: close_calls.append(arg))

    with pytest.raises(ProcessingError, match="Failed to create feature visualization"):
        await viz_main.visualize_data("d1", _Ctx(adata), _params())

    assert close_calls == ["all"]


async def test_visualize_data_builds_plot_type_key_with_subtype(monkeypatch):
    adata = SimpleNamespace(n_obs=8, n_vars=8)
    captured = {"plot_type": None}

    async def _handler(_adata, _params, _ctx):
        return object()

    async def _optimize(fig, params, ctx, data_id=None, plot_type=None):
        assert fig is not None
        assert params.plot_type == "feature"
        assert data_id == "d1"
        captured["plot_type"] = plot_type
        return "saved"

    monkeypatch.setattr(viz_main.sc.settings, "set_figure_params", lambda **_kwargs: None)
    monkeypatch.setattr(viz_main, "PLOT_HANDLERS", {"feature": _handler})
    monkeypatch.setattr(viz_main, "optimize_fig_to_image_with_cache", _optimize)

    result = await viz_main.visualize_data(
        "d1", _Ctx(adata), _params(subtype="umap")
    )

    assert result == "saved"
    assert captured["plot_type"] == "feature_umap"
