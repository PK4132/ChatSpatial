"""Unit tests for cell communication visualization utilities and routing."""

from __future__ import annotations

import sys
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chatspatial.models.data import VisualizationParameters
from chatspatial.tools.cell_communication import CCCStorage, store_ccc_results
from chatspatial.tools.visualization import cell_comm as viz_cc
from chatspatial.utils.exceptions import DataNotFoundError, ParameterError, ProcessingError


class DummyCtx:
    def __init__(self):
        self.infos: list[str] = []

    async def info(self, msg: str):
        self.infos.append(msg)


def _store_cluster_ccc(adata, *, method: str = "cellphonedb") -> None:
    results = pd.DataFrame(
        {
            "T|B": [0.8, 0.2],
            "B|T": [0.1, 0.6],
        },
        index=["L1_R1", "L2-R2"],
    )
    pvals = pd.DataFrame(
        {
            "T|B": [0.05, 0.2],
            "B|T": [0.01, 0.3],
        },
        index=results.index,
    )
    storage = CCCStorage(
        method=method,
        analysis_type="cluster",
        species="human",
        database="cellphonedb",
        lr_pairs=["L1_R1", "L2_R2"],
        top_lr_pairs=["L1_R1"],
        n_pairs=2,
        n_significant=1,
        results=results,
        pvalues=pvals,
    )
    store_ccc_results(adata, storage)


def _mock_liana_results(include_scores: bool = True) -> pd.DataFrame:
    data: dict[str, list] = {
        "source": ["T", "B", "T"],
        "target": ["B", "T", "T"],
        "ligand_complex": ["L1", "L2", "L1"],
        "receptor_complex": ["R1", "R2", "R1"],
    }
    if include_scores:
        data["magnitude_rank"] = [0.1, 0.4, 0.2]
        data["lr_means"] = [0.9, 0.4, 0.7]
    return pd.DataFrame(data)


def _mock_cc_data(
    results: pd.DataFrame,
    *,
    analysis_type: str = "cluster",
    lr_pairs: list[str] | None = None,
    spatial_scores: np.ndarray | None = None,
):
    return viz_cc.CellCommunicationData(
        results=results,
        method="cellphonedb",
        analysis_type=analysis_type,
        lr_pairs=lr_pairs or ["L1_R1", "L2_R2"],
        pvalues=None,
        spatial_scores=spatial_scores,
        spatial_pvals=None,
        source_labels=["T", "B"],
        target_labels=["T", "B"],
        method_data=None,
    )


class _FakePlotnine:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    def draw(self):
        if self.should_fail:
            raise RuntimeError("draw failed")
        fig, _ = plt.subplots()
        return fig


def test_parse_lr_pair_supports_common_formats():
    assert viz_cc._parse_lr_pair("LIG_REC") == ("LIG", "REC")
    assert viz_cc._parse_lr_pair("LIG^REC") == ("LIG", "REC")
    assert viz_cc._parse_lr_pair("complex:LIG-REC") == ("LIG", "REC")
    assert viz_cc._parse_lr_pair("NO_SEPARATOR") == ("NO", "SEPARATOR")


def test_convert_to_liana_format_matrix_contract():
    results = pd.DataFrame(
        {
            "interacting_pair": ["CXCL12_CXCR4", "CCL5^CCR5"],
            "A|B": [1.2, 0.0],
            "B|A": [0.5, np.nan],
        }
    )
    pvals = pd.DataFrame({"A|B": [0.01, 0.2], "B|A": [0.03, 0.4]})

    out = viz_cc._convert_to_liana_format(results, pvals, method="cellphonedb")

    assert not out.empty
    assert {"source", "target", "ligand_complex", "receptor_complex"}.issubset(
        out.columns
    )
    assert set(out["source"]) == {"A", "B"}
    assert set(out["target"]) == {"A", "B"}
    assert out["lr_means"].min() >= 0


def test_convert_to_liana_format_cellchat_3d_contract():
    results = pd.DataFrame(
        {
            "interaction_name": ["L1_R1", "L2_R2"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
        }
    )
    prob = np.zeros((2, 2, 2), dtype=float)
    prob[0, 1, 0] = 0.6
    prob[1, 0, 1] = 0.2
    pval = np.ones((2, 2, 2), dtype=float)
    pval[0, 1, 0] = 0.01
    pval[1, 0, 1] = 0.2

    out = viz_cc._convert_to_liana_format(
        results=results,
        pvalues=None,
        method="cellchat_r",
        method_data={
            "prob_matrix": prob,
            "pval_matrix": pval,
            "cell_type_names": ["A", "B"],
        },
    )

    assert len(out) == 2
    assert set(out["ligand_complex"]) == {"L1", "L2"}
    assert out["magnitude_rank"].between(0, 1).all()


@pytest.mark.asyncio
async def test_get_cell_communication_data_requires_stored_results(minimal_spatial_adata):
    with pytest.raises(DataNotFoundError, match="No cell communication results found"):
        await viz_cc.get_cell_communication_data(minimal_spatial_adata)


@pytest.mark.asyncio
async def test_get_cell_communication_data_converts_and_extracts_labels(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    adata.obsm["ccc_spatial_scores"] = np.ones((adata.n_obs, 2))
    _store_cluster_ccc(adata)
    ctx = DummyCtx()

    data = await viz_cc.get_cell_communication_data(adata, context=ctx)

    assert data.method == "cellphonedb"
    assert not data.results.empty
    assert set(data.source_labels) == {"T", "B"}
    assert set(data.target_labels) == {"T", "B"}
    assert data.spatial_scores is not None
    assert any("converted to LIANA format" in m for m in ctx.infos)


@pytest.mark.asyncio
async def test_create_cell_communication_visualization_default_routes_to_dotplot(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    _store_cluster_ccc(adata)

    sentinel = object()
    called: dict[str, bool] = {}

    def fake_dotplot(*_args, **_kwargs):
        called["dotplot"] = True
        return sentinel

    monkeypatch.setattr(viz_cc, "_create_unified_dotplot", fake_dotplot)

    out = await viz_cc.create_cell_communication_visualization(
        adata, VisualizationParameters(plot_type="communication")
    )

    assert out is sentinel
    assert called["dotplot"] is True


@pytest.mark.asyncio
async def test_create_cell_communication_visualization_spatial_requires_spatial_analysis(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    _store_cluster_ccc(adata, method="liana")

    with pytest.raises(ParameterError, match="requires spatial analysis"):
        await viz_cc.create_cell_communication_visualization(
            adata,
            VisualizationParameters(plot_type="communication", subtype="spatial"),
        )


@pytest.mark.asyncio
async def test_create_cell_communication_visualization_unknown_subtype_error(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    _store_cluster_ccc(adata)

    with pytest.raises(ParameterError, match="Unknown visualization type"):
        await viz_cc.create_cell_communication_visualization(
            adata,
            VisualizationParameters(plot_type="communication", subtype="unknown"),
        )


def test_cellchat_3d_to_liana_format_handles_empty_and_normalization():
    empty = viz_cc._cellchat_3d_to_liana_format(
        results=pd.DataFrame({"interaction_name": ["L1_R1"]}),
        method_data={"pval_matrix": np.ones((1, 1, 1))},
    )
    assert empty.empty

    prob = np.zeros((2, 2, 1), dtype=float)
    prob[0, 1, 0] = 0.8
    prob[1, 0, 0] = 0.3
    pval = np.zeros((2, 2, 1), dtype=float)
    pval[0, 1, 0] = 2.0
    pval[1, 0, 0] = 4.0

    out = viz_cc._cellchat_3d_to_liana_format(
        results=pd.DataFrame(
            {
                "interaction_name": ["L1_R1"],
                "ligand": ["L1"],
                "receptor": ["R1"],
            }
        ),
        method_data={
            "prob_matrix": prob,
            "pval_matrix": pval,
            "cell_type_names": np.array(["A", "B"]),
        },
    )
    assert len(out) == 2
    assert out["magnitude_rank"].max() == pytest.approx(1.0)


def test_matrix_to_liana_format_parses_cell_pairs_and_normalizes():
    results = pd.DataFrame(
        {
            "interacting_pair": ["L1_R1", "L2-R2"],
            "A|B": [0.7, np.nan],
            "B|A": [0.5, 0.4],
        }
    )
    pvals = pd.DataFrame({"A|B": [3.0, 1.0], "B|A": [9.0, 6.0]})

    out = viz_cc._matrix_to_liana_format(results, pvals, method="cellphonedb")
    assert not out.empty
    assert set(out["source"]) == {"A", "B"}
    assert set(out["target"]) == {"A", "B"}
    assert out["magnitude_rank"].max() == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_create_cell_communication_visualization_defaults_to_spatial_for_spatial_analysis(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    sentinel = object()
    data = _mock_cc_data(
        _mock_liana_results(),
        analysis_type="spatial",
        spatial_scores=np.ones((adata.n_obs, 2), dtype=float),
    )

    async def _fake_get(*_args, **_kwargs):
        return data

    monkeypatch.setattr(viz_cc, "get_cell_communication_data", _fake_get)
    monkeypatch.setattr(viz_cc, "_create_spatial_lr_visualization", lambda *_a, **_k: sentinel)

    params = VisualizationParameters(plot_type="communication", subtype="dotplot")
    params.subtype = None
    out = await viz_cc.create_cell_communication_visualization(
        adata,
        params,
    )
    assert out is sentinel


@pytest.mark.asyncio
async def test_create_cell_communication_visualization_routes_tile_and_circle(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    data = _mock_cc_data(_mock_liana_results(), analysis_type="cluster")

    async def _fake_get(*_args, **_kwargs):
        return data

    sentinel_tile = object()
    sentinel_circle = object()

    monkeypatch.setattr(viz_cc, "get_cell_communication_data", _fake_get)
    monkeypatch.setattr(viz_cc, "_create_unified_tileplot", lambda *_a, **_k: sentinel_tile)
    monkeypatch.setattr(viz_cc, "_create_unified_circle_plot", lambda *_a, **_k: sentinel_circle)

    tile = await viz_cc.create_cell_communication_visualization(
        adata, VisualizationParameters(plot_type="communication", subtype="tileplot")
    )
    circle = await viz_cc.create_cell_communication_visualization(
        adata, VisualizationParameters(plot_type="communication", subtype="circle_plot")
    )
    assert tile is sentinel_tile
    assert circle is sentinel_circle


def test_create_spatial_lr_visualization_requires_scores(minimal_spatial_adata):
    data = _mock_cc_data(_mock_liana_results(), analysis_type="spatial", spatial_scores=None)
    with pytest.raises(DataNotFoundError, match="No spatial communication scores found"):
        viz_cc._create_spatial_lr_visualization(
            minimal_spatial_adata.copy(),
            data,
            VisualizationParameters(plot_type="communication", subtype="spatial"),
        )


def test_create_spatial_lr_visualization_uses_metric_and_handles_missing_pair_column(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    results = pd.DataFrame(
        {"morans": [0.3, 0.9]},
        index=["L1_R1", "L2_R2"],
    )
    data = _mock_cc_data(
        results,
        analysis_type="spatial",
        lr_pairs=["L1_R1", "L2_R2"],
        spatial_scores=np.ones((adata.n_obs, 1), dtype=float),
    )
    fig = viz_cc._create_spatial_lr_visualization(
        adata,
        data,
        VisualizationParameters(
            plot_type="communication",
            subtype="spatial",
            plot_top_pairs=2,
        ),
    )
    assert fig._suptitle is not None
    assert "Spatial Cell Communication" in fig._suptitle.get_text()
    fig.clf()


def test_create_unified_dotplot_validates_required_columns():
    bad = pd.DataFrame({"source": ["A"], "target": ["B"]})
    data = _mock_cc_data(bad)
    with pytest.raises(DataNotFoundError, match="Missing required columns"):
        viz_cc._create_unified_dotplot(
            data,
            VisualizationParameters(plot_type="communication", subtype="dotplot"),
        )


def test_create_unified_dotplot_rejects_when_no_score_columns(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(viz_cc, "require", lambda *_a, **_k: None)
    df = _mock_liana_results(include_scores=False)
    data = _mock_cc_data(df)

    fake_liana = ModuleType("liana")
    fake_liana.pl = ModuleType("liana.pl")
    fake_liana.pl.dotplot = lambda **_kwargs: _FakePlotnine()
    monkeypatch.setitem(sys.modules, "liana", fake_liana)

    with pytest.raises(DataNotFoundError, match="No suitable columns for visualization"):
        viz_cc._create_unified_dotplot(
            data,
            VisualizationParameters(plot_type="communication", subtype="dotplot"),
        )


def test_create_unified_dotplot_falls_back_on_liana_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(viz_cc, "require", lambda *_a, **_k: None)
    data = _mock_cc_data(_mock_liana_results())
    sentinel = object()

    fake_liana = ModuleType("liana")
    fake_liana.pl = ModuleType("liana.pl")

    def _boom(**_kwargs):
        raise RuntimeError("liana plot failure")

    fake_liana.pl.dotplot = _boom
    monkeypatch.setitem(sys.modules, "liana", fake_liana)
    monkeypatch.setattr(viz_cc, "_create_fallback_dotplot", lambda *_a, **_k: sentinel)

    out = viz_cc._create_unified_dotplot(
        data,
        VisualizationParameters(plot_type="communication", subtype="dotplot"),
    )
    assert out is sentinel


def test_create_fallback_dotplot_zero_rank_branch_sets_constant_size():
    df = _mock_liana_results()
    df["magnitude_rank"] = 0.0
    data = _mock_cc_data(df)

    fig = viz_cc._create_fallback_dotplot(
        data,
        VisualizationParameters(plot_type="communication", subtype="dotplot"),
    )
    sizes = fig.axes[0].collections[0].get_sizes()
    assert np.allclose(sizes, 100.0)
    fig.clf()


def test_create_unified_tileplot_requires_value_columns(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(viz_cc, "require", lambda *_a, **_k: None)
    df = _mock_liana_results(include_scores=False)
    data = _mock_cc_data(df)

    fake_liana = ModuleType("liana")
    fake_liana.pl = ModuleType("liana.pl")
    fake_liana.pl.tileplot = lambda **_kwargs: _FakePlotnine()
    monkeypatch.setitem(sys.modules, "liana", fake_liana)

    with pytest.raises(DataNotFoundError, match="No suitable columns for tileplot"):
        viz_cc._create_unified_tileplot(
            data,
            VisualizationParameters(plot_type="communication", subtype="tileplot"),
        )


def test_create_unified_tileplot_falls_back_on_liana_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(viz_cc, "require", lambda *_a, **_k: None)
    data = _mock_cc_data(_mock_liana_results())
    sentinel = object()

    fake_liana = ModuleType("liana")
    fake_liana.pl = ModuleType("liana.pl")

    def _boom(**_kwargs):
        raise RuntimeError("tileplot failure")

    fake_liana.pl.tileplot = _boom
    monkeypatch.setitem(sys.modules, "liana", fake_liana)
    monkeypatch.setattr(viz_cc, "_create_fallback_tileplot", lambda *_a, **_k: sentinel)

    out = viz_cc._create_unified_tileplot(
        data,
        VisualizationParameters(plot_type="communication", subtype="tileplot"),
    )
    assert out is sentinel


def test_create_fallback_tileplot_requires_value_column():
    data = _mock_cc_data(_mock_liana_results(include_scores=False))
    with pytest.raises(DataNotFoundError, match="No suitable value column found"):
        viz_cc._create_fallback_tileplot(
            data,
            VisualizationParameters(plot_type="communication", subtype="tileplot"),
        )


def test_create_unified_circle_plot_supports_weighted_and_count_modes():
    weighted = _mock_cc_data(_mock_liana_results())
    fig = viz_cc._create_unified_circle_plot(
        weighted,
        VisualizationParameters(plot_type="communication", subtype="circle_plot"),
    )
    assert fig.axes[0].name == "polar"
    fig.clf()

    counts_df = _mock_liana_results(include_scores=False)
    counts = _mock_cc_data(counts_df)
    fig2 = viz_cc._create_unified_circle_plot(
        counts,
        VisualizationParameters(plot_type="communication", subtype="circle_plot"),
    )
    assert fig2.axes[0].name == "polar"
    fig2.clf()


def test_plotnine_to_matplotlib_success_and_error_wrap():
    fig = viz_cc._plotnine_to_matplotlib(
        _FakePlotnine(should_fail=False),
        VisualizationParameters(plot_type="communication", dpi=123),
    )
    assert fig.get_dpi() == 123
    fig.clf()

    with pytest.raises(ProcessingError, match="Failed to convert plotnine figure"):
        viz_cc._plotnine_to_matplotlib(
            _FakePlotnine(should_fail=True),
            VisualizationParameters(plot_type="communication"),
        )
