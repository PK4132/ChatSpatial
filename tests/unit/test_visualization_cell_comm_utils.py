"""Unit tests for cell communication visualization utilities and routing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chatspatial.models.data import VisualizationParameters
from chatspatial.tools.cell_communication import CCCStorage, store_ccc_results
from chatspatial.tools.visualization import cell_comm as viz_cc
from chatspatial.utils.exceptions import DataNotFoundError, ParameterError


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
