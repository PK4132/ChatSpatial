"""Unit tests for lightweight utilities in cell_communication module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chatspatial.models.data import CellCommunicationParameters
from chatspatial.tools import cell_communication as ccc
from chatspatial.tools.cell_communication import (
    CCCAutocrine,
    CCCStorage,
    _integrate_autocrine_detection,
    get_ccc_results,
    has_ccc_results,
    standardize_lr_pair,
    store_ccc_results,
)
from chatspatial.utils.exceptions import DataCompatibilityError, DataNotFoundError, DependencyError, ParameterError, ProcessingError


class DummyCtx:
    def __init__(self):
        self.warnings: list[str] = []

    async def warning(self, msg: str):
        self.warnings.append(msg)

    async def get_adata(self, _data_id: str):
        return None


def test_standardize_lr_pair_normalizes_separators():
    assert standardize_lr_pair("LIG^REC") == "LIG_REC"
    assert standardize_lr_pair("lig-rec") == "lig_rec"
    assert standardize_lr_pair("LIG_REC") == "LIG_REC"


def test_store_and_get_ccc_results_roundtrip(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    storage = CCCStorage(
        method="liana",
        analysis_type="cluster",
        species="human",
        database="consensus",
        lr_pairs=["L1_R1"],
        top_lr_pairs=["L1_R1"],
        n_pairs=1,
        n_significant=1,
        autocrine=CCCAutocrine(n_loops=1, top_pairs=["L1_R1"]),
    )

    store_ccc_results(adata, storage)
    assert has_ccc_results(adata) is True

    restored = get_ccc_results(adata)
    assert restored is not None
    assert restored.method == "liana"
    assert restored.n_pairs == 1
    assert restored.autocrine.n_loops == 1


def test_integrate_autocrine_detection_for_liana_cluster_results():
    results = pd.DataFrame(
        {
            "source": ["T", "T", "B"],
            "target": ["T", "B", "B"],
            "ligand_complex": ["L1", "L2", "L3"],
            "receptor_complex": ["R1", "R2", "R3"],
            "magnitude_rank": [0.01, 0.2, 0.03],
        }
    )
    storage = CCCStorage(
        method="liana",
        analysis_type="cluster",
        species="human",
        database="consensus",
        results=results,
    )

    _integrate_autocrine_detection(storage, n_top=5)

    assert storage.autocrine.n_loops == 2
    assert "L1_R1" in storage.autocrine.top_pairs
    assert "L3_R3" in storage.autocrine.top_pairs


def test_integrate_autocrine_detection_for_matrix_based_methods():
    # Simulate cellphonedb/fastccc matrix format: columns are source|target
    results = pd.DataFrame(
        {
            "T|T": [0.4, 0.0],
            "T|B": [0.1, 0.2],
            "B|B": [0.0, 0.9],
        },
        index=["L1_R1", "L2_R2"],
    )
    storage = CCCStorage(
        method="cellphonedb",
        analysis_type="cluster",
        species="human",
        database="cellphonedb",
        results=results,
    )

    _integrate_autocrine_detection(storage, n_top=5)

    assert storage.autocrine.n_loops == 2
    assert "L1_R1" in storage.autocrine.top_pairs
    assert "L2_R2" in storage.autocrine.top_pairs


@pytest.mark.asyncio
async def test_validate_ccc_params_requires_spatial_connectivity_for_liana(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)
    params = CellCommunicationParameters(
        method="liana",
        species="human",
        cell_type_key="cell_type",
        perform_spatial_analysis=True,
    )
    with pytest.raises(DataNotFoundError, match="Spatial connectivity required"):
        await ccc._validate_ccc_params(adata, params, DummyCtx())


@pytest.mark.asyncio
async def test_validate_ccc_params_warns_for_mouse_consensus_resource(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)
    ctx = DummyCtx()
    params = CellCommunicationParameters(
        method="liana",
        species="mouse",
        cell_type_key="cell_type",
        perform_spatial_analysis=False,
        liana_resource="consensus",
    )
    await ccc._validate_ccc_params(adata, params, ctx)
    assert any("mouseconsensus" in w for w in ctx.warnings)


@pytest.mark.asyncio
async def test_run_ccc_analysis_dispatches_to_fastccc(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)
    params = CellCommunicationParameters(
        method="fastccc",
        species="human",
        cell_type_key="cell_type",
    )
    expected = CCCStorage(
        method="fastccc",
        analysis_type="cluster",
        species="human",
        database="fastccc",
        n_pairs=2,
        n_significant=1,
    )

    async def _fake_fastccc(*_args, **_kwargs):
        return expected

    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ccc, "_analyze_communication_fastccc", _fake_fastccc)

    out = await ccc._run_ccc_analysis(adata, params, DummyCtx())
    assert out is expected


@pytest.mark.asyncio
async def test_analyze_cell_communication_happy_path_cluster(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)
    params = CellCommunicationParameters(
        method="fastccc",
        species="human",
        cell_type_key="cell_type",
    )
    ctx = DummyCtx()
    ctx.get_adata = lambda _data_id: adata  # type: ignore[method-assign]
    captured: dict[str, object] = {}

    async def _ok_get_adata(_data_id: str):
        return adata

    ctx.get_adata = _ok_get_adata  # type: ignore[method-assign]

    async def _fake_validate(*_args, **_kwargs):
        return None

    async def _fake_run(*_args, **_kwargs):
        return CCCStorage(
            method="fastccc",
            analysis_type="cluster",
            species="human",
            database="fastccc",
            lr_pairs=["L1_R1"],
            top_lr_pairs=["L1_R1"],
            n_pairs=1,
            n_significant=1,
            statistics={"ok": 1.0},
        )

    monkeypatch.setattr(ccc, "_validate_ccc_params", _fake_validate)
    monkeypatch.setattr(ccc, "_run_ccc_analysis", _fake_run)
    monkeypatch.setattr(
        "chatspatial.utils.adata_utils.store_analysis_metadata",
        lambda _adata, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "chatspatial.utils.results_export.export_analysis_result",
        lambda *_args, **_kwargs: [],
    )

    out = await ccc.analyze_cell_communication("d1", ctx, params)
    assert out.method == "fastccc"
    assert out.n_lr_pairs == 1
    assert out.results_key == "ccc"
    assert captured["analysis_name"] == "cell_communication_fastccc"
    assert captured["results_keys"] == {"obs": [], "obsm": [], "uns": ["ccc"]}


@pytest.mark.asyncio
async def test_analyze_cell_communication_wraps_unexpected_error(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)
    params = CellCommunicationParameters(
        method="fastccc",
        species="human",
        cell_type_key="cell_type",
    )

    async def _get_adata(_data_id: str):
        return adata

    ctx = DummyCtx()
    ctx.get_adata = _get_adata  # type: ignore[method-assign]

    async def _raise_validate(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ccc, "_validate_ccc_params", _raise_validate)

    with pytest.raises(ProcessingError, match="Error in cell communication analysis: boom"):
        await ccc.analyze_cell_communication("d1", ctx, params)


@pytest.mark.asyncio
async def test_analyze_cell_communication_passes_through_data_not_found(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)
    params = CellCommunicationParameters(
        method="fastccc",
        species="human",
        cell_type_key="cell_type",
    )

    async def _get_adata(_data_id: str):
        return adata

    async def _raise_not_found(*_args, **_kwargs):
        raise DataNotFoundError("missing neighbor graph")

    ctx = DummyCtx()
    ctx.get_adata = _get_adata  # type: ignore[method-assign]
    monkeypatch.setattr(ccc, "_validate_ccc_params", _raise_not_found)

    with pytest.raises(DataNotFoundError, match="missing neighbor graph"):
        await ccc.analyze_cell_communication("d1", ctx, params)


def test_get_liana_resource_name_maps_mouse_consensus_and_passthrough():
    assert ccc._get_liana_resource_name("mouse", "consensus") == "mouseconsensus"
    assert ccc._get_liana_resource_name("mouse", "cellphonedb") == "cellphonedb"
    assert ccc._get_liana_resource_name("human", "consensus") == "consensus"


@pytest.mark.asyncio
async def test_validate_ccc_params_cellphonedb_warns_low_gene_and_cell_counts(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)
    ctx = DummyCtx()

    monkeypatch.setattr(
        ccc,
        "get_raw_data_source",
        lambda _adata, prefer_complete_genes=True: type("Raw", (), {"var_names": list(range(300))})(),
    )

    await ccc._validate_ccc_params(
        adata,
        CellCommunicationParameters(
            method="cellphonedb",
            species="human",
            cell_type_key="cell_type",
        ),
        ctx,
    )

    assert any("Gene count" in w for w in ctx.warnings)
    assert any("Cell count" in w for w in ctx.warnings)


@pytest.mark.asyncio
async def test_run_ccc_analysis_unknown_method_raises_parameter_error(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)
    params = CellCommunicationParameters(
        method="fastccc", species="human", cell_type_key="cell_type"
    ).model_copy(update={"method": "unknown"})

    with pytest.raises(ParameterError, match="Unsupported method"):
        await ccc._run_ccc_analysis(adata, params, DummyCtx())


@pytest.mark.asyncio
async def test_analyze_cell_communication_spatial_writes_obsm_scores(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)
    params = CellCommunicationParameters(method="liana", species="human", cell_type_key="cell_type")

    async def _get_adata(_data_id: str):
        return adata

    ctx = DummyCtx()
    ctx.get_adata = _get_adata  # type: ignore[method-assign]
    captured: dict[str, object] = {}

    async def _fake_validate(*_args, **_kwargs):
        return None

    async def _fake_run(*_args, **_kwargs):
        return CCCStorage(
            method="liana",
            analysis_type="spatial",
            species="human",
            database="consensus",
            lr_pairs=["L1_R1"],
            top_lr_pairs=["L1_R1"],
            n_pairs=1,
            n_significant=1,
            method_data={
                "spatial_scores": pd.DataFrame({"x": [0.1] * adata.n_obs}, index=adata.obs_names),
                "spatial_pvals": pd.DataFrame({"x": [0.05] * adata.n_obs}, index=adata.obs_names),
            },
        )

    monkeypatch.setattr(ccc, "_validate_ccc_params", _fake_validate)
    monkeypatch.setattr(ccc, "_run_ccc_analysis", _fake_run)
    monkeypatch.setattr(
        "chatspatial.utils.adata_utils.store_analysis_metadata",
        lambda _adata, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "chatspatial.utils.results_export.export_analysis_result",
        lambda *_args, **_kwargs: [],
    )

    out = await ccc.analyze_cell_communication("d1", ctx, params)

    assert out.analysis_type == "spatial"
    assert ccc.CCC_SPATIAL_SCORES_KEY in adata.obsm
    assert ccc.CCC_SPATIAL_PVALS_KEY in adata.obsm
    assert captured["results_keys"]["obsm"] == [ccc.CCC_SPATIAL_SCORES_KEY]


@pytest.mark.asyncio
async def test_analyze_cell_communication_passes_through_data_compatibility_error(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)
    params = CellCommunicationParameters(method="fastccc", species="human", cell_type_key="cell_type")

    async def _get_adata(_data_id: str):
        return adata

    async def _raise_compat(*_args, **_kwargs):
        raise DataCompatibilityError("incompatible matrix")

    ctx = DummyCtx()
    ctx.get_adata = _get_adata  # type: ignore[method-assign]
    monkeypatch.setattr(ccc, "_validate_ccc_params", _raise_compat)

    with pytest.raises(DataCompatibilityError, match="incompatible matrix"):
        await ccc.analyze_cell_communication("d1", ctx, params)


def test_run_liana_cluster_analysis_builds_expected_storage(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * (adata.n_obs // 2) + ["B"] * (adata.n_obs - adata.n_obs // 2))

    def _rank_aggregate(_adata, **_kwargs):
        _adata.uns["liana_res"] = pd.DataFrame(
            {
                "source": ["T", "B"],
                "target": ["B", "T"],
                "ligand_complex": ["L1", "L2"],
                "receptor_complex": ["R1", "R2"],
                "magnitude_rank": [0.01, 0.2],
            }
        )

    fake_liana = type("L", (), {"mt": type("MT", (), {"rank_aggregate": staticmethod(_rank_aggregate)})})()
    monkeypatch.setitem(__import__("sys").modules, "liana", fake_liana)
    monkeypatch.setattr(
        ccc,
        "get_raw_data_source",
        lambda _adata, prefer_complete_genes=True: type("Raw", (), {"source": "raw"})(),
    )

    params = CellCommunicationParameters(
        method="liana",
        species="human",
        cell_type_key="cell_type",
        perform_spatial_analysis=False,
        plot_top_pairs=1,
        liana_significance_alpha=0.05,
    )

    out = ccc._run_liana_cluster_analysis(adata, params, DummyCtx())

    assert out.analysis_type == "cluster"
    assert out.n_pairs == 2
    assert out.n_significant == 1
    assert out.lr_pairs == ["L1_R1", "L2_R2"]
    assert out.top_lr_pairs == ["L1_R1"]
    assert out.statistics["use_raw"] is True


def test_run_liana_spatial_analysis_builds_expected_storage(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()

    var = pd.DataFrame(
        {
            "morans": [0.9, 0.3],
            "morans_pvals": [0.001, 0.3],
        },
        index=["L1^R1", "L2-R2"],
    )
    lrdata = type(
        "LR",
        (),
        {
            "n_vars": 2,
            "var": var,
            "X": np.array([[0.5, 0.2]] * adata.n_obs),
            "layers": {"pvals": np.array([[0.01, 0.2]] * adata.n_obs)},
        },
    )()

    fake_liana = type(
        "L",
        (),
        {"mt": type("MT", (), {"bivariate": staticmethod(lambda *_args, **_kwargs: lrdata)})},
    )()
    monkeypatch.setitem(__import__("sys").modules, "liana", fake_liana)

    params = CellCommunicationParameters(
        method="liana",
        species="human",
        liana_global_metric="morans",
        cell_type_key="cell_type",
        plot_top_pairs=1,
    )

    out = ccc._run_liana_spatial_analysis(adata, params, DummyCtx())

    assert out.analysis_type == "spatial"
    assert out.n_pairs == 2
    assert out.lr_pairs == ["L1_R1", "L2_R2"]
    assert out.top_lr_pairs == ["L1_R1"]
    assert out.method_data["spatial_scores"].shape == (adata.n_obs, 2)
    assert out.method_data["spatial_pvals"].shape == (adata.n_obs, 2)
    assert "morans_pvals_corrected" in out.results.columns
    assert "morans_significant" in out.results.columns


@pytest.mark.asyncio
async def test_analyze_communication_liana_auto_builds_spatial_neighbors(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)

    calls: dict[str, object] = {}

    def _spatial_neighbors(_adata, **kwargs):
        calls["neighbors_kwargs"] = kwargs
        _adata.obsp["spatial_connectivities"] = np.eye(_adata.n_obs)

    fake_sq = type("SQ", (), {"gr": type("GR", (), {"spatial_neighbors": staticmethod(_spatial_neighbors)})})()
    monkeypatch.setitem(__import__("sys").modules, "squidpy", fake_sq)
    monkeypatch.setitem(__import__("sys").modules, "liana", type("L", (), {})())
    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)

    expected = CCCStorage(
        method="liana",
        analysis_type="spatial",
        species="human",
        database="consensus",
        n_pairs=1,
        n_significant=1,
    )

    monkeypatch.setattr(ccc, "_run_liana_spatial_analysis", lambda *_args, **_kwargs: expected)

    params = CellCommunicationParameters(
        method="liana",
        species="human",
        cell_type_key="cell_type",
        perform_spatial_analysis=True,
    )

    out = await ccc._analyze_communication_liana(adata, params, DummyCtx())

    assert out is expected
    assert "neighbors_kwargs" in calls
    assert calls["neighbors_kwargs"]["coord_type"] == "generic"


@pytest.mark.asyncio
async def test_analyze_communication_liana_without_cluster_column_dispatches_spatial(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    if "cell_type" in adata.obs:
        del adata.obs["cell_type"]

    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(__import__("sys").modules, "liana", type("L", (), {})())

    expected = CCCStorage(
        method="liana",
        analysis_type="spatial",
        species="human",
        database="consensus",
        n_pairs=0,
        n_significant=0,
    )

    monkeypatch.setattr(ccc, "_run_liana_spatial_analysis", lambda *_args, **_kwargs: expected)

    params = CellCommunicationParameters(
        method="liana",
        species="human",
        cell_type_key="cell_type",
        perform_spatial_analysis=False,
    )

    out = await ccc._analyze_communication_liana(adata, params, DummyCtx())
    assert out is expected


@pytest.mark.asyncio
async def test_analyze_communication_liana_cluster_dispatch_when_cluster_present(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)

    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(__import__("sys").modules, "liana", type("L", (), {})())

    expected = CCCStorage(
        method="liana",
        analysis_type="cluster",
        species="human",
        database="consensus",
        n_pairs=1,
        n_significant=1,
    )

    monkeypatch.setattr(ccc, "_run_liana_cluster_analysis", lambda *_args, **_kwargs: expected)

    params = CellCommunicationParameters(
        method="liana",
        species="human",
        cell_type_key="cell_type",
        perform_spatial_analysis=False,
    )

    out = await ccc._analyze_communication_liana(adata, params, DummyCtx())
    assert out is expected


@pytest.mark.asyncio
async def test_analyze_communication_cellphonedb_rejects_non_human_early(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)

    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)

    fake_cpdb_method = type("CP", (), {"call": staticmethod(lambda **_kwargs: {})})
    import types, sys
    cellphonedb_pkg = types.ModuleType("cellphonedb")
    src_pkg = types.ModuleType("cellphonedb.src")
    core_pkg = types.ModuleType("cellphonedb.src.core")
    methods_pkg = types.ModuleType("cellphonedb.src.core.methods")
    methods_pkg.cpdb_statistical_analysis_method = fake_cpdb_method
    sys.modules["cellphonedb"] = cellphonedb_pkg
    sys.modules["cellphonedb.src"] = src_pkg
    sys.modules["cellphonedb.src.core"] = core_pkg
    sys.modules["cellphonedb.src.core.methods"] = methods_pkg

    with pytest.raises(ProcessingError, match="CellPhoneDB only supports human"):
        await ccc._analyze_communication_cellphonedb(
            adata,
            CellCommunicationParameters(
                method="cellphonedb",
                species="mouse",
                cell_type_key="cell_type",
            ),
            DummyCtx(),
        )


@pytest.mark.asyncio
async def test_analyze_communication_fastccc_rejects_non_human_early(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)

    with pytest.raises(ProcessingError, match="FastCCC only supports human"):
        await ccc._analyze_communication_fastccc(
            adata,
            CellCommunicationParameters(
                method="fastccc",
                species="mouse",
                cell_type_key="cell_type",
            ),
            DummyCtx(),
        )


@pytest.mark.asyncio
async def test_run_ccc_analysis_dispatches_to_liana(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)

    expected = CCCStorage(
        method="liana",
        analysis_type="cluster",
        species="human",
        database="consensus",
        n_pairs=1,
        n_significant=1,
    )

    async def _fake_liana(*_args, **_kwargs):
        return expected

    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ccc, "_analyze_communication_liana", _fake_liana)

    out = await ccc._run_ccc_analysis(
        adata,
        CellCommunicationParameters(method="liana", species="human", cell_type_key="cell_type"),
        DummyCtx(),
    )
    assert out is expected


@pytest.mark.asyncio
async def test_run_ccc_analysis_dispatches_to_cellphonedb_and_cellchat_r(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)

    cpdb_expected = CCCStorage(
        method="cellphonedb",
        analysis_type="cluster",
        species="human",
        database="cellphonedb",
        n_pairs=2,
        n_significant=1,
    )
    chat_expected = CCCStorage(
        method="cellchat_r",
        analysis_type="cluster",
        species="human",
        database="cellchatdb",
        n_pairs=2,
        n_significant=1,
    )

    async def _fake_cpdb(*_args, **_kwargs):
        return cpdb_expected

    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ccc, "validate_r_package", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ccc, "_analyze_communication_cellphonedb", _fake_cpdb)
    monkeypatch.setattr(ccc, "_analyze_communication_cellchat_r", lambda *_args, **_kwargs: chat_expected)

    cpdb_out = await ccc._run_ccc_analysis(
        adata,
        CellCommunicationParameters(method="cellphonedb", species="human", cell_type_key="cell_type"),
        DummyCtx(),
    )
    chat_out = await ccc._run_ccc_analysis(
        adata,
        CellCommunicationParameters(method="cellchat_r", species="human", cell_type_key="cell_type"),
        DummyCtx(),
    )

    assert cpdb_out is cpdb_expected
    assert chat_out is chat_expected


@pytest.mark.asyncio
async def test_create_microenvironments_file_returns_none_without_spatial(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    if "spatial" in adata.obsm:
        del adata.obsm["spatial"]
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)

    out = await ccc._create_microenvironments_file(
        adata,
        CellCommunicationParameters(
            method="cellphonedb",
            species="human",
            cell_type_key="cell_type",
            cellphonedb_spatial_radius=10.0,
        ),
        DummyCtx(),
    )
    assert out is None


@pytest.mark.asyncio
async def test_create_microenvironments_file_writes_expected_format(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * (adata.n_obs // 2) + ["B"] * (adata.n_obs - adata.n_obs // 2))
    # Force all cells into one connected neighborhood signature.
    adata.obsm["spatial"] = np.column_stack([
        np.linspace(0.0, 1.0, adata.n_obs),
        np.linspace(0.0, 1.0, adata.n_obs),
    ])

    path = await ccc._create_microenvironments_file(
        adata,
        CellCommunicationParameters(
            method="cellphonedb",
            species="human",
            cell_type_key="cell_type",
            cellphonedb_spatial_radius=10.0,
        ),
        DummyCtx(),
    )

    assert path is not None
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    assert lines[0] == "cell_type\tmicroenvironment"
    data_lines = set(lines[1:])
    # Both cell types should map into at least one shared microenvironment.
    assert any(ln.startswith("T\tmicroenv_") for ln in data_lines)
    assert any(ln.startswith("B\tmicroenv_") for ln in data_lines)

    import os

    os.remove(path)


def test_integrate_autocrine_detection_cellchat_prob_matrix_branch():
    results = pd.DataFrame({"interaction_name": ["L1^R1", "L2-R2", "L3_R3"]})
    prob = np.zeros((2, 2, 3), dtype=float)
    prob[0, 0, :] = [1.0, 0.0, 2.0]  # diagonal contributions for pairs 1 and 3
    prob[1, 1, :] = [0.0, 0.0, 0.0]

    storage = CCCStorage(
        method="cellchat_r",
        analysis_type="cluster",
        species="human",
        database="cellchatdb",
        results=results,
        method_data={"prob_matrix": prob},
    )

    ccc._integrate_autocrine_detection(storage, n_top=2)

    assert storage.autocrine.n_loops == 2
    assert storage.autocrine.top_pairs == ["L1_R1", "L3_R3"]


def _install_fake_cellphonedb_modules(monkeypatch: pytest.MonkeyPatch, download_impl):
    import sys
    from types import ModuleType

    cellphonedb_pkg = ModuleType("cellphonedb")
    utils_pkg = ModuleType("cellphonedb.utils")
    db_utils_mod = ModuleType("cellphonedb.utils.db_utils")
    db_utils_mod.download_database = download_impl

    monkeypatch.setitem(sys.modules, "cellphonedb", cellphonedb_pkg)
    monkeypatch.setitem(sys.modules, "cellphonedb.utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "cellphonedb.utils.db_utils", db_utils_mod)


def test_ensure_cellphonedb_database_returns_existing_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    import os

    calls: dict[str, int] = {"download": 0}

    def _download(*_args, **_kwargs):
        calls["download"] += 1

    _install_fake_cellphonedb_modules(monkeypatch, _download)
    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("certifi.where", lambda: str(tmp_path / "ca.pem"))

    db_file = tmp_path / "cellphonedb.zip"
    db_file.write_text("ok", encoding="utf-8")

    out = ccc._ensure_cellphonedb_database(str(tmp_path), DummyCtx())

    assert out == os.path.join(str(tmp_path), "cellphonedb.zip")
    assert calls["download"] == 0


def test_ensure_cellphonedb_database_downloads_and_restores_ssl(tmp_path, monkeypatch: pytest.MonkeyPatch):
    import os
    import ssl

    calls: dict[str, object] = {"download_args": None}

    def _download(output_dir, version):
        calls["download_args"] = (output_dir, version)
        (tmp_path / "cellphonedb.zip").write_text("ok", encoding="utf-8")

    _install_fake_cellphonedb_modules(monkeypatch, _download)
    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("certifi.where", lambda: str(tmp_path / "ca.pem"))

    original_ctx = ssl._create_default_https_context

    out = ccc._ensure_cellphonedb_database(str(tmp_path), DummyCtx())

    assert out == os.path.join(str(tmp_path), "cellphonedb.zip")
    assert calls["download_args"] == (str(tmp_path), "v5.0.0")
    assert ssl._create_default_https_context is original_ctx


@pytest.mark.asyncio
async def test_analyze_communication_fastccc_success_single_method(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * (adata.n_obs // 2) + ["B"] * (adata.n_obs - adata.n_obs // 2))

    interactions_strength = pd.DataFrame(
        {
            "T|T": [0.8, 0.1],
            "T|B": [0.2, 0.9],
        },
        index=["L1^R1", "L2-R2"],
    )
    pvalues = pd.DataFrame(
        {
            "T|T": [0.01, 0.2],
            "T|B": [0.03, 0.8],
        },
        index=["L1^R1", "L2-R2"],
    )
    percentages = pd.DataFrame({"pct": [0.5, 0.6]}, index=["L1^R1", "L2-R2"])

    def _fake_statistical_analysis_method(**_kwargs):
        return interactions_strength, pvalues, percentages

    import sys
    from types import ModuleType

    fastccc_mod = ModuleType("fastccc")
    fastccc_mod.statistical_analysis_method = _fake_statistical_analysis_method
    monkeypatch.setitem(sys.modules, "fastccc", fastccc_mod)

    monkeypatch.setattr(
        ccc,
        "get_raw_data_source",
        lambda _adata, prefer_complete_genes=True: type(
            "Raw",
            (),
            {
                "X": np.asarray(_adata.X),
                "var_names": _adata.var_names,
            },
        )(),
    )

    import os.path as osp

    orig_exists = osp.exists
    monkeypatch.setattr(
        "os.path.exists",
        lambda p: True if str(p).endswith("interaction_table.csv") else orig_exists(p),
    )

    params = CellCommunicationParameters(
        method="fastccc",
        species="human",
        cell_type_key="cell_type",
        fastccc_use_cauchy=False,
        fastccc_pvalue_threshold=0.05,
        plot_top_pairs=1,
    )

    out = await ccc._analyze_communication_fastccc(adata, params, DummyCtx())

    assert out.method == "fastccc"
    assert out.n_pairs == 2
    assert out.n_significant == 1
    assert out.lr_pairs == ["L1_R1", "L2_R2"]
    assert out.top_lr_pairs == ["L2_R2"]
    assert out.method_data["percentages"] is percentages


def test_ensure_cellphonedb_database_raises_dependency_error_and_restores_ssl_on_download_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    import ssl

    def _download_fail(*_args, **_kwargs):
        raise RuntimeError("network down")

    _install_fake_cellphonedb_modules(monkeypatch, _download_fail)
    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("certifi.where", lambda: str(tmp_path / "ca.pem"))

    original_ctx = ssl._create_default_https_context

    with pytest.raises(DependencyError, match="Failed to download CellPhoneDB database"):
        ccc._ensure_cellphonedb_database(str(tmp_path), DummyCtx())

    assert ssl._create_default_https_context is original_ctx


@pytest.mark.asyncio
async def test_create_microenvironments_file_warns_and_returns_none_on_exception(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)

    class _BrokenNN:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *_args, **_kwargs):
            raise RuntimeError("knn fail")

    monkeypatch.setattr("sklearn.neighbors.NearestNeighbors", _BrokenNN)

    ctx = DummyCtx()
    out = await ccc._create_microenvironments_file(
        adata,
        CellCommunicationParameters(
            method="cellphonedb",
            species="human",
            cell_type_key="cell_type",
        ),
        ctx,
    )

    assert out is None
    assert any("Failed to create microenvironments file" in w for w in ctx.warnings)


@pytest.mark.asyncio
async def test_analyze_communication_liana_wraps_internal_errors(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)

    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(__import__("sys").modules, "liana", type("L", (), {})())

    def _boom(*_args, **_kwargs):
        raise RuntimeError("cluster fail")

    monkeypatch.setattr(ccc, "_run_liana_cluster_analysis", _boom)

    with pytest.raises(ProcessingError, match=r"LIANA\+ analysis failed: cluster fail"):
        await ccc._analyze_communication_liana(
            adata,
            CellCommunicationParameters(
                method="liana",
                species="human",
                cell_type_key="cell_type",
                perform_spatial_analysis=False,
            ),
            DummyCtx(),
        )


def test_run_liana_spatial_analysis_without_pvals_layer_sets_none(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()

    var = pd.DataFrame(
        {
            "morans": [0.9],
            "morans_pvals": [0.001],
        },
        index=["L1^R1"],
    )
    lrdata = type(
        "LR",
        (),
        {
            "n_vars": 1,
            "var": var,
            "X": np.array([[0.5]] * adata.n_obs),
            "layers": {},
        },
    )()

    fake_liana = type(
        "L",
        (),
        {"mt": type("MT", (), {"bivariate": staticmethod(lambda *_args, **_kwargs: lrdata)})},
    )()
    monkeypatch.setitem(__import__("sys").modules, "liana", fake_liana)

    out = ccc._run_liana_spatial_analysis(
        adata,
        CellCommunicationParameters(
            method="liana",
            species="human",
            cell_type_key="cell_type",
            liana_global_metric="morans",
            plot_top_pairs=1,
        ),
        DummyCtx(),
    )

    assert out.method_data["spatial_pvals"] is None
    assert out.lr_pairs == ["L1_R1"]


@pytest.mark.asyncio
async def test_analyze_communication_cellphonedb_success_with_correction_stats(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import sys
    import types

    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * (adata.n_obs // 2) + ["B"] * (adata.n_obs - adata.n_obs // 2))

    means = pd.DataFrame(
        {
            "interacting_pair": ["L1^R1", "L2-R2"],
            "T|B": [0.8, 0.2],
            "B|T": [0.6, 0.1],
        },
        index=["pair1", "pair2"],
    )
    pvals = pd.DataFrame(
        {
            "T|B": [0.001, 0.8],
            "B|T": [0.01, 0.7],
        },
        index=["pair1", "pair2"],
    )

    fake_cpdb_method = types.SimpleNamespace(
        call=lambda **_kwargs: {
            "deconvoluted": pd.DataFrame(),
            "means": means,
            "pvalues": pvals,
            "significant_means": means.copy(),
        }
    )
    cellphonedb_pkg = types.ModuleType("cellphonedb")
    src_pkg = types.ModuleType("cellphonedb.src")
    core_pkg = types.ModuleType("cellphonedb.src.core")
    methods_pkg = types.ModuleType("cellphonedb.src.core.methods")
    methods_pkg.cpdb_statistical_analysis_method = fake_cpdb_method
    sys.modules["cellphonedb"] = cellphonedb_pkg
    sys.modules["cellphonedb.src"] = src_pkg
    sys.modules["cellphonedb.src.core"] = core_pkg
    sys.modules["cellphonedb.src.core.methods"] = methods_pkg

    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ccc, "_ensure_cellphonedb_database", lambda *_args, **_kwargs: "/tmp/fake_cpdb.zip")

    params = CellCommunicationParameters(
        method="cellphonedb",
        species="human",
        cell_type_key="cell_type",
        cellphonedb_use_microenvironments=False,
        cellphonedb_iterations=5,
        cellphonedb_pvalue=0.05,
        cellphonedb_correction_method="fdr_bh",
        plot_top_pairs=2,
    )

    out = await ccc._analyze_communication_cellphonedb(adata, params, DummyCtx())

    assert out.method == "cellphonedb"
    assert out.n_pairs == 2
    assert out.n_significant == 1
    assert out.lr_pairs == ["L1_R1", "L2_R2"]
    assert out.top_lr_pairs == ["L1_R1"]
    assert "correction_statistics" in out.statistics


@pytest.mark.asyncio
async def test_analyze_communication_fastccc_success_standard_path(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import sys
    import types

    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * (adata.n_obs // 2) + ["B"] * (adata.n_obs - adata.n_obs // 2))

    interactions_strength = pd.DataFrame(
        {"T|B": [0.7, 0.1], "B|T": [0.5, 0.2]}, index=["L1^R1", "L2-R2"]
    )
    pvalues = pd.DataFrame(
        {"T|B": [0.01, 0.8], "B|T": [0.03, 0.6]}, index=["L1^R1", "L2-R2"]
    )
    percentages = pd.DataFrame(
        {"T": [0.8, 0.2], "B": [0.6, 0.4]}, index=["L1^R1", "L2-R2"]
    )

    fake_fastccc = types.ModuleType("fastccc")
    fake_fastccc.statistical_analysis_method = lambda **_kwargs: (
        interactions_strength,
        pvalues,
        percentages,
    )
    sys.modules["fastccc"] = fake_fastccc

    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ccc, "get_raw_data_source", lambda *_a, **_k: type("Raw", (), {"X": adata.X, "var_names": adata.var_names})())

    real_exists = __import__("os").path.exists

    def _fake_exists(path):
        if path.endswith("interaction_table.csv"):
            return True
        return real_exists(path)

    monkeypatch.setattr("os.path.exists", _fake_exists)

    params = CellCommunicationParameters(
        method="fastccc",
        species="human",
        cell_type_key="cell_type",
        fastccc_use_cauchy=False,
        fastccc_pvalue_threshold=0.05,
        plot_top_pairs=1,
    )

    out = await ccc._analyze_communication_fastccc(adata, params, DummyCtx())

    assert out.method == "fastccc"
    assert out.n_pairs == 2
    assert out.n_significant == 1
    assert out.lr_pairs == ["L1_R1", "L2_R2"]
    assert out.top_lr_pairs == ["L1_R1"]
    assert out.method_data["percentages"].shape == (2, 2)


@pytest.mark.asyncio
async def test_analyze_communication_fastccc_cauchy_without_outputs_raises(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import sys
    import types

    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)

    fake_fastccc = types.ModuleType("fastccc")
    fake_fastccc.Cauchy_combination_of_statistical_analysis_methods = lambda **_kwargs: None
    sys.modules["fastccc"] = fake_fastccc

    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ccc, "get_raw_data_source", lambda *_a, **_k: type("Raw", (), {"X": adata.X, "var_names": adata.var_names})())

    real_exists = __import__("os").path.exists

    def _fake_exists(path):
        if path.endswith("interaction_table.csv"):
            return True
        return real_exists(path)

    monkeypatch.setattr("os.path.exists", _fake_exists)
    monkeypatch.setattr("glob.glob", lambda _pattern: [])

    params = CellCommunicationParameters(
        method="fastccc",
        species="human",
        cell_type_key="cell_type",
        fastccc_use_cauchy=True,
    )

    with pytest.raises(ProcessingError, match="Cauchy combination did not produce output files"):
        await ccc._analyze_communication_fastccc(adata, params, DummyCtx())


@pytest.mark.asyncio
async def test_analyze_communication_cellphonedb_rejects_unexpected_return_format(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import sys
    import types

    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)

    fake_cpdb_method = types.SimpleNamespace(call=lambda **_kwargs: "bad-result")
    cellphonedb_pkg = types.ModuleType("cellphonedb")
    src_pkg = types.ModuleType("cellphonedb.src")
    core_pkg = types.ModuleType("cellphonedb.src.core")
    methods_pkg = types.ModuleType("cellphonedb.src.core.methods")
    methods_pkg.cpdb_statistical_analysis_method = fake_cpdb_method
    sys.modules["cellphonedb"] = cellphonedb_pkg
    sys.modules["cellphonedb.src"] = src_pkg
    sys.modules["cellphonedb.src.core"] = core_pkg
    sys.modules["cellphonedb.src.core.methods"] = methods_pkg

    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ccc, "_ensure_cellphonedb_database", lambda *_args, **_kwargs: "/tmp/fake_cpdb.zip")

    params = CellCommunicationParameters(
        method="cellphonedb",
        species="human",
        cell_type_key="cell_type",
        cellphonedb_use_microenvironments=False,
    )

    with pytest.raises(ProcessingError, match="returned unexpected format"):
        await ccc._analyze_communication_cellphonedb(adata, params, DummyCtx())


@pytest.mark.asyncio
async def test_analyze_communication_cellphonedb_rejects_missing_significant_means(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import sys
    import types

    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)

    means = pd.DataFrame({"interacting_pair": ["L1^R1"]})
    pvals = pd.DataFrame({"T|T": [0.01]})

    fake_cpdb_method = types.SimpleNamespace(
        call=lambda **_kwargs: {
            "means": means,
            "pvalues": pvals,
            # intentionally missing 'significant_means'
        }
    )
    cellphonedb_pkg = types.ModuleType("cellphonedb")
    src_pkg = types.ModuleType("cellphonedb.src")
    core_pkg = types.ModuleType("cellphonedb.src.core")
    methods_pkg = types.ModuleType("cellphonedb.src.core.methods")
    methods_pkg.cpdb_statistical_analysis_method = fake_cpdb_method
    sys.modules["cellphonedb"] = cellphonedb_pkg
    sys.modules["cellphonedb.src"] = src_pkg
    sys.modules["cellphonedb.src.core"] = core_pkg
    sys.modules["cellphonedb.src.core.methods"] = methods_pkg

    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ccc, "_ensure_cellphonedb_database", lambda *_args, **_kwargs: "/tmp/fake_cpdb.zip")

    params = CellCommunicationParameters(
        method="cellphonedb",
        species="human",
        cell_type_key="cell_type",
        cellphonedb_use_microenvironments=False,
    )

    with pytest.raises(ProcessingError, match="found no L-R interactions"):
        await ccc._analyze_communication_cellphonedb(adata, params, DummyCtx())


@pytest.mark.asyncio
async def test_analyze_communication_cellphonedb_rejects_non_numeric_pvalues(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import sys
    import types

    adata = minimal_spatial_adata.copy()
    adata.obs["cell_type"] = pd.Categorical(["T"] * adata.n_obs)

    means = pd.DataFrame({"interacting_pair": ["L1^R1"]})
    pvals = pd.DataFrame({"T|T": ["not_numeric"]})

    fake_cpdb_method = types.SimpleNamespace(
        call=lambda **_kwargs: {
            "deconvoluted": pd.DataFrame(),
            "means": means,
            "pvalues": pvals,
            "significant_means": means.copy(),
        }
    )
    cellphonedb_pkg = types.ModuleType("cellphonedb")
    src_pkg = types.ModuleType("cellphonedb.src")
    core_pkg = types.ModuleType("cellphonedb.src.core")
    methods_pkg = types.ModuleType("cellphonedb.src.core.methods")
    methods_pkg.cpdb_statistical_analysis_method = fake_cpdb_method
    sys.modules["cellphonedb"] = cellphonedb_pkg
    sys.modules["cellphonedb.src"] = src_pkg
    sys.modules["cellphonedb.src.core"] = core_pkg
    sys.modules["cellphonedb.src.core.methods"] = methods_pkg

    monkeypatch.setattr(ccc, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ccc, "_ensure_cellphonedb_database", lambda *_args, **_kwargs: "/tmp/fake_cpdb.zip")

    params = CellCommunicationParameters(
        method="cellphonedb",
        species="human",
        cell_type_key="cell_type",
        cellphonedb_use_microenvironments=False,
    )

    with pytest.raises(ProcessingError, match="p-values are not numeric"):
        await ccc._analyze_communication_cellphonedb(adata, params, DummyCtx())
