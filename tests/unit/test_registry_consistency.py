"""Structural consistency tests: verify registries match parameter model Literals.

These tests ensure that runtime dispatch registries stay in sync with the
Pydantic Literal types that define valid values in the MCP JSON schema.
If a method is added to a registry but not to the Literal (or vice versa),
these tests fail immediately.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_deconvolution_registry_matches_literal():
    """METHOD_REGISTRY keys must equal DeconvolutionParameters.method Literal."""
    from chatspatial.models.data import DeconvolutionParameters
    from chatspatial.tools.deconvolution import METHOD_REGISTRY

    literal_methods = set(
        DeconvolutionParameters.model_fields["method"].annotation.__args__
    )
    assert set(METHOD_REGISTRY) == literal_methods


@pytest.mark.unit
def test_spatial_statistics_registry_matches_literal():
    """_ANALYSIS_REGISTRY keys must equal SpatialStatisticsParameters.analysis_type Literal."""
    from chatspatial.models.data import SpatialStatisticsParameters
    from chatspatial.tools.spatial_statistics import _ANALYSIS_REGISTRY

    literal_types = set(
        SpatialStatisticsParameters.model_fields["analysis_type"].annotation.__args__
    )
    assert set(_ANALYSIS_REGISTRY) == literal_types


@pytest.mark.unit
def test_every_tool_has_annotations():
    """Every registered MCP tool must have ToolAnnotations (inline, not a dict)."""
    from chatspatial.server import mcp

    tools = mcp._tool_manager.list_tools()
    missing = [t.name for t in tools if t.annotations is None]
    assert not missing, f"Tools without annotations: {missing}"
