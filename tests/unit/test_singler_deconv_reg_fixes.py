"""Regression tests for deconvolution count validation and registration fixes."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# _prepare_counts with require_counts=True must raise DataError when only
# normalized (non-integer) data is available.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_counts_require_counts_raises_for_normalized_data(
    minimal_spatial_adata,
):
    """_prepare_counts(require_counts=True) should raise DataError when the
    dataset contains only normalized (float) expression data."""
    from chatspatial.tools.deconvolution.base import _prepare_counts
    from chatspatial.utils.exceptions import DataError

    adata = minimal_spatial_adata.copy()
    # Replace X with clearly normalized (non-integer) data
    adata.X = np.random.default_rng(42).uniform(0, 5, size=adata.shape).astype(
        np.float32
    )
    # Remove any counts layer that might exist
    if "counts" in adata.layers:
        del adata.layers["counts"]

    class FakeCtx:
        async def warning(self, msg: str) -> None:
            pass

    ctx = FakeCtx()

    with pytest.raises(DataError, match="No raw integer counts found"):
        await _prepare_counts(
            adata, "Spatial", ctx,
            require_int_dtype=False,
            require_counts=True,
        )


@pytest.mark.asyncio
async def test_prepare_counts_require_counts_succeeds_for_integer_data(
    minimal_spatial_adata,
):
    """_prepare_counts(require_counts=True) should succeed when integer
    counts are available."""
    from chatspatial.tools.deconvolution.base import _prepare_counts

    adata = minimal_spatial_adata.copy()
    # Fixture X is already Poisson-distributed integers
    adata.X = np.round(adata.X).astype(np.float32)

    class FakeCtx:
        async def warning(self, msg: str) -> None:
            pass

    ctx = FakeCtx()

    result = await _prepare_counts(
        adata, "Spatial", ctx,
        require_int_dtype=False,
        require_counts=True,
    )
    assert result.shape == adata.shape


# ---------------------------------------------------------------------------
# Bug 3: _transform_coordinates must preserve original query coordinates
# for zero-transport-signal rows instead of mapping them to (0, 0).
# ---------------------------------------------------------------------------


def test_transform_coordinates_preserves_query_coords_for_zero_rows():
    """When a row in the transport matrix is all zeros and query_coords
    is provided, the original query coordinates should be preserved
    instead of returning (0, 0)."""
    from chatspatial.tools import spatial_registration as reg

    transport = np.array([
        [0.0, 0.0],  # zero row - no correspondence
        [0.2, 0.8],  # normal row
        [0.0, 0.0],  # another zero row
    ], dtype=float)
    ref = np.array([[1.0, 1.0], [3.0, 5.0]], dtype=float)
    query = np.array([[10.0, 20.0], [0.0, 0.0], [30.0, 40.0]], dtype=float)

    result = reg._transform_coordinates(transport, ref, query_coords=query)

    assert result.shape == (3, 2)
    assert np.isfinite(result).all()

    # Zero rows should retain their original query coordinates
    np.testing.assert_allclose(result[0], [10.0, 20.0])
    np.testing.assert_allclose(result[2], [30.0, 40.0])

    # Normal row should be transformed: 0.2*[1,1] + 0.8*[3,5] = [2.6, 4.2]
    np.testing.assert_allclose(result[1], [2.6, 4.2])


def test_transform_coordinates_without_query_coords_backward_compat():
    """Without query_coords, zero rows map to (0,0) as before (no regression
    for callers that don't pass query_coords)."""
    from chatspatial.tools import spatial_registration as reg

    transport = np.array([[0.0, 0.0], [0.2, 0.8]], dtype=float)
    ref = np.array([[1.0, 1.0], [3.0, 5.0]], dtype=float)

    result = reg._transform_coordinates(transport, ref)

    assert result.shape == (2, 2)
    assert np.isfinite(result).all()
    # Zero row maps to (0, 0) when no query_coords
    np.testing.assert_allclose(result[0], [0.0, 0.0])
    np.testing.assert_allclose(result[1], [2.6, 4.2])
