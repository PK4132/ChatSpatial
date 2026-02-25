"""Branch-focused contracts for compatibility fallback paths."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from chatspatial.utils import compat


def test_numpy2_compat_assert_array_equal_requires_two_arrays() -> None:
    with pytest.raises(ValueError, match="requires two arrays"):
        compat._numpy2_compat_assert_array_equal()


def test_patch_numpy_testing_is_idempotent() -> None:
    unpatch = compat._patch_numpy_testing()
    try:
        noop_unpatch = compat._patch_numpy_testing()  # already patched path
        # no-op unpatch should not restore original function
        noop_unpatch()
        np.testing.assert_array_equal(x=np.array([1]), y=np.array([1]))
    finally:
        unpatch()


def test_derivative_compat_uses_richardson_for_higher_order() -> None:
    def f(x: float) -> float:
        return x**4

    # n=3 forces Richardson branch.
    out = compat._derivative_compat(f, x0=2.0, dx=1e-3, n=3, order=5)
    assert np.isfinite(out)


def test_patch_scipy_misc_derivative_adds_missing_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(compat.scipy_misc, "derivative", raising=False)
    compat.patch_scipy_misc_derivative()
    assert hasattr(compat.scipy_misc, "derivative")
    assert compat.scipy_misc.derivative is compat._derivative_compat


def test_patch_scipy_numpy_aliases_sets_missing_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    import scipy

    # Ensure alias is missing before patching, regardless of scipy version behavior.
    monkeypatch.delattr(scipy, "arange", raising=False)
    monkeypatch.delattr(scipy, "linspace", raising=False)

    compat.patch_scipy_numpy_aliases()

    assert hasattr(scipy, "arange")
    assert hasattr(scipy, "linspace")
    np.testing.assert_array_equal(scipy.arange(3), np.arange(3))


def test_patch_scipy_sparse_matrix_A_is_idempotent() -> None:
    compat.patch_scipy_sparse_matrix_A()
    # already patched fast-return branch
    compat.patch_scipy_sparse_matrix_A()


def test_ensure_cellrank_compat_cleans_up_numpy_patch_when_numpy2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"cleanup_called": False}

    monkeypatch.setattr(compat, "_is_numpy2", lambda: True)
    monkeypatch.setattr(
        compat,
        "_patch_numpy_testing",
        lambda: (lambda: state.__setitem__("cleanup_called", True)),
    )

    cleanup = compat.ensure_cellrank_compat()
    cleanup()
    assert state["cleanup_called"]


def test_ensure_cellrank_compat_noop_when_not_numpy2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compat, "_is_numpy2", lambda: False)
    cleanup = compat.ensure_cellrank_compat()
    cleanup()  # should not raise


def test_ensure_spagcn_compat_dispatches_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"patched": False}
    monkeypatch.setattr(
        compat,
        "patch_scipy_sparse_matrix_A",
        lambda: called.__setitem__("patched", True),
    )
    compat.ensure_spagcn_compat()
    assert called["patched"]
