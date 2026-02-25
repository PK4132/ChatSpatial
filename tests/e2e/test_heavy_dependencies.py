"""Optional heavy dependency checks (excluded from default fast suite)."""

import pytest


@pytest.mark.slow
@pytest.mark.e2e
def test_optional_r_stack_available():
    pytest.importorskip("rpy2")


@pytest.mark.slow
@pytest.mark.e2e
def test_optional_scvi_available():
    pytest.importorskip("scvi")


@pytest.mark.slow
@pytest.mark.e2e
def test_optional_tangram_available():
    pytest.importorskip("tangram")


@pytest.mark.slow
@pytest.mark.e2e
def test_optional_liana_available():
    pytest.importorskip("liana")
