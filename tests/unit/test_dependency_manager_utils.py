"""Unit contracts for dependency manager behavior and error messaging."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from chatspatial.utils import dependency_manager as dm


def _install_fake_rpy2(
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_packages: set[str] | None = None,
) -> None:
    """Install lightweight fake rpy2/anndata2ri modules into sys.modules."""

    missing_packages = missing_packages or set()

    class _Lock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _LocalConverter:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_anndata2ri = ModuleType("anndata2ri")
    fake_ro = ModuleType("rpy2.robjects")
    fake_conversion = ModuleType("rpy2.robjects.conversion")
    fake_packages = ModuleType("rpy2.robjects.packages")
    fake_rinterface_lib = ModuleType("rpy2.rinterface_lib")
    fake_scvi = ModuleType("scvi")
    fake_scvi_model = ModuleType("scvi.model")
    fake_scvi_external = ModuleType("scvi.external")

    def _fake_importr(name: str):
        if name in missing_packages:
            raise RuntimeError(f"missing {name}")
        return object()

    fake_conversion.localconverter = _LocalConverter
    fake_ro.conversion = fake_conversion
    fake_ro.default_converter = object()
    fake_ro.numpy2ri = object()
    fake_ro.pandas2ri = object()
    fake_ro.r = lambda _expr: "ok"
    fake_packages.importr = _fake_importr
    fake_rinterface_lib.openrlib = SimpleNamespace(rlock=_Lock())
    fake_scvi.model = fake_scvi_model
    fake_scvi.external = fake_scvi_external

    monkeypatch.setitem(sys.modules, "anndata2ri", fake_anndata2ri)
    monkeypatch.setitem(sys.modules, "rpy2", ModuleType("rpy2"))
    monkeypatch.setitem(sys.modules, "rpy2.robjects", fake_ro)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.conversion", fake_conversion)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.packages", fake_packages)
    monkeypatch.setitem(sys.modules, "rpy2.rinterface_lib", fake_rinterface_lib)
    monkeypatch.setitem(sys.modules, "scvi", fake_scvi)
    monkeypatch.setitem(sys.modules, "scvi.model", fake_scvi_model)
    monkeypatch.setitem(sys.modules, "scvi.external", fake_scvi_external)


def test_get_info_supports_registered_alias_and_unknown_defaults():
    spatialde_by_name = dm._get_info("spatialde")
    spatialde_by_module = dm._get_info("NaiveDE")
    unknown = dm._get_info("some-new-package")

    assert spatialde_by_name.module_name == "NaiveDE"
    assert spatialde_by_module.install_cmd == "pip install SpatialDE"
    assert unknown.install_cmd == "pip install some-new-package"
    assert unknown.description == "Optional: some-new-package"


def test_is_available_uses_registered_module_name(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(dm, "_check_spec", lambda module_name: module_name == "NaiveDE")
    assert dm.is_available("spatialde")
    assert not dm.is_available("flashs")


def test_get_warn_if_missing_emits_install_hint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dm, "_try_import", lambda _module_name: None)
    with pytest.warns(UserWarning, match="Install:"):
        module = dm.get("flashs", warn_if_missing=True)
    assert module is None


def test_require_includes_feature_install_and_description(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(dm, "_try_import", lambda _module_name: None)

    with pytest.raises(ImportError) as exc:
        dm.require("flashs", feature="SVG analysis")
    msg = str(exc.value)

    assert "for SVG analysis" in msg
    assert "pip install flashs" in msg
    assert "FlashS" in msg


def test_validate_r_environment_reports_missing_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(dm, "is_available", lambda name: name != "rpy2")
    with pytest.raises(ImportError, match="rpy2 is required"):
        dm.validate_r_environment()

    monkeypatch.setattr(dm, "is_available", lambda name: name != "anndata2ri")
    with pytest.raises(ImportError, match="anndata2ri is required"):
        dm.validate_r_environment()


def test_validate_r_environment_reports_missing_r_packages(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(dm, "is_available", lambda _name: True)
    _install_fake_rpy2(monkeypatch, missing_packages={"SPARK"})

    with pytest.raises(ImportError, match="Missing R packages: 'SPARK'"):
        dm.validate_r_environment(required_packages=["base", "SPARK"])


def test_validate_r_package_success_and_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dm, "is_available", lambda _name: True)
    _install_fake_rpy2(monkeypatch, missing_packages={"MissingPkg"})

    assert dm.validate_r_package("stats")
    with pytest.raises(ImportError, match="MissingPkg"):
        dm.validate_r_package("MissingPkg")


def test_check_r_packages_returns_all_when_rpy2_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dm, "is_available", lambda _name: False)
    assert dm.check_r_packages(["a", "b"]) == ["a", "b"]


def test_check_r_packages_returns_only_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dm, "is_available", lambda _name: True)
    monkeypatch.setattr(
        dm,
        "validate_r_package",
        lambda pkg: (_ for _ in ()).throw(ImportError("x")) if pkg == "bad" else True,
    )
    assert dm.check_r_packages(["ok", "bad"]) == ["bad"]


def test_validate_scvi_tools_component_guard(monkeypatch: pytest.MonkeyPatch):
    _install_fake_rpy2(monkeypatch)
    fake_scvi = sys.modules["scvi"]
    monkeypatch.setattr(dm, "require", lambda *_args, **_kwargs: fake_scvi)

    with pytest.raises(ImportError, match="SCANVI"):
        dm.validate_scvi_tools(components=["SCANVI"])
