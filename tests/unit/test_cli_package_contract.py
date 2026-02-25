"""Unit contract for chatspatial.cli package metadata."""

from __future__ import annotations

import importlib


def test_cli_package_import_contract():
    mod = importlib.import_module("chatspatial.cli")
    assert mod.__all__ == []
