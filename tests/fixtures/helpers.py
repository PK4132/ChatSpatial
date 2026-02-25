"""Shared test helpers to keep behavior checks consistent and DRY."""

from __future__ import annotations

from pathlib import Path

from chatspatial.server import load_data


def extract_saved_path(viz_message: str) -> Path:
    """Parse saved file path from visualize_data return text."""
    prefix = "Visualization saved: "
    for line in viz_message.splitlines():
        if line.startswith(prefix):
            return Path(line[len(prefix) :].strip())
    raise AssertionError(f"Cannot parse visualization output path from: {viz_message}")


async def load_generic_dataset(spatial_dataset_path, *, name: str):
    """Load the shared temporary h5ad fixture via public load_data API."""
    return await load_data(str(spatial_dataset_path), "generic", name=name)
