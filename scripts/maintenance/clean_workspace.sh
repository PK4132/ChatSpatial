#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "Cleaning common local artifacts (safe, non-destructive)"
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name ".DS_Store" -delete
rm -rf .pytest_cache .ruff_cache .mypy_cache build dist

echo "Done."
