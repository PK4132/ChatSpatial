# Contributing

This page is the docs-site entry point for contributors.

If you want the full contributor guide used on GitHub, see the repository file as well. Within the docs site, this page keeps the contribution path discoverable without breaking site navigation.

---

## Getting Started

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/ChatSpatial.git
cd ChatSpatial

# Create environment and install
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Verify
pytest tests/unit/ -x
```

**Prerequisites:** Python 3.11-3.13, Git. For R-based methods (RCTD, CellChat, SPARK-X, etc.): R 4.4+ and rpy2.

---

## Contribution Paths

### Documentation

- clarify terminology
- reduce duplicated guidance
- improve examples and workflows
- keep method/reference docs aligned with schemas

### Code

Common areas:
- `chatspatial/` — package source
- `tests/` — automated verification
- `docs/` — user-facing documentation

### New analysis methods

Follow the existing pattern:
- add a parameter model in `models/data.py`
- add a result model in `models/analysis.py`
- implement the tool in `tools/`
- register it in `server.py`
- add unit/integration tests

---

## Quality Gate

```bash
black chatspatial/
isort chatspatial/
ruff check chatspatial/ --fix
mypy chatspatial/
pytest tests/unit/
```

---

## Submitting Changes

1. Create a branch
2. Make focused changes
3. Run tests and linting
4. Open a PR against `main`

Suggested commit styles:

```text
feat: add new spatial analysis method
fix: handle edge case in deconvolution
docs: update methods reference
test: add integration test for trajectory
```

---

## More Detail

For the full contributor guide, open [CONTRIBUTING.md](https://github.com/cafferychen777/ChatSpatial/blob/main/CONTRIBUTING.md).

For user-facing docs, return to:
- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Methods Reference](advanced/methods-reference.md)
