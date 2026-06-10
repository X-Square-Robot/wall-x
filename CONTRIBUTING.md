# Contributing to Wall-X

Thanks for your interest in contributing to Wall-X.

## Development Setup

Install the project dependencies and the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

Run checks before opening a pull request:

```bash
pre-commit run --all-files
python -m pytest tests -q
```

## Pull Requests

- Keep changes focused and include reproduction steps for bug fixes.
- Update documentation when changing user-facing behavior or commands.
- Do not include private paths, credentials, internal service URLs, or large
  checkpoint artifacts.
- Use placeholder paths such as `<path/to/checkpoint>` in examples.
