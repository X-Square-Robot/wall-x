# Contributing to Wall-X

Thank you for your interest in contributing to Wall-X! This document provides guidelines to help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
  - [Reporting Issues](#reporting-issues)
  - [Suggesting Features](#suggesting-features)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Coding Guidelines](#coding-guidelines)
- [License](#license)

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to foster an inclusive and welcoming community.

## How to Contribute

### Reporting Issues

If you find a bug or have a problem using Wall-X, please open an issue on GitHub:

1. Search [existing issues](https://github.com/X-Square-Robot/wall-x/issues) to avoid duplicates
2. Use a clear, descriptive title
3. Include steps to reproduce the issue
4. Provide your environment details (OS, Python version, CUDA version, PyTorch version)
5. Attach relevant logs or error messages

### Suggesting Features

Feature suggestions are welcome! Open an issue with:

- A clear description of the feature
- Motivation (what problem does it solve?)
- Any alternative solutions you've considered

### Pull Requests

Before opening a pull request, please:

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** following the [coding guidelines](#coding-guidelines)
3. **Test your changes** thoroughly
4. **Ensure lint checks pass:**

```bash
# Install pre-commit hooks (run once)
pre-commit install
```

Or run manually:

```bash
# Manually run all checks
pre-commit run --all-files
```

5. **Submit a pull request** with a clear description of your changes

## Development Setup

See the [README](README.md#environment-setup) for full environment setup instructions.

In short:

```bash
conda create --name wallx python=3.10
conda activate wallx
pip install -r requirements.txt
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation
git submodule update --init --recursive
MAX_JOBS=4 pip install --no-build-isolation --verbose -e .
```

## Coding Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code style
- Use type hints where practical
- Write docstrings for public functions and classes
- Keep changes focused — one feature or fix per PR
- Add or update tests when applicable

## License

By contributing to Wall-X, you agree that your contributions will be licensed under the [BSD 3-Clause License](LICENSE).
