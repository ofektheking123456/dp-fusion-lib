# Contributing to DP-Fusion-Lib

Thank you for your interest in contributing to DP-Fusion-Lib! This document provides guidelines for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions.

## Getting Started

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rushil-thareja/dp-fusion-lib.git
   cd dp-fusion-lib
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks (optional):
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ -v --cov=dp_fusion_lib --cov-report=term-missing
```

### Code Style

We use:
- **black** for code formatting
- **ruff** for linting

Format code:
```bash
black src/ tests/ examples/
```

Check linting:
```bash
ruff check src/ tests/ examples/
```

## Making Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring

### Commit Messages

Use clear, descriptive commit messages:
- `feat: Add support for batch generation`
- `fix: Handle edge case in lambda search`
- `docs: Update installation instructions`
- `test: Add tests for epsilon computation`

### Pull Requests

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Reporting Issues

### Bug Reports

Please include:
- Python version
- Package versions (torch, transformers, dp-fusion-lib)
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Please include:
- Clear description of the feature
- Use case / motivation
- Possible implementation approach (optional)

## Questions

For questions about using the library, please open a GitHub issue with the "question" label.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).
