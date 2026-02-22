# robotframework-testselection v0.2.0

**Vector-based diverse test case selection for Robot Framework**

## What's New

### Robot CLI Passthrough (`--` separator)

You can now pass **any** Robot Framework CLI option through `testcase-select` by placing `--` after your selection arguments. Everything after `--` is forwarded directly to `robot`.

```bash
# Pass variables and set log level
testcase-select run --suite tests/ --k 20 \
  -- --variable ENV:staging --variable USER:admin --loglevel DEBUG

# Combine diversity selection with Robot's own tag filtering
testcase-select execute --suite tests/ --selection sel.json \
  -- --include smoke --exclude manual

# Add metadata and custom output name
testcase-select run --suite tests/ --k 30 \
  -- --metadata Version:2.1 --name "Smoke Regression"

# Set variables file and debug log
testcase-select run --suite tests/ --k 50 \
  -- --variablefile config/env_staging.py --debugfile debug.log
```

This works with both `run` and `execute` subcommands, including during graceful fallback (when selection fails and all tests are run).

### CI/CD Pipeline

Added a comprehensive GitHub Actions CI workflow:

- **Lint**: ruff check + mypy type checking
- **Test**: pytest matrix across Python 3.10, 3.11, 3.12, 3.13 with coverage reporting
- **Build**: wheel/sdist verification with install smoke test
- 60% coverage threshold gate on Python 3.11

### License Change

Switched from MIT to **Apache 2.0** license.

## Changes Since v0.1.0

### Added
- `--` passthrough for robot CLI options (variables, tags, log levels, listeners, metadata, etc.)
- GitHub Actions CI workflow (`.github/workflows/ci.yml`)
- 10 new CLI unit tests for passthrough argument handling

### Changed
- License: MIT → Apache 2.0
- Improved type annotations for mypy compatibility

### Fixed
- Ruff lint violations (E501, F841, F401, I001) across test files
- Mypy type errors for optional dependency stubs and Robot Framework API

## Installation

```bash
pip install robotframework-testselection

# With sentence-transformers for vectorization
pip install robotframework-testselection[vectorize]

# Everything
pip install robotframework-testselection[all]
```

Requires Python 3.10+.

## Links

- **Repository**: https://github.com/manykarim/robotframework-testselection
- **Full Changelog**: https://github.com/manykarim/robotframework-testselection/compare/v0.1.0...v0.2.0
- **Issues**: https://github.com/manykarim/robotframework-testselection/issues
