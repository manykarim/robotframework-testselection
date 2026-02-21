# Packaging Proposal: Reusable PyPI Distribution

## 1. Name Proposals

### PyPI Package Name (must start with `robotframework-`)

| Option | PyPI install command | Availability |
|--------|---------------------|--------------|
| **`robotframework-diversetest`** | `pip install robotframework-diversetest` | Available |
| **`robotframework-diverse`** | `pip install robotframework-diverse` | Available |
| **`robotframework-testselection`** | `pip install robotframework-testselection` | Available |
| **`robotframework-selector`** | `pip install robotframework-selector` | Available |
| **`robotframework-diversityselector`** | `pip install robotframework-diversityselector` | Available |

### Python Package Name (import name)

The Robot Framework ecosystem has two conventions:

- **CamelCase** for packages that provide RF keywords (e.g. `SeleniumLibrary`, `DataDriver`)
- **lowercase** for CLI tools (e.g. `robocop`, `pabot`)

Since this project is **both** a CLI tool and provides RF integration components (PreRunModifier, Listener), the recommended approach is a hybrid:

| PyPI Name | Python Import | CLI Command | Style |
|-----------|---------------|-------------|-------|
| `robotframework-diversetest` | `DiverseTest` | `rfdiverse` | CamelCase (RF convention) |
| `robotframework-diversetest` | `diversetest` | `rfdiverse` | lowercase (Python convention) |
| `robotframework-diverse` | `RFDiverse` | `rfdiverse` | CamelCase (RF convention) |
| `robotframework-diverse` | `rfdiverse` | `rfdiverse` | lowercase (Python convention) |
| `robotframework-testselection` | `TestSelection` | `testcase-select` | CamelCase (RF convention) |
| `robotframework-testselection` | `testselection` | `testcase-select` | lowercase (Python convention) |

### Recommendation

**PyPI name:** `robotframework-diversetest`
**Python import:** `diversetest` (lowercase, PEP 8 compliant, short)
**CLI entry point:** `rfdiverse` (short, memorable, follows `rf` prefix pattern from robocop/robotidy)
**Alternative CLI alias:** `testcase-select` (backward compatible)

Rationale:
- `diversetest` is concise, descriptive, and unique
- lowercase follows PEP 8 and matches the tool-oriented nature (like `robocop`, `pabot`)
- `rfdiverse` CLI prefix makes it instantly recognizable as an RF tool
- The name communicates the core value proposition: diverse test selection

---

## 2. Required Changes

### 2.1 Rename Python Package Directory

```
Current:  src/testcase_selection/
Target:   src/diversetest/
```

All internal imports change from `testcase_selection.*` to `diversetest.*`:

| Current | New |
|---------|-----|
| `from testcase_selection.parsing.suite_collector import ...` | `from diversetest.parsing.suite_collector import ...` |
| `from testcase_selection.selection.fps import ...` | `from diversetest.selection.fps import ...` |
| `from testcase_selection.embedding.embedder import ...` | `from diversetest.embedding.embedder import ...` |
| `from testcase_selection.pipeline.select import ...` | `from diversetest.pipeline.select import ...` |
| `from testcase_selection.shared.types import ...` | `from diversetest.shared.types import ...` |

Files requiring import updates:
- All 33 source files under `src/`
- All 35 test files under `tests/`
- `pyproject.toml` (package references, mypy config)
- CI configs referencing CLI entry point

### 2.2 Update `pyproject.toml`

```toml
[project]
name = "robotframework-diversetest"
version = "0.1.0"
description = "Vector-based diverse test case selection for Robot Framework"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "Your Name", email = "you@example.com" },
]
keywords = [
    "robotframework",
    "testing",
    "test-selection",
    "diversity",
    "machine-learning",
    "nlp",
    "embeddings",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Framework :: Robot Framework",
    "Framework :: Robot Framework :: Library",
    "Framework :: Robot Framework :: Tool",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Testing",
    "Topic :: Software Development :: Quality Assurance",
]
dependencies = [
    "robotframework>=7.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
]

[project.urls]
Homepage = "https://github.com/<owner>/robotframework-diversetest"
Documentation = "https://github.com/<owner>/robotframework-diversetest#readme"
Repository = "https://github.com/<owner>/robotframework-diversetest"
Issues = "https://github.com/<owner>/robotframework-diversetest/issues"
Changelog = "https://github.com/<owner>/robotframework-diversetest/blob/main/CHANGELOG.md"

[project.scripts]
rfdiverse = "diversetest.cli:main"
testcase-select = "diversetest.cli:main"  # backward-compat alias

[project.optional-dependencies]
vectorize = ["sentence-transformers>=2.2"]
selection-extras = [
    "scikit-learn-extra>=0.3",
    "dppy>=0.3",
    "apricot-select>=0.6",
]
chromadb = ["chromadb>=0.4"]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "pytest-benchmark",
    "ruff",
    "mypy",
]
all = ["robotframework-diversetest[vectorize,selection-extras,chromadb,dev]"]

[tool.hatch.build.targets.wheel]
packages = ["src/diversetest"]

[tool.mypy]
python_version = "3.11"
strict = false
packages = ["diversetest"]
mypy_path = "src"
```

### 2.3 Add Missing Package Metadata Files

| File | Purpose | Currently Exists |
|------|---------|-----------------|
| `README.md` | PyPI long description | Yes |
| `LICENSE` | MIT license text | **No — must create** |
| `CHANGELOG.md` | Version history | **No — must create** |
| `py.typed` | PEP 561 marker for type checkers | **No — should create** |
| `.gitignore` | Standard Python gitignore | **No — should create** |
| `MANIFEST.in` | Source distribution includes | Not needed (hatchling handles) |

### 2.4 Add `py.typed` Marker

Create `src/diversetest/py.typed` (empty file) so type checkers can find inline types.

### 2.5 Add `long_description` for PyPI

```toml
[project]
readme = "README.md"
```

### 2.6 Update Robot Framework Integration Paths

The PreRunModifier and Listener are referenced by fully-qualified module path in `robot` commands. These change:

```bash
# Current
robot --prerunmodifier testcase_selection.execution.prerun_modifier.DiversePreRunModifier:file tests/

# New
robot --prerunmodifier diversetest.execution.prerun_modifier.DiversePreRunModifier:file tests/
```

The README, CI configs, and any documentation must be updated to reflect new module paths.

### 2.7 Update CI Configs

Files in `config/`:
- `github-actions.yml` — update `uv run testcase-select` to `uv run rfdiverse`
- `gitlab-ci.yml` — same
- `Jenkinsfile` — same

---

## 3. Build and Publish Workflow

### 3.1 Build

```bash
# Using uv (recommended)
uv build

# Produces:
#   dist/robotframework_diversetest-0.1.0-py3-none-any.whl
#   dist/robotframework_diversetest-0.1.0.tar.gz
```

### 3.2 Test the Built Package

```bash
# Install from wheel in a fresh venv
uv venv /tmp/test-install
uv pip install --python /tmp/test-install dist/robotframework_diversetest-0.1.0-py3-none-any.whl

# Verify imports work
/tmp/test-install/bin/python -c "import diversetest; print(diversetest.__version__)"

# Verify CLI works
/tmp/test-install/bin/rfdiverse --help

# Verify RF integration works
/tmp/test-install/bin/python -c "from diversetest.execution.prerun_modifier import DiversePreRunModifier; print('OK')"
```

### 3.3 Publish to TestPyPI (first)

```bash
# Upload to TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/

# Test install from TestPyPI
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ robotframework-diversetest
```

### 3.4 Publish to PyPI

```bash
# Requires PyPI API token
uv publish
```

### 3.5 GitHub Actions Publish Workflow

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI
on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # trusted publisher
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

---

## 4. Version Strategy

### Semantic Versioning (SemVer)

```
0.1.0  — Initial release (current)
0.2.0  — Add new selection strategies
0.x.y  — Pre-1.0 development (breaking changes allowed in minor)
1.0.0  — Stable public API
```

### Single Source of Truth for Version

Currently the version is in two places:
- `pyproject.toml` → `version = "0.1.0"`
- `src/testcase_selection/__init__.py` → `__version__ = "0.1.0"`

Options:
1. **Use `hatch-vcs`** — derive version from git tags (recommended for automated releases)
2. **Use `hatchling` dynamic versioning** — read from `__init__.py`
3. **Keep manual** — update both files (current approach, simple but error-prone)

Recommended: Option 2 (dynamic from `__init__.py`):

```toml
[project]
dynamic = ["version"]

[tool.hatch.version]
path = "src/diversetest/__init__.py"
```

---

## 5. Dependency Considerations for Distribution

### 5.1 Core Dependencies (always installed)

| Package | Min Version | Size | Notes |
|---------|-------------|------|-------|
| `robotframework` | >=7.0 | ~3 MB | Required. Consider >=6.0 for wider compat |
| `numpy` | >=1.24 | ~30 MB | Required for vector operations |
| `scikit-learn` | >=1.3 | ~30 MB | Required for cosine_distances in FPS |

**Total base install: ~63 MB** (plus RF's own deps)

### 5.2 Concern: `scikit-learn` in Base

`scikit-learn` is heavy. Consider whether it should be in base or optional:

- **Current:** `scikit-learn` is in base deps because FPS uses `cosine_distances`
- **Alternative:** Implement cosine distance with pure numpy to remove scikit-learn from base deps:

```python
# numpy-only cosine distance (no scikit-learn needed)
def cosine_distances(a, b):
    similarity = (a @ b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True).T)
    return 1 - similarity
```

This would reduce base install from ~63 MB to ~33 MB and make the package much lighter for users who only need FPS.

### 5.3 Optional Dependency Groups

| Group | Use Case | Added Size |
|-------|----------|-----------|
| `vectorize` | Real ML embeddings | ~2 GB (torch + transformers) |
| `selection-extras` | k-Medoids, DPP, Facility | ~20 MB |
| `chromadb` | Alternative vector storage | ~50 MB |

Users must be clearly informed that `vectorize` pulls in PyTorch (~2 GB). The README should prominently document this.

### 5.4 Python Version Support

Current: `>=3.11`. Consider:
- `>=3.10` — wider compatibility (sentence-transformers supports 3.10+)
- `>=3.11` — current, allows `frozenset` type hints, `tomllib` stdlib
- `>=3.12` — too restrictive for a public package

**Recommendation:** `>=3.10` for maximum adoption. The `from __future__ import annotations` already used everywhere handles type hint syntax.

---

## 6. Project Structure After Rename

```
robotframework-diversetest/          # repository name
    pyproject.toml
    README.md
    LICENSE
    CHANGELOG.md
    .gitignore
    src/
        diversetest/
            __init__.py              # __version__, top-level imports
            py.typed                 # PEP 561 marker
            cli.py                   # CLI entry point
            shared/
                __init__.py
                types.py
                config.py
            parsing/
                __init__.py
                suite_collector.py
                keyword_resolver.py
                text_builder.py
                datadriver_reader.py
            embedding/
                __init__.py
                ports.py
                embedder.py
                models.py
            selection/
                __init__.py
                strategy.py
                fps.py
                kmedoids.py
                dpp.py
                facility.py
                registry.py
                filtering.py
            execution/
                __init__.py
                prerun_modifier.py
                listener.py
                runner.py
            pipeline/
                __init__.py
                vectorize.py
                select.py
                execute.py
                cache.py
                artifacts.py
                errors.py
    tests/
        ...
    docs/
        adr/
        architecture/
    config/
        github-actions.yml
        gitlab-ci.yml
        Jenkinsfile
```

---

## 7. Checklist for Package Release

### Pre-release

- [ ] Choose final names (PyPI, Python package, CLI command)
- [ ] Rename `src/testcase_selection/` to `src/<chosen_name>/`
- [ ] Update all internal imports (33 source + 35 test files)
- [ ] Update `pyproject.toml` with full metadata (authors, classifiers, urls, readme)
- [ ] Add `LICENSE` file
- [ ] Add `CHANGELOG.md`
- [ ] Add `py.typed` marker
- [ ] Add `.gitignore`
- [ ] Consider removing `scikit-learn` from base deps (implement numpy cosine distance)
- [ ] Consider lowering `requires-python` to `>=3.10`
- [ ] Update README with new import paths and CLI commands
- [ ] Update CI configs with new CLI command name
- [ ] Update Robot Framework integration paths in docs
- [ ] Run full test suite
- [ ] Run `uv build` and verify wheel contents
- [ ] Test install in clean venv
- [ ] Verify CLI works after install
- [ ] Verify RF PreRunModifier and Listener work after install

### Publish

- [ ] Create GitHub repository with chosen name
- [ ] Push code
- [ ] Register PyPI account and create API token
- [ ] Publish to TestPyPI first
- [ ] Test install from TestPyPI
- [ ] Publish to PyPI
- [ ] Create GitHub release with tag
- [ ] Set up trusted publisher (GitHub Actions OIDC) for future releases

---

## 8. Alternative Name Matrix (Full Comparison)

| # | PyPI Name | Python Import | CLI Command | Pros | Cons |
|---|-----------|---------------|-------------|------|------|
| 1 | `robotframework-diversetest` | `diversetest` | `rfdiverse` | Descriptive, unique, short import | Slightly long PyPI name |
| 2 | `robotframework-diverse` | `rfdiverse` | `rfdiverse` | Short, clean | "diverse" alone is vague |
| 3 | `robotframework-testselection` | `testselection` | `testcase-select` | Very descriptive | Generic, long |
| 4 | `robotframework-selector` | `rfselector` | `rfselector` | Short | Could be confused with CSS selectors |
| 5 | `robotframework-diversityselector` | `diversityselector` | `rfdiverse` | Most descriptive | Too long |
| 6 | `robotframework-testdiversity` | `testdiversity` | `rfdiverse` | Clear purpose | Slightly generic |
