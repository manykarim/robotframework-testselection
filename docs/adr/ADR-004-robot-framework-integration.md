# ADR-004: Robot Framework Integration Approach

## Status

Proposed

## Date

2026-02-21

## Context

The Vector-Based Diverse Test Selection system must integrate with Robot Framework's execution pipeline to filter tests before they run. The system needs to parse test suites, extract semantic content for embedding, and then restrict execution to only the selected diverse subset.

Two integration mechanisms exist within `robot.api`:

1. **PreRunModifier** (extending `SuiteVisitor`) -- runs after parsing but before execution begins.
2. **Listener v3** -- fires events during execution, including `start_suite` which provides access to suite data after all libraries have initialized.

A critical complication arises from the **DataDriver library**. DataDriver generates test cases dynamically from CSV/Excel data sources via its own Listener v3 `start_suite` handler. This generation happens **after** PreRunModifiers have already run, meaning a PreRunModifier only sees the single template test case, not the hundreds of data-driven tests that DataDriver will create. This timing gap requires a second integration mechanism specifically for DataDriver suites.

### Robot Framework Execution Order

The full execution sequence relevant to test filtering is:

1. Robot Framework parses `.robot` files and builds the `TestSuite` model.
2. **PreRunModifiers run** -- at this point, DataDriver suites contain only the template test.
3. Tag filtering (`--include`/`--exclude`) is processed.
4. **Execution begins** -- Listener v3 `start_suite` fires for each suite.
5. **DataDriver's `start_suite` fires** (at default listener priority) -- generates test cases from the data source.
6. **Our listener's `start_suite` fires** (at lower priority) -- filters the now-populated test list.
7. Selected tests execute.

### Programmatic vs CLI Execution

There is a subtle but important behavioral difference:

- **CLI mode** (`robot` command): Both `--prerunmodifier` and `--listener` flags work as expected.
- **Programmatic mode** (`TestSuite.run()`): The `--prerunmodifier` argument is **silently ignored**. Instead, the visitor must be applied directly via `suite.visit()`.
- **Programmatic mode** (`robot.run_cli()`): Behaves like CLI mode, supporting both `--prerunmodifier` and `--listener` flags.

## Decision

We adopt a **dual-mechanism integration** using exclusively public `robot.api` features. No monkey-patching, no access to framework internals, no private APIs.

### 1. PreRunModifier for Standard Test Suites

A `PreRunModifier` extending `robot.api.SuiteVisitor` handles all non-DataDriver test suites. It accepts a selection file path as a constructor argument and filters `suite.tests` in-place.

```python
from robot.api import SuiteVisitor

class DiversePreRunModifier(SuiteVisitor):

    def __init__(self, selection_file):
        # Load selected test names from JSON produced by Stage 2
        with open(selection_file) as f:
            data = json.load(f)
        self.selected_names = set(
            t["name"] for t in data["selected"]
            if not t.get("is_datadriver", False)
        )

    def start_suite(self, suite):
        suite.tests = [
            t for t in suite.tests
            if t.name in self.selected_names
        ]

    def end_suite(self, suite):
        suite.suites = [s for s in suite.suites if s.test_count > 0]

    def visit_test(self, test):
        pass  # skip test internals for performance
```

Design rationale:

- **`start_suite`**: Filters `suite.tests` to retain only selected test names. This is the primary filtering point.
- **`end_suite`**: Prunes child suites that became empty after filtering, keeping output clean.
- **`visit_test`**: Overridden as a no-op to skip walking test body internals, since we only need name-based matching, not introspection.
- **Constructor argument**: The selection file path is passed as the sole argument, matching Robot Framework's convention for PreRunModifier parameterization (`--prerunmodifier module.Class:arg`).

### 2. Listener v3 for DataDriver-Generated Tests

A Listener v3 class handles suites that use DataDriver. It fires after DataDriver has populated the test list, filtering the generated tests down to the selected subset.

```python
class DiverseDataDriverListener:
    ROBOT_LISTENER_API_VERSION = 3
    ROBOT_LISTENER_PRIORITY = 50

    def __init__(self, selection_file):
        with open(selection_file) as f:
            data = json.load(f)
        self.selected_dd_names = set(
            t["name"] for t in data["selected"]
            if t.get("is_datadriver", False)
        )

    def start_suite(self, data, result):
        if not self.selected_dd_names or len(data.tests) <= 1:
            return
        data.tests = [
            t for t in data.tests
            if t.name in self.selected_dd_names
        ]
```

Design rationale:

- **`ROBOT_LISTENER_API_VERSION = 3`**: Uses the modern Listener v3 API where `start_suite` receives both `data` (mutable test model) and `result` (result model).
- **`ROBOT_LISTENER_PRIORITY = 50`**: Lower than DataDriver's default priority, ensuring our listener fires **after** DataDriver has generated its tests. In Robot Framework's listener priority scheme, lower values execute later.
- **`start_suite(data, result)`**: The `data` parameter provides the mutable `TestSuite` model. After DataDriver has populated `data.tests`, we filter it in-place.
- **DataDriver test name patterns**: DataDriver generates test names from data row values (e.g., "Login with user admin and password secret"). The embedding pipeline must account for this by pre-reading the CSV/Excel data source during the vectorization stage.

### 3. Parsing Approach for Vectorization (Stage 1)

The vectorization stage uses these `robot.api` features to extract test content:

- **`TestSuite.from_file_system(path)`**: Primary entry point for parsing `.robot` files into a traversable suite model.
- **`suite.tests`**: Access to test case objects, each with `.name`, `.tags`, and `.body` (keyword call sequence).
- **`suite.resource.keywords`**: Access to user keyword definitions for building a resolution map.
- **`build_keyword_map(suite)`**: Recursive function that collects all user keywords across the suite hierarchy, enabling keyword tree resolution. Keywords are indexed by normalized name (lowercased, spaces replaced with underscores).
- **`SuiteVisitor` pattern**: Used for traversal when collecting tests across nested suite hierarchies via `start_suite`, `start_keyword`, and related methods.
- **DataDriver CSV pre-reading**: For DataDriver suites, the data source CSV/Excel file is read directly during vectorization to generate test names and descriptions, since those tests do not exist in the parsed model until runtime.

### 4. CLI and Programmatic Invocation

**CLI invocation** (standard suites only):

```bash
robot --prerunmodifier DiversePreRunModifier:selected_tests.json tests/
```

**CLI invocation** (with DataDriver suites):

```bash
robot --prerunmodifier DiversePreRunModifier:selected_tests.json \
      --listener DiverseDataDriverListener:selected_tests.json \
      tests/
```

**Programmatic invocation** (standard suites via `TestSuite.run()`):

```python
suite = TestSuite.from_file_system('tests/')
suite.visit(DiversePreRunModifier('selected_tests.json'))
suite.run(output='output.xml')
```

Note: `--prerunmodifier` passed to `TestSuite.run()` is silently ignored, so `suite.visit()` must be called explicitly.

**Programmatic invocation** (combined PreRunModifier + Listener via `robot.run_cli()`):

```python
import robot
robot.run_cli([
    '--prerunmodifier', 'DiversePreRunModifier:selected_tests.json',
    '--listener', 'DiverseDataDriverListener:selected_tests.json',
    'tests/'
], exit=False)
```

### 5. robot.api Features Used

The integration relies exclusively on public `robot.api` features:

| Feature | Purpose |
|---------|---------|
| `TestSuite.from_file_system()` | Parse `.robot` files into traversable model |
| `SuiteVisitor` | Base class for PreRunModifier; provides `start_suite`, `end_suite`, `visit_test`, `start_keyword` |
| Listener v3 API | `start_suite(data, result)` for post-DataDriver filtering |
| `ROBOT_LISTENER_API_VERSION` | Declares Listener v3 protocol |
| `ROBOT_LISTENER_PRIORITY` | Controls listener execution order relative to DataDriver |
| `suite.tests` | Mutable list of test cases within a suite |
| `test.name`, `test.tags`, `test.body` | Test case properties for name matching, tag filtering, and keyword extraction |
| `suite.resource.keywords` | User keyword definitions for keyword tree resolution |
| `suite.suites` | Child suite access for hierarchical traversal and pruning |

No private APIs, internal modules, or monkey-patching is used.

## Consequences

### Positive

- **Uses only public, stable APIs**: All integration points are documented `robot.api` features, reducing the risk of breakage on Robot Framework upgrades.
- **Handles the DataDriver timing gap**: The dual-mechanism approach (PreRunModifier + Listener v3 with priority ordering) correctly filters both standard and DataDriver-generated tests.
- **Minimal runtime overhead**: Name-based set lookup in `start_suite` is O(1) per test. The `visit_test` no-op avoids unnecessary traversal of test body internals.
- **Clean separation of concerns**: The PreRunModifier and Listener are independent classes with no shared state. They can be used individually or together depending on whether DataDriver is present.
- **Compatible with both CLI and programmatic usage**: The explicit `suite.visit()` pattern for programmatic execution avoids the silent-ignore pitfall of `TestSuite.run()`.
- **Preserves downstream Robot Framework features**: Tag filtering (`--include`/`--exclude`) still functions correctly because PreRunModifiers run before tag processing, and our Listener runs at a separate lifecycle point.

### Negative

- **DataDriver test names must be known ahead of time**: The vectorization stage must pre-read DataDriver CSV/Excel sources to predict generated test names. If the data source format changes or test name generation logic is customized, the pre-reading must be updated to match.
- **Two integration points to maintain**: Having both a PreRunModifier and a Listener adds complexity compared to a single mechanism. Teams not using DataDriver can omit the Listener entirely.
- **Listener priority coupling**: The `ROBOT_LISTENER_PRIORITY = 50` value depends on DataDriver using a higher (default) priority. If DataDriver changes its priority scheme, the value may need adjustment. This is a fragile ordering dependency.
- **Programmatic execution requires awareness of the `suite.visit()` pattern**: Developers using `TestSuite.run()` must know to call `suite.visit()` explicitly rather than passing `--prerunmodifier` as an argument, since the latter is silently ignored. This is a documented Robot Framework behavior but is easy to miss.
- **No filtering of tests generated by other runtime-generating libraries**: The Listener v3 approach is designed for DataDriver specifically. Other libraries that generate tests at runtime may require priority adjustments or additional listeners.
