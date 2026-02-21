# ADR-006: Text Representation and Keyword Tree Resolution Strategy

## Status

Proposed

## Date

2026-02-21

## Context

The Vector-Based Diverse Test Selection system for Robot Framework relies on transformer-based embeddings to place test cases in a vector space, then selects maximally diverse subsets for execution. The quality of this vector representation is determined almost entirely by the quality of the text string fed into the embedding model. Robot Framework test cases are not plain text -- they are structured artifacts composed of keyword calls, arguments, tags, metadata, and nested keyword definitions. Converting these structured artifacts into text suitable for embedding is a non-trivial design decision with measurable impact on selection quality.

Research demonstrates a **19-27% improvement** in retrieval metrics when structured data is converted to natural language before embedding, compared to embedding raw structured formats directly (documented in project research: `docs/research/compass_artifact_wf-fd8b16a0-0008-4bc3-9105-e4553444d251_text_markdown.md`). This makes the text representation layer a critical component of the pipeline.

Three specific challenges must be addressed:

1. **Signal-to-noise ratio**: Robot Framework test cases contain high-semantic-value elements (test names, keyword names, tags) mixed with low-semantic-value noise (DOM locators, variable placeholders, XPaths). Including noise dilutes embedding quality, causing semantically different tests to appear similar in vector space.

2. **Keyword tree depth**: A test case's top-level keyword calls may be abstract (e.g., `Login As Admin`) or concrete (e.g., `Input Text    id:username    admin`). Resolving user keywords to their sub-keywords provides richer text at the cost of increased processing time and potential noise introduction. The optimal resolution depth depends on the test suite's abstraction patterns.

3. **DataDriver-generated tests**: The DataDriver library generates test cases dynamically at runtime from CSV/Excel data sources. These tests do not exist at parse time, so their text representations must be constructed by reading the data sources directly rather than from the Robot Framework model.

## Decision

### 1. Natural Language Text Representation Format

Test cases will be converted to a single natural language string before embedding. The format concatenates semantic elements in descending order of value:

```
Test: {test_name}. Tags: {tag1}, {tag2}. {keyword1_text} {keyword2_text} ...
```

Example output:
```
Test: Login With Valid Credentials. Tags: smoke, regression. Open Browser with https://app.example.com Log In with admin Check Dashboard Title
```

### 2. Semantic Element Inclusion (Ranked by Value)

Elements are included in the text representation in this priority order:

| Priority | Element | Rationale | Example |
|----------|---------|-----------|---------|
| 1 (highest) | Test name | Most distinctive identifier; often describes intent | `Login With Valid Credentials` |
| 2 | Tags | Categorical context that groups related tests | `smoke`, `regression`, `api` |
| 3 | Suite path / name | Grouping context when suite structure reflects domains | `tests/authentication/login.robot` |
| 4 | Keyword names | Action semantics; underscores converted to spaces | `Open Browser`, `Log In` |
| 5 | Semantic arguments | Meaningful values that distinguish test scenarios | Page titles, expected text, URLs, usernames |

### 3. Noise Exclusion Rules

The following patterns are filtered from the text representation because they add implementation detail without semantic value, causing unrelated tests to cluster together in vector space:

| Noise Category | Detection Pattern | Example |
|----------------|-------------------|---------|
| DOM locators (id) | Starts with `id:` | `id:username`, `id:submit-btn` |
| DOM locators (CSS) | Starts with `css:` | `css:.btn-primary`, `css:#login-form` |
| DOM locators (XPath prefix) | Starts with `xpath:` | `xpath://div[@class='main']` |
| Raw XPaths | Starts with `//` | `//div[@id='content']/span` |
| Scalar variable placeholders | Starts with `${` | `${USERNAME}`, `${EXPECTED_TITLE}` |
| List variable placeholders | Starts with `@{` | `@{ITEMS}`, `@{TEST_DATA}` |
| Environment variables | Starts with `%{` | `%{HOME}`, `%{CI_ENV}` |
| Dictionary variables | Starts with `&{` | `&{CREDENTIALS}`, `&{CONFIG}` |

The filtering is implemented as a prefix check on each argument string:

```python
NOISE_PREFIXES = ('id:', 'css:', 'xpath:', '//', '${', '@{', '%{', '&{')

def is_semantic_arg(arg):
    return not any(str(arg).startswith(p) for p in NOISE_PREFIXES)
```

### 4. Keyword Tree Resolution

Keyword tree resolution converts abstract keyword calls into their constituent sub-keyword calls, producing richer text representations. This is implemented through three functions:

- **`build_keyword_map(suite)`**: Recursively collects all user keywords from `suite.resource.keywords` across the suite hierarchy into a name-to-definition lookup dictionary. Keyword names are normalized (lowercased, spaces replaced with underscores) for matching.

- **`resolve_keyword_tree(kw_name, kw_args, kw_map, depth, max_depth)`**: Starting from a keyword call, recursively matches the keyword name against the registry and descends into its body to build a tree of sub-keyword calls, up to `max_depth` levels.

- **`flatten_tree(node)`**: Converts the keyword tree into a natural language string. Underscores in keyword names are replaced with spaces. Semantic arguments (those passing the noise filter) are appended with "with" joining syntax. Children are concatenated with spaces.

Resolution depth is configurable:

| Depth | Behavior | Performance | Best For |
|-------|----------|-------------|----------|
| 0 (default) | Top-level keyword names only; no resolution | Fastest | Most suites; sufficient when keyword names are descriptive |
| 1 | Resolves one level of user keywords | Moderate | Suites with generic top-level keywords (e.g., `Setup`, `Teardown`) |
| 2+ | Deep resolution into sub-keywords | Slower | Suites where multiple tests share the same high-level keywords but differ in implementation details |

Depth 0 is the default because it provides the best speed-to-quality tradeoff for most suites. When tests share identical top-level keywords (e.g., many tests calling `Perform Login` with different sub-steps), increasing depth differentiates them in embedding space.

### 5. DataDriver Test Handling

DataDriver generates tests at runtime via a Listener v3 `start_suite` event, which fires after PreRunModifiers have already executed. The text representation stage therefore cannot rely on the Robot Framework model for DataDriver tests. Instead, it reads the data source files directly.

For CSV sources, the implementation uses `csv.DictReader`:

```python
def read_datadriver_csv(csv_path, template_name, delimiter=','):
    tests = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            test_name = row.get('*** Test Cases ***', '').strip()
            if not test_name or test_name.startswith('#'):
                continue
            args = {k: v for k, v in row.items()
                    if k.startswith('${') and v}
            description = f"Template: {template_name}. Test: {test_name}. "
            description += " ".join(f"{k}={v}" for k, v in args.items())
            tests.append({"name": test_name, "description": description})
    return tests
```

Key design points:

- The `*** Test Cases ***` column is DataDriver's convention for the generated test name.
- Variable columns (starting with `${`) contain the parameterized values that differentiate each generated test.
- The CSV delimiter is configurable (DataDriver defaults to `;` in some configurations).
- Rows starting with `#` are treated as comments and skipped.
- The template name is included in the text to provide context about what the data-driven test does.

### 6. TextRepresentationBuilder Service

The text representation logic is encapsulated in a `TextRepresentationBuilder` service with configurable options:

```python
@dataclass
class TextRepresentationConfig:
    resolve_depth: int = 0           # Keyword tree resolution depth
    include_suite_name: bool = False  # Include suite path in text
    include_tags: bool = True         # Include tags in text
    noise_prefixes: tuple = NOISE_PREFIXES  # Customizable noise filters
    underscore_to_space: bool = True  # Convert keyword underscores to spaces
    datadriver_csvs: list = None      # DataDriver CSV configurations
    csv_delimiter: str = ','          # Default CSV delimiter
```

This design supports:

- **Configurable noise filters**: Teams can add or remove noise prefix patterns to match their locator conventions.
- **Optional suite name inclusion**: Useful when suite directory structure carries semantic meaning (e.g., `tests/api/auth/` vs `tests/ui/auth/`).
- **Extensibility for new text formats**: The builder pattern allows adding structured or hybrid output formats without modifying the core pipeline.

## Consequences

### Positive

- **Improved embedding quality**: Filtering noise and converting to natural language yields 19-27% better retrieval metrics compared to raw structured embedding, directly improving diversity selection accuracy.
- **Configurable depth**: Teams can tune keyword resolution depth to match their suite's abstraction level without code changes. Depth 0 covers most cases with minimal overhead.
- **DataDriver support**: Pre-reading CSV sources enables the pipeline to handle data-driven test suites that would otherwise be invisible at parse time.
- **Deterministic output**: For a given configuration and test suite, the text representation is deterministic, enabling reliable caching of embeddings in Stage 1 of the pipeline.
- **Low overhead**: Text generation for 2,000 tests at depth 0 completes in under 1 second. Even at depth 2, the overhead remains under 5 seconds.

### Negative

- **Noise filter maintenance**: The prefix-based noise filter is a heuristic. Custom locator strategies (e.g., `data-testid:`) or non-standard variable syntax require adding new prefixes to the filter configuration.
- **Keyword resolution limitations**: `build_keyword_map` only resolves user keywords defined in the suite's resource files. Library keywords (e.g., SeleniumLibrary's `Click Element`) are not resolved -- they appear as-is in the text. This is acceptable because library keyword names are already descriptive natural language.
- **DataDriver coupling**: The CSV reading logic assumes DataDriver's column naming conventions. Changes to DataDriver's CSV format would require updating the reader. Excel data sources require additional library dependencies (e.g., `openpyxl`).
- **Argument filtering is lossy**: Some filtered arguments may carry semantic value (e.g., a `${PAGE_TITLE}` variable whose resolved value would be informative). The current approach accepts this loss because variable values are not available at parse time, and the noise reduction benefit outweighs the information loss in aggregate.

### Risks

- **Domain-specific vocabulary**: The general-purpose embedding model (`all-MiniLM-L6-v2`) may not capture domain-specific test terminology as well as a fine-tuned model. The text representation layer cannot compensate for embedding model limitations, but producing clean natural language text gives the model the best possible input.
- **Very similar tests**: When many tests share the same keywords and differ only in filtered arguments (e.g., data-driven tests differing only in `${VAR}` values), they will produce nearly identical text representations and cluster together in embedding space. DataDriver CSV pre-reading mitigates this for data-driven suites, but purely variable-parameterized standard tests remain a known limitation.

## References

- Project research: `docs/research/compass_artifact_wf-fd8b16a0-0008-4bc3-9105-e4553444d251_text_markdown.md`
- Project research: `docs/research/multistage_pipeline_report.md`
- Gonzalez, T.F. (1985). Clustering to Minimize the Maximum Intercluster Distance. Theoretical Computer Science, 38, 293-306.
- Robot Framework User Guide: Creating Test Libraries, Listener Interface
- DataDriver Library Documentation: CSV/Excel data source conventions
