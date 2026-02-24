"""Experiment: Vectorize pytest test functions and compute pairwise distances.

Extracts test functions from a Python test file, builds multiple text
representations (name-only, docstring, full-source, AST-based, combined),
embeds them with all-MiniLM-L6-v2, and prints cosine distance matrices.

Usage:
    uv run python scripts/experiments/vectorize_pytest_experiment.py [test_file]

Default test file: tests/unit/test_cli.py
"""
from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PytestTestInfo:
    """Extracted information about a single pytest test function."""

    name: str
    qualname: str  # e.g. "TestClass.test_method"
    docstring: str
    source: str
    markers: list[str]
    fixtures: list[str]
    assertions: list[str]
    called_functions: list[str]
    imported_names: list[str]  # from the module level
    class_name: str | None
    lineno: int
    complexity: int  # number of control-flow branches


@dataclass
class TextRepresentations:
    """Multiple text representations of a test for comparison."""

    name_only: str
    name_plus_docstring: str
    full_source: str
    ast_based: str
    combined: str


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------

class TestExtractor(ast.NodeVisitor):
    """Walk a Python AST and extract information about test functions."""

    def __init__(self, source_text: str, module_imports: list[str]) -> None:
        self.source_lines = source_text.splitlines()
        self.module_imports = module_imports
        self.tests: list[PytestTestInfo] = []
        self._current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        old_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not node.name.startswith("test_"):
            return

        qualname = (
            f"{self._current_class}.{node.name}"
            if self._current_class
            else node.name
        )

        # Docstring
        docstring = ast.get_docstring(node) or ""

        # Source code (from line numbers)
        start = node.lineno - 1
        end = node.end_lineno or start + 1
        source = "\n".join(self.source_lines[start:end])

        # Markers (decorators like @pytest.mark.xxx)
        markers = []
        for dec in node.decorator_list:
            marker_name = _get_marker_name(dec)
            if marker_name:
                markers.append(marker_name)

        # Fixtures (arguments other than 'self')
        fixtures = [
            arg.arg
            for arg in node.args.args
            if arg.arg not in ("self", "cls")
        ]

        # Assertions
        assertions = _extract_assertions(node)

        # Called functions
        called = _extract_function_calls(node)

        # Complexity (if/for/while/try/with branches)
        complexity = _compute_complexity(node)

        self.tests.append(PytestTestInfo(
            name=node.name,
            qualname=qualname,
            docstring=docstring,
            source=source,
            markers=markers,
            fixtures=fixtures,
            assertions=assertions,
            called_functions=called,
            imported_names=self.module_imports,
            class_name=self._current_class,
            lineno=node.lineno,
            complexity=complexity,
        ))

    visit_AsyncFunctionDef = visit_FunctionDef


def _get_marker_name(node: ast.expr) -> str | None:
    """Extract pytest marker name from a decorator node."""
    if isinstance(node, ast.Attribute):
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        full = ".".join(reversed(parts))
        if "pytest.mark" in full or "mark." in full:
            return full
    elif isinstance(node, ast.Call):
        return _get_marker_name(node.func)
    return None


def _extract_assertions(node: ast.FunctionDef) -> list[str]:
    """Extract assertion patterns from a test function."""
    assertions: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            # Summarize the assert expression
            assertions.append(ast.dump(child.test, indent=0)[:120])
        elif isinstance(child, ast.Call):
            func_name = _call_to_str(child)
            if func_name and any(
                kw in func_name.lower()
                for kw in ("assert", "raises", "warns", "expect")
            ):
                assertions.append(func_name)
    return assertions


def _extract_function_calls(node: ast.FunctionDef) -> list[str]:
    """Extract all function/method calls within a test body."""
    calls: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _call_to_str(child)
            if name:
                calls.append(name)
    return list(dict.fromkeys(calls))  # deduplicate, preserve order


def _call_to_str(node: ast.Call) -> str | None:
    """Convert a Call node to a readable string like 'foo.bar'."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        current = func.value
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _compute_complexity(node: ast.FunctionDef) -> int:
    """Count control flow branches as a simple complexity metric."""
    count = 0
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
            count += 1
    return count


def extract_module_imports(tree: ast.Module) -> list[str]:
    """Extract top-level import names from a module."""
    imports: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    return imports


def extract_tests(file_path: Path) -> list[PytestTestInfo]:
    """Parse a Python file and extract all test function metadata."""
    source_text = file_path.read_text()
    tree = ast.parse(source_text)
    imports = extract_module_imports(tree)
    extractor = TestExtractor(source_text, imports)
    extractor.visit(tree)
    return extractor.tests


# ---------------------------------------------------------------------------
# Text representation builders
# ---------------------------------------------------------------------------

def build_representations(test: PytestTestInfo) -> TextRepresentations:
    """Build multiple text representations for a single test."""

    # Option A: Name + markers only (minimal)
    name_text = f"Test: {test.qualname}."
    if test.markers:
        name_text += f" Markers: {', '.join(test.markers)}."

    # Option A+: Name + docstring + markers
    doc_text = name_text
    if test.docstring:
        doc_text += f" {test.docstring}"

    # Option B: Full source code
    full_source = test.source

    # Option C: AST-based (structural features)
    ast_parts = [f"Test: {test.qualname}."]
    if test.fixtures:
        ast_parts.append(f"Fixtures: {', '.join(test.fixtures)}.")
    if test.assertions:
        # Summarize assertion types rather than full AST dumps
        assert_types = set()
        for a in test.assertions:
            if "Compare" in a:
                assert_types.add("equality check")
            elif "Is" in a and "None" in a:
                assert_types.add("none check")
            elif "Not" in a:
                assert_types.add("negation check")
            elif "In" in a:
                assert_types.add("membership check")
            elif "hasattr" in a.lower():
                assert_types.add("attribute check")
            elif "isinstance" in a.lower():
                assert_types.add("type check")
            elif "raises" in a.lower():
                assert_types.add("exception check")
            else:
                assert_types.add("assertion")
        ast_parts.append(f"Checks: {', '.join(sorted(assert_types))}.")
    if test.called_functions:
        # Filter out common noise (assert, len, etc.)
        meaningful = [
            f for f in test.called_functions
            if f not in ("len", "set", "list", "dict", "tuple", "str", "int",
                         "print", "range", "type", "isinstance", "hasattr")
        ]
        if meaningful:
            ast_parts.append(f"Calls: {', '.join(meaningful[:10])}.")
    ast_parts.append(f"Complexity: {test.complexity}.")
    ast_text = " ".join(ast_parts)

    # Option D: Combined (name + markers + fixtures + simplified body)
    combined_parts = [f"Test: {test.qualname}."]
    if test.markers:
        combined_parts.append(f"Markers: {', '.join(test.markers)}.")
    if test.docstring:
        combined_parts.append(test.docstring)
    if test.fixtures:
        combined_parts.append(f"Uses fixtures: {', '.join(test.fixtures)}.")
    if test.called_functions:
        meaningful = [
            f for f in test.called_functions
            if f not in ("len", "set", "list", "dict", "tuple", "str", "int",
                         "print", "range", "type", "isinstance", "hasattr")
        ]
        if meaningful:
            combined_parts.append(
                f"Calls: {', '.join(meaningful[:10])}."
            )
    if test.assertions:
        assert_summary = []
        for a in test.assertions:
            if "Compare" in a:
                assert_summary.append("checks equality")
            elif "Is" in a:
                assert_summary.append("checks identity/none")
            elif "Not" in a:
                assert_summary.append("checks negation")
            else:
                assert_summary.append("asserts condition")
        # deduplicate
        assert_summary = list(dict.fromkeys(assert_summary))
        combined_parts.append(
            f"Verifies: {', '.join(assert_summary)}."
        )
    combined_text = " ".join(combined_parts)

    return TextRepresentations(
        name_only=name_text,
        name_plus_docstring=doc_text,
        full_source=full_source,
        ast_based=ast_text,
        combined=combined_text,
    )


# ---------------------------------------------------------------------------
# Embedding and distance computation
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> NDArray[np.float32]:
    """Embed texts using all-MiniLM-L6-v2 via sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors: NDArray[np.float32] = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False,
    )
    return vectors


def cosine_distance_matrix(vectors: NDArray[np.float32]) -> NDArray[np.float64]:
    """Compute pairwise cosine distances (1 - cosine_similarity)."""
    from sklearn.metrics.pairwise import cosine_distances
    return cosine_distances(vectors)


def print_distance_matrix(
    matrix: NDArray[np.float64],
    labels: list[str],
    title: str,
) -> None:
    """Pretty-print a distance matrix."""
    n = len(labels)
    # Truncate labels for display
    short_labels = [
        (lbl[:35] + "..") if len(lbl) > 37 else lbl for lbl in labels
    ]
    col_width = max(len(s) for s in short_labels) + 2

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # Header
    header = " " * col_width
    for lbl in short_labels:
        header += f"{lbl:>{col_width}}"
    print(header)

    # Rows
    for i in range(n):
        row = f"{short_labels[i]:<{col_width}}"
        for j in range(n):
            row += f"{matrix[i, j]:>{col_width}.4f}"
        print(row)

    # Statistics
    upper = matrix[np.triu_indices(n, k=1)]
    if len(upper) > 0:
        print(f"\n  Mean pairwise distance: {upper.mean():.4f}")
        print(f"  Min  pairwise distance: {upper.min():.4f}")
        print(f"  Max  pairwise distance: {upper.max():.4f}")
        print(f"  Std  pairwise distance: {upper.std():.4f}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(test_file: Path) -> None:
    """Run the full experiment on a test file."""
    print(f"Experiment: Vectorizing pytest tests from {test_file}")
    print(f"{'=' * 80}\n")

    # Step 1: Extract tests
    tests = extract_tests(test_file)
    print(f"Found {len(tests)} test functions:\n")
    for i, t in enumerate(tests):
        print(f"  [{i:2d}] {t.qualname}")
        print(f"       fixtures={t.fixtures}, markers={t.markers}")
        print(f"       assertions={len(t.assertions)}, "
              f"calls={len(t.called_functions)}, "
              f"complexity={t.complexity}")
        if t.docstring:
            print(f"       docstring: {t.docstring[:80]}")
        print()

    # Step 2: Build text representations
    all_reps = [build_representations(t) for t in tests]

    print("\n--- Text Representations (first 3 tests) ---\n")
    for i, (test, rep) in enumerate(zip(tests[:3], all_reps[:3])):
        print(f"[{i}] {test.qualname}")
        print(f"    NAME_ONLY:    {rep.name_only[:100]}")
        print(f"    NAME+DOC:     {rep.name_plus_docstring[:100]}")
        print(f"    AST_BASED:    {rep.ast_based[:100]}")
        print(f"    COMBINED:     {rep.combined[:100]}")
        print(f"    FULL_SOURCE:  {rep.full_source[:100]}...")
        print()

    # Step 3: Embed each representation strategy
    labels = [t.qualname for t in tests]
    # Truncate labels for matrix display
    short_labels = []
    for lbl in labels:
        # Remove common class prefix for readability
        parts = lbl.split(".")
        if len(parts) == 2:
            short_labels.append(parts[1][:30])
        else:
            short_labels.append(lbl[:30])

    strategies = {
        "A: Name + Markers Only": [r.name_only for r in all_reps],
        "B: Full Source Code": [r.full_source for r in all_reps],
        "C: AST-Based Features": [r.ast_based for r in all_reps],
        "D: Combined (Recommended)": [r.combined for r in all_reps],
    }

    print("\nEmbedding all strategies with all-MiniLM-L6-v2 (384-dim)...")

    strategy_stats: dict[str, dict] = {}

    for strategy_name, texts in strategies.items():
        vectors = embed_texts(texts)
        dist_matrix = cosine_distance_matrix(vectors)
        print_distance_matrix(dist_matrix, short_labels, strategy_name)

        upper = dist_matrix[np.triu_indices(len(tests), k=1)]
        strategy_stats[strategy_name] = {
            "mean": float(upper.mean()),
            "min": float(upper.min()),
            "max": float(upper.max()),
            "std": float(upper.std()),
            "spread": float(upper.max() - upper.min()),
        }

    # Step 4: Comparative summary
    print(f"\n{'=' * 80}")
    print("  COMPARATIVE SUMMARY: Which strategy spreads tests best?")
    print(f"{'=' * 80}\n")
    print(f"  {'Strategy':<35} {'Mean':>8} {'Min':>8} {'Max':>8} "
          f"{'Std':>8} {'Spread':>8}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for name, stats in strategy_stats.items():
        print(f"  {name:<35} {stats['mean']:>8.4f} {stats['min']:>8.4f} "
              f"{stats['max']:>8.4f} {stats['std']:>8.4f} "
              f"{stats['spread']:>8.4f}")

    print(f"\n  Interpretation:")
    print(f"  - Higher MEAN distance = tests are more spread in embedding space")
    print(f"  - Higher STD = better discrimination between similar/different tests")
    print(f"  - Higher SPREAD (max-min) = strategy captures more variation")
    print(f"  - The strategy with highest spread best distinguishes test behaviors")

    # Step 5: Cluster analysis -- group by test class
    if any(t.class_name for t in tests):
        print(f"\n{'=' * 80}")
        print("  INTRA-CLASS vs INTER-CLASS DISTANCES (Combined strategy)")
        print(f"{'=' * 80}\n")

        combined_vectors = embed_texts([r.combined for r in all_reps])
        dist = cosine_distance_matrix(combined_vectors)

        classes = {}
        for i, t in enumerate(tests):
            cls = t.class_name or "<module>"
            classes.setdefault(cls, []).append(i)

        intra_dists = []
        inter_dists = []
        for i in range(len(tests)):
            for j in range(i + 1, len(tests)):
                if tests[i].class_name == tests[j].class_name:
                    intra_dists.append(dist[i, j])
                else:
                    inter_dists.append(dist[i, j])

        if intra_dists and inter_dists:
            print(f"  Test classes found: {list(classes.keys())}")
            print(f"  Intra-class mean distance: {np.mean(intra_dists):.4f} "
                  f"(n={len(intra_dists)} pairs)")
            print(f"  Inter-class mean distance: {np.mean(inter_dists):.4f} "
                  f"(n={len(inter_dists)} pairs)")
            ratio = np.mean(inter_dists) / np.mean(intra_dists)
            print(f"  Ratio (inter/intra):        {ratio:.4f}")
            print(f"\n  A ratio > 1.0 means embeddings cluster tests by class,")
            print(f"  indicating the model captures functional groupings.")


if __name__ == "__main__":
    default_file = Path(__file__).resolve().parents[2] / "tests" / "unit" / "test_cli.py"
    test_file = Path(sys.argv[1]) if len(sys.argv) > 1 else default_file

    if not test_file.exists():
        print(f"Error: {test_file} does not exist")
        sys.exit(1)

    run_experiment(test_file)
