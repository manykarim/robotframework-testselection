"""AST-based text representation builder for pytest test functions.

Uses the Combined strategy (D) validated experimentally to produce
embeddable natural-language descriptions of test behavior.
"""
from __future__ import annotations

import ast

from TestSelection.pytest.collector import PytestTestRecord


def build_combined_text(record: PytestTestRecord) -> str:
    """Build a combined text representation for embedding.

    Combines test name, markers, docstring, fixtures, called functions,
    and assertion types into a natural-language string. This approach
    achieved a 5.9x inter/intra class ratio in experiments.
    """
    parts: list[str] = []

    # Qualified name
    if record.class_name:
        parts.append(f"Test: {record.class_name}.{record.originalname}.")
    else:
        parts.append(f"Test: {record.originalname}.")

    # Markers
    if record.markers:
        parts.append(f"Markers: {', '.join(record.markers)}.")

    # Docstring
    if record.docstring:
        parts.append(record.docstring)

    # Fixtures
    if record.fixtures:
        parts.append(f"Uses fixtures: {', '.join(record.fixtures)}.")

    # AST-based features from source code
    if record.source_code:
        try:
            calls, assertions = _extract_ast_features(record.source_code)
            if calls:
                parts.append(f"Calls: {', '.join(calls[:10])}.")
            if assertions:
                parts.append(f"Verifies: {', '.join(assertions)}.")
        except SyntaxError:
            pass

    return " ".join(parts)


def build_combined_text_from_item(item) -> str:
    """Build combined text directly from a pytest.Item (for plugin use).

    Avoids the full PytestTestRecord overhead when called from the plugin hook.
    """
    import inspect

    parts: list[str] = []

    # Name
    cls = getattr(item, "cls", None)
    originalname = getattr(item, "originalname", item.name)
    if cls:
        parts.append(f"Test: {cls.__name__}.{originalname}.")
    else:
        parts.append(f"Test: {originalname}.")

    # Markers
    marker_names = [
        m.name for m in item.iter_markers()
        if m.name not in ("parametrize", "usefixtures")
    ]
    if marker_names:
        parts.append(f"Markers: {', '.join(marker_names)}.")

    # Docstring
    func = getattr(item, "function", None) or getattr(item, "obj", None)
    if func:
        doc = inspect.getdoc(func)
        if doc:
            parts.append(doc)

    # Fixtures
    fixtureinfo = getattr(item, "_fixtureinfo", None)
    if fixtureinfo:
        fixtures = [
            n for n in fixtureinfo.argnames
            if n not in ("self", "cls", "request")
        ]
        if fixtures:
            parts.append(f"Uses fixtures: {', '.join(fixtures)}.")

    # AST features from source
    if func:
        try:
            source = inspect.getsource(func)
            calls, assertions = _extract_ast_features(source)
            if calls:
                parts.append(f"Calls: {', '.join(calls[:10])}.")
            if assertions:
                parts.append(f"Verifies: {', '.join(assertions)}.")
        except (OSError, TypeError, SyntaxError):
            pass

    return " ".join(parts)


def _extract_ast_features(
    source: str,
) -> tuple[list[str], list[str]]:
    """Extract function calls and assertion types from source code AST.

    Returns (called_functions, assertion_types).
    """
    # Parse the source into an AST
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []

    # Find the function definition (first one)
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_node = node
            break

    if func_node is None:
        return [], []

    calls: list[str] = []
    assertion_types: set[str] = set()

    _noise = frozenset((
        "len", "set", "list", "dict", "tuple", "str", "int", "float",
        "print", "range", "type", "isinstance", "hasattr", "getattr",
        "setattr", "sorted", "enumerate", "zip", "map", "filter",
        "repr", "bool", "any", "all", "min", "max", "sum", "abs",
    ))

    for child in ast.walk(func_node):
        if isinstance(child, ast.Assert):
            assertion_types.add(_classify_assertion(child.test))

        if isinstance(child, ast.Call):
            name = _call_name(child)
            if name:
                if any(
                    kw in name.lower()
                    for kw in ("assert", "raises", "warns", "expect")
                ):
                    assertion_types.add(_classify_call_assertion(name))
                elif name not in _noise:
                    calls.append(name)

    # Deduplicate calls, preserving order
    seen: set[str] = set()
    unique_calls: list[str] = []
    for c in calls:
        if c not in seen:
            seen.add(c)
            unique_calls.append(c)

    return unique_calls, sorted(assertion_types)


def _classify_assertion(node: ast.expr) -> str:
    """Classify an assert expression into a human-readable type."""
    if isinstance(node, ast.Compare):
        for op in node.ops:
            if isinstance(op, (ast.Eq, ast.NotEq)):
                return "checks equality"
            if isinstance(op, (ast.Is, ast.IsNot)):
                return "checks identity"
            if isinstance(op, (ast.In, ast.NotIn)):
                return "checks membership"
            if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                return "checks ordering"
        return "checks comparison"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return "checks negation"
    if isinstance(node, ast.Call):
        name = _call_name(node)
        if name and "isinstance" in name:
            return "checks type"
        if name and "hasattr" in name:
            return "checks attribute"
    return "asserts condition"


def _classify_call_assertion(name: str) -> str:
    """Classify an assertion-like function call."""
    lower = name.lower()
    if "raises" in lower:
        return "checks exception"
    if "warns" in lower:
        return "checks warning"
    if "equal" in lower:
        return "checks equality"
    return "asserts condition"


def _call_name(node: ast.Call) -> str | None:
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
