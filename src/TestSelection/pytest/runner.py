"""Programmatic pytest execution for the testcase-select CLI."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


def run_pytest_with_selection(
    suite_path: Path,
    selection_file: Path,
    output_dir: str = "./results",
    extra_args: list[str] | None = None,
    strategy: str = "fps",
    k: int = 0,
    seed: int = 42,
    model_name: str = "all-MiniLM-L6-v2",
) -> int:
    """Run pytest with diverse selection enabled.

    If selection_file is provided and exists, uses it as a pre-computed
    node list. Otherwise, uses the plugin for on-the-fly selection.

    Returns the pytest exit code.
    """
    args: list[str] = [str(suite_path)]

    if selection_file.exists():
        # Use pre-computed selection: pass nodeids directly
        nodeids = _load_nodeids(selection_file)
        if nodeids:
            # Replace suite path with specific nodeids
            args = list(nodeids)
    elif k > 0:
        # Use plugin for on-the-fly selection
        args.extend([
            f"--diverse-k={k}",
            f"--diverse-strategy={strategy}",
            f"--diverse-seed={seed}",
            f"--diverse-model={model_name}",
        ])

    # Output configuration
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    args.extend([
        f"--junitxml={output_path / 'results.xml'}",
        f"--html={output_path / 'report.html'}",
    ] if _has_pytest_html() else [
        f"--junitxml={output_path / 'results.xml'}",
    ])

    # Extra args (from -- passthrough)
    if extra_args:
        args.extend(extra_args)

    logger.info(
        "[DIVERSE-SELECT] stage=execute framework=pytest args=%s",
        " ".join(args),
    )

    return pytest.main(args)


def run_pytest_pipeline(
    suite_path: Path,
    k: int,
    strategy: str = "fps",
    seed: int = 42,
    model_name: str = "all-MiniLM-L6-v2",
    output_dir: str = "./results",
    extra_args: list[str] | None = None,
) -> int:
    """Run the full pipeline for pytest: vectorize + select + execute.

    This is the pytest equivalent of _cmd_run for Robot Framework.
    Uses the plugin directly for simplicity.
    """
    args = [
        str(suite_path),
        f"--diverse-k={k}",
        f"--diverse-strategy={strategy}",
        f"--diverse-seed={seed}",
        f"--diverse-model={model_name}",
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    args.append(f"--junitxml={output_path / 'results.xml'}")

    if extra_args:
        args.extend(extra_args)

    return pytest.main(args)


def _load_nodeids(selection_file: Path) -> list[str]:
    """Load test nodeids from a selection JSON file."""
    data = json.loads(selection_file.read_text())
    nodeids: list[str] = []
    for test in data.get("selected", []):
        nodeid = test.get("nodeid") or test.get("name", "")
        if nodeid:
            nodeids.append(nodeid)
    return nodeids


def _has_pytest_html() -> bool:
    """Check if pytest-html is available."""
    try:
        import pytest_html  # noqa: F401
        return True
    except ImportError:
        return False
