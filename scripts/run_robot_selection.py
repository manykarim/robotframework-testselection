"""Run robot with selected tests and coverage."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DOCTEST_REPO = Path("/home/many/workspace/robotframework-doctestlibrary")
RESULTS_BASE = Path("/home/many/workspace/testcase-selection/results/coverage_comparison")


def run_robot_with_selection(selection_file: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    sel = json.loads(selection_file.read_text())
    tests = sel.get("selected_tests", sel.get("selected", []))
    names = [t.get("name", t) if isinstance(t, dict) else t for t in tests]

    # Build --test args
    test_args = []
    for n in names:
        test_args.extend(["--test", n])

    cmd = [
        sys.executable, "-m", "coverage", "run", "--source=DocTest",
        "-m", "robot",
        "--outputdir", str(results_dir / "robot_output"),
        "--loglevel", "WARN",
        *test_args,
        "atest/",
    ]

    print(f"Running {len(names)} selected robot tests...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(DOCTEST_REPO))
    # Print last lines of output
    lines = result.stdout.strip().splitlines()
    for line in lines[-15:]:
        print(line)

    # Generate coverage reports
    cov_json = results_dir / "coverage.json"
    subprocess.run(
        [sys.executable, "-m", "coverage", "json", "-o", str(cov_json)],
        capture_output=True, cwd=str(DOCTEST_REPO),
    )
    r = subprocess.run(
        [sys.executable, "-m", "coverage", "report"],
        capture_output=True, text=True, cwd=str(DOCTEST_REPO),
    )
    for line in r.stdout.strip().splitlines()[-5:]:
        print(line)


if __name__ == "__main__":
    label = sys.argv[1]  # e.g. "robot_50pct"
    selection_file = RESULTS_BASE / label / "selection.json"
    results_dir = RESULTS_BASE / label
    run_robot_with_selection(selection_file, results_dir)
