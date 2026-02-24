#!/usr/bin/env python3
"""Deep coverage analysis: all strategies x selection levels x frameworks.

Strategies: fps, fps_multi, kmedoids, dpp, facility, random (baseline)
Levels: 50%, 20%, 10%
Frameworks: pytest (594 tests), robot (126 tests)
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# --- Configuration ---

VENV_PYTHON = Path("/home/many/workspace/testcase-selection/.venv/bin/python")
DOCTEST_REPO = Path("/home/many/workspace/robotframework-doctestlibrary")
RESULTS_BASE = Path("/home/many/workspace/testcase-selection/results/deep_analysis")
COV_SOURCE = str(DOCTEST_REPO / "DocTest")
UTEST_DIR = DOCTEST_REPO / "utest"
ATEST_DIR = DOCTEST_REPO / "atest"

# Reuse embeddings from previous run
ROBOT_ARTIFACTS = Path("/home/many/workspace/testcase-selection/results/coverage_comparison/robot_50pct")

STRATEGIES = ["fps", "fps_multi", "kmedoids", "dpp", "facility"]
LEVELS = [0.50, 0.20, 0.10]

PYTEST_TOTAL = 594
ROBOT_TOTAL = 126


@dataclass
class ExperimentResult:
    framework: str
    strategy: str
    level: float
    k: int
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    lines_covered: int
    lines_total: int
    coverage_pct: float
    duration_sec: float
    avg_pairwise_dist: float = 0.0
    min_pairwise_dist: float = 0.0
    suite_coverage: str = ""


def load_env() -> dict[str, str]:
    """Load .env from doctestlibrary."""
    env = dict(os.environ)
    dotenv = DOCTEST_REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip()
    return env


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
        timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
    )


def parse_coverage_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def extract_totals(cov: dict) -> tuple[int, int, float]:
    """Return (covered, total, pct)."""
    if not cov:
        return 0, 0, 0.0
    t = cov.get("totals", {})
    return t.get("covered_lines", 0), t.get("num_statements", 0), round(t.get("percent_covered", 0.0), 2)


def parse_pytest_summary(stdout: str) -> tuple[int, int, int, int]:
    """Parse '295 passed, 2 skipped, 297 deselected' from pytest output."""
    import re
    passed = failed = skipped = deselected = 0
    m = re.search(r"(\d+) passed", stdout)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", stdout)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+) skipped", stdout)
    if m:
        skipped = int(m.group(1))
    m = re.search(r"(\d+) deselected", stdout)
    if m:
        deselected = int(m.group(1))
    return passed, failed, skipped, deselected


def parse_robot_summary(stdout: str) -> tuple[int, int, int]:
    """Parse robot summary line."""
    import re
    m = re.search(r"(\d+) tests?, (\d+) passed, (\d+) failed", stdout)
    if m:
        return int(m.group(2)), int(m.group(3)), 0
    return 0, 0, 0


# === PYTEST EXPERIMENTS ===

def run_pytest_full(env: dict) -> ExperimentResult:
    """Full pytest suite - baseline."""
    out_dir = RESULTS_BASE / "pytest" / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_json = out_dir / "coverage.json"

    print("  [pytest] Running FULL suite (594 tests)...")
    t0 = time.time()
    result = run(
        [str(VENV_PYTHON), "-m", "pytest", str(UTEST_DIR),
         f"--cov={COV_SOURCE}",
         "--cov-report", f"json:{cov_json}",
         "--cov-report", "term",
         "-q"],
        cwd=DOCTEST_REPO, env=env, timeout=600,
    )
    elapsed = time.time() - t0

    passed, failed, skipped, _ = parse_pytest_summary(result.stdout)
    covered, total, pct = extract_totals(parse_coverage_json(cov_json))

    print(f"    -> {pct}% coverage, {passed}p/{failed}f/{skipped}s in {elapsed:.1f}s")
    return ExperimentResult(
        framework="pytest", strategy="full", level=1.0, k=PYTEST_TOTAL,
        tests_run=passed + failed + skipped, tests_passed=passed,
        tests_failed=failed, tests_skipped=skipped,
        lines_covered=covered, lines_total=total, coverage_pct=pct,
        duration_sec=round(elapsed, 1),
    )


def run_pytest_random(k: int, level: float, seed: int, env: dict) -> ExperimentResult:
    """Random selection baseline - select k random tests."""
    out_dir = RESULTS_BASE / "pytest" / f"random_{int(level * 100)}pct"
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_json = out_dir / "coverage.json"

    # Collect test nodeids, then randomly pick k
    collect = run(
        [str(VENV_PYTHON), "-m", "pytest", str(UTEST_DIR), "--collect-only", "-q"],
        cwd=DOCTEST_REPO, env=env,
    )
    nodeids = [
        line for line in collect.stdout.splitlines()
        if "::" in line and not line.startswith(" ")
    ]

    import random
    rng = random.Random(seed)
    selected = rng.sample(nodeids, min(k, len(nodeids)))

    # Write to a temp file for --co deselection
    nodeid_file = out_dir / "selected_nodeids.txt"
    nodeid_file.write_text("\n".join(selected))

    print(f"  [pytest] Running RANDOM {int(level * 100)}% ({k} tests)...")
    t0 = time.time()
    result = run(
        [str(VENV_PYTHON), "-m", "pytest", *selected,
         f"--cov={COV_SOURCE}",
         "--cov-report", f"json:{cov_json}",
         "--cov-report", "term",
         "-q"],
        cwd=DOCTEST_REPO, env=env, timeout=600,
    )
    elapsed = time.time() - t0

    passed, failed, skipped, _ = parse_pytest_summary(result.stdout)
    covered, total, pct = extract_totals(parse_coverage_json(cov_json))

    print(f"    -> {pct}% coverage, {passed}p/{failed}f/{skipped}s in {elapsed:.1f}s")
    return ExperimentResult(
        framework="pytest", strategy="random", level=level, k=k,
        tests_run=passed + failed + skipped, tests_passed=passed,
        tests_failed=failed, tests_skipped=skipped,
        lines_covered=covered, lines_total=total, coverage_pct=pct,
        duration_sec=round(elapsed, 1),
    )


def run_pytest_strategy(strategy: str, k: int, level: float, env: dict) -> ExperimentResult:
    """Diverse selection with a specific strategy via the pytest plugin."""
    out_dir = RESULTS_BASE / "pytest" / f"{strategy}_{int(level * 100)}pct"
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_json = out_dir / "coverage.json"

    print(f"  [pytest] Running {strategy} {int(level * 100)}% ({k} tests)...")
    t0 = time.time()
    result = run(
        [str(VENV_PYTHON), "-m", "pytest", str(UTEST_DIR),
         f"--diverse-k={k}",
         f"--diverse-strategy={strategy}",
         "--diverse-seed=42",
         f"--cov={COV_SOURCE}",
         "--cov-report", f"json:{cov_json}",
         "--cov-report", "term",
         "-q"],
        cwd=DOCTEST_REPO, env=env, timeout=600,
    )
    elapsed = time.time() - t0

    passed, failed, skipped, deselected = parse_pytest_summary(result.stdout)
    covered, total, pct = extract_totals(parse_coverage_json(cov_json))

    print(f"    -> {pct}% coverage, {passed}p/{failed}f/{skipped}s, {deselected} deselected in {elapsed:.1f}s")
    return ExperimentResult(
        framework="pytest", strategy=strategy, level=level, k=k,
        tests_run=passed + failed + skipped, tests_passed=passed,
        tests_failed=failed, tests_skipped=skipped,
        lines_covered=covered, lines_total=total, coverage_pct=pct,
        duration_sec=round(elapsed, 1),
    )


# === ROBOT EXPERIMENTS ===

def run_robot_full(env: dict) -> ExperimentResult:
    """Full robot suite baseline."""
    out_dir = RESULTS_BASE / "robot" / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_json = out_dir / "coverage.json"

    print("  [robot] Running FULL suite (126 tests)...")
    t0 = time.time()
    result = run(
        [str(VENV_PYTHON), "-m", "coverage", "run", "--source", COV_SOURCE,
         "-m", "robot",
         "--outputdir", str(out_dir / "robot_output"),
         "--loglevel", "WARN",
         str(ATEST_DIR)],
        cwd=DOCTEST_REPO, env=env, timeout=600,
    )
    elapsed = time.time() - t0

    run([str(VENV_PYTHON), "-m", "coverage", "json", "-o", str(cov_json)], cwd=DOCTEST_REPO)

    passed, failed, _ = parse_robot_summary(result.stdout)
    covered, total, pct = extract_totals(parse_coverage_json(cov_json))

    print(f"    -> {pct}% coverage, {passed}p/{failed}f in {elapsed:.1f}s")
    return ExperimentResult(
        framework="robot", strategy="full", level=1.0, k=ROBOT_TOTAL,
        tests_run=passed + failed, tests_passed=passed,
        tests_failed=failed, tests_skipped=0,
        lines_covered=covered, lines_total=total, coverage_pct=pct,
        duration_sec=round(elapsed, 1),
    )


def run_robot_strategy(strategy: str, k: int, level: float, env: dict) -> ExperimentResult:
    """Robot with diverse selection via CLI select command."""
    label = f"{strategy}_{int(level * 100)}pct"
    out_dir = RESULTS_BASE / "robot" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_json = out_dir / "coverage.json"
    sel_json = out_dir / "selection.json"

    # Select using pre-computed artifacts
    print(f"  [robot] Selecting {strategy} {int(level * 100)}% ({k} tests)...")
    sel_result = run(
        [str(VENV_PYTHON), "-m", "TestSelection.cli", "select",
         "--artifacts", str(ROBOT_ARTIFACTS),
         "--k", str(k),
         "--strategy", strategy,
         "--seed", "42",
         "--output", str(sel_json)],
        cwd=DOCTEST_REPO,
    )
    if sel_result.returncode != 0:
        print(f"    [WARN] Select failed: {sel_result.stderr[-300:]}")
        return ExperimentResult(
            framework="robot", strategy=strategy, level=level, k=k,
            tests_run=0, tests_passed=0, tests_failed=0, tests_skipped=0,
            lines_covered=0, lines_total=0, coverage_pct=0.0, duration_sec=0.0,
        )

    # Parse selection
    sel = json.loads(sel_json.read_text())
    tests = sel.get("selected_tests", sel.get("selected", []))
    names = [t.get("name", t) if isinstance(t, dict) else t for t in tests]
    metrics = sel.get("diversity_metrics", {})

    # Build --test args
    test_args = []
    for n in names:
        test_args.extend(["--test", n])

    print(f"  [robot] Running {len(names)} tests...")
    t0 = time.time()
    result = run(
        [str(VENV_PYTHON), "-m", "coverage", "run", "--source", COV_SOURCE,
         "-m", "robot",
         "--outputdir", str(out_dir / "robot_output"),
         "--loglevel", "WARN",
         *test_args,
         str(ATEST_DIR)],
        cwd=DOCTEST_REPO, env=env, timeout=600,
    )
    elapsed = time.time() - t0

    run([str(VENV_PYTHON), "-m", "coverage", "json", "-o", str(cov_json)], cwd=DOCTEST_REPO)

    passed, failed, _ = parse_robot_summary(result.stdout)
    covered, total, pct = extract_totals(parse_coverage_json(cov_json))

    print(f"    -> {pct}% coverage, {passed}p/{failed}f in {elapsed:.1f}s")
    return ExperimentResult(
        framework="robot", strategy=strategy, level=level, k=k,
        tests_run=passed + failed, tests_passed=passed,
        tests_failed=failed, tests_skipped=0,
        lines_covered=covered, lines_total=total, coverage_pct=pct,
        duration_sec=round(elapsed, 1),
        avg_pairwise_dist=metrics.get("avg_pairwise_distance", 0),
        min_pairwise_dist=metrics.get("min_pairwise_distance", 0),
        suite_coverage=f"{metrics.get('suite_coverage', 0)}/{metrics.get('suite_total', 0)}",
    )


def run_robot_random(k: int, level: float, seed: int, env: dict) -> ExperimentResult:
    """Random robot baseline."""
    out_dir = RESULTS_BASE / "robot" / f"random_{int(level * 100)}pct"
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_json = out_dir / "coverage.json"

    # Get all test names from dryrun
    dryrun = run(
        [str(VENV_PYTHON), "-m", "robot", "--dryrun",
         "--outputdir", str(out_dir / "dryrun"),
         "--loglevel", "WARN",
         str(ATEST_DIR)],
        cwd=DOCTEST_REPO, env=env,
    )

    # Parse test names from dryrun output.xml
    import xml.etree.ElementTree as ET
    output_xml = out_dir / "dryrun" / "output.xml"
    all_tests = []
    if output_xml.exists():
        tree = ET.parse(output_xml)
        for test_elem in tree.iter("test"):
            all_tests.append(test_elem.get("name", ""))

    import random
    rng = random.Random(seed)
    selected = rng.sample(all_tests, min(k, len(all_tests)))

    test_args = []
    for n in selected:
        test_args.extend(["--test", n])

    print(f"  [robot] Running RANDOM {int(level * 100)}% ({len(selected)} tests)...")
    t0 = time.time()
    result = run(
        [str(VENV_PYTHON), "-m", "coverage", "run", "--source", COV_SOURCE,
         "-m", "robot",
         "--outputdir", str(out_dir / "robot_output"),
         "--loglevel", "WARN",
         *test_args,
         str(ATEST_DIR)],
        cwd=DOCTEST_REPO, env=env, timeout=600,
    )
    elapsed = time.time() - t0

    run([str(VENV_PYTHON), "-m", "coverage", "json", "-o", str(cov_json)], cwd=DOCTEST_REPO)

    passed, failed, _ = parse_robot_summary(result.stdout)
    covered, total, pct = extract_totals(parse_coverage_json(cov_json))

    print(f"    -> {pct}% coverage, {passed}p/{failed}f in {elapsed:.1f}s")
    return ExperimentResult(
        framework="robot", strategy="random", level=level, k=k,
        tests_run=passed + failed, tests_passed=passed,
        tests_failed=failed, tests_skipped=0,
        lines_covered=covered, lines_total=total, coverage_pct=pct,
        duration_sec=round(elapsed, 1),
    )


# === REPORTING ===

def print_results_table(results: list[ExperimentResult], framework: str) -> None:
    """Print formatted results for one framework."""
    fw_results = [r for r in results if r.framework == framework]
    if not fw_results:
        return

    full = next((r for r in fw_results if r.strategy == "full"), None)
    full_pct = full.coverage_pct if full else 0

    print(f"\n{'=' * 110}")
    total = PYTEST_TOTAL if framework == "pytest" else ROBOT_TOTAL
    print(f"  {framework.upper()} RESULTS ({total} total tests, {full_pct}% full coverage)")
    print(f"{'=' * 110}")
    print(f"  {'Strategy':<14} {'Level':<8} {'K':<6} {'Run':<6} {'Pass':<6} {'Fail':<6} "
          f"{'Covered':<9} {'Coverage':<10} {'Retention':<11} {'Time(s)':<9} {'Speedup':<8}")
    print(f"  {'-' * 106}")

    full_time = full.duration_sec if full else 1

    # Sort: full first, then by strategy name, then by level descending
    sorted_results = sorted(
        fw_results,
        key=lambda r: (
            0 if r.strategy == "full" else 1,
            r.strategy,
            -r.level,
        ),
    )

    for r in sorted_results:
        if r.strategy == "full":
            retention = "baseline"
            speedup = "1.0x"
        else:
            retention = f"{r.coverage_pct / full_pct * 100:.1f}%" if full_pct > 0 else "N/A"
            speedup = f"{full_time / r.duration_sec:.1f}x" if r.duration_sec > 0 else "N/A"

        level_str = f"{int(r.level * 100)}%" if r.strategy != "full" else "100%"
        print(f"  {r.strategy:<14} {level_str:<8} {r.k:<6} {r.tests_run:<6} {r.tests_passed:<6} "
              f"{r.tests_failed:<6} {r.lines_covered:<9} {r.coverage_pct:<10.1f} "
              f"{retention:<11} {r.duration_sec:<9.1f} {speedup:<8}")


def print_strategy_comparison(results: list[ExperimentResult], framework: str) -> None:
    """Print strategy comparison at each level."""
    full = next((r for r in results if r.framework == framework and r.strategy == "full"), None)
    if not full:
        return

    print(f"\n  Strategy ranking by coverage retention ({framework}):")
    for level in LEVELS:
        level_results = [
            r for r in results
            if r.framework == framework and abs(r.level - level) < 0.01
        ]
        level_results.sort(key=lambda r: r.coverage_pct, reverse=True)

        pct_label = f"{int(level * 100)}%"
        print(f"\n    {pct_label} selection:")
        for i, r in enumerate(level_results, 1):
            ret = r.coverage_pct / full.coverage_pct * 100 if full.coverage_pct > 0 else 0
            delta = r.coverage_pct - full.coverage_pct
            print(f"      {i}. {r.strategy:<14} {r.coverage_pct:>6.1f}% coverage "
                  f"({ret:.1f}% retention, {delta:+.1f}pp vs full)")


def save_results(results: list[ExperimentResult]) -> None:
    """Save all results to JSON."""
    data = [
        {
            "framework": r.framework,
            "strategy": r.strategy,
            "level": r.level,
            "k": r.k,
            "tests_run": r.tests_run,
            "tests_passed": r.tests_passed,
            "tests_failed": r.tests_failed,
            "tests_skipped": r.tests_skipped,
            "lines_covered": r.lines_covered,
            "lines_total": r.lines_total,
            "coverage_pct": r.coverage_pct,
            "duration_sec": r.duration_sec,
            "avg_pairwise_dist": r.avg_pairwise_dist,
            "min_pairwise_dist": r.min_pairwise_dist,
            "suite_coverage": r.suite_coverage,
        }
        for r in results
    ]
    out = RESULTS_BASE / "all_results.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to {out}")


def main() -> None:
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    env = load_env()
    all_results: list[ExperimentResult] = []

    total_start = time.time()

    # === PHASE 1: PYTEST ===
    print("\n" + "=" * 60)
    print("PHASE 1: PYTEST EXPERIMENTS")
    print("=" * 60)

    # Full baseline
    all_results.append(run_pytest_full(env))

    # Random baselines
    for level in LEVELS:
        k = math.ceil(PYTEST_TOTAL * level)
        all_results.append(run_pytest_random(k, level, seed=42, env=env))

    # Each strategy at each level
    for strategy in STRATEGIES:
        for level in LEVELS:
            k = math.ceil(PYTEST_TOTAL * level)
            all_results.append(run_pytest_strategy(strategy, k, level, env))

    # === PHASE 2: ROBOT ===
    print("\n" + "=" * 60)
    print("PHASE 2: ROBOT FRAMEWORK EXPERIMENTS")
    print("=" * 60)

    # Full baseline
    all_results.append(run_robot_full(env))

    # Random baselines
    for level in LEVELS:
        k = math.ceil(ROBOT_TOTAL * level)
        all_results.append(run_robot_random(k, level, seed=42, env=env))

    # Each strategy at each level
    for strategy in STRATEGIES:
        for level in LEVELS:
            k = math.ceil(ROBOT_TOTAL * level)
            all_results.append(run_robot_strategy(strategy, k, level, env))

    total_elapsed = time.time() - total_start

    # === RESULTS ===
    print("\n\n" + "#" * 110)
    print("#" + " " * 40 + "DEEP ANALYSIS RESULTS" + " " * 40 + "#")
    print("#" * 110)

    print_results_table(all_results, "pytest")
    print_strategy_comparison(all_results, "pytest")

    print_results_table(all_results, "robot")
    print_strategy_comparison(all_results, "robot")

    print(f"\n\nTotal experiment time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"Total experiments: {len(all_results)}")

    save_results(all_results)


if __name__ == "__main__":
    main()
