#!/usr/bin/env python3
"""Rerun kmedoids, dpp, facility experiments with deps now installed."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

VENV_PYTHON = Path("/home/many/workspace/testcase-selection/.venv/bin/python")
DOCTEST_REPO = Path("/home/many/workspace/robotframework-doctestlibrary")
RESULTS_BASE = Path("/home/many/workspace/testcase-selection/results/deep_analysis")
COV_SOURCE = str(DOCTEST_REPO / "DocTest")
UTEST_DIR = DOCTEST_REPO / "utest"
ATEST_DIR = DOCTEST_REPO / "atest"
ROBOT_ARTIFACTS = Path(
    "/home/many/workspace/testcase-selection/results/coverage_comparison/robot_50pct"
)

STRATEGIES = ["kmedoids", "dpp", "facility"]
LEVELS = [0.50, 0.20, 0.10]
PYTEST_TOTAL = 594
ROBOT_TOTAL = 126


def load_env():
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


def run(cmd, cwd=None, env=None, timeout=600):
    return subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
    )


def extract_totals(cov_file):
    if not cov_file.exists():
        return 0, 0, 0.0
    data = json.loads(cov_file.read_text())
    t = data.get("totals", {})
    return (
        t.get("covered_lines", 0),
        t.get("num_statements", 0),
        round(t.get("percent_covered", 0.0), 2),
    )


def parse_pytest_summary(stdout):
    passed = failed = skipped = deselected = 0
    for pat, name in [
        (r"(\d+) passed", "p"),
        (r"(\d+) failed", "f"),
        (r"(\d+) skipped", "s"),
        (r"(\d+) deselected", "d"),
    ]:
        m = re.search(pat, stdout)
        if m:
            val = int(m.group(1))
            if name == "p":
                passed = val
            elif name == "f":
                failed = val
            elif name == "s":
                skipped = val
            elif name == "d":
                deselected = val
    return passed, failed, skipped, deselected


def parse_robot_summary(stdout):
    m = re.search(r"(\d+) tests?, (\d+) passed, (\d+) failed", stdout)
    if m:
        return int(m.group(2)), int(m.group(3))
    return 0, 0


def run_pytest_strategy(strategy, k, level, env):
    label = f"{strategy}_{int(level * 100)}pct"
    out_dir = RESULTS_BASE / "pytest" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_json = out_dir / "coverage.json"

    print(
        f"  [pytest] {strategy} {int(level*100)}% (k={k})...",
        end=" ",
        flush=True,
    )
    t0 = time.time()
    result = run(
        [
            str(VENV_PYTHON), "-m", "pytest", str(UTEST_DIR),
            f"--diverse-k={k}", f"--diverse-strategy={strategy}",
            "--diverse-seed=42",
            f"--cov={COV_SOURCE}",
            "--cov-report", f"json:{cov_json}",
            "--cov-report", "term", "-q",
        ],
        cwd=DOCTEST_REPO, env=env, timeout=600,
    )
    elapsed = time.time() - t0
    passed, failed, skipped, deselected = parse_pytest_summary(result.stdout)
    covered, total, pct = extract_totals(cov_json)
    tests_run = passed + failed + skipped

    # Sanity check: if tests_run == PYTEST_TOTAL, the strategy fell back
    if tests_run == PYTEST_TOTAL and k < PYTEST_TOTAL:
        print(f"FALLBACK (ran all {tests_run}), stderr: {result.stderr[-200:]}")
    else:
        print(f"{pct}% cov, {passed}p/{failed}f, {deselected} desel, {elapsed:.1f}s")

    return {
        "framework": "pytest", "strategy": strategy, "level": level, "k": k,
        "tests_run": tests_run, "tests_passed": passed, "tests_failed": failed,
        "tests_skipped": skipped, "lines_covered": covered, "lines_total": total,
        "coverage_pct": pct, "duration_sec": round(elapsed, 1),
    }


def run_robot_strategy(strategy, k, level, env):
    label = f"{strategy}_{int(level * 100)}pct"
    out_dir = RESULTS_BASE / "robot" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_json = out_dir / "coverage.json"
    sel_json = out_dir / "selection.json"

    print(
        f"  [robot] {strategy} {int(level*100)}% (k={k})...",
        end=" ",
        flush=True,
    )

    sel_result = run(
        [
            str(VENV_PYTHON), "-m", "TestSelection.cli", "select",
            "--artifacts", str(ROBOT_ARTIFACTS),
            "--k", str(k), "--strategy", strategy, "--seed", "42",
            "--output", str(sel_json),
        ],
        cwd=DOCTEST_REPO,
    )
    if sel_result.returncode != 0:
        print(f"SELECT FAILED: {sel_result.stderr[-300:]}")
        return {
            "framework": "robot", "strategy": strategy, "level": level, "k": k,
            "tests_run": 0, "tests_passed": 0, "tests_failed": 0,
            "tests_skipped": 0, "lines_covered": 0, "lines_total": 0,
            "coverage_pct": 0.0, "duration_sec": 0.0,
        }

    sel = json.loads(sel_json.read_text())
    tests = sel.get("selected_tests", sel.get("selected", []))
    names = [t.get("name", t) if isinstance(t, dict) else t for t in tests]
    metrics = sel.get("diversity_metrics", {})

    test_args = []
    for n in names:
        test_args.extend(["--test", n])

    t0 = time.time()
    result = run(
        [
            str(VENV_PYTHON), "-m", "coverage", "run", "--source", COV_SOURCE,
            "-m", "robot",
            "--outputdir", str(out_dir / "robot_output"),
            "--loglevel", "WARN",
            *test_args, str(ATEST_DIR),
        ],
        cwd=DOCTEST_REPO, env=env, timeout=600,
    )
    elapsed = time.time() - t0

    run(
        [str(VENV_PYTHON), "-m", "coverage", "json", "-o", str(cov_json)],
        cwd=DOCTEST_REPO,
    )

    passed, failed = parse_robot_summary(result.stdout)
    covered, total, pct = extract_totals(cov_json)

    print(f"{pct}% cov, {passed}p/{failed}f, {elapsed:.1f}s")
    return {
        "framework": "robot", "strategy": strategy, "level": level, "k": k,
        "tests_run": passed + failed, "tests_passed": passed, "tests_failed": failed,
        "tests_skipped": 0, "lines_covered": covered, "lines_total": total,
        "coverage_pct": pct, "duration_sec": round(elapsed, 1),
        "avg_pairwise_dist": metrics.get("avg_pairwise_distance", 0),
        "min_pairwise_dist": metrics.get("min_pairwise_distance", 0),
        "suite_coverage": (
            f"{metrics.get('suite_coverage', 0)}/{metrics.get('suite_total', 0)}"
        ),
    }


def main():
    env = load_env()
    new_results = []
    total_start = time.time()

    print("=" * 60)
    print("RERUNNING: kmedoids, dpp, facility (all deps installed)")
    print("=" * 60)

    print("\n--- PYTEST ---")
    for strategy in STRATEGIES:
        for level in LEVELS:
            k = math.ceil(PYTEST_TOTAL * level)
            new_results.append(run_pytest_strategy(strategy, k, level, env))

    print("\n--- ROBOT ---")
    for strategy in STRATEGIES:
        for level in LEVELS:
            k = math.ceil(ROBOT_TOTAL * level)
            new_results.append(run_robot_strategy(strategy, k, level, env))

    elapsed = time.time() - total_start

    # Load existing results and replace broken entries
    existing_file = RESULTS_BASE / "all_results.json"
    if existing_file.exists():
        existing = json.loads(existing_file.read_text())
    else:
        existing = []

    # Remove old entries for these strategies (they had fallback/zero data)
    existing = [
        r for r in existing
        if r["strategy"] not in STRATEGIES
    ]
    existing.extend(new_results)
    existing_file.write_text(json.dumps(existing, indent=2))

    # Print summary
    print(f"\n{'=' * 100}")
    print("RERUN RESULTS")
    print(f"{'=' * 100}")

    full_pytest = next(
        (r for r in existing if r["framework"] == "pytest" and r["strategy"] == "full"),
        None,
    )
    full_robot = next(
        (r for r in existing if r["framework"] == "robot" and r["strategy"] == "full"),
        None,
    )

    for fw, full in [("pytest", full_pytest), ("robot", full_robot)]:
        full_pct = full["coverage_pct"] if full else 0
        full_time = full["duration_sec"] if full else 1
        fw_results = sorted(
            [r for r in new_results if r["framework"] == fw],
            key=lambda r: (r["strategy"], -r["level"]),
        )
        if not fw_results:
            continue
        print(f"\n  {fw.upper()} (full baseline: {full_pct}%)")
        print(
            f"  {'Strategy':<14} {'Level':<8} {'K':<6} {'Run':<6} "
            f"{'Coverage':<10} {'Retention':<11} {'Time(s)':<9} {'Speedup'}"
        )
        print(f"  {'-' * 80}")
        for r in fw_results:
            if full_pct > 0 and r["coverage_pct"] > 0:
                ret = f"{r['coverage_pct']/full_pct*100:.1f}%"
            else:
                ret = "FAILED"
            spd = (
                f"{full_time/r['duration_sec']:.1f}x"
                if r["duration_sec"] > 0 else "N/A"
            )
            print(
                f"  {r['strategy']:<14} {int(r['level']*100)}%{'':>4} "
                f"{r['k']:<6} {r['tests_run']:<6} "
                f"{r['coverage_pct']:<10.1f} {ret:<11} "
                f"{r['duration_sec']:<9.1f} {spd}"
            )

    print(f"\nTotal rerun time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Updated results saved to {existing_file}")


if __name__ == "__main__":
    main()
