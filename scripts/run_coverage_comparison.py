#!/usr/bin/env python3
"""Coverage comparison: full suite vs diverse selection on robotframework-doctestlibrary.

Runs the doctestlibrary acceptance tests with coverage measurement:
1. Full suite run with coverage
2. Vectorize + select diverse subset
3. Selected subset run with coverage
4. Compare coverage reports
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DOCTEST_DIR = ROOT / "example-suite" / "robotframework-doctestlibrary"
ATEST_DIR = DOCTEST_DIR / "atest"
DOCTEST_SRC = DOCTEST_DIR / "DocTest"
RESULTS_DIR = ROOT / "results"
FULL_DIR = RESULTS_DIR / "full_suite"
SELECTED_DIR = RESULTS_DIR / "selected_suite"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"

# tests that require external infra (browser, specific HW) — skip them
SKIP_SUITES = {"Browser.robot"}

# ── env ────────────────────────────────────────────────────────────────────
_OPENROUTER_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)
_OPENROUTER_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
ENV = {
    **os.environ,
    "OPENAI_API_KEY": _OPENROUTER_KEY,
    "OPENAI_BASE_URL": _OPENROUTER_URL,
    # Route DocTest LLM to a vision-capable model on OpenRouter
    "DOCTEST_LLM_MODEL": "openai/gpt-4o-mini",
    "DOCTEST_LLM_VISION_MODEL": "openai/gpt-4o-mini",
}


def ensure_dirs() -> None:
    for d in (FULL_DIR, SELECTED_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def list_runnable_suites() -> list[Path]:
    """Return .robot files we can actually run."""
    return sorted(
        p for p in ATEST_DIR.glob("*.robot") if p.name not in SKIP_SUITES
    )


def run_with_coverage(
    tag: str,
    output_dir: Path,
    suites: list[Path],
    extra_robot_args: list[str] | None = None,
) -> dict:
    """Run robot tests under coverage, return stats dict."""
    cov_data = output_dir / ".coverage"
    cov_html = output_dir / "htmlcov"
    cov_json = output_dir / "coverage.json"

    # Build robot command
    robot_args = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--source",
        str(DOCTEST_SRC),
        "--data-file",
        str(cov_data),
        "-m",
        "robot",
        "--outputdir",
        str(output_dir),
        "--loglevel",
        "WARN",
        "--nostatusrc",
    ]
    if extra_robot_args:
        robot_args.extend(extra_robot_args)
    robot_args.extend(str(s) for s in suites)

    print(f"\n{'='*60}")
    print(f"  Running {tag}: {len(suites)} suite file(s)")
    print(f"{'='*60}")

    t0 = time.monotonic()
    result = subprocess.run(
        robot_args,
        cwd=str(DOCTEST_DIR),
        env=ENV,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.monotonic() - t0

    print(f"  Robot exit code: {result.returncode}")
    if result.returncode not in (0, 1):
        # 0=pass, 1=some tests failed — both OK for coverage
        print(f"  STDERR (last 500 chars): {result.stderr[-500:]}")

    # Generate coverage reports
    subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "json",
            "--data-file",
            str(cov_data),
            "-o",
            str(cov_json),
        ],
        cwd=str(DOCTEST_DIR),
        capture_output=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "html",
            "--data-file",
            str(cov_data),
            "-d",
            str(cov_html),
        ],
        cwd=str(DOCTEST_DIR),
        capture_output=True,
    )

    # Parse output.xml for test counts
    from robot.api import ExecutionResult

    try:
        rf_result = ExecutionResult(str(output_dir / "output.xml"))
        stats = rf_result.statistics.total
        total_tests = stats.passed + stats.failed + stats.skipped
        passed = stats.passed
        failed = stats.failed
        skipped = stats.skipped
    except Exception:
        total_tests = passed = failed = skipped = -1

    # Parse coverage JSON
    cov_stats = {"total_stmts": 0, "covered_stmts": 0, "pct": 0.0, "files": {}}
    if cov_json.exists():
        with open(cov_json) as f:
            cov_raw = json.load(f)
        totals = cov_raw.get("totals", {})
        cov_stats["total_stmts"] = totals.get("num_statements", 0)
        cov_stats["covered_stmts"] = totals.get("covered_lines", 0)
        cov_stats["missing_stmts"] = totals.get("missing_lines", 0)
        cov_stats["pct"] = totals.get("percent_covered", 0.0)
        for fname, fdata in cov_raw.get("files", {}).items():
            short = fname.replace(str(DOCTEST_SRC) + "/", "")
            cov_stats["files"][short] = {
                "stmts": fdata["summary"]["num_statements"],
                "covered": fdata["summary"]["covered_lines"],
                "missing": fdata["summary"]["missing_lines"],
                "pct": fdata["summary"]["percent_covered"],
            }

    return {
        "tag": tag,
        "suites": len(suites),
        "total_tests": total_tests,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "elapsed_s": round(elapsed, 1),
        "coverage": cov_stats,
    }


def run_vectorize_and_select(
    suite_dir: Path, k: int, strategy: str = "fps"
) -> list[str]:
    """Run testcase-selection vectorize + select, return selected test names."""
    # Stage 1: Vectorize
    print(f"\n{'='*60}")
    print(f"  Vectorizing {suite_dir} with sentence-transformers")
    print(f"{'='*60}")
    vec_args = [
        sys.executable,
        "-m",
        "TestSelection.cli",
        "vectorize",
        "--suite",
        str(suite_dir),
        "--output",
        str(ARTIFACTS_DIR),
        "--model",
        "all-MiniLM-L6-v2",
        "--force",
    ]

    t0 = time.monotonic()
    result = subprocess.run(
        vec_args, capture_output=True, text=True, timeout=300
    )
    vec_time = time.monotonic() - t0
    print(f"  Vectorize exit code: {result.returncode} ({vec_time:.1f}s)")
    if result.stdout:
        print(f"  STDOUT: {result.stdout[-500:]}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        sys.exit(1)

    # Stage 2: Select
    print(f"\n{'='*60}")
    print(f"  Selecting k={k} tests via {strategy}")
    print(f"{'='*60}")
    sel_args = [
        sys.executable,
        "-m",
        "TestSelection.cli",
        "select",
        "--artifacts",
        str(ARTIFACTS_DIR),
        "--k",
        str(k),
        "--strategy",
        strategy,
        "--seed",
        "42",
    ]
    result = subprocess.run(
        sel_args, capture_output=True, text=True, timeout=60
    )
    print(f"  Select exit code: {result.returncode}")
    if result.stdout:
        print(f"  STDOUT: {result.stdout[-500:]}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        sys.exit(1)

    # Load selection result
    sel_file = ARTIFACTS_DIR / "selected_tests.json"
    with open(sel_file) as f:
        sel_data = json.load(f)
    selected_names = [t["name"] for t in sel_data["selected"]]
    print(f"  Selected {len(selected_names)} tests")
    for name in selected_names:
        print(f"    - {name}")
    return selected_names


def build_include_args(selected_names: list[str]) -> list[str]:
    """Build --test arguments for robot to run only selected tests."""
    args = []
    for name in selected_names:
        args.extend(["--test", name])
    return args


def print_comparison(full: dict, selected: dict) -> None:
    """Print a side-by-side comparison table."""
    fc = full["coverage"]
    sc = selected["coverage"]

    print(f"\n{'='*70}")
    print("  COVERAGE COMPARISON: Full Suite vs Diverse Selection")
    print(f"{'='*70}")
    print(f"  {'Metric':<35} {'Full':>12} {'Selected':>12} {'Ratio':>8}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*8}")
    print(
        f"  {'Test files (suites)':<35} {full['suites']:>12} {selected['suites']:>12}"
    )
    print(
        f"  {'Total tests':<35} {full['total_tests']:>12} {selected['total_tests']:>12} {selected['total_tests']/max(full['total_tests'],1)*100:>7.1f}%"
    )
    print(
        f"  {'Passed':<35} {full['passed']:>12} {selected['passed']:>12}"
    )
    print(
        f"  {'Failed':<35} {full['failed']:>12} {selected['failed']:>12}"
    )
    print(
        f"  {'Skipped':<35} {full['skipped']:>12} {selected['skipped']:>12}"
    )
    print(
        f"  {'Execution time (s)':<35} {full['elapsed_s']:>12.1f} {selected['elapsed_s']:>12.1f} {selected['elapsed_s']/max(full['elapsed_s'],0.1)*100:>7.1f}%"
    )
    print()
    print(
        f"  {'Code statements':<35} {fc['total_stmts']:>12} {sc['total_stmts']:>12}"
    )
    print(
        f"  {'Covered statements':<35} {fc['covered_stmts']:>12} {sc['covered_stmts']:>12} {sc['covered_stmts']/max(fc['covered_stmts'],1)*100:>7.1f}%"
    )
    print(
        f"  {'Missing statements':<35} {fc.get('missing_stmts',0):>12} {sc.get('missing_stmts',0):>12}"
    )
    print(
        f"  {'Overall coverage %':<35} {fc['pct']:>11.1f}% {sc['pct']:>11.1f}%"
    )

    # Per-file breakdown
    print(f"\n  {'Per-file coverage breakdown':}")
    print(f"  {'File':<40} {'Full %':>8} {'Sel %':>8} {'Delta':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
    all_files = sorted(set(list(fc["files"].keys()) + list(sc["files"].keys())))
    for fname in all_files:
        f_pct = fc["files"].get(fname, {}).get("pct", 0.0)
        s_pct = sc["files"].get(fname, {}).get("pct", 0.0)
        delta = s_pct - f_pct
        marker = "" if abs(delta) < 0.1 else (" +" if delta > 0 else " -")
        print(f"  {fname:<40} {f_pct:>7.1f}% {s_pct:>7.1f}% {delta:>+7.1f}%{marker}")

    # Efficiency metric
    test_ratio = selected["total_tests"] / max(full["total_tests"], 1)
    cov_ratio = sc["pct"] / max(fc["pct"], 0.1)
    print(f"\n  Coverage efficiency: {cov_ratio:.2%} coverage with {test_ratio:.2%} of tests")
    print(f"  Time saved: {full['elapsed_s'] - selected['elapsed_s']:.1f}s ({(1 - selected['elapsed_s']/max(full['elapsed_s'],0.1))*100:.1f}% faster)")


def main() -> None:
    ensure_dirs()

    suites = list_runnable_suites()
    print(f"Found {len(suites)} runnable suite files:")
    for s in suites:
        print(f"  - {s.name}")

    # ── 1. Full suite run with coverage ────────────────────────────────
    full_stats = run_with_coverage("full_suite", FULL_DIR, suites)

    # ── 2. Vectorize & select ──────────────────────────────────────────
    total_tests = full_stats["total_tests"]
    # Select roughly 40% of tests for a meaningful comparison
    k = max(5, total_tests * 40 // 100)
    print(f"\n  Total tests found: {total_tests}, selecting k={k}")
    selected_names = run_vectorize_and_select(ATEST_DIR, k=k)

    # ── 3. Selected subset run with coverage ───────────────────────────
    include_args = build_include_args(selected_names)
    selected_stats = run_with_coverage(
        "selected_suite", SELECTED_DIR, suites, extra_robot_args=include_args
    )

    # ── 4. Compare ─────────────────────────────────────────────────────
    print_comparison(full_stats, selected_stats)

    # Save comparison JSON
    comparison = {
        "full_suite": full_stats,
        "selected_suite": selected_stats,
    }
    comp_file = RESULTS_DIR / "comparison.json"
    with open(comp_file, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\n  Detailed results saved to {comp_file}")
    print(f"  Full coverage HTML:     {FULL_DIR / 'htmlcov' / 'index.html'}")
    print(f"  Selected coverage HTML: {SELECTED_DIR / 'htmlcov' / 'index.html'}")


if __name__ == "__main__":
    main()
