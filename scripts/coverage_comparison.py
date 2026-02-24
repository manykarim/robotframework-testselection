"""Coverage comparison: full suite vs diverse selection at 50% and 20%.

Runs pytest tests from robotframework-doctestlibrary with coverage,
then compares results across selection levels.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

DOCTEST_REPO = Path("/home/many/workspace/robotframework-doctestlibrary")
TESTSEL_REPO = Path("/home/many/workspace/testcase-selection")
UTEST_DIR = DOCTEST_REPO / "utest"
ATEST_DIR = DOCTEST_REPO / "atest"
RESULTS_DIR = TESTSEL_REPO / "results" / "coverage_comparison"

# Source package to measure coverage on
COV_SOURCE = str(DOCTEST_REPO / "DocTest")


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    merged_env = {**os.environ, **(env or {})}
    print(f"  $ {' '.join(cmd[:6])}{'...' if len(cmd) > 6 else ''}")
    return subprocess.run(cmd, cwd=cwd, env=merged_env, capture_output=True, text=True, timeout=600)


def load_dotenv_manual(path: Path) -> dict[str, str]:
    """Load .env file into a dict (no library needed)."""
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip()
    return env


def get_coverage_data(cov_file: Path) -> dict:
    """Parse coverage JSON report."""
    if not cov_file.exists():
        return {}
    return json.loads(cov_file.read_text())


def run_full_pytest(results_dir: Path, env_vars: dict) -> dict:
    """Run full pytest suite with coverage."""
    results_dir.mkdir(parents=True, exist_ok=True)
    cov_data = results_dir / "coverage.json"

    result = run_cmd(
        [
            sys.executable, "-m", "pytest", str(UTEST_DIR),
            f"--cov={COV_SOURCE}",
            "--cov-report", f"json:{cov_data}",
            "--cov-report", f"html:{results_dir / 'htmlcov'}",
            "--cov-report", "term",
            "-x",  # stop on first failure to save time
            "--timeout=120",
            "-q",
        ],
        cwd=DOCTEST_REPO,
        env=env_vars,
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"  [WARN] pytest exited {result.returncode}")
        print(result.stderr[-1000:] if result.stderr else "")

    return get_coverage_data(cov_data)


def run_diverse_pytest(k: int, label: str, results_dir: Path, env_vars: dict) -> dict:
    """Run pytest with diverse selection plugin at k tests."""
    results_dir.mkdir(parents=True, exist_ok=True)
    cov_data = results_dir / "coverage.json"

    result = run_cmd(
        [
            sys.executable, "-m", "pytest", str(UTEST_DIR),
            f"--diverse-k={k}",
            "--diverse-strategy=fps",
            "--diverse-seed=42",
            f"--cov={COV_SOURCE}",
            "--cov-report", f"json:{cov_data}",
            "--cov-report", f"html:{results_dir / 'htmlcov'}",
            "--cov-report", "term",
            "--timeout=120",
            "-q",
        ],
        cwd=DOCTEST_REPO,
        env=env_vars,
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"  [WARN] pytest exited {result.returncode}")
        print(result.stderr[-1000:] if result.stderr else "")

    return get_coverage_data(cov_data)


def run_full_robot(results_dir: Path, env_vars: dict) -> dict:
    """Run full robot suite with coverage via coverage.py."""
    results_dir.mkdir(parents=True, exist_ok=True)
    cov_data = results_dir / "coverage.json"

    result = run_cmd(
        [
            sys.executable, "-m", "coverage", "run",
            "--source", COV_SOURCE,
            "-m", "robot",
            "--outputdir", str(results_dir / "robot_output"),
            "--loglevel", "WARN",
            str(ATEST_DIR),
        ],
        cwd=DOCTEST_REPO,
        env=env_vars,
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"  [WARN] robot exited {result.returncode}")

    # Generate JSON report
    run_cmd(
        [sys.executable, "-m", "coverage", "json", "-o", str(cov_data)],
        cwd=DOCTEST_REPO,
    )
    run_cmd(
        [sys.executable, "-m", "coverage", "html", "-d", str(results_dir / "htmlcov")],
        cwd=DOCTEST_REPO,
    )

    return get_coverage_data(cov_data)


def run_diverse_robot(k: int, label: str, results_dir: Path, env_vars: dict) -> dict:
    """Run robot suite with diverse selection via testcase-select."""
    results_dir.mkdir(parents=True, exist_ok=True)
    cov_data = results_dir / "coverage.json"

    # Step 1: Vectorize
    vec_result = run_cmd(
        [
            sys.executable, "-m", "TestSelection.cli",
            "vectorize", str(ATEST_DIR),
            "--output-dir", str(results_dir),
        ],
        cwd=DOCTEST_REPO,
    )
    if vec_result.returncode != 0:
        print(f"  [WARN] vectorize failed: {vec_result.stderr[-500:]}")
        return {}

    # Step 2: Select
    sel_result = run_cmd(
        [
            sys.executable, "-m", "TestSelection.cli",
            "select",
            "--input-dir", str(results_dir),
            "--output-dir", str(results_dir),
            "--k", str(k),
            "--strategy", "fps",
            "--seed", "42",
        ],
        cwd=DOCTEST_REPO,
    )
    if sel_result.returncode != 0:
        print(f"  [WARN] select failed: {sel_result.stderr[-500:]}")
        return {}

    # Read selected test names
    selection_file = results_dir / "selection.json"
    if not selection_file.exists():
        # Try finding it
        for f in results_dir.glob("*selection*.json"):
            selection_file = f
            break

    if not selection_file.exists():
        print("  [WARN] No selection file found")
        return {}

    selection = json.loads(selection_file.read_text())
    selected_tests = selection.get("selected_tests", selection.get("selected", []))
    if isinstance(selected_tests, list) and selected_tests and isinstance(selected_tests[0], dict):
        selected_names = [t.get("name", t.get("test_name", "")) for t in selected_tests]
    else:
        selected_names = selected_tests

    print(f"  Selected {len(selected_names)} tests out of total suite")

    # Step 3: Run with coverage using --include
    include_args = []
    for name in selected_names:
        include_args.extend(["--include", name])

    result = run_cmd(
        [
            sys.executable, "-m", "coverage", "run",
            "--source", COV_SOURCE,
            "-m", "robot",
            "--outputdir", str(results_dir / "robot_output"),
            "--loglevel", "WARN",
            *include_args,
            str(ATEST_DIR),
        ],
        cwd=DOCTEST_REPO,
        env=env_vars,
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    run_cmd(
        [sys.executable, "-m", "coverage", "json", "-o", str(cov_data)],
        cwd=DOCTEST_REPO,
    )

    return get_coverage_data(cov_data)


def extract_summary(cov_data: dict) -> dict:
    """Extract key metrics from coverage JSON."""
    if not cov_data:
        return {"lines_covered": 0, "lines_total": 0, "pct": 0.0}
    totals = cov_data.get("totals", {})
    return {
        "lines_covered": totals.get("covered_lines", 0),
        "lines_total": totals.get("num_statements", 0),
        "pct": totals.get("percent_covered", 0.0),
        "branches_covered": totals.get("covered_branches", 0),
        "branches_total": totals.get("num_branches", 0),
        "missing": totals.get("missing_lines", 0),
    }


def print_comparison(results: dict[str, dict]) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("COVERAGE COMPARISON RESULTS")
    print("=" * 80)

    # Header
    print(f"\n{'Configuration':<30} {'Tests':<8} {'Lines Covered':<15} {'Total Lines':<13} {'Coverage %':<12} {'Retention':<10}")
    print("-" * 88)

    full_pct = None
    for label, data in results.items():
        summary = data["summary"]
        tests = data.get("test_count", "?")
        pct = summary["pct"]
        if full_pct is None:
            full_pct = pct
            retention = "baseline"
        else:
            retention = f"{pct / full_pct * 100:.1f}%" if full_pct > 0 else "N/A"

        print(
            f"{label:<30} {str(tests):<8} {summary['lines_covered']:<15} "
            f"{summary['lines_total']:<13} {pct:<12.1f} {retention:<10}"
        )

    print("-" * 88)
    print()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load .env
    dotenv = load_dotenv_manual(DOCTEST_REPO / ".env")
    env_vars = {**dotenv}

    results: dict[str, dict] = {}

    # --- PYTEST TESTS ---
    print("\n" + "=" * 60)
    print("PHASE 1: PYTEST TESTS (utest/)")
    print("=" * 60)

    # Count total tests
    count_result = run_cmd(
        [sys.executable, "-m", "pytest", str(UTEST_DIR), "--collect-only", "-q"],
        cwd=DOCTEST_REPO,
        env=env_vars,
    )
    total_line = [l for l in count_result.stdout.splitlines() if "collected" in l]
    total_tests = 594
    if total_line:
        import re
        m = re.search(r"(\d+)\s+tests?\s+collected", total_line[0])
        if m:
            total_tests = int(m.group(1))

    k_50 = math.ceil(total_tests * 0.5)
    k_20 = math.ceil(total_tests * 0.2)

    print(f"\nTotal pytest tests: {total_tests}")
    print(f"50% selection: k={k_50}")
    print(f"20% selection: k={k_20}")

    # 1a. Full pytest suite
    print(f"\n--- Running FULL pytest suite ({total_tests} tests) ---")
    full_cov = run_full_pytest(RESULTS_DIR / "pytest_full", env_vars)
    results["pytest: 100% (full)"] = {
        "summary": extract_summary(full_cov),
        "test_count": total_tests,
    }

    # 1b. 50% diverse selection
    print(f"\n--- Running 50% diverse pytest ({k_50} tests) ---")
    cov_50 = run_diverse_pytest(k_50, "50pct", RESULTS_DIR / "pytest_50pct", env_vars)
    results["pytest: 50% diverse"] = {
        "summary": extract_summary(cov_50),
        "test_count": k_50,
    }

    # 1c. 20% diverse selection
    print(f"\n--- Running 20% diverse pytest ({k_20} tests) ---")
    cov_20 = run_diverse_pytest(k_20, "20pct", RESULTS_DIR / "pytest_20pct", env_vars)
    results["pytest: 20% diverse"] = {
        "summary": extract_summary(cov_20),
        "test_count": k_20,
    }

    # --- ROBOT FRAMEWORK TESTS ---
    print("\n" + "=" * 60)
    print("PHASE 2: ROBOT FRAMEWORK TESTS (atest/)")
    print("=" * 60)

    # Count robot tests
    robot_total = 126  # from dryrun earlier

    k_50_rf = math.ceil(robot_total * 0.5)
    k_20_rf = math.ceil(robot_total * 0.2)

    print(f"\nTotal robot tests: {robot_total}")
    print(f"50% selection: k={k_50_rf}")
    print(f"20% selection: k={k_20_rf}")

    # 2a. Full robot suite
    print(f"\n--- Running FULL robot suite ({robot_total} tests) ---")
    full_rf_cov = run_full_robot(RESULTS_DIR / "robot_full", env_vars)
    results["robot: 100% (full)"] = {
        "summary": extract_summary(full_rf_cov),
        "test_count": robot_total,
    }

    # 2b. 50% diverse robot
    print(f"\n--- Running 50% diverse robot ({k_50_rf} tests) ---")
    cov_50_rf = run_diverse_robot(k_50_rf, "50pct", RESULTS_DIR / "robot_50pct", env_vars)
    results["robot: 50% diverse"] = {
        "summary": extract_summary(cov_50_rf),
        "test_count": k_50_rf,
    }

    # 2c. 20% diverse robot
    print(f"\n--- Running 20% diverse robot ({k_20_rf} tests) ---")
    cov_20_rf = run_diverse_robot(k_20_rf, "20pct", RESULTS_DIR / "robot_20pct", env_vars)
    results["robot: 20% diverse"] = {
        "summary": extract_summary(cov_20_rf),
        "test_count": k_20_rf,
    }

    # --- PRINT RESULTS ---
    print_comparison(results)

    # Save results JSON
    report_file = RESULTS_DIR / "comparison_report.json"
    report_file.write_text(json.dumps(results, indent=2))
    print(f"Full report saved to: {report_file}")


if __name__ == "__main__":
    main()
