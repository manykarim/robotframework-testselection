"""CLI entry points for the diverse test selection pipeline."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("TestSelection")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )


def _add_framework_arg(parser: argparse.ArgumentParser) -> None:
    """Add --framework option to a parser."""
    parser.add_argument(
        "--framework",
        choices=["robot", "pytest"],
        default="robot",
        help="Test framework: robot (default) or pytest.",
    )


def _add_vectorize_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "vectorize", help="Stage 1: vectorize test suite",
    )
    _add_framework_arg(p)
    p.add_argument(
        "--suite", required=True, type=Path,
        help="Path to test suite directory",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Artifact output dir",
    )
    p.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    p.add_argument(
        "--resolve-depth", type=int, default=0,
        help="Keyword resolve depth (Robot Framework only)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Force re-indexing",
    )
    p.add_argument(
        "--datadriver-csv", nargs="*", type=Path,
        help="DataDriver CSV files (Robot Framework only)",
    )
    p.set_defaults(func=_cmd_vectorize)


def _add_select_parser(subparsers: argparse._SubParsersAction) -> None:
    default_k = int(os.environ.get("DIVERSE_K", "50"))
    default_strategy = os.environ.get("DIVERSE_STRATEGY", "fps")
    default_seed = int(os.environ.get("DIVERSE_SEED", "42"))
    default_output = os.environ.get("DIVERSE_OUTPUT", "")

    p = subparsers.add_parser(
        "select", help="Stage 2: select diverse subset",
    )
    p.add_argument(
        "--artifacts", required=True, type=Path,
        help="Artifact directory",
    )
    p.add_argument(
        "--k", type=int, default=default_k,
        help="Number of tests to select",
    )
    p.add_argument(
        "--strategy", default=default_strategy,
        help="Selection strategy",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path(default_output) if default_output else None,
        help="Output selection file",
    )
    p.add_argument(
        "--include-tags", nargs="*",
        help="Include only tests with these tags",
    )
    p.add_argument(
        "--exclude-tags", nargs="*",
        help="Exclude tests with these tags",
    )
    p.add_argument(
        "--seed", type=int, default=default_seed,
        help="Random seed",
    )
    p.add_argument(
        "--no-datadriver", action="store_true",
        help="Exclude DataDriver tests",
    )
    p.set_defaults(func=_cmd_select)


def _add_execute_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "execute",
        help="Stage 3: execute selected tests",
        epilog=(
            "All arguments after -- are passed directly to the test runner. "
            "Example: testcase-select execute --suite tests/ "
            "--selection sel.json "
            "-- --variable ENV:staging --loglevel DEBUG"
        ),
    )
    _add_framework_arg(p)
    p.add_argument(
        "--suite", required=True, type=Path,
        help="Path to test suite directory",
    )
    p.add_argument(
        "--selection", required=True, type=Path,
        help="Selection JSON file",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("./results"),
        help="Output dir",
    )
    p.set_defaults(func=_cmd_execute)


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    default_k = int(os.environ.get("DIVERSE_K", "50"))
    default_strategy = os.environ.get("DIVERSE_STRATEGY", "fps")
    default_seed = int(os.environ.get("DIVERSE_SEED", "42"))

    p = subparsers.add_parser(
        "run",
        help="Full pipeline: vectorize + select + execute",
        epilog=(
            "All arguments after -- are passed directly to the test runner. "
            "Example: testcase-select run --suite tests/ --k 20 "
            "-- --include smoke --variable ENV:staging"
        ),
    )
    _add_framework_arg(p)
    p.add_argument(
        "--suite", required=True, type=Path,
        help="Path to test suite directory",
    )
    p.add_argument(
        "--k", type=int, default=default_k,
        help="Number of tests to select",
    )
    p.add_argument(
        "--strategy", default=default_strategy,
        help="Selection strategy",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("./results"),
        help="Output dir",
    )
    p.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    p.add_argument(
        "--resolve-depth", type=int, default=0,
        help="Keyword resolve depth (Robot Framework only)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Force re-indexing",
    )
    p.add_argument(
        "--seed", type=int, default=default_seed,
        help="Random seed",
    )
    p.set_defaults(func=_cmd_run)


def _cmd_vectorize(args: argparse.Namespace) -> int:
    framework = getattr(args, "framework", "robot")

    if framework == "pytest":
        return _cmd_vectorize_pytest(args)

    from TestSelection.pipeline.vectorize import run_vectorize

    try:
        indexed = run_vectorize(
            suite_path=args.suite,
            artifact_dir=args.output,
            model_name=args.model,
            resolve_depth=args.resolve_depth,
            force=args.force,
            datadriver_csvs=args.datadriver_csv,
        )
        if indexed:
            logger.info("[DIVERSE-SELECT] Vectorization complete")
        else:
            logger.info(
                "[DIVERSE-SELECT] Vectorization skipped (no changes)"
            )
        return 0
    except Exception as exc:
        logger.error("[DIVERSE-SELECT] Vectorization failed: %s", exc)
        return 2


def _cmd_vectorize_pytest(args: argparse.Namespace) -> int:
    """Vectorize pytest tests."""
    from TestSelection.pipeline.vectorize_pytest import run_vectorize_pytest

    try:
        indexed = run_vectorize_pytest(
            suite_path=args.suite,
            artifact_dir=args.output,
            model_name=args.model,
            force=args.force,
        )
        if indexed:
            logger.info("[DIVERSE-SELECT] Vectorization complete (pytest)")
        else:
            logger.info(
                "[DIVERSE-SELECT] Vectorization skipped (no changes)"
            )
        return 0
    except Exception as exc:
        logger.error("[DIVERSE-SELECT] Vectorization failed: %s", exc)
        return 2


def _cmd_select(args: argparse.Namespace) -> int:
    from TestSelection.pipeline.select import run_select

    try:
        result = run_select(
            artifact_dir=args.artifacts,
            k=args.k,
            strategy=args.strategy,
            output_file=args.output,
            include_tags=args.include_tags,
            exclude_tags=args.exclude_tags,
            seed=args.seed,
            include_datadriver=not args.no_datadriver,
        )
        logger.info(
            "[DIVERSE-SELECT] Selected %d tests (strategy=%s)",
            len(result.selected),
            result.strategy,
        )
        return 0
    except Exception as exc:
        logger.error("[DIVERSE-SELECT] Selection failed: %s", exc)
        return 2


def _cmd_execute(args: argparse.Namespace) -> int:
    framework = getattr(args, "framework", "robot")

    if framework == "pytest":
        return _cmd_execute_pytest(args)

    from TestSelection.pipeline.execute import run_execute

    return run_execute(
        suite_path=args.suite,
        selection_file=args.selection,
        output_dir=str(args.output_dir),
        extra_robot_args=args.robot_passthrough,
    )


def _cmd_execute_pytest(args: argparse.Namespace) -> int:
    """Execute pytest with selection."""
    from TestSelection.pytest.runner import run_pytest_with_selection

    return run_pytest_with_selection(
        suite_path=args.suite,
        selection_file=args.selection,
        output_dir=str(args.output_dir),
        extra_args=args.robot_passthrough,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    framework = getattr(args, "framework", "robot")

    if framework == "pytest":
        return _cmd_run_pytest(args)

    from TestSelection.pipeline.execute import run_execute
    from TestSelection.pipeline.select import run_select
    from TestSelection.pipeline.vectorize import run_vectorize

    artifact_dir = args.output_dir / ".artifacts"

    # Stage 1: Vectorize
    try:
        run_vectorize(
            suite_path=args.suite,
            artifact_dir=artifact_dir,
            model_name=args.model,
            resolve_depth=args.resolve_depth,
            force=args.force,
        )
    except Exception as exc:
        logger.warning(
            "[DIVERSE-SELECT] Vectorization failed, "
            "falling back to all tests: %s",
            exc,
        )
        return _fallback_execute(args)

    # Stage 2: Select
    selection_file = artifact_dir / "selected_tests.json"
    try:
        run_select(
            artifact_dir=artifact_dir,
            k=args.k,
            strategy=args.strategy,
            output_file=selection_file,
            seed=args.seed,
        )
    except Exception as exc:
        logger.warning(
            "[DIVERSE-SELECT] Selection failed, "
            "falling back to all tests: %s",
            exc,
        )
        return _fallback_execute(args)

    # Stage 3: Execute
    return run_execute(
        suite_path=args.suite,
        selection_file=selection_file,
        output_dir=str(args.output_dir),
        extra_robot_args=args.robot_passthrough,
    )


def _cmd_run_pytest(args: argparse.Namespace) -> int:
    """Run full pipeline for pytest."""
    from TestSelection.pytest.runner import run_pytest_pipeline

    try:
        return run_pytest_pipeline(
            suite_path=args.suite,
            k=args.k,
            strategy=args.strategy,
            seed=args.seed,
            model_name=args.model,
            output_dir=str(args.output_dir),
            extra_args=args.robot_passthrough,
        )
    except Exception as exc:
        logger.warning(
            "[DIVERSE-SELECT] pytest pipeline failed, "
            "falling back to all tests: %s",
            exc,
        )
        return _fallback_execute_pytest(args)


def _fallback_execute(args: argparse.Namespace) -> int:
    """Execute all tests without selection (graceful degradation)."""
    logger.info("[DIVERSE-SELECT] Running all tests (no selection)")
    try:
        import robot

        robot_args = [
            "--outputdir",
            str(args.output_dir),
        ]
        if args.robot_passthrough:
            robot_args.extend(args.robot_passthrough)
        robot_args.append(str(args.suite))
        return robot.run_cli(robot_args, exit=False)  # type: ignore[attr-defined]
    except Exception as exc:
        logger.error(
            "[DIVERSE-SELECT] Fallback execution failed: %s", exc,
        )
        return 2


def _fallback_execute_pytest(args: argparse.Namespace) -> int:
    """Execute all pytest tests without selection (graceful degradation)."""
    logger.info("[DIVERSE-SELECT] Running all pytest tests (no selection)")
    try:
        import pytest

        pytest_args = [str(args.suite)]
        if args.robot_passthrough:
            pytest_args.extend(args.robot_passthrough)
        return pytest.main(pytest_args)
    except Exception as exc:
        logger.error(
            "[DIVERSE-SELECT] Fallback execution failed: %s", exc,
        )
        return 2


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="testcase-select",
        description=(
            "Vector-based diverse test case selection "
            "for Robot Framework and pytest"
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands",
    )
    _add_vectorize_parser(subparsers)
    _add_select_parser(subparsers)
    _add_execute_parser(subparsers)
    _add_run_parser(subparsers)
    return parser


def _split_robot_passthrough(
    argv: list[str],
) -> tuple[list[str], list[str]]:
    """Split argv at -- into our args and passthrough args.

    Returns (our_args, passthrough_args). If no -- is present,
    passthrough_args is empty.
    """
    try:
        sep = argv.index("--")
        return argv[:sep], argv[sep + 1:]
    except ValueError:
        return argv, []


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Arguments after -- are passed through to the test runner:
        testcase-select run --suite tests/ --k 20 \\
            -- --variable ENV:staging --loglevel DEBUG --include smoke
    """
    if argv is None:
        argv = sys.argv[1:]

    our_args, robot_args = _split_robot_passthrough(argv)

    parser = build_parser()
    args = parser.parse_args(our_args)
    args.robot_passthrough = robot_args or None

    _setup_logging(getattr(args, "verbose", False))

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
