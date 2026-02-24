"""Tests for CLI --framework option and pytest dispatch."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from TestSelection.cli import build_parser, main


class TestBuildParserFrameworkOption:
    def test_run_subcommand_accepts_framework_pytest(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "run", "--suite", "tests/", "--framework", "pytest",
        ])
        assert args.framework == "pytest"

    def test_run_subcommand_defaults_to_robot(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--suite", "tests/"])
        assert args.framework == "robot"

    def test_execute_subcommand_accepts_framework_pytest(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "execute", "--suite", "tests/",
            "--selection", "sel.json",
            "--framework", "pytest",
        ])
        assert args.framework == "pytest"

    def test_execute_subcommand_defaults_to_robot(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "execute", "--suite", "tests/",
            "--selection", "sel.json",
        ])
        assert args.framework == "robot"

    def test_vectorize_subcommand_accepts_framework_pytest(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "vectorize", "--suite", "tests/",
            "--output", "artifacts/",
            "--framework", "pytest",
        ])
        assert args.framework == "pytest"

    def test_vectorize_subcommand_defaults_to_robot(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "vectorize", "--suite", "tests/",
            "--output", "artifacts/",
        ])
        assert args.framework == "robot"


class TestCmdRunPytestDispatch:
    @patch("TestSelection.cli._cmd_run_pytest")
    def test_run_with_framework_pytest_dispatches(
        self, mock_run_pytest: MagicMock,
    ) -> None:
        mock_run_pytest.return_value = 0

        exit_code = main([
            "run", "--suite", "tests/", "--framework", "pytest",
        ])

        mock_run_pytest.assert_called_once()
        assert exit_code == 0

    @patch("TestSelection.cli._cmd_run_pytest")
    def test_run_passes_args_to_cmd_run_pytest(
        self, mock_run_pytest: MagicMock,
    ) -> None:
        mock_run_pytest.return_value = 0

        main([
            "run", "--suite", "my_tests/",
            "--framework", "pytest",
            "--k", "10",
            "--strategy", "kmedoids",
        ])

        args = mock_run_pytest.call_args[0][0]
        assert args.framework == "pytest"
        assert args.k == 10
        assert args.strategy == "kmedoids"


class TestCmdExecutePytestDispatch:
    @patch("TestSelection.cli._cmd_execute_pytest")
    def test_execute_with_framework_pytest_dispatches(
        self, mock_exec_pytest: MagicMock,
    ) -> None:
        mock_exec_pytest.return_value = 0

        exit_code = main([
            "execute", "--suite", "tests/",
            "--selection", "sel.json",
            "--framework", "pytest",
        ])

        mock_exec_pytest.assert_called_once()
        assert exit_code == 0


class TestCmdVectorizePytestDispatch:
    @patch("TestSelection.cli._cmd_vectorize_pytest")
    def test_vectorize_with_framework_pytest_dispatches(
        self, mock_vec_pytest: MagicMock,
    ) -> None:
        mock_vec_pytest.return_value = 0

        exit_code = main([
            "vectorize", "--suite", "tests/",
            "--output", "artifacts/",
            "--framework", "pytest",
        ])

        mock_vec_pytest.assert_called_once()
        assert exit_code == 0
