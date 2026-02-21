"""Tests for CLI argument parsing and robot passthrough."""
from __future__ import annotations

from TestSelection.cli import _split_robot_passthrough, build_parser


class TestSplitRobotPassthrough:
    def test_no_separator_returns_empty(self) -> None:
        our, robot = _split_robot_passthrough(
            ["run", "--suite", "tests/", "--k", "20"],
        )
        assert our == ["run", "--suite", "tests/", "--k", "20"]
        assert robot == []

    def test_separator_splits_correctly(self) -> None:
        our, robot = _split_robot_passthrough(
            ["run", "--suite", "tests/", "--",
             "--variable", "ENV:staging", "--loglevel", "DEBUG"],
        )
        assert our == ["run", "--suite", "tests/"]
        assert robot == [
            "--variable", "ENV:staging", "--loglevel", "DEBUG",
        ]

    def test_separator_at_end_gives_empty_robot_args(self) -> None:
        our, robot = _split_robot_passthrough(
            ["run", "--suite", "tests/", "--"],
        )
        assert our == ["run", "--suite", "tests/"]
        assert robot == []

    def test_separator_at_start(self) -> None:
        our, robot = _split_robot_passthrough(
            ["--", "--include", "smoke"],
        )
        assert our == []
        assert robot == ["--include", "smoke"]

    def test_empty_argv(self) -> None:
        our, robot = _split_robot_passthrough([])
        assert our == []
        assert robot == []

    def test_multiple_robot_flags(self) -> None:
        our, robot = _split_robot_passthrough([
            "execute", "--suite", "t/", "--selection", "s.json",
            "--",
            "--variable", "USER:admin",
            "--variable", "PASS:secret",
            "--include", "smoke",
            "--exclude", "manual",
            "--loglevel", "DEBUG",
            "--metadata", "Version:1.0",
        ])
        assert our == [
            "execute", "--suite", "t/", "--selection", "s.json",
        ]
        assert robot == [
            "--variable", "USER:admin",
            "--variable", "PASS:secret",
            "--include", "smoke",
            "--exclude", "manual",
            "--loglevel", "DEBUG",
            "--metadata", "Version:1.0",
        ]


class TestParserWithPassthrough:
    def test_execute_parser_no_passthrough(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "execute", "--suite", "tests/",
            "--selection", "sel.json",
        ])
        assert args.command == "execute"
        assert not hasattr(args, "robot_passthrough")

    def test_run_parser_no_passthrough(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "run", "--suite", "tests/",
        ])
        assert args.command == "run"
        assert not hasattr(args, "robot_passthrough")


class TestMainPassthrough:
    def test_main_sets_robot_passthrough(self) -> None:
        # We can't easily test main() end-to-end without
        # running robot, but we can test the arg splitting
        our, robot = _split_robot_passthrough([
            "run", "--suite", "tests/", "--k", "5",
            "--", "--variable", "X:1",
        ])
        parser = build_parser()
        args = parser.parse_args(our)
        args.robot_passthrough = robot or None

        assert args.command == "run"
        assert args.robot_passthrough == ["--variable", "X:1"]

    def test_main_sets_none_without_passthrough(self) -> None:
        our, robot = _split_robot_passthrough([
            "run", "--suite", "tests/", "--k", "5",
        ])
        parser = build_parser()
        args = parser.parse_args(our)
        args.robot_passthrough = robot or None

        assert args.robot_passthrough is None
