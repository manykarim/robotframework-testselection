from __future__ import annotations

import json
from pathlib import Path

from TestSelection.execution.runner import ExecutionRunner


def _write_selection(
    tmp_path: Path, selected: list[dict], strategy: str = "fps",
) -> Path:
    """Write a selection JSON file and return its path."""
    selection_file = tmp_path / "selected_tests.json"
    data = {
        "strategy": strategy,
        "k": len(selected),
        "total_tests": 14,
        "filtered_tests": 14,
        "seed": 42,
        "selected": selected,
    }
    selection_file.write_text(json.dumps(data))
    return selection_file


class TestExecutionRunner:
    def test_build_args_includes_prerunmodifier(self, tmp_path: Path) -> None:
        """Build args always include --prerunmodifier."""
        selection_file = _write_selection(
            tmp_path,
            [{"name": "Test A", "is_datadriver": False}],
        )
        runner = ExecutionRunner(
            suite_path="/some/tests",
            selection_file=selection_file,
            output_dir=tmp_path / "results",
        )

        args = runner.build_robot_args()

        assert "--prerunmodifier" in args
        # Find the value after --prerunmodifier
        idx = args.index("--prerunmodifier")
        modifier_value = args[idx + 1]
        assert "DiversePreRunModifier" in modifier_value
        assert str(selection_file) in modifier_value

    def test_build_args_includes_listener_with_datadriver(self, tmp_path: Path) -> None:
        """Build args include --listener when DataDriver tests are present."""
        selection_file = _write_selection(
            tmp_path,
            [
                {"name": "Standard Test", "is_datadriver": False},
                {"name": "DD Test 1", "is_datadriver": True},
            ],
        )
        runner = ExecutionRunner(
            suite_path="/some/tests",
            selection_file=selection_file,
            output_dir=tmp_path / "results",
        )

        args = runner.build_robot_args()

        assert "--listener" in args
        idx = args.index("--listener")
        listener_value = args[idx + 1]
        assert "DiverseDataDriverListener" in listener_value

    def test_build_args_omits_listener_without_datadriver(self, tmp_path: Path) -> None:
        """Build args omit --listener when no DataDriver tests."""
        selection_file = _write_selection(
            tmp_path,
            [
                {"name": "Test A", "is_datadriver": False},
                {"name": "Test B", "is_datadriver": False},
            ],
        )
        runner = ExecutionRunner(
            suite_path="/some/tests",
            selection_file=selection_file,
            output_dir=tmp_path / "results",
        )

        args = runner.build_robot_args()

        assert "--listener" not in args

    def test_build_args_includes_outputdir(self, tmp_path: Path) -> None:
        """Build args include --outputdir."""
        selection_file = _write_selection(
            tmp_path,
            [{"name": "Test A", "is_datadriver": False}],
        )
        output_dir = tmp_path / "results"
        runner = ExecutionRunner(
            suite_path="/some/tests",
            selection_file=selection_file,
            output_dir=output_dir,
        )

        args = runner.build_robot_args()

        assert "--outputdir" in args
        idx = args.index("--outputdir")
        assert args[idx + 1] == str(output_dir)

    def test_generate_report_creates_json(self, tmp_path: Path) -> None:
        """Generate report creates selection_report.json with correct structure."""
        selection_file = _write_selection(
            tmp_path,
            [
                {"name": "Test A", "is_datadriver": False},
                {"name": "Test B", "is_datadriver": False},
                {"name": "DD Test", "is_datadriver": True},
            ],
            strategy="fps_multi",
        )
        output_dir = tmp_path / "results"
        runner = ExecutionRunner(
            suite_path="/some/tests",
            selection_file=selection_file,
            output_dir=output_dir,
        )

        report = runner.generate_report(return_code=0)

        assert report["return_code"] == 0
        assert report["selected_tests"] == 3
        assert report["datadriver_tests"] == 1
        assert report["standard_tests"] == 2
        assert report["strategy"] == "fps_multi"
        assert report["status"] == "pass"

        # Verify file was written
        report_path = output_dir / "selection_report.json"
        assert report_path.exists()
        written = json.loads(report_path.read_text())
        assert written == report

    def test_generate_report_fail_status(self, tmp_path: Path) -> None:
        """Generate report marks status as fail for non-zero return code."""
        selection_file = _write_selection(
            tmp_path,
            [{"name": "Test A", "is_datadriver": False}],
        )
        runner = ExecutionRunner(
            suite_path="/some/tests",
            selection_file=selection_file,
            output_dir=tmp_path / "results",
        )

        report = runner.generate_report(return_code=1)

        assert report["status"] == "fail"
        assert report["return_code"] == 1
