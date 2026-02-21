"""Integration tests for Stage 3: execution filtering with PreRunModifier."""
from __future__ import annotations

import json

import pytest
from robot.api import TestSuite as RobotTestSuite

from TestSelection.execution.prerun_modifier import DiversePreRunModifier
from TestSelection.execution.runner import ExecutionRunner
from TestSelection.pipeline.select import run_select


@pytest.mark.integration
class TestStage3Execute:
    """End-to-end tests for execution filtering using fake artifacts."""

    def test_prerun_modifier_filters_tests(self, fake_artifacts, sample_suite_path, tmp_path):
        artifact_dir, raw_tests = fake_artifacts
        k = 5

        # Run selection to get a selection file
        output_file = tmp_path / "selected.json"
        result = run_select(artifact_dir=artifact_dir, k=k, output_file=output_file)

        # Apply PreRunModifier to the real suite
        modifier = DiversePreRunModifier(str(output_file))
        suite = RobotTestSuite.from_file_system(str(sample_suite_path))
        suite.visit(modifier)

        # Count remaining tests
        remaining = list(suite.tests)
        selected_names = {t.name for t in result.selected}
        remaining_names = {t.name for t in remaining}

        assert remaining_names == selected_names
        assert len(remaining) == k

    def test_modifier_stats_are_correct(self, fake_artifacts, sample_suite_path, tmp_path):
        artifact_dir, raw_tests = fake_artifacts
        k = 5

        output_file = tmp_path / "selected.json"
        run_select(artifact_dir=artifact_dir, k=k, output_file=output_file)

        modifier = DiversePreRunModifier(str(output_file))
        suite = RobotTestSuite.from_file_system(str(sample_suite_path))
        suite.visit(modifier)

        assert modifier.stats["kept"] == k
        assert modifier.stats["removed"] == len(raw_tests) - k

    def test_selection_report_generated(self, fake_artifacts, sample_suite_path, tmp_path):
        artifact_dir, raw_tests = fake_artifacts
        k = 5

        selection_file = tmp_path / "selected.json"
        run_select(artifact_dir=artifact_dir, k=k, output_file=selection_file)

        report_dir = tmp_path / "report_output"
        runner = ExecutionRunner(
            suite_path=sample_suite_path,
            selection_file=selection_file,
            output_dir=report_dir,
        )

        # Generate report without actually running robot
        report = runner.generate_report(return_code=0)

        assert report["selected_tests"] == k
        assert report["standard_tests"] == k
        assert report["datadriver_tests"] == 0
        assert report["strategy"] == "fps"
        assert report["status"] == "pass"

        report_path = report_dir / "selection_report.json"
        assert report_path.exists()

        saved_report = json.loads(report_path.read_text())
        assert saved_report["selected_tests"] == k

    def test_runner_builds_correct_robot_args(self, fake_artifacts, sample_suite_path, tmp_path):
        artifact_dir, raw_tests = fake_artifacts

        selection_file = tmp_path / "selected.json"
        run_select(artifact_dir=artifact_dir, k=5, output_file=selection_file)

        runner = ExecutionRunner(
            suite_path=sample_suite_path,
            selection_file=selection_file,
            output_dir=tmp_path / "results",
        )
        args = runner.build_robot_args()

        assert "--prerunmodifier" in args
        assert "--outputdir" in args
        # No datadriver tests in fake artifacts, so no --listener
        assert "--listener" not in args
