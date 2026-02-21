"""Integration test for the full pipeline: parse -> fake embed -> select -> filter."""
from __future__ import annotations

import pytest
from robot.api import TestSuite as RobotTestSuite

from TestSelection.execution.prerun_modifier import DiversePreRunModifier
from TestSelection.pipeline.select import run_select


@pytest.mark.integration
class TestFullPipeline:
    """End-to-end pipeline test using real parsing and fake embeddings."""

    def test_full_parse_select_filter_flow(self, fake_artifacts, sample_suite_path, tmp_path):
        artifact_dir, raw_tests = fake_artifacts
        k = 6

        # Stage 2: Select
        selection_file = tmp_path / "selected.json"
        result = run_select(artifact_dir=artifact_dir, k=k, output_file=selection_file)

        # Stage 3: Filter via PreRunModifier
        modifier = DiversePreRunModifier(str(selection_file))
        suite = RobotTestSuite.from_file_system(str(sample_suite_path))
        suite.visit(modifier)

        remaining = list(suite.tests)
        assert len(remaining) == k

    def test_selected_names_are_valid_subset(self, fake_artifacts, sample_suite_path, tmp_path):
        artifact_dir, raw_tests = fake_artifacts
        k = 4

        selection_file = tmp_path / "selected.json"
        result = run_select(artifact_dir=artifact_dir, k=k, output_file=selection_file)

        all_names = {t["name"] for t in raw_tests}
        selected_names = {t.name for t in result.selected}

        assert selected_names.issubset(all_names)
        assert len(selected_names) == k

    def test_pipeline_with_different_strategies(self, fake_artifacts, sample_suite_path, tmp_path):
        artifact_dir, raw_tests = fake_artifacts
        k = 5

        for strategy in ("fps", "fps_multi"):
            selection_file = tmp_path / f"selected_{strategy}.json"
            result = run_select(
                artifact_dir=artifact_dir,
                k=k,
                strategy=strategy,
                output_file=selection_file,
            )
            assert result.strategy == strategy
            assert len(result.selected) == k

            modifier = DiversePreRunModifier(str(selection_file))
            suite = RobotTestSuite.from_file_system(str(sample_suite_path))
            suite.visit(modifier)
            remaining = list(suite.tests)
            assert len(remaining) == k

    def test_pipeline_with_tag_filtering(self, fake_artifacts, sample_suite_path, tmp_path):
        artifact_dir, raw_tests = fake_artifacts

        # Select from only "smoke" tagged tests
        selection_file = tmp_path / "selected_smoke.json"
        result = run_select(
            artifact_dir=artifact_dir,
            k=10,
            include_tags=["smoke"],
            output_file=selection_file,
        )

        # Apply modifier
        modifier = DiversePreRunModifier(str(selection_file))
        suite = RobotTestSuite.from_file_system(str(sample_suite_path))
        suite.visit(modifier)

        remaining = list(suite.tests)
        # Smoke-tagged tests in sample.robot: Login With Valid Credentials, Search Products By Name,
        # Add Single Item To Cart, Complete Purchase Flow = 4
        assert len(remaining) == len(result.selected)
        assert len(remaining) <= 10

    def test_diversity_metrics_meaningful(self, fake_artifacts, tmp_path):
        artifact_dir, raw_tests = fake_artifacts

        result = run_select(artifact_dir=artifact_dir, k=5)

        metrics = result.diversity_metrics
        # With random vectors, pairwise distances should be positive
        assert metrics.avg_pairwise_distance > 0
        assert metrics.min_pairwise_distance >= 0
        assert metrics.suite_coverage >= 1
        assert metrics.suite_total >= 1
