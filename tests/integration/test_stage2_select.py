"""Integration tests for Stage 2: selection with fake artifacts."""
from __future__ import annotations

import json

import pytest

from TestSelection.pipeline.select import run_select
from TestSelection.selection.strategy import SelectionResult


@pytest.mark.integration
class TestStage2Select:
    """End-to-end tests for the selection pipeline using fake embeddings."""

    def test_run_select_produces_valid_result(self, fake_artifacts):
        artifact_dir, raw_tests = fake_artifacts
        k = 5
        result = run_select(artifact_dir=artifact_dir, k=k)
        assert isinstance(result, SelectionResult)
        assert len(result.selected) == k

    def test_selected_tests_are_subset_of_total(self, fake_artifacts):
        artifact_dir, raw_tests = fake_artifacts
        k = 5
        result = run_select(artifact_dir=artifact_dir, k=k)

        all_names = {t["name"] for t in raw_tests}
        for selected in result.selected:
            assert selected.name in all_names

    def test_selection_json_file_created(self, fake_artifacts, tmp_path):
        artifact_dir, raw_tests = fake_artifacts
        output_file = tmp_path / "selection_output.json"
        run_select(artifact_dir=artifact_dir, k=5, output_file=output_file)
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert "selected" in data
        assert len(data["selected"]) == 5

    def test_diversity_metrics_computed(self, fake_artifacts):
        artifact_dir, raw_tests = fake_artifacts
        result = run_select(artifact_dir=artifact_dir, k=5)

        assert result.diversity_metrics.avg_pairwise_distance > 0
        assert result.diversity_metrics.suite_coverage > 0

    def test_different_k_values_produce_different_selections(self, fake_artifacts):
        artifact_dir, raw_tests = fake_artifacts
        result_3 = run_select(artifact_dir=artifact_dir, k=3)
        result_7 = run_select(artifact_dir=artifact_dir, k=7)

        assert len(result_3.selected) == 3
        assert len(result_7.selected) == 7
        assert result_3.k != result_7.k

    def test_fps_strategy_works(self, fake_artifacts):
        artifact_dir, raw_tests = fake_artifacts
        result = run_select(artifact_dir=artifact_dir, k=5, strategy="fps")
        assert result.strategy == "fps"
        assert len(result.selected) == 5

    def test_fps_multi_strategy_works(self, fake_artifacts):
        artifact_dir, raw_tests = fake_artifacts
        result = run_select(artifact_dir=artifact_dir, k=5, strategy="fps_multi")
        assert result.strategy == "fps_multi"
        assert len(result.selected) == 5

    def test_tag_filtering_include(self, fake_artifacts):
        artifact_dir, raw_tests = fake_artifacts
        # Filter to only "smoke" tests
        result = run_select(
            artifact_dir=artifact_dir,
            k=10,
            include_tags=["smoke"],
        )
        # All selected tests should have "smoke" tag
        manifest_data = json.loads(
            (artifact_dir / "test_manifest.json").read_text()
        )
        test_by_name = {t["name"]: t for t in manifest_data["tests"]}
        for selected in result.selected:
            entry = test_by_name[selected.name]
            tags_lower = [t.lower() for t in entry["tags"]]
            assert "smoke" in tags_lower

    def test_tag_filtering_exclude(self, fake_artifacts):
        artifact_dir, raw_tests = fake_artifacts
        result = run_select(
            artifact_dir=artifact_dir,
            k=10,
            exclude_tags=["negative"],
        )
        manifest_data = json.loads(
            (artifact_dir / "test_manifest.json").read_text()
        )
        test_by_name = {t["name"]: t for t in manifest_data["tests"]}
        for selected in result.selected:
            entry = test_by_name[selected.name]
            tags_lower = [t.lower() for t in entry["tags"]]
            assert "negative" not in tags_lower

    def test_k_clamped_to_available_tests(self, fake_artifacts):
        artifact_dir, raw_tests = fake_artifacts
        # Request more tests than available
        result = run_select(artifact_dir=artifact_dir, k=1000)
        assert len(result.selected) == len(raw_tests)

    def test_seed_determinism(self, fake_artifacts):
        artifact_dir, raw_tests = fake_artifacts
        result1 = run_select(artifact_dir=artifact_dir, k=5, seed=99)
        result2 = run_select(artifact_dir=artifact_dir, k=5, seed=99)
        names1 = [t.name for t in result1.selected]
        names2 = [t.name for t in result2.selected]
        assert names1 == names2
