"""Integration tests for Stage 1: parsing and text building (no ML model required)."""
from __future__ import annotations

import pytest

from TestSelection.parsing.keyword_resolver import KeywordTreeResolver
from TestSelection.parsing.suite_collector import RobotApiAdapter
from TestSelection.parsing.text_builder import TextRepresentationBuilder
from TestSelection.shared.config import TextBuilderConfig
from TestSelection.shared.types import Tag


@pytest.mark.integration
class TestStage1Vectorize:
    """End-to-end tests for parsing + text building (without actual embedding)."""

    def test_parse_sample_suite_finds_all_tests(self, sample_suite_path):
        adapter = RobotApiAdapter()
        raw_tests, kw_map = adapter.parse_suite(sample_suite_path)
        assert len(raw_tests) == 14

    def test_parse_builds_keyword_map(self, sample_suite_path):
        adapter = RobotApiAdapter()
        raw_tests, kw_map = adapter.parse_suite(sample_suite_path)
        # sample.robot defines user keywords like "Login As User"
        assert len(kw_map) > 0
        # Check a known keyword is in the map (normalized)
        assert "login_as_user" in kw_map

    def test_text_representations_contain_test_names(self, sample_suite_path):
        adapter = RobotApiAdapter()
        raw_tests, kw_map = adapter.parse_suite(sample_suite_path)

        resolver = KeywordTreeResolver(kw_map)
        builder = TextRepresentationBuilder(resolver, TextBuilderConfig())

        for test_dict in raw_tests:
            tags = frozenset(Tag(value=t) for t in test_dict.get("tags", []))
            text_rep = builder.build(
                test_name=test_dict["name"],
                tags=tags,
                body_items=test_dict.get("body", []),
            )
            assert test_dict["name"] in text_rep.text

    def test_text_representations_filter_noise(self, sample_suite_path):
        adapter = RobotApiAdapter()
        raw_tests, kw_map = adapter.parse_suite(sample_suite_path)

        resolver = KeywordTreeResolver(kw_map)
        builder = TextRepresentationBuilder(resolver, TextBuilderConfig())

        for test_dict in raw_tests:
            tags = frozenset(Tag(value=t) for t in test_dict.get("tags", []))
            text_rep = builder.build(
                test_name=test_dict["name"],
                tags=tags,
                body_items=test_dict.get("body", []),
            )
            # Variable placeholders like ${...} should be filtered
            assert "${" not in text_rep.text

    def test_all_tests_have_source_and_suite_name(self, sample_suite_path):
        adapter = RobotApiAdapter()
        raw_tests, kw_map = adapter.parse_suite(sample_suite_path)

        for test_dict in raw_tests:
            assert test_dict["source"], f"Test {test_dict['name']} missing source"
            name = test_dict["name"]
            assert test_dict["suite_name"], f"Test {name} missing suite_name"

    def test_tags_are_extracted_correctly(self, sample_suite_path):
        adapter = RobotApiAdapter()
        raw_tests, kw_map = adapter.parse_suite(sample_suite_path)

        # Find the "Login With Valid Credentials" test and check its tags
        login_test = next(
            t for t in raw_tests if t["name"] == "Login With Valid Credentials"
        )
        assert "smoke" in login_test["tags"]
        assert "authentication" in login_test["tags"]

    def test_text_builder_with_resolve_depth(self, sample_suite_path):
        adapter = RobotApiAdapter()
        raw_tests, kw_map = adapter.parse_suite(sample_suite_path)

        resolver = KeywordTreeResolver(kw_map)
        config_depth0 = TextBuilderConfig(resolve_depth=0)
        config_depth1 = TextBuilderConfig(resolve_depth=1)
        builder0 = TextRepresentationBuilder(resolver, config_depth0)
        builder1 = TextRepresentationBuilder(resolver, config_depth1)

        # Pick a test that uses nested keywords
        test_dict = raw_tests[0]
        tags = frozenset(Tag(value=t) for t in test_dict.get("tags", []))

        text0 = builder0.build(
            test_name=test_dict["name"], tags=tags, body_items=test_dict.get("body", [])
        )
        text1 = builder1.build(
            test_name=test_dict["name"], tags=tags, body_items=test_dict.get("body", [])
        )
        # Both should be valid non-empty text representations
        assert len(text0.text) > 0
        assert len(text1.text) > 0
