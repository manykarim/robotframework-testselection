"""Benchmarks for text building and keyword resolution."""
from __future__ import annotations

import pytest

from TestSelection.parsing.keyword_resolver import KeywordTreeResolver
from TestSelection.parsing.text_builder import TextRepresentationBuilder
from TestSelection.shared.config import TextBuilderConfig
from TestSelection.shared.types import Tag


@pytest.mark.benchmark
class TestTextBuilderBenchmark:
    """Benchmark text builder on synthetic test dicts."""

    def test_build_100_tests(self, benchmark, synthetic_test_dicts):
        resolver = KeywordTreeResolver({})
        builder = TextRepresentationBuilder(resolver, TextBuilderConfig())

        def build_all():
            results = []
            for test_dict in synthetic_test_dicts:
                tags = frozenset(Tag(value=t) for t in test_dict.get("tags", []))
                rep = builder.build(
                    test_name=test_dict["name"],
                    tags=tags,
                    body_items=test_dict.get("body", []),
                )
                results.append(rep)
            return results

        results = benchmark(build_all)
        assert len(results) == 100


@pytest.mark.benchmark
class TestKeywordResolverBenchmark:
    """Benchmark keyword resolver on nested keyword trees."""

    def test_resolve_depth_1(self, benchmark, synthetic_keyword_map):
        resolver = KeywordTreeResolver(synthetic_keyword_map)

        def resolve_all():
            results = []
            for i in range(5):
                tree = resolver.resolve(f"Top Keyword {i}", (), max_depth=1)
                results.append(tree)
            return results

        results = benchmark(resolve_all)
        assert len(results) == 5

    def test_resolve_depth_3(self, benchmark, synthetic_keyword_map):
        resolver = KeywordTreeResolver(synthetic_keyword_map)

        def resolve_all():
            results = []
            for i in range(5):
                tree = resolver.resolve(f"Top Keyword {i}", (), max_depth=3)
                results.append(tree)
            return results

        results = benchmark(resolve_all)
        assert len(results) == 5
        # Depth 3 should resolve children
        for tree in results:
            assert tree.keyword_name.startswith("Top Keyword")
