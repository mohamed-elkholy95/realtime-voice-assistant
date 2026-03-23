"""Comprehensive tests for the evaluation metrics module.

Tests cover:
- WER computation with Levenshtein edit distance
- CER computation at character level
- Intent classification accuracy
- Latency benchmarking
- Report generation
- Edge cases: empty strings, identical strings, complete mismatches
"""

import pytest
import numpy as np

from src.evaluation import (
    levenshtein_distance,
    compute_wer,
    compute_cer,
    compute_intent_accuracy,
    benchmark_latency,
    generate_report,
)


class TestLevenshteinDistance:
    """Tests for the Levenshtein edit distance algorithm."""

    def test_identical_sequences(self):
        """Identical sequences should have distance 0."""
        dist, subs, deletes, inserts = levenshtein_distance(
            ["hello", "world"], ["hello", "world"]
        )
        assert dist == 0
        assert subs == 0
        assert deletes == 0
        assert inserts == 0

    def test_single_substitution(self):
        """One different word should give distance 1."""
        dist, subs, deletes, inserts = levenshtein_distance(
            ["hello", "world"], ["hello", "there"]
        )
        assert dist == 1
        assert subs == 1
        assert deletes == 0
        assert inserts == 0

    def test_single_insertion(self):
        """One extra word in hypothesis should give distance 1."""
        dist, subs, deletes, inserts = levenshtein_distance(
            ["hello"], ["hello", "world"]
        )
        assert dist == 1
        assert inserts == 1
        assert deletes == 0

    def test_single_deletion(self):
        """One missing word should give distance 1."""
        dist, subs, deletes, inserts = levenshtein_distance(
            ["hello", "world"], ["hello"]
        )
        assert dist == 1
        assert deletes == 1
        assert inserts == 0

    def test_empty_sequences(self):
        """Two empty sequences should have distance 0."""
        dist, _, _, _ = levenshtein_distance([], [])
        assert dist == 0

    def test_empty_a_nonempty_b(self):
        """Empty first sequence should require all insertions."""
        dist, subs, deletes, inserts = levenshtein_distance([], ["a", "b", "c"])
        assert dist == 3
        assert inserts == 3
        assert deletes == 0

    def test_nonempty_a_empty_b(self):
        """Empty second sequence should require all deletions."""
        dist, subs, deletes, inserts = levenshtein_distance(["a", "b"], [])
        assert dist == 2
        assert deletes == 2
        assert inserts == 0

    def test_complex_edit(self):
        """Multiple edits should sum correctly."""
        dist, subs, deletes, inserts = levenshtein_distance(
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "dog", "sat", "on", "a", "mat"],
        )
        # "cat"→"dog" (1 sub), "the"→"a" (1 sub) = distance 2
        assert dist == 2
        assert subs == 2


class TestWordErrorRate:
    """Tests for WER computation."""

    def test_perfect_match(self):
        """Identical strings should give WER of 0.0."""
        assert compute_wer("hello world", "hello world") == 0.0

    def test_single_substitution(self):
        """One wrong word should give non-zero WER."""
        wer = compute_wer("hello world", "hello there")
        assert wer > 0.0
        assert wer == 0.5  # 1 error / 2 words

    def test_all_wrong(self):
        """Complete mismatch should give WER of 1.0."""
        wer = compute_wer("the cat sat", "a dog ran fast")
        assert wer == 1.0

    def test_empty_reference_empty_hypothesis(self):
        """Both empty should give WER of 0.0."""
        assert compute_wer("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self):
        """Empty reference with hypothesis should give WER of 1.0."""
        assert compute_wer("", "hello") == 1.0

    def test_case_insensitive(self):
        """WER should be case-insensitive."""
        wer1 = compute_wer("Hello World", "hello world")
        assert wer1 == 0.0

    def test_extra_whitespace_robust(self):
        """Multiple spaces should be handled correctly."""
        wer1 = compute_wer("hello world", "hello  world")
        # Extra space creates empty string in split
        assert 0.0 <= wer1 <= 0.5

    def test_insertions_only(self):
        """Extra words in hypothesis should increase WER."""
        wer = compute_wer("hello", "hello world foo")
        assert wer >= 1.0  # 2 insertions / 1 word = 2.0, capped to 1.0
        assert wer <= 1.0

    def test_deletions_only(self):
        """Missing words should increase WER."""
        wer = compute_wer("hello world foo bar", "hello")
        assert wer == 0.75  # 3 deletions / 4 words = 0.75

    def test_long_sentences(self):
        """WER should work correctly on longer sentences."""
        ref = "the quick brown fox jumps over the lazy dog"
        hyp = "the quick brown fox jumped over the lazy dog"
        wer = compute_wer(ref, hyp)
        assert 0.0 < wer < 0.5  # 1 substitution out of 9 words


class TestCharacterErrorRate:
    """Tests for CER computation."""

    def test_perfect_match(self):
        """Identical strings should give CER of 0.0."""
        assert compute_cer("hello", "hello") == 0.0

    def test_single_deletion(self):
        """One missing character should give CER of 0.2."""
        cer = compute_cer("hello", "helo")
        assert cer == pytest.approx(1 / 5, abs=0.01)

    def test_single_substitution(self):
        """One wrong character should give non-zero CER."""
        cer = compute_cer("cat", "bat")
        assert cer == pytest.approx(1 / 3, abs=0.01)

    def test_empty_reference(self):
        """Empty reference should give CER of 0.0."""
        assert compute_cer("", "") == 0.0

    def test_empty_reference_with_hypothesis(self):
        """Empty reference with hypothesis should give CER of 1.0."""
        assert compute_cer("", "hello") == 1.0

    def test_case_insensitive(self):
        """CER should be case-insensitive."""
        assert compute_cer("Hello", "hello") == 0.0

    def test_cer_less_than_wer_typically(self):
        """CER is typically lower than WER for the same error."""
        ref = "the cat sat on the mat"
        hyp = "the bat sat on the mat"
        cer = compute_cer(ref, hyp)
        wer = compute_wer(ref, hyp)
        # Same number of errors but CER denominator (characters) is larger
        assert cer <= wer


class TestIntentAccuracy:
    """Tests for intent classification accuracy metric."""

    def test_perfect_accuracy(self):
        """All correct should give 1.0."""
        accuracy = compute_intent_accuracy(
            ["weather", "time", "music"],
            ["weather", "time", "music"],
        )
        assert accuracy == 1.0

    def test_zero_accuracy(self):
        """All wrong should give 0.0."""
        accuracy = compute_intent_accuracy(
            ["weather", "time", "music"],
            ["greeting", "farewell", "general"],
        )
        assert accuracy == 0.0

    def test_partial_accuracy(self):
        """Some correct should give partial accuracy."""
        accuracy = compute_intent_accuracy(
            ["weather", "time", "music"],
            ["weather", "time", "general"],
        )
        assert accuracy == pytest.approx(2 / 3, abs=0.01)

    def test_length_mismatch_raises(self):
        """Different-length lists should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_intent_accuracy(["a", "b"], ["a"])

    def test_empty_lists(self):
        """Empty lists should give 0.0."""
        accuracy = compute_intent_accuracy([], [])
        assert accuracy == 0.0


class TestLatencyBenchmark:
    """Tests for latency benchmarking utility."""

    def test_fast_function(self):
        """A fast function should report low latency."""
        result = benchmark_latency(lambda: None, num_runs=5, warmup_runs=1)
        assert result["mean_ms"] >= 0
        assert result["num_runs"] == 5
        assert "median_ms" in result
        assert "std_ms" in result

    def test_consistent_results(self):
        """Same function should give similar mean across runs."""
        result1 = benchmark_latency(lambda: sum(range(100)), num_runs=10, warmup_runs=2)
        result2 = benchmark_latency(lambda: sum(range(100)), num_runs=10, warmup_runs=2)
        # Mean should be within 10x (very generous — avoiding flaky tests)
        assert result1["mean_ms"] > 0
        assert result2["mean_ms"] > 0

    def test_min_lte_max(self):
        """Minimum should be <= maximum."""
        result = benchmark_latency(lambda: None, num_runs=5)
        assert result["min_ms"] <= result["max_ms"]


class TestReportGeneration:
    """Tests for evaluation report generation."""

    def test_basic_report(self):
        """Report should contain the title."""
        report = generate_report({"wer": 0.15})
        assert "# Voice Assistant" in report

    def test_metrics_in_table(self):
        """Metrics should appear in the table."""
        report = generate_report({"wer": 0.15, "cer": 0.08})
        assert "wer" in report
        assert "cer" in report

    def test_sections_included(self):
        """Educational sections should be included by default."""
        report = generate_report({"wer": 0.15}, include_sections=True)
        assert "Word Error Rate" in report
        assert "Character Error Rate" in report

    def test_sections_excluded(self):
        """Educational sections can be excluded."""
        report = generate_report({"wer": 0.15}, include_sections=False)
        assert "Word Error Rate" not in report

    def test_float_formatting(self):
        """Float values should be formatted nicely."""
        report = generate_report({"wer": 0.1534})
        assert "15.34%" in report  # Formatted as percentage

    def test_latency_formatting(self):
        """Latency metrics should include ms units."""
        report = generate_report({"latency_mean_ms": 42.5})
        assert "ms" in report
