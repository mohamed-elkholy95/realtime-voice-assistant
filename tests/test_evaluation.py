import pytest
from src.evaluation import compute_wer, generate_report

class TestWER:
    def test_perfect(self): assert compute_wer("hello world", "hello world") == 0.0
    def test_substitution(self): assert compute_wer("hello world", "hello there") > 0
    def test_empty_ref(self): assert compute_wer("", "hello") == 1.0

class TestReport:
    def test_output(self):
        assert "# Voice Assistant" in generate_report({"wer": 0.15})
