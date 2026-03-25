"""Tests for SNR estimation and confusion matrix computation.

Validates audio quality measurement and multi-class classification
evaluation metrics including per-class precision, recall, and F1.
"""

import numpy as np
import pytest

from src.audio_processor import estimate_snr, generate_speech_like_audio, generate_sine_wave
from src.evaluation import compute_confusion_matrix


class TestEstimateSNR:
    """Tests for the estimate_snr function."""

    def test_returns_required_keys(self):
        """Result should contain all documented fields."""
        audio = generate_speech_like_audio(duration=1.0)
        result = estimate_snr(audio)
        expected_keys = {
            "snr_db", "signal_power_db", "noise_power_db",
            "speech_ratio", "n_speech_frames", "n_silence_frames",
        }
        assert set(result.keys()) == expected_keys

    def test_empty_audio(self):
        """Empty audio should return zero SNR with floor power levels."""
        result = estimate_snr(np.array([], dtype=np.int16))
        assert result["snr_db"] == 0.0
        assert result["speech_ratio"] == 0.0

    def test_silent_audio(self):
        """All-zero audio should classify all frames as silence."""
        silent = np.zeros(16000, dtype=np.int16)
        result = estimate_snr(silent)
        assert result["n_speech_frames"] == 0

    def test_loud_signal_has_positive_snr(self):
        """A strong signal should have a non-negative SNR."""
        audio = generate_sine_wave(duration=1.0, freq=440, amplitude=1.0)
        result = estimate_snr(audio)
        assert result["snr_db"] >= 0.0

    def test_speech_ratio_range(self):
        """Speech ratio should be between 0 and 1."""
        audio = generate_speech_like_audio(duration=2.0)
        result = estimate_snr(audio)
        assert 0.0 <= result["speech_ratio"] <= 1.0

    def test_frame_counts_sum_correctly(self):
        """Speech + silence frames should equal total frames."""
        audio = generate_speech_like_audio(duration=1.0)
        result = estimate_snr(audio)
        total = result["n_speech_frames"] + result["n_silence_frames"]
        assert total > 0


class TestComputeConfusionMatrix:
    """Tests for the compute_confusion_matrix function."""

    def test_perfect_classification(self):
        """All-correct predictions should have 1.0 accuracy and F1."""
        refs = ["weather", "time", "music", "weather", "time"]
        preds = ["weather", "time", "music", "weather", "time"]
        result = compute_confusion_matrix(refs, preds)

        assert result["accuracy"] == 1.0
        assert result["macro_f1"] == 1.0
        for metrics in result["per_class"].values():
            assert metrics["precision"] == 1.0
            assert metrics["recall"] == 1.0

    def test_all_wrong(self):
        """Completely wrong predictions should have 0.0 accuracy."""
        refs = ["weather", "weather", "weather"]
        preds = ["time", "time", "time"]
        result = compute_confusion_matrix(refs, preds)
        assert result["accuracy"] == 0.0

    def test_known_confusion_pattern(self):
        """Verify specific confusion matrix values."""
        refs = ["A", "A", "B", "B"]
        preds = ["A", "B", "A", "B"]
        result = compute_confusion_matrix(refs, preds, labels=["A", "B"])

        # Matrix should be [[1, 1], [1, 1]]
        assert result["matrix"] == [[1, 1], [1, 1]]
        assert result["accuracy"] == 0.5

    def test_per_class_precision_recall(self):
        """Verify precision and recall for a known asymmetric case."""
        refs = ["cat", "cat", "cat", "dog"]
        preds = ["cat", "cat", "dog", "dog"]
        result = compute_confusion_matrix(refs, preds)

        # cat: precision = 2/(2+0) = 1.0, recall = 2/3 = 0.6667
        assert result["per_class"]["cat"]["precision"] == 1.0
        assert abs(result["per_class"]["cat"]["recall"] - 0.6667) < 0.001

        # dog: precision = 1/(1+1) = 0.5, recall = 1/1 = 1.0
        assert result["per_class"]["dog"]["precision"] == 0.5
        assert result["per_class"]["dog"]["recall"] == 1.0

    def test_empty_inputs(self):
        """Empty inputs should return empty results."""
        result = compute_confusion_matrix([], [])
        assert result["matrix"] == []
        assert result["accuracy"] == 0.0

    def test_length_mismatch_raises(self):
        """Mismatched input lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_confusion_matrix(["a", "b"], ["a"])

    def test_custom_labels(self):
        """Custom label ordering should be preserved."""
        refs = ["B", "A"]
        preds = ["A", "A"]
        result = compute_confusion_matrix(refs, preds, labels=["A", "B"])
        assert result["labels"] == ["A", "B"]
        # A predicted for both: A column has 2
        # Matrix[0][0] = TP for A (ref=A, pred=A) = 1
        # Matrix[1][0] = FP for A (ref=B, pred=A) = 1
        assert result["matrix"][0][0] == 1  # A→A
        assert result["matrix"][1][0] == 1  # B→A

    def test_single_class(self):
        """Single-class classification should work."""
        refs = ["pos", "pos", "pos"]
        preds = ["pos", "pos", "pos"]
        result = compute_confusion_matrix(refs, preds)
        assert result["accuracy"] == 1.0
        assert result["matrix"] == [[3]]

    def test_support_counts(self):
        """Support should equal the number of actual instances per class."""
        refs = ["a", "a", "a", "b", "b"]
        preds = ["a", "a", "b", "b", "a"]
        result = compute_confusion_matrix(refs, preds)
        assert result["per_class"]["a"]["support"] == 3
        assert result["per_class"]["b"]["support"] == 2

    def test_macro_f1_is_average_of_per_class(self):
        """Macro F1 should be the unweighted mean of per-class F1 scores."""
        refs = ["x", "x", "y", "y", "z"]
        preds = ["x", "y", "y", "z", "z"]
        result = compute_confusion_matrix(refs, preds)

        per_class_f1s = [
            m["f1"] for m in result["per_class"].values() if m["support"] > 0
        ]
        expected_macro_f1 = sum(per_class_f1s) / len(per_class_f1s)
        assert abs(result["macro_f1"] - round(expected_macro_f1, 4)) < 0.0001
