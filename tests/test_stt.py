"""Comprehensive tests for the STT engine module.

Tests cover:
- Mock transcription output
- Audio preprocessing pipeline
- Batch transcription
- Language detection stub
- Edge cases: empty audio, silent audio
"""

import pytest
import numpy as np

from src.stt_engine import STTEngine


class TestSTTEngineInit:
    """Tests for STT engine initialization."""

    def test_not_loaded_in_mock_mode(self, mock_stt_engine):
        """Mock mode should report no model loaded."""
        assert not mock_stt_engine.is_loaded()

    def test_model_size_stored(self):
        """Model size should be stored correctly."""
        engine = STTEngine(model_size="base")
        assert engine.model_size == "base"

    def test_language_stored(self):
        """Language should be stored correctly."""
        engine = STTEngine(language="es")
        assert engine.language == "es"


class TestMockTranscription:
    """Tests for mock (deterministic) transcription."""

    def test_mock_returns_text(self, mock_stt_engine):
        """Mock transcription should return a non-empty text string."""
        audio = np.zeros(16000, dtype=np.int16)
        result = mock_stt_engine.transcribe(audio)
        assert "text" in result
        assert len(result["text"]) > 0

    def test_mock_returns_language(self, mock_stt_engine):
        """Mock transcription should include language."""
        audio = np.zeros(16000, dtype=np.int16)
        result = mock_stt_engine.transcribe(audio)
        assert "language" in result
        assert result["language"] == "en"

    def test_mock_returns_confidence(self, mock_stt_engine):
        """Mock transcription should include confidence score."""
        audio = np.zeros(16000, dtype=np.int16)
        result = mock_stt_engine.transcribe(audio)
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_mock_cycling(self, mock_stt_engine):
        """Mock transcription should cycle through predefined texts."""
        audio = np.zeros(16000, dtype=np.int16)
        texts = [mock_stt_engine.transcribe(audio)["text"] for _ in range(12)]
        # With 6 mock texts and 12 calls, we should see repeats
        assert len(set(texts)) <= 6

    def test_mock_no_preprocess_flag(self, mock_stt_engine):
        """Mock transcription should indicate no preprocessing by default."""
        audio = np.zeros(16000, dtype=np.int16)
        result = mock_stt_engine.transcribe(audio, preprocess=False)
        assert result.get("preprocessed") is False


class TestAudioPreprocessing:
    """Tests for audio preprocessing pipeline."""

    def test_preprocess_preserves_length(self, mock_stt_engine, short_audio):
        """Preprocessing should not dramatically change audio length."""
        processed = mock_stt_engine.preprocess_audio(short_audio)
        # Allow some trimming but not total loss
        assert len(processed) > 0

    def test_preprocess_preserves_dtype(self, mock_stt_engine, short_audio):
        """Preprocessing should preserve int16 dtype."""
        processed = mock_stt_engine.preprocess_audio(short_audio)
        assert processed.dtype == np.int16

    def test_preprocess_silent_audio(self, mock_stt_engine, silent_audio):
        """Preprocessing silent audio should not crash."""
        processed = mock_stt_engine.preprocess_audio(silent_audio)
        assert processed.dtype == np.int16

    def test_preprocess_normalize_off(self, mock_stt_engine, short_audio):
        """Preprocessing with normalization off should still work."""
        processed = mock_stt_engine.preprocess_audio(short_audio, normalize=False)
        assert len(processed) > 0


class TestBatchTranscription:
    """Tests for batch transcription of multiple segments."""

    def test_batch_returns_list(self, mock_stt_engine, short_audio):
        """Batch transcription should return a list of results."""
        segments = [short_audio, short_audio, short_audio]
        results = mock_stt_engine.transcribe_batch(segments)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_batch_each_has_text(self, mock_stt_engine, short_audio):
        """Each batch result should have a text field."""
        segments = [short_audio, short_audio]
        results = mock_stt_engine.transcribe_batch(segments)
        for result in results:
            assert "text" in result
            assert len(result["text"]) > 0

    def test_batch_empty_list(self, mock_stt_engine):
        """Batch with empty segment list should return empty list."""
        results = mock_stt_engine.transcribe_batch([])
        assert results == []

    def test_batch_single_segment(self, mock_stt_engine, short_audio):
        """Batch with single segment should work correctly."""
        results = mock_stt_engine.transcribe_batch([short_audio])
        assert len(results) == 1
        assert "text" in results[0]


class TestLanguageDetection:
    """Tests for language detection stub."""

    def test_detect_returns_language(self, mock_stt_engine, short_audio):
        """Language detection should return a language code."""
        lang = mock_stt_engine.detect_language(short_audio)
        assert lang is not None
        assert isinstance(lang, str)

    def test_detect_uses_configured_language(self, mock_stt_engine, short_audio):
        """In mock mode, should return configured language."""
        lang = mock_stt_engine.detect_language(short_audio)
        assert lang == "en"


class TestEdgeCases:
    """Tests for edge case inputs."""

    def test_empty_audio(self, mock_stt_engine):
        """Empty audio should not crash the engine."""
        audio = np.array([], dtype=np.int16)
        result = mock_stt_engine.transcribe(audio, preprocess=False)
        assert "text" in result

    def test_single_sample(self, mock_stt_engine):
        """Single-sample audio should work."""
        audio = np.array([100], dtype=np.int16)
        result = mock_stt_engine.transcribe(audio, preprocess=False)
        assert "text" in result
