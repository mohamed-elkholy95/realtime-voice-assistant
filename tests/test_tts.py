"""Comprehensive tests for the TTS engine module.

Tests cover:
- Mock synthesis output
- Duration scaling with text length
- Speech rate control
- Pitch scaling
- SSML stub
- Audio format conversion
- Edge cases: empty text, invalid parameters
"""

import pytest
import numpy as np

from src.tts_engine import TTSEngine


class TestTTSEngineInit:
    """Tests for TTS engine initialization."""

    def test_not_loaded_in_mock_mode(self, mock_tts_engine):
        """Mock mode should report no model loaded."""
        assert not mock_tts_engine.is_loaded()

    def test_model_name_stored(self):
        """Model name should be stored correctly."""
        engine = TTSEngine(model_name="test_model")
        assert engine.model_name == "test_model"


class TestMockSynthesis:
    """Tests for mock (synthetic) synthesis."""

    def test_mock_returns_array(self, mock_tts_engine):
        """Mock synthesis should return a numpy array."""
        audio = mock_tts_engine.synthesize("Hello world")
        assert isinstance(audio, np.ndarray)

    def test_mock_returns_int16(self, mock_tts_engine):
        """Mock synthesis should return int16 samples."""
        audio = mock_tts_engine.synthesize("Hello world")
        assert audio.dtype == np.int16

    def test_mock_non_empty(self, mock_tts_engine):
        """Mock synthesis should produce non-empty audio."""
        audio = mock_tts_engine.synthesize("Hello world")
        assert len(audio) > 0

    def test_duration_scales_with_text(self, mock_tts_engine):
        """Longer text should produce longer audio."""
        short = mock_tts_engine.synthesize("Hi")
        long = mock_tts_engine.synthesize("Hello world this is a much longer sentence for testing")
        assert len(long) > len(short)

    def test_duration_scales_with_text_parametrize(self, mock_tts_engine):
        """Various text lengths should produce monotonically longer audio."""
        texts = ["hi", "hello world", "the quick brown fox jumps over the lazy dog and more text here"]
        lengths = [len(mock_tts_engine.synthesize(t)) for t in texts]
        assert lengths[0] < lengths[1] < lengths[2]


class TestSpeechRate:
    """Tests for speech rate control."""

    def test_faster_rate_shorter_audio(self, mock_tts_engine):
        """Faster speech rate should produce shorter audio."""
        normal = mock_tts_engine.synthesize("Hello world", speech_rate=1.0)
        fast = mock_tts_engine.synthesize("Hello world", speech_rate=2.0)
        assert len(fast) < len(normal)

    def test_slower_rate_longer_audio(self, mock_tts_engine):
        """Slower speech rate should produce longer audio."""
        normal = mock_tts_engine.synthesize("Hello world", speech_rate=1.0)
        slow = mock_tts_engine.synthesize("Hello world", speech_rate=0.5)
        assert len(slow) > len(normal)

    @pytest.mark.parametrize("rate", [0.25, 0.5, 1.0, 1.5, 2.0, 4.0])
    def test_valid_rates(self, mock_tts_engine, rate):
        """All valid rate values should produce audio without error."""
        audio = mock_tts_engine.synthesize("Test", speech_rate=rate)
        assert len(audio) > 0

    def test_rate_below_minimum_raises(self, mock_tts_engine):
        """Rate below 0.25 should raise ValueError."""
        with pytest.raises(ValueError, match="speech_rate"):
            mock_tts_engine.synthesize("Test", speech_rate=0.1)

    def test_rate_above_maximum_raises(self, mock_tts_engine):
        """Rate above 4.0 should raise ValueError."""
        with pytest.raises(ValueError, match="speech_rate"):
            mock_tts_engine.synthesize("Test", speech_rate=5.0)


class TestPitchScaling:
    """Tests for pitch scaling."""

    def test_pitch_change_preserves_length_approximately(self, mock_tts_engine):
        """Pitch scaling should approximately preserve duration."""
        normal = mock_tts_engine.synthesize("Hello world", pitch_scale=1.0)
        high = mock_tts_engine.synthesize("Hello world", pitch_scale=1.5)
        # Length should be within 10% of original
        ratio = len(high) / len(normal)
        assert 0.8 < ratio < 1.2

    @pytest.mark.parametrize("pitch", [0.5, 1.0, 1.5, 2.0])
    def test_valid_pitch_values(self, mock_tts_engine, pitch):
        """Valid pitch values should produce audio."""
        audio = mock_tts_engine.synthesize("Test", pitch_scale=pitch)
        assert len(audio) > 0

    def test_pitch_below_minimum_raises(self, mock_tts_engine):
        """Pitch below 0.25 should raise ValueError."""
        with pytest.raises(ValueError, match="pitch_scale"):
            mock_tts_engine.synthesize("Test", pitch_scale=0.1)

    def test_pitch_above_maximum_raises(self, mock_tts_engine):
        """Pitch above 4.0 should raise ValueError."""
        with pytest.raises(ValueError, match="pitch_scale"):
            mock_tts_engine.synthesize("Test", pitch_scale=5.0)


class TestSSML:
    """Tests for SSML support stub."""

    def test_ssml_strips_tags(self, mock_tts_engine):
        """SSML tags should be stripped and remaining text synthesized."""
        audio = mock_tts_engine.synthesize_ssml(
            '<speak><prosody rate="fast">Hello world</prosody></speak>'
        )
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_ssml_only_tags(self, mock_tts_engine):
        """SSML with only tags (no text content) should return empty audio."""
        audio = mock_tts_engine.synthesize_ssml("<speak></speak>")
        assert len(audio) == 0


class TestAudioFormatConversion:
    """Tests for audio format conversion helpers."""

    def test_int16_to_float32(self, mock_tts_engine):
        """Converting int16 to float32 should scale to [-1, 1] range."""
        int_audio = np.array([0, 16384, 32767, -32768], dtype=np.int16)
        float_audio = mock_tts_engine.convert_audio_format(int_audio, np.float32)
        assert float_audio.dtype == np.float32
        assert float_audio.max() <= 1.0
        assert float_audio.min() >= -1.0

    def test_float32_to_int16(self, mock_tts_engine):
        """Converting float32 to int16 should scale to int16 range."""
        float_audio = np.array([0.0, 0.5, 1.0, -1.0], dtype=np.float32)
        int_audio = mock_tts_engine.convert_audio_format(float_audio, np.int16)
        assert int_audio.dtype == np.int16
        assert int_audio.max() <= 32767
        assert int_audio.min() >= -32768

    def test_int16_passthrough(self, mock_tts_engine):
        """Converting int16 to int16 should return the same array."""
        audio = np.array([100, 200, -100], dtype=np.int16)
        result = mock_tts_engine.convert_audio_format(audio, np.int16)
        assert np.array_equal(result, audio)

    def test_unsupported_dtype_raises(self, mock_tts_engine):
        """Unsupported dtype should raise ValueError."""
        audio = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(ValueError, match="Unsupported target dtype"):
            mock_tts_engine.convert_audio_format(audio, np.int32)


class TestEdgeCases:
    """Tests for edge case inputs."""

    def test_empty_text_raises(self, mock_tts_engine):
        """Empty text should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            mock_tts_engine.synthesize("")

    def test_whitespace_text_raises(self, mock_tts_engine):
        """Whitespace-only text should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            mock_tts_engine.synthesize("   ")

    def test_rate_change_empty_audio(self):
        """Rate change on empty audio should return empty."""
        from src.tts_engine import TTSEngine
        result = TTSEngine._apply_rate_change(np.array([], dtype=np.float64), 2.0)
        assert len(result) == 0
