"""Shared pytest fixtures for all test modules.

This module provides reusable fixtures for testing the voice assistant
components, including:
- Sample audio signals with known properties
- Mock engine instances for isolated testing
- Common test parameters
"""

import pytest
import numpy as np


@pytest.fixture
def sample_rate() -> int:
    """Provide the standard test sample rate of 16000 Hz."""
    return 16000


@pytest.fixture
def short_audio(sample_rate: int) -> np.ndarray:
    """Provide a short (1 second) sine wave at 440 Hz for basic tests.

    This is a pure tone — ideal for testing signal processing functions
    that should handle simple, predictable inputs.

    Returns:
        1D int16 array of 16000 samples.
    """
    from src.audio_processor import generate_sine_wave
    return generate_sine_wave(duration=1.0, freq=440, sample_rate=sample_rate)


@pytest.fixture
def long_audio(sample_rate: int) -> np.ndarray:
    """Provide a longer (2 second) speech-like audio signal.

    Multi-harmonic signal that mimics some properties of speech:
    multiple frequency components, amplitude variation, noise.

    Returns:
        1D int16 array of 32000 samples.
    """
    from src.audio_processor import generate_speech_like_audio
    return generate_speech_like_audio(duration=2.0, sample_rate=sample_rate)


@pytest.fixture
def silent_audio(sample_rate: int) -> np.ndarray:
    """Provide a completely silent audio signal (all zeros).

    Useful for testing edge cases where input has no energy.

    Returns:
        1D int16 array of 16000 zero-valued samples.
    """
    return np.zeros(sample_rate, dtype=np.int16)


@pytest.fixture
def empty_audio() -> np.ndarray:
    """Provide an empty audio signal (zero length).

    Tests edge case handling for functions that should gracefully
    handle empty inputs.

    Returns:
        Empty 1D int16 array.
    """
    return np.array([], dtype=np.int16)


@pytest.fixture
def clipped_audio(sample_rate: int) -> np.ndarray:
    """Provide audio with clipping (maximum amplitude values).

    Clipping occurs when the signal exceeds the representable range,
    causing distortion. This fixture tests normalization and
    dynamic range handling.

    Returns:
        1D int16 array with values at the clipping boundaries.
    """
    audio = np.zeros(sample_rate, dtype=np.int16)
    # Set some samples to max/min int16 values (clipping)
    audio[:sample_rate // 4] = 32767
    audio[sample_rate // 4:sample_rate // 2] = -32768
    audio[sample_rate // 2:] = 32767
    return audio


@pytest.fixture
def multi_freq_audio(sample_rate: int) -> np.ndarray:
    """Provide audio with multiple frequency components.

    Returns:
        1D int16 array combining 440 Hz, 880 Hz, and 1320 Hz sine waves.
    """
    duration = 1.0
    num_samples = int(sample_rate * duration)
    time_array = np.linspace(0, duration, num_samples, endpoint=False)
    signal = (
        np.sin(2 * np.pi * 440 * time_array) +
        0.5 * np.sin(2 * np.pi * 880 * time_array) +
        0.3 * np.sin(2 * np.pi * 1320 * time_array)
    )
    # Normalize to prevent clipping when summing
    signal = signal / np.max(np.abs(signal)) * 0.9
    return (signal * 32767).astype(np.int16)


@pytest.fixture
def mock_stt_engine():
    """Provide a mock STT engine for isolated testing."""
    from src.stt_engine import STTEngine
    return STTEngine()


@pytest.fixture
def mock_tts_engine():
    """Provide a mock TTS engine for isolated testing."""
    from src.tts_engine import TTSEngine
    return TTSEngine()


@pytest.fixture
def intent_pipeline():
    """Provide a fresh intent classification pipeline for testing."""
    from src.intent_classifier import IntentClassifierPipeline
    return IntentClassifierPipeline()


@pytest.fixture
def voice_assistant():
    """Provide a fresh voice assistant instance for testing."""
    from src.voice_assistant import VoiceAssistant
    return VoiceAssistant()
