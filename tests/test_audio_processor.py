import pytest
import numpy as np
from src.audio_processor import generate_sine_wave, generate_speech_like_audio, compute_mfcc, compute_spectrogram, apply_preemphasis, normalize_audio

class TestSineWave:
    def test_shape(self):
        wave = generate_sine_wave(duration=1.0, sample_rate=16000)
        assert len(wave) == 16000
    def test_dtype(self):
        assert generate_sine_wave().dtype == np.int16

class TestSpeechLike:
    def test_shape(self):
        audio = generate_speech_like_audio(duration=2.0)
        assert len(audio) == 32000

class TestMFCC:
    def test_shape(self):
        mfcc = compute_mfcc(np.zeros(16000))
        assert mfcc.shape[0] == 13

class TestSpectrogram:
    def test_output(self):
        stats = compute_spectrogram(np.zeros(16000))
        assert "duration" in stats
        assert stats["duration"] == 1.0

class TestPreemphasis:
    def test_length(self):
        audio = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        result = apply_preemphasis(audio)
        assert len(result) == 5

class TestNormalize:
    def test_peak(self):
        audio = np.array([100, -100, 50], dtype=np.int16)
        result = normalize_audio(audio, target_peak=0.5)
        assert np.max(np.abs(result)) <= 32767
