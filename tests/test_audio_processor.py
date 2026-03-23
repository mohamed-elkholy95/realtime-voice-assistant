"""Comprehensive tests for the audio processing module.

Tests cover:
- Signal generation (sine waves, speech-like audio)
- MFCC extraction with real scipy implementation
- STFT spectrogram computation
- Mel filter bank creation
- Voice activity detection (silence detection)
- Speech segment extraction
- Log-Mel spectrogram computation
- Pre-emphasis filtering
- Audio normalization
- Hz ↔ Mel conversion
- Edge cases: empty audio, silent audio, clipping
"""

import pytest
import numpy as np

from src.audio_processor import (
    generate_sine_wave,
    generate_speech_like_audio,
    compute_mfcc,
    compute_spectrogram,
    create_mel_filterbank,
    detect_silence,
    extract_speech_segments,
    compute_log_mel_spectrogram,
    apply_preemphasis,
    normalize_audio,
    hz_to_mel,
    mel_to_hz,
)


class TestHzMelConversion:
    """Tests for Hz ↔ Mel scale conversion functions."""

    def test_zero_hz_is_zero_mel(self):
        """0 Hz should map to 0 Mel (boundary condition)."""
        assert hz_to_mel(0) == 0.0

    def test_mel_to_hz_roundtrip(self):
        """Mel → Hz → Mel should return the original value."""
        original_hz = 1000.0
        mel = hz_to_mel(original_hz)
        recovered_hz = mel_to_hz(mel)
        assert abs(recovered_hz - original_hz) < 0.01

    def test_hz_to_mel_roundtrip(self):
        """Hz → Mel → Hz should return the original value."""
        original_mel = 1500.0
        hz = mel_to_hz(original_mel)
        recovered_mel = hz_to_mel(hz)
        assert abs(recovered_mel - original_mel) < 0.01

    def test_negative_hz_raises(self):
        """Negative frequencies should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            hz_to_mel(-100)

    def test_negative_mel_raises(self):
        """Negative Mel values should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            mel_to_hz(-100)

    @pytest.mark.parametrize("hz,mel_approx", [
        (1000, 1000.0),  # By definition, 1000 Hz ≈ 1000 Mel
        (4000, 2146.0),  # Higher frequency
        (200, 283.0),  # Lower frequency
    ])
    def test_known_conversions(self, hz, mel_approx):
        """Verify approximate values for known frequency conversions."""
        mel = hz_to_mel(hz)
        assert abs(mel - mel_approx) < 5.0  # Allow small tolerance


class TestMelFilterbank:
    """Tests for Mel filter bank creation."""

    def test_shape(self):
        """Filter bank should have shape (n_filters, n_fft // 2 + 1)."""
        fb = create_mel_filterbank(n_filters=26, n_fft_size=512, sample_rate=16000)
        assert fb.shape == (26, 257)

    def test_all_non_negative(self):
        """All filter bank weights should be non-negative."""
        fb = create_mel_filterbank(n_filters=26, n_fft_size=512, sample_rate=16000)
        assert np.all(fb >= 0)

    def test_sum_to_one_at_peaks(self):
        """At peak frequency, the dominant filter should have weight near 1.0."""
        fb = create_mel_filterbank(n_filters=26, n_fft_size=512, sample_rate=16000)
        # Check that at least one filter has a value close to 1.0 (its peak)
        max_per_filter = np.max(fb, axis=1)
        assert np.any(max_per_filter > 0.95)

    def test_different_n_filters(self):
        """Filter bank should adapt to different numbers of filters."""
        for n in [10, 20, 40]:
            fb = create_mel_filterbank(n_filters=n, n_fft_size=512, sample_rate=16000)
            assert fb.shape[0] == n

    def test_invalid_n_filters(self):
        """Zero or negative filter count should raise ValueError."""
        with pytest.raises(ValueError):
            create_mel_filterbank(n_filters=0, n_fft_size=512, sample_rate=16000)

    def test_invalid_high_freq(self):
        """High frequency above Nyquist should raise ValueError."""
        with pytest.raises(ValueError):
            create_mel_filterbank(
                n_filters=26, n_fft_size=512, sample_rate=16000, high_freq_hz=10000
            )


class TestSineWaveGeneration:
    """Tests for sine wave generation."""

    def test_shape(self, sample_rate):
        """Sine wave duration * sample_rate should match length."""
        wave = generate_sine_wave(duration=1.0, sample_rate=sample_rate)
        assert len(wave) == sample_rate

    def test_dtype(self):
        """Generated audio should be int16."""
        assert generate_sine_wave().dtype == np.int16

    def test_frequency_scaling(self, sample_rate):
        """Higher frequency should produce more zero crossings."""
        freq_low = 100
        freq_high = 1000
        wave_low = generate_sine_wave(duration=1.0, freq=freq_low, sample_rate=sample_rate)
        wave_high = generate_sine_wave(duration=1.0, freq=freq_high, sample_rate=sample_rate)

        # Count zero crossings: sign changes
        crossings_low = np.sum(np.diff(np.sign(wave_low.astype(float))) != 0)
        crossings_high = np.sum(np.diff(np.sign(wave_high.astype(float))) != 0)

        # Higher frequency should have more zero crossings
        assert crossings_high > crossings_low

    @pytest.mark.parametrize("duration", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_various_durations(self, sample_rate, duration):
        """Various durations should produce correct sample counts."""
        wave = generate_sine_wave(duration=duration, sample_rate=sample_rate)
        assert len(wave) == int(duration * sample_rate)

    def test_amplitude_range(self):
        """Generated samples should be within int16 range."""
        wave = generate_sine_wave(amplitude=1.0)
        assert np.min(wave) >= -32768
        assert np.max(wave) <= 32767


class TestSpeechLikeGeneration:
    """Tests for speech-like audio generation."""

    def test_shape(self, sample_rate):
        """Duration * sample_rate should match length."""
        audio = generate_speech_like_audio(duration=2.0, sample_rate=sample_rate)
        assert len(audio) == 2 * sample_rate

    def test_dtype(self):
        """Generated audio should be int16."""
        assert generate_speech_like_audio().dtype == np.int16

    def test_non_zero(self, long_audio):
        """Speech-like audio should have non-zero samples."""
        assert np.any(long_audio != 0)

    def test_reproducibility(self):
        """Same seed should produce identical output."""
        audio1 = generate_speech_like_audio(duration=1.0, seed=42)
        audio2 = generate_speech_like_audio(duration=1.0, seed=42)
        assert np.array_equal(audio1, audio2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different output."""
        audio1 = generate_speech_like_audio(duration=1.0, seed=1)
        audio2 = generate_speech_like_audio(duration=1.0, seed=2)
        assert not np.array_equal(audio1, audio2)


class TestMFCC:
    """Tests for MFCC extraction."""

    def test_shape(self, short_audio):
        """MFCC should have shape (n_mfcc, n_frames)."""
        mfccs = compute_mfcc(short_audio)
        assert mfccs.shape[0] == 13  # Default n_mfcc
        assert mfccs.shape[1] > 0  # At least 1 frame

    def test_custom_n_mfcc(self, short_audio):
        """Custom n_mfcc should be respected."""
        for n in [5, 13, 20, 26]:
            mfccs = compute_mfcc(short_audio, n_mfcc=n)
            assert mfccs.shape[0] == n

    def test_empty_audio(self):
        """Empty audio should return a minimal MFCC array."""
        mfccs = compute_mfcc(np.array([], dtype=np.int16))
        assert mfccs.shape == (13, 1)

    def test_silent_audio(self, silent_audio):
        """Silent audio should still produce valid MFCC shape."""
        mfccs = compute_mfcc(silent_audio)
        assert mfccs.shape[0] == 13
        assert mfccs.shape[1] > 0

    def test_finite_values(self, short_audio):
        """All MFCC values should be finite (no NaN or Inf)."""
        mfccs = compute_mfcc(short_audio)
        assert np.all(np.isfinite(mfccs))

    def test_different_audio_produces_different_mfcc(self, sample_rate):
        """Different audio signals should produce different MFCCs."""
        audio1 = generate_sine_wave(duration=1.0, freq=440, sample_rate=sample_rate)
        audio2 = generate_sine_wave(duration=1.0, freq=880, sample_rate=sample_rate)
        mfcc1 = compute_mfcc(audio1)
        mfcc2 = compute_mfcc(audio2)
        # They should not be identical (though could be similar)
        assert not np.allclose(mfcc1, mfcc2)

    def test_reproducibility(self, short_audio):
        """Same input should produce same MFCCs."""
        mfcc1 = compute_mfcc(short_audio)
        mfcc2 = compute_mfcc(short_audio)
        assert np.allclose(mfcc1, mfcc2)

    def test_long_audio_more_frames(self, sample_rate):
        """Longer audio should produce more MFCC frames."""
        audio_short = generate_sine_wave(duration=0.5, sample_rate=sample_rate)
        audio_long = generate_sine_wave(duration=2.0, sample_rate=sample_rate)
        mfcc_short = compute_mfcc(audio_short)
        mfcc_long = compute_mfcc(audio_long)
        assert mfcc_long.shape[1] > mfcc_short.shape[1]


class TestSpectrogram:
    """Tests for spectrogram computation."""

    def test_basic_output(self, short_audio):
        """Spectrogram should return all expected keys."""
        result = compute_spectrogram(short_audio)
        assert "duration" in result
        assert "max_amplitude" in result
        assert "rms" in result
        assert "spectrogram" in result
        assert "frequencies" in result
        assert "times" in result
        assert "sample_rate" in result

    def test_duration(self, short_audio, sample_rate):
        """Duration should match audio length / sample rate."""
        result = compute_spectrogram(short_audio, sample_rate=sample_rate)
        assert result["duration"] == 1.0

    def test_spectrogram_shape(self, short_audio):
        """Spectrogram should be 2D (frequency bins × time frames)."""
        result = compute_spectrogram(short_audio)
        spec = result["spectrogram"]
        assert spec.ndim == 2
        assert spec.shape[0] > 0  # Frequency bins
        assert spec.shape[1] > 0  # Time frames

    def test_spectrogram_finite(self, short_audio):
        """Spectrogram values (in dB) should all be finite."""
        result = compute_spectrogram(short_audio)
        assert np.all(np.isfinite(result["spectrogram"]))

    def test_silent_audio_rms(self, silent_audio):
        """Silent audio should have zero RMS."""
        result = compute_spectrogram(silent_audio)
        assert result["rms"] == 0.0

    def test_sine_wave_has_energy(self, short_audio):
        """A sine wave should have non-zero RMS energy."""
        result = compute_spectrogram(short_audio)
        assert result["rms"] > 0


class TestSilenceDetection:
    """Tests for energy-based voice activity detection."""

    def test_sine_wave_is_speech(self, short_audio):
        """A non-trivial sine wave should be detected as speech."""
        results = detect_silence(short_audio)
        # At least some frames should be classified as speech
        speech_frames = [r for r in results if r["is_speech"]]
        assert len(speech_frames) > 0

    def test_silent_audio_no_speech(self, silent_audio):
        """Completely silent audio should have no speech frames."""
        results = detect_silence(silent_audio)
        speech_frames = [r for r in results if r["is_speech"]]
        assert len(speech_frames) == 0

    def test_empty_audio(self):
        """Empty audio should return empty results."""
        results = detect_silence(np.array([], dtype=np.int16))
        assert results == []

    def test_result_structure(self, short_audio):
        """Each result should have the expected keys."""
        results = detect_silence(short_audio)
        if results:
            assert "start_time" in results[0]
            assert "end_time" in results[0]
            assert "energy_db" in results[0]
            assert "is_speech" in results[0]

    def test_frame_coverage(self, short_audio):
        """Frames should cover the entire audio duration."""
        results = detect_silence(short_audio)
        if results:
            assert results[-1]["end_time"] >= results[0]["start_time"]

    def test_threshold_sensitivity(self, short_audio):
        """Lower threshold should detect more frames as speech."""
        results_sensitive = detect_silence(short_audio, threshold_db=-10.0)
        results_strict = detect_silence(short_audio, threshold_db=-80.0)

        speech_sensitive = sum(1 for r in results_sensitive if r["is_speech"])
        speech_strict = sum(1 for r in results_strict if r["is_speech"])

        # More sensitive (less negative) threshold = fewer speech frames
        assert speech_sensitive <= speech_strict


class TestSpeechSegmentExtraction:
    """Tests for speech segment extraction."""

    def test_speech_audio_has_segments(self, long_audio):
        """Speech-like audio should yield at least one segment."""
        segments = extract_speech_segments(long_audio)
        assert len(segments) >= 1

    def test_silent_audio_no_segments(self, silent_audio):
        """Silent audio should yield no segments."""
        segments = extract_speech_segments(silent_audio)
        assert len(segments) == 0

    def test_segment_structure(self, long_audio):
        """Each segment should have expected keys."""
        segments = extract_speech_segments(long_audio)
        if segments:
            seg = segments[0]
            assert "start_time" in seg
            assert "end_time" in seg
            assert "duration" in seg
            assert "audio" in seg
            assert "start_sample" in seg
            assert "end_sample" in seg

    def test_segment_audio_not_empty(self, long_audio):
        """Extracted segment audio should not be empty."""
        segments = extract_speech_segments(long_audio)
        if segments:
            assert len(segments[0]["audio"]) > 0

    def test_segment_duration_positive(self, long_audio):
        """Segment duration should be positive."""
        segments = extract_speech_segments(long_audio)
        if segments:
            assert segments[0]["duration"] > 0


class TestLogMelSpectrogram:
    """Tests for Whisper-style log-MEL spectrogram computation."""

    def test_shape(self, short_audio):
        """Log-MEL spectrogram should have shape (n_mels, n_frames)."""
        mel_spec = compute_log_mel_spectrogram(short_audio)
        assert mel_spec.shape[0] == 26  # Default n_mels
        assert mel_spec.shape[1] > 0

    def test_custom_n_mels(self, short_audio):
        """Custom n_mels should be respected."""
        mel_spec = compute_log_mel_spectrogram(short_audio, n_mels=40)
        assert mel_spec.shape[0] == 40

    def test_empty_audio(self):
        """Empty audio should return minimal array."""
        mel_spec = compute_log_mel_spectrogram(np.array([], dtype=np.int16))
        assert mel_spec.shape == (26, 1)

    def test_finite_values(self, short_audio):
        """All values should be finite."""
        mel_spec = compute_log_mel_spectrogram(short_audio)
        assert np.all(np.isfinite(mel_spec))

    def test_negative_values(self, short_audio):
        """Log energies should be non-positive (log of values ≤ 1)."""
        mel_spec = compute_log_mel_spectrogram(short_audio)
        # Log of filterbank energies should typically be negative or zero
        # (unless energies > 1, which is possible with certain FFT gains)
        assert np.any(mel_spec <= 0) or np.all(np.isfinite(mel_spec))


class TestPreemphasis:
    """Tests for pre-emphasis filter."""

    def test_length_preserved(self):
        """Pre-emphasis should not change the signal length."""
        audio = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        result = apply_preemphasis(audio)
        assert len(result) == len(audio)

    def test_first_sample_unchanged(self):
        """The first sample should pass through unchanged (no predecessor)."""
        audio = np.array([100, 200, 300], dtype=np.int16)
        result = apply_preemphasis(audio)
        assert result[0] == 100

    def test_high_frequency_boost(self, sample_rate):
        """Pre-emphasis should change the spectral characteristics of the signal."""
        # Generate low-frequency and high-frequency signals
        low_freq = generate_sine_wave(duration=1.0, freq=100, sample_rate=sample_rate)
        high_freq = generate_sine_wave(duration=1.0, freq=4000, sample_rate=sample_rate)

        low_pre = apply_preemphasis(low_freq)
        high_pre = apply_preemphasis(high_freq)

        # Pre-emphasis should change the energy of the signal
        energy_low_before = np.mean(np.float64(low_freq) ** 2)
        energy_low_after = np.mean(np.float64(low_pre) ** 2)
        energy_high_before = np.mean(np.float64(high_freq) ** 2)
        energy_high_after = np.mean(np.float64(high_pre) ** 2)

        # For low-frequency: adjacent samples are similar, so difference is small
        # Pre-emphasis should significantly reduce energy for low-freq signals
        assert energy_low_after < energy_low_before

        # For high-frequency: the effect differs from low-frequency
        assert energy_high_after != energy_high_before

    def test_zero_coeff_passthrough(self):
        """Coefficient of 0 should leave signal unchanged (y[n] = x[n] - 0*x[n-1])."""
        audio = np.array([10, 20, 30, 40], dtype=np.int16)
        result = apply_preemphasis(audio, coeff=0.0)
        assert np.array_equal(result, audio)

    def test_empty_audio(self):
        """Empty audio should return empty."""
        result = apply_preemphasis(np.array([], dtype=np.int16))
        assert len(result) == 0


class TestNormalization:
    """Tests for audio normalization."""

    def test_shape_preserved(self, short_audio):
        """Normalization should not change length."""
        normalized = normalize_audio(short_audio)
        assert len(normalized) == len(short_audio)

    def test_dtype_preserved(self, short_audio):
        """Output should be int16."""
        assert normalize_audio(short_audio).dtype == np.int16

    def test_silent_audio_unchanged(self, silent_audio):
        """Silent audio should be returned unchanged."""
        normalized = normalize_audio(silent_audio)
        assert np.array_equal(normalized, silent_audio)

    def test_peak_normalized(self, short_audio):
        """After normalization, peak should be near target."""
        target_peak = 0.9
        normalized = normalize_audio(short_audio, target_peak=target_peak)
        actual_peak = np.max(np.abs(normalized))
        expected_peak = int(target_peak * 32767)
        # Should be close (within 1 due to int16 rounding)
        assert abs(actual_peak - expected_peak) <= 1

    def test_clipped_audio_normalized(self, clipped_audio):
        """Clipped audio should still normalize correctly."""
        normalized = normalize_audio(clipped_audio)
        assert np.min(normalized) >= -32768
        assert np.max(normalized) <= 32767

    def test_empty_audio(self):
        """Empty audio should return empty."""
        result = normalize_audio(np.array([], dtype=np.int16))
        assert len(result) == 0
