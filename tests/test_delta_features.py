"""Tests for delta and delta-delta MFCC feature computation.

Validates the regression-based delta computation, edge handling,
and the full 39-dimensional feature extraction pipeline.
"""

import numpy as np
import pytest

from src.audio_processor import (
    compute_delta_features,
    compute_mfcc_with_deltas,
    generate_speech_like_audio,
    generate_sine_wave,
)


class TestComputeDeltaFeatures:
    """Tests for the compute_delta_features function."""

    def test_constant_features_produce_zero_deltas(self):
        """Constant features should have zero temporal change."""
        # All frames have the same feature values
        features = np.ones((5, 20), dtype=np.float64) * 3.0
        deltas = compute_delta_features(features, width=2)
        assert deltas.shape == features.shape
        np.testing.assert_allclose(deltas, 0.0, atol=1e-10)

    def test_linear_ramp_produces_constant_deltas(self):
        """A linearly increasing feature should have constant delta."""
        # Feature that increases by 1.0 per frame
        n_frames = 20
        features = np.arange(n_frames, dtype=np.float64).reshape(1, -1)
        deltas = compute_delta_features(features, width=2)

        # For a linear ramp with slope 1, delta should be ~1.0
        # (edge frames may differ due to padding)
        interior = deltas[0, 3:-3]  # Skip edge-affected frames
        np.testing.assert_allclose(interior, 1.0, atol=0.01)

    def test_output_shape_matches_input(self):
        """Delta output should have the same shape as input."""
        features = np.random.default_rng(42).standard_normal((13, 50))
        for width in [1, 2, 3]:
            deltas = compute_delta_features(features, width=width)
            assert deltas.shape == features.shape, f"Shape mismatch for width={width}"

    def test_width_1_vs_width_2(self):
        """Wider windows should produce smoother deltas."""
        rng = np.random.default_rng(42)
        noisy_features = np.cumsum(rng.standard_normal((3, 100)), axis=1)

        delta_w1 = compute_delta_features(noisy_features, width=1)
        delta_w2 = compute_delta_features(noisy_features, width=2)

        # Wider window should produce lower variance (smoother) deltas
        assert np.std(delta_w2) < np.std(delta_w1)

    def test_empty_frames_return_zeros(self):
        """Empty feature matrix should return zeros."""
        features = np.zeros((5, 0), dtype=np.float64)
        deltas = compute_delta_features(features, width=2)
        assert deltas.shape == (5, 0)

    def test_single_frame(self):
        """Single-frame features should produce zero deltas (no change)."""
        features = np.array([[5.0], [3.0], [1.0]])
        deltas = compute_delta_features(features, width=2)
        np.testing.assert_allclose(deltas, 0.0, atol=1e-10)

    def test_invalid_width_raises(self):
        """Width < 1 should raise ValueError."""
        features = np.ones((3, 10))
        with pytest.raises(ValueError, match="width must be >= 1"):
            compute_delta_features(features, width=0)

    def test_1d_input_raises(self):
        """1D input should raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D"):
            compute_delta_features(np.array([1, 2, 3]), width=2)

    def test_known_regression_values(self):
        """Verify delta computation against hand-calculated values.

        For features = [1, 3, 2, 5, 4] with width=1:
        delta[t] = (f[t+1] - f[t-1]) / (2 * 1²) = (f[t+1] - f[t-1]) / 2

        At t=1: (2 - 1) / 2 = 0.5
        At t=2: (5 - 3) / 2 = 1.0
        At t=3: (4 - 2) / 2 = 1.0
        """
        features = np.array([[1.0, 3.0, 2.0, 5.0, 4.0]])
        deltas = compute_delta_features(features, width=1)
        # Interior frames (indices 1, 2, 3)
        assert abs(deltas[0, 1] - 0.5) < 1e-10
        assert abs(deltas[0, 2] - 1.0) < 1e-10
        assert abs(deltas[0, 3] - 1.0) < 1e-10


class TestComputeMFCCWithDeltas:
    """Tests for the compute_mfcc_with_deltas function."""

    def test_full_39d_feature_vector(self):
        """Default settings should produce 39-dimensional features."""
        audio = generate_speech_like_audio(duration=1.0)
        features = compute_mfcc_with_deltas(audio)
        assert features.shape[0] == 39  # 13 static + 13 delta + 13 delta-delta

    def test_static_only(self):
        """With include_delta=False, should produce 13 features."""
        audio = generate_speech_like_audio(duration=1.0)
        features = compute_mfcc_with_deltas(audio, include_delta=False)
        assert features.shape[0] == 13

    def test_static_plus_delta_only(self):
        """With include_delta_delta=False, should produce 26 features."""
        audio = generate_speech_like_audio(duration=1.0)
        features = compute_mfcc_with_deltas(
            audio, include_delta=True, include_delta_delta=False
        )
        assert features.shape[0] == 26

    def test_frame_count_matches_static_mfcc(self):
        """Number of frames should match plain compute_mfcc output."""
        from src.audio_processor import compute_mfcc

        audio = generate_speech_like_audio(duration=1.0)
        static = compute_mfcc(audio)
        full = compute_mfcc_with_deltas(audio)
        assert static.shape[1] == full.shape[1]

    def test_sine_wave_input(self):
        """Should work on a pure sine wave (not just speech-like audio)."""
        audio = generate_sine_wave(duration=0.5, freq=440)
        features = compute_mfcc_with_deltas(audio)
        assert features.shape[0] == 39
        assert features.shape[1] > 0

    def test_custom_n_mfcc(self):
        """Custom n_mfcc should scale the feature dimensions."""
        audio = generate_speech_like_audio(duration=1.0)
        features = compute_mfcc_with_deltas(audio, n_mfcc=20)
        assert features.shape[0] == 60  # 20 * 3
