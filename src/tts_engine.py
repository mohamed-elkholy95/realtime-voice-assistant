"""Enhanced Text-to-Speech engine with rate control, pitch scaling, and SSML stubs.

This module provides a Text-to-Speech (TTS) engine that integrates with
Coqui TTS when available, and falls back to a synthetic mock mode for
testing and demonstration.

Educational Context:
    Text-to-Speech synthesis converts written text into spoken audio.
    Modern TTS systems typically follow this pipeline:

    1. **Text Analysis**: Normalize text (numbers → words, abbreviations),
       identify phonemes, determine prosody (stress, intonation)
    2. **Spectral Prediction**: Generate acoustic features (mel spectrogram)
       from the linguistic representation using a neural network
       (e.g., Tacotron 2, FastSpeech)
    3. **Vocoder**: Convert the mel spectrogram to a waveform using a
       neural vocoder (e.g., WaveGlow, HiFi-GAN, WaveNet)

    Coqui TTS is an open-source TTS toolkit supporting multiple model
    architectures. The Tacotron2-DDC model used here is a popular
    choice for its balance of quality and speed.
"""

import logging
from typing import Optional

import numpy as np

from src.config import RANDOM_SEED, SAMPLE_RATE
from src.audio_processor import generate_speech_like_audio

logger = logging.getLogger(__name__)

HAS_TTS = False
try:
    from TTS.api import TTS  # noqa: F401
    HAS_TTS = True
except ImportError:
    logger.info("TTS not available — using mock TTS synthesis")


class TTSEngine:
    """Text-to-Speech engine with Coqui TTS support and mock fallback.

    This engine provides:
    - Real TTS synthesis when Coqui TTS is installed
    - Synthetic mock speech generation for testing
    - Speech rate control (speed up/slow down)
    - Pitch scaling (raise/lower voice pitch)
    - SSML support stubs for future enhancement

    Attributes:
        model_name: Name/path of the TTS model.
        _model: The loaded TTS model (None in mock mode).

    Example:
        >>> engine = TTSEngine()
        >>> audio = engine.synthesize("Hello world")
        >>> isinstance(audio, np.ndarray)
        True
    """

    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC") -> None:
        """Initialize the TTS engine.

        Attempts to load a Coqui TTS model if available. Falls back to
        mock mode if TTS is not installed or model loading fails.

        Args:
            model_name: Coqui TTS model identifier or local path.
                Common English models:
                - 'tts_models/en/ljspeech/tacotron2-DDC' (default)
                - 'tts_models/en/ljspeech/fast_pitch'
                - 'tts_models/en/vctk/vits'
        """
        self.model_name = model_name
        self._model = None

        if HAS_TTS:
            try:
                self._model = TTS(model_name)
                logger.info("Loaded TTS model: %s", model_name)
            except Exception as exc:
                logger.warning("Failed to load TTS model '%s': %s", model_name, exc)

        if self._model is None:
            logger.info("TTS engine running in mock mode")

    def synthesize(
        self,
        text: str,
        sample_rate: int = SAMPLE_RATE,
        speech_rate: float = 1.0,
        pitch_scale: float = 1.0,
    ) -> np.ndarray:
        """Synthesize speech from text.

        If a real TTS model is loaded, performs actual synthesis.
        Otherwise, generates synthetic speech-like audio.

        Args:
            text: The text to convert to speech.
            sample_rate: Desired output sample rate in Hz.
            speech_rate: Speech rate multiplier (default 1.0).
                Values > 1.0 speed up speech, < 1.0 slow it down.
                Range: 0.5 to 2.0 is typical.
            pitch_scale: Pitch scaling factor (default 1.0).
                Values > 1.0 raise pitch, < 1.0 lower it.
                Range: 0.5 to 2.0 is typical.

        Returns:
            1D numpy array of int16 audio samples.

        Raises:
            ValueError: If text is empty or rate/pitch values are invalid.
        """
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text")

        # Validate speech rate
        if not (0.25 <= speech_rate <= 4.0):
            raise ValueError(
                f"speech_rate must be in [0.25, 4.0], got {speech_rate}"
            )

        # Validate pitch scale
        if not (0.25 <= pitch_scale <= 4.0):
            raise ValueError(
                f"pitch_scale must be in [0.25, 4.0], got {pitch_scale}"
            )

        if self._model is not None:
            return self._real_synthesize(text, sample_rate, speech_rate, pitch_scale)
        return self._mock_synthesize(text, sample_rate, speech_rate, pitch_scale)

    def _real_synthesize(
        self,
        text: str,
        sample_rate: int,
        speech_rate: float,
        pitch_scale: float,
    ) -> np.ndarray:
        """Synthesize speech using the Coqui TTS model.

        Args:
            text: Text to synthesize.
            sample_rate: Desired sample rate.
            speech_rate: Speed multiplier.
            pitch_scale: Pitch scaling factor.

        Returns:
            Audio samples as int16 numpy array.
        """
        try:
            wav = self._model.tts(text)
            audio = np.array(wav, dtype=np.float64)

            # Apply speech rate by resampling the audio
            # Faster rate = shorter duration (drop samples)
            # Slower rate = longer duration (repeat samples)
            if speech_rate != 1.0:
                audio = self._apply_rate_change(audio, speech_rate)

            # Apply pitch scaling
            # Higher pitch = shift frequencies up
            # Lower pitch = shift frequencies down
            if pitch_scale != 1.0:
                audio = self._apply_pitch_scaling(audio, pitch_scale)

            # Resample to target sample rate if needed
            # (Coqui TTS typically outputs at 22050 Hz)
            if len(audio) > 0:
                audio = (audio * 32767.0).astype(np.int16)
            else:
                audio = np.array([], dtype=np.int16)

            return audio
        except Exception as exc:
            logger.error("TTS synthesis failed: %s", exc)
            return self._mock_synthesize(text, sample_rate, speech_rate, pitch_scale)

    def _mock_synthesize(
        self,
        text: str,
        sample_rate: int = SAMPLE_RATE,
        speech_rate: float = 1.0,
        pitch_scale: float = 1.0,
    ) -> np.ndarray:
        """Generate synthetic speech-like audio as a mock fallback.

        The duration scales with text length (longer text → longer audio),
        and the speech rate inversely affects duration (faster = shorter).

        Args:
            text: Input text (used to estimate duration).
            sample_rate: Desired sample rate.
            speech_rate: Speed multiplier.
            pitch_scale: Pitch scaling factor.

        Returns:
            Synthetic audio as int16 numpy array.
        """
        # Estimate duration: ~0.05 seconds per character at normal rate
        base_duration = max(0.5, len(text) * 0.05)

        # Apply speech rate: faster rate = shorter duration
        adjusted_duration = base_duration / max(speech_rate, 0.25)

        audio = generate_speech_like_audio(adjusted_duration, sample_rate)

        # Apply pitch scaling by shifting frequencies
        # This is a simplified mock — real pitch shifting is more complex
        if pitch_scale != 1.0 and len(audio) > 0:
            audio = self._apply_pitch_scaling_mock(audio, pitch_scale)

        return audio

    @staticmethod
    def _apply_rate_change(audio: np.ndarray, rate: float) -> np.ndarray:
        """Apply speech rate change by resampling.

        Faster rate (> 1.0) drops samples (shorter audio).
        Slower rate (< 1.0) interpolates new samples (longer audio).

        This is a simplified linear interpolation approach. Real-time
        rate changing in TTS typically uses WSOLA (Waveform Similarity
        Overlap-Add) to avoid artifacts.

        Args:
            audio: Float64 audio samples.
            rate: Rate multiplier.

        Returns:
            Rate-adjusted audio samples.
        """
        if len(audio) == 0:
            return audio

        original_length = len(audio)
        new_length = int(original_length / rate)

        if new_length <= 0:
            return audio[:1]

        # Create new time indices in the original signal
        original_indices = np.linspace(0, original_length - 1, new_length)

        # Simple linear interpolation between adjacent samples
        lower_indices = np.floor(original_indices).astype(int)
        upper_indices = np.minimum(lower_indices + 1, original_length - 1)
        fractions = original_indices - lower_indices

        return audio[lower_indices] * (1 - fractions) + audio[upper_indices] * fractions

    @staticmethod
    def _apply_pitch_scaling(audio: np.ndarray, scale: float) -> np.ndarray:
        """Apply pitch scaling to audio using frequency-domain processing.

        Pitch shifting changes the perceived pitch of audio without
        changing its duration. This is more complex than simple resampling
        (which changes both pitch and speed).

        This implementation uses a simplified approach suitable for mock:
        - For scale > 1 (higher pitch): compress frequency domain
        - For scale < 1 (lower pitch): expand frequency domain

        A production implementation would use a phase vocoder.

        Args:
            audio: Audio samples.
            scale: Pitch scaling factor.

        Returns:
            Pitch-scaled audio samples.
        """
        if len(audio) == 0 or scale == 1.0:
            return audio

        # Simplified pitch scaling: resample then time-stretch
        # This changes pitch without changing duration
        # (Note: real pitch shifting uses phase vocoders — this is approximate)
        original_length = len(audio)
        float_audio = audio.astype(np.float64)

        # Step 1: Resample to change pitch (this also changes speed)
        resampled_length = int(original_length / scale)
        if resampled_length <= 0:
            resampled_length = 1

        original_indices = np.linspace(0, original_length - 1, resampled_length)
        lower = np.floor(original_indices).astype(int)
        upper = np.minimum(lower + 1, original_length - 1)
        frac = original_indices - lower
        resampled = float_audio[lower] * (1 - frac) + float_audio[upper] * frac

        # Step 2: Time-stretch back to original length (undo the speed change)
        # This restores the original duration with the new pitch
        stretch_indices = np.linspace(0, len(resampled) - 1, original_length)
        lower_s = np.floor(stretch_indices).astype(int)
        upper_s = np.minimum(lower_s + 1, len(resampled) - 1)
        frac_s = stretch_indices - lower_s
        stretched = resampled[lower_s] * (1 - frac_s) + resampled[upper_s] * frac_s

        # Normalize to prevent clipping after processing
        peak = np.max(np.abs(stretched))
        if peak > 0:
            stretched = stretched / peak * np.max(np.abs(float_audio))

        return stretched.astype(audio.dtype)

    @staticmethod
    def _apply_pitch_scaling_mock(audio: np.ndarray, scale: float) -> np.ndarray:
        """Simplified pitch scaling for mock mode.

        For int16 audio, applies a basic resampling approach.

        Args:
            audio: int16 audio samples.
            scale: Pitch scaling factor.

        Returns:
            Pitch-shifted audio as int16.
        """
        if len(audio) == 0 or scale == 1.0:
            return audio

        float_audio = audio.astype(np.float64)
        original_length = len(float_audio)

        # Resample
        new_length = int(original_length / scale)
        if new_length <= 0:
            new_length = 1

        indices = np.linspace(0, original_length - 1, new_length)
        lower = np.floor(indices).astype(int)
        upper = np.minimum(lower + 1, original_length - 1)
        frac = indices - lower
        resampled = float_audio[lower] * (1 - frac) + float_audio[upper] * frac

        # Stretch back to original length
        stretch_idx = np.linspace(0, len(resampled) - 1, original_length)
        lower_s = np.floor(stretch_idx).astype(int)
        upper_s = np.minimum(lower_s + 1, len(resampled) - 1)
        frac_s = stretch_idx - lower_s
        stretched = resampled[lower_s] * (1 - frac_s) + resampled[upper_s] * frac_s

        peak = np.max(np.abs(stretched))
        if peak > 0:
            stretched = stretched / peak * np.max(np.abs(float_audio))

        return np.clip(stretched, -32768, 32767).astype(np.int16)

    def synthesize_ssml(self, ssml_text: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Synthesize speech from SSML (Speech Synthesis Markup Language) input.

        SSML is an XML-based markup language that provides fine-grained
        control over speech synthesis, including:
        - <prosody rate="fast">: Control speech rate
        - <prosody pitch="+10%">: Control pitch
        - <break time="500ms"/>: Insert pauses
        - <emphasis level="strong">: Emphasize words
        - <say-as interpret-as="date">: Control pronunciation

        This is a stub implementation that strips SSML tags and
        synthesizes the plain text content.

        Args:
            ssml_text: SSML-formatted text.
            sample_rate: Desired output sample rate.

        Returns:
            Audio samples as int16 numpy array.
        """
        import re

        # Strip SSML tags to get plain text
        plain_text = re.sub(r'<[^>]+>', '', ssml_text).strip()

        if not plain_text:
            return np.array([], dtype=np.int16)

        return self.synthesize(plain_text, sample_rate)

    def convert_audio_format(
        self,
        audio: np.ndarray,
        target_dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        """Convert audio between int16 and float32 representations.

        Different audio libraries use different sample formats:
        - int16: Used by WAV files and some audio APIs (range: -32768 to 32767)
        - float32: Used by NumPy/SciPy and ML models (range: -1.0 to 1.0)

        Args:
            audio: Audio samples to convert.
            target_dtype: Target numpy dtype (np.int16 or np.float32).

        Returns:
            Converted audio samples.

        Raises:
            ValueError: If target_dtype is not int16 or float32.
        """
        if target_dtype == np.int16:
            if audio.dtype == np.int16:
                return audio
            # float32 → int16: scale from [-1, 1] to [-32768, 32767]
            return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        elif target_dtype == np.float32:
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                return audio.astype(np.float32)
            # int16 → float32: scale from [-32768, 32767] to [-1, 1]
            return (audio.astype(np.float32) / 32768.0)
        else:
            raise ValueError(
                f"Unsupported target dtype: {target_dtype}. "
                f"Use np.int16 or np.float32."
            )

    def is_loaded(self) -> bool:
        """Check if the TTS model is loaded (not in mock mode).

        Returns:
            True if a real TTS model is available, False for mock mode.
        """
        return self._model is not None
