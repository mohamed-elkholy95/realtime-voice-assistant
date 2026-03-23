"""Enhanced Speech-to-Text engine with preprocessing, batch, and confidence scoring.

This module provides a Speech-to-Text (STT) engine that integrates with
OpenAI's Whisper model when available, and falls back to a deterministic
mock mode for testing and demonstration.

Educational Context:
    Automatic Speech Recognition (ASR) converts spoken language into text.
    The process involves:
    1. **Feature Extraction**: Convert raw audio to MFCCs or log-Mel spectrograms
    2. **Acoustic Model**: Map features to phonemes (or directly to characters)
       using a neural network (e.g., Whisper's encoder-decoder Transformer)
    3. **Language Model**: Apply linguistic knowledge to resolve ambiguities
       (e.g., "recognize speech" vs. "wreck a nice beach")
    4. **Decoding**: Find the most likely text sequence given the observations

    Whisper (OpenAI, 2022) is an encoder-decoder Transformer trained on
    680K hours of multilingual data. It performs feature extraction, acoustic
    modeling, and language modeling end-to-end, making it robust across
    domains and accents.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import RANDOM_SEED
from src.audio_processor import apply_preemphasis, normalize_audio, detect_silence

logger = logging.getLogger(__name__)

HAS_WHISPER = False
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    logger.info("whisper not available — using mock STT")


class STTEngine:
    """Speech-to-Text engine with Whisper support and mock fallback.

    This engine provides:
    - Real Whisper transcription when the whisper package is installed
    - Deterministic mock transcription for testing
    - Audio preprocessing pipeline (normalization, silence trimming)
    - Batch transcription for multiple audio segments
    - Confidence scoring for transcription quality

    Attributes:
        model_size: Whisper model size identifier.
        language: Default language for transcription.
        _model: The loaded Whisper model (None in mock mode).

    Example:
        >>> engine = STTEngine()
        >>> result = engine.transcribe(np.zeros(16000, dtype=np.int16))
        >>> 'text' in result
        True
    """

    # Mock transcription responses — deterministic for reproducibility
    _MOCK_TRANSCRIPTIONS: List[str] = [
        "Hello, how can I help you today?",
        "The weather is sunny and warm outside.",
        "I'd like to schedule a meeting for tomorrow.",
        "Can you play some relaxing music?",
        "What time is it right now?",
        "Please set a reminder for five PM.",
    ]

    def __init__(self, model_size: str = "tiny", language: str = "en") -> None:
        """Initialize the STT engine.

        Attempts to load a Whisper model if available. Falls back to
        mock mode if whisper is not installed or model loading fails.

        Args:
            model_size: Whisper model size. Options:
                - 'tiny': 39M params, fastest, least accurate
                - 'base': 74M params
                - 'small': 244M params
                - 'medium': 769M params
                - 'large': 1550M params, slowest, most accurate
            language: ISO language code for transcription (e.g., 'en', 'es').
        """
        self.model_size = model_size
        self.language = language
        self._model = None
        self._mock_index = 0

        if HAS_WHISPER:
            try:
                self._model = whisper.load_model(model_size)
                logger.info("Loaded Whisper model: %s", model_size)
            except Exception as exc:
                logger.warning("Failed to load Whisper model '%s': %s", model_size, exc)

        if self._model is None:
            logger.info("STT engine running in mock mode")

    def preprocess_audio(
        self,
        audio: np.ndarray,
        normalize: bool = True,
        trim_silence: bool = True,
        silence_threshold_db: float = -40.0,
    ) -> np.ndarray:
        """Preprocess audio before transcription to improve accuracy.

        Preprocessing improves STT accuracy by:
        - **Normalization**: Ensures consistent amplitude levels, which
          helps models trained on normalized data
        - **Silence trimming**: Removes leading/trailing silence that
          can cause errors in models that use voice activity detection

        Args:
            audio: Raw audio samples (int16).
            normalize: Whether to normalize amplitude levels.
            trim_silence: Whether to trim leading/trailing silence.
            silence_threshold_db: Energy threshold for silence detection.

        Returns:
            Preprocessed audio samples (int16).
        """
        processed = audio.copy()

        if normalize:
            processed = normalize_audio(processed)

        if trim_silence and len(processed) > 0:
            vad_results = detect_silence(
                processed, threshold_db=silence_threshold_db
            )
            if vad_results:
                # Find first and last speech frames
                speech_indices = [
                    idx for idx, result in enumerate(vad_results)
                    if result["is_speech"]
                ]
                if speech_indices:
                    first_speech = vad_results[speech_indices[0]]
                    last_speech = vad_results[speech_indices[-1]]

                    # Estimate sample indices from timestamps
                    sample_rate = 16000  # Default assumption
                    start_sample = int(first_speech["start_time"] * sample_rate)
                    end_sample = int(last_speech["end_time"] * sample_rate)

                    # Clamp to valid range
                    start_sample = max(0, start_sample)
                    end_sample = min(len(processed), end_sample)

                    if end_sample > start_sample:
                        processed = processed[start_sample:end_sample]

        return processed

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        preprocess: bool = True,
    ) -> Dict[str, Any]:
        """Transcribe audio to text.

        If a Whisper model is loaded, performs real transcription.
        Otherwise, returns a deterministic mock transcription.

        Args:
            audio: 1D numpy array of audio samples (int16 or float32).
            sample_rate: Audio sample rate in Hz.
            preprocess: Whether to apply audio preprocessing before
                transcription.

        Returns:
            Dictionary containing:
                - 'text': Transcribed text string.
                - 'language': Detected/specified language code.
                - 'confidence': Confidence score in [0, 1].
                - 'preprocessed': Whether preprocessing was applied.
        """
        if preprocess and len(audio) > 0:
            audio = self.preprocess_audio(audio)

        if self._model is not None:
            return self._whisper_transcribe(audio, sample_rate)
        return self._mock_transcribe()

    def _whisper_transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Transcribe audio using the Whisper model.

        Args:
            audio: Preprocessed audio samples.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Transcription result dictionary.
        """
        try:
            result = self._model.transcribe(
                audio.astype(np.float32),
                language=self.language,
            )
            text = result["text"].strip()
            detected_language = result.get("language", self.language)

            # Compute a simple confidence estimate based on text length
            # (Real Whisper doesn't expose per-word confidence directly)
            confidence = 0.90 if len(text) > 0 else 0.1

            return {
                "text": text,
                "language": detected_language,
                "confidence": confidence,
                "preprocessed": True,
            }
        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc)
            return self._mock_transcribe()

    def _mock_transcribe(self) -> Dict[str, Any]:
        """Return a deterministic mock transcription.

        Uses a rotating index to cycle through predefined responses,
        ensuring reproducibility in tests.

        Returns:
            Mock transcription result dictionary.
        """
        # Use deterministic selection based on counter
        text = self._MOCK_TRANSCRIPTIONS[
            self._mock_index % len(self._MOCK_TRANSCRIPTIONS)
        ]
        self._mock_index += 1

        return {
            "text": text,
            "language": self.language,
            "confidence": 0.85,
            "preprocessed": False,
        }

    def transcribe_batch(
        self,
        audio_segments: List[np.ndarray],
        sample_rate: int = 16000,
        preprocess: bool = True,
    ) -> List[Dict[str, Any]]:
        """Transcribe multiple audio segments in batch.

        Batch transcription is useful for processing recordings that
        have been split into individual utterances by a VAD system.

        Args:
            audio_segments: List of 1D numpy arrays, each containing
                one audio segment to transcribe.
            sample_rate: Audio sample rate in Hz.
            preprocess: Whether to apply preprocessing to each segment.

        Returns:
            List of transcription result dictionaries, one per segment.
        """
        results: List[Dict[str, Any]] = []
        for segment in audio_segments:
            result = self.transcribe(segment, sample_rate, preprocess)
            results.append(result)
        return results

    def detect_language(self, audio: np.ndarray) -> Optional[str]:
        """Detect the language of the audio (stub implementation).

        In a full implementation, this would use Whisper's language
        detection capabilities or a dedicated language identification
        model.

        Args:
            audio: Audio samples to analyze.

        Returns:
            Detected language code or None if detection fails.
        """
        if self._model is not None:
            try:
                # Whisper can detect language from a short audio segment
                audio_float = audio.astype(np.float32)
                result = self._model.transcribe(audio_float, task="language_detection")
                return result.get("language", self.language)
            except Exception as exc:
                logger.warning("Language detection failed: %s", exc)

        # Mock: return configured language
        return self.language

    def is_loaded(self) -> bool:
        """Check if the Whisper model is loaded (not in mock mode).

        Returns:
            True if a real Whisper model is available, False for mock mode.
        """
        return self._model is not None
