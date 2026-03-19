"""Speech-to-Text engine (mock)."""
import logging
from typing import Dict, List, Optional
import numpy as np
from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

HAS_WHISPER = False
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    logger.info("whisper not available — mock STT")


class STTEngine:
    """Speech-to-Text with mock fallback."""

    def __init__(self, model_size: str = "tiny", language: str = "en") -> None:
        self.model_size = model_size
        self.language = language
        self._model = None
        if HAS_WHISPER:
            try:
                self._model = whisper.load_model(model_size)
                logger.info("Loaded Whisper %s", model_size)
            except Exception as e:
                logger.warning("Failed to load Whisper: %s", e)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, str]:
        if self._model is None:
            return self._mock_transcribe()
        try:
            result = self._model.transcribe(audio.astype(np.float32), language=self.language)
            return {"text": result["text"].strip(), "language": result.get("language", "en"),
                    "confidence": 0.95}
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return self._mock_transcribe()

    def _mock_transcribe(self) -> Dict[str, str]:
        texts = [
            "Hello, how can I help you today?",
            "The weather is sunny and warm outside.",
            "I'd like to schedule a meeting for tomorrow.",
            "Can you play some relaxing music?",
            "What time is it right now?",
            "Please set a reminder for five PM.",
        ]
        rng = np.random.default_rng(RANDOM_SEED)
        return {"text": rng.choice(texts), "language": "en", "confidence": 0.85}

    def is_loaded(self) -> bool:
        return self._model is not None
