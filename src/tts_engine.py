"""Text-to-Speech engine (mock)."""
import logging
from typing import Optional
import numpy as np
from src.config import RANDOM_SEED, SAMPLE_RATE

logger = logging.getLogger(__name__)

HAS_TTS = False
try:
    from TTS.api import TTS  # noqa: F401
    HAS_TTS = True
except ImportError:
    logger.info("TTS not available — mock mode")


class TTSEngine:
    """Text-to-Speech with mock fallback."""

    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC") -> None:
        self.model_name = model_name
        self._model = None
        if HAS_TTS:
            try:
                self._model = TTS(model_name)
                logger.info("Loaded TTS: %s", model_name)
            except Exception as e:
                logger.warning("Failed to load TTS: %s", e)

    def synthesize(self, text: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        if self._model is None:
            return self._mock_synthesize(text, sample_rate)
        try:
            wav = self._model.tts(text)
            return np.array(wav * 32767, dtype=np.int16)
        except Exception as e:
            logger.error("TTS failed: %s", e)
            return self._mock_synthesize(text, sample_rate)

    def _mock_synthesize(self, text: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        duration = max(0.5, len(text) * 0.05)
        return generate_speech_like_audio(duration, sample_rate)

    def is_loaded(self) -> bool:
        return self._model is not None

import numpy as np
from src.audio_processor import generate_speech_like_audio
