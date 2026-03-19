"""Voice assistant pipeline."""
import numpy as np
import logging
from typing import Any, Dict, Optional
import numpy as np
from src.config import SAMPLE_RATE

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """End-to-end voice assistant: STT → Process → TTS."""

    def __init__(self) -> None:
        from src.stt_engine import STTEngine
        from src.tts_engine import TTSEngine
        self.stt = STTEngine()
        self.tts = TTSEngine()
        self._conversation_history: list = []

    def process_audio(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Dict[str, Any]:
        transcription = self.stt.transcribe(audio, sample_rate)
        response_text = self._generate_response(transcription["text"])
        response_audio = self.tts.synthesize(response_text, sample_rate)
        self._conversation_history.append({"user": transcription["text"], "assistant": response_text})
        return {"transcription": transcription, "response_text": response_text,
                "response_audio_length": len(response_audio), "sample_rate": sample_rate}

    def _generate_response(self, user_text: str) -> str:
        lower = user_text.lower()
        if "weather" in lower: return "It's currently sunny and warm, about 72 degrees. Perfect weather for a walk!"
        if "time" in lower: return "The current time is 3:30 PM Eastern Time."
        if "music" in lower: return "Playing relaxing lo-fi beats on your speaker."
        if "reminder" in lower: return "Reminder set for 5:00 PM today."
        if "name" in lower: return "I'm your AI voice assistant, powered by local models."
        return f"I heard you say: '{user_text}'. How else can I help?"

    @property
    def history(self) -> list:
        return list(self._conversation_history)

    @property
    def stt_loaded(self) -> bool:
        return self.stt.is_loaded()

    @property
    def tts_loaded(self) -> bool:
        return self.tts.is_loaded()
