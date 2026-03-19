import pytest
import numpy as np
from src.voice_assistant import VoiceAssistant

class TestVoiceAssistant:
    def test_process(self):
        va = VoiceAssistant()
        result = va.process_audio(np.zeros(16000, dtype=np.int16))
        assert "transcription" in result
        assert "response_text" in result

    def test_weather_intent(self):
        va = VoiceAssistant()
        result = va._generate_response("What's the weather?")
        assert "sunny" in result.lower()

    def test_history(self):
        va = VoiceAssistant()
        va.process_audio(np.zeros(16000, dtype=np.int16))
        assert len(va.history) == 1
