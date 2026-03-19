import pytest
import numpy as np
from src.stt_engine import STTEngine

class TestSTTEngine:
    def test_mock_transcribe(self):
        engine = STTEngine()
        result = engine.transcribe(np.zeros(16000, dtype=np.int16))
        assert "text" in result
        assert len(result["text"]) > 0
    def test_not_loaded(self):
        assert not STTEngine().is_loaded()
