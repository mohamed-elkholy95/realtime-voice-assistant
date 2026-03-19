import pytest
import numpy as np
from src.tts_engine import TTSEngine

class TestTTSEngine:
    def test_mock_synth(self):
        engine = TTSEngine()
        audio = engine.synthesize("Hello world")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
    def test_duration_scales(self):
        engine = TTSEngine()
        short = len(engine.synthesize("Hi"))
        long = len(engine.synthesize("Hello world this is a longer sentence"))
        assert long > short
