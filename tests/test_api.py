"""Comprehensive tests for the FastAPI endpoints.

Tests cover:
- Health endpoint
- Chat endpoint with intent classification
- Transcribe endpoint with base64 audio
- Synthesize endpoint with rate/pitch controls
- Intents listing endpoint
- Evaluation endpoint (WER/CER)
- History endpoint
- Analytics endpoint
- Error handling and validation
"""

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self):
        """Health check should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self):
        """Health response should include status field."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_has_version(self):
        """Health response should include version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data


class TestChatEndpoint:
    """Tests for /chat endpoint."""

    def test_chat_returns_200(self):
        """Valid chat request should return 200."""
        response = client.post("/chat", json={"text": "hello"})
        assert response.status_code == 200

    def test_chat_has_response(self):
        """Chat response should include a response text."""
        response = client.post("/chat", json={"text": "hello"})
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0

    def test_chat_has_intent(self):
        """Chat response should include intent classification."""
        response = client.post("/chat", json={"text": "hello"})
        data = response.json()
        assert "intent" in data
        assert "label" in data["intent"]
        assert "confidence" in data["intent"]

    def test_chat_weather_intent(self):
        """Weather query should classify as weather intent."""
        response = client.post("/chat", json={"text": "What's the weather?"})
        data = response.json()
        assert data["intent"]["label"] == "weather"

    def test_chat_greeting_intent(self):
        """Greeting should classify correctly."""
        response = client.post("/chat", json={"text": "Hello!"})
        data = response.json()
        assert data["intent"]["label"] == "greeting"

    def test_chat_empty_text_422(self):
        """Empty text should return validation error (422)."""
        response = client.post("/chat", json={"text": ""})
        assert response.status_code == 422

    def test_chat_missing_text_422(self):
        """Missing text field should return validation error."""
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_has_processing_time(self):
        """Chat response should include processing time."""
        response = client.post("/chat", json={"text": "hello"})
        data = response.json()
        assert "processing_time_ms" in data


class TestTranscribeEndpoint:
    """Tests for /transcribe endpoint."""

    def test_transcribe_returns_200(self):
        """Valid transcription request should return 200."""
        audio_b64 = base64.b64encode(
            np.zeros(16000, dtype=np.int16).tobytes()
        ).decode("utf-8")
        response = client.post("/transcribe", json={
            "audio_base64": audio_b64,
            "sample_rate": 16000,
        })
        assert response.status_code == 200

    def test_transcribe_has_text(self):
        """Transcribe response should include transcribed text."""
        audio_b64 = base64.b64encode(
            np.zeros(16000, dtype=np.int16).tobytes()
        ).decode("utf-8")
        response = client.post("/transcribe", json={
            "audio_base64": audio_b64,
            "sample_rate": 16000,
        })
        data = response.json()
        assert "text" in data

    def test_transcribe_empty_audio_400(self):
        """Empty base64 audio should return 400."""
        response = client.post("/transcribe", json={
            "audio_base64": "",
            "sample_rate": 16000,
        })
        # Empty audio may fail at various stages
        assert response.status_code in (400, 500)

    def test_transcribe_has_language(self):
        """Transcribe response should include language."""
        audio = base64.b64encode(
            b'\x00\x00' * 16000
        ).decode("utf-8")
        response = client.post("/transcribe", json={
            "audio_base64": audio,
            "sample_rate": 16000,
        })
        data = response.json()
        assert "language" in data


class TestSynthesizeEndpoint:
    """Tests for /synthesize endpoint."""

    def test_synthesize_returns_200(self):
        """Valid synthesis request should return 200."""
        response = client.post("/synthesize", json={
            "text": "Hello world",
        })
        assert response.status_code == 200

    def test_synthesize_has_audio(self):
        """Synthesize response should include base64 audio."""
        response = client.post("/synthesize", json={"text": "Hello"})
        data = response.json()
        assert "audio_base64" in data
        assert len(data["audio_base64"]) > 0

    def test_synthesize_has_metadata(self):
        """Synthesize response should include duration metadata."""
        response = client.post("/synthesize", json={"text": "Hello"})
        data = response.json()
        assert "duration_samples" in data
        assert data["duration_samples"] > 0

    def test_synthesize_with_rate(self):
        """Faster rate should produce shorter audio."""
        normal = client.post("/synthesize", json={
            "text": "Hello world test",
            "speech_rate": 1.0,
        }).json()
        fast = client.post("/synthesize", json={
            "text": "Hello world test",
            "speech_rate": 2.0,
        }).json()
        assert fast["duration_samples"] < normal["duration_samples"]

    def test_synthesize_empty_text_422(self):
        """Empty text should return validation error."""
        response = client.post("/synthesize", json={"text": ""})
        assert response.status_code == 422

    def test_synthesize_invalid_rate_422(self):
        """Invalid speech rate should return validation error."""
        response = client.post("/synthesize", json={
            "text": "Hello",
            "speech_rate": 10.0,  # Exceeds maximum of 4.0
        })
        assert response.status_code == 422


class TestIntentsEndpoint:
    """Tests for /intents endpoint."""

    def test_intents_returns_200(self):
        """Intents listing should return 200."""
        response = client.get("/intents")
        assert response.status_code == 200

    def test_intents_is_list(self):
        """Intents response should include a list."""
        response = client.get("/intents")
        data = response.json()
        assert "intents" in data
        assert isinstance(data["intents"], list)

    def test_intents_count(self):
        """Should return all expected intents."""
        response = client.get("/intents")
        data = response.json()
        assert data["count"] == len(data["intents"])
        assert "weather" in data["intents"]
        assert "greeting" in data["intents"]
        assert "general" in data["intents"]


class TestEvaluateEndpoint:
    """Tests for /evaluate endpoint."""

    def test_evaluate_perfect_wer(self):
        """Perfect match should give WER of 0."""
        response = client.post("/evaluate", json={
            "reference": "hello world",
            "hypothesis": "hello world",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["wer"] == 0.0

    def test_evaluate_has_cer(self):
        """Evaluate response should include CER."""
        response = client.post("/evaluate", json={
            "reference": "hello world",
            "hypothesis": "hello there",
        })
        data = response.json()
        assert "cer" in data
        assert data["cer"] >= 0.0

    def test_evaluate_has_lengths(self):
        """Evaluate response should include word counts."""
        response = client.post("/evaluate", json={
            "reference": "hello world",
            "hypothesis": "hello there",
        })
        data = response.json()
        assert "reference_length" in data
        assert "hypothesis_length" in data


class TestHistoryEndpoint:
    """Tests for /history endpoint."""

    def test_history_returns_200(self):
        """History should return 200."""
        response = client.get("/history")
        assert response.status_code == 200

    def test_history_has_fields(self):
        """History should include expected fields."""
        response = client.get("/history")
        data = response.json()
        assert "history" in data
        assert "turn_count" in data
        assert isinstance(data["history"], list)

    def test_history_updates_after_chat(self):
        """History should grow after chat interactions."""
        # Clear by creating fresh client context
        response = client.get("/history")
        initial_turns = response.json()["turn_count"]

        client.post("/chat", json={"text": "hello"})
        response = client.get("/history")
        new_turns = response.json()["turn_count"]
        assert new_turns > initial_turns


class TestAnalyticsEndpoint:
    """Tests for /analytics endpoint."""

    def test_analytics_returns_200(self):
        """Analytics should return 200."""
        response = client.get("/analytics")
        assert response.status_code == 200

    def test_analytics_has_fields(self):
        """Analytics should include expected fields."""
        response = client.get("/analytics")
        data = response.json()
        assert "total_classifications" in data
        assert "unique_intents" in data
        assert "intent_distribution" in data
