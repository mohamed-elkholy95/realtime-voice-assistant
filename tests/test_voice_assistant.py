"""Comprehensive tests for the voice assistant module.

Tests cover:
- Full audio processing pipeline
- Text-only processing
- Intent integration with response generation
- Conversation history tracking
- Context management
- Intent analytics
- Edge cases
"""

import pytest
import numpy as np

from src.voice_assistant import VoiceAssistant, ConversationContext, IntentAnalytics


class TestVoiceAssistantPipeline:
    """Tests for the full voice assistant pipeline."""

    def test_process_audio_returns_dict(self, voice_assistant, short_audio):
        """process_audio should return a dictionary with expected keys."""
        result = voice_assistant.process_audio(short_audio)
        assert isinstance(result, dict)
        assert "transcription" in result
        assert "intent" in result
        assert "response_text" in result
        assert "response_audio_length" in result

    def test_process_audio_has_intent(self, voice_assistant, short_audio):
        """Pipeline result should include intent classification."""
        result = voice_assistant.process_audio(short_audio)
        assert "label" in result["intent"]
        assert "confidence" in result["intent"]

    def test_process_audio_intent_valid(self, voice_assistant, short_audio):
        """Classified intent should be a valid intent type."""
        result = voice_assistant.process_audio(short_audio)
        from src.intent_classifier import INTENT_TYPES
        assert result["intent"]["label"] in INTENT_TYPES

    def test_process_audio_response_non_empty(self, voice_assistant, short_audio):
        """Response text should be non-empty."""
        result = voice_assistant.process_audio(short_audio)
        assert len(result["response_text"]) > 0

    def test_process_audio_timing(self, voice_assistant, short_audio):
        """Processing time should be recorded."""
        result = voice_assistant.process_audio(short_audio)
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0


class TestTextProcessing:
    """Tests for text-only processing (bypassing STT/TTS)."""

    def test_process_text_returns_dict(self, voice_assistant):
        """process_text should return a dictionary."""
        result = voice_assistant.process_text("Hello")
        assert isinstance(result, dict)
        assert "response_text" in result
        assert "intent" in result

    def test_weather_text(self, voice_assistant):
        """Weather-related text should classify correctly."""
        result = voice_assistant.process_text("What's the weather?")
        assert result["intent"]["label"] == "weather"

    def test_greeting_text(self, voice_assistant):
        """Greeting text should classify correctly."""
        result = voice_assistant.process_text("Hello!")
        assert result["intent"]["label"] == "greeting"

    def test_music_text(self, voice_assistant):
        """Music request should classify correctly."""
        result = voice_assistant.process_text("Play some music")
        assert result["intent"]["label"] == "music"

    def test_time_text(self, voice_assistant):
        """Time query should classify correctly."""
        result = voice_assistant.process_text("What time is it?")
        assert result["intent"]["label"] == "time"

    def test_response_is_string(self, voice_assistant):
        """Response text should be a string."""
        result = voice_assistant.process_text("Hello")
        assert isinstance(result["response_text"], str)


class TestConversationHistory:
    """Tests for conversation history tracking."""

    def test_history_starts_empty(self, voice_assistant):
        """History should be empty before any interaction."""
        assert len(voice_assistant.history) == 0

    def test_history_increments(self, voice_assistant):
        """Each interaction should add to history."""
        voice_assistant.process_text("Hello")
        assert len(voice_assistant.history) == 2  # User + assistant turn

    def test_history_has_roles(self, voice_assistant):
        """History entries should have role and content."""
        voice_assistant.process_text("Hello")
        assert voice_assistant.history[0]["role"] == "user"
        assert voice_assistant.history[1]["role"] == "assistant"
        assert "content" in voice_assistant.history[0]

    def test_history_accumulates(self, voice_assistant):
        """Multiple interactions should accumulate history."""
        voice_assistant.process_text("Hello")
        voice_assistant.process_text("What's the weather?")
        assert len(voice_assistant.history) == 4  # 2 interactions × 2 turns

    def test_reset_clears_history(self, voice_assistant):
        """Reset should clear all conversation history."""
        voice_assistant.process_text("Hello")
        voice_assistant.reset_conversation()
        assert len(voice_assistant.history) == 0


class TestContextManagement:
    """Tests for conversation context tracking."""

    def test_turn_count_increments(self):
        """Turn count should increment with each user message."""
        ctx = ConversationContext()
        assert ctx.turn_count == 0
        ctx.add_turn("user", "Hello")
        assert ctx.turn_count == 1
        ctx.add_turn("assistant", "Hi!")
        assert ctx.turn_count == 1  # Only user turns count
        ctx.add_turn("user", "Weather?")
        assert ctx.turn_count == 2

    def test_last_intent_tracked(self):
        """Last intent should be updated on user turns."""
        ctx = ConversationContext()
        ctx.add_turn("user", "Hello", intent="greeting")
        assert ctx.last_intent == "greeting"
        ctx.add_turn("user", "Weather?", intent="weather")
        assert ctx.last_intent == "weather"

    def test_clear_resets_all(self):
        """Clear should reset all context fields."""
        ctx = ConversationContext()
        ctx.add_turn("user", "Hello", intent="greeting")
        ctx.awaiting_confirmation = True
        ctx.clear()
        assert ctx.turn_count == 0
        assert ctx.last_intent is None
        assert ctx.awaiting_confirmation is False
        assert len(ctx.history) == 0


class TestIntentAnalytics:
    """Tests for intent classification analytics."""

    def test_initial_summary(self):
        """Initial analytics should have zero classifications."""
        analytics = IntentAnalytics()
        summary = analytics.get_summary()
        assert summary["total_classifications"] == 0
        assert summary["unique_intents"] == 0

    def test_record_classification(self):
        """Recording a classification should increment counts."""
        analytics = IntentAnalytics()
        analytics.record_classification("weather", 0.9)
        summary = analytics.get_summary()
        assert summary["total_classifications"] == 1
        assert summary["unique_intents"] == 1

    def test_multiple_classifications(self):
        """Multiple classifications should accumulate correctly."""
        analytics = IntentAnalytics()
        analytics.record_classification("weather", 0.9)
        analytics.record_classification("weather", 0.8)
        analytics.record_classification("greeting", 0.95)
        summary = analytics.get_summary()
        assert summary["total_classifications"] == 3
        assert summary["unique_intents"] == 2
        assert summary["most_common_intent"] == "weather"
        assert summary["most_common_count"] == 2

    def test_distribution(self):
        """Distribution should sum to approximately 1.0."""
        analytics = IntentAnalytics()
        analytics.record_classification("weather", 0.9)
        analytics.record_classification("greeting", 0.8)
        dist = analytics.get_distribution()
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.01

    def test_reset(self):
        """Reset should clear all analytics data."""
        analytics = IntentAnalytics()
        analytics.record_classification("weather", 0.9)
        analytics.reset()
        summary = analytics.get_summary()
        assert summary["total_classifications"] == 0

    def test_assistant_analytics(self, voice_assistant):
        """Voice assistant should track analytics across interactions."""
        voice_assistant.process_text("Hello")
        voice_assistant.process_text("What's the weather?")
        voice_assistant.process_text("Play music")
        summary = voice_assistant.intent_summary
        assert summary["total_classifications"] == 3
