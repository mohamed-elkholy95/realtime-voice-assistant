"""Comprehensive tests for the intent classification module.

Tests cover:
- Keyword rule classifier with exact and fuzzy matching
- Regex pattern classifier
- Intent classification pipeline fusion
- Confidence scoring
- All supported intents
- Edge cases: empty input, ambiguous input, unknown words
"""

import pytest

from src.intent_classifier import (
    KeywordRuleClassifier,
    RegexPatternClassifier,
    IntentClassifierPipeline,
    IntentResult,
    INTENT_TYPES,
)


class TestIntentResult:
    """Tests for the IntentResult dataclass."""

    def test_valid_creation(self):
        """Valid intent and confidence should create successfully."""
        result = IntentResult(intent="weather", confidence=0.9)
        assert result.intent == "weather"
        assert result.confidence == 0.9

    def test_invalid_intent_raises(self):
        """Unknown intent should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown intent"):
            IntentResult(intent="nonexistent", confidence=0.9)

    def test_confidence_above_one_raises(self):
        """Confidence > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="Confidence"):
            IntentResult(intent="weather", confidence=1.5)

    def test_confidence_negative_raises(self):
        """Negative confidence should raise ValueError."""
        with pytest.raises(ValueError, match="Confidence"):
            IntentResult(intent="weather", confidence=-0.1)

    def test_confidence_zero_valid(self):
        """Confidence of exactly 0.0 should be valid."""
        result = IntentResult(intent="general", confidence=0.0)
        assert result.confidence == 0.0

    def test_confidence_one_valid(self):
        """Confidence of exactly 1.0 should be valid."""
        result = IntentResult(intent="general", confidence=1.0)
        assert result.confidence == 1.0


class TestKeywordRuleClassifier:
    """Tests for keyword-based intent classification."""

    def test_weather_intent(self, intent_pipeline):
        """Weather-related queries should classify as 'weather'."""
        result = intent_pipeline.classify("What's the weather like today?")
        assert result.intent == "weather"

    def test_time_intent(self, intent_pipeline):
        """Time-related queries should classify as 'time'."""
        result = intent_pipeline.classify("What time is it?")
        assert result.intent == "time"

    def test_music_intent(self, intent_pipeline):
        """Music-related queries should classify as 'music'."""
        result = intent_pipeline.classify("Play some music")
        assert result.intent == "music"

    def test_reminder_intent(self, intent_pipeline):
        """Reminder queries should classify as 'reminder'."""
        result = intent_pipeline.classify("Set a reminder for 5 PM")
        assert result.intent == "reminder"

    def test_greeting_intent(self, intent_pipeline):
        """Greeting phrases should classify as 'greeting'."""
        result = intent_pipeline.classify("Hello there!")
        assert result.intent == "greeting"

    def test_farewell_intent(self, intent_pipeline):
        """Farewell phrases should classify as 'farewell'."""
        result = intent_pipeline.classify("Goodbye, see you later")
        assert result.intent == "farewell"

    def test_thanks_intent(self, intent_pipeline):
        """Thank you phrases should classify as 'thanks'."""
        result = intent_pipeline.classify("Thank you very much")
        assert result.intent == "thanks"

    def test_calculation_intent(self, intent_pipeline):
        """Calculation queries should classify as 'calculation'."""
        result = intent_pipeline.classify("Calculate 15% of 200")
        assert result.intent == "calculation"

    def test_name_intent(self, intent_pipeline):
        """Identity questions should classify as 'name'."""
        result = intent_pipeline.classify("Who are you?")
        assert result.intent == "name"

    def test_general_fallback(self, intent_pipeline):
        """Unrecognized input should fall back to 'general'."""
        result = intent_pipeline.classify("xyzzy foobar baz")
        assert result.intent == "general"


class TestFuzzyMatching:
    """Tests for fuzzy keyword matching."""

    def test_typo_in_weather(self):
        """Minor typo in 'weather' should still match via fuzzy matching."""
        classifier = KeywordRuleClassifier(fuzzy_threshold=0.7)
        result = classifier.classify("What's the wether like?")
        assert result.intent == "weather"

    def test_typo_in_music(self):
        """Minor typo in 'music' should still match."""
        classifier = KeywordRuleClassifier(fuzzy_threshold=0.7)
        result = classifier.classify("Play some musik")
        assert result.intent == "music"

    def test_strict_threshold_no_fuzzy(self):
        """High threshold should prevent fuzzy matching."""
        classifier = KeywordRuleClassifier(fuzzy_threshold=0.99)
        result = classifier.classify("wether")  # Typo
        # With strict threshold, typo shouldn't match, falls to general
        assert result.intent == "general"

    def test_confidence_decreases_with_typos(self):
        """Fuzzy matches should have lower confidence than exact matches."""
        classifier = KeywordRuleClassifier(fuzzy_threshold=0.7)
        exact = classifier.classify("What is the weather?")
        fuzzy = classifier.classify("What is the wether?")
        assert exact.confidence >= fuzzy.confidence


class TestRegexPatternClassifier:
    """Tests for regex pattern-based classification."""

    def test_regex_greeting(self):
        """Regex should match greeting patterns."""
        classifier = RegexPatternClassifier()
        result = classifier.classify("Hello there")
        assert result is not None
        assert result.intent == "greeting"

    def test_regex_weather(self):
        """Regex should match weather patterns."""
        classifier = RegexPatternClassifier()
        result = classifier.classify("It will rain tomorrow")
        assert result is not None
        assert result.intent == "weather"

    def test_regex_calculation(self):
        """Regex should match calculation patterns with numbers."""
        classifier = RegexPatternClassifier()
        result = classifier.classify("What is 15 + 27?")
        assert result is not None
        assert result.intent == "calculation"

    def test_regex_no_match(self):
        """Non-matching input should return None."""
        classifier = RegexPatternClassifier()
        result = classifier.classify("xyzzyplugh")
        assert result is None

    def test_regex_custom_patterns(self):
        """Custom patterns should be added and matched."""
        classifier = RegexPatternClassifier(
            custom_patterns=[(r"\b(grocery|shopping)\b", "reminder")]
        )
        result = classifier.classify("Add milk to my grocery list")
        assert result is not None
        assert result.intent == "reminder"


class TestIntentPipeline:
    """Tests for the multi-classifier fusion pipeline."""

    def test_pipeline_returns_result(self, intent_pipeline):
        """Pipeline should always return an IntentResult."""
        result = intent_pipeline.classify("Hello")
        assert isinstance(result, IntentResult)

    def test_pipeline_confidence_range(self, intent_pipeline):
        """Pipeline confidence should always be in [0, 1]."""
        result = intent_pipeline.classify("Hello")
        assert 0.0 <= result.confidence <= 1.0

    def test_pipeline_classified_by(self, intent_pipeline):
        """Pipeline result should indicate the classifiers used."""
        result = intent_pipeline.classify("Hello")
        assert "Pipeline" in result.classifier_name

    def test_batch_classification(self, intent_pipeline):
        """Batch classification should return correct number of results."""
        texts = ["Hello", "What's the weather?", "Goodbye"]
        results = intent_pipeline.classify_batch(texts)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, IntentResult)

    def test_supported_intents(self, intent_pipeline):
        """Supported intents list should include all expected intents."""
        intents = intent_pipeline.get_supported_intents()
        assert "weather" in intents
        assert "greeting" in intents
        assert "general" in intents
        assert len(intents) == len(INTENT_TYPES)

    def test_empty_input(self, intent_pipeline):
        """Empty input should return general with low confidence."""
        result = intent_pipeline.classify("")
        assert result.intent == "general"
        assert result.confidence < 0.5

    def test_whitespace_input(self, intent_pipeline):
        """Whitespace-only input should return general."""
        result = intent_pipeline.classify("   ")
        assert result.intent == "general"


class TestAllIntents:
    """Parametrized test ensuring all intents are reachable."""

    @pytest.mark.parametrize("text,expected_intent", [
        ("hello", "greeting"),
        ("good morning", "greeting"),
        ("hey there", "greeting"),
        ("goodbye", "farewell"),
        ("see you later", "farewell"),
        ("take care", "farewell"),
        ("thanks", "thanks"),
        ("thank you", "thanks"),
        ("appreciate it", "thanks"),
        ("weather", "weather"),
        ("is it raining", "weather"),
        ("temperature outside", "weather"),
        ("what time", "time"),
        ("current time", "time"),
        ("what day", "time"),
        ("play music", "music"),
        ("listen to a song", "music"),
        ("pause the music", "music"),
        ("set a reminder", "reminder"),
        ("remind me", "reminder"),
        ("schedule a meeting", "reminder"),
        ("calculate", "calculation"),
        ("what is 2 plus 2", "calculation"),
        ("your name", "name"),
        ("who are you", "name"),
        ("help", "general"),
        ("tell me something", "general"),
    ])
    def test_intent_classification(self, text, expected_intent, intent_pipeline):
        """Each test input should classify to the expected intent."""
        result = intent_pipeline.classify(text)
        assert result.intent == expected_intent, (
            f"Input '{text}' classified as '{result.intent}', expected '{expected_intent}'"
        )
