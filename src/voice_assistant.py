"""Enhanced Voice Assistant with intent pipeline, context tracking, and analytics.

This module implements the core voice assistant pipeline:
    Audio Input → STT (Speech-to-Text) → Intent Classification → Response Generation → TTS

Educational Context:
    A voice assistant is a task-oriented dialogue system that processes
    spoken language to complete user requests. The pipeline involves:

    1. **Audio Capture**: Recording the user's speech via microphone
    2. **STT (Automatic Speech Recognition)**: Converting speech to text
       using acoustic models (e.g., Whisper) that map audio features to words
    3. **NLU (Natural Language Understanding)**: Understanding the meaning
       of the text — primarily intent classification and entity extraction
    4. **Dialogue Management**: Tracking conversation state, context, and
       history for multi-turn conversations
    5. **Response Generation**: Creating an appropriate response based on
       the understood intent and dialogue state
    6. **TTS (Text-to-Speech)**: Converting the text response back to audio
       for the user to hear
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import SAMPLE_RATE

logger = logging.getLogger(__name__)


# Response templates organized by intent
# These use simple slot-filling: {slot_name} gets replaced with extracted values
# In production, templates would come from a content management system
RESPONSE_TEMPLATES: Dict[str, List[str]] = {
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Hey! I'm ready to assist. What do you need?",
        "Greetings! How may I be of service?",
    ],
    "farewell": [
        "Goodbye! Have a great day!",
        "See you later! Don't hesitate to ask if you need anything.",
        "Take care! It was nice chatting with you.",
        "Bye for now! Come back anytime.",
    ],
    "thanks": [
        "You're welcome! Happy to help.",
        "No problem at all!",
        "My pleasure! Is there anything else I can help with?",
        "Glad I could assist!",
    ],
    "weather": [
        "It's currently sunny and warm, about 72 degrees. Perfect weather for a walk!",
        "The weather today is clear with a high of 75°F and a low of 58°F.",
        "Expect partly cloudy skies with temperatures around 68°F today.",
    ],
    "time": [
        "The current time is 3:30 PM Eastern Time.",
        "It's 3:30 PM. Is there something scheduled you'd like to check?",
    ],
    "music": [
        "Playing relaxing lo-fi beats on your speaker. Enjoy!",
        "I'll start your favorite playlist. Music incoming!",
        "Queuing up some ambient music for you.",
    ],
    "reminder": [
        "Reminder set for 5:00 PM today. I'll notify you when it's time.",
        "Done! I've set your reminder. You'll get a notification.",
        "Reminder is active. I won't let you forget!",
    ],
    "calculation": [
        "I'd be happy to help with calculations! Please give me the specific numbers and operation.",
        "Sure, I can do math! What would you like me to calculate?",
    ],
    "name": [
        "I'm your AI voice assistant, powered by local models. You can call me Assistant!",
        "I'm the Realtime Voice Assistant — designed to help you with tasks using voice commands.",
    ],
    "general": [
        "I can help you with weather, time, music, reminders, and calculations. What would you like?",
        "I'm here to help! Try asking about the weather, setting a reminder, or playing some music.",
    ],
}


class ConversationContext:
    """Tracks conversation state for multi-turn dialogue support.

    In a multi-turn conversation, the assistant needs to remember what
    was discussed to provide coherent responses. This context tracker
    maintains a history of interactions and the current dialogue state.

    Attributes:
        turn_count: Number of conversation turns (user + assistant = 1 turn).
        last_intent: The intent of the most recent user utterance.
        last_entities: Any extracted entities from the last utterance.
        awaiting_confirmation: Whether the assistant is waiting for a
            yes/no response from the user.

    Example:
        >>> ctx = ConversationContext()
        >>> ctx.add_turn("user", "hello")
        >>> ctx.add_turn("assistant", "Hi! How can I help?")
        >>> ctx.turn_count
        1
    """

    def __init__(self) -> None:
        """Initialize an empty conversation context."""
        self.turn_count: int = 0
        self.last_intent: Optional[str] = None
        self.last_entities: Dict[str, Any] = {}
        self.awaiting_confirmation: bool = False
        self._history: List[Dict[str, str]] = []

    def add_turn(
        self,
        role: str,
        content: str,
        intent: Optional[str] = None,
    ) -> None:
        """Add a conversation turn to the history.

        Args:
            role: Either 'user' or 'assistant'.
            content: The text content of the message.
            intent: The classified intent (for user turns).
        """
        turn: Dict[str, str] = {"role": role, "content": content}
        if intent:
            turn["intent"] = intent
        self._history.append(turn)

        if role == "user":
            self.turn_count += 1
            if intent:
                self.last_intent = intent

    @property
    def history(self) -> List[Dict[str, str]]:
        """Return a copy of the conversation history."""
        return list(self._history)

    def clear(self) -> None:
        """Reset the conversation context."""
        self.turn_count = 0
        self.last_intent = None
        self.last_entities = {}
        self.awaiting_confirmation = False
        self._history.clear()


class IntentAnalytics:
    """Collects and reports analytics about intent classifications.

    Tracking intent distribution and accuracy is important for:
    - **Model improvement**: Identifying frequently misclassified intents
    - **UX optimization**: Knowing which features users use most
    - **Coverage analysis**: Finding gaps in intent coverage

    Attributes:
        intent_counts: Dictionary tracking how many times each intent
            was classified.
        total_classifications: Total number of classifications performed.
    """

    def __init__(self) -> None:
        """Initialize an empty analytics tracker."""
        self.intent_counts: Dict[str, int] = {}
        self.total_classifications: int = 0

    def record_classification(self, intent: str, confidence: float) -> None:
        """Record a single intent classification event.

        Args:
            intent: The classified intent label.
            confidence: The confidence score (0-1).
        """
        self.intent_counts[intent] = self.intent_counts.get(intent, 0) + 1
        self.total_classifications += 1

    def get_distribution(self) -> Dict[str, float]:
        """Get the distribution of intents as proportions.

        Returns:
            Dictionary mapping intent labels to their proportion (0-1)
            of all classifications.
        """
        if self.total_classifications == 0:
            return {}
        return {
            intent: count / self.total_classifications
            for intent, count in self.intent_counts.items()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of classification analytics.

        Returns:
            Dictionary with total classifications, intent distribution,
            and most common intent.
        """
        most_common_intent = ""
        most_common_count = 0
        for intent, count in self.intent_counts.items():
            if count > most_common_count:
                most_common_intent = intent
                most_common_count = count

        return {
            "total_classifications": self.total_classifications,
            "unique_intents": len(self.intent_counts),
            "intent_distribution": self.get_distribution(),
            "most_common_intent": most_common_intent,
            "most_common_count": most_common_count,
        }

    def reset(self) -> None:
        """Reset all analytics data."""
        self.intent_counts.clear()
        self.total_classifications = 0


class VoiceAssistant:
    """End-to-end voice assistant with STT, intent classification, and TTS.

    This is the main orchestrator that connects all components of the
    voice assistant pipeline:

    1. Receives audio input
    2. Converts speech to text (STT)
    3. Classifies the intent of the transcribed text
    4. Generates an appropriate response using templates
    5. Converts the response text to speech (TTS)
    6. Tracks conversation history and analytics

    Attributes:
        stt: The speech-to-text engine instance.
        tts: The text-to-speech engine instance.
        context: Conversation context tracker.
        analytics: Intent classification analytics.

    Example:
        >>> va = VoiceAssistant()
        >>> result = va.process_audio(np.zeros(16000, dtype=np.int16))
        >>> 'transcription' in result
        True
    """

    def __init__(self) -> None:
        """Initialize the voice assistant with all pipeline components."""
        from src.intent_classifier import IntentClassifierPipeline
        from src.stt_engine import STTEngine
        from src.tts_engine import TTSEngine

        self.stt = STTEngine()
        self.tts = TTSEngine()
        self.intent_pipeline = IntentClassifierPipeline()
        self.context = ConversationContext()
        self.analytics = IntentAnalytics()
        logger.info("VoiceAssistant initialized with intent pipeline")

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> Dict[str, Any]:
        """Process audio through the full voice assistant pipeline.

        This is the main entry point that runs the complete pipeline:
        STT → Intent Classification → Response Generation → TTS

        Args:
            audio: 1D numpy array of audio samples.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Dictionary containing:
                - 'transcription': STT result with text and metadata.
                - 'intent': Classified intent result.
                - 'response_text': Generated text response.
                - 'response_audio_length': Length of TTS audio in samples.
                - 'sample_rate': The sample rate used.
                - 'processing_time_ms': Total pipeline processing time.
        """
        start_time = time.time()

        # Step 1: Speech-to-Text
        transcription = self.stt.transcribe(audio, sample_rate)
        user_text = transcription.get("text", "")

        # Step 2: Intent Classification
        intent_result = self.intent_pipeline.classify(user_text)

        # Record for analytics
        self.analytics.record_classification(
            intent_result.intent, intent_result.confidence
        )

        # Step 3: Response Generation
        response_text = self._generate_response(
            user_text, intent_result.intent
        )

        # Step 4: Text-to-Speech
        response_audio = self.tts.synthesize(response_text, sample_rate)

        # Update conversation context
        self.context.add_turn(
            role="user",
            content=user_text,
            intent=intent_result.intent,
        )
        self.context.add_turn(
            role="assistant",
            content=response_text,
        )

        processing_time = (time.time() - start_time) * 1000.0

        logger.info(
            "Pipeline complete: intent=%s, confidence=%.2f, "
            "processing_time=%.1fms",
            intent_result.intent,
            intent_result.confidence,
            processing_time,
        )

        return {
            "transcription": transcription,
            "intent": {
                "label": intent_result.intent,
                "confidence": intent_result.confidence,
                "classifier": intent_result.classifier_name,
                "matched_keyword": intent_result.matched_keyword,
            },
            "response_text": response_text,
            "response_audio_length": len(response_audio),
            "sample_rate": sample_rate,
            "processing_time_ms": round(processing_time, 2),
        }

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process a text input (bypassing STT, for chat-style interaction).

        This is useful for the Streamlit chat interface and API endpoints
        where the input is already text (typed, not spoken).

        Args:
            text: The user's text input.

        Returns:
            Dictionary with intent classification and response, similar
            to process_audio but without transcription/TTS.
        """
        start_time = time.time()

        intent_result = self.intent_pipeline.classify(text)
        self.analytics.record_classification(
            intent_result.intent, intent_result.confidence
        )

        response_text = self._generate_response(
            text, intent_result.intent
        )

        self.context.add_turn(
            role="user",
            content=text,
            intent=intent_result.intent,
        )
        self.context.add_turn(
            role="assistant",
            content=response_text,
        )

        processing_time = (time.time() - start_time) * 1000.0

        return {
            "intent": {
                "label": intent_result.intent,
                "confidence": intent_result.confidence,
                "classifier": intent_result.classifier_name,
                "matched_keyword": intent_result.matched_keyword,
            },
            "response_text": response_text,
            "processing_time_ms": round(processing_time, 2),
        }

    def _generate_response(self, user_text: str, intent: str) -> str:
        """Generate a response based on the classified intent.

        Uses template-based response generation with slot filling.
        In production, this would be replaced with an LLM or more
        sophisticated dialogue manager.

        The template selection uses a hash of the user text to
        deterministically (but seemingly randomly) pick a template
        variation, making responses feel more natural.

        Args:
            user_text: The user's original input text.
            intent: The classified intent label.

        Returns:
            A response string appropriate for the classified intent.
        """
        templates = RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES["general"])

        if not templates:
            return "I'm not sure how to help with that. Could you rephrase?"

        # Select a template deterministically based on input hash
        # This provides variety while remaining reproducible
        template_index = hash(user_text) % len(templates)
        response = templates[template_index]

        return response

    @property
    def history(self) -> List[Dict[str, str]]:
        """Return the conversation history."""
        return self.context.history

    @property
    def intent_summary(self) -> Dict[str, Any]:
        """Return analytics summary of classified intents."""
        return self.analytics.get_summary()

    def reset_conversation(self) -> None:
        """Clear conversation history and start fresh."""
        self.context.clear()
        logger.info("Conversation reset")

    def reset_analytics(self) -> None:
        """Clear intent analytics data."""
        self.analytics.reset()
        logger.info("Analytics reset")

    @property
    def stt_loaded(self) -> bool:
        """Check if the STT model is loaded (not in mock mode)."""
        return self.stt.is_loaded()

    @property
    def tts_loaded(self) -> bool:
        """Check if the TTS model is loaded (not in mock mode)."""
        return self.tts.is_loaded()
