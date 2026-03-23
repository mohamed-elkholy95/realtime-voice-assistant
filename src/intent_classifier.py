"""Intent classification engine for the voice assistant.

This module provides a multi-strategy intent classification system that
combines keyword matching with fuzzy similarity and regex pattern matching.
The pipeline fuses results from multiple classifiers to produce a single
classified intent with a confidence score.

Educational Context:
    Intent classification is a core component of any task-oriented dialogue
    system. Given a user's utterance, the classifier determines WHAT the
    user wants to do (the "intent"). This is the first step in natural
    language understanding (NLU).

    Production systems typically use:
    - **Supervised ML**: Train a classifier (SVM, neural net) on labeled
      examples (e.g., "What's the weather?" → weather intent)
    - **Deep Learning**: BERT, RoBERTa fine-tuned for intent classification
    - **Few-shot/Zero-shot**: GPT-style models with prompt engineering

    This module implements a rule-based approach suitable for demonstration
    and education, showing the principles behind intent classification
    without requiring a large labeled dataset.
"""

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Supported intent types — extend this as needed
INTENT_TYPES = [
    "greeting",
    "farewell",
    "thanks",
    "weather",
    "time",
    "music",
    "reminder",
    "calculation",
    "name",
    "general",
]


@dataclass
class IntentResult:
    """Result of intent classification with confidence score.

    Attributes:
        intent: The classified intent label (e.g., 'weather', 'greeting').
        confidence: Confidence score in [0, 1] where 1.0 is certain.
        classifier_name: Which classifier produced this result.
        matched_keyword: The keyword or pattern that triggered this intent.
            Useful for debugging and explaining classification decisions.
    """

    intent: str
    confidence: float
    classifier_name: str = ""
    matched_keyword: str = ""

    def __post_init__(self) -> None:
        """Validate the intent result fields."""
        if self.intent not in INTENT_TYPES:
            raise ValueError(
                f"Unknown intent '{self.intent}'. "
                f"Must be one of {INTENT_TYPES}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be in [0, 1], got {self.confidence}"
            )


class KeywordRuleClassifier:
    """Intent classifier based on keyword matching with fuzzy similarity.

    This classifier checks if the user's utterance contains keywords
    associated with each intent. It supports exact matching and fuzzy
    matching using the Ratcliff/Obershelp algorithm (SequenceMatcher)
    to handle typos and word variations.

    The fuzzy matching computes the ratio of matching characters in the
    longest common subsequence to the total string length. A ratio above
    the threshold (default 0.7) is considered a match.

    Attributes:
        intent_keywords: Dictionary mapping intent names to lists of
            associated keywords/phrases.
        fuzzy_threshold: Minimum similarity ratio (0-1) for fuzzy matching.
            Higher values require closer matches (more strict).

    Example:
        >>> classifier = KeywordRuleClassifier()
        >>> result = classifier.classify("What's the weather like today?")
        >>> result.intent
        'weather'
        >>> result.confidence > 0.5
        True
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.7,
        custom_keywords: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Initialize the keyword rule classifier.

        Args:
            fuzzy_threshold: Minimum similarity ratio for fuzzy matching.
            custom_keywords: Optional custom keyword dictionary to
                merge with the defaults.
        """
        self.fuzzy_threshold = fuzzy_threshold

        # Default keyword mapping for each intent
        # Each intent has a list of keywords that indicate that intent
        self.intent_keywords: Dict[str, List[str]] = {
            "greeting": [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "good evening", "howdy", "greetings", "what's up", "sup",
                "yo",
            ],
            "farewell": [
                "bye", "goodbye", "see you", "farewell", "good night",
                "later", "take care", "goodbye for now",
            ],
            "thanks": [
                "thank", "thanks", "thank you", "appreciate", "grateful",
                "cheers", "thx", "ty",
            ],
            "weather": [
                "weather", "temperature", "forecast", "rain", "snow",
                "sunny", "cold", "hot", "humid", "wind", "climate",
            ],
            "time": [
                "time", "clock", "hour", "minute", "date", "day",
                "today", "tomorrow", "yesterday", "what day",
            ],
            "music": [
                "play", "music", "song", "artist", "album", "playlist",
                "radio", "tune", "listen", "pause", "stop music",
            ],
            "reminder": [
                "remind", "reminder", "alarm", "alert", "notify",
                "schedule", "remember", "timer", "set",
            ],
            "calculation": [
                "calculate", "compute", "what is", "how much", "plus",
                "minus", "times", "divided", "percent", "equals",
                "math", "sum",
            ],
            "name": [
                "your name", "who are you", "what are you", "name",
                "identify", "introduce yourself",
            ],
            "general": [
                "help", "can you", "do you", "what can", "how do",
                "tell me", "explain", "know",
            ],
        }

        # Merge any custom keywords provided by the user
        if custom_keywords:
            for intent, keywords in custom_keywords.items():
                if intent in self.intent_keywords:
                    self.intent_keywords[intent].extend(keywords)
                else:
                    self.intent_keywords[intent] = keywords

        logger.info(
            "KeywordRuleClassifier initialized with %d intents, "
            "fuzzy_threshold=%.2f",
            len(self.intent_keywords),
            self.fuzzy_threshold,
        )

    def _fuzzy_match_ratio(self, text: str, keyword: str) -> float:
        """Compute fuzzy similarity ratio between text and a keyword.

        Uses Python's SequenceMatcher which implements the Ratcliff/Obershelp
        algorithm — it finds the longest contiguous matching subsequence,
        then recursively matches the remaining parts on both sides.

        The ratio is: 2.0 * M / T where M is the number of matching
        characters and T is the total number of characters in both strings.

        Args:
            text: The text to search in.
            keyword: The keyword to match against.

        Returns:
            Similarity ratio in [0, 1].
        """
        return SequenceMatcher(None, text.lower(), keyword.lower()).ratio()

    def classify(self, text: str) -> IntentResult:
        """Classify the intent of a user utterance using keyword matching.

        The classification strategy:
        1. Check for exact keyword matches (highest confidence)
        2. Check for fuzzy matches (lower confidence based on similarity)
        3. If no match found, return 'general' with low confidence

        Args:
            text: The user's input text (utterance).

        Returns:
            IntentResult with the best-matching intent and confidence.
        """
        text_lower = text.lower()

        best_intent = "general"
        best_confidence = 0.1  # Low confidence default
        best_keyword = ""

        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                # Strategy 1: Exact substring match (most confident)
                if keyword in text_lower:
                    # Confidence based on keyword specificity:
                    # Longer keywords → more specific → higher confidence
                    confidence = min(0.5 + len(keyword.split()) * 0.15, 0.95)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent
                        best_keyword = keyword

                # Strategy 2: Fuzzy match for handling typos
                # Check both full-text and individual word-level matching
                elif self.fuzzy_threshold > 0:
                    # Word-level fuzzy matching: check each word in text
                    # against each keyword word. This handles typos like
                    # "wether" matching "weather" even in a longer sentence.
                    words = text_lower.split()
                    max_word_ratio = 0.0
                    matched_word = ""
                    for word in words:
                        word_ratio = self._fuzzy_match_ratio(word, keyword)
                        if word_ratio > max_word_ratio:
                            max_word_ratio = word_ratio
                            matched_word = word

                    # Also check full-text ratio
                    full_text_ratio = self._fuzzy_match_ratio(text_lower, keyword)
                    best_ratio = max(max_word_ratio, full_text_ratio)

                    if best_ratio >= self.fuzzy_threshold:
                        # Fuzzy matches get lower confidence than exact
                        fuzzy_confidence = best_ratio * 0.7
                        if fuzzy_confidence > best_confidence:
                            best_confidence = fuzzy_confidence
                            best_intent = intent
                            if max_word_ratio > full_text_ratio:
                                best_keyword = f"(fuzzy word: {matched_word}≈{keyword}, ratio={max_word_ratio:.2f})"
                            else:
                                best_keyword = f"(fuzzy: {keyword}, ratio={full_text_ratio:.2f})"

        return IntentResult(
            intent=best_intent,
            confidence=best_confidence,
            classifier_name="KeywordRuleClassifier",
            matched_keyword=best_keyword,
        )


class RegexPatternClassifier:
    """Intent classifier based on regular expression patterns.

    This classifier uses regex patterns to match structured or
    semi-structured user inputs. Patterns are more powerful than simple
    keyword matching because they can capture:

    - **Word order**: "play music" vs. "music play" can have different intents
    - **Specific structures**: Phone numbers, times, dates
    - **Optional/alternative words**: "what's|what is the weather"
    - **Word boundaries**: Whole-word matching to avoid false positives

    Regex patterns are evaluated in order, and the first match wins.
    This means patterns should be ordered from most specific to least.

    Example:
        >>> classifier = RegexPatternClassifier()
        >>> result = classifier.classify("Calculate 15% of 200")
        >>> result.intent
        'calculation'
    """

    def __init__(
        self,
        custom_patterns: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Initialize the regex pattern classifier.

        Args:
            custom_patterns: Optional list of (pattern, intent) tuples
                to add to the defaults.
        """
        # Each tuple: (compiled regex pattern, intent_label)
        # Patterns use word boundaries (\\b) to match whole words only
        self.patterns: List[Tuple[re.Pattern, str]] = [
            # Greetings — common greeting phrases
            (re.compile(
                r"^(hi|hello|hey|howdy|yo|what'?s up|greetings|good\s+(morning|afternoon|evening))\b",
                re.IGNORECASE,
            ), "greeting"),
            # Farewells
            (re.compile(
                r"\b(bye|goodbye|farewell|see\s+you|take\s+care|good\s+night|later)\b",
                re.IGNORECASE,
            ), "farewell"),
            # Thanks
            (re.compile(
                r"\b(thank(s| you)?|appreciate|grateful|cheers|thx|ty)\b",
                re.IGNORECASE,
            ), "thanks"),
            # Weather — questions about weather conditions
            (re.compile(
                r"\b(weather|temperature|forecast|rain(ing)?|snow(ing)?|"
                r"sunny|cloudy|wind(y)?|humid|degrees|celsius|fahrenheit)\b",
                re.IGNORECASE,
            ), "weather"),
            # Time/date questions
            (re.compile(
                r"\b(what\s+(time|day|date)|current\s+time|"
                r"tell\s+me\s+the\s+(time|date))\b",
                re.IGNORECASE,
            ), "time"),
            # Music playback
            (re.compile(
                r"\b(play|pause|stop|skip|next|previous|volume|"
                r"(listen|queue|add).*(music|song|track|album|playlist))\b",
                re.IGNORECASE,
            ), "music"),
            # Reminders and scheduling
            (re.compile(
                r"\b(remind(er)?|alarm|alert|schedule|notify|timer|"
                r"set\s+(a|an|the)\s+(reminder|alarm|timer))\b",
                re.IGNORECASE,
            ), "reminder"),
            # Calculations — mathematical expressions
            (re.compile(
                r"\b(calculate|compute|what\s+is|how\s+much\s+is|"
                r"\d+\s*[\+\-\*\/\%]\s*\d+|\d+\s*(plus|minus|times|divided|percent))\b",
                re.IGNORECASE,
            ), "calculation"),
            # Identity questions
            (re.compile(
                r"\b(your\s+name|who\s+are\s+you|what\s+are\s+you|"
                r"identify\s+yourself|introduce\s+yourself)\b",
                re.IGNORECASE,
            ), "name"),
        ]

        # Add any custom patterns provided
        if custom_patterns:
            for pattern_str, intent in custom_patterns:
                compiled = re.compile(pattern_str, re.IGNORECASE)
                self.patterns.append((compiled, intent))

        logger.info(
            "RegexPatternClassifier initialized with %d patterns",
            len(self.patterns),
        )

    def classify(self, text: str) -> Optional[IntentResult]:
        """Classify intent by matching regex patterns against the input.

        Patterns are evaluated in order; the first match determines the
        intent. If no pattern matches, returns None (letting the pipeline
        fall back to other classifiers).

        Args:
            text: The user's input text.

        Returns:
            IntentResult if a pattern matched, None otherwise.
        """
        for pattern, intent in self.patterns:
            match = pattern.search(text)
            if match:
                # Confidence is based on how much of the text the pattern
                # covers (longer match = more specific = higher confidence)
                match_length = len(match.group())
                text_length = len(text)
                # Ratio of matched text, bounded to [0.5, 0.95]
                coverage_ratio = match_length / max(text_length, 1)
                confidence = min(0.5 + coverage_ratio * 0.45, 0.95)

                logger.debug(
                    "Regex matched intent=%s, confidence=%.2f, pattern=%s",
                    intent, confidence, pattern.pattern,
                )

                return IntentResult(
                    intent=intent,
                    confidence=confidence,
                    classifier_name="RegexPatternClassifier",
                    matched_keyword=f"regex: {match.group()}",
                )

        # No pattern matched
        logger.debug("No regex pattern matched for input: %s", text[:50])
        return None


class IntentClassifierPipeline:
    """Pipeline that combines multiple intent classifiers with result fusion.

    In production NLU systems, multiple classifiers are often combined:
    - **Ensemble approach**: Average/confidence-weight the scores
    - **Cascade approach**: Try classifiers in order until one is confident
    - **Voting approach**: Multiple classifiers vote on the intent

    This pipeline uses a weighted fusion approach:
    1. Run all classifiers on the input
    2. For each intent, compute a weighted sum of confidence scores
    3. Apply a softmax-like normalization
    4. Return the intent with the highest fused score

    The weights allow prioritizing more reliable classifiers. For example,
    if the regex classifier is highly precise, it gets a higher weight.

    Attributes:
        classifiers: Ordered list of (classifier_instance, weight) tuples.
        fallback_intent: Default intent when all classifiers have low
            confidence.

    Example:
        >>> pipeline = IntentClassifierPipeline()
        >>> result = pipeline.classify("What's the weather like?")
        >>> result.intent
        'weather'
        >>> result.confidence > 0.3
        True
    """

    def __init__(
        self,
        classifiers: Optional[List[Tuple]] = None,
        fallback_intent: str = "general",
    ) -> None:
        """Initialize the intent classification pipeline.

        Args:
            classifiers: Custom list of (classifier, weight) tuples.
                If None, uses the default KeywordRule and Regex classifiers.
            fallback_intent: Intent to return when confidence is below
                the minimum threshold.
        """
        self.fallback_intent = fallback_intent

        if classifiers is not None:
            self.classifiers = classifiers
        else:
            # Default: keyword classifier (weight 0.6) + regex classifier (weight 0.4)
            # Keyword is higher weight because it handles more utterance variations
            keyword_classifier = KeywordRuleClassifier()
            regex_classifier = RegexPatternClassifier()
            self.classifiers = [
                (keyword_classifier, 0.6),
                (regex_classifier, 0.4),
            ]

        logger.info(
            "IntentClassifierPipeline initialized with %d classifiers",
            len(self.classifiers),
        )

    def classify(self, text: str) -> IntentResult:
        """Classify intent using all classifiers and fuse results.

        The fusion process:
        1. Run each classifier on the input text
        2. For each intent that any classifier proposed, sum the
           weighted confidence scores from all classifiers
        3. Return the intent with the highest fused score

        If all classifiers agree on an intent, the confidence is boosted.
        If classifiers disagree, the intent with the highest weighted
        confidence wins, but the final confidence is moderated.

        Args:
            text: The user's input text.

        Returns:
            IntentResult with the best fused intent and confidence.
        """
        if not text or not text.strip():
            return IntentResult(
                intent=self.fallback_intent,
                confidence=0.1,
                classifier_name="IntentClassifierPipeline",
                matched_keyword="empty input",
            )

        # Collect results from all classifiers
        all_results: List[IntentResult] = []
        for classifier, weight in self.classifiers:
            result = classifier.classify(text)
            if result is not None:
                all_results.append((result, weight))

        if not all_results:
            return IntentResult(
                intent=self.fallback_intent,
                confidence=0.1,
                classifier_name="IntentClassifierPipeline",
                matched_keyword="no classifier match",
            )

        # Fuse results: accumulate weighted confidence per intent
        intent_scores: Dict[str, float] = {}
        intent_match_info: Dict[str, str] = {}
        classifiers_used: set = set()

        for result, weight in all_results:
            intent = result.intent
            weighted_score = result.confidence * weight
            intent_scores[intent] = intent_scores.get(intent, 0.0) + weighted_score
            if intent not in intent_match_info or len(result.matched_keyword) > len(
                intent_match_info[intent]
            ):
                intent_match_info[intent] = result.matched_keyword
            classifiers_used.add(result.classifier_name)

        # Find the intent with the highest fused score
        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]

        # Boost confidence if multiple classifiers agree
        # Agreement from multiple sources increases reliability
        agreement_bonus = 0.0
        if len(all_results) > 1:
            agreeing_classifiers = sum(
                1 for r, _ in all_results if r.intent == best_intent
            )
            if agreeing_classifiers > 1:
                # Each agreeing classifier adds a small confidence boost
                agreement_bonus = 0.05 * (agreeing_classifiers - 1)

        final_confidence = min(best_score + agreement_bonus, 1.0)

        return IntentResult(
            intent=best_intent,
            confidence=final_confidence,
            classifier_name=f"Pipeline({'+'.join(sorted(classifiers_used))})",
            matched_keyword=intent_match_info.get(best_intent, ""),
        )

    def classify_batch(self, texts: List[str]) -> List[IntentResult]:
        """Classify intents for multiple texts.

        Args:
            texts: List of user input strings.

        Returns:
            List of IntentResult objects, one per input text.
        """
        return [self.classify(text) for text in texts]

    def get_supported_intents(self) -> List[str]:
        """Return the list of all supported intent types.

        Returns:
            List of intent label strings.
        """
        return list(INTENT_TYPES)
