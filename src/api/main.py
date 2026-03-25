"""Enhanced FastAPI application for the voice assistant.

This module provides a REST API for interacting with the voice assistant,
including endpoints for transcription, synthesis, intent classification,
evaluation, and conversation management.

Educational Context:
    RESTful APIs are the standard way to expose ML model functionality
    to other applications. A voice assistant API typically provides:

    1. **Transcription endpoints**: Accept audio, return text (STT)
    2. **Synthesis endpoints**: Accept text, return audio (TTS)
    3. **Chat endpoints**: Accept text, return response (end-to-end pipeline)
    4. **Management endpoints**: Model info, health checks, metrics

    FastAPI is a modern Python web framework that provides:
    - Automatic OpenAPI/Swagger documentation
    - Request/response validation with Pydantic
    - Async support for concurrent request handling
    - Type hints → automatic schema generation
"""

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Rate Limiting ──────────────────────────────────────────────────────────
# Token bucket algorithm: each client gets a bucket of tokens that refills
# at a constant rate. Each request consumes one token. When the bucket is
# empty, requests are rejected with HTTP 429 (Too Many Requests).
#
# This is the most common rate limiting algorithm because it allows short
# bursts (up to bucket capacity) while enforcing a long-term average rate.
# Alternatives include:
# - Sliding window: More precise but higher memory overhead
# - Leaky bucket: Smoother output rate, no bursts allowed
# - Fixed window: Simpler but allows burst at window boundaries


class TokenBucketRateLimiter:
    """In-memory token bucket rate limiter keyed by client IP.

    Each client IP address gets an independent token bucket. Tokens
    refill at a constant rate (refill_rate tokens per second) up to
    a maximum capacity. Each request consumes one token.

    This implementation is suitable for single-process deployments.
    For multi-process or distributed deployments, use Redis-backed
    rate limiting (e.g., redis + lua scripts for atomic operations).

    Attributes:
        capacity: Maximum tokens a bucket can hold (burst limit).
        refill_rate: Tokens added per second (sustained rate limit).
        _buckets: Per-client token counts.
        _last_refill: Per-client last refill timestamps.

    Example:
        >>> limiter = TokenBucketRateLimiter(capacity=5, refill_rate=1.0)
        >>> limiter.allow("192.168.1.1")  # First request
        True
    """

    def __init__(
        self,
        capacity: int = 60,
        refill_rate: float = 10.0,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            capacity: Maximum burst size (tokens per bucket).
                60 tokens means a client can make 60 requests instantly.
            refill_rate: Tokens per second added to each bucket.
                10.0 means 10 requests/second sustained rate.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: Dict[str, float] = defaultdict(lambda: float(capacity))
        self._last_refill: Dict[str, float] = defaultdict(time.monotonic)

    def allow(self, client_id: str) -> bool:
        """Check if a request from this client should be allowed.

        Refills the client's bucket based on elapsed time, then
        attempts to consume one token.

        Args:
            client_id: Unique client identifier (typically IP address).

        Returns:
            True if the request is allowed, False if rate limited.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill[client_id]

        # Refill tokens based on elapsed time
        self._buckets[client_id] = min(
            self.capacity,
            self._buckets[client_id] + elapsed * self.refill_rate,
        )
        self._last_refill[client_id] = now

        # Try to consume one token
        if self._buckets[client_id] >= 1.0:
            self._buckets[client_id] -= 1.0
            return True
        return False


# Global rate limiter instance
_rate_limiter = TokenBucketRateLimiter(capacity=60, refill_rate=10.0)

app = FastAPI(
    title="Voice Assistant API",
    version="2.0.0",
    description=(
        "REST API for the Realtime Voice Assistant. "
        "Provides endpoints for speech-to-text, text-to-speech, "
        "intent classification, and conversation management."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply token bucket rate limiting per client IP.

    Rate limiting protects the API from abuse and ensures fair usage:
    - Prevents a single client from monopolizing server resources
    - Protects expensive operations (STT/TTS model inference)
    - Returns standard HTTP 429 with Retry-After header

    The client IP is extracted from X-Forwarded-For (when behind a
    reverse proxy) or falls back to the direct connection IP.
    """
    # Extract client IP — check proxy headers first
    forwarded = request.headers.get("X-Forwarded-For")
    client_ip = forwarded.split(",")[0].strip() if forwarded else (
        request.client.host if request.client else "unknown"
    )

    # Skip rate limiting for health checks (monitoring should never be blocked)
    if request.url.path == "/health":
        return await call_next(request)

    if not _rate_limiter.allow(client_ip):
        logger.warning("Rate limited client: %s on %s", client_ip, request.url.path)
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Too many requests. Please slow down.",
                "retry_after_seconds": 1.0 / _rate_limiter.refill_rate,
            },
            headers={"Retry-After": str(int(1.0 / _rate_limiter.refill_rate))},
        )

    return await call_next(request)


# ── Request/Response Models ────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Request model for the chat endpoint.

    Attributes:
        text: User's text input to process.
        session_id: Optional session identifier for multi-turn conversations.
    """

    text: str = Field(..., min_length=1, max_length=5000, description="User input text")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class ChatResponse(BaseModel):
    """Response model for the chat endpoint.

    Attributes:
        response: The assistant's text response.
        intent: Classified intent information.
        stt_loaded: Whether STT model is loaded.
        tts_loaded: Whether TTS model is loaded.
        processing_time_ms: Time to process the request.
    """

    response: str
    intent: Dict[str, Any] = Field(default_factory=dict)
    stt_loaded: bool = False
    tts_loaded: bool = False
    processing_time_ms: float = 0.0


class TranscribeRequest(BaseModel):
    """Request model for batch transcription.

    Attributes:
        audio_base64: Base64-encoded audio data.
        sample_rate: Audio sample rate in Hz.
    """

    audio_base64: str = Field(..., description="Base64-encoded audio data")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")


class TranscribeResponse(BaseModel):
    """Response model for transcription endpoint."""

    text: str
    language: str = "en"
    confidence: float = 0.0


class SynthesizeRequest(BaseModel):
    """Request model for text-to-speech synthesis.

    Attributes:
        text: Text to synthesize.
        speech_rate: Speed multiplier (0.25 to 4.0).
        pitch_scale: Pitch scaling factor (0.25 to 4.0).
    """

    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    speech_rate: float = Field(1.0, ge=0.25, le=4.0, description="Speed multiplier")
    pitch_scale: float = Field(1.0, ge=0.25, le=4.0, description="Pitch scaling factor")


class SynthesizeResponse(BaseModel):
    """Response model for synthesis endpoint.

    Attributes:
        audio_base64: Base64-encoded audio data.
        sample_rate: Output sample rate.
        duration_samples: Number of audio samples.
    """

    audio_base64: str
    sample_rate: int = 16000
    duration_samples: int = 0


class IntentsResponse(BaseModel):
    """Response model for supported intents listing."""

    intents: List[str]
    count: int


class EvaluateRequest(BaseModel):
    """Request model for running WER evaluation.

    Attributes:
        reference: Ground truth text.
        hypothesis: Predicted text to evaluate.
    """

    reference: str = Field(..., min_length=1, description="Reference text")
    hypothesis: str = Field(..., min_length=1, description="Hypothesis text")


class EvaluateResponse(BaseModel):
    """Response model for evaluation results."""

    wer: float
    cer: float
    reference_length: int
    hypothesis_length: int


class HistoryResponse(BaseModel):
    """Response model for conversation history."""

    history: List[Dict[str, Any]]
    turn_count: int


# ── Shared assistant instance ─────────────────────────────────────────────

# Singleton assistant instance for session state
_assistant_instance: Optional[Any] = None


def _get_assistant():
    """Get or create the shared voice assistant instance.

    Using a singleton avoids recreating the assistant (and its models)
    on every request.

    Returns:
        VoiceAssistant instance.
    """
    global _assistant_instance
    if _assistant_instance is None:
        from src.voice_assistant import VoiceAssistant
        _assistant_instance = VoiceAssistant()
    return _assistant_instance


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        JSON with status indicator.
    """
    return {"status": "healthy", "version": "2.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Process a text chat message through the voice assistant pipeline.

    This endpoint bypasses STT (the input is already text) and runs
    intent classification → response generation.

    Args:
        req: Chat request with user text.

    Returns:
        Assistant response with intent classification metadata.
    """
    try:
        assistant = _get_assistant()
        result = assistant.process_text(req.text)

        return ChatResponse(
            response=result["response_text"],
            intent=result["intent"],
            stt_loaded=assistant.stt_loaded,
            tts_loaded=assistant.tts_loaded,
            processing_time_ms=result["processing_time_ms"],
        )
    except Exception as exc:
        logger.error("Chat endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}")


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    """Transcribe base64-encoded audio to text.

    Accepts audio data as a base64-encoded string, decodes it,
    and runs it through the STT engine.

    Args:
        req: Transcription request with audio data.

    Returns:
        Transcription result with text and confidence.
    """
    try:
        import base64

        from src.stt_engine import STTEngine

        # Decode base64 audio
        audio_bytes = base64.b64decode(req.audio_base64)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)

        if len(audio) == 0:
            raise HTTPException(status_code=400, detail="Empty audio data")

        engine = STTEngine()
        result = engine.transcribe(audio, req.sample_rate)

        return TranscribeResponse(
            text=result["text"],
            language=result.get("language", "en"),
            confidence=result.get("confidence", 0.0),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Transcribe endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Transcription error: {exc}")


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(req: SynthesizeRequest) -> SynthesizeResponse:
    """Synthesize speech from text, returning base64-encoded audio.

    Args:
        req: Synthesis request with text and optional rate/pitch controls.

    Returns:
        Base64-encoded audio data with metadata.
    """
    try:
        import base64

        from src.tts_engine import TTSEngine

        engine = TTSEngine()
        audio = engine.synthesize(
            req.text,
            speech_rate=req.speech_rate,
            pitch_scale=req.pitch_scale,
        )

        # Encode audio as base64
        audio_base64 = base64.b64encode(audio.tobytes()).decode("utf-8")

        return SynthesizeResponse(
            audio_base64=audio_base64,
            sample_rate=16000,
            duration_samples=len(audio),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Synthesize endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Synthesis error: {exc}")


@app.get("/intents", response_model=IntentsResponse)
async def list_intents() -> IntentsResponse:
    """List all supported intent types.

    Returns the full list of intents the classifier can recognize,
    useful for client-side validation and UI display.

    Returns:
        List of supported intent labels.
    """
    from src.intent_classifier import INTENT_TYPES

    return IntentsResponse(intents=INTENT_TYPES, count=len(INTENT_TYPES))


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Evaluate WER and CER between reference and hypothesis text.

    Useful for benchmarking STT model quality by comparing
    model output against known reference transcriptions.

    Args:
        req: Evaluation request with reference and hypothesis text.

    Returns:
        WER, CER, and text length information.
    """
    try:
        from src.evaluation import compute_wer, compute_cer

        wer = compute_wer(req.reference, req.hypothesis)
        cer = compute_cer(req.reference, req.hypothesis)

        return EvaluateResponse(
            wer=round(wer, 4),
            cer=round(cer, 4),
            reference_length=len(req.reference.split()),
            hypothesis_length=len(req.hypothesis.split()),
        )
    except Exception as exc:
        logger.error("Evaluate endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Evaluation error: {exc}")


@app.get("/history", response_model=HistoryResponse)
async def get_history() -> HistoryResponse:
    """Get the current conversation history.

    Returns all turns (user and assistant messages) from the
    current session's conversation.

    Returns:
        Conversation history with turn count.
    """
    assistant = _get_assistant()
    history = assistant.history

    return HistoryResponse(
        history=history,
        turn_count=assistant.context.turn_count,
    )


@app.get("/export")
async def export_conversation(format: str = "markdown") -> Dict[str, str]:
    """Export the current conversation in the specified format.

    Useful for saving conversation logs, generating training data,
    or creating reports from voice assistant interactions.

    Supported formats:
    - 'markdown': Human-readable with intent annotations
    - 'jsonl': JSON Lines for ML training pipelines
    - 'csv': Spreadsheet-compatible tabular format

    Args:
        format: Output format (query parameter, default: 'markdown').

    Returns:
        Dictionary with 'format' and 'content' keys.
    """
    try:
        assistant = _get_assistant()
        content = assistant.export_conversation(format=format)
        return {"format": format, "content": content}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Export endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Export error: {exc}")


@app.get("/analytics")
async def get_analytics() -> Dict[str, Any]:
    """Get intent classification analytics.

    Returns statistics about intent classifications performed
    during this session, including distribution and most common intents.

    Returns:
        Analytics summary dictionary.
    """
    assistant = _get_assistant()
    return assistant.intent_summary


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8008)
