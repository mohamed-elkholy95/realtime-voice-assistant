"""Centralized configuration for the Realtime Voice Assistant project.

This module uses a dataclass-based configuration system with validation and
environment variable override support. All configurable parameters for audio
processing, STT/TTS engines, API server, and evaluation are centralized here.

Environment Variables:
    VOICE_ASSISTANT_SAMPLE_RATE: Override the audio sample rate (default: 16000).
    VOICE_ASSISTANT_LANGUAGE: Override the default language (default: "en").
    VOICE_ASSISTANT_API_HOST: Override the API host (default: "0.0.0.0").
    VOICE_ASSISTANT_API_PORT: Override the API port (default: 8008).
    VOICE_ASSISTANT_LOG_LEVEL: Override the log level (default: "INFO").

Example:
    >>> from src.config import config
    >>> print(config.sample_rate)
    16000
    >>> print(config.audio_duration_seconds)
    10
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Base directory is the project root (two levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class AudioConfig:
    """Configuration for audio processing parameters.

    These parameters control how audio signals are generated, processed,
    and analyzed throughout the voice assistant pipeline.

    Attributes:
        sample_rate: Audio sample rate in Hz. 16 kHz is standard for
            speech processing (covers frequencies up to 8 kHz, capturing
            most speech information per Nyquist theorem).
        audio_duration_seconds: Default duration for generated audio in seconds.
        frame_duration_ms: Frame length in milliseconds for feature extraction.
            25 ms is standard for speech — long enough for a stable spectral
            estimate, short enough to capture transient changes.
        frame_step_ms: Step between consecutive frames in milliseconds.
            10 ms gives 15 ms overlap (60% overlap) at 25 ms frames,
            ensuring smooth transitions in spectral features.
        preemphasis_coeff: Pre-emphasis filter coefficient (0.95–0.97 typical).
            Boosts high frequencies to compensate for the natural spectral
            tilt of voiced speech, which has more energy at low frequencies.
        n_mfcc: Number of Mel-frequency cepstral coefficients to extract.
            13 is standard — captures the vocal tract shape information
            that characterizes different speech sounds.
        n_mels: Number of Mel filter bank channels.
            26–40 is typical; 26 is the original Davis & Mermelstein standard.
        n_fft: FFT size in samples. Should be >= frame size for zero-padding.
            512 samples at 16 kHz = 32 ms, giving good frequency resolution.
        low_freq_hz: Lower edge of Mel filter bank in Hz.
        high_freq_hz: Upper edge of Mel filter bank (None = Nyquist frequency).
        silence_threshold_db: Energy threshold in dB below peak for silence detection.
    """

    sample_rate: int = 16000
    audio_duration_seconds: float = 10.0
    frame_duration_ms: float = 25.0
    frame_step_ms: float = 10.0
    preemphasis_coeff: float = 0.97
    n_mfcc: int = 13
    n_mels: int = 26
    n_fft: int = 512
    low_freq_hz: float = 0.0
    high_freq_hz: Optional[float] = None
    silence_threshold_db: float = -40.0

    def __post_init__(self) -> None:
        """Validate audio configuration parameters after initialization."""
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.audio_duration_seconds <= 0:
            raise ValueError(
                f"audio_duration_seconds must be positive, got {self.audio_duration_seconds}"
            )
        if not (0.0 <= self.preemphasis_coeff <= 1.0):
            raise ValueError(
                f"preemphasis_coeff must be in [0, 1], got {self.preemphasis_coeff}"
            )
        if self.n_mfcc < 1:
            raise ValueError(f"n_mfcc must be >= 1, got {self.n_mfcc}")
        if self.n_mels < 1:
            raise ValueError(f"n_mels must be >= 1, got {self.n_mels}")
        if self.n_fft < 1:
            raise ValueError(f"n_fft must be >= 1, got {self.n_fft}")
        # Compute high frequency as Nyquist if not specified
        if self.high_freq_hz is None:
            self.high_freq_hz = self.sample_rate / 2.0

    @property
    def frame_size_samples(self) -> int:
        """Frame size in samples, computed from frame duration and sample rate.

        Returns:
            Integer frame length in audio samples.
        """
        return int(self.frame_duration_ms / 1000.0 * self.sample_rate)

    @property
    def frame_step_samples(self) -> int:
        """Frame step in samples, computed from step duration and sample rate.

        Returns:
            Integer hop length in audio samples.
        """
        return int(self.frame_step_ms / 1000.0 * self.sample_rate)


@dataclass
class ModelConfig:
    """Configuration for STT/TTS model parameters.

    Attributes:
        whisper_model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
        tts_model_name: Name/path of the TTS model to use.
        language: Default language code for STT processing.
        random_seed: Seed for reproducible random number generation.
    """

    whisper_model_size: str = "tiny"
    tts_model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
    language: str = "en"
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validate model configuration parameters."""
        valid_whisper_sizes = {"tiny", "base", "small", "medium", "large",
                               "tiny.en", "base.en", "small.en", "medium.en"}
        if self.whisper_model_size not in valid_whisper_sizes:
            raise ValueError(
                f"Invalid whisper model size '{self.whisper_model_size}'. "
                f"Must be one of {valid_whisper_sizes}"
            )
        if not self.language or len(self.language) > 10:
            raise ValueError(f"Invalid language code '{self.language}'")


@dataclass
class APIConfig:
    """Configuration for the FastAPI server.

    Attributes:
        host: Host address to bind the server to.
        port: Port number for the server.
        cors_origins: List of allowed CORS origins ('*' allows all).
    """

    host: str = "0.0.0.0"
    port: int = 8008
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    def __post_init__(self) -> None:
        """Validate API configuration parameters."""
        if not (0 < self.port <= 65535):
            raise ValueError(f"Port must be in 1-65535, got {self.port}")


@dataclass
class Config:
    """Root configuration container holding all sub-configurations.

    This is the main entry point for accessing any configuration value.
    Environment variables can override individual settings.

    Example:
        >>> from src.config import config
        >>> config.audio.sample_rate
        16000
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    base_dir: Path = BASE_DIR
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    model_dir: Path = field(default_factory=lambda: BASE_DIR / "models")
    output_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs")
    log_dir: Path = field(default_factory=lambda: BASE_DIR / "logs")
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Apply environment variable overrides and create directories."""
        self._apply_env_overrides()
        self._ensure_directories()

    def _apply_env_overrides(self) -> None:
        """Override configuration values from environment variables.

        Environment variables take precedence over defaults, allowing
        deployment-specific configuration without code changes.
        """
        env_sample_rate = os.environ.get("VOICE_ASSISTANT_SAMPLE_RATE")
        if env_sample_rate:
            try:
                self.audio.sample_rate = int(env_sample_rate)
            except ValueError:
                logger.warning(
                    "Invalid VOICE_ASSISTANT_SAMPLE_RATE='%s', using default %d",
                    env_sample_rate, self.audio.sample_rate
                )

        env_language = os.environ.get("VOICE_ASSISTANT_LANGUAGE")
        if env_language:
            self.model.language = env_language

        env_host = os.environ.get("VOICE_ASSISTANT_API_HOST")
        if env_host:
            self.api.host = env_host

        env_port = os.environ.get("VOICE_ASSISTANT_API_PORT")
        if env_port:
            try:
                self.api.port = int(env_port)
            except ValueError:
                logger.warning(
                    "Invalid VOICE_ASSISTANT_API_PORT='%s', using default %d",
                    env_port, self.api.port
                )

        env_log_level = os.environ.get("VOICE_ASSISTANT_LOG_LEVEL")
        if env_log_level:
            self.log_level = env_log_level

    def _ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for directory in [self.data_dir, self.model_dir, self.output_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import this for configuration access
# Example: from src.config import config; print(config.audio.sample_rate)
config = Config()

# Configure logging with the configured level
logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Convenience aliases for backward compatibility with older code
RANDOM_SEED = config.model.random_seed
SAMPLE_RATE = config.audio.sample_rate
LANGUAGE = config.model.language
API_HOST = config.api.host
API_PORT = config.api.port
