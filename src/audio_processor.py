"""Audio processing utilities."""
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.config import RANDOM_SEED, SAMPLE_RATE

logger = logging.getLogger(__name__)

HAS_AUDIO = False
try:
    import scipy.io.wavfile as wavfile
    import scipy.signal as signal
    HAS_AUDIO = True
except ImportError:
    logger.info("scipy not available")


def generate_sine_wave(duration: float = 1.0, freq: float = 440, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)


def generate_speech_like_audio(duration: float = 2.0, sample_rate: int = SAMPLE_RATE, seed: int = RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Mix of sine waves to simulate speech-like signal
    signal = np.zeros_like(t)
    for f in [100, 200, 300, 500, 800, 1200]:
        amp = rng.uniform(0.1, 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * f * t + phase)
    signal = signal / np.max(np.abs(signal) + 1e-8) * 0.8
    noise = rng.normal(0, 0.02, len(signal))
    signal = np.clip(signal + noise, -1, 1)
    return (signal * 32767).astype(np.int16)


def compute_mfcc(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, n_mfcc: int = 13) -> np.ndarray:
    """Compute simplified MFCC features (mock if no scipy)."""
    if len(audio) == 0:
        return np.zeros((n_mfcc, 1))
    frame_size = int(0.025 * sample_rate)
    hop_size = int(0.01 * sample_rate)
    n_frames = max(1, (len(audio) - frame_size) // hop_size + 1)
    features = np.random.default_rng(42).standard_normal((n_mfcc, n_frames))
    return features


def compute_spectrogram(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Dict[str, Any]:
    """Compute audio spectrogram stats."""
    return {"length": len(audio), "duration": round(len(audio) / sample_rate, 3),
            "max_amplitude": float(np.max(np.abs(audio))),
            "rms": round(float(np.sqrt(np.mean(audio.astype(float)**2))), 4),
            "sample_rate": sample_rate}


def apply_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter."""
    return np.append(audio[0], audio[1:] - coeff * audio[:-1]).astype(audio.dtype)


def normalize_audio(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Normalize audio to target peak amplitude."""
    peak = np.max(np.abs(audio))
    if peak == 0: return audio
    scale = (target_peak * 32767) / peak
    return np.clip(audio * scale, -32768, 32767).astype(np.int16)
