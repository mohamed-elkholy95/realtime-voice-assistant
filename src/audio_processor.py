"""Audio processing utilities with real scipy-based DSP implementations.

This module provides comprehensive audio signal processing functions used
throughout the voice assistant pipeline, including:

- **MFCC Extraction**: Real Mel-frequency cepstral coefficients using
  DFT → Mel filterbank → log compression → DCT (Davis & Mermelstein, 1980).
- **Spectrogram Computation**: Short-Time Fourier Transform (STFT) with
  Hanning window for time-frequency representation.
- **Silence Detection**: Energy-based Voice Activity Detection (VAD) to
  identify speech vs. silence segments in audio.
- **Audio Generation**: Synthetic sine waves and speech-like signals for
  testing and demonstration purposes.

Educational Context:
    MFCCs are the dominant feature representation for speech processing.
    They approximate the human auditory system's non-linear frequency perception
    (the Mel scale) and the way the cochlea analyzes sound into frequency
    components (like a filter bank). The cepstral analysis step removes the
    excitation source information (pitch), leaving the vocal tract filter
    characteristics that define phonemes.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import RANDOM_SEED, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Check if scipy is available for real DSP operations
HAS_SCIPY = False
try:
    from scipy.fft import dct
    from scipy.signal import get_window
    HAS_SCIPY = True
except ImportError:
    logger.info("scipy not available — using simplified implementations")


def hz_to_mel(frequency_hz: float) -> float:
    """Convert a frequency in Hertz to the Mel scale.

    The Mel scale is a perceptual scale of pitches where equal distances
    on the scale sound equally distant to a human listener. It maps
    linear frequency to a quasi-logarithmic scale.

    The formula used here (HTK-style) is: mel = 2595 * log10(1 + f/700)

    This is the most commonly used Mel scale formula in speech processing.

    Args:
        frequency_hz: Frequency in Hertz. Must be non-negative.

    Returns:
        The corresponding Mel-scale value.

    Raises:
        ValueError: If frequency_hz is negative.

    Example:
        >>> hz_to_mel(1000)
        1000.0  # By definition, 1000 Hz ≈ 1000 Mel (approximately)
        >>> hz_to_mel(0)
        0.0
    """
    if frequency_hz < 0:
        raise ValueError(f"Frequency must be non-negative, got {frequency_hz}")
    return 2595.0 * np.log10(1.0 + frequency_hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert a Mel-scale value to frequency in Hertz.

    This is the inverse of :func:`hz_to_mel`, using the formula:
    hz = 700 * (10^(mel/2595) - 1)

    Args:
        mel: Value on the Mel scale. Must be non-negative.

    Returns:
        The corresponding frequency in Hertz.

    Raises:
        ValueError: If mel is negative.

    Example:
        >>> mel_to_hz(1000.0)  # Approximately 1000 Hz
        999.99...
    """
    if mel < 0:
        raise ValueError(f"Mel value must be non-negative, got {mel}")
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _hz_to_mel_array(frequencies_hz: np.ndarray) -> np.ndarray:
    """Vectorized Mel conversion for an array of frequencies.

    Args:
        frequencies_hz: Array of frequencies in Hertz.

    Returns:
        Array of corresponding Mel-scale values.
    """
    return 2595.0 * np.log10(1.0 + frequencies_hz / 700.0)


def _mel_to_hz_array(mels: np.ndarray) -> np.ndarray:
    """Vectorized inverse Mel conversion for an array of Mel values.

    Args:
        mels: Array of Mel-scale values.

    Returns:
        Array of corresponding frequencies in Hertz.
    """
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def create_mel_filterbank(
    n_filters: int,
    n_fft_size: int,
    sample_rate: int,
    low_freq_hz: float = 0.0,
    high_freq_hz: Optional[float] = None,
) -> np.ndarray:
    """Create a Mel-scale triangular filter bank.

    A Mel filter bank is a set of triangular bandpass filters spaced
    according to the Mel scale. It approximates the frequency selectivity
    of the human auditory system — we perceive frequency on a logarithmic
    scale, not a linear one.

    Each filter has a triangular shape that starts at the center frequency
    of the previous filter, peaks at its own center, and drops to zero at
    the next filter's center. The overlap between filters captures spectral
    energy smoothly across frequencies.

    The filter bank converts a linear frequency axis (from FFT) into a
    perceptually-relevant Mel-frequency representation.

    Args:
        n_filters: Number of Mel filters (typically 26 for standard MFCCs).
        n_fft_size: FFT size used for spectral analysis.
            The filter bank has n_fft_size // 2 + 1 frequency bins.
        sample_rate: Audio sample rate in Hz.
        low_freq_hz: Lower edge of the first Mel filter in Hz.
        high_freq_hz: Upper edge of the last Mel filter in Hz.
            If None, defaults to the Nyquist frequency (sample_rate / 2).

    Returns:
        A 2D numpy array of shape (n_filters, n_fft_size // 2 + 1)
        containing the Mel filter bank weights.

    Raises:
        ValueError: If n_filters < 1, n_fft_size < 1, or frequency bounds
            are invalid.

    Example:
        >>> fb = create_mel_filterbank(26, 512, 16000)
        >>> fb.shape
        (26, 257)
        >>> np.all(fb >= 0)  # All weights non-negative
        True
    """
    if n_filters < 1:
        raise ValueError(f"Number of filters must be >= 1, got {n_filters}")
    if n_fft_size < 1:
        raise ValueError(f"FFT size must be >= 1, got {n_fft_size}")

    if high_freq_hz is None:
        high_freq_hz = sample_rate / 2.0
    if high_freq_hz > sample_rate / 2.0:
        raise ValueError(
            f"high_freq_hz ({high_freq_hz}) exceeds Nyquist ({sample_rate / 2.0})"
        )
    if low_freq_hz < 0 or low_freq_hz >= high_freq_hz:
        raise ValueError(
            f"Invalid frequency bounds: low={low_freq_hz}, high={high_freq_hz}"
        )

    n_freq_bins = n_fft_size // 2 + 1

    # Convert frequency boundaries to Mel scale
    low_mel = hz_to_mel(low_freq_hz)
    high_mel = hz_to_mel(high_freq_hz)

    # Create n_filters + 2 evenly spaced points in Mel space
    # The extra 2 points define the edges of the first and last filters
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = _mel_to_hz_array(mel_points)

    # Map Mel center frequencies to FFT bin indices
    # Each bin covers sample_rate / n_fft_size Hz, so:
    #   bin_index = frequency * n_fft_size / sample_rate
    bin_indices = np.floor((n_fft_size + 1) * hz_points / sample_rate).astype(int)

    filter_bank = np.zeros((n_filters, n_freq_bins))

    for filter_idx in range(n_filters):
        # Three key points for the triangular filter
        left_bin = bin_indices[filter_idx]
        center_bin = bin_indices[filter_idx + 1]
        right_bin = bin_indices[filter_idx + 2]

        # Rising slope: from left to center
        # The filter rises linearly from 0 at left_bin to 1 at center_bin
        for bin_idx in range(left_bin, center_bin):
            if center_bin != left_bin:
                filter_bank[filter_idx, bin_idx] = (
                    (bin_idx - left_bin) / (center_bin - left_bin)
                )

        # Falling slope: from center to right
        # The filter falls linearly from 1 at center_bin to 0 at right_bin
        for bin_idx in range(center_bin, right_bin):
            if right_bin != center_bin:
                filter_bank[filter_idx, bin_idx] = (
                    (right_bin - bin_idx) / (right_bin - center_bin)
                )

    return filter_bank


def generate_sine_wave(
    duration: float = 1.0,
    freq: float = 440,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a pure sine wave tone.

    A sine wave is the simplest periodic waveform, defined as:
    x(t) = A * sin(2π * f * t)

    where A is amplitude, f is frequency, and t is time. Sine waves are
    fundamental in audio DSP because ANY complex waveform can be
    decomposed into a sum of sine waves (Fourier's theorem).

    Args:
        duration: Duration of the tone in seconds.
        freq: Frequency in Hz. A440 (440 Hz) is concert pitch A.
        sample_rate: Samples per second. Higher rates capture higher
            frequencies (Nyquist limit = sample_rate / 2).
        amplitude: Peak amplitude, typically in [0, 1].

    Returns:
        1D numpy array of int16 samples representing the sine wave.

    Example:
        >>> wave = generate_sine_wave(duration=1.0, freq=440)
        >>> wave.dtype
        dtype('int16')
        >>> len(wave)
        16000  # 1 second * 16000 samples/sec
    """
    num_samples = int(sample_rate * duration)
    # Create time array: t = [0, 1/sample_rate, 2/sample_rate, ...]
    time_array = np.linspace(0, duration, num_samples, endpoint=False)
    # Generate sine wave and scale to int16 range
    waveform = np.sin(2.0 * np.pi * freq * time_array) * amplitude
    # Scale to full int16 dynamic range (-32768 to 32767)
    scaled_waveform = (waveform * 32767.0).astype(np.int16)
    return scaled_waveform


def generate_speech_like_audio(
    duration: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Generate synthetic audio that mimics speech-like characteristics.

    This creates a multi-harmonic signal that approximates some properties
    of speech: multiple frequency components (like formants), random amplitude
    variation, and a small amount of noise (simulating breath sounds).

    Real speech has a fundamental frequency (F0, typically 80-300 Hz for
    adults) plus harmonics that are shaped by the vocal tract resonances
    (formants, typically at 300-3500 Hz for vowels). This generator uses
    fixed frequencies to approximate that structure.

    Args:
        duration: Duration of the generated audio in seconds.
        sample_rate: Samples per second.
        seed: Random seed for reproducibility.

    Returns:
        1D numpy array of int16 samples.

    Example:
        >>> audio = generate_speech_like_audio(duration=2.0)
        >>> audio.dtype
        dtype('int16')
        >>> len(audio)
        32000
    """
    rng = np.random.default_rng(seed)
    num_samples = int(sample_rate * duration)
    time_array = np.linspace(0, duration, num_samples, endpoint=False)

    # Mix multiple harmonics to simulate speech formant structure
    # These frequencies approximate typical speech spectral peaks
    formant_frequencies = [100, 200, 300, 500, 800, 1200]
    signal = np.zeros(num_samples, dtype=np.float64)

    for freq in formant_frequencies:
        amplitude = rng.uniform(0.1, 1.0)
        phase = rng.uniform(0, 2.0 * np.pi)
        signal += amplitude * np.sin(2.0 * np.pi * freq * time_array + phase)

    # Normalize to [-1, 1] and scale down to leave headroom
    peak_amplitude = np.max(np.abs(signal))
    if peak_amplitude > 0:
        signal = signal / peak_amplitude * 0.8

    # Add a small amount of noise to simulate ambient/breath sounds
    noise = rng.normal(0, 0.02, num_samples)
    signal = np.clip(signal + noise, -1.0, 1.0)

    return (signal * 32767.0).astype(np.int16)


def compute_mfcc(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = 13,
    n_mels: int = 26,
    n_fft: int = 512,
    frame_duration_ms: float = 25.0,
    frame_step_ms: float = 10.0,
    preemphasis_coeff: float = 0.97,
) -> np.ndarray:
    """Compute Mel-Frequency Cepstral Coefficients (MFCCs) from audio.

    MFCCs are the most widely used features in speech and audio processing.
    The computation pipeline transforms raw audio through several stages
    that mirror human auditory processing:

    1. **Pre-emphasis**: Boost high frequencies (compensate for natural
       spectral tilt where voiced speech has more energy at low freqs).
       Formula: y[n] = x[n] - α * x[n-1]  (α ≈ 0.97)

    2. **Framing & Windowing**: Split signal into overlapping frames
       (25 ms window, 10 ms hop) and apply a Hanning window to reduce
       spectral leakage from framing discontinuities.

    3. **FFT (DFT)**: Compute the Discrete Fourier Transform of each
       frame to convert from time domain to frequency domain.
       This reveals the spectral content (which frequencies are present
       and with what energy).

    4. **Mel Filter Bank**: Apply triangular bandpass filters spaced on
       the Mel scale (perceptual frequency scale). This compresses the
       frequency axis and emphasizes perceptually important regions.

    5. **Log Compression**: Take the log of filter bank energies.
       This approximates human loudness perception (Weber-Fechner law:
       perceived loudness ∝ log(stimulus intensity)).

    6. **DCT (Discrete Cosine Transform)**: Apply Type-II DCT to the
       log Mel energies. The DCT decorrelates the filter bank outputs
       (which are correlated because adjacent filters overlap) and
       concentrates the information into fewer coefficients. The lower
       coefficients capture the overall spectral envelope (vocal tract
       shape = phoneme identity), while higher coefficients capture
       fine spectral details.

    The final output is typically 13 coefficients (0th-12th). The 0th
    coefficient represents the overall spectral energy and is often
    replaced with delta (velocity) and delta-delta (acceleration)
    coefficients for temporal dynamics.

    Args:
        audio: 1D numpy array of audio samples (int16 or float).
        sample_rate: Audio sample rate in Hz.
        n_mfcc: Number of MFCC coefficients to return (typically 13).
        n_mels: Number of Mel filter bank channels (typically 26).
        n_fft: FFT size in samples. Larger FFT gives better frequency
            resolution but poorer time resolution (uncertainty principle).
        frame_duration_ms: Frame length in milliseconds.
        frame_step_ms: Hop between frames in milliseconds.
        preemphasis_coeff: Pre-emphasis filter coefficient.

    Returns:
        2D numpy array of shape (n_mfcc, n_frames) containing the
        MFCC coefficients for each frame.

    Example:
        >>> audio = generate_speech_like_audio(duration=1.0)
        >>> mfccs = compute_mfcc(audio)
        >>> mfccs.shape[0]
        13
        >>> mfccs.shape[1] > 0
        True
    """
    if len(audio) == 0:
        return np.zeros((n_mfcc, 1))

    # Step 0: Convert to float for processing
    float_audio = audio.astype(np.float64)

    # Step 1: Pre-emphasis — boost high frequencies
    # The first sample has no predecessor, so we pass it through unchanged
    preemphasized = np.append(
        float_audio[0], float_audio[1:] - preemphasis_coeff * float_audio[:-1]
    )

    # Step 2: Framing — split into overlapping windows
    frame_size = int(frame_duration_ms / 1000.0 * sample_rate)
    frame_step = int(frame_step_ms / 1000.0 * sample_rate)
    signal_length = len(preemphasized)

    if signal_length < frame_size:
        # Signal too short for even one frame — pad with zeros
        preemphasized = np.pad(preemphasized, (0, frame_size - signal_length))
        signal_length = frame_size

    # Number of complete frames we can extract
    num_frames = 1 + (signal_length - frame_size) // frame_step

    # Create frame indices using broadcasting for efficiency
    frame_indices = np.arange(frame_size)[np.newaxis, :] + \
        (np.arange(num_frames) * frame_step)[:, np.newaxis]

    # Extract all frames into a 2D array
    frames = preemphasized[frame_indices]

    # Step 2b: Apply Hanning window to each frame
    # The Hanning window tapers the frame edges to zero, reducing
    # spectral leakage caused by the abrupt start/end of each frame.
    hanning_window = np.hanning(frame_size)
    windowed_frames = frames * hanning_window

    # Step 3: FFT — convert each frame from time domain to frequency domain
    # The magnitude spectrum shows the energy at each frequency
    magnitude_spectrum = np.abs(np.fft.rfft(windowed_frames, n=n_fft, axis=1))

    # Step 4: Mel filter bank — apply perceptual frequency warping
    # This converts the linear frequency spectrum to Mel-scale energies
    mel_filterbank = create_mel_filterbank(
        n_filters=n_mels,
        n_fft_size=n_fft,
        sample_rate=sample_rate,
    )
    mel_energies = np.dot(magnitude_spectrum, mel_filterbank.T)

    # Replace zeros to avoid log(0) = -inf
    mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)

    # Step 5: Log compression — model human loudness perception
    log_mel_energies = np.log(mel_energies)

    # Step 6: DCT — decorrelate and compress information
    # DCT Type-II is standard for MFCC computation
    # We take only the first n_mfcc coefficients (they contain most
    # of the spectral envelope information)
    if HAS_SCIPY:
        mfccs = dct(log_mel_energies, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    else:
        # Fallback: simplified DCT using matrix multiplication
        # DCT-II can be computed as: C = M * log_mel_energies
        # where M[k,n] = cos(π * k * (2n+1) / (2N))
        n_filters_actual = log_mel_energies.shape[1]
        n_coeff = min(n_mfcc, n_filters_actual)
        k_indices = np.arange(n_coeff)[:, np.newaxis]
        n_indices = np.arange(n_filters_actual)[np.newaxis, :]
        dct_matrix = np.cos(
            np.pi * k_indices * (2.0 * n_indices + 1.0) / (2.0 * n_filters_actual)
        )
        # Orthogonal normalization factor
        dct_matrix[0, :] *= np.sqrt(1.0 / (2.0 * n_filters_actual))
        dct_matrix[1:, :] *= np.sqrt(1.0 / n_filters_actual)
        mfccs = np.dot(log_mel_energies, dct_matrix.T)

    # Transpose to shape (n_mfcc, n_frames) — convention: features × time
    return mfccs.T


def compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = 512,
    frame_step_ms: float = 10.0,
    frame_duration_ms: float = 25.0,
) -> Dict[str, Any]:
    """Compute Short-Time Fourier Transform (STFT) spectrogram and audio statistics.

    The STFT splits a signal into short overlapping frames and computes
    the FFT of each frame. This gives a time-frequency representation
    showing how the frequency content of the signal evolves over time.

    The trade-off in STFT is between time and frequency resolution:
    - Longer frames → better frequency resolution, worse time resolution
    - Shorter frames → better time resolution, worse frequency resolution
    This is a consequence of the Gabor limit (uncertainty principle
    applied to signal processing).

    Args:
        audio: 1D numpy array of audio samples.
        sample_rate: Audio sample rate in Hz.
        n_fft: FFT size in samples.
        frame_step_ms: Hop between frames in milliseconds.
        frame_duration_ms: Frame length in milliseconds.

    Returns:
        Dictionary containing:
            - 'length': Number of samples in the audio.
            - 'duration': Duration in seconds.
            - 'max_amplitude': Peak amplitude value.
            - 'rms': Root Mean Square energy (proxy for loudness).
            - 'sample_rate': The sample rate used.
            - 'spectrogram': 2D array (frequency_bins × time_frames) of
              magnitude spectrum values in dB.
            - 'frequencies': Array of frequency values for each bin in Hz.
            - 'times': Array of time values for each frame in seconds.

    Example:
        >>> audio = generate_sine_wave(duration=1.0, freq=440)
        >>> result = compute_spectrogram(audio)
        >>> result['duration']
        1.0
        >>> 'spectrogram' in result
        True
    """
    float_audio = audio.astype(np.float64)
    signal_length = len(float_audio)
    duration = signal_length / sample_rate

    frame_size = int(frame_duration_ms / 1000.0 * sample_rate)
    frame_step = int(frame_step_ms / 1000.0 * sample_rate)

    if signal_length < frame_size:
        padded = np.pad(float_audio, (0, frame_size - signal_length))
    else:
        padded = float_audio

    num_frames = 1 + (len(padded) - frame_size) // frame_step

    # Extract frames and apply Hanning window
    frame_indices = np.arange(frame_size)[np.newaxis, :] + \
        (np.arange(num_frames) * frame_step)[:, np.newaxis]
    frames = padded[frame_indices]
    hanning_window = np.hanning(frame_size)
    windowed_frames = frames * hanning_window

    # Compute magnitude spectrum in dB
    magnitude_spectrum = np.abs(
        np.fft.rfft(windowed_frames, n=n_fft, axis=1)
    )
    # Convert to dB: 20 * log10(magnitude), with small epsilon to avoid log(0)
    magnitude_db = 20.0 * np.log10(magnitude_spectrum + 1e-10)

    # Frequency and time axes
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    times = np.arange(num_frames) * frame_step / sample_rate

    # Basic statistics
    rms = float(np.sqrt(np.mean(float_audio ** 2)))
    max_amplitude = float(np.max(np.abs(float_audio)))

    return {
        "length": signal_length,
        "duration": round(duration, 3),
        "max_amplitude": max_amplitude,
        "rms": round(rms, 4),
        "sample_rate": sample_rate,
        "spectrogram": magnitude_db,
        "frequencies": frequencies,
        "times": times,
    }


def detect_silence(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_duration_ms: float = 25.0,
    frame_step_ms: float = 10.0,
    threshold_db: float = -40.0,
) -> List[Dict[str, Any]]:
    """Detect silent regions in audio using energy-based Voice Activity Detection (VAD).

    Voice Activity Detection identifies which portions of an audio signal
    contain speech (or other non-silence) and which are silent. This is
    crucial for:

    - **Endpointing**: Detecting when a user starts/stops speaking
    - **Noise reduction**: Focusing processing on speech frames only
    - **Bandwidth optimization**: Only transmitting speech frames

    This implementation uses a simple energy-based approach:
    1. Split audio into short frames (25 ms)
    2. Compute the energy of each frame: E = (1/N) * Σ x²[n]
    3. Convert to decibels: dB = 10 * log10(E)
    4. Compare against a threshold to classify speech vs. silence

    More sophisticated VAD approaches include:
    - Statistical models (GMM-based)
    - Neural network-based classifiers
    - Multi-feature (energy + zero-crossing rate + spectral features)

    Args:
        audio: 1D numpy array of audio samples.
        sample_rate: Audio sample rate in Hz.
        frame_duration_ms: Frame length for energy computation in ms.
        frame_step_ms: Hop between frames in ms.
        threshold_db: Energy threshold in dB. Frames below this
            are classified as silence. -40 dB is a reasonable default
            for typical recording conditions.

    Returns:
        List of dictionaries, one per frame, with keys:
            - 'start_time': Frame start time in seconds.
            - 'end_time': Frame end time in seconds.
            - 'energy_db': Frame energy in decibels.
            - 'is_speech': Boolean indicating if the frame contains speech.

    Example:
        >>> audio = generate_sine_wave(duration=1.0, freq=440)
        >>> results = detect_silence(audio)
        >>> len(results) > 0
        True
    """
    float_audio = audio.astype(np.float64)
    signal_length = len(float_audio)

    if signal_length == 0:
        return []

    frame_size = int(frame_duration_ms / 1000.0 * sample_rate)
    frame_step = int(frame_step_ms / 1000.0 * sample_rate)

    if signal_length < frame_size:
        # Too short — treat as a single frame
        frame_size = signal_length
        frame_step = signal_length

    num_frames = 1 + (signal_length - frame_size) // frame_step

    # Compute the overall peak energy for reference
    frame_indices = np.arange(frame_size)[np.newaxis, :] + \
        (np.arange(num_frames) * frame_step)[:, np.newaxis]
    frames = float_audio[frame_indices]

    # Energy of each frame: E = mean(x^2)
    # This is the mean square energy, a proxy for loudness
    frame_energies = np.mean(frames ** 2, axis=1)

    # Convert to decibels: dB = 10 * log10(E)
    # Add epsilon to avoid log(0) for silent frames
    epsilon = 1e-10
    frame_energies_db = 10.0 * np.log10(frame_energies + epsilon)

    # Compute peak energy as reference for adaptive thresholding
    peak_energy_db = np.max(frame_energies_db)

    # For completely silent audio, all frames should be classified as silence
    # (epsilon in log gives ~-100 dB for zero-energy frames)
    if peak_energy_db < -90:
        return [
            {
                "start_time": round(frame_idx * frame_step / sample_rate, 4),
                "end_time": round((frame_idx * frame_step + frame_size) / sample_rate, 4),
                "energy_db": round(float(frame_energies_db[frame_idx]), 2),
                "is_speech": False,
            }
            for frame_idx in range(num_frames)
        ]

    # The effective threshold is relative to the peak
    # This adapts to varying recording levels
    effective_threshold = peak_energy_db + threshold_db

    # Classify each frame
    results: List[Dict[str, Any]] = []
    for frame_idx in range(num_frames):
        start_time = frame_idx * frame_step / sample_rate
        end_time = (frame_idx * frame_step + frame_size) / sample_rate
        energy_db = float(frame_energies_db[frame_idx])
        is_speech = energy_db >= effective_threshold

        results.append({
            "start_time": round(start_time, 4),
            "end_time": round(end_time, 4),
            "energy_db": round(energy_db, 2),
            "is_speech": is_speech,
        })

    return results


def extract_speech_segments(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    threshold_db: float = -40.0,
    min_speech_duration_ms: float = 100.0,
    min_silence_duration_ms: float = 300.0,
) -> List[Dict[str, Any]]:
    """Extract speech segments from audio, removing silence.

    This function uses energy-based VAD to identify regions of speech
    and returns the start/end times and audio data for each speech segment.
    Adjacent speech frames are merged into continuous segments if the gap
    between them is shorter than min_silence_duration_ms.

    This is useful for:
    - Pre-processing audio before STT to remove leading/trailing silence
    - Splitting recordings into individual utterances
    - Reducing computational cost by processing only speech portions

    Args:
        audio: 1D numpy array of audio samples.
        sample_rate: Audio sample rate in Hz.
        threshold_db: Energy threshold in dB for speech detection.
        min_speech_duration_ms: Minimum duration of a speech segment to keep.
            Shorter segments (clicks, pops) are discarded.
        min_silence_duration_ms: Minimum silence gap to split segments.
            Gaps shorter than this merge adjacent speech regions.

    Returns:
        List of dictionaries, one per speech segment:
            - 'start_time': Segment start in seconds.
            - 'end_time': Segment end in seconds.
            - 'duration': Segment duration in seconds.
            - 'audio': Extracted audio samples as numpy array.
            - 'start_sample': Start index in the original audio.
            - 'end_sample': End index in the original audio.

    Example:
        >>> audio = generate_speech_like_audio(duration=2.0)
        >>> segments = extract_speech_segments(audio)
    """
    vad_results = detect_silence(
        audio, sample_rate, threshold_db=threshold_db
    )
    if not vad_results:
        return []

    frame_step_ms = 10.0  # Default from detect_silence
    frame_duration_ms = 25.0
    frame_step_samples = int(frame_step_ms / 1000.0 * sample_rate)

    # Identify continuous speech regions
    # A segment starts when we transition from silence to speech
    # and ends when we transition from speech to silence
    segments: List[Dict[str, Any]] = []
    in_speech = False
    segment_start_idx = 0

    min_silence_frames = int(min_silence_duration_ms / frame_step_ms)
    silence_counter = 0

    for frame_idx, result in enumerate(vad_results):
        if result["is_speech"]:
            if not in_speech:
                # Start a new segment
                segment_start_idx = frame_idx
                in_speech = True
            silence_counter = 0
        else:
            if in_speech:
                silence_counter += 1
                if silence_counter >= min_silence_frames:
                    # End the segment
                    segment_end_idx = frame_idx
                    start_sample = segment_start_idx * frame_step_samples
                    end_sample = segment_end_idx * frame_step_samples + \
                        int(frame_duration_ms / 1000.0 * sample_rate)
                    end_sample = min(end_sample, len(audio))

                    duration_ms = (end_sample - start_sample) / sample_rate * 1000
                    if duration_ms >= min_speech_duration_ms:
                        segments.append({
                            "start_time": round(start_sample / sample_rate, 4),
                            "end_time": round(end_sample / sample_rate, 4),
                            "duration": round(
                                (end_sample - start_sample) / sample_rate, 4
                            ),
                            "audio": audio[start_sample:end_sample],
                            "start_sample": start_sample,
                            "end_sample": end_sample,
                        })
                    in_speech = False

    # Handle case where audio ends during a speech segment
    if in_speech:
        start_sample = segment_start_idx * frame_step_samples
        end_sample = len(audio)
        duration_ms = (end_sample - start_sample) / sample_rate * 1000
        if duration_ms >= min_speech_duration_ms:
            segments.append({
                "start_time": round(start_sample / sample_rate, 4),
                "end_time": round(end_sample / sample_rate, 4),
                "duration": round(
                    (end_sample - start_sample) / sample_rate, 4
                ),
                "audio": audio[start_sample:end_sample],
                "start_sample": start_sample,
                "end_sample": end_sample,
            })

    return segments


def compute_log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = 26,
    n_fft: int = 512,
    frame_duration_ms: float = 25.0,
    frame_step_ms: float = 10.0,
    preemphasis_coeff: float = 0.97,
) -> np.ndarray:
    """Compute Whisper-style log-MEL spectrogram.

    This computes a log-Mel spectrogram as used in Whisper and similar
    modern speech models. It follows the same pipeline as MFCC computation
    but stops before the DCT step, preserving the full Mel-frequency
    representation.

    The log-Mel spectrogram is a 2D time-frequency image where:
    - X-axis: time (frame index)
    - Y-axis: Mel frequency bins (perceptual frequency scale)
    - Color/value: log energy in decibels

    Whisper normalizes this further by converting to dB relative to peak
    and clipping to [-1, 1], but this function returns raw log values.

    Args:
        audio: 1D numpy array of audio samples.
        sample_rate: Audio sample rate in Hz.
        n_mels: Number of Mel filter bank channels.
        n_fft: FFT size in samples.
        frame_duration_ms: Frame length in milliseconds.
        frame_step_ms: Hop between frames in milliseconds.
        preemphasis_coeff: Pre-emphasis filter coefficient.

    Returns:
        2D numpy array of shape (n_mels, n_frames) containing log Mel
        energies for each Mel band and time frame.

    Example:
        >>> audio = generate_speech_like_audio(duration=1.0)
        >>> mel_spec = compute_log_mel_spectrogram(audio)
        >>> mel_spec.shape[0]
        26
    """
    if len(audio) == 0:
        return np.zeros((n_mels, 1))

    float_audio = audio.astype(np.float64)

    # Pre-emphasis
    preemphasized = np.append(
        float_audio[0], float_audio[1:] - preemphasis_coeff * float_audio[:-1]
    )

    # Framing
    frame_size = int(frame_duration_ms / 1000.0 * sample_rate)
    frame_step = int(frame_step_ms / 1000.0 * sample_rate)
    signal_length = len(preemphasized)

    if signal_length < frame_size:
        preemphasized = np.pad(preemphasized, (0, frame_size - signal_length))
        signal_length = frame_size

    num_frames = 1 + (signal_length - frame_size) // frame_step
    frame_indices = np.arange(frame_size)[np.newaxis, :] + \
        (np.arange(num_frames) * frame_step)[:, np.newaxis]
    frames = preemphasized[frame_indices]

    # Hanning window
    hanning_window = np.hanning(frame_size)
    windowed_frames = frames * hanning_window

    # FFT → magnitude spectrum
    magnitude_spectrum = np.abs(np.fft.rfft(windowed_frames, n=n_fft, axis=1))

    # Mel filter bank
    mel_filterbank = create_mel_filterbank(
        n_filters=n_mels, n_fft_size=n_fft, sample_rate=sample_rate,
    )
    mel_energies = np.dot(magnitude_spectrum, mel_filterbank.T)

    # Log compression with floor to avoid -inf
    mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)
    log_mel = np.log(mel_energies)

    # Transpose: (n_mels, n_frames) — features × time
    return log_mel.T


def compute_delta_features(
    features: np.ndarray,
    width: int = 2,
) -> np.ndarray:
    """Compute delta (velocity) features from a feature matrix.

    Delta features capture the temporal dynamics of spectral features —
    how the features change over time. This is critical for speech
    recognition because phonemes are defined not just by their spectral
    shape but by how that shape transitions (e.g., a stop consonant like
    /b/ is characterized by a rapid spectral transition, not a static shape).

    The delta is computed using a regression formula over a window of
    ±width frames:

        delta[t] = Σ_{n=1}^{width} n * (features[t+n] - features[t-n])
                   / (2 * Σ_{n=1}^{width} n²)

    This is equivalent to fitting a straight line to the features over
    the window and taking the slope, which is more robust than a simple
    first difference (features[t] - features[t-1]).

    Standard practice is to append delta (Δ) and delta-delta (ΔΔ)
    features to the static MFCCs, tripling the feature vector from
    13 to 39 dimensions. The delta-delta features capture acceleration
    — how the rate of change itself is changing.

    Args:
        features: 2D array of shape (n_features, n_frames). Each column
            is one frame's feature vector.
        width: Number of frames on each side for the regression window.
            width=2 uses a 5-frame window (t-2, t-1, t, t+1, t+2).

    Returns:
        2D array of shape (n_features, n_frames) containing the delta
        features. Edge frames are padded by repeating the nearest frame.

    Raises:
        ValueError: If width < 1 or features has fewer than 2 dimensions.

    Example:
        >>> import numpy as np
        >>> features = np.array([[1, 2, 4, 7, 11]], dtype=np.float64)
        >>> deltas = compute_delta_features(features, width=1)
        >>> deltas.shape == features.shape
        True
        >>> deltas[0, 1]  # (4 - 1) / 2 = 1.5
        1.5
    """
    if width < 1:
        raise ValueError(f"Delta width must be >= 1, got {width}")
    if features.ndim != 2:
        raise ValueError(
            f"Features must be 2D (n_features, n_frames), got shape {features.shape}"
        )

    n_features, n_frames = features.shape
    if n_frames == 0:
        return np.zeros_like(features)

    # Pad the features at both ends by repeating edge frames
    # This avoids boundary effects in the regression computation
    padded = np.pad(features, ((0, 0), (width, width)), mode="edge")

    # Denominator: 2 * Σ_{n=1}^{width} n²
    denominator = 2.0 * sum(n * n for n in range(1, width + 1))

    # Compute deltas using the regression formula
    deltas = np.zeros_like(features)
    for n in range(1, width + 1):
        # padded[:, width+n : width+n+n_frames] = features shifted left by n
        # padded[:, width-n : width-n+n_frames] = features shifted right by n
        deltas += n * (
            padded[:, width + n : width + n + n_frames]
            - padded[:, width - n : width - n + n_frames]
        )
    deltas /= denominator

    return deltas


def compute_mfcc_with_deltas(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = 13,
    n_mels: int = 26,
    n_fft: int = 512,
    include_delta: bool = True,
    include_delta_delta: bool = True,
    delta_width: int = 2,
) -> np.ndarray:
    """Compute MFCCs with optional delta and delta-delta features.

    This is the standard feature extraction pipeline used in most ASR
    systems. The full 39-dimensional feature vector consists of:

    - **Static MFCCs** (coefficients 0–12): Capture the spectral envelope
      shape at each time frame — what phoneme is being produced.
    - **Delta MFCCs** (Δ, coefficients 13–25): Capture the velocity of
      spectral change — how the sound is transitioning between phonemes.
    - **Delta-delta MFCCs** (ΔΔ, coefficients 26–38): Capture the
      acceleration of spectral change — useful for detecting transient
      events like stop consonant releases.

    Research has consistently shown that adding delta and delta-delta
    features improves ASR accuracy by 10–20% over static MFCCs alone,
    because speech is inherently a dynamic process.

    Args:
        audio: 1D numpy array of audio samples.
        sample_rate: Audio sample rate in Hz.
        n_mfcc: Number of static MFCC coefficients.
        n_mels: Number of Mel filter bank channels.
        n_fft: FFT size in samples.
        include_delta: Whether to append delta (Δ) features.
        include_delta_delta: Whether to append delta-delta (ΔΔ) features.
            Only applies if include_delta is True.
        delta_width: Regression window width for delta computation.

    Returns:
        2D numpy array of shape (n_features, n_frames) where n_features
        is n_mfcc * (1 + include_delta + include_delta_delta).
        With defaults: (39, n_frames) for the full feature set.

    Example:
        >>> audio = generate_speech_like_audio(duration=1.0)
        >>> features = compute_mfcc_with_deltas(audio)
        >>> features.shape[0]
        39
        >>> static_only = compute_mfcc_with_deltas(audio, include_delta=False)
        >>> static_only.shape[0]
        13
    """
    # Compute static MFCCs
    static_mfccs = compute_mfcc(
        audio,
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
    )

    if not include_delta:
        return static_mfccs

    # Compute delta (velocity) features
    delta_mfccs = compute_delta_features(static_mfccs, width=delta_width)
    feature_stack = [static_mfccs, delta_mfccs]

    if include_delta_delta:
        # Delta-delta = delta of delta (acceleration)
        delta_delta_mfccs = compute_delta_features(delta_mfccs, width=delta_width)
        feature_stack.append(delta_delta_mfccs)

    # Stack vertically: [static; delta; delta-delta]
    return np.vstack(feature_stack)


def estimate_snr(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_duration_ms: float = 25.0,
    frame_step_ms: float = 10.0,
    silence_threshold_db: float = -40.0,
) -> Dict[str, float]:
    """Estimate the Signal-to-Noise Ratio (SNR) of an audio signal.

    SNR measures how much louder the desired signal (speech) is compared
    to the background noise. It is defined as:

        SNR = 10 * log10(P_signal / P_noise)   [in dB]

    where P_signal and P_noise are the average power of the speech and
    noise segments respectively. Higher SNR means cleaner audio:

    - **>40 dB**: Studio-quality recording
    - **20-40 dB**: Good quality, typical quiet room
    - **10-20 dB**: Noisy environment, ASR accuracy starts degrading
    - **<10 dB**: Very noisy, significant ASR degradation expected

    This function estimates SNR by:
    1. Running Voice Activity Detection to classify frames as speech/silence
    2. Computing the mean power of speech frames (signal + noise)
    3. Computing the mean power of silence frames (noise only)
    4. SNR ≈ 10 * log10(P_speech / P_silence)

    This is an approximation because silence frames contain only noise,
    while speech frames contain signal + noise. For a more precise estimate,
    spectral subtraction or NIST WADA-SNR methods would be needed.

    Args:
        audio: 1D numpy array of audio samples.
        sample_rate: Audio sample rate in Hz.
        frame_duration_ms: Frame length in milliseconds for VAD.
        frame_step_ms: Hop between frames in milliseconds.
        silence_threshold_db: Energy threshold for speech/silence classification.

    Returns:
        Dictionary containing:
            - 'snr_db': Estimated SNR in decibels.
            - 'signal_power_db': Average power of speech frames in dB.
            - 'noise_power_db': Average power of silence frames in dB.
            - 'speech_ratio': Fraction of frames classified as speech.
            - 'n_speech_frames': Number of speech frames found.
            - 'n_silence_frames': Number of silence frames found.

    Example:
        >>> audio = generate_speech_like_audio(duration=2.0)
        >>> snr = estimate_snr(audio)
        >>> 'snr_db' in snr
        True
        >>> snr['speech_ratio'] >= 0.0
        True
    """
    if len(audio) == 0:
        return {
            "snr_db": 0.0,
            "signal_power_db": -100.0,
            "noise_power_db": -100.0,
            "speech_ratio": 0.0,
            "n_speech_frames": 0,
            "n_silence_frames": 0,
        }

    float_audio = audio.astype(np.float64)

    # Use VAD to separate speech and silence frames
    vad_results = detect_silence(
        audio,
        sample_rate=sample_rate,
        frame_duration_ms=frame_duration_ms,
        frame_step_ms=frame_step_ms,
        threshold_db=silence_threshold_db,
    )

    if not vad_results:
        return {
            "snr_db": 0.0,
            "signal_power_db": -100.0,
            "noise_power_db": -100.0,
            "speech_ratio": 0.0,
            "n_speech_frames": 0,
            "n_silence_frames": 0,
        }

    # Compute frame-level power from the VAD energy values
    speech_energies = []
    silence_energies = []

    for result in vad_results:
        # Convert dB back to linear power: P = 10^(dB/10)
        linear_power = 10.0 ** (result["energy_db"] / 10.0)
        if result["is_speech"]:
            speech_energies.append(linear_power)
        else:
            silence_energies.append(linear_power)

    n_speech = len(speech_energies)
    n_silence = len(silence_energies)
    total_frames = n_speech + n_silence
    speech_ratio = n_speech / total_frames if total_frames > 0 else 0.0

    epsilon = 1e-10  # Floor to avoid log(0)

    if n_speech == 0:
        # No speech detected — can't compute meaningful SNR
        noise_power = np.mean(silence_energies) if silence_energies else epsilon
        return {
            "snr_db": 0.0,
            "signal_power_db": -100.0,
            "noise_power_db": round(10.0 * np.log10(noise_power + epsilon), 2),
            "speech_ratio": 0.0,
            "n_speech_frames": 0,
            "n_silence_frames": n_silence,
        }

    if n_silence == 0:
        # All speech, no silence reference — estimate noise floor
        # Use the lowest 10% of speech frame energies as noise estimate
        sorted_energies = sorted(speech_energies)
        noise_count = max(1, len(sorted_energies) // 10)
        noise_power = np.mean(sorted_energies[:noise_count])
    else:
        noise_power = np.mean(silence_energies)

    signal_power = np.mean(speech_energies)

    # Ensure noise power is positive for valid SNR computation
    noise_power = max(noise_power, epsilon)
    signal_power = max(signal_power, epsilon)

    snr_db = 10.0 * np.log10(signal_power / noise_power)
    signal_power_db = 10.0 * np.log10(signal_power)
    noise_power_db = 10.0 * np.log10(noise_power)

    return {
        "snr_db": round(float(snr_db), 2),
        "signal_power_db": round(float(signal_power_db), 2),
        "noise_power_db": round(float(noise_power_db), 2),
        "speech_ratio": round(speech_ratio, 4),
        "n_speech_frames": n_speech,
        "n_silence_frames": n_silence,
    }


def apply_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply a first-order pre-emphasis filter to audio.

    Pre-emphasis boosts high frequencies by applying the filter:
    y[n] = x[n] - α * x[n-1]

    This compensates for the natural spectral tilt in speech where
    voiced sounds have a -12 dB/octave falloff. Boosting high frequencies:
    - Improves the signal-to-noise ratio for high-frequency components
    - Balances the spectrum for better FFT-based analysis
    - Makes the spectral features more uniform across frequencies

    The coefficient α (typically 0.95–0.97) controls how much emphasis
    is applied. Higher values produce more high-frequency boost.

    Args:
        audio: 1D numpy array of audio samples.
        coeff: Pre-emphasis coefficient in [0, 1].

    Returns:
        Pre-emphasized audio as numpy array with the same dtype.

    Example:
        >>> import numpy as np
        >>> audio = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        >>> result = apply_preemphasis(audio, coeff=0.97)
        >>> len(result) == len(audio)
        True
        >>> result[0] == 1  # First sample passes through unchanged
        True
    """
    if len(audio) == 0:
        return audio.copy()
    # The first sample has no predecessor, so we keep it as-is
    emphasized = np.append(audio[0], audio[1:] - coeff * audio[:-1])
    return emphasized.astype(audio.dtype)


def normalize_audio(
    audio: np.ndarray,
    target_peak: float = 0.9,
) -> np.ndarray:
    """Normalize audio amplitude to a target peak level.

    Normalization scales the audio so that the maximum absolute sample
    value equals target_peak * 32767 (for int16). This ensures consistent
    loudness across different audio segments.

    Args:
        audio: 1D numpy array of audio samples.
        target_peak: Target peak amplitude as a fraction of the int16 range.
            Should be in (0, 1]. Values close to 1.0 maximize dynamic range
            but leave less headroom for further processing.

    Returns:
        Normalized audio as int16 numpy array. If the input is silent
        (all zeros), returns the input unchanged.

    Example:
        >>> audio = np.array([1000, 2000, -3000], dtype=np.int16)
        >>> normalized = normalize_audio(audio, target_peak=0.9)
        >>> normalized.dtype
        dtype('int16')
    """
    if len(audio) == 0:
        return audio.copy()
    peak = np.max(np.abs(audio))

    if peak == 0:
        # Silent audio — nothing to normalize
        return audio

    # Scale so that peak amplitude = target_peak * full int16 range
    scale_factor = (target_peak * 32767.0) / peak
    scaled = audio * scale_factor

    # Clip to valid int16 range to prevent overflow
    return np.clip(scaled, -32768, 32767).astype(np.int16)
