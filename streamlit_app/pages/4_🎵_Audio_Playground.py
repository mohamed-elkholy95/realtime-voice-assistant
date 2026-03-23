import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st

from src.audio_processor import (
    generate_sine_wave,
    generate_speech_like_audio,
    compute_mfcc,
    compute_spectrogram,
    compute_log_mel_spectrogram,
    detect_silence,
    apply_preemphasis,
    normalize_audio,
    create_mel_filterbank,
    hz_to_mel,
    mel_to_hz,
)

st.title("🎵 Audio Playground")
st.markdown("Interactive demos for audio signal processing and DSP concepts.")

# ── Signal Generator ─────────────────────────────────────────────────────

st.subheader("📡 Signal Generator")

signal_type = st.selectbox("Waveform Type", ["Sine Wave", "Speech-Like (Multi-harmonic)"])

col_freq, col_dur, col_amp = st.columns(3)
with col_freq:
    frequency = st.slider("Frequency (Hz)", 20, 8000, 440, step=10)
with col_dur:
    duration = st.slider("Duration (s)", 0.1, 5.0, 1.0, step=0.1)
with col_amp:
    amplitude = st.slider("Amplitude", 0.0, 1.0, 0.8, step=0.05)

if signal_type == "Sine Wave":
    audio = generate_sine_wave(duration=duration, freq=frequency, amplitude=amplitude)
else:
    audio = generate_speech_like_audio(duration=duration)

# Plot waveform
st.write("**Time-Domain Waveform**")
st.line_chart(audio[:min(len(audio), 5000)])  # Show first 5000 samples

st.caption(f"Signal: {len(audio):,} samples | dtype: {audio.dtype}")

st.divider()

# ── MFCC Explorer ────────────────────────────────────────────────────────

st.subheader("🌀 MFCC Feature Explorer")

col_n_mfcc, col_n_mels, col_n_fft = st.columns(3)
with col_n_mfcc:
    n_mfcc = st.slider("Number of MFCCs", 1, 26, 13, key="mfcc_n")
with col_n_mels:
    n_mels = st.slider("Mel filter banks", 10, 80, 26, key="mel_n")
with col_n_fft:
    n_fft = st.selectbox("FFT Size", [256, 512, 1024, 2048], index=1, key="fft_n")

mfccs = compute_mfcc(audio, n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft)

st.write(f"**Output shape:** `{mfccs.shape[0]}` coefficients × `{mfccs.shape[1]}` time frames")
st.image(mfccs, use_column_width=True, clamp=True)
st.caption("Heatmap: rows = MFCC coefficients, columns = time frames. Color intensity = coefficient magnitude.")

# Show coefficient statistics
st.write("**Coefficient Statistics:**")
mfcc_stats = []
for i in range(min(n_mfcc, mfccs.shape[0])):
    mfcc_stats.append({
        "Coefficient": f"MFCC-{i}",
        "Mean": f"{np.mean(mfccs[i]):.4f}",
        "Std": f"{np.std(mfccs[i]):.4f}",
        "Min": f"{np.min(mfccs[i]):.4f}",
        "Max": f"{np.max(mfccs[i]):.4f}",
    })
st.dataframe(mfcc_stats, use_container_width=True, hide_index=True)

st.divider()

# ── Mel Filter Bank Visualization ────────────────────────────────────────

st.subheader("🎷 Mel Filter Bank")

mel_fb = create_mel_filterbank(
    n_filters=n_mels, n_fft_size=n_fft, sample_rate=16000
)

st.write(f"**Filter bank shape:** `{mel_fb.shape[0]}` filters × `{mel_fb.shape[1]}` frequency bins")
st.image(mel_fb, use_column_width=True, clamp=True)
st.caption("Each row is a triangular filter. Notice the filters are spaced closer together at low frequencies (Mel scale).")

st.divider()

# ── Spectrogram Comparison ────────────────────────────────────────────────

st.subheader("📊 Spectrogram Comparison")

spec_result = compute_spectrogram(audio, n_fft=n_fft)
log_mel = compute_log_mel_spectrogram(audio, n_mels=n_mels, n_fft=n_fft)

col_spec, col_mel = st.columns(2)

with col_spec:
    st.write("**Linear Spectrogram (STFT + dB)**")
    st.image(spec_result["spectrogram"], use_column_width=True, clamp=True)
    st.caption(f"{spec_result['spectrogram'].shape[0]} freq bins × {spec_result['spectrogram'].shape[1]} frames")

with col_mel:
    st.write("**Log-MEL Spectrogram (Whisper-style)**")
    st.image(log_mel, use_column_width=True, clamp=True)
    st.caption(f"{log_mel.shape[0]} Mel bins × {log_mel.shape[1]} frames")

st.info(
    "💡 **Key Difference:** The linear spectrogram shows uniform frequency resolution, "
    "while the Mel spectrogram compresses higher frequencies (matching human hearing). "
    "This is why Mel features are so effective for speech processing."
)

st.divider()

# ── Pre-emphasis & Normalization ────────────────────────────────────────

st.subheader("🎛️ Audio Effects")

effects_col1, effects_col2 = st.columns(2)

with effects_col1:
    st.write("**Pre-emphasis Filter**")
    pre_coeff = st.slider("Coefficient α", 0.0, 1.0, 0.97, step=0.01, key="preemph")
    preemphasized = apply_preemphasis(audio, coeff=pre_coeff)
    st.line_chart(preemphasized[:min(len(preemphasized), 5000)])

    col_orig, col_pre = st.columns(2)
    with col_orig:
        st.metric("Original RMS", f"{np.sqrt(np.mean(np.float64(audio)**2)):.2f}")
    with col_pre:
        st.metric("Pre-emph RMS", f"{np.sqrt(np.mean(np.float64(preemphasized)**2)):.2f}")

with effects_col2:
    st.write("**Normalization**")
    target_peak = st.slider("Target Peak", 0.1, 1.0, 0.9, step=0.05, key="norm_peak")
    normalized = normalize_audio(audio, target_peak=target_peak)
    st.line_chart(normalized[:min(len(normalized), 5000)])

    col_orig2, col_norm = st.columns(2)
    with col_orig2:
        st.metric("Original Peak", f"{np.max(np.abs(audio))}")
    with col_norm:
        st.metric("Normalized Peak", f"{np.max(np.abs(normalized))}")

st.divider()

# ── Hz ↔ Mel Converter ──────────────────────────────────────────────────

st.subheader("🔄 Hz ↔ Mel Converter")

hz_input = st.number_input("Frequency (Hz):", value=1000, min_value=0, step=100)
mel_value = hz_to_mel(hz_input)
recovered_hz = mel_to_hz(mel_value)

col_hz, col_mel, col_back = st.columns(3)
col_hz.metric("Input (Hz)", f"{hz_input:.0f}")
col_mel.metric("Mel Scale", f"{mel_value:.1f}")
col_back.metric("Roundtrip (Hz)", f"{recovered_hz:.1f}")

st.caption(
    "The Mel scale is perceptual: equal distances sound equally distant to humans. "
    "Formula: mel = 2595 × log₁₀(1 + f/700)"
)

# Mel scale chart
mel_range = np.linspace(0, 8000, 200)
mel_values = [hz_to_mel(f) for f in mel_range]
chart_data = {"Frequency (Hz)": mel_range, "Mel Scale": mel_values}
st.line_chart(chart_data, x="Frequency (Hz)", y="Mel Scale")
