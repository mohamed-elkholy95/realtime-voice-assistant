import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st

from src.evaluation import compute_wer, compute_cer, generate_report
from src.audio_processor import (
    generate_sine_wave,
    generate_speech_like_audio,
    compute_mfcc,
    compute_spectrogram,
    detect_silence,
)

st.title("📈 Metrics & Evaluation")
st.markdown("Explore evaluation metrics for the voice assistant pipeline.")

# ── WER/CER Interactive Demo ─────────────────────────────────────────────

st.subheader("🔤 Word Error Rate (WER) Calculator")

col_ref, col_hyp = st.columns(2)
with col_ref:
    reference = st.text_area("Reference (ground truth):", value="the quick brown fox jumps over the lazy dog")
with col_hyp:
    hypothesis = st.text_area("Hypothesis (ASR output):", value="the quick brown fox jumped over the lazy dog")

if reference and hypothesis:
    wer = compute_wer(reference, hypothesis)
    cer = compute_cer(reference, hypothesis)

    col_wer, col_cer, col_status = st.columns(3)
    with col_wer:
        color = "🟢" if wer < 0.1 else "🟡" if wer < 0.25 else "🔴"
        st.metric(f"{color} WER", f"{wer:.2%}")
    with col_cer:
        color = "🟢" if cer < 0.1 else "🟡" if cer < 0.25 else "🔴"
        st.metric(f"{color} CER", f"{cer:.2%}")
    with col_status:
        ref_words = len(reference.split())
        hyp_words = len(hypothesis.split())
        st.metric("Word Count", f"{ref_words} → {hyp_words}")

    # Interpretation
    if wer == 0.0:
        st.success("✅ Perfect match! No errors detected.")
    elif wer < 0.1:
        st.info("🟢 Excellent! WER is below 10% (human-level for clean speech).")
    elif wer < 0.25:
        st.warning("🟡 Acceptable. WER is in the 10-25% range.")
    else:
        st.error("🔴 High error rate. WER exceeds 25%.")

st.divider()

# ── STT Quality Benchmarks ───────────────────────────────────────────────

st.subheader("📊 STT Quality Benchmarks")

benchmark_data = [
    ("Perfect Match", "hello world", "hello world"),
    ("1 Substitution", "hello world", "hello there"),
    ("1 Insertion", "hello", "hello world"),
    ("1 Deletion", "hello world", "hello"),
    ("Multiple Errors", "the cat sat on the mat", "the dog sat on a hat"),
    ("Complete Mismatch", "good morning everyone", "night goodbye nobody"),
]

results = []
for label, ref, hyp in benchmark_data:
    w = compute_wer(ref, hyp)
    c = compute_cer(ref, hyp)
    results.append({
        "Test Case": label,
        "Reference": ref,
        "Hypothesis": hyp,
        "WER": f"{w:.2%}",
        "CER": f"{c:.2%}",
    })

st.dataframe(results, use_container_width=True, hide_index=True)

st.divider()

# ── Audio Analysis Demo ──────────────────────────────────────────────────

st.subheader("🎵 Audio Signal Analysis")

signal_type = st.radio("Signal Type:", ["Sine Wave", "Speech-Like"], horizontal=True)

if signal_type == "Sine Wave":
    col_freq, col_dur = st.columns(2)
    with col_freq:
        frequency = st.slider("Frequency (Hz)", 20, 4000, 440, step=10)
    with col_dur:
        duration = st.slider("Duration (s)", 0.1, 5.0, 1.0, step=0.1)
    audio = generate_sine_wave(duration=duration, freq=frequency)
else:
    duration = st.slider("Duration (s)", 0.5, 5.0, 2.0, step=0.5)
    audio = generate_speech_like_audio(duration=duration)

# Basic stats
spec_info = compute_spectrogram(audio)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Duration", f"{spec_info['duration']:.3f}s")
col2.metric("Samples", f"{spec_info['length']:,}")
col3.metric("Peak Amplitude", f"{spec_info['max_amplitude']}")
col4.metric("RMS Energy", f"{spec_info['rms']:.4f}")

# MFCC visualization
st.subheader("MFCC Features")
mfccs = compute_mfcc(audio, n_mfcc=13)

col_mfcc, mel_spec_col = st.columns(2)
with col_mfcc:
    st.write(f"MFCC shape: {mfccs.shape[0]} coefficients × {mfccs.shape[1]} frames")
    st.caption("First 13 Mel-frequency cepstral coefficients (time → color)")
    st.image(mfccs, use_column_width=True, clamp=True)

# Spectrogram
with mel_spec_col:
    st.write(f"Spectrogram shape: {spec_info['spectrogram'].shape[0]} bins × {spec_info['spectrogram'].shape[1]} frames")
    st.caption("Short-Time Fourier Transform magnitude (dB)")
    st.image(spec_info["spectrogram"], use_column_width=True, clamp=True)

# VAD
st.subheader("🎙️ Voice Activity Detection")
vad_results = detect_silence(audio)
speech_frames = sum(1 for r in vad_results if r["is_speech"])
total_frames = len(vad_results) if vad_results else 1
speech_pct = speech_frames / total_frames * 100

col_speech, col_silence = st.columns(2)
col_speech.metric("Speech Frames", f"{speech_frames}/{total_frames} ({speech_pct:.1f}%)")
col_silence.metric("Silence Frames", f"{total_frames - speech_frames}/{total_frames}")

st.divider()

# ── Report Generation ────────────────────────────────────────────────────

st.subheader("📋 Generate Evaluation Report")

if st.button("Generate Markdown Report", use_container_width=True):
    metrics = {
        "wer": wer if reference and hypothesis else 0.0,
        "cer": cer if reference and hypothesis else 0.0,
        "avg_latency_ms": 45.2,
        "intent_accuracy": 0.92,
        "total_tests": 100,
    }
    report = generate_report(metrics)
    st.code(report, language="markdown")
    st.download_button(
        "Download Report",
        report,
        file_name="evaluation_report.md",
        mime="text/markdown",
    )
