import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.title("🎤 Realtime Voice Assistant")
st.markdown(
    "An end-to-end voice assistant demonstrating the full speech processing pipeline: "
    "**Audio Capture → STT → Intent Classification → Response Generation → TTS**"
)

# ── Architecture Diagram ──────────────────────────────────────────────────

st.subheader("🏗️ System Architecture")

st.markdown("""
```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐
│   🎙️ Audio   │───▶│  📝 STT      │───▶│ 🧠 Intent Class. │───▶│ 💬 Response │───▶│  🔊 TTS      │
│   Capture     │    │  (Whisper)   │    │ (Keyword+Regex)  │    │ Generation  │    │  Synthesis   │
└──────────────┘    └──────────────┘    └──────────────────┘    └──────────────┘    └──────────────┘
       │                    │                      │                      │                    │
       │              ┌─────┴─────┐          ┌────┴────┐         ┌─────┴─────┐        ┌────┴────┐
       │              │ MFCC      │          │ Greeting│         │ Template  │        │ Coqui   │
       │              │ Log-Mel   │          │ Weather │         │ Filling   │        │ Vocoder │
       │              │ Pre-emph  │          │ Music   │         │ Context   │        │ Pitch   │
       │              └───────────┘          │ Time    │         └───────────┘        │ Rate    │
       │                                     │ ...     │                              └─────────┘
       │                                     └─────────┘
```
""")

# ── Pipeline Stages ──────────────────────────────────────────────────────

st.subheader("🔄 Pipeline Stages")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1. Speech-to-Text (STT)")
    st.markdown("""
    **Model:** OpenAI Whisper (with mock fallback)

    - Converts audio → text using encoder-decoder Transformer
    - Feature extraction: audio → log-Mel spectrogram
    - Supports 99 languages with automatic detection
    - Performance: Tiny model runs in <1s on CPU
    """)

with col2:
    st.markdown("### 2. Intent Classification")
    st.markdown("""
    **Method:** Keyword + Regex pipeline with fusion

    - Keyword matching with fuzzy similarity (SequenceMatcher)
    - Regex patterns for structured inputs
    - Multi-classifier fusion with confidence scoring
    - 10 supported intents: greeting, weather, time, music, etc.
    """)

with col3:
    st.markdown("### 3. Response Generation → TTS")
    st.markdown("""
    **Response:** Template-based with slot filling
    **TTS:** Coqui Tacotron2-DDC (with mock fallback)

    - Context-aware multi-turn dialogue tracking
    - Intent analytics for usage patterns
    - Speech rate and pitch control
    - SSML support (stub)
    """)

# ── Component Cards ───────────────────────────────────────────────────────

st.subheader("📦 Components")

components = {
    "src/audio_processor.py": [
        "Real MFCC extraction (DFT → Mel → log → DCT)",
        "STFT spectrogram with Hanning window",
        "Energy-based Voice Activity Detection (VAD)",
        "Log-MEL spectrogram (Whisper-style)",
        "Speech segment extraction",
    ],
    "src/stt_engine.py": [
        "Whisper integration with mock fallback",
        "Audio preprocessing (normalize, trim silence)",
        "Batch transcription support",
        "Language detection stub",
        "Confidence scoring",
    ],
    "src/intent_classifier.py": [
        "KeywordRuleClassifier with fuzzy matching",
        "RegexPatternClassifier for structured input",
        "IntentClassifierPipeline with weighted fusion",
        "10 intent types supported",
        "Batch classification",
    ],
    "src/voice_assistant.py": [
        "End-to-end pipeline orchestration",
        "Conversation context tracking",
        "Intent analytics collection",
        "Template-based response generation",
        "Processing time benchmarking",
    ],
    "src/evaluation.py": [
        "Real WER via Levenshtein dynamic programming",
        "Character Error Rate (CER)",
        "Intent classification accuracy",
        "Latency benchmarking",
        "Markdown report generation",
    ],
    "src/tts_engine.py": [
        "Coqui TTS integration with mock fallback",
        "Speech rate control (0.25x–4.0x)",
        "Pitch scaling",
        "SSML support stub",
        "Audio format conversion (int16 ↔ float32)",
    ],
    "src/api/main.py": [
        "7 REST endpoints (health, chat, transcribe, synthesize, intents, evaluate, history)",
        "Pydantic request/response validation",
        "Singleton assistant instance",
        "Analytics endpoint",
    ],
}

for filepath, features in components.items():
    with st.expander(f"📄 `{filepath}`"):
        for feature in features:
            st.markdown(f"- {feature}")

# ── Quick Links ───────────────────────────────────────────────────────────

st.subheader("🚀 Quick Start")

st.code("""
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Start the API server
uvicorn src.api.main:app --reload --port 8008

# Launch the Streamlit dashboard
streamlit run streamlit_app/app.py
""", language="bash")
