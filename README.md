<div align="center">

# 🎤 Realtime Voice Assistant

**End-to-end voice assistant** with Whisper STT, intent classification, NLU, and TTS synthesis — built for learning and portfolio demonstration.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-100%2B%20passed-success?style=flat-square)](#)
[![scipy](https://img.shields.io/badge/scipy-1.11-8CAAE6?style=flat-square)](https://scipy.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square)](https://streamlit.io)

</div>

## 🎯 What You'll Learn

This project demonstrates core speech processing concepts with **real implementations** (not just mocks):

- **DSP Fundamentals**: FFT, STFT, Mel filter banks, DCT, pre-emphasis, SNR estimation
- **Feature Extraction**: Real MFCC computation using scipy (DFT → Mel → log → DCT), plus delta and delta-delta temporal features
- **Voice Activity Detection**: Energy-based silence detection and speech segmentation
- **Intent Classification**: Multi-strategy pipeline (keyword + regex + fuzzy matching)
- **Speech Recognition**: Whisper integration with audio preprocessing
- **Speech Synthesis**: TTS with rate control, pitch scaling, and SSML stubs
- **Evaluation Metrics**: WER via Levenshtein dynamic programming, CER, intent accuracy, confusion matrix with per-class precision/recall/F1
- **Streaming ASR**: Chunk-based transcription with overlap handling for real-time audio streams
- **Audio Quality**: Signal-to-Noise Ratio estimation using VAD-based speech/silence segmentation
- **API Design**: RESTful API with Pydantic validation, token bucket rate limiting, and 9 endpoints
- **Visualization**: Interactive Streamlit dashboard with 5 pages

> 📖 New to speech processing? Check out the **[Glossary](GLOSSARY.md)** for definitions of all key terms used in this project.

## 🏗️ Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐
│   🎙️ Audio   │───▶│  📝 STT      │───▶│ 🧠 Intent Class. │───▶│ 💬 Response │───▶│  🔊 TTS      │
│   Capture     │    │  (Whisper)   │    │ (Keyword+Regex)  │    │ Generation  │    │  Synthesis   │
└──────────────┘    └──────────────┘    └──────────────────┘    └──────────────┘    └──────────────┘
       │                    │                      │                      │                    │
       │              ┌─────┴─────┐          ┌────┴────┐         ┌─────┴─────┐        ┌────┴────┐
       │              │ MFCC      │          │ 10      │         │ Template  │        │ Pitch   │
       │              │ Log-Mel   │          │ intents │         │ Filling   │        │ Rate    │
       │              │ Pre-emph  │          │ Fuzzy   │         │ Context   │        │ SSML    │
       │              │ VAD       │          │ Fusion  │         │ History   │        │ Vocoder │
       │              └───────────┘          └─────────┘         └───────────┘        └─────────┘
```

## 📊 How MFCCs Work

MFCCs (Mel-Frequency Cepstral Coefficients) are the dominant feature representation in speech processing. Here's the computation pipeline:

```
Audio Waveform
     │
     ▼
[Pre-emphasis]  y[n] = x[n] - α·x[n-1]   ← Boost high frequencies
     │
     ▼
[Framing]       25ms windows, 10ms hop     ← Short, stationary segments
     │
     ▼
[Hanning Window] Taper edges to zero        ← Reduce spectral leakage
     │
     ▼
[FFT]           Time → Frequency domain     ← Reveal spectral content
     │
     ▼
[Mel Filter Bank] 26 triangular filters     ← Match human hearing (Mel scale)
     │
     ▼
[Log]           log(filter energies)        ← Model loudness perception
     │
     ▼
[DCT]           Decorrelate features        ← Separate source from filter
     │
     ▼
MFCCs (13 coefficients per frame)          ← Vocal tract shape = phoneme identity
```

**Why MFCCs?** The DCT step is the key insight — it performs *cepstral analysis*, which separates the vocal tract filter (what we want for speech recognition) from the excitation source (pitch, which we don't want). Lower coefficients capture the spectral envelope (vocal tract shape), while higher coefficients capture fine pitch details.

## 📁 Project Structure

```
08-realtime-voice-assistant/
├── src/
│   ├── config.py              # Dataclass-based config with validation & env vars
│   ├── audio_processor.py     # Real DSP: MFCC, STFT, Mel filter bank, VAD
│   ├── stt_engine.py          # Whisper STT with preprocessing & batch support
│   ├── tts_engine.py          # TTS with rate/pitch control & SSML stubs
│   ├── intent_classifier.py   # Multi-strategy intent classification pipeline
│   ├── voice_assistant.py     # Pipeline orchestrator with context & analytics
│   ├── evaluation.py          # WER (Levenshtein), CER, accuracy, benchmarking
│   └── api/
│       └── main.py            # FastAPI with 7 endpoints & Pydantic models
├── streamlit_app/
│   ├── app.py                 # Dashboard entry point (5 pages)
│   └── pages/
│       ├── 1_📊_Overview.py           # Architecture diagram & component cards
│       ├── 2_🎤_Chat.py               # Chat with intent display & confidence bars
│       ├── 3_📈_Metrics.py            # WER/CER calculator & audio analysis
│       ├── 4_🎵_Audio_Playground.py    # Interactive DSP demos & signal generator
│       └── 5_📚_Learn.py              # Educational deep-dives
├── tests/
│   ├── conftest.py            # Shared fixtures (audio signals, mock engines)
│   ├── test_audio_processor.py  # 45+ tests: MFCC, spectrogram, VAD, Mel filter
│   ├── test_stt.py            # STT engine, preprocessing, batch transcription
│   ├── test_tts.py            # TTS synthesis, rate/pitch, format conversion
│   ├── test_intent_classifier.py  # All classifiers, fuzzy matching, pipeline
│   ├── test_voice_assistant.py  # Pipeline, history, context, analytics
│   ├── test_evaluation.py     # WER, CER, Levenshtein, accuracy, benchmarks
│   └── test_api.py            # All 7 API endpoints, validation, errors
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/mohamed-elkholy95/realtime-voice-assistant.git
cd realtime-voice-assistant

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the test suite (100+ tests)
python -m pytest tests/ -v

# Start the API server
uvicorn src.api.main:app --reload --port 8008
# API docs: http://localhost:8008/docs

# Launch the Streamlit dashboard
streamlit run streamlit_app/app.py
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Process text through the assistant pipeline |
| `POST` | `/transcribe` | Convert base64 audio to text (STT) |
| `POST` | `/synthesize` | Convert text to base64 audio (TTS) |
| `GET` | `/intents` | List all supported intent types |
| `POST` | `/evaluate` | Compute WER/CER between reference and hypothesis |
| `GET` | `/history` | Get conversation history |
| `GET` | `/analytics` | Get intent classification analytics |

### Example API Usage

```bash
# Chat
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the weather today?"}'

# Evaluate WER
curl -X POST http://localhost:8008/evaluate \
  -H "Content-Type: application/json" \
  -d '{"reference": "hello world", "hypothesis": "hello there"}'
```

## 📈 Streamlit Dashboard

The interactive dashboard has 5 pages:

1. **📊 Overview** — Architecture diagram, pipeline visualization, component cards
2. **🎤 Chat** — Voice chat with intent display, confidence bars, session analytics
3. **📈 Metrics** — Interactive WER/CER calculator, audio analysis, spectrograms
4. **🎵 Audio Playground** — Signal generator, MFCC explorer, Mel filter bank visualizer, pre-emphasis & normalization demos
5. **📚 Learn** — Educational deep-dives into STT, MFCCs, intent classification, TTS, and WER

## 🧪 Testing

The project has **100+ tests** covering all components:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test module
python -m pytest tests/test_audio_processor.py -v

# Run specific test
python -m pytest tests/test_evaluation.py::TestWordErrorRate -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for your changes
4. Ensure all tests pass: `python -m pytest tests/ -v`
5. Commit with conventional commits: `feat: add new feature`
6. Push and open a Pull Request

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
