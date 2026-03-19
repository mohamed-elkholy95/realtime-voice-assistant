<div align="center">

# 🎤 Realtime Voice Assistant

**End-to-end voice assistant** with Whisper STT, intent classification, NLU, and TTS synthesis

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-20%20passed-success?style=flat-square)](#)
[![scipy](https://img.shields.io/badge/scipy-1.11-8CAAE6?style=flat-square)](https://scipy.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square)](https://streamlit.io)

</div>

## Overview

A **real-time voice assistant** implementing the full audio pipeline: speech-to-text (Whisper), natural language understanding (intent classification), response generation, and text-to-speech synthesis. Includes audio processing utilities (MFCC, pre-emphasis, normalization) with mock fallbacks.

## Features

- 🎙️ **Audio Processing** — MFCC extraction, pre-emphasis filtering, normalization
- 📝 **Whisper STT** — OpenAI Whisper integration with mock fallback
- 🧠 **Intent Classification** — Weather, time, music, reminder, and name intents
- 🔊 **TTS Synthesis** — Text-to-speech with duration scaling
- 💬 **Chat Interface** — Conversational UI with history tracking
- 📊 **WER Evaluation** — Word Error Rate computation for STT quality
- ✅ **20 Tests** — Full pipeline coverage

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/realtime-voice-assistant.git
cd realtime-voice-assistant
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run streamlit_app/app.py
```

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
