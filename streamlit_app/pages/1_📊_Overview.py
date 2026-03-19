import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
st.title("🎤 Realtime Voice Assistant")
st.markdown("End-to-end voice assistant with STT, NLU, and TTS pipelines.")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Pipeline"); st.markdown("🎤 Audio Input → 📝 STT (Whisper) → 🧠 NLU → 🔊 TTS → Audio Output")
with col2:
    st.subheader("Components"); st.markdown("- Whisper STT (with mock fallback)\n- Intent classification\n- Template-based responses\n- TTS synthesis")
