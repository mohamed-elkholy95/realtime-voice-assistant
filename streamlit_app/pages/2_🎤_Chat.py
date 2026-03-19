import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
from src.voice_assistant import VoiceAssistant
st.title("🎤 Voice Chat")
if "history" not in st.session_state: st.session_state.history = []
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])
if text := st.chat_input("Type a message (voice input simulated)..."):
    va = VoiceAssistant()
    result = va._generate_response(text)
    st.session_state.history.append({"role": "user", "content": text})
    st.session_state.history.append({"role": "assistant", "content": result})
    st.chat_message("user").write(text)
    st.chat_message("assistant").write(result)
