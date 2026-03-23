import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time

import streamlit as st
from src.voice_assistant import VoiceAssistant

st.title("🎤 Voice Chat")
st.markdown("Interact with the voice assistant. Type messages and see intent classification in real time.")

# Initialize assistant in session state for conversation continuity
if "assistant" not in st.session_state:
    st.session_state.assistant = VoiceAssistant()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display conversation history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "intent" in msg:
            col_intent, col_conf = st.columns(2)
            with col_intent:
                st.caption(f"🏷️ Intent: **{msg['intent']['label']}**")
            with col_conf:
                st.caption(
                    f"📊 Confidence: **{msg['intent']['confidence']:.1%}**"
                )
                # Visual confidence bar
                conf = msg["intent"]["confidence"]
                bar_color = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
                st.progress(conf)

# Chat input
if text := st.chat_input("Type a message (voice input simulated)..."):
    assistant = st.session_state.assistant

    # Display user message
    st.chat_message("user").write(text)

    # Process through the assistant pipeline
    start = time.time()
    result = assistant.process_text(text)
    elapsed = (time.time() - start) * 1000

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(result["response_text"])

        # Show intent classification details
        intent = result["intent"]
        conf = intent["confidence"]
        bar_color = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"

        col_intent, col_conf, col_time = st.columns(3)
        with col_intent:
            st.caption(f"🏷️ Intent: **{intent['label']}**")
        with col_conf:
            st.caption(f"{bar_color} Confidence: **{conf:.1%}**")
            st.progress(conf)
        with col_time:
            st.caption(f"⏱️ Latency: **{result['processing_time_ms']:.1f} ms**")

        # Show classifier details in expander
        with st.expander("🔍 Classification Details"):
            st.json({
                "intent": intent["label"],
                "confidence": round(conf, 4),
                "classifier": intent["classifier"],
                "matched_keyword": intent["matched_keyword"],
                "processing_time_ms": result["processing_time_ms"],
            })

    # Add to session history
    st.session_state.chat_history.append({"role": "user", "content": text})
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["response_text"],
        "intent": intent,
    })

# Sidebar: conversation controls
with st.sidebar:
    st.subheader("⚙️ Controls")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.assistant.reset_conversation()
        st.session_state.chat_history.clear()
        st.rerun()

    st.subheader("📊 Session Analytics")
    analytics = st.session_state.assistant.intent_summary
    st.metric("Total Interactions", analytics["total_classifications"])
    st.metric("Unique Intents", analytics["unique_intents"])

    if analytics["intent_distribution"]:
        st.subheader("Intent Distribution")
        for intent_name, proportion in sorted(
            analytics["intent_distribution"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            st.write(f"- **{intent_name}**: {proportion:.1%}")

    # Model status
    st.subheader("🔧 Model Status")
    st.write(f"- STT: {'✅ Loaded' if st.session_state.assistant.stt_loaded else '⚠️ Mock mode'}")
    st.write(f"- TTS: {'✅ Loaded' if st.session_state.assistant.tts_loaded else '⚠️ Mock mode'}")
