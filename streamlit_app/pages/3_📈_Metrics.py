import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
st.title("📈 Metrics")
from src.evaluation import compute_wer
col1, col2 = st.columns(2)
with col1: st.metric("WER (Mock)", f"{compute_wer('hello world', 'hello'):.2%}")
with col2: st.metric("Response Time", "~50ms")
