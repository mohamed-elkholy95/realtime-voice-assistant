"""FastAPI for voice assistant."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
logger = logging.getLogger(__name__)
app = FastAPI(title="Voice Assistant API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class ChatResponse(BaseModel):
    response: str; stt_loaded: bool; tts_loaded: bool

@app.get("/health")
async def health(): return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    from src.voice_assistant import VoiceAssistant
    va = VoiceAssistant()
    result = va._generate_response(req.text)
    return ChatResponse(response=result, stt_loaded=va.stt_loaded, tts_loaded=va.tts_loaded)

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8008)
