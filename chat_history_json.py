# chat_history_json.py

"""This implementation will response in chunks of json object {"json/application", ""}"""

import os
from pathlib import Path
from typing import List
import json

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import google.generativeai as genai
from pydantic import BaseModel


# --- 1. INIT ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], 
    allow_credentials = True,
    allow_methods = ["*"],  
    allow_headers = ["*"],
)

# --- Load .env explicitly ---
ENV_PATH = Path(__file__).parent / ".env"

if not ENV_PATH.exists():
    raise RuntimeError(f".env file not found at: {ENV_PATH}")

load_dotenv(dotenv_path = ENV_PATH)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key :
    raise RuntimeError("GEMINI_API_KEY not found in .env")

genai.configure(api_key = api_key)


# --- Data Model ----

class HistoryItem (BaseModel):
    """Defines a history item object with 'user' or 'model' roles"""
    role:str
    text:str

class Request (BaseModel):
    """Request object"""
    prompt:str
    temperature: float = 0.7
    history: List[HistoryItem]

@app.get("/")
def read_root():
    """Basic health check"""
    return { "status" : "ok", "message": "LLM API is running and ready."}

@app.post("/api/chat/stream/json")
async def query(request_data: Request):

    try:

        contents =[
            *[
                {
                    "role": item.role,
                    "parts": [{"text": item.text}]
                }              
                for item in request_data.history
            ],
            {
                "role":"user",
                "parts": [{"text": request_data.prompt}]
            }
        ]

        model = genai.GenerativeModel(    
            model_name="gemini-2.5-flash",
            # system_instruction="Respond as if you are Yoda."
        )

        async_generator = await model.generate_content_async(
            stream=True,
            generation_config=genai.types.GenerationConfig(
                temperature=request_data.temperature
            ),
            contents = contents
        )
        

        async def stream_response_async():
            async for chunk in async_generator:
                if not hasattr(chunk, "candidates") or not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]

                if not hasattr(candidate, "content") or not candidate.content:
                    continue

                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text is not None:
                        yield (json.dumps({"text": part.text}) + "\n").encode("utf-8")

        return StreamingResponse(stream_response_async(), media_type="application/x-ndjson")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

