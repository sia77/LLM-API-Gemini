"""
Main application file for the LLM Query API.

This module initializes the FastAPI application, loads environment variables,
sets up the CORS middleware, and defines the primary /api/query endpoint
for interacting with the Gemini model.
"""

import os
import asyncio

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel


from dotenv import load_dotenv
import google.generativeai as genai


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

load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

genai.configure(api_key = api_key)

# --- 2. DATA MODELS ---
class QueryRequest(BaseModel):
    """Request object"""
    prompt: str
    temperature: float = 0.7


# class LLMResponse(BaseModel):
#     model_name: str
#     generated_text: str


# --- 3. ENDPOINTS ---
@app.get("/")
def read_root():
    """Basic health check endpoint."""
    return {"status": "ok", "message": "LLM API is running and ready."}


@app.post("/api/query")
async def query(request_data: QueryRequest):
    """
    Handles the query, calls the Gemini API, and returns the response as stream chunks.
    """
    try:
        model = genai.GenerativeModel(    
            model_name="gemini-2.5-flash",
            # system_instruction="Respond as if you are Yoda."
        )

        # streaming generator using the current SDK
        def stream_response():
            for chunk in model.generate_content(
                contents = request_data.prompt,
                generation_config = genai.types.GenerationConfig(
                    temperature = request_data.temperature
                ),
                
                stream = True  # enable streaming here
            ):
                if chunk.text:
                    #print(f"chunk.text: {chunk.text}")
                    yield chunk.text.encode("utf-8")

        async def async_stream():
            chunk = stream_response()
            for text in chunk:
                #print(f"text: {text}")
                yield text
                await asyncio.sleep(0)

        return StreamingResponse(async_stream(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
