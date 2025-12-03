"""This implementation is an example of EventSource
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import google.generativeai as genai
from pydantic import BaseModel


# --- INIT ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], 
    allow_credentials = True,
    allow_methods = ["*"],  
    allow_headers = ["*"],
)

# --- LOAD ENV ---
ENV_PATH = Path(__file__).parent / ".env"

if not ENV_PATH.exists():
    raise RuntimeError(f".env file not found at: {ENV_PATH}")

load_dotenv(dotenv_path = ENV_PATH)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key: 
    raise RuntimeError("GEMINI_API_KEY not found in .env")

genai.configure(api_key = api_key)   

# --- Health Check ---
@app.get("/")
def health_check():
    """Basic health check"""
    return { "status" : "ok", "message": "LLM API is running and ready."}

@app.get("/api/chat/stream/sse")
async def query(prompt: str):

    try:

        model = genai.GenerativeModel(    
            model_name="gemini-2.5-flash",
            # system_instruction="Respond as if you are Yoda."
        )

        async_generator = await model.generate_content_async(
            stream=True,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7 #request_data.temperature
            ),
            contents = [{"role": "user", "parts": [{"text": prompt}]}],
        )


        async def stream_response():
            async for chunk in async_generator:
                if not hasattr(chunk, "candidates") or not chunk.candidates:
                    continue
                print(f"chunk.candidates: {chunk.candidates}")
                candidate = chunk.candidates[0]

                if not hasattr(candidate, "content") or not candidate.content:
                    continue

                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text is not None:
                        payload = json.dumps({"text": part.text})
                        yield f"data: {payload}\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")    

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e