# main.py

import os
import asyncio
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai.errors import APIError

# --- 1. CONFIGURATION ---
# Load environment variables from the .env file
load_dotenv()

# Initialize the Gemini Client using the key from the environment
# The client automatically looks for the GEMINI_API_KEY variable
try:
    gemini_client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    # You might want to exit or handle this error differently
    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# --- 2. Pydantic Data Models ---
# Defines the structure of the data coming IN
class QueryRequest(BaseModel):
    prompt: str
    temperature: float = 0.7 # Default is 0.7
    
# Defines the structure of the data going OUT
class LLMResponse(BaseModel):
    model_name: str
    generated_text: str

# --- 3. API Endpoints ---
@app.get("/")
def read_root():
    """Basic health check endpoint."""
    return {"status": "ok", "message": "LLM API is running and ready."}

@app.post("/api/query", response_model=LLMResponse)
async def handle_llm_query(request_data: QueryRequest):
    """
    Handles the query, calls the Gemini API, and returns the response.
    """
    try:

        print(f"request_data: {request_data}")
        # Call the Gemini API asynchronously
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model='gemini-2.5-flash',
            contents=request_data.prompt,
            config=genai.types.GenerateContentConfig(
                temperature=request_data.temperature
            )
        )
        
        return LLMResponse(
            model_name='gemini-2.5-flash',
            generated_text=response.text
        )

    except APIError as e:
        # Handle specific API errors (e.g., rate limit exceeded)
        return LLMResponse(
            model_name='Error',
            generated_text=f"LLM API Error: {e}"
        )
    except Exception as e:
        # Handle general errors
        return LLMResponse(
            model_name='Error',
            generated_text=f"An unexpected error occurred: {e}"
        )