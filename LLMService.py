"""This class provide LLM configuration and communication"""

import os
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

from models import HistoryItem 


def load_and_get_api_key():
    """Loads environment variables, and returns api_key"""
    ENV_PATH = Path(__file__).parent / ".env"
    if not ENV_PATH.exists():
        raise RuntimeError(f".env file doesn't exist on this path: {ENV_PATH}")
    
    load_dotenv(dotenv_path = ENV_PATH)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")
    
    return api_key


class LLMService:
    """A service class to manage interaction with the Gemini API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model_name = "gemini-2.5-flash"
        genai.configure(api_key=self.api_key)

    
    async def get_stream_sse(self, prompt:str, temperature:float):
        """
        Handles the logic for calling the streaming Gemini API using SSE.
        """
        try:
            contents = [{"role": "user", "parts": [{"text": prompt}]}]

            model = genai.GenerativeModel(    
                model_name=self.model_name,
            )

            async for chunk in await model.generate_content_async(
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                ),
                contents = contents,
            ):
                yield chunk.text
        except Exception as e:
            print(f"Gemini API Communication Error: {e}")
            raise LLMServiceError(f"Failed to communicate with the LLM API: {e}") from e
        

    async def get_stream_no_history(self, prompt:str, temperature:float):
        """
        Handles the logic for calling the streaming Gemini API, but doesn't include the existing history - It was meant as a test
        """

        try:

            contents = [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]

            model = genai.GenerativeModel(    
                model_name = self.model_name,
            )

            async for chunk in await model.generate_content_async(
                stream = True,
                generation_config = genai.types.GenerationConfig(
                    temperature = temperature
                ),
                contents= contents
            ):
                yield chunk.text
                
        except Exception as e:
            print(f"Gemini API Communication Error: {e}")
            raise LLMServiceError(f"Failed to communicate with the LLM API: {e}") from e





    async def get_stream(self, prompt: str, history: list[HistoryItem], temperature: float):
        """
        Handles the logic for calling the streaming Gemini API.
        """
        try:

            contents = [
                *[
                    {
                        "role": item.role,
                        "parts": [{"text": item.text}]
                    }
                    for item in history
                ],
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]

            model = genai.GenerativeModel(    
                model_name=self.model_name,
            )

            async for chunk in await model.generate_content_async(
                stream = True,
                generation_config = genai.types.GenerationConfig(
                    temperature = temperature
                ),
                contents= contents
            ):
                yield chunk.text
                    
                    # if not hasattr(chunk, "candidates") or not chunk.candidates:
                    #     continue

                    # candidate = chunk.candidates[0]

                    # if not hasattr(candidate, "content") or not candidate.content:
                    #     continue

                    # for part in candidate.content.parts:
                    #     if hasattr(part, "text") and part.text is not None:
                    #         yield chunk.text

             
        except Exception as e:
            print(f"Gemini API Communication Error: {e}")
            raise LLMServiceError(f"Failed to communicate with the LLM API: {e}") from e
        
class LLMServiceError(Exception):
    """Custom exception raised when the LLM service fails to communicate 
    or returns an unexpected non-standard error."""


def get_llm_service() -> LLMService:
    """
    FastAPI Dependency Provider.
    Loads the API key and returns a configured LLMService instance.
    """
    try:
        # 1. Load the necessary configuration (API key)
        api_key = load_and_get_api_key()
        
        # 2. Instantiate and return the service object
        return LLMService(api_key=api_key)
    
    except RuntimeError as e:
        # If config/env loading fails, raise an exception that will 
        # result in a 500 error before the request even hits the endpoint.
        # It's okay to raise a standard exception here, or a custom one.
        raise Exception(f"Configuration Error: {e}") from e
