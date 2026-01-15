"""This class provide LLM configuration and communication"""

import os
import logging
from typing import List, AsyncGenerator

from dotenv import load_dotenv
import google.generativeai as genai
from models import HistoryItem 

logger = logging.getLogger(__name__)

# Load environment variables once at the module level
load_dotenv(override=False)

def load_and_get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return api_key

class LLMService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Use the stable production name unless you are specifically testing 2.0/Experimental
        self.model_name = "gemini-1.5-flash" 
        genai.configure(api_key=self.api_key)
        # Initialize once to save resources
        self.model = genai.GenerativeModel(model_name=self.model_name)

    async def _generate_stream(self, contents: list, temperature: float) -> AsyncGenerator[str, None]:
        """Core private method to handle the actual API communication."""
        try:
            config = genai.types.GenerationConfig(temperature=temperature)
            async for chunk in await self.model.generate_content_async(
                contents=contents,
                stream=True,
                generation_config=config,
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.exception("Gemini API communication failed")
            raise LLMServiceError(
                internal_message="Gemini API call failed",
                public_message="LLM service is temporarily unavailable",
                status_code=502,
                raw_response=str(e),
            ) from e

    async def get_stream_sse(self, prompt: str, temperature: float):
        """Experimental version 1: Standard SSE wrapper."""
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        async for text in self._generate_stream(contents, temperature):
            yield text

    async def get_stream_no_history(self, prompt: str, temperature: float):
        """Experimental version 2: Explicitly ignoring history."""
        # Functionally identical to SSE version for now
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        async for text in self._generate_stream(contents, temperature):
            yield text

    async def get_stream(self, prompt: str, history: List[HistoryItem], temperature: float):
        """Experimental version 3: Full history injection."""
        contents = [
            {"role": item.role, "parts": [{"text": item.text}]}
            for item in history
        ]
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        async for text in self._generate_stream(contents, temperature):
            yield text

class LLMServiceError(Exception):
    def __init__(self, *, internal_message: str, public_message: str, status_code: int, raw_response: str):
        self.internal_message = internal_message
        self.public_message = public_message
        self.status_code = status_code
        self.raw_response = raw_response
        super().__init__(internal_message)

def get_llm_service() -> LLMService:
    try:
        return LLMService(api_key=load_and_get_api_key())
    except Exception as e:
        logger.exception("LLM service configuration failed")
        raise LLMServiceError(
            internal_message="Failed to initialize",
            public_message="Service unavailable",
            status_code=500,
            raw_response=str(e),
        ) from e