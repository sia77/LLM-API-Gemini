"""This class provide LLM configuration and communication"""

from functools import lru_cache

import logging
from typing import List, AsyncGenerator

import google.generativeai as genai
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr
from models import HistoryItem 

logger = logging.getLogger(__name__)

# --- 1. Configuration Layer (Pydantic Settings) ---

class Settings(BaseSettings):
    """
    Centralized configuration. Pydantic handles validation and .env loading.
    If GEMINI_API_KEY is missing, this will raise an error immediately.
    """
    gemini_api_key: SecretStr = Field(..., alias="GEMINI_API_KEY")
    # You can easily swap models for experiments here
    gemini_model: str = Field(default="gemini-2.5-flash-lite")
    
    # Standard Pydantic V2 way to link a .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    """Returns a cached version of settings so .env is only read once."""
    return Settings() # type: ignore

# --- 2. Error Handling ---

class LLMServiceError(Exception):
    def __init__(self, *, internal_message: str, public_message: str, status_code: int, raw_response: str):
        self.internal_message = internal_message
        self.public_message = public_message
        self.status_code = status_code
        self.raw_response = raw_response
        super().__init__(internal_message)

# --- 3. The LLM Service ---

class LLMService:
    def __init__(self, api_key: str, model_name: str):
        # Configuration happens once when the class is instantiated
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
        logger.info(f"LLMService initialized with model: {model_name}")

    async def _generate_stream(self, contents: list, temperature: float) -> AsyncGenerator[str, None]:
        """Private engine ensures all experimental methods share the same logic/error handling."""
        try:
            config = genai.types.GenerationConfig(temperature=temperature)
            # Standard async streaming call
            response = await self.model.generate_content_async(
                contents=contents,
                stream=True,
                generation_config=config,
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.exception("Gemini API communication failed")
            raise LLMServiceError(
                internal_message=f"Gemini API failure: {str(e)}",
                public_message="The AI service is currently unavailable.",
                status_code=502,
                raw_response=str(e),
            ) from e

    # --- Experimental Wrappers ---

    async def get_stream_sse(self, prompt: str, temperature: float):
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        async for text in self._generate_stream(contents, temperature):
            yield text

    async def get_stream_no_history(self, prompt: str, temperature: float):
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        async for text in self._generate_stream(contents, temperature):
            yield text

    async def get_stream(self, prompt: str, history: List[HistoryItem], temperature: float):
        contents = [{"role": item.role, "parts": [{"text": item.text}]} for item in history]
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        async for text in self._generate_stream(contents, temperature):
            yield text


# --- 4. The Singleton Provider (FastAPI Dependency) ---

@lru_cache
def get_llm_service() -> LLMService:
    """
    This is the "Magic" function. 
    @lru_cache ensures that LLMService is only instantiated ONCE.
    """
    settings = get_settings()
    return LLMService(
        api_key=settings.gemini_api_key.get_secret_value(), 
        model_name=settings.gemini_model
    )