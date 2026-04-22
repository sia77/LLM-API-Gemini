import asyncio
from functools import lru_cache
import logging
from typing import Any, Dict, List, AsyncGenerator, Optional

from google import genai
from google.genai import types
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr
from models import HistoryItem 

logger = logging.getLogger(__name__)

# --- 1. Configuration Layer (STAYS UNCHANGED) ---

class Settings(BaseSettings):
    gemini_api_key: SecretStr = Field(..., alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash-lite")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    """Remains cached to avoid redundant .env reads."""
    return Settings() 

# --- 2. Error Handling ---

class LLMServiceError(Exception):
    def __init__(self, *, internal_message: str, public_message: str, status_code: int, raw_response: str):
        self.internal_message = internal_message
        self.public_message = public_message
        self.status_code = status_code
        self.raw_response = raw_response
        super().__init__(internal_message)

# --- 3. The Modernized LLM Service ---

class LLMService:
    def __init__(self, api_key: str, default_model: str):
        # We initialize the Client once per service instance
        self.client = genai.Client(api_key=api_key)
        self.default_model = default_model
        logger.info(f"LLMService initialized with default model: {default_model}")

    async def list_available_models(self) -> List[dict]:
        """GET implementation: Fetches models dynamically from the API."""
        try:
            # .aio is the modern namespace for all async operations
            models_pager = await self.client.aio.models.list()
            return [
                {"value": m.name, "label": m.display_name}
                for m in models_pager if 'generateContent' in m.supported_actions
            ]
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return []

    async def _core_generator(
        self, 
        contents: list, 
        temperature: float, 
        model_override: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Private engine for all streaming/SSE methods."""
        try:
            # Use the frontend's choice, or fall back to the .env default
            print(f"default: {self.default_model}")
            target_model = model_override or self.default_model
            config = types.GenerateContentConfig(temperature=temperature)

            async for chunk in await self.client.aio.models.generate_content_stream(
                model=target_model,
                contents=contents,
                config=config
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.exception("Gemini API communication failed")
            # You can keep your custom LLMServiceError here
            raise e

    # --- Your Required Methods ---

    async def get_complete(
        self, 
        prompt: str, 
        temperature: float,
        model_name: Optional[str] = None,
        history: Optional[List[HistoryItem]] = None
    ) -> str:
        """Full response implementation."""
        contents = self._prepare_contents(prompt, history)
        config = types.GenerateContentConfig(temperature=temperature)
        
        response = await self.client.aio.models.generate_content(
            model=model_name or self.default_model,
            contents=contents,
            config=config
        )
        return response.text or "No response generated."

    async def get_stream(
        self, 
        prompt: str,
        temperature: float, 
        model_name: Optional[str] = None,
        history: Optional[List[HistoryItem]] = None
    ) -> AsyncGenerator[str, None]:
        """Standard chunk streaming."""
        contents = self._prepare_contents(prompt, history)
        async for text in self._core_generator(contents, temperature, model_name):
            yield text

    async def get_stream_sse(
        self, 
        prompt: str, 
        temperature: float, 
        model_name: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """SSE formatted streaming."""
        async for text in self.get_stream(prompt, temperature, model_name=model_name):
            yield f"data: {text}\n\n"

    def _prepare_contents(self, prompt: str, history: Optional[List[HistoryItem]]) -> list:
        contents = []
        if history:
            contents.extend({"role": item.role, "parts": [{"text": item.text}]} for item in history)
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        return contents
    
    async def ping_model_by_id(self, model_id:str) -> Optional[str]:
        """
            The 'Functional' coroutine: Tries to generate 1 token.
            If it hits a 403 (Permission) or 429 (Quota), it returns None
        """

        try:
            await self.client.aio.models.generate_content(
                model=model_id,
                contents="ping",
                config={
                    "max_output_tokens":1,
                    "candidate_count": 1
                } # Minimal cost/latency
            )
            return model_id

        except Exception as e:            
            print(f"Model {model_id} rejected ping: {e}")
            return None
    

    async def get_available_models(self) -> Dict[str, Any]:
        """Retrives a list of available models to be consumed by a user"""

        sem = asyncio.Semaphore(5)
        async def safe_ping(m_id):
            async with sem:
                return await self.ping_model_by_id(m_id)
        try:

            all_models = await self.client.aio.models.list()

            candidate_ids = [
                m.name for m in all_models
                if 'generateContent' in m.supported_actions
            ]

            tasks = [safe_ping(m_id) for m_id in candidate_ids]

            #print(f"{candidate_ids}")
            #print(f"Len: {len(candidate_ids)}")

            # tasks = [
            #     self.ping_model_by_id(m_id)
            #     for m_id in candidate_ids
            # ]

            #print(f"***len: {len(tasks)}")
            results = await asyncio.gather(*tasks)

            #print(f"results: {results}")

            accessible_ids = [
                res for res in results
                if res is not None
            ]

            #print(f"accessible_ids:{accessible_ids}")


            model_list = [
                {"id":m.name.split("/")[-1], "display_name":m.display_name}
                for m in all_models
                if m.name in accessible_ids
            ]

            return {
                "models":model_list,
                "total_count":len(model_list)
            }   

        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            return {"models":[], "total_count":0}


# --- 3. The Singleton Provider (STAYS UNCHANGED) ---
@lru_cache
def get_llm_service() -> LLMService:
    """
    Still using lru_cache to ensure the service (and its client) 
    is only created once, using the cached settings.
    """
    settings = get_settings()
    return LLMService(
        api_key=settings.gemini_api_key.get_secret_value(), 
        default_model=settings.gemini_model
    )





















# """This class provide LLM configuration and communication"""

# from functools import lru_cache

# import logging
# from typing import List, AsyncGenerator, Optional

# import google.generativeai as genai
# from pydantic_settings import BaseSettings, SettingsConfigDict
# from pydantic import Field, SecretStr
# from models import HistoryItem 

# logger = logging.getLogger(__name__)

# # --- 1. Configuration Layer (Pydantic Settings) ---

# class Settings(BaseSettings):
#     """
#     Centralized configuration. Pydantic handles validation and .env loading.
#     If GEMINI_API_KEY is missing, this will raise an error immediately.
#     """
#     gemini_api_key: SecretStr = Field(..., alias="GEMINI_API_KEY")
#     # You can easily swap models for experiments here
#     gemini_model: str = Field(default="gemini-2.5-flash-lite")
    
#     # Standard Pydantic V2 way to link a .env file
#     model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# @lru_cache
# def get_settings() -> Settings:
#     """Returns a cached version of settings so .env is only read once."""
#     return Settings() # type: ignore

# # --- 2. Error Handling ---

# class LLMServiceError(Exception):
#     def __init__(self, *, internal_message: str, public_message: str, status_code: int, raw_response: str):
#         self.internal_message = internal_message
#         self.public_message = public_message
#         self.status_code = status_code
#         self.raw_response = raw_response
#         super().__init__(internal_message)

# # --- 3. The LLM Service ---

# class LLMService:
#     def __init__(self, api_key: str, model_name: str):
#         # Configuration happens once when the class is instantiated
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel(model_name=model_name)
#         logger.info(f"LLMService initialized with model: {model_name}")

#     async def _core_generator(self, contents: list, temperature: float) -> AsyncGenerator[str, None]:
#         """Private engine ensures all experimental methods share the same logic/error handling."""
#         try:
#             config = genai.types.GenerationConfig(temperature = temperature)
#             # Standard async streaming call
#             response = await self.model.generate_content_async(
#                 contents=contents,
#                 stream=True,
#                 generation_config=config,
#             )
#             async for chunk in response:
#                 if chunk.text:
#                     yield chunk.text
#         except Exception as e:
#             logger.exception("Gemini API communication failed")
#             raise LLMServiceError(
#                 internal_message=f"Gemini API failure: {str(e)}",
#                 public_message="The AI service is currently unavailable.",
#                 status_code=502,
#                 raw_response=str(e),
#             ) from e

#     # --- Experimental Wrappers ---

#     async def get_complete(
#         self, 
#         prompt: str, 
#         temperature: float,
#         history: Optional[List[HistoryItem]] = None
#     ) -> str:
        
#         contents = []

#         if history:
#             contents.extend( { "role": item.role, "parts": [{"text": item.text}]} for item in history )

#         contents.append({"role": "user", "parts": [{"text": prompt}]})

#         bits = [text async for text in self._core_generator(contents, temperature)]
#         return "".join(bits) or "No response generated."

#     async def get_stream_sse(self, prompt: str, temperature: float) -> AsyncGenerator[str, None]:
#         contents = [{"role": "user", "parts": [{"text": prompt}]}]
#         async for text in self._core_generator(contents, temperature):
#             yield text

#     async def get_stream(
#         self, 
#         prompt: str,
#         temperature: float, 
#         history: Optional[List[HistoryItem]] = None
#     ) -> AsyncGenerator[str, None]:
        
#         contents = [] 
        
#         if history: 
#             contents.extend( {"role": item.role, "parts": [{"text": item.text}]} for item in history ) 
            
#         contents.append({"role": "user", "parts": [{"text": prompt}]})

#         async for text in self._core_generator(contents, temperature):
#             yield text


# # --- 4. The Singleton Provider (FastAPI Dependency) ---

# @lru_cache
# def get_llm_service() -> LLMService:
#     """
#     This is the "Magic" function. 
#     @lru_cache ensures that LLMService is only instantiated ONCE.
#     """
#     settings = get_settings()
#     return LLMService(
#         api_key=settings.gemini_api_key.get_secret_value(), 
#         model_name=settings.gemini_model
#     )