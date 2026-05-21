import ast
import asyncio
from functools import lru_cache
import logging
import re
import sys
from typing import Any, Dict, List, AsyncGenerator, Optional, TypedDict

from google import genai
from google.genai import types
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr
from models import HistoryItem

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(levelname)s - %(name)s - %(message)s"
)


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

class ErrorData(TypedDict):
    public_message:str
    status_code:int
    is_retryable:bool
    raw_info:str
    internal_message:str
    

class LLMServiceError(Exception):
    def __init__(self, *, internal_message: str, public_message: str, status_code: int, raw_response: str, is_retryable:bool):
        self.internal_message = internal_message
        self.public_message = public_message
        self.status_code = status_code
        self.raw_response = raw_response
        self.is_retryable = is_retryable
        super().__init__(internal_message)

# --- 3. The Modernized LLM Service ---

class LLMService:
    def __init__(self, api_key: str, default_model: str):
        # We initialize the Client once per service instance
        self.client = genai.Client(api_key=api_key)
        self.default_model = default_model
        logger.info(f"LLMService initialized with default model: {default_model}")


    @staticmethod
    def extract_error_details(raw_error_val: Any) -> ErrorData:

        raw_error_str= str(raw_error_val)

        # Default: Assume a generic server-side issue
        errorData:ErrorData = {
            "public_message":"An AI provider error occurred.",
            "status_code":500,
            "is_retryable":True,
            "raw_info":raw_error_str,
            "internal_message":raw_error_str
        }
        
        try:
            # \{.*\} means "Find the first { and match everything until the last }"
            match = re.search(r"\{.*\}", raw_error_str, re.DOTALL)

            if match:
                dict_str = match.group(0)                

                # We use literal_eval because the SDK uses single quotes '
                error_dict = ast.literal_eval(dict_str)

                nested_msg = error_dict.get('error', {}).get('message', "")
                lowercase_msg = nested_msg.lower()

                if any(word in lowercase_msg for word in ["temperature", "invalid", "400"]):
                    errorData["public_message"] = nested_msg.split(':')[-1].strip()
                    errorData["status_code"] = 400
                    errorData["is_retryable"] = False

                elif "tier" in lowercase_msg or "limit: 0" in lowercase_msg:
                    errorData["public_message"] = "The selected model is unavailable on this account tier."
                    errorData["status_code"] = 429
                    errorData["is_retryable"] = False

                elif any(word in lowercase_msg for word in ["quota", "limit", "429"]):
                    errorData["public_message"] = "Rate limit reached. Please wait a moment."
                    errorData["status_code"] = 429
                    errorData["is_retryable"] = True
                else:
                    if nested_msg:
                        errorData["public_message"] = nested_msg

                return errorData

        except Exception as e:
            logger.error(f"Surgical parsing failed, falling back to 500: {e}")
            pass

        return errorData


    # --- Methods ---

    async def get_complete(
        self, 
        prompt: str,
        **kwargs
    ) -> str:
        """Full response implementation."""
        contents = self._prepare_contents(prompt, history = kwargs.get("history"))
        config = types.GenerateContentConfig(temperature = kwargs.get("temperature"))

        try:
            response = await self.client.aio.models.generate_content(
                model= kwargs.get("model_name") or self.default_model,
                contents=contents,
                config=config
            )
            return response.text or "No response generated." 
        except Exception as e: 
            error_data = LLMService.extract_error_details(e)
            logger.exception(f"LLM error has been captured: {error_data}")

            raise LLMServiceError(
                public_message = error_data["public_message"],
                internal_message = error_data["internal_message"],
                status_code = error_data["status_code"],
                raw_response = error_data["raw_info"],
                is_retryable = error_data["is_retryable"]
            )  


    async def get_stream(
        self, 
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Standard chunk streaming."""
        contents = self._prepare_contents(prompt, kwargs.get("history"))
        target_model = kwargs.get("model_name") or self.default_model
        config = types.GenerateContentConfig(temperature=kwargs.get("temperature"))

        init_stream = await self.client.aio.models.generate_content_stream(
            model=target_model,
            contents=contents,
            config=config
        )

        async def _generator():

            try:
                async for chunk in init_stream:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:

                error_data = LLMService.extract_error_details(e)
                logger.exception(f"LLM error has been captured: {error_data}")

                raise LLMServiceError(
                    public_message = error_data["public_message"],
                    internal_message = error_data["internal_message"],
                    status_code = error_data["status_code"],
                    raw_response = error_data["raw_info"],
                    is_retryable = error_data["is_retryable"]
                )

        return _generator()
        

    async def get_raw_sse_stream(
        self, 
        prompt: str, 
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """SSE formatted streaming."""
        try:
            generator = await self.get_stream(prompt, **kwargs)
            async for text in generator:
                yield text #There is already a utility that formats SSE responses
        except Exception as e:
            logger.exception(f"***Error: {e}")
            error_data = LLMService.extract_error_details(e)
            logger.exception(f"LLM error has been captured: {error_data}")

            raise LLMServiceError(
                public_message = error_data["public_message"],
                internal_message = error_data["internal_message"],
                status_code = error_data["status_code"],
                raw_response = error_data["raw_info"],
                is_retryable = error_data["is_retryable"]
            )
        
            # yield f"event: sse_error\n data:{error_payload}\n\n"

    def _prepare_contents(self, prompt: str, history: Optional[List[HistoryItem]]) -> list:
        contents = []
        if history:
            contents.extend({"role": item.role, "parts": [{"text": item.text}]} for item in history)
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        return contents


    """This is going to be used as some sort of checkup to update the list of available models. 
        It could run every often and then update the list, and save it to DB"""    
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

