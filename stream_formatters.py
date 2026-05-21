import json
from typing import AsyncGenerator
from venv import logger

from fastapi import HTTPException

from LLMService import LLMServiceError

def complete_formatter_text(result: str) -> str: 
    return result 

def complete_formatter_json(result: str) -> dict: 
    return {"text": result}


async def stream_formatter_json(raw_stream):
    """
    Takes the raw text stream from LLMService and formats it 
    into line-delimited JSON (NDJSON) and encodes it for HTTP streaming.
    """
    try:
        async for text_chunk in raw_stream:
        
            data = {"text": text_chunk}
            
            # 2. Serialize to a JSON string and add a newline (NDJSON)
            json_string = json.dumps(data, ensure_ascii=False) + "\n"
            
            # Encode the string to bytes for the StreamingResponse
            yield json_string.encode("utf-8")
    except LLMServiceError as e: 
        raise HTTPException( 
            status_code=e.status_code, 
            detail = { 
                "error": "LLM_PROVIDER_ERROR", 
                "message": e.public_message 
            } 
        ) 
    except Exception as e:
        logger.error(f"Streaming Error: {e}", exc_info=True)
        message = getattr(e, "public_message", "There was an issue with streaming")
        error_payload = json.dumps({"text": message}) + "\n"
        yield error_payload.encode("utf-8")


async def stream_formatter_text(raw_stream):
    """
    Takes the raw text stream from LLMService and formats it 
    into plain text.
    """
    try:
        async for text_chunk in raw_stream:
            # Encode the string to bytes for the StreamingResponse
            yield text_chunk.encode("utf-8")
    except Exception as e:
        logger.error(f"Streaming Error: {e}", exc_info=True)
        message = getattr(e, "public_message", "There was an issue with streaming")
        error_payload = message
        yield error_payload.encode("utf-8")

    
async def stream_formatter_sse(raw_stream) -> AsyncGenerator[str, None]:
    """
    Formats an async text stream into SSE events:
        - 'chunk' for incremental text
        - 'done' when the stream finishes normally
        - 'error' if an exception occurs
    """
    try:
        async for text_chunk in raw_stream:
            payload = json.dumps({"text":text_chunk})
            yield f"event: chunk\ndata: {payload}\n\n"

        # When the generator finishes normallly
        yield "event: done\ndata: {}\n\n"
    
    except Exception as e:
        # Send an error event with a message
        logger.error(f"Streaming Error: {e}", exc_info=True)
        message = getattr(e, "public_message", "There was an issue with streaming")
        error_payload = json.dumps({"message": message})
        yield f"event: sse_error\ndata: {error_payload}\n\n"

