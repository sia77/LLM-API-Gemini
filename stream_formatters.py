import json
from typing import AsyncGenerator

def complete_formatter_text(result: str) -> str: 
    return result 

def complete_formatter_json(result: str) -> dict: 
    return {"text": result}


async def stream_formatter_json(raw_stream):
    """
    Takes the raw text stream from LLMService and formats it 
    into line-delimited JSON (NDJSON) and encodes it for HTTP streaming.
    """
    async for text_chunk in raw_stream:
       
        data = {"text": text_chunk}
        
        # 2. Serialize to a JSON string and add a newline (NDJSON)
        json_string = json.dumps(data) + "\n"
        
        # 3. Encode the string to bytes for the StreamingResponse
        yield json_string.encode("utf-8")


async def stream_formatter_text(raw_stream):
    """
    Takes the raw text stream from LLMService and formats it 
    into plain text.
    """
    async for text_chunk in raw_stream:       
       
        # 3. Encode the string to bytes for the StreamingResponse
        yield text_chunk.encode("utf-8")

    
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
        error_payload = json.dumps({"message": str(e)})
        yield f"event: sse_error\ndata:{error_payload}\n\n"

        




