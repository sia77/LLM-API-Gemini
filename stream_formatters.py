import json


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

    
async def stream_formatter_sse(raw_stream):
    """
    Takes the raw text stream from LLMService and formats it 
    into plain text.
    """
    async for text_chunk in raw_stream:       
    
        data = { "text":text_chunk }
        payload = json.dumps(data)
        yield f"data:{payload}\n\n"


