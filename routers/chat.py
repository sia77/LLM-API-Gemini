from enum import Enum
import json
import logging
from pathlib import Path
from typing import Annotated, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Header, Response
from fastapi.responses import StreamingResponse

from LLMService import LLMService, get_llm_service, LLMServiceError
from models import QueryRequest
from stream_formatters import complete_formatter_json, complete_formatter_text, stream_formatter_json, stream_formatter_text, stream_formatter_sse
logger = logging.getLogger(__name__)

router = APIRouter(
    tags = ["chat"]
)

class AcceptHeader(str, Enum):
    json = "application/json"
    text = "text/plain"
    # sse  = "text/event-stream"
    # ndjson = "application/x-ndjson"

@router.get("/")
def health():
    """API health"""
    return { "status" : "ok", "message": "LLM API is running and ready."}

@router.get("/stream/sse")
async def query_stream_sse(
    prompt: str,
    temperature:float=0.7,
    model_name:Optional[str]="gemini-2.5-flash-lite",
    llm_service:LLMService = Depends(get_llm_service)
):
    """This endpoint handles sse GET requests - no history due to GET string length limitations"""

    settings = { 
        k:v 
        for k,v in {"temperature":temperature, "model_name":model_name}.items() 
        if v is not None
    }        
    raw_stream = llm_service.get_raw_sse_stream(prompt, **settings)
    return StreamingResponse(stream_formatter_sse(raw_stream), media_type="text/event-stream")

    
COMPLETE_FORMATTERS = { 
    "text": (complete_formatter_text, "text/plain"), 
    "json": (complete_formatter_json, "application/json"), 
}
    
@router.post("/complete")
async def query_complete(
    request_data:QueryRequest,
    # I switeched form accept to x_format so that swagger doesn't override the selection picked. 
    # I discovered swagger's internal logic always picks 'application/json' for document purposes
    #### The client MUST send 'X-Format' in the request header. ###
    x_format: Annotated[AcceptHeader, Header(alias="X-Format")] = AcceptHeader.json, 
    #format_key:Literal["text", "json"] = "text",
    llm_service:LLMService = Depends(get_llm_service)
):
    """This end point responds in complete in the form of text/plain or application/json format"""

    is_json = (x_format == AcceptHeader.json)
    format_key = "json" if is_json else "text"  
    formatter, media_type = COMPLETE_FORMATTERS[format_key]  
    settings = { 
        k:v
        for k,v in
        {
            "temperature":request_data.temperature, 
            "history":request_data.history, 
            "model_name":request_data.model_name
        }.items()
        if v is not None 
    }
    
    result = await llm_service.get_complete(
        request_data.prompt, 
        **settings
    )

    body = formatter(result)

    if isinstance(body, dict):
            return body # FastAPI auto-converts dict to JSON response
            
    return Response(content=body, media_type=media_type)


    
FORMATTERS = {
    "text": (stream_formatter_text, "text/plain"),
    "json": (stream_formatter_json, "application/x-ndjson"),
}


@router.post("/stream")
async def query_stream(
    request_data:QueryRequest,
    # I switeched form accept to x_format so that swagger doesn't override the selection picked. 
    # I discovered swagger's internal logic always picks 'application/json' for document purposes
    #### The client MUST send 'X-Format' in the request header. ###
    #format_key:Literal["text", "json"] = "json",
    x_format: Annotated[AcceptHeader, Header(alias="X-Format")] = AcceptHeader.json, # FastAPI extracts 'Accept' header here
    llm_service:LLMService = Depends(get_llm_service)
):
    """This end point responds in stream in the form of text/plain or application/json format"""

    is_json = (x_format == AcceptHeader.json)
    format_key = "json" if is_json else "text"

    formatter, media_type = FORMATTERS[format_key]
    settings = {
        k:v
        for k,v in {
            "temperature":request_data.temperature, 
            "history":request_data.history,
            "model_name":request_data.model_name
        }.items()
        if v is not None
    }

    gen_obj = await llm_service.get_stream(
        request_data.prompt,
        **settings            
    )

    try:
        # We use __anext__() to manually grab the first chunk
        first_chunk = await gen_obj.__anext__()
    except StopAsyncIteration:
        first_chunk = "" # Handle empty responses gracefully


    async def combined_gen():
        if first_chunk:
            yield first_chunk
        async for chunk in gen_obj:
            yield chunk

    return StreamingResponse(
        formatter(combined_gen()), 
        media_type=media_type
    )
    

    
@router.get('/models')
async def get_models():   
    # Locate the file relative to this python file 
    file_path = Path(__file__).parent.parent / "data" / "supported_models.json"
    try:
        with open(file_path, "r") as f:
        # This converts the JSON text directly into a Python list/dict
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Configuration Error: {e}")
        #Return the 'Safe Shape' so React doesn't break
        return {"models":[], "total_count":0}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"models":[], "total_count":0}
    
