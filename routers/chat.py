from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from LLMService import LLMService, get_llm_service, LLMServiceError
from models import QueryRequest
from stream_formatters import stream_formatter_json, stream_formatter_text, stream_formatter_sse

router = APIRouter(
    tags = ["chat"]
)

@router.get("/")
def health():
    """API health"""
    return { "status" : "ok", "message": "LLM API is running and ready."}

@router.get("/stream/sse")
async def query_stream_sse(
    prompt: str,
    llm_service:LLMService = Depends(get_llm_service)
):
    """This endpoint handles sse GET requests - no history due to GET string length limitations"""
    try:
        raw_stream = llm_service.get_stream_sse(prompt, 0.7)
        return StreamingResponse(stream_formatter_sse(raw_stream), media_type="text/event-stream")
    except LLMServiceError as e:
        raise HTTPException(
            status_code=500,
            detail=f"External LLM service failed: {e}"
        ) from e

@router.post("/stream/text")
async def query_stream_text(
    request_data:QueryRequest,
    llm_service:LLMService = Depends(get_llm_service)
):
    """This end point responds in streams of plain text format"""

    try:
        raw_stream = llm_service.get_stream(
            request_data.prompt, 
            request_data.history, 
            request_data.temperature
        )

        return StreamingResponse(stream_formatter_text(raw_stream), media_type="text/plain")
    except LLMServiceError as e:
        # This now returns the ACTUAL status code (e.g., 401, 429) 
        # and the ACTUAL reason why it failed.
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": "LLM_PROVIDER_ERROR",
                "message": e.message,
                "provider_details": e.raw_response
            }
        )
    except Exception as e:
        # BLUNT FIX: This catches everything else (NameError, AttributeError, etc.)
        # and sends the actual error string to your browser/curl
        raise HTTPException(
            status_code=500,
            detail={
                "error": "UNEXPECTED_CRASH",
                "type": type(e).__name__,
                "message": str(e)
            }
        )
    # except LLMServiceError as e:
    #     raise HTTPException(
    #         status_code=500,
    #         detail=f"External LLM service failed: {e}"
    #     ) from e
    
@router.post("/stream/json")
async def query_stream_json(
    request_data:QueryRequest,
    llm_service: LLMService = Depends(get_llm_service)
):
    """This end point responds in streams of Json format"""

    try:
        raw_stream = llm_service.get_stream(
            request_data.prompt, 
            request_data.history, 
            request_data.temperature
        )
        return StreamingResponse(
            stream_formatter_json(raw_stream), 
            media_type="application/x-ndjson"
        )
    except LLMServiceError as e:
        raise HTTPException(
            status_code=500,
            detail=f"External LLM service failed: {e}"
        ) from e
    
