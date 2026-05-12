# main.py
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


from LLMService import LLMServiceError
from routers import chat

logger = logging.getLogger(__name__)

app = FastAPI(
    title = "LLM streaming API", 
    version = "1.0.0"
)

#adding middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:5173", "https://micharlar.netlify.app"], 
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

#Registering chat.py
app.include_router(chat.router, prefix="/api/v1/chat")

@app.exception_handler(LLMServiceError)
async def llm_error_handler(request: Request, e: LLMServiceError):
    return JSONResponse(
        status_code=e.status_code,
        content = {
            "detail":{
                "error":"LLM_PROVIDER_ERROR",
                "type": type(e).__name__,
                "message":e.public_message
            }
        }

    )

@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, e: Exception):
    #log the full traceback for the devs
    logger.error(f"Unhandled system crash: {str(e)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content = {
            "detail":{
                "error":"UNEXPECTED_CRASH",
                "type": type(e).__name__,
                "message":"An internal error occurred. Please try again later."
            }
        }
    )
