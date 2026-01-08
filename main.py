# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import chat

app = FastAPI(
    title = "LLM streaming API", 
    version = "1.0.0"
)

#adding middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], 
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.include_router(chat.router, prefix="/api/v1/chat")