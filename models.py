from typing import List, Optional
from pydantic import BaseModel, Field


class HistoryItem(BaseModel):
    """Defines a history item object with 'user' or 'model' roles"""
    role:str
    text: str

class QueryRequest(BaseModel):
    """Request object"""
    prompt: str
    temperature: float = 0.7
    history: Optional[List[HistoryItem]] = None
    model_name: Optional[str] = Field(None, examples=["gemini-2.5-flash-lite"])