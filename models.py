from typing import List, Literal
from pydantic import BaseModel, Field


class HistoryItem(BaseModel):
    """Defines a history item object with 'user' or 'model' roles"""
    role:Literal["user", "model"]
    text: str

class QueryRequest(BaseModel):
    """Request object"""
    prompt: str
    temperature: float = 0.7
    history: List[HistoryItem] = Field(default_factory=list)