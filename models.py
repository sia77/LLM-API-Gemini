from typing import List
from pydantic import BaseModel


class HistoryItem(BaseModel):
    """Defines a history item object with 'user' or 'model' roles"""
    role:str
    text: str

class QueryRequest(BaseModel):
    """Request object"""
    prompt: str
    temperature: float = 0.7
    history: List[HistoryItem]