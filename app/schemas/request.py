from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    role: str
    session_id: str | None = None