from datetime import datetime
from pydantic import BaseModel


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    agent_steps: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationWithMessages(ConversationOut):
    messages: list[MessageOut]


class ChatRequest(BaseModel):
    message: str
