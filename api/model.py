from pydantic import BaseModel


class PromptRequest(BaseModel):
    prompt: str


class ChatSource(BaseModel):
    score: float
    source: str | None = None
    chunk: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
