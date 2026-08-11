import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from api.model import ChatResponse, PromptRequest
from api.services.chat import ChatService


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


async def enforce_chat_rate_limit(request: Request) -> None:
    await request.app.state.chat_rate_limiter(request)


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(enforce_chat_rate_limit)],
    responses={429: {"description": "Rate limit exceeded"}},
)
async def chat(
    req: PromptRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    try:
        return await service.answer(req.prompt)
    except Exception as error:
        logger.exception(error)
        raise HTTPException(status_code=500, detail=str(error)) from error
