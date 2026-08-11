import logging

from fastapi import FastAPI
from mangum import Mangum

from api.core.config import Settings
from api.core.rate_limit import RateLimiter
from api.infrastructure.ai import create_chat_service
from api.routes.chat import router as chat_router
from api.services.chat import ChatService


logging.basicConfig(level=logging.INFO)


def create_app(
    settings: Settings | None = None,
    chat_service: ChatService | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    chat_service = chat_service or create_chat_service(settings)

    app = FastAPI(
        title="Resume AI Orchestrator API",
        description="AI-powered assistant for Angga Bachtiar's resume",
        version="1.0",
    )
    app.state.chat_service = chat_service
    app.state.chat_rate_limiter = RateLimiter(
        requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    app.include_router(chat_router)
    return app


app = create_app()
handler = Mangum(app)
