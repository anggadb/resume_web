import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("PINECONE_INDEX", "test-index")

from fastapi import HTTPException
from starlette.requests import Request

with (
    patch("api.infrastructure.ai.Pinecone"),
    patch("api.infrastructure.ai.ChatGroq"),
):
    from api.main import create_app

from api.core.config import Settings
from api.core.rate_limit import RateLimiter
from api.model import PromptRequest
from api.routes.chat import chat
from api.services.chat import ChatService


def make_request(ip: str = "127.0.0.1", forwarded_for: str | None = None) -> Request:
    headers = []
    if forwarded_for:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": headers,
            "client": (ip, 12345),
        }
    )


class SettingsTests(unittest.TestCase):
    def test_reads_rate_limit_defaults(self):
        with patch.dict(
            os.environ,
            {
                "GROQ_API_KEY": "groq",
                "PINECONE_API_KEY": "pinecone",
                "PINECONE_INDEX": "resume",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.rate_limit_requests, 10)
        self.assertEqual(settings.rate_limit_window_seconds, 60)


class RateLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_requests_over_limit_with_retry_after(self):
        clock = Mock(side_effect=[0.0, 0.0, 1.0, 2.0])
        limiter = RateLimiter(requests=2, window_seconds=60, clock=clock)
        request = make_request()

        await limiter(request)
        await limiter(request)

        with self.assertRaises(HTTPException) as raised:
            await limiter(request)

        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.headers, {"Retry-After": "58"})

    async def test_allows_requests_after_window_expires(self):
        clock = Mock(side_effect=[0.0, 0.0, 61.0])
        limiter = RateLimiter(requests=1, window_seconds=60, clock=clock)
        request = make_request()

        await limiter(request)
        await limiter(request)

    async def test_uses_forwarded_client_ip(self):
        limiter = RateLimiter(requests=1, window_seconds=60)

        await limiter(make_request(forwarded_for="203.0.113.10, 10.0.0.1"))
        await limiter(make_request(forwarded_for="203.0.113.11, 10.0.0.1"))


class ChatServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pinecone = Mock()
        self.index = Mock()
        self.chain = SimpleNamespace(ainvoke=AsyncMock())
        self.service = ChatService(self.pinecone, self.index, self.chain)

    def test_retrieve_embeds_question_and_queries_index(self):
        matches = [Mock(name="match")]
        self.pinecone.inference.embed.return_value = SimpleNamespace(
            data=[{"values": [0.1, 0.2]}]
        )
        self.index.query.return_value = SimpleNamespace(matches=matches)

        result = self.service.retrieve("What has Angga built?", top_k=3)

        self.assertEqual(result, matches)
        self.pinecone.inference.embed.assert_called_once_with(
            model="llama-text-embed-v2",
            inputs=["What has Angga built?"],
            parameters={"input_type": "query"},
        )
        self.index.query.assert_called_once_with(
            vector=[0.1, 0.2],
            top_k=3,
            include_metadata=True,
        )

    async def test_answer_returns_answer_and_sources(self):
        matches = [
            {
                "score": 0.95,
                "metadata": {
                    "text": "Built a resume assistant.",
                    "source": "resume.pdf",
                    "chunk": 4,
                },
            },
            {
                "score": 0.82,
                "metadata": {"text": "Uses retrieval augmented generation."},
            },
        ]
        self.chain.ainvoke.return_value = "<p>I built a resume assistant.</p>"

        with patch.object(self.service, "retrieve", return_value=matches):
            response = await self.service.answer("What did you build?")

        self.assertEqual(response.answer, "<p>I built a resume assistant.</p>")
        self.assertEqual(
            [source.model_dump() for source in response.sources],
            [
                {"score": 0.95, "source": "resume.pdf", "chunk": 4},
                {"score": 0.82, "source": None, "chunk": None},
            ],
        )
        self.chain.ainvoke.assert_awaited_once_with(
            {
                "context": (
                    "Built a resume assistant.\n\n"
                    "Uses retrieval augmented generation."
                ),
                "question": "What did you build?",
            }
        )


class ChatRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_converts_service_errors_to_http_500(self):
        service = SimpleNamespace(
            answer=AsyncMock(side_effect=RuntimeError("Pinecone unavailable"))
        )

        with self.assertRaises(HTTPException) as raised:
            await chat(PromptRequest(prompt="Tell me about Angga"), service)

        self.assertEqual(raised.exception.status_code, 500)
        self.assertEqual(raised.exception.detail, "Pinecone unavailable")

    def test_app_registers_chat_route(self):
        settings = Settings("groq", "pinecone", "resume")
        service = Mock(spec=ChatService)

        app = create_app(settings=settings, chat_service=service)

        self.assertIn("/api/chat", app.openapi()["paths"])


if __name__ == "__main__":
    unittest.main()
