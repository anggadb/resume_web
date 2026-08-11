import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


# api.main validates configuration while it is imported. These values are only
# used to construct clients; every external operation is mocked below.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("PINECONE_INDEX", "test-index")

from fastapi import HTTPException
from starlette.requests import Request

# Prevent client construction during module import from contacting external
# services. Individual tests configure the mocked clients they exercise.
with (
    patch("pinecone.Pinecone"),
    patch("langchain_groq.ChatGroq"),
):
    from api import main

from api.model import PromptRequest
from api.rate_limit import RateLimiter


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


class RetrieveTests(unittest.TestCase):
    def test_retrieve_embeds_question_and_queries_index(self):
        matches = [Mock(name="match")]
        main.pc.inference.embed = Mock(
            return_value=SimpleNamespace(data=[{"values": [0.1, 0.2]}])
        )
        main.index.query = Mock(return_value=SimpleNamespace(matches=matches))

        result = main.retrieve("What has Angga built?", top_k=3)

        self.assertEqual(result, matches)
        main.pc.inference.embed.assert_called_once_with(
            model="llama-text-embed-v2",
            inputs=["What has Angga built?"],
            parameters={"input_type": "query"},
        )
        main.index.query.assert_called_once_with(
            vector=[0.1, 0.2],
            top_k=3,
            include_metadata=True,
        )


class ChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_returns_answer_and_sources(self):
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
        fake_chain = SimpleNamespace(
            ainvoke=AsyncMock(return_value="<p>I built a resume assistant.</p>")
        )

        with (
            patch("api.main.asyncio.to_thread", AsyncMock(return_value=matches)) as to_thread,
            patch.object(main, "chain", fake_chain),
        ):
            response = await main.chat(PromptRequest(prompt="What did you build?"))

        self.assertEqual(response.answer, "<p>I built a resume assistant.</p>")
        self.assertEqual(
            [source.model_dump() for source in response.sources],
            [
                {"score": 0.95, "source": "resume.pdf", "chunk": 4},
                {"score": 0.82, "source": None, "chunk": None},
            ],
        )
        to_thread.assert_awaited_once_with(main.retrieve, "What did you build?")
        fake_chain.ainvoke.assert_awaited_once_with(
            {
                "context": (
                    "Built a resume assistant.\n\n"
                    "Uses retrieval augmented generation."
                ),
                "question": "What did you build?",
            }
        )

    async def test_chat_converts_internal_errors_to_http_500(self):
        with patch(
            "api.main.asyncio.to_thread",
            AsyncMock(side_effect=RuntimeError("Pinecone unavailable")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await main.chat(PromptRequest(prompt="Tell me about Angga"))

        self.assertEqual(raised.exception.status_code, 500)
        self.assertEqual(raised.exception.detail, "Pinecone unavailable")


if __name__ == "__main__":
    unittest.main()
