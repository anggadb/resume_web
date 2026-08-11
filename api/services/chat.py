import asyncio
from typing import Any

from api.model import ChatResponse


class ChatService:
    def __init__(self, pinecone: Any, index: Any, chain: Any) -> None:
        self._pinecone = pinecone
        self._index = index
        self._chain = chain

    def retrieve(self, question: str, top_k: int = 5) -> list[Any]:
        embedding = self._pinecone.inference.embed(
            model="llama-text-embed-v2",
            inputs=[question],
            parameters={"input_type": "query"},
        )
        query_vector = embedding.data[0]["values"]
        result = self._index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )
        return result.matches

    async def answer(self, question: str) -> ChatResponse:
        matches = await asyncio.to_thread(self.retrieve, question)
        context = "\n\n".join(match["metadata"]["text"] for match in matches)
        answer = await self._chain.ainvoke(
            {"context": context, "question": question}
        )

        return ChatResponse(
            answer=answer,
            sources=[
                {
                    "score": match["score"],
                    "source": match["metadata"].get("source"),
                    "chunk": match["metadata"].get("chunk"),
                }
                for match in matches
            ],
        )
