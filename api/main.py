import os
import asyncio
import logging

from fastapi import FastAPI, HTTPException
from mangum import Mangum

from pinecone import Pinecone

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from contextlib import asynccontextmanager

from api.model import PromptRequest


# --------------------------------------------------------
# Config
# --------------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not all([GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX]):
    raise RuntimeError("Missing environment variables.")

app = FastAPI(
    title="Resume AI Orchestrator API",
    description="AI Orchestrator for Angga Bachtiar's Resume using Gemini 1.5 Flash",
    version="1.0",
)

handler = Mangum(app)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX) # type: ignore

llm = ChatGroq(
    api_key=GROQ_API_KEY, # type: ignore
    model="openai/gpt-oss-120b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant.

Answer ONLY using the provided context.

If the answer is not contained in the context, simply say you don't know.

Context:
{context}

Question:
{question}
"""
)

chain = (
    prompt
    | llm
    | StrOutputParser()
)


# --------------------------------------------------------
# Pinecone Retrieval
# --------------------------------------------------------

def retrieve(question: str, top_k: int = 5):

    embedding = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[question],
        parameters={
            "input_type": "query",
        },
    )

    query_vector = embedding.data[0]["values"]

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    return result.matches


# --------------------------------------------------------
# Endpoint
# --------------------------------------------------------

@app.post("/api/chat")
async def chat(req: PromptRequest) -> dict[str, object]:
    try:
        matches = await asyncio.to_thread(
            retrieve,
            req.question, # type: ignore
        )

        context = "\n\n".join(
            match["metadata"]["text"]
            for match in matches
        )

        answer = await chain.ainvoke(
            {
                "context": context,
                "question": req.question, # type: ignore
            }
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "score": match["score"],
                    "source": match["metadata"].get("source"),
                    "chunk": match["metadata"].get("chunk"),
                }
                for match in matches
            ],
        }

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )