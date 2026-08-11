import os
import asyncio
import logging

from fastapi import Depends, FastAPI, HTTPException
from mangum import Mangum

from pinecone import Pinecone

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from api.model import ChatResponse, PromptRequest
from api.rate_limit import RateLimiter


# --------------------------------------------------------
# Config
# --------------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

if not all([GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX]):
    raise RuntimeError("Missing environment variables.")

app = FastAPI(
    title="Resume AI Orchestrator API",
    description="AI Orchestrator for Angga Bachtiar's Resume using Gemini 1.5 Flash",
    version="1.0",
)

chat_rate_limiter = RateLimiter(
    requests=RATE_LIMIT_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
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
You are Angga Bachtiar's AI representative.

Your role is to answer questions from guests, recruiters, clients, or hiring managers about Angga Bachtiar's career, experience, skills, achievements, and projects.

Use ONLY the information provided in the context below, which comes from Angga Bachtiar's documents stored in the knowledge base.

Instructions:
- Answer in first person, as if you are Angga Bachtiar.
- Be professional, confident, and concise.
- Do not invent or assume information that is not present in the context.
- If the context does not contain enough information, say:
  "Hi, thank you for your question! I apologize, but I think for this one we need to have a coffee talk or just contact me directly, thanks!."
- When appropriate, summarize multiple projects or experiences into a clear and natural response.

Formatting rules:
- Output valid HTML only.
- Allowed tags:
  <p>, <br>, <ul>, <ol>, <li>,
  <strong>, <em>, <code>, <pre>,
  <h2>, <h3>, <blockquote>
- Do NOT output <html>, <body>, <style>, <script>, <iframe>, or inline CSS.
- Use <ul><li> for lists.
- Use <code> for technologies.
- Use <strong> for important keywords.
- Keep HTML semantic and minimal.

Context:
{context}

Guest Question:
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

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    dependencies=[Depends(chat_rate_limiter)],
    responses={429: {"description": "Rate limit exceeded"}},
)
async def chat(req: PromptRequest) -> ChatResponse:
    try:
        matches = await asyncio.to_thread(
            retrieve,
            req.prompt, # type: ignore
        )

        context = "\n\n".join(
            match["metadata"]["text"]
            for match in matches
        )

        answer = await chain.ainvoke(
            {
                "context": context,
                "question": req.prompt, # type: ignore
            }
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

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
