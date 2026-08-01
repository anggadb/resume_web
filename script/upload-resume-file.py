import uuid
import os

from dotenv import load_dotenv
from pypdf import PdfReader
from pinecone import Pinecone

load_dotenv()

# PUT YOUR CREDENTIALS, BETTER TO USE ENVIRONMENT VARIABLES FOR SECURITY.
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX")) # type: ignore

# SAME LEVEL DIRECTORY
PDF_PATH = "profile.pdf"


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks: list[str] = []

    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap

    return chunks


def embed(texts: list[str]):
    response = pc.inference.embed( # type: ignore
        model="llama-text-embed-v2",
        inputs=texts,
        parameters={
            "input_type": "passage"
        }
    )

    return [item["values"] for item in response.data]


def upload(chunks: list[str]):
    vectors: list[dict[str, object]] = []

    embeddings = embed(chunks)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "chunk": i,
                "text": chunk,
                "source": PDF_PATH,
            }
        })

    index.upsert(vectors=vectors) # type: ignore


text = read_pdf(PDF_PATH)
chunks = chunk_text(text)

upload(chunks)

print("Upload complete.")