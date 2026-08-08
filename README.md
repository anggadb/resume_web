# Resume AI Assistant

An interactive portfolio that lets recruiters, hiring managers, and collaborators explore my professional experience through conversation.

Instead of searching through a traditional résumé, visitors can ask direct questions about my background, projects, skills, and achievements. The assistant retrieves relevant information from my résumé and responds in my voice—clearly, professionally, and without inventing details.

## What it does

- Presents my profile, experience, and contact information in a responsive portfolio
- Answers natural-language questions through an embedded AI assistant
- Uses retrieval-augmented generation (RAG) to ground answers in my résumé
- Returns concise, sanitized HTML responses for safe rendering in the browser
- Includes source metadata for retrieved résumé sections
- Runs as a FastAPI serverless API and is configured for Vercel deployment

## How it works

1. A visitor submits a question from the portfolio.
2. Pinecone converts the question into an embedding and retrieves the most relevant résumé sections.
3. Groq generates a concise response using only the retrieved context.
4. The frontend sanitizes the generated HTML with DOMPurify before displaying it.

## Technology

- **Frontend:** HTML, Tailwind CSS, JavaScript, DOMPurify
- **API:** Python 3.14, FastAPI, Mangum
- **AI orchestration:** LangChain and Groq
- **Vector search:** Pinecone
- **PDF ingestion:** PyPDF
- **Deployment:** Vercel
- **Dependency management:** uv

## Project structure

```text
resume_web/
├── api/
│   ├── main.py             # FastAPI application and RAG pipeline
│   ├── model.py            # Request model
│   └── test_main.py        # Unit tests
├── assets/
│   ├── script.js           # Portfolio and chat interactions
│   └── style.css           # Custom styling
├── script/
│   └── upload-resume-file.py
├── index.html              # Portfolio interface
├── pyproject.toml          # Project metadata and dependencies
└── vercel.json             # Serverless API routing
```

## Local setup

### Prerequisites

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- A Groq API key
- A Pinecone account and index

Clone the repository and install the locked dependencies:

```bash
git clone https://github.com/anggadb/resume_web.git
cd resume_web
uv sync
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_pinecone_index_name
```

Environment files are excluded from Git. Never commit real credentials.

## Add résumé content to Pinecone

Place the résumé PDF at `profile.pdf` in the project root, then run:

```bash
uv run python script/upload-resume-file.py
```

The ingestion script extracts the PDF text, splits it into overlapping chunks, generates passage embeddings, and uploads the vectors and source metadata to Pinecone.

## Run locally

Start the API:

```bash
uv run uvicorn api.main:app --reload
```

Serve the frontend from a second terminal:

```bash
uv run python -m http.server 3000
```

Open [http://localhost:3000](http://localhost:3000). The frontend sends chat requests to `/api/chat`; use a local reverse proxy or serve the frontend and API under the same origin when testing the complete integration locally.

You can test the API directly at [http://localhost:8000/docs](http://localhost:8000/docs).

Example request:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is your backend engineering experience?"}'
```

## Run the tests

With `uv`:

```bash
uv run python -m unittest api.test_main -v
```

From PowerShell using the repository's Linux-based virtual environment through WSL:

```powershell
wsl .venv/bin/python -m unittest api.test_main -v
```

The tests mock Pinecone and Groq, so they do not require network access or real API credentials.

## Deploy to Vercel

Add the following environment variables to the Vercel project:

- `GROQ_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX`

The included `vercel.json` routes `/api/*` requests to the FastAPI application in `api/main.py`.

## API

### `POST /api/chat`

Request:

```json
{
  "prompt": "Which high-scale systems have you worked on?"
}
```

Response:

```json
{
  "answer": "<p>...</p>",
  "sources": [
    {
      "score": 0.91,
      "source": "profile.pdf",
      "chunk": 2
    }
  ]
}
```

## Author

**Angga Bachtiar**

Software Engineer focused on backend development, distributed systems, and high-scale applications.

Contact: [bachtiar.angga@gmail.com](mailto:bachtiar.angga@gmail.com)
