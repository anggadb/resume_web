import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from mangum import Mangum

app = FastAPI(
    title="Resume AI Orchestrator API",
    description="AI Orchestrator for Angga Bachtiar's Resume using Gemini 1.5 Flash",
    version="1.0"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def load_resume_context():
    try:
        with open("context.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "AI context is not found."

RESUME_CONTEXT = load_resume_context()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/chat")
async def chat_with_ai(request: PromptRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY is not set. Please set the environment variable to use the AI features."
        )
    
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=RESUME_CONTEXT
        )
        
        response = model.generate_content(request.prompt)
        
        return {"answer": response.text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pada AI: {str(e)}")

handler = Mangum(app)