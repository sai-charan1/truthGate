import sys
sys.path.insert(0, ".")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from src.qa_engine import query

app = FastAPI(
    title="TruthGate",
    description="A RAG system over FastAPI documentation that knows when to shut up.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer_type: str
    answer: str
    citations: list
    confidence: float
    cost_usd: float
    latency_ms: int
    retrieved_chunks: int
    top_retrieval_score: float
    is_multi_section: bool
    reasoning: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "corpus": "FastAPI official documentation"}


@app.post("/query", response_model=QuestionResponse)
def ask(req: QuestionRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = query(req.question.strip())
    return result


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
