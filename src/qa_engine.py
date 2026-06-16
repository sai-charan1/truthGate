import os
import time
import json
import tiktoken
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv
from src.retrieval.retriever import retrieve_multi_section

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

MODEL = "gpt-4o-mini"
REFUSAL_THRESHOLD = 0.12
COST_PER_INPUT_TOKEN = 0.00015 / 1000
COST_PER_OUTPUT_TOKEN = 0.0006 / 1000

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


SYSTEM_PROMPT = """You are TruthGate, a precise question-answering assistant restricted to the FastAPI official documentation.

Your response MUST be valid JSON with exactly this structure:
{
  "answer_type": "answered" | "unanswerable" | "false_premise",
  "answer": "your answer text, or empty string",
  "citations": ["Exact Source Title 1", "Exact Source Title 2"],
  "confidence": 0.0-1.0,
  "reasoning": "one-sentence internal reasoning"
}

Critical rules:
1. ONLY use information from the provided context. NEVER use your training knowledge.
2. If the context answers the question → answer_type = "answered". Cite exact source titles.
3. If the question contains a FALSE claim about FastAPI (e.g., "since FastAPI uses XML by default") → answer_type = "false_premise". Explain the false assumption in "answer".
4. If the answer is NOT present in the context, even though the topic is related → answer_type = "unanswerable".
5. NEVER answer from training knowledge even if you know the answer. If it's not in the context, refuse.
6. For multi-section questions, synthesize across sections and cite all relevant ones.
7. False-premise detection: watch for claims that FastAPI requires X, uses Y by default, or has Z feature when the context shows otherwise.
"""


def build_context(chunks: List[Dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        parts.append(f"[{i}] SOURCE: {meta['source']}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def query(question: str) -> Dict[str, Any]:
    t0 = time.time()

    chunks, is_multi_section = retrieve_multi_section(question, top_k=8)
    if not chunks:
        return {
            "answer_type": "unanswerable",
            "answer": "No relevant documentation found.",
            "citations": [],
            "confidence": 0.0,
            "cost_usd": 0.0,
            "latency_ms": int((time.time() - t0) * 1000),
            "retrieved_chunks": 0,
            "top_retrieval_score": 0.0,
            "is_multi_section": False,
        }

    top_score = chunks[0]["score"]

    if top_score < REFUSAL_THRESHOLD:
        return {
            "answer_type": "unanswerable",
            "answer": "The question appears to be outside the scope of the FastAPI documentation.",
            "citations": [],
            "confidence": top_score,
            "cost_usd": 0.0,
            "latency_ms": int((time.time() - t0) * 1000),
            "retrieved_chunks": len(chunks),
            "top_retrieval_score": round(top_score, 4),
            "is_multi_section": is_multi_section,
        }

    context = build_context(chunks)

    user_msg = f"""Context from FastAPI documentation:

{context}

---

Question: {question}

Respond ONLY with valid JSON. No text outside the JSON."""

    input_tokens = count_tokens(SYSTEM_PROMPT + user_msg)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=600,
        response_format={"type": "json_object"},
    )

    output_tokens = response.usage.completion_tokens
    cost = (input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN)
    latency_ms = int((time.time() - t0) * 1000)

    raw = response.choices[0].message.content
    try:
        result = json.loads(raw)
    except Exception:
        result = {
            "answer_type": "unanswerable",
            "answer": "Failed to parse model response.",
            "citations": [],
            "confidence": 0.0,
            "reasoning": "parse error",
        }

    result["cost_usd"] = round(cost, 6)
    result["latency_ms"] = latency_ms
    result["retrieved_chunks"] = len(chunks)
    result["top_retrieval_score"] = round(top_score, 4)
    result["is_multi_section"] = is_multi_section

    return result
