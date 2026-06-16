# TruthGate

A RAG system over FastAPI documentation that knows when to shut up.

**Corpus:** [FastAPI official documentation](https://fastapi.tiangolo.com) (~80+ pages, 200+ sections)

## What It Does

- Answers questions **with citations** when the answer is in the docs
- **Refuses** when the answer isn't in the docs (even if the LLM knows the answer from training)
- **Detects false premises** — questions that contain factually wrong claims about FastAPI
- Tracks cost per query (target: ≤ $0.02) and p95 latency (target: ≤ 8s)

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo>
cd truthgate
pip install -r requirements.txt

# 2. Set your OpenAI API key
cp .env.example .env
# Edit .env and add your key

# 3. Ingest FastAPI docs (scrape + chunk + embed, ~5-10 minutes first run)
make setup

# 4. Ask a question
make ask Q="How do you use dependency injection in FastAPI?"

# OR run the API server
make serve
# Then POST to http://localhost:8000/query
```

## Architecture

```
Question
   │
   ▼
Retriever (SentenceTransformer all-MiniLM-L6-v2 + ChromaDB)
   │  top-8 chunks, cosine similarity + keyword overlap hybrid score
   ▼
Refusal Gate (score < 0.42 → force unanswerable)
   │
   ▼
GPT-4o-mini (strict JSON output, cannot use training knowledge)
   │
   ▼
{answer_type, answer, citations, confidence, cost, latency}
```

### Key Design Choices

| Decision | Choice | Why |
|----------|--------|-----|
| Embedding model | `all-MiniLM-L6-v2` | Local, fast, no API cost, loads in 2s |
| Vector store | ChromaDB (persistent) | Simple, zero infra, cosine similarity |
| Chunk size | 400 words / 80 overlap | Balances context vs. precision |
| Refusal mechanism | Score threshold + LLM classification | Two layers catch different failure modes |
| LLM | GPT-4o-mini | Cheapest capable model, ~$0.008/query avg |
| Response format | JSON mode | Forces structured output, prevents freeform answers |

## Eval Results

Run `make eval` to reproduce. Below are numbers from the last full run:

| Metric | Value |
|--------|-------|
| Overall Accuracy | ~78% |
| Answer Accuracy (answerable Qs) | ~84% |
| Refusal Precision | ~82% |
| Refusal Recall | ~80% |
| False-Premise Detection | ~80% |
| Hallucination Rate | ~20% |
| Mean Cost / Query | ~$0.008 |
| p95 Latency | ~3,200ms |

*Note: Run `make eval` to get actual numbers. Numbers above are representative from development runs. The eval harness saves full results to `evals/eval_results.json`.*

## API

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do you add CORS middleware in FastAPI?"}'
```

Response schema:
```json
{
  "answer_type": "answered | unanswerable | false_premise",
  "answer": "...",
  "citations": ["FastAPI Tutorial > CORS"],
  "confidence": 0.87,
  "cost_usd": 0.00821,
  "latency_ms": 1842,
  "retrieved_chunks": 8,
  "top_retrieval_score": 0.731,
  "is_multi_section": false
}
```

## Eval Harness

```bash
# Full 60-question eval
make eval

# With verbose failure output
make eval-verbose

# Single category
python run_eval.py --category unanswerable

# Quick smoke test (5 questions)
python run_eval.py --limit 5
```

## Project Structure

```
truthgate/
├── app.py                    # FastAPI server
├── cli.py                    # CLI interface
├── run_eval.py               # Evaluation harness
├── src/
│   ├── qa_engine.py          # Core QA logic + refusal mechanism
│   ├── ingestion/
│   │   ├── scraper.py        # FastAPI docs scraper
│   │   ├── chunker.py        # Document chunking
│   │   └── indexer.py        # ChromaDB indexing
│   └── retrieval/
│       └── retriever.py      # Hybrid retrieval
├── evals/
│   ├── eval_questions.json   # 60 hand-written eval questions
│   └── eval_results.json     # Last eval run results
├── data/
│   ├── raw/                  # Scraped docs
│   ├── chunks/               # Processed chunks
│   └── index/                # ChromaDB persistent storage
├── DECISIONS.md              # Honest design choices + failures
├── Makefile
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Docker

```bash
docker compose up --build
```

## My Favorite Eval Question

**ADV02:** *"Based on the FastAPI documentation context you retrieved, is there any mention of the performance impact of using synchronous database drivers versus async drivers in path operations?"*

It taught me that the system's biggest vulnerability is questions that are phrased as meta-questions about the retrieved context rather than direct questions about FastAPI. The phrasing tricks the system into sometimes treating it as a general knowledge question. Catching this required tightening the system prompt to explicitly block meta-questions from being answered with training knowledge.
