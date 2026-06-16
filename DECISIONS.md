# DECISIONS.md

## 1. Three Things That Didn't Work

**Attempting cross-encoder reranking with ms-marco-MiniLM**
My first instinct was to do a two-stage retrieval: dense retrieval top-20, then rerank with a cross-encoder. Pulled in `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers. The reranking actually *hurt* refusal precision — the cross-encoder was trained on passage relevance, not on "does this chunk actually answer the question." It kept confidently surfacing semi-related passages for unanswerable questions, which pushed my refusal threshold decision in the wrong direction. Dropped it. Ended up doing a simpler weighted combo of cosine similarity + keyword overlap, which turned out more honest about what it didn't know.

**LLM-as-judge for refusal calibration**
Originally tried using a second LLM call to judge whether the retrieved context was "sufficient" to answer the question. The idea: ask GPT-4o-mini "given this context, can you answer X? yes/no." In practice, the model just... answered anyway. It's deeply trained to be helpful and it struggled to say "no I can't answer this from just context." The LLM-as-judge approach also doubled cost per query which blew the $0.02 budget immediately. Scrapped it. The current approach — a hard retrieval score threshold (0.42 cosine similarity) combined with a strict system prompt — works better and costs less.

**Sentence-level chunking**
First version of the chunker split on sentences using `nltk.sent_tokenize`. The FastAPI docs have a lot of short sentences that only make sense with surrounding context ("This will work." / "But only if you declare it."). Sentence-level chunks were too narrow — retrieval kept returning context-free fragments. Moved to fixed-size word-count chunks with overlap (400 words, 80 overlap), which preserves enough local context for the LLM to actually reason about. Still not perfect — see failure mode below.

## 2. Chunking Strategy and Its Failure Mode

**Strategy:** Fixed-size word chunks (400 words) with 80-word overlap, split at the section level (heading → text). Each chunk is tagged with page title, section heading, and URL for citation.

**Still-broken failure mode:** The chunks don't respect code block boundaries. FastAPI docs frequently have a pattern like: "Here's how you use it: [code example] The important thing here is X." If the code block gets split across chunk boundaries, the LLM sees either (a) prose with no code, or (b) code with no explanation. For answerable questions that require understanding a specific code pattern, this causes partial answers. I'd need a code-aware chunker that treats ` ```python ... ``` ` blocks as atomic units to fix this properly.

## 3. How Refusal Works and Where It Still Fails

**Mechanism:** Two-layer. First, semantic retrieval score — if the best-matched chunk has cosine similarity < 0.42, we override whatever the LLM said and force `unanswerable`. This catches "plausible but truly absent" questions pretty reliably. Second, the LLM itself gets a strict system prompt that says it cannot use training knowledge, must classify as `answered/unanswerable/false_premise`, and must return structured JSON. For false-premise detection, the LLM does the heavy lifting since it needs to understand what FastAPI *actually* does.

**Category where it still fails:** Adjacent technology questions. If you ask "How do you configure SQLAlchemy's connection pool timeout when using it with FastAPI?" — this is unanswerable from the docs, but the retrieval score is high (the docs do mention SQLAlchemy). The LLM sees real-looking context about SQLAlchemy + FastAPI and sometimes answers from training knowledge about SQLAlchemy specifically, even though that wasn't in the corpus. The threshold helps but doesn't fully catch these. I'd need explicit source-grounding verification to handle this properly.

## 4. One More Week + $500/month Budget

With a week and real budget, I'd:
- Switch from sentence-transformers (local) to OpenAI `text-embedding-3-small` for better retrieval quality. The budget covers ~15M tokens/month easily.
- Build a proper semantic cache using Redis with similarity matching on question embeddings. The same question asked 100 different ways shouldn't cost $0.02 every time.
- Add a retrieval audit layer: before calling the LLM, run a quick check asking whether any retrieved chunk explicitly mentions the key entity in the question. This would kill the "adjacent technology" hallucination problem.
- Instrument latency properly — the current logging is coarse-grained. I'd want per-step timing (embed query, vector search, LLM call) to know where the p95 budget is actually being spent.
- Expand the eval to 200 questions with a proper split and have a few teammates blind-label 40 of them to establish human baseline.

## 5. Shortcuts Taken Due to 24h Limit

The embedding model choice. I used `all-MiniLM-L6-v2` because it's small and fast — loads in ~2 seconds, runs inference locally, zero API cost. But it's an older model with a max sequence length of 256 tokens, which means long chunks get silently truncated during embedding. For a proper system I'd evaluate `bge-large-en-v1.5` or `text-embedding-3-small` and run actual retrieval quality benchmarks (recall@5 on a labeled set) before committing to an embedding model. I didn't have time for that comparison so I went with the safe/fast choice and acknowledged the truncation risk in a code comment.
