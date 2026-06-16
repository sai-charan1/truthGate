import json
import pickle
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = Path("data/index")

_vectorizer = None
_tfidf_matrix = None
_chunks = None


def _load_index():
    global _vectorizer, _tfidf_matrix, _chunks
    if _vectorizer is None:
        with open(INDEX_DIR / "vectorizer.pkl", "rb") as f:
            _vectorizer = pickle.load(f)
        with open(INDEX_DIR / "tfidf_matrix.pkl", "rb") as f:
            _tfidf_matrix = pickle.load(f)
        with open(INDEX_DIR / "chunks_meta.json") as f:
            _chunks = json.load(f)


def keyword_overlap_score(text: str, query: str) -> float:
    query_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
    doc_terms = set(re.findall(r'\b\w{3,}\b', text.lower()))
    if not query_terms:
        return 0.0
    return len(query_terms & doc_terms) / len(query_terms)


def retrieve(query: str, top_k: int = 8) -> List[Dict]:
    _load_index()

    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _tfidf_matrix)[0]

    top_indices = np.argsort(scores)[::-1][:top_k * 2]

    results = []
    for idx in top_indices:
        chunk = _chunks[idx]
        tfidf_score = float(scores[idx])
        kw_score = keyword_overlap_score(chunk["text"], query)
        combined = 0.75 * tfidf_score + 0.25 * kw_score
        results.append({
            "text": chunk["text"],
            "metadata": {
                "url": chunk["url"],
                "page_title": chunk["page_title"],
                "section_heading": chunk["section_heading"],
                "source": chunk["source"],
            },
            "tfidf_score": tfidf_score,
            "keyword_score": kw_score,
            "score": combined,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def retrieve_multi_section(query: str, top_k: int = 8) -> Tuple[List[Dict], bool]:
    results = retrieve(query, top_k=top_k)
    unique_sections = {r["metadata"]["section_heading"] for r in results}
    return results, len(unique_sections) >= 2
