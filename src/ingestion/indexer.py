import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNKS_FILE = Path("data/chunks/chunks.json")
INDEX_DIR = Path("data/index")


def build_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    print(f"Building TF-IDF index over {len(chunks)} chunks...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    with open(INDEX_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(INDEX_DIR / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(INDEX_DIR / "chunks_meta.json", "w") as f:
        json.dump(chunks, f)

    print(f"Index built: {tfidf_matrix.shape[0]} docs, {tfidf_matrix.shape[1]} features -> {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
