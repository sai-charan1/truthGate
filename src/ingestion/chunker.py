import json
import hashlib
from pathlib import Path
from typing import List, Dict

RAW_FILE = Path("data/raw/fastapi_docs.json")
CHUNKS_FILE = Path("data/chunks/chunks.json")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 60:
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_chunks() -> List[Dict]:
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_FILE) as f:
        pages = json.load(f)

    chunks = []
    for page in pages:
        url = page["url"]
        title = page["title"]
        for section in page.get("sections", []):
            heading = section["heading"]
            text = section["text"]
            if not text or len(text.split()) < 15:
                continue
            text_chunks = chunk_text(text)
            for i, chunk_text_part in enumerate(text_chunks):
                chunk_id = hashlib.md5(f"{url}:{heading}:{i}".encode()).hexdigest()[:12]
                chunks.append({
                    "id": chunk_id,
                    "url": url,
                    "page_title": title,
                    "section_heading": heading,
                    "text": chunk_text_part,
                    "source": f"{title} > {heading}",
                    "chunk_index": i,
                })

    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Built {len(chunks)} chunks -> {CHUNKS_FILE}")
    return chunks


if __name__ == "__main__":
    build_chunks()
