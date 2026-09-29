import hashlib
import time

import httpx

from anchor.config import CACHE_DIR

HEADERS = {"User-Agent": "anchor-rag/0.1 (personal project)"}


def get(url: str, pause: float = 0.4) -> str:
    """Download a page, keeping a copy on disk so re-running ingest costs nothing."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".html")

    if cached.exists():
        return cached.read_text(encoding="utf-8")

    response = httpx.get(url, headers=HEADERS, follow_redirects=True, timeout=30)
    response.raise_for_status()
    cached.write_text(response.text, encoding="utf-8")

    time.sleep(pause)
    return response.text
