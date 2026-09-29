import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://anchor:anchor@localhost:5433/anchor")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

CACHE_DIR = ROOT / "data" / "raw"

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# Two tiers. The big model writes answers, the small one does the mechanical jobs
# (judging, splitting text into claims, rewriting a question) where speed and cost
# matter more than eloquence.
WRITER_MODEL = "openai/gpt-oss-120b"
WORKER_MODEL = "openai/gpt-oss-20b"

TOP_K = 5
# Distances below this are clearly relevant, above it clearly not. Anything in
# between is what actually gets sent to the judge.
CONFIDENT_DISTANCE = 0.55
HOPELESS_DISTANCE = 0.85
MAX_REWRITES = 1
