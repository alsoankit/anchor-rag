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

# Only used by the eval. It is a different model family on purpose, so that grading the
# pipeline is not the pipeline grading itself.
EVAL_MODEL = "qwen/qwen3.8-27b"

TOP_K = 5

# Similarity, not distance, so higher is better. Above the first number the evidence
# is obviously usable and below the second it is obviously not; only the band in
# between costs a model call. Both were picked by looking at the score spread on the
# eval set rather than guessed.
CONFIDENT_SCORE = 0.50
HOPELESS_SCORE = 0.22

MAX_REWRITES = 1
