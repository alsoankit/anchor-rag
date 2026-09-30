import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://anchor:anchor@localhost:5433/anchor")
# More than one key because the daily token budget is enforced per organisation, and a
# single free tier account does not have enough of it to finish an eval run. Anything
# named GROQ_API_KEY, GROQ_API_KEY2, GROQ_API_KEY3 and so on gets picked up.
GROQ_API_KEYS = [
    value for name, value in sorted(os.environ.items())
    if name.startswith("GROQ_API_KEY") and value.strip()
]
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

CACHE_DIR = ROOT / "data" / "raw"

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# Two tiers. The big model writes answers, the small one does the mechanical jobs
# (judging, splitting text into claims, rewriting a question) where speed and cost
# matter more than eloquence.
WRITER_MODEL = os.getenv("ANCHOR_WRITER_MODEL", "openai/gpt-oss-120b")
WORKER_MODEL = os.getenv("ANCHOR_WORKER_MODEL", "openai/gpt-oss-20b")

# Only used by the eval. It is a different model family on purpose, so that grading the
# pipeline is not the pipeline grading itself.
EVAL_MODEL = "qwen/qwen3.8-27b"

# A third opinion for the calibration, and the one that carries the most weight because it
# is the only judge that is not an open-weight model served by the same provider as the
# pipeline. If all three agree, the agreement is not an artefact of a shared lineage.
#
# The free tier allows twenty requests per day per model, which is nowhere near enough to
# judge every claim. So this one grades a sample and the two Groq judges grade everything,
# and the write-up reports those as two separate numbers rather than pretending otherwise.
THIRD_SAMPLE = 18
THIRD_MODEL = "gemini-flash-lite-latest"

# Five was the guess, eight is what the retrieval sweep actually picked: with the
# citation hop on, k=8 finds the gold unit for every question in the eval set, while
# k=5 has to drop a direct hit to make room for a followed one. See eval/sweep.py.
TOP_K = 8

# Similarity, not distance, so higher is better. Above the first number the evidence
# is obviously usable and below the second it is obviously not; only the band in
# between costs a model call. Both were picked by looking at the score spread on the
# eval set rather than guessed.
CONFIDENT_SCORE = 0.50
HOPELESS_SCORE = 0.22

MAX_REWRITES = 1
