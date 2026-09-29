from pydantic import BaseModel

from anchor.config import EVAL_MODEL
from anchor.llm import structured
from anchor.verify import format_sources

FAITHFUL_SYSTEM = """You check an answer against the sources it was supposed to come from.

Count how many separate factual statements the answer makes, and how many of those the
sources actually support. A statement is supported only if the sources state it or state
something it follows directly from. Do not credit a statement for being true in general."""

FAITHFUL_USER = """Sources:

{sources}

Answer: {answer}"""


class Faithfulness(BaseModel):
    statements: int
    supported: int


def retrieval_hit(hits, gold: list[str]) -> bool:
    return bool(gold) and bool({h.unit_id for h in hits} & set(gold))


def gold_recall(hits, gold: list[str]) -> float:
    if not gold:
        return 0.0
    return len({h.unit_id for h in hits} & set(gold)) / len(gold)


def faithfulness(answer: str, hits) -> float:
    """Scored by the writer model, while the pipeline's own checker runs on the small one.

    Grading the system with the exact component it uses to police itself would only
    show that the component agrees with itself, so the eval deliberately asks a
    different model with a differently worded prompt.
    """
    if not answer.strip() or not hits:
        return 0.0
    try:
        result = structured(
            FAITHFUL_USER.format(sources=format_sources(hits), answer=answer),
            Faithfulness, system=FAITHFUL_SYSTEM, max_tokens=300, model=EVAL_MODEL,
        )
    except ValueError:
        return 0.0
    if result.statements <= 0:
        return 0.0
    return min(result.supported / result.statements, 1.0)
