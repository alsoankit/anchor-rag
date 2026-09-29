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


DECLINE_SYSTEM = """You decide whether a reply actually answered the question or backed out of it.

Backing out covers saying the sources do not cover it, saying it cannot be determined from
what was given, or apologising and stopping. Giving the answer and then adding a caveat is
still answering. Only judge whether an answer was given, never whether it was correct."""

DECLINE_USER = """Question: {question}

Reply: {answer}"""


class Declined(BaseModel):
    declined: bool


def declined(question: str, answer: str) -> bool:
    """Whether the system backed out, judged by reading the reply rather than grepping it.

    This started as a string check for the NOT IN SOURCES token the grounded prompt is told
    to emit, which worked fine until the naive baseline was added. That one has no refusal
    instruction, so when it had nothing to work with it declined in ordinary English and
    the string check counted every one of those as an answer. The baseline looked like it
    was inventing things when it was doing the opposite.

    The lesson is that the metric was measuring a prompt's vocabulary rather than the
    behaviour it was supposed to be measuring, and the two only lined up by accident while
    every configuration shared one prompt.
    """
    if not answer.strip():
        return True
    try:
        return structured(DECLINE_USER.format(question=question, answer=answer),
                          Declined, system=DECLINE_SYSTEM, max_tokens=200,
                          model=EVAL_MODEL).declined
    except ValueError:
        return False
