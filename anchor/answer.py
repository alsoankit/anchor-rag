import re
import time
from dataclasses import dataclass, field

from pydantic import BaseModel

from anchor.config import CONFIDENT_SCORE, HOPELESS_SCORE, MAX_REWRITES, TOP_K
from anchor.corpora import ABOUT
from anchor.llm import structured, write
from anchor.prompts import (ANSWER_SYSTEM, ANSWER_USER, JUDGE_SYSTEM, JUDGE_USER,
                            NAIVE_SYSTEM, REWRITE_SYSTEM, REWRITE_USER)
from anchor.retrieve import retrieve
from anchor.verify import Checked, format_sources, verify

REFUSAL = "I don't have enough in the sources to answer that."


class Verdict(BaseModel):
    can_answer: bool
    reason: str = ""


class Rewritten(BaseModel):
    question: str


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    reason: str = ""
    hits: list = field(default_factory=list)
    checks: list[Checked] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def unsupported(self) -> list[Checked]:
        return [c for c in self.checks if not c.supported]


def enough_evidence(question: str, hits, trace: list[str]) -> bool:
    """Two stage check, cheap part first.

    Scoring every question with a model call is slow and mostly wasted, because the
    clearly good and clearly hopeless cases are obvious from the similarity alone. Only
    the middle is genuinely ambiguous, and that is the only place worth paying for.
    """
    if not hits:
        trace.append("gate: nothing retrieved")
        return False

    best = hits[0].score
    if best >= CONFIDENT_SCORE:
        trace.append(f"gate: passed on score alone ({best:.2f})")
        return True
    if best < HOPELESS_SCORE:
        trace.append(f"gate: rejected on score alone ({best:.2f})")
        return False

    try:
        verdict = structured(
            JUDGE_USER.format(sources=format_sources(hits), question=question),
            Verdict, system=JUDGE_SYSTEM, max_tokens=300,
        )
    except ValueError:
        # If the judge cannot be reached the score is all there is, so fall back to it
        # rather than refusing outright. Refusing on an infrastructure problem would
        # look identical to refusing on weak evidence, which makes the metric a lie.
        trace.append(f"gate: judge unavailable, fell back to the score ({best:.2f})")
        return best >= (CONFIDENT_SCORE + HOPELESS_SCORE) / 2

    trace.append(f"gate: score {best:.2f} was borderline, judge said "
                 f"{'yes' if verdict.can_answer else 'no'}")
    return verdict.can_answer


def rewrite(question: str, corpus: str) -> str:
    try:
        result = structured(
            REWRITE_USER.format(about=ABOUT.get(corpus, corpus), question=question),
            Rewritten, system=REWRITE_SYSTEM, max_tokens=200,
        )
    except ValueError:
        return question
    return result.question.strip() or question


def ask(question: str, corpus: str = "aiact", strategy: str = "structural",
        k: int = TOP_K, hop: bool = True, gate: bool = True, check: bool = True,
        grounded_prompt: bool = True, conn=None) -> Answer:
    started = time.time()
    trace: list[str] = []
    asked = question

    hits = retrieve(asked, corpus, strategy, k, hop=hop, conn=conn)
    if hits:
        trace.append(f"retrieved {len(hits)} chunks, best score {hits[0].score:.2f}")
    else:
        trace.append("retrieved nothing")
    if hop:
        followed = [h.citation for h in hits if h.followed]
        if followed:
            trace.append(f"followed citations into {', '.join(followed)}")

    if gate:
        attempts = 0
        while not enough_evidence(asked, hits, trace):
            if attempts >= MAX_REWRITES:
                return Answer(question=question, text=REFUSAL, refused=True,
                              reason="the retrieved sources do not cover it",
                              hits=hits, trace=trace, seconds=time.time() - started)
            attempts += 1
            asked = rewrite(asked, corpus)
            trace.append(f"rewrote the question as: {asked}")
            hits = retrieve(asked, corpus, strategy, k, hop=hop, conn=conn)

    text = write(
        ANSWER_USER.format(sources=format_sources(hits), question=question),
        system=ANSWER_SYSTEM if grounded_prompt else NAIVE_SYSTEM,
    )

    text = re.sub(r"【(\d+(?:\]\[|\s*,\s*)?\d*)】", r"[\1]", text)

    if "NOT IN SOURCES" in text.upper():
        trace.append("the writer refused after reading the sources")
        return Answer(question=question, text=REFUSAL, refused=True,
                      reason="the sources were retrieved but do not answer it",
                      hits=hits, trace=trace, seconds=time.time() - started)

    checks = verify(text, hits) if check else []
    if checks:
        bad = sum(1 for c in checks if not c.supported)
        trace.append(f"checked {len(checks)} statements, {bad} unsupported")

    return Answer(question=question, text=text, refused=False, hits=hits,
                  checks=checks, trace=trace, seconds=time.time() - started)
