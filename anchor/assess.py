from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from anchor.corpora import ABOUT
from anchor.llm import structured
from anchor.prompts import ASSESS_SYSTEM, ASSESS_USER
from anchor.retrieve import retrieve
from anchor.verify import format_sources

PLAN_SYSTEM = """You turn a description of a system into searches over a document collection.

The description is written the way its builder would describe it. The documents are
written formally and organised by topic, so a single search over the whole description
matches everything weakly and nothing well. Break it into a few searches, each aimed at
one aspect that the documents would have a section about."""

PLAN_USER = """Collection: {about}

The system: {description}

Give between three and five searches."""


class Plan(BaseModel):
    searches: list[str] = Field(default_factory=list)


class Obligation(BaseModel):
    obligation: str
    requires: str
    source: int


class Assessment(BaseModel):
    summary: str
    obligations: list[Obligation] = Field(default_factory=list)


@dataclass
class Report:
    description: str
    summary: str
    obligations: list = field(default_factory=list)
    hits: list = field(default_factory=list)
    searches: list[str] = field(default_factory=list)


def plan(description: str, about: str) -> list[str]:
    try:
        result = structured(
            PLAN_USER.format(about=about, description=description),
            Plan, system=PLAN_SYSTEM, max_tokens=400,
        )
    except ValueError:
        return [description]
    return [s.strip() for s in result.searches if s.strip()][:5]


def assess(description: str, corpus: str = "aiact", strategy: str = "structural",
           per_search: int = 4, conn=None) -> Report:
    """Describe a system in plain words, get back the duties the documents put on it.

    This is the part that is not question answering. Nobody asks a document collection
    "what applies to me", because the answer is spread across provisions that each cover
    one aspect and none of which mention the system being described. So the description
    is broken into searches first, each aspect is retrieved separately, and the writing
    step works over the union.
    """
    about = ABOUT.get(corpus, corpus)
    searches = plan(description, about)

    gathered: dict[int, object] = {}
    for search in searches:
        for hit in retrieve(search, corpus, strategy, per_search, hop=True, conn=conn):
            gathered.setdefault(hit.chunk_id, hit)

    hits = sorted(gathered.values(), key=lambda h: h.score, reverse=True)[:12]

    result = structured(
        ASSESS_USER.format(sources=format_sources(hits), description=description),
        Assessment, system=ASSESS_SYSTEM, max_tokens=1600,
    )

    obligations = []
    for item in result.obligations:
        index = item.source - 1
        citation = hits[index].citation if 0 <= index < len(hits) else "unknown"
        obligations.append({"obligation": item.obligation, "requires": item.requires,
                            "citation": citation})

    return Report(description=description, summary=result.summary,
                  obligations=obligations, hits=hits, searches=searches)
