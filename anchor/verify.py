import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pydantic import BaseModel, Field

from anchor.llm import structured
from anchor.prompts import CHECK_SYSTEM, CHECK_USER, CLAIMS_SYSTEM, CLAIMS_USER


class Claims(BaseModel):
    claims: list[str] = Field(default_factory=list)


class Support(BaseModel):
    supported: bool
    quote: str = ""


@dataclass
class Checked:
    claim: str
    supported: bool
    quote: str


def format_sources(hits) -> str:
    blocks = []
    for n, hit in enumerate(hits, 1):
        blocks.append(f"[{n}] {hit.citation} — {hit.title}\n{hit.text}")
    return "\n\n".join(blocks)


def split_claims(answer: str) -> list[str]:
    try:
        result = structured(CLAIMS_USER.format(answer=answer), Claims, system=CLAIMS_SYSTEM)
    except ValueError:
        # Falling back to sentences is worse than what the model does, because pronouns
        # stay unresolved and a sentence can carry two claims. But checking a rough split
        # is much better than the whole verification step disappearing on one bad reply.
        rough = re.split(r"(?<=[.!?])\s+", re.sub(r"\[\d+\]", "", answer))
        return [s.strip() for s in rough if len(s.strip()) > 25]
    return [c.strip() for c in result.claims if c.strip()]


def verify(answer: str, hits) -> list[Checked]:
    """Check the answer against the sources one statement at a time.

    The model is asked to cite as it writes, but a citation marker is just text it
    produced and it will attach one to a sentence the source does not support. So the
    answer gets taken apart and every piece is put back in front of the sources on its
    own, without the rest of the answer around it making it sound plausible.
    """
    claims = split_claims(answer)
    if not claims:
        return []

    sources = format_sources(hits)

    def check(claim: str) -> Checked:
        try:
            result = structured(
                CHECK_USER.format(sources=sources, claim=claim),
                Support, system=CHECK_SYSTEM, max_tokens=400,
            )
        except ValueError:
            # A claim the checker cannot produce a verdict on is treated as unsupported.
            # Failing the other way would let anything through on a bad JSON reply.
            return Checked(claim=claim, supported=False, quote="")
        return Checked(claim=claim, supported=result.supported, quote=result.quote.strip())

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(check, claims))
