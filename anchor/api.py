from fastapi import FastAPI
from pydantic import BaseModel

from anchor.answer import ask
from anchor.assess import assess

app = FastAPI(title="Anchor")


class Question(BaseModel):
    question: str
    corpus: str = "aiact"
    k: int = 5
    hop: bool = True
    gate: bool = True
    check: bool = True


class Source(BaseModel):
    marker: int
    citation: str
    score: float
    followed_from: str | None
    url: str
    text: str


class Claim(BaseModel):
    claim: str
    supported: bool
    quote: str


class Reply(BaseModel):
    question: str
    answer: str
    refused: bool
    reason: str
    sources: list[Source]
    claims: list[Claim]
    trace: list[str]
    seconds: float


class Description(BaseModel):
    description: str
    corpus: str = "aiact"


class Duty(BaseModel):
    obligation: str
    requires: str
    citation: str


class Assessed(BaseModel):
    summary: str
    obligations: list[Duty]
    searches: list[str]


@app.post("/ask", response_model=Reply)
def post_ask(body: Question) -> Reply:
    result = ask(body.question, corpus=body.corpus, k=body.k,
                 hop=body.hop, gate=body.gate, check=body.check)
    return Reply(
        question=result.question,
        answer=result.text,
        refused=result.refused,
        reason=result.reason,
        sources=[
            Source(marker=n, citation=h.citation, score=round(h.score, 3),
                   followed_from=h.via, url=h.url, text=h.text)
            for n, h in enumerate(result.hits, 1)
        ],
        claims=[Claim(claim=c.claim, supported=c.supported, quote=c.quote)
                for c in result.checks],
        trace=result.trace,
        seconds=round(result.seconds, 2),
    )


@app.post("/assess", response_model=Assessed)
def post_assess(body: Description) -> Assessed:
    report = assess(body.description, corpus=body.corpus)
    return Assessed(
        summary=report.summary,
        obligations=[Duty(**item) for item in report.obligations],
        searches=report.searches,
    )
