import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from anchor.answer import ask
from anchor.assess import assess

app = FastAPI(title="Anchor")
HERE = Path(__file__).resolve().parent.parent


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


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/ask/stream")
def ask_stream(question: str, corpus: str = "aiact", k: int = 8,
               hop: bool = True, gate: bool = True, check: bool = True):
    """The same pipeline, but reporting each stage as it reaches it.

    A full answer takes around ten seconds and nearly all of that is spent waiting on a
    model. Returning only at the end makes the system look like a black box that pauses.
    Server-sent events are the smallest thing that fixes it: the work runs on a thread,
    pushes a line per stage onto a queue, and this generator forwards them as they arrive.
    No websocket, no extra dependency, and the browser side is one EventSource.
    """
    events: queue.Queue = queue.Queue()

    def run():
        try:
            result = ask(question, corpus=corpus, k=k, hop=hop, gate=gate, check=check,
                         on_step=lambda stage, detail: events.put({"stage": stage,
                                                                   "detail": detail}))
            events.put({"stage": "result", "payload": {
                "answer": result.text,
                "refused": result.refused,
                "reason": result.reason,
                "seconds": round(result.seconds, 1),
                "sources": [{"marker": n, "citation": h.citation,
                             "score": round(h.score, 2), "followed_from": h.via,
                             "text": h.text[:300]}
                            for n, h in enumerate(result.hits, 1)],
                "claims": [{"claim": c.claim, "supported": c.supported, "quote": c.quote}
                           for c in result.checks],
                "trace": result.trace,
            }})
        except Exception as error:
            events.put({"stage": "error", "detail": str(error)[:300]})
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            item = events.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
