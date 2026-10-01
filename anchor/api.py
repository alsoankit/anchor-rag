import json
import queue
import re
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
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


# Uploaded files, kept until the process restarts. Keyed by a token the browser hands back
# when it decides to go ahead, so parsing and ingesting are two separate decisions.
PENDING: dict = {}
UPLOADS = HERE / "data" / "uploads"

# Measured on this machine. Used only to tell the user roughly how long ingest will take,
# so being a little wrong is fine and being silent is not.
CHUNKS_PER_SECOND = 140
TOKENS_PER_QUESTION = 2200
TOKENS_PER_VERIFIED_QUESTION = 6800


@app.post("/upload")
async def upload(file: UploadFile):
    """Parse a PDF and report what ingesting it would involve. Nothing is stored yet."""
    from anchor import pdfs

    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "that is not a PDF")

    UPLOADS.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^a-z0-9]+", "-", Path(file.filename).stem.lower()).strip("-")[:40] or "upload"
    path = UPLOADS / f"{name}.pdf"
    path.write_bytes(await file.read())

    try:
        units, info = pdfs.load(path, name)
    except Exception as error:
        raise HTTPException(400, f"could not read that PDF: {str(error)[:150]}")
    if not units:
        raise HTTPException(400, "no readable text in that PDF - it may be scanned images")

    token = uuid.uuid4().hex
    PENDING[token] = {"name": name, "units": units, "info": info}

    return {
        "token": token,
        "corpus": name,
        "filename": file.filename,
        **info,
        "seconds": max(1, round(info["units"] / CHUNKS_PER_SECOND)),
        "ingest_tokens": 0,
        "tokens_per_question": TOKENS_PER_QUESTION,
        "tokens_per_verified_question": TOKENS_PER_VERIFIED_QUESTION,
    }


@app.get("/ingest/stream")
def ingest_stream(token: str):
    """Embed and store a parsed upload, reporting progress as it goes."""
    pending = PENDING.get(token)
    if not pending:
        raise HTTPException(404, "nothing pending for that token")

    from anchor.store import ingest_units

    events: queue.Queue = queue.Queue()

    def run():
        try:
            total = len(pending["units"])
            events.put({"stage": "start", "total": total, "detail":
                        f"embedding {total} chunks"})
            ingest_units(pending["name"], pending["units"],
                         on_progress=lambda done, n: events.put(
                             {"stage": "progress", "done": done, "total": n}))
            events.put({"stage": "done", "corpus": pending["name"],
                        "detail": f"{total} chunks ready"})
            PENDING.pop(token, None)
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
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/corpora")
def corpora():
    """Everything currently in the database, so the page can list it."""
    from anchor.corpora import ABOUT
    from anchor.store import connect

    conn = connect()
    try:
        rows = conn.execute(
            """select corpus, count(*) from chunks where strategy = 'structural'
               group by corpus order by corpus""").fetchall()
    finally:
        conn.close()
    return [{"id": c, "label": ABOUT.get(c, c), "chunks": n, "builtin": c in ABOUT}
            for c, n in rows]
