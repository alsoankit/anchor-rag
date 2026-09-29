from dataclasses import dataclass

from anchor.store import connect, embed

SELECT = """
    select id, unit_id, citation, title, text, url, 1 - (embedding <=> %s) as similarity
    from chunks
    where corpus = %s and strategy = %s
"""


@dataclass
class Hit:
    chunk_id: int
    unit_id: str
    citation: str
    title: str
    text: str
    url: str
    score: float
    via: str | None = None

    @property
    def followed(self) -> bool:
        return self.via is not None


def _rows_to_hits(rows) -> list[Hit]:
    return [Hit(chunk_id=r[0], unit_id=r[1], citation=r[2], title=r[3],
                text=r[4], url=r[5], score=float(r[6])) for r in rows]


def search(question: str, corpus: str, strategy: str = "structural", k: int = 5,
           conn=None) -> list[Hit]:
    vector = embed([question])[0]
    own = conn is None
    conn = conn or connect()
    try:
        rows = conn.execute(
            SELECT + " order by embedding <=> %s limit %s",
            (vector, corpus, strategy, vector, k),
        ).fetchall()
    finally:
        if own:
            conn.close()
    return _rows_to_hits(rows)


def follow(seeds: list[Hit], question: str, corpus: str, strategy: str = "structural",
           k: int = 5, decay: float = 0.75, breadth: int = 3, conn=None) -> list[Hit]:
    """Pull in whatever the strongest hits point at, then rank everything together.

    A referenced provision usually shares almost no wording with the question, so on
    similarity alone it never surfaces and never survives a rerank either. So instead
    of scoring it on its own, it inherits a faded copy of the score of whatever cited
    it, and keeps the better of the two. Relevance spreads outward along citations and
    gets weaker with distance.
    """
    if not seeds:
        return seeds

    vector = embed([question])[0]
    own = conn is None
    conn = conn or connect()
    try:
        # Only the best few seeds get to drag things in. Letting every hit expand turns
        # a five document context into forty and the generator drowns.
        parents = sorted(seeds, key=lambda h: h.score, reverse=True)[:breadth]
        already = {h.unit_id for h in seeds}

        inherited: dict[str, tuple[float, str]] = {}
        for parent in parents:
            refs = conn.execute(
                "select to_unit from citations where corpus = %s and from_unit = %s",
                (corpus, parent.unit_id),
            ).fetchall()
            for (target,) in refs:
                if target in already:
                    continue
                passed = parent.score * decay
                if passed > inherited.get(target, (0.0, ""))[0]:
                    inherited[target] = (passed, parent.citation)

        extra: list[Hit] = []
        for unit_id, (passed, via) in inherited.items():
            rows = conn.execute(
                SELECT + " and unit_id = %s order by embedding <=> %s limit 2",
                (vector, corpus, strategy, unit_id, vector),
            ).fetchall()
            for hit in _rows_to_hits(rows):
                hit.score = max(hit.score, passed)
                hit.via = via
                extra.append(hit)
    finally:
        if own:
            conn.close()

    merged = {h.chunk_id: h for h in seeds}
    for hit in extra:
        if hit.chunk_id not in merged:
            merged[hit.chunk_id] = hit

    return sorted(merged.values(), key=lambda h: h.score, reverse=True)[:k]


def retrieve(question: str, corpus: str, strategy: str = "structural", k: int = 5,
             hop: bool = True, conn=None) -> list[Hit]:
    seeds = search(question, corpus, strategy, k, conn=conn)
    if not hop:
        return seeds
    return follow(seeds, question, corpus, strategy, k, conn=conn)
