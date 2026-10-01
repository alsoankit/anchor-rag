import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from anchor.config import DATABASE_URL, EMBED_MODEL, ROOT
from anchor.corpora import REGISTRY, Unit

_model = None


def encoder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(texts: list[str]):
    # Normalised so cosine distance and dot product agree, which is what the
    # pgvector index is set up for.
    return encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def setup():
    # Plain connection here. Registering the vector type needs the extension to
    # already exist, and this is the function that creates it.
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute((ROOT / "schema.sql").read_text())
        conn.commit()


def fixed_size(units: list[Unit], size: int = 1000, overlap: int = 200) -> list[Unit]:
    """The naive splitter, kept so the structural one has something to be measured against.

    It runs over the same text a unit already holds, which means the only thing that
    changes between the two arms of the comparison is where the boundaries fall.
    """
    joined = " ".join(u.text for u in units)
    if not joined:
        return []

    first = units[0]
    step = size - overlap
    out = []
    for i in range(0, len(joined), step):
        window = joined[i: i + size]
        if len(window) < 100:
            continue
        out.append(Unit(
            unit_id=first.unit_id,
            label=first.label,
            title=first.title,
            part=f"chars {i}-{i + len(window)}",
            text=window,
            url=first.url,
            refs=(),
        ))
    return out


def ingest(corpus_name: str, strategy: str = "structural", limit: int | None = None):
    corpus = REGISTRY[corpus_name]()
    sources = corpus.sources()
    if limit:
        sources = sources[:limit]

    conn = connect()
    conn.execute("delete from chunks where corpus = %s and strategy = %s", (corpus_name, strategy))
    conn.execute("delete from citations where corpus = %s", (corpus_name,))
    conn.commit()

    total = 0
    for n, source in enumerate(sources, 1):
        units = corpus.units(source)
        if not units:
            continue

        edges = {(source, ref) for u in units for ref in u.refs if ref != source}
        if strategy != "structural":
            size, overlap = (int(x) for x in strategy.split("-")[1:])
            units = fixed_size(units, size, overlap)

        # The heading goes into the embedding but not into storage. A provision states
        # its rule without ever restating what it is a rule about, so on its own text it
        # only matches questions that happen to share its wording. Carrying the title
        # gives every chunk the topic it was written under.
        vectors = embed([f"{u.label}. {u.title}. {u.text}" for u in units])
        with conn.cursor() as cur:
            for unit, vector in zip(units, vectors):
                cur.execute(
                    """insert into chunks
                       (corpus, strategy, unit_id, label, title, part, citation, text, url, embedding)
                       values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (corpus_name, strategy, unit.unit_id, unit.label, unit.title,
                     unit.part, unit.citation, unit.text, unit.url, vector),
                )
            for from_unit, to_unit in edges:
                cur.execute(
                    "insert into citations values (%s, %s, %s) on conflict do nothing",
                    (corpus_name, from_unit, to_unit),
                )
        conn.commit()
        total += len(units)
        print(f"  [{n}/{len(sources)}] {source}: {len(units)} chunks", flush=True)

    conn.close()
    print(f"{corpus_name} / {strategy}: {total} chunks")


def ingest_units(corpus_name: str, units: list[Unit], on_progress=None, batch: int = 32):
    """Store units that were built elsewhere, reporting progress as it goes.

    The corpus adapters produce their units lazily per source, which suits a long ingest
    from the web. An uploaded file is already in hand, so it arrives here as one list and
    the only thing a caller wants to know is how far along it is.
    """
    conn = connect()
    conn.execute("delete from chunks where corpus = %s", (corpus_name,))
    conn.execute("delete from citations where corpus = %s", (corpus_name,))
    conn.commit()

    done = 0
    try:
        for start in range(0, len(units), batch):
            window = units[start:start + batch]
            vectors = embed([f"{u.label}. {u.title}. {u.text}" for u in window])
            with conn.cursor() as cur:
                for unit, vector in zip(window, vectors):
                    cur.execute(
                        """insert into chunks
                           (corpus, strategy, unit_id, label, title, part, citation, text,
                            url, embedding)
                           values (%s, 'structural', %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (corpus_name, unit.unit_id, unit.label, unit.title, unit.part,
                         unit.citation, unit.text, unit.url, vector),
                    )
                    for target in unit.refs:
                        cur.execute(
                            "insert into citations values (%s, %s, %s) on conflict do nothing",
                            (corpus_name, unit.unit_id, target),
                        )
            conn.commit()
            done += len(window)
            if on_progress:
                on_progress(done, len(units))
    finally:
        conn.close()
    return done
