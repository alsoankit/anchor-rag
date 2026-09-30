"""How much should the grounding numbers be trusted?

Everything the eval reports about faithfulness is one model's opinion of another model's
output. There are no human labels behind it, so the honest thing is to measure how stable
that opinion is rather than present it as ground truth.

Three judges rule on the same statements. The pipeline's own checker, a second open-weight
model from a different family, and a Gemini model from a different provider entirely. The
third one matters more than it looks: the first two are both open-weight models served
through the same API, so their agreeing could easily be a shared-lineage artefact rather
than the statements being genuinely clear-cut. A judge trained by someone else, running
somewhere else, is the only one that can rule that out.

Agreement is still not accuracy. It only says whether the signal is stable enough that a
difference between two configurations means something rather than noise.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anchor.config import EVAL_MODEL, THIRD_MODEL, THIRD_SAMPLE, WORKER_MODEL
from anchor.llm import structured
from anchor.prompts import CHECK_SYSTEM, CHECK_USER
from anchor.store import connect
from anchor.verify import Support, format_sources
from anchor.retrieve import Hit

HERE = Path(__file__).resolve().parent


def rebuild_hits(chunk_ids: list[int], conn) -> list[Hit]:
    """The exact chunks the answer was generated from, by id.

    This used to look up one chunk per unit with no ordering, which quietly handed the
    judges a different paragraph from the one the model had read. Article 99 has nineteen
    chunks; the reconstruction picked the first and the answer had been written from the
    one with the fine in it, so every judge but the pipeline's own was grading against
    evidence that did not contain the answer. They all said unsupported, agreed with each
    other, and the agreement looked like a finding.
    """
    hits = []
    for chunk_id in chunk_ids:
        row = conn.execute(
            """select id, unit_id, citation, title, text, url from chunks where id = %s""",
            (chunk_id,)).fetchone()
        if row:
            hits.append(Hit(chunk_id=row[0], unit_id=row[1], citation=row[2],
                            title=row[3], text=row[4], url=row[5], score=0.0))
    return hits


def judge(claim: str, sources: str, model: str) -> bool | None:
    try:
        result = structured(CHECK_USER.format(sources=sources, claim=claim),
                            Support, system=CHECK_SYSTEM, max_tokens=400, model=model)
    except Exception:
        return None
    return result.supported


def main():
    run = json.loads((HERE / "results" / "aiact-full.json").read_text())
    conn = connect()

    jobs = []
    for row in run["rows"]:
        if row["refused"] or not row.get("claim_detail") or not row.get("chunk_ids"):
            continue
        sources = format_sources(rebuild_hits(row["chunk_ids"], conn))
        for claim, pipeline_said in row["claim_detail"]:
            jobs.append({"question": row["question"], "claim": claim,
                         "pipeline": pipeline_said, "sources": sources})
    conn.close()

    if not jobs:
        print("nothing to compare — run the full config first")
        return

    # The third judge is on a free tier that allows twenty calls a day, so it grades an
    # evenly spaced sample rather than everything. Spacing it rather than taking the first
    # N keeps the sample spread across questions instead of concentrated in the first two.
    step = max(len(jobs) // THIRD_SAMPLE, 1)
    sampled = {i for i in range(0, len(jobs), step)}

    def ask(pair):
        i, job = pair
        with ThreadPoolExecutor(max_workers=2) as inner:
            second = inner.submit(judge, job["claim"], job["sources"], EVAL_MODEL)
            third = inner.submit(judge, job["claim"], job["sources"], THIRD_MODEL) \
                if i in sampled else None
            return second.result(), (third.result() if third else None)

    with ThreadPoolExecutor(max_workers=6) as pool:
        answers = list(pool.map(ask, enumerate(jobs)))

    for job, (second, third) in zip(jobs, answers):
        job["second"], job["third"] = second, third

    pair_two = [j for j in jobs if j["second"] is not None]
    trio = [j for j in jobs if j["second"] is not None and j["third"] is not None]

    def agree(items, a, b):
        return sum(1 for j in items if j[a] == j[b]) / len(items) if items else None

    print(f"claims judged: {len(jobs)}")
    print()
    print(f"full comparison, both Groq judges, {len(pair_two)} claims")
    print(f"  pipeline ({WORKER_MODEL}) vs second ({EVAL_MODEL}): {agree(pair_two, 'pipeline', 'second'):.0%}")
    print()
    if trio:
        unanimous = sum(1 for j in trio if j["pipeline"] == j["second"] == j["third"])
        print(f"sample cross-checked against a third provider, {len(trio)} claims ({THIRD_MODEL})")
        print(f"  pipeline vs third:  {agree(trio, 'pipeline', 'third'):.0%}")
        print(f"  second vs third:    {agree(trio, 'second', 'third'):.0%}")
        print(f"  all three agreed:   {unanimous}/{len(trio)} ({unanimous / len(trio):.0%})")
    else:
        print("third judge unavailable (daily quota); two-judge comparison only")

    split = [j for j in pair_two if j["pipeline"] != j["second"]]
    print(f"\nthe two Groq judges disagreed on {len(split)} claims:")
    for j in split[:8]:
        print(f"\n  claim: {j['claim'][:105]}")
        print(f"  pipeline={j['pipeline']}  second={j['second']}  third={j['third']}")

    for j in jobs:
        j.pop("sources", None)
    (HERE / "results" / "calibration.json").write_text(json.dumps({
        "claims": len(jobs),
        "compared_two": len(pair_two),
        "compared_three": len(trio),
        "pipeline_vs_second": agree(pair_two, "pipeline", "second"),
        "pipeline_vs_third": agree(trio, "pipeline", "third"),
        "second_vs_third": agree(trio, "second", "third"),
        "verdicts": jobs,
    }, indent=2))


if __name__ == "__main__":
    main()
