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

from anchor.config import EVAL_MODEL, THIRD_MODEL, WORKER_MODEL
from anchor.llm import structured
from anchor.prompts import CHECK_SYSTEM, CHECK_USER
from anchor.store import connect
from anchor.verify import Support, format_sources
from anchor.retrieve import Hit

HERE = Path(__file__).resolve().parent


def rebuild_hits(unit_ids: list[str], conn) -> list[Hit]:
    hits = []
    for unit_id in unit_ids:
        row = conn.execute(
            """select id, unit_id, citation, title, text, url from chunks
               where corpus='aiact' and strategy='structural' and unit_id=%s limit 1""",
            (unit_id,)).fetchone()
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

    # Every claim, paired with the sources its answer was generated from.
    jobs = []
    for row in run["rows"]:
        if row["refused"] or not row.get("claim_detail"):
            continue
        sources = format_sources(rebuild_hits(row["retrieved"], conn))
        for claim, pipeline_said in row["claim_detail"]:
            jobs.append((row["question"], claim, pipeline_said, sources))

    # Judged in parallel. The first version walked the list one call at a time, twice
    # over, and a single request that never came back stalled the whole thing for half
    # an hour with nothing to show for it. verify.py already had a thread pool for
    # exactly this shape of work; this should have had one from the start.
    def ask(job):
        _, claim, _, sources = job
        with ThreadPoolExecutor(max_workers=2) as pair:
            second = pair.submit(judge, claim, sources, EVAL_MODEL)
            third = pair.submit(judge, claim, sources, THIRD_MODEL)
            return second.result(), third.result()

    with ThreadPoolExecutor(max_workers=6) as pool:
        answers = list(pool.map(ask, jobs))

    verdicts = []
    for (question, claim, pipeline_said, _), (second, third) in zip(jobs, answers):
        if second is None or third is None:
            continue
        verdicts.append({"question": question, "claim": claim,
                         "pipeline": pipeline_said, "second": second, "third": third})

    conn.close()

    if not verdicts:
        print("nothing to compare — run the full config first")
        return

    def agreement(a: str, b: str) -> float:
        return sum(1 for v in verdicts if v[a] == v[b]) / len(verdicts)

    unanimous = sum(1 for v in verdicts if v["pipeline"] == v["second"] == v["third"])

    print(f"statements compared: {len(verdicts)}")
    print(f"  pipeline ({WORKER_MODEL}) vs second ({EVAL_MODEL}): {agreement('pipeline', 'second'):.0%}")
    print(f"  pipeline vs third ({THIRD_MODEL}): {agreement('pipeline', 'third'):.0%}")
    print(f"  second vs third: {agreement('second', 'third'):.0%}")
    print(f"  all three agreed: {unanimous}/{len(verdicts)} ({unanimous / len(verdicts):.0%})")

    split = [v for v in verdicts if not (v["pipeline"] == v["second"] == v["third"])]
    print(f"\nstatements the judges split on ({len(split)}):")
    for v in split[:10]:
        print(f"\n  claim: {v['claim'][:110]}")
        print(f"  pipeline={v['pipeline']}  second={v['second']}  third={v['third']}")

    (HERE / "results" / "calibration.json").write_text(json.dumps({
        "compared": len(verdicts),
        "pipeline_vs_second": agreement("pipeline", "second"),
        "pipeline_vs_third": agreement("pipeline", "third"),
        "second_vs_third": agreement("second", "third"),
        "unanimous": unanimous,
        "verdicts": verdicts,
    }, indent=2))


if __name__ == "__main__":
    main()
