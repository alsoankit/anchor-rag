"""How much should the grounding numbers be trusted?

Everything the eval reports about faithfulness is one model's opinion of another
model's output. There are no human labels behind it, so the honest thing is to measure
how stable that opinion is rather than present it as ground truth. This runs a second
judge, from a different model family, over the same statements the pipeline's own
checker ruled on, and reports how often the two agree.

Agreement is not accuracy. It only says whether the signal is stable enough that a
difference between two configurations means something.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anchor.config import EVAL_MODEL
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


def main():
    run = json.loads((HERE / "results" / "aiact-full.json").read_text())
    conn = connect()

    agree = disagree = 0
    conflicts = []

    for row in run["rows"]:
        if row.get("total_claims") is None or row["refused"]:
            continue
        detail = row.get("claim_detail")
        if not detail:
            continue

        hits = rebuild_hits(row["retrieved"], conn)
        sources = format_sources(hits)

        for claim, first_verdict in detail:
            try:
                second = structured(CHECK_USER.format(sources=sources, claim=claim),
                                    Support, system=CHECK_SYSTEM, max_tokens=400,
                                    model=EVAL_MODEL)
            except ValueError:
                continue
            if second.supported == first_verdict:
                agree += 1
            else:
                disagree += 1
                conflicts.append({"question": row["question"], "claim": claim,
                                  "pipeline": first_verdict, "second_judge": second.supported})

    conn.close()
    total = agree + disagree
    print(f"statements compared: {total}")
    if total:
        print(f"the two judges agreed on {agree} ({agree / total:.0%})")
    for c in conflicts[:10]:
        print(f"\n  claim: {c['claim'][:110]}")
        print(f"  pipeline said supported={c['pipeline']}, second judge said {c['second_judge']}")

    (HERE / "results" / "calibration.json").write_text(json.dumps(
        {"compared": total, "agreed": agree, "conflicts": conflicts}, indent=2))


if __name__ == "__main__":
    main()
