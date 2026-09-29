"""Retrieval-only sweep over the settings that decide what reaches the model.

Everything measured here is embeddings and SQL, so it costs nothing to run and can be
repeated as often as the questions change. That is the point of separating it from the
main eval: the retrieval knobs are the ones worth exploring widely, and tying every
exploration to a generation call makes it expensive enough that you stop exploring.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anchor.retrieve import follow, search
from anchor.store import connect

HERE = Path(__file__).resolve().parent


def run(questions, corpus, strategy, k, hop, budget, decay, conn):
    counts = {}
    for item in questions:
        if not item["gold"]:
            continue
        hits = search(item["question"], corpus, strategy, k, conn=conn)
        if hop:
            hits = follow(hits, item["question"], corpus, strategy, k,
                          decay=decay, budget=budget, conn=conn)
        got = {h.unit_id for h in hits}
        hit = bool(got & set(item["gold"]))
        counts.setdefault(item["kind"], []).append(hit)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="aiact")
    parser.add_argument("--questions", default=str(HERE / "questions.json"))
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    conn = connect()

    settings = [
        ("structural", 5, False, 0, 0.75),
        ("structural", 5, True, 1, 0.75),
        ("structural", 5, True, 2, 0.75),
        ("structural", 5, True, 2, 0.90),
        ("structural", 5, True, 2, 0.60),
        ("structural", 8, False, 0, 0.75),
        ("structural", 8, True, 2, 0.75),
        ("structural", 3, False, 0, 0.75),
        ("structural", 3, True, 1, 0.75),
        ("fixed-1000-200", 5, False, 0, 0.75),
        ("fixed-1000-200", 5, True, 2, 0.75),
        ("fixed-1000-200", 8, False, 0, 0.75),
    ]

    rows = []
    print(f"{'chunking':16s}{'k':>3s}{'hop':>5s}{'budget':>8s}{'decay':>7s}"
          f"{'single':>9s}{'multihop':>10s}{'overall':>9s}")
    print("-" * 67)

    for strategy, k, hop, budget, decay in settings:
        counts = run(questions, args.corpus, strategy, k, hop, budget, decay, conn)
        single = counts.get("single", [])
        multi = counts.get("multihop", [])
        every = single + multi

        def pct(items):
            return f"{sum(items) / len(items):.2f}" if items else "-"

        print(f"{strategy:16s}{k:>3d}{str(hop):>5s}{budget:>8d}{decay:>7.2f}"
              f"{pct(single):>9s}{pct(multi):>10s}{pct(every):>9s}")

        rows.append({"strategy": strategy, "k": k, "hop": hop, "budget": budget,
                     "decay": decay,
                     "single": sum(single) / len(single) if single else None,
                     "multihop": sum(multi) / len(multi) if multi else None,
                     "overall": sum(every) / len(every) if every else None})

    conn.close()
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{args.corpus}-sweep.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
