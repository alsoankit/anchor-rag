import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anchor.answer import ask
from anchor.llm import usage
from anchor.store import connect
from eval.metrics import declined, faithfulness, gold_recall, retrieval_hit

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

CONFIGS = {
    "naive": dict(strategy="structural", hop=False, gate=False, check=False,
                  grounded_prompt=False),
    "plain": dict(strategy="structural", hop=False, gate=False, check=False),
    "fixed-size": dict(strategy="fixed-1000-200", hop=False, gate=False, check=False),
    "hop": dict(strategy="structural", hop=True, gate=False, check=False),
    "hop+gate": dict(strategy="structural", hop=True, gate=True, check=False),
    "full": dict(strategy="structural", hop=True, gate=True, check=True),
}


def run_config(name: str, questions: list[dict], corpus: str, k: int) -> dict:
    settings = CONFIGS[name]
    before = dict(usage)
    started = time.time()
    rows = []

    conn = connect()
    try:
        for item in questions:
            result = ask(item["question"], corpus=corpus, k=k, conn=conn, **settings)
            answerable = item["kind"] != "unanswerable"

            # The pipeline knows when it refused on purpose. It does not know when the
            # writer declined in its own words, which is the only way the naive baseline
            # can decline at all, so the reply gets read rather than pattern matched.
            backed_out = result.refused or declined(item["question"], result.text)

            row = {
                "id": item["id"],
                "kind": item["kind"],
                "question": item["question"],
                "refused": backed_out,
                "refused_by_pipeline": result.refused,
                "answer": result.text,
                "retrieved": [h.unit_id for h in result.hits],
                "followed": [h.citation for h in result.hits if h.followed],
                "hit": retrieval_hit(result.hits, item.get("gold", [])),
                "recall": gold_recall(result.hits, item.get("gold", [])),
                "seconds": round(result.seconds, 2),
            }

            if answerable and not backed_out:
                row["faithfulness"] = round(faithfulness(result.text, result.hits), 3)
            if result.checks:
                row["unsupported_claims"] = len(result.unsupported)
                row["total_claims"] = len(result.checks)
                row["claim_detail"] = [[c.claim, c.supported] for c in result.checks]

            rows.append(row)
            print(f"  {item['id']:<6} {item['kind']:<13} "
                  f"{'refused' if result.refused else 'answered':<9} "
                  f"hit={'y' if row['hit'] else 'n'}", flush=True)
    finally:
        conn.close()

    return {
        "config": name,
        "settings": settings,
        "rows": rows,
        "seconds": round(time.time() - started, 1),
        "model_calls": usage["calls"] - before["calls"],
        "tokens": (usage["prompt_tokens"] + usage["completion_tokens"]
                   - before["prompt_tokens"] - before["completion_tokens"]),
    }


def summarise(run: dict) -> dict:
    rows = run["rows"]
    answerable = [r for r in rows if r["kind"] != "unanswerable"]
    multihop = [r for r in rows if r["kind"] == "multihop"]
    single = [r for r in rows if r["kind"] == "single"]
    impossible = [r for r in rows if r["kind"] == "unanswerable"]
    scored = [r for r in answerable if "faithfulness" in r]

    def rate(items, key="hit"):
        return round(sum(1 for r in items if r[key]) / len(items), 3) if items else None

    return {
        "config": run["config"],
        "hit_single": rate(single),
        "hit_multihop": rate(multihop),
        "refused_when_it_should": rate(impossible, "refused"),
        "refused_when_it_should_not": rate(answerable, "refused"),
        "faithfulness": round(sum(r["faithfulness"] for r in scored) / len(scored), 3) if scored else None,
        "seconds_per_question": round(run["seconds"] / max(len(rows), 1), 1),
        "model_calls": run["model_calls"],
    }


def table(summaries: list[dict]) -> str:
    columns = [
        ("config", "config", 12),
        ("hit_single", "hit single", 11),
        ("hit_multihop", "hit multihop", 13),
        ("refused_when_it_should", "good refusals", 14),
        ("refused_when_it_should_not", "bad refusals", 13),
        ("faithfulness", "faithful", 9),
        ("seconds_per_question", "sec/q", 6),
        ("model_calls", "calls", 6),
    ]
    lines = ["".join(head.ljust(width) for _, head, width in columns)]
    lines.append("-" * sum(w for _, _, w in columns))
    for summary in summaries:
        cells = []
        for key, _, width in columns:
            value = summary.get(key)
            cells.append(("-" if value is None else str(value)).ljust(width))
        lines.append("".join(cells))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="aiact")
    parser.add_argument("--questions", default=str(HERE / "questions.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--tag", default="",
                        help="suffix for the result files, for when the same configs are "
                             "re-measured on a different writer model")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    RESULTS.mkdir(exist_ok=True)

    summaries = []
    for name in args.configs:
        print(f"\n{name}")
        run = run_config(name, questions, args.corpus, args.k)
        (RESULTS / f"{args.corpus}-{name}{args.tag}.json").write_text(json.dumps(run, indent=2))
        summaries.append(summarise(run))

    print("\n" + table(summaries))
    (RESULTS / f"{args.corpus}-summary{args.tag}.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
