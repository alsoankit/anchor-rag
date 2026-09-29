import argparse
import textwrap

from anchor import store
from anchor.answer import ask
from anchor.assess import assess
from anchor.llm import usage
from anchor.retrieve import retrieve

WIDTH = 88


def wrap(text: str, indent: str = "") -> str:
    return textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent)


def show_sources(hits):
    print("\nsources")
    for n, hit in enumerate(hits, 1):
        tag = f" (followed from {hit.via})" if hit.followed else ""
        print(f"  [{n}] {hit.citation}  score {hit.score:.2f}{tag}")
        print(wrap(hit.text[:200] + ("..." if len(hit.text) > 200 else ""), "      "))


def cmd_ask(args):
    result = ask(args.question, corpus=args.corpus, k=args.k,
                 hop=not args.no_hop, gate=not args.no_gate, check=not args.no_check)

    print()
    print(wrap(result.text))

    if result.refused:
        print(f"\nreason: {result.reason}")
    elif result.checks:
        bad = result.unsupported
        print(f"\nchecked {len(result.checks)} statements, {len(bad)} not supported")
        for item in bad:
            print(wrap(f"unsupported: {item.claim}", "  "))

    show_sources(result.hits)

    print("\nwhat it did")
    for step in result.trace:
        print(wrap(step, "  "))
    print(f"\n{result.seconds:.1f}s, {usage['calls']} model calls")


def cmd_assess(args):
    report = assess(args.description, corpus=args.corpus)

    print()
    print(wrap(report.summary))
    print("\nsearches it ran")
    for search in report.searches:
        print(wrap(search, "  "))

    print(f"\nobligations ({len(report.obligations)})")
    for item in report.obligations:
        print(f"\n  {item['obligation']}  [{item['citation']}]")
        print(wrap(item["requires"], "    "))

    print(f"\n{usage['calls']} model calls")


def cmd_retrieve(args):
    hits = retrieve(args.question, args.corpus, k=args.k, hop=not args.no_hop)
    show_sources(hits)


def cmd_ingest(args):
    store.setup()
    store.ingest(args.corpus, args.strategy, limit=args.limit)


def main():
    parser = argparse.ArgumentParser(prog="anchor")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest")
    p.add_argument("corpus")
    p.add_argument("--strategy", default="structural")
    p.add_argument("--limit", type=int)
    p.set_defaults(run=cmd_ingest)

    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--corpus", default="aiact")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--no-hop", action="store_true")
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--no-check", action="store_true")
    p.set_defaults(run=cmd_ask)

    p = sub.add_parser("retrieve")
    p.add_argument("question")
    p.add_argument("--corpus", default="aiact")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--no-hop", action="store_true")
    p.set_defaults(run=cmd_retrieve)

    p = sub.add_parser("assess")
    p.add_argument("description")
    p.add_argument("--corpus", default="aiact")
    p.set_defaults(run=cmd_assess)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
