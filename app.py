"""A UI that shows the mechanism rather than hiding it.

A chat box would make this look like every other wrapper around a model. The point of the
project is what happens between the question and the answer, so the retrieved chunks, which
of them arrived by following a citation, the gate's decision and the per-claim verdicts are
all on screen next to the answer. The toggles in the sidebar are there so the difference the
citation hop makes can be demonstrated live rather than described.
"""
import streamlit as st

from anchor.answer import ask
from anchor.corpora import ABOUT
from anchor.retrieve import retrieve
from anchor.store import connect

st.set_page_config(page_title="Anchor", page_icon="::", layout="wide")

EXAMPLES = {
    "aiact": [
        "Is an AI system used to evaluate job applicants considered high risk?",
        "Which AI practices does the Regulation prohibit outright?",
        "What is the maximum fine under the GDPR for unlawful profiling?",
    ],
    "peps": [
        "How many spaces should be used per indentation level?",
        "Where should the closing quotes of a multi-line docstring go?",
        "What is the scheduled release date of Python 3.15?",
    ],
}


@st.cache_resource
def db():
    return connect()


st.title("Anchor")
st.caption("Answers only what the documents support. Cites every claim. Refuses when the "
           "evidence is not there.")

with st.sidebar:
    st.header("Corpus")
    corpus = st.radio("", list(ABOUT), format_func=lambda c: ABOUT[c], label_visibility="collapsed")

    st.header("Pipeline")
    hop = st.checkbox("Follow citations", value=True,
                      help="Pull in documents that the top results reference. Turn this off to "
                           "see what plain vector search misses.")
    gate = st.checkbox("Refusal gate", value=True,
                       help="Score the evidence before generating, and refuse if it is too weak.")
    check = st.checkbox("Verify each claim", value=True,
                        help="Split the answer into claims and check each one against the sources.")
    k = st.slider("Chunks retrieved (k)", 3, 10, 8)

    st.divider()
    st.caption("Retrieval only needs the database. Generation and verification call an LLM.")

# ?q=...&show=retrieve makes a link that runs itself, which is handy for sharing a
# particular result rather than telling someone what to type.
params = st.query_params
question = st.text_input("Question", value=params.get("q", ""),
                         placeholder="Ask something about the corpus")
st.caption("Try: " + "  |  ".join(f"*{q}*" for q in EXAMPLES[corpus]))

col_a, col_b = st.columns([1, 1])
run_retrieval = col_a.button("Retrieve only", use_container_width=True,
                             help="No model call. Shows what the question pulls back.")
run_full = col_b.button("Answer", type="primary", use_container_width=True)

run_retrieval = run_retrieval or params.get("show") == "retrieve"
run_full = run_full or params.get("show") == "answer"


def show_sources(hits):
    st.subheader("Sources")
    for n, h in enumerate(hits, 1):
        label = f"**[{n}] {h.citation}**  ·  score {h.score:.2f}"
        if h.followed:
            label += f"  ·  :orange[followed from {h.via}]"
        st.markdown(label)
        st.caption(h.text[:320] + ("..." if len(h.text) > 320 else ""))


if run_retrieval and question:
    with st.spinner("Searching"):
        hits = retrieve(question, corpus, k=k, hop=hop, conn=db())
    followed = [h for h in hits if h.followed]
    st.info(f"{len(hits)} chunks. {len(followed)} arrived by following a citation."
            if followed else
            f"{len(hits)} chunks, none from a citation hop.")
    show_sources(hits)

if run_full and question:
    with st.spinner("Retrieving, checking the evidence, answering, verifying"):
        result = ask(question, corpus=corpus, k=k, hop=hop, gate=gate, check=check, conn=db())

    if result.refused:
        st.warning(f"**{result.text}**\n\n{result.reason}")
    else:
        st.success(result.text)

    if result.checks:
        bad = result.unsupported
        st.subheader(f"Claim check — {len(result.checks) - len(bad)} of {len(result.checks)} supported")
        for c in result.checks:
            if c.supported:
                st.markdown(f":green[supported] &nbsp; {c.claim}")
                if c.quote:
                    st.caption(f"“{c.quote}”")
            else:
                st.markdown(f":red[**not supported**] &nbsp; {c.claim}")

    left, right = st.columns([2, 1])
    with left:
        show_sources(result.hits)
    with right:
        st.subheader("What it did")
        for step in result.trace:
            st.markdown(f"- {step}")
        st.caption(f"{result.seconds:.1f}s")
