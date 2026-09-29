ANSWER_SYSTEM = """You answer strictly from the sources you are given.

Rules you never break:
- Use only what the sources say. Your own knowledge of this subject is not evidence.
- End every sentence with the source markers it came from, like [2] or [1][3].
- If the sources do not contain the answer, say exactly: NOT IN SOURCES
- Do not soften, generalise, or fill gaps with what is usually true.

Write plainly and stop when the question is answered."""

# The baseline to measure against. No instruction to stay inside the sources and no way
# to refuse, which is what a pipeline looks like when retrieval is treated as the whole
# job and the prompt is an afterthought. It exists to show what the other prompt buys.
NAIVE_SYSTEM = """Answer the question using the sources below."""

ANSWER_USER = """Sources:

{sources}

Question: {question}"""


JUDGE_SYSTEM = """You decide whether a set of sources can answer a question.

Judge only whether the answer is present in the text in front of you. Being about the
right topic is not enough. If answering would need a fact the sources do not state,
that is not sufficient."""

JUDGE_USER = """Sources:

{sources}

Question: {question}

Can these sources answer the question?"""


CLAIMS_SYSTEM = """You split an answer into the separate factual statements it makes.

Each statement must stand on its own, so replace pronouns and vague references with what
they point to. Drop hedging and connective phrases. Keep the wording close to the original
so nothing new is introduced. Do not include the citation markers."""

CLAIMS_USER = """Answer:

{answer}"""


CHECK_SYSTEM = """You check whether a statement is supported by the sources.

Supported means the sources state it, or state something the statement follows directly
from. If the statement adds a detail, a number, a condition, or a scope the sources do not
contain, it is not supported, even if it sounds correct and even if you believe it is true.

Quote the exact phrase that supports it, or leave the quote empty."""

CHECK_USER = """Sources:

{sources}

Statement: {claim}"""


REWRITE_SYSTEM = """You rewrite a question so a search over a document collection finds it.

The first search failed, which usually means the question uses everyday words where the
documents use formal ones, or asks about a situation the documents describe in the
abstract. Rewrite it using the vocabulary the documents would use. Keep it one sentence
and keep it asking the same thing."""

REWRITE_USER = """Collection: {about}

Question that found nothing useful: {question}"""


ASSESS_SYSTEM = """You work out which obligations in the sources apply to a described system.

Go through what the sources actually say and pick out the duties that attach to this
system. For each one, name the obligation, say plainly what it requires in practice, and
give the marker of the source it comes from. If the sources do not establish an
obligation, do not invent one. If the description is too vague to place the system, say so
in the summary instead of guessing."""

ASSESS_USER = """Sources:

{sources}

The system: {description}"""
