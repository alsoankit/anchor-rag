# Anchor

Anchor is a question answering engine that stays inside its evidence. You point it at a set
of documents, ask it something, and it either answers with a citation for every claim it
makes or it tells you it doesn't have enough to go on. It is corpus agnostic: the same
engine runs over EU regulation and over Python PEPs, and the only thing that changes between
them is a small adapter class.

The short version of what it does: **retrieval decides what the model is allowed to see, and
the verification layer decides what it is allowed to say.**

## Why I built it this way

The first version of any RAG project is easy. Embed some chunks, search by cosine
similarity, paste the top five into a prompt, print whatever comes back. That works often
enough to demo and it fails in ways that are hard to see, which is worse than failing
loudly.

Three failures kept showing up while I tested a plain pipeline, and each one pushed a piece
of this design.

**The model answers from memory instead of from the documents.** Ask about something it saw
during pretraining and it produces a fluent, confident, sometimes correct answer that has
nothing to do with what retrieval returned. You can't tell by reading the output. That is
why every claim gets checked against the retrieved text before the answer is returned,
rather than trusting the citation markers the model wrote itself.

**Similarity search can't follow a reference.** Real documents point at each other
constantly. An article of the EU AI Act says a system is high risk if it is listed in Annex
III, and the text you actually need sits in a different document that shares almost no
vocabulary with the question. Cosine similarity will never surface it. So Anchor builds a
citation graph over the corpus during ingest and follows those edges after the first search.

**A plain pipeline has no way to say no.** If retrieval comes back with garbage, the prompt
still gets filled and the model still writes something. Anchor scores the evidence before
generating, refuses when it is too weak, and rewrites the question once before giving up.

## How it works

Ingest pulls a corpus through an adapter, splits it into units that respect the document's
own structure, embeds them, and stores them in Postgres with pgvector. While parsing it also
records which unit cites which, which gives the citation graph for free.

A question is embedded and searched against that store. The top results are expanded one hop
along citation edges. Followed text does not compete on similarity — it can't, since it
inherits a faded copy of its parent's score and so can never outrank it — so a couple of
context slots are reserved for it instead. Total context size stays the same either way,
which is what makes the comparison below mean something.

Before anything is generated the evidence is scored. A cheap distance check handles the
obvious cases and only borderline ones cost a model call. If the evidence fails, the engine
rewrites the question and tries once more; if it fails again it refuses.

If the evidence holds, the answer is generated under a prompt that only permits what it was
given. That answer is then split into individual claims and each one is checked back against
the sources on its own. Claims that aren't supported get flagged rather than shipped.

There is a second mode. Describe a system in plain English and `assess` breaks the
description into several searches, retrieves for each, and returns the obligations that
apply to it with a citation on each one.

## What the numbers say

Two corpora. The EU AI Act: 739 chunks, 126 units, 402 citation edges, with 30 hand-written
questions whose gold labels were each verified against the actual text. Python PEPs: 635
chunks, 60 units, 235 edges, 11 questions. Questions come in three kinds: answerable from one
unit, needing a reference followed, and genuinely unanswerable.

### Retrieval

This sweep costs nothing to run, because it is embeddings and SQL with no model calls at
all. Splitting it out from the expensive evaluation is what let it explore widely.

| chunking | k | hop | single | multihop | overall |
|---|---|---|---|---|---|
| structural | 5 | no | 1.00 | 0.70 | 0.86 |
| structural | 5 | yes | 0.92 | 1.00 | 0.95 |
| **structural** | **8** | **yes** | **1.00** | **1.00** | **1.00** |
| structural | 3 | no | 0.92 | 0.60 | 0.77 |
| structural | 3 | yes | 0.92 | 1.00 | 0.95 |
| fixed-size | 5 | no | 0.92 | 0.80 | 0.86 |

Following citations takes multi-hop retrieval from **0.70 to 1.00** on the AI Act and from
**0.00 to 1.00** on the PEPs. The three AI Act questions it rescues are all the same shape:
plain retrieval found the article that *points at* the answer and stopped there. Asked which
criminal offences permit real-time biometric identification, it returned Article 5 three
times over; the list of offences is in Annex II.

It is not free. At k=5 a reserved slot pushes a direct hit out and single-hop drops to 0.92.
At k=8 there is room for both and everything reaches 1.00, which is why k=8 is the default —
that came from the table, not from a guess.

The line I find most useful: **k=3 with the hop beats k=8 without it**, 0.95 against 0.86, on
under half the context. Following one reference is worth more than five extra chunks of
similar text.

### Generation

Every row below uses the same writer model, the same metric and k=5, so they are comparable
to each other.

| config | hit single | hit multihop | good refusals | bad refusals | faithful | tokens/question |
|---|---|---|---|---|---|---|
| naive | 1.00 | 0.70 | 0.625 | 0.045 | 0.975 | 2,500 |
| plain | 1.00 | 0.70 | 1.00 | 0.136 | 1.00 | 1,914 |
| fixed-size | 0.92 | 0.80 | 1.00 | 0.045 | 0.943 | 2,737 |
| hop | 0.92 | 1.00 | 1.00 | 0.091 | 0.944 | 2,150 |
| hop+gate | 0.92 | 1.00 | 1.00 | 0.091 | 0.944 | 2,208 |
| full | 0.92 | 1.00 | 1.00 | 0.091 | 0.958 | 6,823 |

`naive` has no instruction to stay inside the sources and no way to refuse. It is what a
pipeline looks like when retrieval is treated as the whole job.

**The grounding instruction takes correct refusals from 0.625 to 1.00.** The naive prompt
answers three of the eight unanswerable questions anyway. The cost, in the same breath: false
refusals rise from 0.045 to 0.136, so the instruction makes the system more cautious than it
needs to be on two answerable questions. That is a trade, and it belongs in the table rather
than buried.

Worth noticing that the hop *reduces* false refusals, 0.136 down to 0.091. Better evidence
means fewer questions the model can't find enough to answer.

### The result I did not expect

The `full` config decomposed 20 answers into 67 individual claims and found **12 of them
unsupported by the retrieved text — 18%.** On the PEPs it was 5 of 22, **23%**. Two unrelated
corpora landing in the same range.

That is with the grounding prompt and the refusal gate both active. And the answer-level
faithfulness judge scores those same answers at 0.958.

Those two measurements disagree, and that disagreement is the point. A judge reading a whole
answer sees something mostly grounded and scores it as grounded. Decomposing it catches the
one sentence in five that quietly drifted past what the sources actually say. **Grounding
prompts and refusal gates are not sufficient on their own**, which is a narrower and more
useful claim than "this system prevents hallucination".

Verification costs what you would expect: 6,823 tokens per question against 2,208 without it,
roughly 3.1x, because every claim is its own call. That is the right trade for a compliance
question and the wrong one for a chatbot.

### How much to trust the faithfulness numbers

There are no human labels anywhere in this evaluation. Faithfulness is one model's opinion of
another model's output. What `eval/calibrate.py` measures is whether three judges from
different families agree on the same statements: the pipeline's own checker, a second
open-weight model, and a Gemini model from a different provider entirely. That third one
matters most, because the first two are served through the same API and their agreeing could
be an artefact of shared lineage rather than the statements being clear-cut.

| comparison | claims | agreement |
|---|---|---|
| pipeline checker vs second judge (both Groq, different families) | 58 | 78% |
| pipeline checker vs third judge (different provider) | 20 | 80% |
| second judge vs third judge | 20 | 90% |
| all three agreed | 20 | 70% |

The third judge grades a sample rather than everything, because its free tier allows twenty
requests a day. Two numbers, two sample sizes, reported separately rather than blended into
one figure that implies more coverage than exists.

Around 78 to 80 percent is a reasonable place to land. It says the faithfulness signal is
stable enough that a difference between two configurations means something, and not so
stable that any single number should be quoted as fact. Agreement is not accuracy: three
models agreeing says they read the evidence the same way, not that they read it correctly.

Where they disagree is itself readable. Most of the splits are the pipeline's checker being
stricter than the others on claims that paraphrase a provision rather than restate it — it
wants the words to be there, the others accept the sense. For a compliance system that is
arguably the right bias, but it is a bias, and it is why the unsupported-claim rate above
should be read as an upper bound.

## Running it

```
docker compose up -d
pip install -r requirements.txt
cp .env.example .env          # add a Groq key; several are supported
python -m anchor.cli ingest aiact
python -m anchor.cli ask "Is an AI system used to screen job applicants high risk?"
python -m eval.sweep --corpus aiact          # free, no model calls
python -m eval.run_eval --corpus aiact
```

`anchor.cli retrieve` shows what came back and what was followed, without generating.
`anchor.cli assess` takes a system description. `anchor.api` exposes both over FastAPI.

## What it doesn't do

- **One hop only.** If the answer is two references away it isn't found. Multi-hop traversal
  wasn't worth the retrieval noise at this corpus size.
- **No reranker.** A cross-encoder over the retrieved set is the obvious next improvement and
  would probably beat the hop on single-hop questions.
- **Recitals of the AI Act are excluded.** They explain the reasoning behind the law but are
  not the law, and they roughly triple the corpus with text that sounds authoritative without
  being binding.
- **The eval set is 30 questions written by one person.** Enough to compare configurations
  against each other, not enough to say anything about absolute quality. At k=8 with the hop
  it is saturated at 1.00 and can no longer distinguish improvements; the next version needs
  harder questions, not more mechanisms.
- **`all-MiniLM-L6-v2` is small and old.** Chosen because it is free, local, and fast enough
  to re-embed the whole corpus in under a minute while iterating.
- **The third calibration judge samples 20 of 67 claims**, because its free tier allows
  twenty requests per day. The two-judge comparison covers everything; the cross-provider
  check does not.

## Notes on things that broke

A silent regex bug cost a fifth of the citation graph: `\bAnnexes?` doesn't mean "Annex" with
an optional "s" — the `?` binds to the preceding character, so it only ever matched "Annexe"
and "Annexes", never the singular that appears everywhere.

The citation hop did nothing for its first implementation. Followed chunks inherit a decayed
copy of their parent's score, so sorting the merged set by score threw away everything the
hop had found. Output was byte-identical with the feature on and off.

The classic RAG failure showed up in miniature: asked whether CV screening is high risk,
retrieval returned Annex III's *header* at 0.588 and the employment provision that answers it
at 0.531. The header repeats the question's vocabulary; the answer doesn't. Folding list
lead-ins into their items and putting unit headings into the embedding moved it to 0.74 and
rank one.

The calibration had a bug that produced a confident, plausible, completely wrong result. It
reconstructed "the sources" a judge should grade against by taking one chunk per retrieved
unit, with no ordering. Article 99 has nineteen paragraphs; the answer had been written from
the one containing the fine, and the reconstruction handed over the first. Every judge but
the pipeline's own was grading against evidence that did not contain the answer. They said
unsupported, agreed with each other at 94%, and agreement between two measurements looked
like confirmation when it was a shared input. Agreement across configurations that were
reading the same wrong thing is not evidence of anything. The eval now records exact chunk
ids and rebuilds from those, and the agreement numbers went from 28% to 78%.

And the evaluation itself had a bug worth more than any of them. Refusal was detected by
checking the answer for the string `NOT IN SOURCES` — the exact phrase the grounding prompt
tells the model to emit. That worked until the naive baseline was added, which has no such
instruction and declined in its own words instead. The metric was measuring a prompt's
vocabulary rather than the behaviour it was named after, and the two only lined up by
accident. It agreed with my expectations for four configurations before the fifth exposed it.
Every number measured before the fix was thrown away and re-run.
