# Anchor

Anchor is a question answering engine that stays inside its evidence. You point it at a set of
documents, ask it something, and it either answers with a citation for every claim it makes or it
tells you it doesn't have enough to go on. It is built to be corpus agnostic, so the same engine
runs over EU regulation and over Python PEPs without changing anything except a small adapter.

## Why I built it this way

The first version of any RAG project is easy. Embed some chunks, search by cosine similarity, paste
the top five into a prompt, print whatever comes back. That works often enough to demo and it fails
in ways that are hard to see, which is worse than failing loudly.

Three failures kept showing up when I tested a plain pipeline, and each one pushed a piece of this
design.

The first is that the model answers from memory instead of from the documents. Ask a question about
something it saw during pretraining and it will produce a fluent, confident, sometimes correct
answer that has nothing to do with what retrieval actually returned. You can't tell the difference
by reading the output. That is why every claim here gets checked against the retrieved text before
the answer is returned, rather than trusting the model's own citation markers.

The second is that similarity search can't follow a reference. Real documents point at each other
constantly. An article of the EU AI Act will say that something is high risk "except under the
conditions in Article 6(3)", and the text you actually need is sitting in a different document that
shares almost no vocabulary with the question. Cosine similarity will never surface it. So Anchor
builds a citation graph over the corpus during ingest and follows those edges after the initial
search.

The third is that a plain pipeline has no way to say no. If retrieval comes back with garbage, the
prompt still gets filled and the model still writes something. Anchor scores the retrieved evidence
before generating and refuses when it is too weak, and it tries rewriting the question once before
it gives up.

## How it works

Ingest pulls the corpus through an adapter, splits it into units that respect the document's own
structure, embeds them, and stores them in Postgres with pgvector. While doing that it also records
which unit cites which, which gives a citation graph for free.

A question is embedded and searched against that store. The top results are expanded by walking one
hop out along citation edges, then the whole set is reranked together so a followed reference has to
earn its place rather than being appended for free.

Before anything is generated, the evidence is scored. A cheap distance check handles the obvious
cases and only borderline ones get sent to a model for judgement, which keeps the cost down. If the
evidence fails, the engine rewrites the question and tries once more. If it fails again it refuses.

If the evidence holds, the answer is generated under a prompt that only allows it to use what it was
given. That answer is then broken into individual claims and each one is checked back against the
retrieved text. Claims that aren't supported get flagged rather than silently shipped.

## Evaluation

There is an eval set of questions written by hand over the corpus, split into ones answerable from a
single unit, ones that need a reference to be followed, and ones the corpus genuinely cannot answer.
That last group is the interesting one, because it is where a plain pipeline invents things and
where refusal has to work.

The harness measures whether the right source came back, whether the answer stayed inside the
evidence, and whether refusals happened when they should have. It runs the same questions across
different configurations so the effect of each design decision shows up as a number instead of an
opinion.

Numbers go here once the runs are done.

## Status

Work in progress. Building it in the order listed above.
