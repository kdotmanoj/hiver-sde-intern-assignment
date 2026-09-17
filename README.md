# Twitter support agent for SpotifyCares

A retrieval-grounded support agent over the Kaggle Customer Support on Twitter
corpus. It takes one customer tweet, classifies it into one of nine intents,
retrieves the k=3 most similar past SpotifyCares conversations, drafts a reply
grounded in those past replies, and decides whether the case needs a human
instead. It is evaluated against two baselines on 150 hand-labelled examples,
with metrics implemented by hand and an LLM-as-judge from a different model
family than the generator. Read [REPORT.md](REPORT.md) for results and their
limitations, and [decisions.md](decisions.md) for the decision log.

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11
```

```bash
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

The extra index is required, not optional: `requirements.txt` pins
`torch==2.14.0+cpu`, which is published on PyTorch's CPU index and not on PyPI.
A plain `uv pip install -r requirements.txt` fails to resolve. Expect about a
minute with a warm uv cache, a few minutes cold, most of it the torch download.

There is no `pyproject.toml`, so `uv run` does not install anything implicitly.
Create the venv first or every target fails on a missing import.

**The Kaggle CSV is not needed to reproduce the results.** `twcs.csv` goes in
`data/raw/` (gitignored, 493MB) and is only required if you want to re-run the
pipeline from raw tweets. The committed LLM cache, embedding cache and
evaluation inputs cover everything below.

## Headline command

```bash
make eval
```

Runs in about 3 seconds, offline, with no API key. It sets `CACHE_ONLY=1` and
scores files already on disk, so it never calls a model. It prints:

- intent classification: accuracy, macro-F1, a 95% paired bootstrap CI and
  Cohen's kappa, for the agent and both baselines
- per-class F1 across all nine intents, naming any class a system never
  predicted and reporting its 0/0 precision as 0.000 rather than 1.000
- the escalation decision as a binary problem, with precision, recall and F1
- escalation broken out by signal family, which is where the negative result
  lives: the intent family alone beats the combined rule
- the agent's 9x9 confusion matrix

The numbers it prints are the ones quoted in REPORT.md. Verified from a clean
checkout: agent macro-F1 0.716 [0.632, 0.780], k-NN 0.376, trivial 0.039.

## Make targets

Work on a fresh clone, offline, no API key. These are the ones a grader needs:

| target | what it does |
|---|---|
| `make eval` | score agent and both baselines on the golden set, ~3s |
| `make test` | 333 tests, ~8s |
| `make judge` | re-score the 150 replies, replaying every call from the committed cache, ~2s |

The judge-versus-human agreement in REPORT.md comes from one more offline
command, which reads the 60 hand-scored replies and joins them to the judge's
scores:

```bash
uv run python -m src.judge --judge-agreement
```

Everything below needs `data/interim/spotify.parquet`, which is 11MB of derived
data that stays gitignored. Run `make all` first, which needs `data/raw/twcs.csv`
from Kaggle. No API key for any of them.

| target | what it does |
|---|---|
| `make ingest` | parse 2.8M tweets into `data/interim/tweets.parquet` |
| `make threads` | reconstruct the reply graph into 798,197 threads |
| `make conversations` | split broadcast threads into individual conversations |
| `make sample` | cut to SpotifyCares, merge split replies, write `spotify.parquet` |
| `make taxonomy` | embed and cluster 2,000 openings into `notes/clusters_k{6,8,10}.md` |
| `make all` | the five above, in order |
| `make baselines` | regenerate the trivial and k-NN baseline predictions |
| `make diagnose` | top-1 similarity spread over the golden set, the evidence behind `SIMILARITY_FLOOR` |
| `make clean` | delete `data/interim/`, forcing a rebuild |

Live model calls. Each replays from the committed cache afterwards, so a second
run needs no key:

| target | what it does |
|---|---|
| `make agent` | run the agent over all 150 golden openings, ~20 min live. Also needs `spotify.parquet` for retrieval |
| `make judge-consistency` | re-judge 20 replies twice to test whether the judge agrees with itself |

## What needs an API key

Nothing you need to reproduce the reported results.

Every model call goes through `src/llm.py`, which caches responses to
`data/cache/` keyed by a hash of model, system prompt and prompt. Embeddings
cache the same way to `data/cache/embeddings/` as `.npy`. Both caches are
committed: 493 LLM responses and 2,002 embedding vectors. `CACHE_ONLY=1` turns
a cache miss into a loud error instead of a network call, which is why the
offline targets set it rather than merely tolerating a missing key.

`make eval`, `make test`, `make baselines` and `make diagnose` never call a
model. `make agent` and `make judge` do, but only on a cache miss, so re-running
them against the committed cache reproduces the same output with no key set.
Verified: `make judge` with no key set prints the reported means of 4.80, 4.97
and 4.91.

You only need keys to run the agent or judge on inputs that are not already
cached, which means changing a prompt, a model id or the golden set. Generation
and classification use Gemini via an AI Studio key; the judge uses Groq. Both
sit on free tiers with daily caps, so both stages count their uncached calls in
a preflight and refuse to start a run that would not fit.

## AI assistance

Per the assignment's citation rule.

Written with Claude Code: most of the implementation in `src/`, the test suite,
the Makefile, and the formatting of the two markdown deliverables. The working
style was constrained by `CLAUDE.md`, which required plain explicit code in
`src/metrics.py`, `src/escalate.py` and `src/threads.py`, hand-implemented
metrics with no `sklearn.metrics`, and a row-count summary from every stage.

Mine, not delegated: all 150 golden labels and the labelling guide, the second
blind re-labelling pass, the nine-class intent taxonomy (defined by reading
cluster exemplars, since the clustering did not separate intents on its own),
the 60 human reply scores behind the judge-agreement numbers, the hypotheses in
the failure analysis and the hand classification of all 40 intent
disagreements, every claim in REPORT.md, and every entry in decisions.md.
