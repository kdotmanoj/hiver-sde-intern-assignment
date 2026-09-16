PY := uv run python

.PHONY: help all ingest threads conversations sample taxonomy diagnose baselines agent judge judge-consistency eval test clean

help:
	@echo "make all            ingest -> threads -> conversations -> sample -> taxonomy"
	@echo "make ingest         parse data/raw/twcs.csv -> data/interim/tweets.parquet"
	@echo "make threads        reconstruct the reply graph -> data/interim/threads.parquet"
	@echo "make conversations  split threads into conversations -> data/interim/conversations.parquet"
	@echo "make sample         cut to SpotifyCares, merge split replies -> data/interim/spotify.parquet"
	@echo "make taxonomy       embed + cluster 2,000 openings -> notes/clusters_k{6,8,10}.md"
	@echo "make diagnose       top-1 similarity spread over the golden set (offline, no key)"
	@echo "make baselines      trivial + k-NN baselines over the golden set (offline, no key)"
	@echo "make agent          run the agent over all 150 golden openings (LIVE, ~20 min)"
	@echo "make judge          LLM-as-judge over the agent's 150 replies (LIVE, ~10 min)"
	@echo "make judge-consistency  re-judge 20 replies twice; does the judge agree with itself?"
	@echo "make eval           score agent + both baselines on the golden set (offline, no key)"
	@echo "make test           run the test suite"
	@echo "make clean          delete data/interim/ (derived data; regenerate with make ingest)"

all: ingest threads conversations sample taxonomy

ingest:
	$(PY) -m src.ingest

threads:
	$(PY) -m src.threads

conversations:
	$(PY) -m src.conversations

sample:
	$(PY) -m src.sample

taxonomy:
	$(PY) -m src.taxonomy

# Evidence for escalate.SIMILARITY_FLOOR. Embeddings only, no LLM calls, so
# CACHE_ONLY=1 proves the retrieval corpus is fully cached at the same time.
diagnose:
	CACHE_ONLY=1 $(PY) -m src.agent --diagnose

# The floor the agent has to clear. No LLM calls in either baseline, so
# CACHE_ONLY=1 is the guard, not a convenience: if this ever needs a key, a
# baseline has started calling a model and stopped being a baseline.
baselines:
	CACHE_ONLY=1 $(PY) -m src.baselines --out-dir data/interim

# --force-reply: the eval stage needs a draft for all 150, including the ones
# the rule escalates, or the judge's scores become conditional on the escalation
# rule being correct. Live run against Gemini; replays from cache afterwards.
agent:
	$(PY) -m src.agent --golden --force-reply --out data/interim/agent_golden.jsonl

# Groq scores the replies Gemini wrote -- a different model family, so the judge
# is not grading its own output. Live run, ~150 calls at 4s apart, ~97K estimated
# tokens against Groq's 200K/day. One system per run: all three would be ~290K and
# the preflight refuses it. Replays from cache afterwards.
judge:
	$(PY) -m src.judge --system agent --out data/interim/judge_scores.jsonl

# Does the judge give identical input the same score twice? TEMPERATURE is 0, but
# temperature 0 is not a determinism guarantee on a hosted MoE model, and a mean
# over 150 scores is only worth reporting if the answer here is yes. 40 calls,
# ~26K tokens. Run it AFTER `make judge`; it refreshes only its own 40 rows.
judge-consistency:
	$(PY) -m src.judge --system agent --limit 20 --repeat 2 --out data/interim/judge_scores.jsonl

# Scores jsonl already on disk -- no model, no embeddings, no key. CACHE_ONLY=1
# is the guard, not a convenience: if this ever needs a key, the scoring stage
# has started calling a model and stopped being reproducible.
eval:
	CACHE_ONLY=1 $(PY) -m src.evaluate

test:
	uv run pytest -q

# data/interim/ is derived from data/raw/ and does NOT invalidate itself.
# Run this whenever twcs.csv changes or the parsing rules in src/ingest.py
# change, otherwise every downstream stage silently inherits the old answer.
clean:
	rm -rf data/interim
