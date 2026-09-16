PY := uv run python

.PHONY: help all ingest threads conversations sample taxonomy diagnose agent test clean

help:
	@echo "make all            ingest -> threads -> conversations -> sample -> taxonomy"
	@echo "make ingest         parse data/raw/twcs.csv -> data/interim/tweets.parquet"
	@echo "make threads        reconstruct the reply graph -> data/interim/threads.parquet"
	@echo "make conversations  split threads into conversations -> data/interim/conversations.parquet"
	@echo "make sample         cut to SpotifyCares, merge split replies -> data/interim/spotify.parquet"
	@echo "make taxonomy       embed + cluster 2,000 openings -> notes/clusters_k{6,8,10}.md"
	@echo "make diagnose       top-1 similarity spread over the golden set (offline, no key)"
	@echo "make agent          run the agent over all 150 golden openings (LIVE, ~20 min)"
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

# --force-reply: the eval stage needs a draft for all 150, including the ones
# the rule escalates, or the judge's scores become conditional on the escalation
# rule being correct. Live run against Gemini; replays from cache afterwards.
agent:
	$(PY) -m src.agent --golden --force-reply --out data/interim/agent_golden.jsonl

test:
	uv run pytest -q

# data/interim/ is derived from data/raw/ and does NOT invalidate itself.
# Run this whenever twcs.csv changes or the parsing rules in src/ingest.py
# change, otherwise every downstream stage silently inherits the old answer.
clean:
	rm -rf data/interim
