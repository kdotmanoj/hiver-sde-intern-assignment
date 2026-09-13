PY := uv run python

.PHONY: help all ingest threads conversations sample test clean

help:
	@echo "make all            ingest -> threads -> conversations -> sample"
	@echo "make ingest         parse data/raw/twcs.csv -> data/interim/tweets.parquet"
	@echo "make threads        reconstruct the reply graph -> data/interim/threads.parquet"
	@echo "make conversations  split threads into conversations -> data/interim/conversations.parquet"
	@echo "make sample         cut to SpotifyCares, merge split replies -> data/interim/spotify.parquet"
	@echo "make test           run the test suite"
	@echo "make clean          delete data/interim/ (derived data; regenerate with make ingest)"

all: ingest threads conversations sample

ingest:
	$(PY) -m src.ingest

threads:
	$(PY) -m src.threads

conversations:
	$(PY) -m src.conversations

sample:
	$(PY) -m src.sample

test:
	uv run pytest -q

# data/interim/ is derived from data/raw/ and does NOT invalidate itself.
# Run this whenever twcs.csv changes or the parsing rules in src/ingest.py
# change, otherwise every downstream stage silently inherits the old answer.
clean:
	rm -rf data/interim
