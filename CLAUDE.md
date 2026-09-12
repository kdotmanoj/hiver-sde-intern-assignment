# Hiver take-home: Twitter support agent

## Non-negotiable constraints
- Python 3.11. Deps via `uv`. Everything runnable through the Makefile.
- I am a full-stack dev, not an ML engineer. Write code I can explain
  line-by-line in a live interview. 
- Implement metrics (precision, recall, macro-F1, Cohen's kappa,
  bootstrap CI) by hand in src/metrics.py. Do NOT import
  sklearn.metrics. Each gets a unit test against a hand-computed value.
- Every LLM call goes through src/llm.py, which caches responses to
  data/cache/ keyed by hash(model + prompt). Cache is committed to git.
- `make eval` must replay from cache with no API key and finish in <60s.
- Deterministic: seed=42 everywhere, no unseeded shuffles.
- Embeddings are cached too, same mechanism as LLM calls. Never re-embed
  the same text twice.
- The LLM-as-judge must use a different model family than the reply
  generator, to avoid self-preference bias. If that becomes impossible,
  stop and tell me — it changes what I can claim in the report.
- Every pipeline stage prints a one-line row-count summary
  (e.g. "threads 41,203 -> brand 8,914 -> with reply 7,102 (lost 1,812)").
  I read these every run.

## Code style
- Idiomatic pandas is fine and preferred for data manipulation. Do not
  write explicit loops to avoid vectorized operations.
- Cap chains at ~3 operations. Beyond that, materialize a named
  intermediate so it can be inspected.
- After every merge, groupby, filter, or explode: assert the expected
  row count or log before/after shapes. Merges must specify
  validate= ('one_to_one', 'one_to_many', etc). Groupbys that should
  not drop nulls must pass dropna=False explicitly.
- Exception — write these as plain, explicit code with no library
  shortcuts: src/metrics.py, src/escalate.py, src/threads.py.
  These encode decisions or math I must be able to derive from scratch.

## Never do these without asking me
- Do not write or modify anything in data/golden/. Those labels are mine.
- Do not name or define the intent taxonomy. I do that after reading clusters.
- Do not write report.md or decisions.md prose. You may format; I write the claims.

## Workflow
- Use plan mode for anything touching more than one file. Show me the plan first.
- After each change, tell me in 3 sentences what you did and what could break.
- Commit granularly with meaningful messages. Never batch unrelated
  changes. My git log is evidence of process and will be reviewed.
- When I make a non-obvious choice (threshold, sampling rule, brand,
  k value, prompt design), remind me to add it to decisions.md before
  we move on. Do not write the entry for me.
- data/raw/ is gitignored. data/cache/, data/sample/, data/golden/ are
  committed.

  ## Model stack (fixed — do not substitute without asking)
- Embeddings: local sentence-transformers all-MiniLM-L6-v2. Never an API.
  Cache vectors to data/cache/embeddings/ as .npy.
- Generator + classifier: Gemini Flash-tier via AI Studio free key.
  ~15 RPM. Min 4s gap between live calls, exponential backoff on 429.
- Judge: Groq openai/gpt-oss-120b. 30 RPM, 8K TPM, 200K TPD.
  TPD is our tightest constraint — keep judge prompts under ~500 tokens
  and warn me if a planned change would push a full judge run over 120K.
- Never enable billing on the Gemini project; it deletes the free tier.
- Environment: pandas 3.0.5 (copy-on-write default), numpy 2.4, Python 3.11.
  Do not write pandas 2.x-era workarounds. If unsure whether an API
  changed in pandas 3, check rather than assume.