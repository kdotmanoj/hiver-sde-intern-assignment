# Decision log

Non-obvious choices made while building this, with the number or observation that
drove each one. Kept as I went rather than reconstructed at the end.

---

## Model stack

**1. Embeddings run locally, not through an API**

Chose `sentence-transformers` (all-MiniLM-L6-v2) on CPU. Cloudflare Workers AI would
have worked, but its free tier is 10,000 neurons per day and embedding ~20k tweets at
6,058 neurons per million input tokens would consume roughly half of that daily budget
for something a CPU does for free. Local also means byte-identical vectors on every
run, which matters because the whole pipeline is supposed to be deterministic.

Rejected: any embeddings API.

**2. Gemini Flash for generation and classification**

A full evaluation run is about 400 calls (200 classify + 200 reply). Gemini's AI Studio
free tier covers that comfortably at ~15 RPM with a generous token-per-minute ceiling.
Cloudflare's 10,000 neurons per day works out to roughly one full run per day, which
leaves no room to iterate. 

Rejected: Cloudflare as the primary generator, kept as a
fallback only.

**3. Groq `openai/gpt-oss-120b` as the judge**

Deliberately a different model family from the generator. Models rate text from their
own family higher than equivalent text from another family, so a Gemini judge scoring
Gemini output would inflate quality scores in a direction I cannot measure or correct
for. Groq's free tier caps at 200K tokens per day, which is the tightest constraint in
the project and is why judge prompts are kept under about 500 tokens. 

Rejected: using Gemini to judge its own output.

**4. Pinned `gemini-3.6-flash`, not an alias**

`gemini-2.5-flash` appears in the models-list endpoint but returns 404 "no longer
available to new users" on a key created this month, so the model list is not evidence
that a model is callable. 

Rejected `gemini-flash-latest` for a different reason: the
response cache is committed and keyed by model id, so an alias that silently re-points
to a newer model would leave the cache looking valid while the outputs inside it came
from a model I never tested. That breaks reproducibility invisibly, which is worse than
breaking it loudly.

---

## LLM gateway

**5. Every call goes through one cached function**

`complete(prompt, model, system=None)` in `src/llm.py` is the only thing in the project
that talks to an API. Responses are cached to disk keyed by sha256 of model + system +
prompt, and the cache is committed. This does two jobs at once: it keeps me inside the
free tier while iterating, and it lets the grader reproduce headline results with no API
key in under 60 seconds. 

Rejected: calling providers directly from each stage.

**6. Four second minimum gap between live Gemini calls**

Free tier is around 15 requests per minute. Pacing at the gateway means no stage has to
think about rate limits. Cache hits skip the sleep entirely, so replay stays instant.

Rejected: unthrottled calls plus retry-on-429, which wastes quota and time.

**7. Generation parameters sit outside the cache key**

Temperature and max output tokens are module constants, not part of the hash. That means
changing one and re-running would silently return responses generated under the old
setting. The mitigation is that both are stored inside every cache record, so a stale
entry is detectable by inspection, and the module docstring says the cache must be
cleared when they change. Chose this over hashing them in because the params are fixed
for the whole project and including them would fragment the cache for no benefit.

**8. Prompts stored as plaintext in cache files**

Each cache file holds the model, system, prompt, response and timestamp in readable
JSON. The cache is then greppable and diffable evidence of exactly what was sent, rather
than a directory of opaque hashes. 

Rejected: storing only the hash and the response.

**9. `CACHE_DIR` anchored to the repo root via `__file__`**

A relative path would create a second cache anywhere I invoked things from a different
directory. There is a test that imports the module from a temp directory in a subprocess
and asserts the path still resolves to the repo root, because a comment would not survive
a refactor but a failing test will.

---

## Environment and process

**10. Project moved off NTFS to ext4**

Originally on an auto-mounted Windows partition. NTFS under fuseblk does not carry Unix
permission bits or handle symlinks properly, which breaks git file modes and venv
internals. Related incident: for a while two copies of the project existed and a session
wrote code into the stale one. Consolidated to a single ext4 location and deleted the
duplicate.

**11. Raw CSV copied locally rather than symlinked to the external drive**

The symlink pointed into the auto-mounted partition, which dropped after a sleep cycle
and broke `make ingest` with a file-not-found. 493MB against available headroom is not
worth a dependency that disappears overnight.

**12. `pytest.ini` with `pythonpath = .` instead of packaging `src`**

Bare `pytest` does not put the project root on `sys.path`, so `from src import llm`
failed. A three line config fixes it for every invocation style. Rejected: turning `src`
into an installed package, which is more machinery than this needs.

---

## Ingestion

**13. DuckDB for reading, never line splitting**

`wc -l` reports 3,002,524 but the file has 2,811,774 rows. The difference is tweets
containing raw newlines inside quoted fields. Any line-based reader would shred those
rows silently. Rejected: naive splitting, and also rejected trusting `wc -l` as a row
count anywhere.

**14. Explicit column types, no inference**

`response_tweet_id` is declared VARCHAR because it is a comma-separated list, not a
number. Dates are parsed with an explicit `%a %b %d %H:%M:%S %z %Y` format. Type
inference samples the head of the file and guesses, then fails somewhere deep in 2.8M
rows where the first chunk was unrepresentative.

**15. Parquet intermediates in `data/interim/`, gitignored, with `make clean`**

Re-parsing the full CSV on every run wastes minutes that add up over a week. The
intermediates are derived from gitignored raw data so they are not committed, and
`make clean` removes them. Same staleness trap as the LLM cache, handled the same way:
the artifact must be regenerated when its inputs change, and that is written down.

---

## Thread reconstruction

**16. Threads built by walking the reply graph, never by sorting**

Tweet ids are not chronological. Tweet 1 has parent 3 and reply 2, so the true order is
3 then 1 then 2. Any approach that sorts by id produces conversations in the wrong order
and would not obviously look broken. Traversal is an iterative DFS with an explicit
stack, since 2.8M nodes would blow the recursion limit.

**17. `in_response_to_tweet_id` is the only source of graph edges**

It holds a single id, so every tweet has at most one parent and threads partition the
data exactly. `response_tweet_id` is used only as a cross-check and never to draw edges,
because building from it would give some tweets two parents and break the partition.
The cross-check result: 172,500 `response_tweet_id` claims name a tweet that is not in
the file at all, and **zero** name a tweet whose parent column disagrees. So the dangling
references are a dataset completeness issue, not a contradiction, and the reconstructed
structure is a tree rather than a DAG.

**18. Siblings ordered by `created_at`, with `tweet_id` only as a tiebreak**

Timestamp first because ids are non-chronological; id second only to make identical
timestamps deterministic. Verified this is safe: across all 2,013,577 edges, zero run
backwards in time and 56 are simultaneous, minimum gap 0 seconds. So timestamp ordering
along an edge is reliable in this dataset.

**19. Orphans and cycles labelled and kept, never dropped**

An orphan is a tweet whose parent id is absent from the file. There are 15,909 of them
across 3,862 threads, about 0.6% of all tweets. They are promoted to roots of their own
threads and flagged, so they can be filtered downstream with one condition instead of
vanishing from the row count. Zero cycles were found, but the handling exists so a cycle
could never hang the traversal. Every tweet appears exactly once in the output, which
makes row reconciliation a one line assert.

**20. Dialogues reconstructed by walking `parent_id`, not by filtering `branch_id`**

Caught during plan review before any code was written. Under a one-row-per-tweet schema,
a root with two children carries only one `branch_id`, so filtering on the other branch
silently loses the shared ancestors. `branch_id` is now documented as a leaf-path counter
only, and `path_to_root()` is the correct way to get a linear dialogue.

---

## Conversations

**21. A thread is not a conversation, so broadcasts get split**

The largest thread is 1,390 tweets: ATVIAssist posts a single service status update and
hundreds of unrelated customers reply underneath it, about pistol grips, missing
pre-order content, error codes, connection failures. Treating that as one conversation
would feed a brand announcement into the pipeline as if it were a customer's opening
message. Under a brand-initiated thread, each inbound reply with no inbound ancestor
becomes its own conversation root and carries its subtree.

**22. Deep inbound replies promoted rather than discarded**

The original rule split only at depth 1. Measuring what that would strand gave 2,186
conversations sitting below depth 1, covering 4,047 tweets, which is 4.5% of the 89,612
broadcast tweets. Inspection showed these are real complaints with real brand replies
underneath them ("I am paying you 40$ a month and I can't even use the service. FIX
IT!"). Promoted under separate `*_deep` source labels so they stay countable and can be
filtered out later if they turn out to be noisier than depth-1 conversations. Related
figure worth keeping straight: 3,212 is the number of inbound tweets inside those
conversations, of which 2,186 became roots and 1,026 inherited from an inbound ancestor.

Consequence for the report: broadcast conversation roots are no longer uniformly at
depth 1, so anything downstream assuming that is wrong and should use
`depth_in_conversation` instead.

**23. Orphan threads routed by their root's inbound flag**

2,201 inbound-rooted orphans are treated like customer conversations; 1,664
outbound-rooted orphans are treated like broadcasts and split. They carry distinct source
labels so they never silently inflate the main counts.

**24. Brand-to-brand tweets left unassigned**

8,952 tweets sit on an all-outbound path from a broadcast root, of which 8,650 are the
broadcast posts themselves. These are a brand talking to itself and belong to no
customer's conversation. Left with a null `conversation_id` and reported on their own
line rather than forced into a conversation to reach 100% assignment. Any downstream join
on `conversation_id` has to decide inner versus left rather than inheriting this choice
by accident.

**25. Conversation ids drawn from two namespaces**

Customer-rooted conversations use `thread_id`; broadcast-derived ones use the root
tweet's `tweet_id`. Verified rather than assumed that these cannot collide:
`thread_id` equals the root's `tweet_id` for all 798,197 threads, and `tweet_id` is
asserted unique at ingest. The summary also asserts `conversation_id.is_unique` so the
guarantee does not depend on that reasoning surviving a future change.