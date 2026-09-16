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

## Brand selection

**26. Deflection measured with a residual-word threshold of 3**

A reply counts as a pure deflection only if it contains a channel-switch phrase
AND, after stripping greetings, apologies, agent signatures, willingness
boilerplate ("we'd be happy to help"), purpose clauses ("so we can look into
this") and the redirect clause itself, three or fewer content words remain.

The reason for a residual rule rather than a plain regex match: 24.80% of all
top-8 replies contain a redirect phrase, but many carry real content alongside
it, for example "which iPhone and version of iOS are you using? please dm us".
That is a diagnostic question with a redirect attached, not a brush-off.

What the threshold does not measure: it captures bare redirects, not "redirect
instead of helping". A brand that always asks for a DM but always attaches a
real question scores low here even though every conversation still leaves
Twitter. That is why redirect-present rate is reported alongside it rather
than instead of it. 3 is a tunable knob and the rate moves if it moves;
validated against a 400-reply sample via `--residuals`, where every flagged
reply was a genuine bare redirect.

**27. Both bounds reported, not one number**

Pure deflection is the floor, redirect-present is the ceiling, and the truth
sits between. Reporting only one would make the headline depend entirely on
the threshold above.

This changed a reading rather than just decorating the table. comcastcares
looked mid-band at 7.64% pure deflection but is 54.37% redirect-present, the
widest gap in the table, with a 7-word median residual. It is a "redirect with
substance" account, which the single column hid. Uber_Support went the other
way: top on both bounds (63.31 / 37.50) and lowest median residual at 4 words,
so three independent signals agree and the read does not rest on the threshold.

**28. Median residual word count added as a third measure**

Deflection rate measures what a brand refuses to do. Median residual words,
computed across all replies rather than just redirects, measures how much
substance a typical reply actually carries. For the question that matters here
(can generated replies be grounded in this brand's history?) the second is the
direct measure.

It separates the field better than deflection rate: AppleSupport, SpotifyCares
and SouthwestAir at 9 words; AmazonHelp and AmericanAir at 8; Delta at 6,
depressed by its 10.08% multi-part rate since half a split reply carries half
the substance.

**29. Help-article links do not count as deflection**

Only channel switches count: DM, call, email, contact form. A link to a fix
("a clean reinstall of the app should help out: `<url>`") is an answer that
happens to include a link, not a refusal to answer here. The distinction is
whether the reply resolves the problem in public or moves it elsewhere, and
that is the whole question the survey exists to answer.

This is not a neutral call. It materially lowers AppleSupport, whose single
largest template is "here's what you can do to work around the issue: `<url>`"
at 5.88% of its replies. Chose the strict definition precisely because the
looser one would have flattered the result I already expected.

**30. Two classifier leaks documented rather than fixed**

Found while reading the 400-reply residual sample:

- "let's work together in dm here: `<url>`" leaves 4 residual words and so
  survives the threshold, despite being a pure deflection.
- Form-fill redirects ("please fill in this form: `<url>`") are not in the
  pattern set at all.

Both bias every brand's deflection rate downward. Chose to document the
direction of the bias rather than keep tuning, because this is a one-off
selection script and further tuning would have cost time the graded
deliverables need. The reported rates should be read as lower bounds.

**31. Brand chosen: SpotifyCares**

Not the highest volume. Chosen because its public replies contain actual
resolution content to ground on: "hold Sleep/Wake + Volume Down for 10
seconds", "what device, operating system, and Spotify version are you using?",
"a clean reinstall should help out", "Windows Phone is currently in
maintenance mode".

Numbers: 28,380 conversations, 2.46% pure deflection, 30.38% redirect-present,
9-word median residual, 0.07% non-English.

What each alternative lost on:

- **AmazonHelp** (82,623 conversations, the largest) is 14.57% non-English
  through the same handle, mostly Japanese and German. Multilingual handling
  is explicitly out of scope, and the generator and judge would both have to
  cope with it.
- **Uber_Support** is 37.50% pure deflection and 63.31% redirect-present with
  a 4-word median residual. It routes to DM rather than resolving in public,
  so there would be little historical resolution to ground a draft in and
  little for a judge to score.
- **AppleSupport** (81,640 conversations, 0% non-English, 0% multi-part) is
  the closest rejected option and looks cleanest on paper. Rejected because
  its single biggest template is a bare help-article URL at 5.88% of replies,
  and a URL cannot be grounded in.
- **SouthwestAir** matches Spotify on median residual (9) and beats it on
  deflection (1.21%), but has less troubleshooting-shaped content; airline
  issues resolve through account lookups more than through steps a reply can
  contain.

Honest caveat for the report: this was not unambiguous. AppleSupport and
SouthwestAir also sit at 9 median residual words. SpotifyCares wins on the
combination of low deflection, clean language and troubleshooting-shaped
content rather than on any single column.

**32. Multi-part replies concatenated before use**

6.23% of SpotifyCares replies are split across tweets ("1:", "2:"), so a
single tweet is often half an answer. Consecutive replies by the same brand
author within a conversation are joined in `created_at` order into one logical
reply before indexing, so retrieval cannot return half a sentence as a
"historical resolution".

**33. Reply-rate column kept but explicitly disclaimed**

"% of conversations where the brand replied" is 100% by construction under the
attribution rule (a conversation belongs to brand X iff X authored an outbound
tweet in it), so it was recomputed over conversations where a customer tweeted
`@brand`.

That still does not make it a service level. All 49,517 inbound `@AmericanAir`
mentions in the corpus are already assigned to a conversation, meaning the
dataset only contains threads that got engaged. The unanswered population is
not in the data at all, so the residual few percent below 100 is just
conversations answered by a different brand. Kept the column with a hard
footnote rather than deleting it, because the fact that it cannot be computed
is itself a finding.

## Working dataset (stage 4)
 
**34. Multi-part replies merged on shared parent or self-reply chain, never on
time-adjacency alone**
 
Split replies turned out to be *siblings*, not chains: in conversation 1878,
tweets 1880 (`1: ...`) and 1879 (`2: ...`) are both children of customer tweet
1877. Of 1,483 adjacent same-author outbound pairs in the SpotifyCares subset,
1,359 share a parent, 17 are a parent-to-child chain, and 106 are neither.
 
That last group is the trap. Those 106 are a brand replying to two different
customers under one broadcast, adjacent in time. A merge rule keyed on
"consecutive replies by the same author" would have fused two unrelated
customers' answers into one text and fed that to retrieval as a single
historical resolution.
 
The rule therefore requires the parent condition: a tweet continues a merge
group only when it shares the previous tweet's parent, or its parent *is* the
previous tweet.
 
**35. Part markers outrank `created_at` when ordering siblings**
 
The verified fact that timestamps never run backwards covers parent-to-child
edges only. Siblings are not on an edge with each other, so that guarantee does
not extend to them, and sibling order is exactly what determines whether a
merged reply reads forwards or backwards.
 
Measured separately: across 1,258 fully-marked sibling groups, `created_at`
order and `1:`/`2:` marker order agree in every case. Zero disagreements.
 
The precedence rule is encoded anyway. Sort by `created_at`, then assert marker
numbers are non-decreasing; where they are not, re-sort by marker and increment
a counter in the stage summary. This is a guard against a disagreement that does
not currently exist, not a fix for one that does. It fires zero times today, and
if the dataset ever changes it fails loudly instead of silently producing a
backwards answer.
 
**36. Conversations over 20 tweets dropped, and what that actually removes**
 
54 conversations (2,773 tweets) of 28,380 fall out, leaving 28,326.
 
The cap was intended as a size guard but its main value turned out to be
different. The largest thing it removes is conversation 2812, rooted at
`@115888` — a Spotify marketing handle whose tweets carry `inbound=True`. A
promotional post is therefore indistinguishable from a customer opening message
at the schema level, and it had roughly 130 replies underneath it. The cap is
doing double duty as a brand-handle-as-customer guard.
 
Applied to the raw tweet count before merging, so "20 tweets" means tweets, not
post-merge turns.
 
**37. Orphan part-markers flagged, not dropped**
 
65 replies carry a part marker but had no sibling to merge with: a `2:` whose
`1:` is not in the corpus. These are permanently half-answers and cannot be
repaired.
 
They are flagged with an `orphan_part` boolean rather than removed, for the same
reason orphan tweets and cycle tweets were kept in earlier stages: dropping rows
inside a stage changes the dataset's shape invisibly. The exclusion happens
visibly at the retrieval index instead, as one predicate.
 
The count understates the impact. 37 of the 65 (57%) are a conversation's
`first_reply`, which is the field retrieval actually uses. Half-answers are
disproportionately opening replies, so the flag matters far more than 65 out of
41,980 suggests.
 
**38. `first_reply` reuses the deflection rule rather than defining "substantive"
again**
 
The first SpotifyCares reply in a conversation is often a bare "DM us /LS", so
the target reply is the first one that is *not* a pure deflection under decision
26's existing rule. No new threshold was introduced.
 
27,425 conversations have a substantive first reply. 901 are deflection-only and
are kept with null `first_reply_*` fields rather than dropped, so the count stays
visible.
 
**39. Deflection rule extracted to `src/deflection.py`**
 
`scripts/brand_survey.py` is a one-off selection script whose own docstring says
nothing downstream reads it. Once `src/sample.py` needed the same rule, that
stopped being true, and a pipeline stage importing from a one-off script is
backwards.
 
Pure move, no logic change, verified rather than asserted: captured
`brand_survey --report` output before and after the extraction and diffed them.
Byte-identical, and still matching the table in `notes/findings.md`.
 
`MULTIPART` deliberately did not move. `sample.py` needs the part *number* as a
capture group while `brand_survey` needs only a boolean, and changing
`brand_survey`'s regex would have changed its published table and broken the
byte-diff test.
 
**40. Part markers and continuation mentions stripped from merged replies**
 
A merged reply has to read as one reply. Two separate strips do that:
 
- **Continuation parts** lose their leading `@mention` and their part marker.
  Without this, conversation 1878's merged reply reads "...there's info
  about... @116129 2: Spotify content here...". Both are Twitter routing
  artifacts, not content.
- **The first part** loses its part marker too, but keeps its `@mention`. A
  `1:` means "part 1 of N", and after merging there is no part N, so it is
  stale. The mention stays because every reply has one, so it is uniform.
Unmerged replies are untouched, which keeps `orphan_part` honest: a lone `2:`
with no sibling stays visible as the half-answer it is.
 
The marker regex requires a single digit and allows a letter directly after the
delimiter. Both constraints came from real cases: "24/7 support" parsed as part
24, and one reply reads `1:Thanks` with no space, which the original pattern
missed. A bare "any whitespace or nothing" delimiter would have made "we close
at 5:30pm" parse as part 5.
 
Original tweet ids are retained in `first_reply_tweet_ids`, so nothing is lost.
This edits reply text, which is why it is recorded rather than done quietly.
 
**41. Nested timestamps stored UTC-naive**
 
A tz-aware `created_at` nested inside the `turns` struct produces a parquet file
that this project's own `read_parquet()` cannot open: duckdb needs `pytz` for
nested TIMESTAMPTZ but not for a top-level one. Stored the nested timestamp
UTC-naive instead of adding a dependency to work around a library gap. Lossless,
because ingest forces `utc=True` across the whole corpus.
 
Found by a round-trip regression test, not by inspection. The write succeeded and
the pipeline looked healthy; only reading the file back revealed it. That test
now runs in the suite.
 
**42. `PART_MARKER` restricted to a single digit**
 
A test caught "24/7 support..." parsing as part 24. No live false positives in
the current data, but it was a mis-sort waiting to happen the first time a reply
mentioned a time range or a date.

**43. Retrieval corpus is the 2,000-row taxonomy sample, not all 27,400**
 
~1,849 conversations after removing the 150 golden ids and orphan first
replies. The eligible pool is roughly 27,400.
 
The reason is the repro requirement, not a modelling judgement. `embed.py`
writes one committed `.npy` per vector so that `make eval` replays offline with
no API key and no model download. Embedding the full pool would add about 27,000
blobs and 45MB to the repo purely to keep that property.
 
This is a real cost and belongs in the report's limitations: retrieval draws
from a pool roughly 7% the size of what is available, so grounding quality is
almost certainly worse than it would be with the full corpus. Expanding it is
the first item on the "one more week" list.
 
**44. k = 3 retrieved neighbours**
 
Enough precedent to ground a reply that is itself tweet-length, while keeping
the prompt small. The generator runs on a free tier with a token-per-minute
ceiling, and a full evaluation run is 150 reply calls.
 
**45. `SIMILARITY_FLOOR = 0.60`**
 
Set from the measured distribution, not picked as a round number. Top-1 cosine
similarity across the 150 golden openings against the corpus: min 0.405,
p10 0.588, p50 0.736, p90 0.833.
 
0.60 sits just above the 10th percentile and fires the `no_similar_precedent`
signal on 11.3% of the golden set. 0.70 was rejected because it would fire on
35.3%, which is not a plausible rate of "we have never seen anything like this"
when the median similarity is 0.736.
 
The claim the signal encodes: if nothing in the corpus resembles this message,
there is no past reply to ground a draft in, so a human should write it.
 
**46. Legal/safety lexicon narrowed to exclude minor-related terms**
 
The labelling guide says escalate on "anything involving a minor". Encoding that
literally turned out to be wrong on this corpus: "my son", "my daughter", "my
kid" and "my child" produced 89 of the 101 hits for that signal, and every one
of them is family-plan mechanics ("when I try to add my son to my family account
it says whoops"), not safeguarding.
 
Left in, the signal would have escalated nearly every family-plan message on a
safety basis, which is both wrong and would have swamped the per-family
escalation metrics with noise. The clause now covers chargeback, fraud, lawyer
and legal action.
 
Same standard as the deflection patterns in decision 26: counted against the
real corpus before being accepted, rather than written from intuition.
 
Related and deliberately left unfixed: the abuse lexicon fires on golden id
93962 ("Screw @115888 for the massive mobile app update..."), which is labelled
`escalate=false` because it is ambient swearing rather than abuse aimed at
anyone. It is commented as a known false positive in the source so the
per-family metrics quantify it rather than hide it.
 
**47. Escalation is a flat OR over named signals, with per-family attribution**
 
No weights, no scoring model. Six signals in three families — intent class,
retrieval similarity, and text — and the decision is `escalate` if any fired.
The rule is deliberately simple enough to derive on a whiteboard, which is the
point of a policy that decides whether a human sees a ticket.
 
The attribution exists because the combined number alone would be misleading.
`billing_dispute` and `account_access` are about 27% of the golden set and
almost all escalate, so intent alone was expected to carry most of the decision.
`signals_by_family()` records which families fired on each case so the
evaluation can score each one independently and show whether similarity and text
add anything over the intent lookup. The expectation gets measured rather than
assumed.
 
The attribution is derived from a `SIGNAL_FAMILY` name-to-family lookup rather
than reimplemented, so it cannot drift from the rule, and a test asserts every
emittable signal has a family assigned — a new signal cannot silently vanish
from the report.
 
**48. Escalated cases skip reply generation in production, but not in evaluation**
 
By default `answer()` does not generate a reply when the decision is `escalate`.
That saves quota on exactly the cases where the draft would be discarded, and it
makes the escalation decision load-bearing rather than decorative.
 
Evaluation needs the opposite. Scoring the judge only on auto-handled cases
would make every quality number conditional on the escalation rule being
correct, which is one of the things under test, and it would drop the hardest
cases from the judge set. `--force-reply` generates regardless; the decision is
still computed and recorded in full, and `reply_generated` is stored separately
from `decision` so a forced run stays distinguishable from a production one.
 
No wasted calls either way: the reply prompt does not contain the escalation
decision, so the cache key is identical and a later default run reuses the
replies a forced run already cached.

**49** Model switched to gemini-3.5-flash-lite after Gemini 3.6 Flash turned out to be 20 requests/day, not the ~1,000 the docs suggest. Discovered by exhausting it at call 21 of a 300-call run. Second time Gemini's own metadata proved unreliable (the first being the 404 on a model the list endpoint advertised). 22 cache entries under the old model are stranded and kept as a record.

**50** MAX_LIVE_CALLS = 320 with a preflight that counts uncached calls before the loop starts. A daily quota doesn't queue — it fails, and the calls already spent stay spent. The preflight is an upper bound when the classify cache is cold, since the reply prompt contains the intent and can't be known in advance.

Then start the baselines session. Don't wait for the run to finish — the two can proceed in parallel since baselines make no LLM calls.