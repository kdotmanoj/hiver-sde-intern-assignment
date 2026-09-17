# Findings

Things that are true about the data, measured rather than assumed. Separate
from decisions.md, which records choices. This file is the raw material for the
report's data-framing and misleading-numbers sections.

---

## The file itself

`wc -l` reports 3,002,524 lines but there are **2,811,774 rows**. The gap is
tweets containing raw newlines inside quoted fields. Any line-based row count
is wrong by about 190,000, which is a 6.8% error.

Tweet text also contains commas and quotes, so the CSV is properly quoted and
needs a real parser.

**1,537,843 inbound** (customer) tweets and **1,273,931 outbound** (brand).
108 distinct brand handles appear as outbound authors.

`tweet_id` is unique across the file, so `tweet_id -> parent` is a
well-defined function.

## Identifiers and text

**Tweet ids are not chronological.** Tweet 1 has parent 3 and reply 2, so the
true conversation order is 3, then 1, then 2. Sorting by id produces wrong
conversation order and does not look obviously broken.

**Customers are anonymised as `@<number>`.** So are brand alternate handles.
`@115873` is Uber's main account and `@158` appears as the brand in a thread
of its own. A numeric mention does not mean "a customer", which matters for
any text cleaning that assumes it does.

**Agent signatures vary by brand**: SpotifyCares uses `/LS`, Delta uses `*QB`,
AmazonHelp uses `^TN`, Tesco writes `- Nick`. Any signature stripper has to
cover `^`, `*`, `/` and trailing `- name`.

**Brand replies get split across several tweets.** Delta writes `1/2` and
`2/2`, Tesco `1/3`, Spotify `1:` and `2:`. A single tweet is often half an
answer, so retrieval over individual tweets can return half a sentence.
Measured per brand below.

`created_at` is in Twitter's own format (`Tue Oct 31 22:10:47 +0000 2017`) and
needs an explicit `%a %b %d %H:%M:%S %z %Y`, not inference.

**Spotify agent signatures are more common than a naive probe suggests.** 40,766
of the SpotifyCares replies carry a trailing `/XX`: 39,851 two-letter and 915
one-letter. An initial probe counting only end-of-text occurrences found 31,465,
missing the 9,752 that sit immediately before a trailing `https://t.co/...`.

32 merged replies are signed by more than one agent, meaning a split reply was
finished by a different person than started it.

## Graph structure

**798,197 threads**, reconstructed from the parent column.

Root breakdown: 787,346 customer-initiated (inbound, null parent), 6,989
brand-initiated (outbound, null parent), 3,862 orphan threads.

**Orphans are small.** 15,909 tweets across 3,862 threads have a parent id
that is absent from the file, about 0.6% of all tweets. They look like opening
messages but are fragments starting mid-conversation. Small enough to keep and
flag rather than needing special treatment.

**Zero cycles.** The handling exists so traversal cannot hang, but the data has
none.

**172,500 `response_tweet_id` claims point at a tweet that is not in the file
at all, and zero point at a tweet whose parent column disagrees.** So the two
columns never contradict each other about parentage; the dangling references
are a dataset completeness issue. The reconstructed structure is a tree, not a
DAG.

**Timestamps are monotonic along every edge.** Across 2,013,577 edges, zero run
backwards in time, 56 are simultaneous, minimum gap 0 seconds. Ordering
siblings by `created_at` is therefore safe in this dataset. (The dataset was
probably assembled by walking forward from roots, which would explain it.)

**Branching is common, not an edge case.** 222,426 rows have a comma-separated
`response_tweet_id`, and 172,399 tweets actually have more than one child. The
difference is explained by the dangling references above.

Max depth 649. Max 982 branches in a single thread.

## Threads are not conversations

**The largest thread is 1,390 tweets and is not a conversation.** ATVIAssist
posts one service status update and hundreds of unrelated customers reply
underneath it: pistol grips, missing pre-order content, error code 4220,
connection failures, abuse. Treating that as one conversation would feed a
brand announcement into the pipeline as if it were a customer's opening
message.

**The deepest thread (650 tweets, depth 649) is a customer pile-on**, not
support at all. Every author is a bare number, every tweet is a wall of 15+
@mentions, and the only brand-ish participant (`@158`) never speaks. It
disappears automatically once conversations are filtered to those where the
target brand actually replied, so no special rule is needed.

After splitting broadcasts at inbound replies: **813,446 conversations**, with
8,952 tweets left unassigned because they sit on an all-outbound path (a brand
talking to itself), of which 8,650 are the broadcast posts themselves.

**Broadcast-derived and customer-rooted conversations have the same shape.**
Both: median 2 tweets, p25 2, p75 4, p90 6, mean 3.4. They diverge only in the
tail (max 974 vs 162). Once split correctly, a broadcast-derived conversation
is not a different kind of object. This is only visible because the two
distributions were computed separately.

**Half of all conversations are two tweets: one complaint, one reply.** Median
2, mean 3.4. So the reply-drafting task is usually "given one message, produce
one reply", not multi-turn dialogue management. This shapes the scope of what
needs building.

**2,186 conversations sit below depth 1** inside broadcast threads, covering
4,047 tweets (4.5% of the 89,612 broadcast tweets). A depth-1-only split rule
would have discarded them. Inspection shows they are genuine complaints with
genuine brand replies underneath, for example "I am paying you 40$ a month and
I can't even use the service. FIX IT!".

**0.37% of conversations involve more than one brand** (3,018 of 811,906).
Small enough to count under each rather than resolve.

## Brand behaviour

Top 8 brands by conversation count, over 1,264,979 brand replies total:

| brand | conversations | median tweets/conv | redirect-present % | pure deflection % | median residual words | non-English % | multi-part % |
|---|---|---|---|---|---|---|---|
| AmazonHelp | 82,623 | 3.00 | 7.92 | 2.35 | 8 | 14.57 | 2.49 |
| AppleSupport | 81,640 | 2.00 | 34.32 | 7.25 | 9 | 0.00 | 0.00 |
| Uber_Support | 42,223 | 2.00 | 63.31 | 37.50 | 4 | 0.03 | 0.00 |
| SpotifyCares | 28,380 | 2.00 | 30.38 | 2.46 | 9 | 0.07 | 6.23 |
| AmericanAir | 26,564 | 2.00 | 12.52 | 1.31 | 8 | 0.00 | 0.03 |
| Delta | 26,508 | 2.00 | 13.47 | 2.54 | 6 | 0.05 | 10.08 |
| comcastcares | 24,204 | 2.00 | 54.37 | 7.64 | 7 | 0.18 | 0.03 |
| SouthwestAir | 22,909 | 2.00 | 15.99 | 1.21 | 9 | 0.06 | 0.08 |

Definitions: redirect-present is any channel-switch phrase anywhere in the
reply. Pure deflection is a redirect with three or fewer content words left
after boilerplate is stripped. Median residual words is how much content a
typical reply carries once greetings, apologies, signatures and redirect
clauses are removed.

**Uber_Support barely does support in public.** Top on both bounds (63.31 /
37.50) and lowest median residual at 4 words. Three independent signals agree,
so this is not an artifact of the deflection threshold.

**comcastcares is a "redirect with substance" account.** The widest gap in the
table: 54.37% of replies push to DM, but only 7.64% are bare, and the median
residual is 7 words. On the deflection column alone it looked mid-band. This
is the case that justifies reporting both bounds.

**Median residual words separates brands better than deflection rate does.**
AppleSupport, SpotifyCares and SouthwestAir at 9; AmazonHelp and AmericanAir
at 8; Delta at 6, depressed by its 10.08% multi-part rate since half a split
reply carries half the substance.

**AmazonHelp is 14.57% non-English** through the same handle, mostly Japanese
(`こんにちは、アマゾン公式です`) and German. The script heuristic catches this
cleanly.

**AppleSupport's single biggest template is a link**, "here's what you can do
to work around the issue: `<url>`", at 5.88% of its replies. Large volume, low
non-English, but you cannot ground a generated reply in a URL.

**SpotifyCares gives real troubleshooting in public.** Examples from the data:
hold Sleep/Wake + Volume Down for 10 seconds; what device, OS and Spotify
version are you using; a clean reinstall should help; Windows Phone is
currently in maintenance mode. This is the material retrieval needs.

## Dataset-level bias (for the misleading-numbers section)

**Unanswered customers do not exist in this corpus.** There are 49,517 inbound
tweets mentioning `@AmericanAir`, and all 49,517 are assigned to a
conversation. The dataset only contains threads that were engaged, so there is
no observable population of customers a brand ignored.

Consequences: any "brand reply rate" is 96 to 99% by construction and is not a
service level. An escalation model trained here can never learn what "ignored"
looks like. Every claim about coverage is conditioned on the conversation
having been engaged in the first place.

**Non-English percentages are a floor, not an estimate.** Detection is a
non-Latin script ratio plus a function-word test, not a language model. It
catches Japanese and Arabic cleanly and undercounts Spanish, Portuguese and
French.

**Two known misses in the deflection classifier**, both biasing the rate
downward: replies like "let's work together in dm here: `<url>`" survive at 4
residual words, and form-fill redirects ("please fill in this form: `<url>`")
are not in the pattern set.

**A brand's own marketing handle can appear as a customer.** Conversation 2812 is
rooted at `@115888`, a Spotify promotional account whose tweets carry
`inbound=True`. At the schema level there is nothing distinguishing a promo post
from a customer's opening message. The >20-tweet cap removes this particular one,
but the general problem is unbounded: any brand handle flagged inbound will be
read as a customer. Intent clusters and golden-set samples drawn from opening
messages can therefore contain brand marketing copy.

## The SpotifyCares working set

28,380 conversations where SpotifyCares authored at least one outbound tweet;
28,326 after dropping the 54 over 20 tweets. 91,827 tweets, 89,054 after the cap,
87,677 turns after merging.
 
**Multi-part replies are siblings, not self-reply chains.** Of 1,483 adjacent
same-author outbound pairs: 1,359 share a parent, 17 are parent-to-child, 106 are
neither (a brand replying to two different customers under one broadcast). Sibling
groups: 1,408 pairs, 31 triples, 2 quads.
 
**1,345 groups actually merged, across 1,290 conversations (4.55%).** This is
lower than the 1,441 sibling sets that exist, because 115 of those sets have a
customer tweet falling between the parts in time, so the parts are not consecutive
and correctly do not merge. Decomposes exactly: 1,327 sibling-shaped + 17 chain +
1 mixed.
 
**Zero groups needed re-ordering by part marker.** Across 1,258 fully-marked
groups, `created_at` order and marker order agree in every case. This is a
separate measurement from the parent-child timestamp monotonicity finding;
siblings are not on an edge with each other, so the earlier guarantee does not
cover them.
 
**65 replies are unrepairable half-answers** — a part marker with no sibling in
the corpus. 37 of those 65 (57%) are a conversation's `first_reply`, so
half-answers are heavily concentrated in exactly the field retrieval uses.
 
**901 of 28,326 conversations (3.2%) have no substantive Spotify reply at all** —
every reply in them is a pure deflection. Kept with null `first_reply` fields.
This is the usable-pool ceiling for retrieval: 27,425 conversations.
 
## Pre-clustering intuition
- Library/Playback clusters
- Account access clusters
- Billing issues

## The clustering did not produce the taxonomy

k-means silhouette scores were 0.037-0.045 across k=5..12 with no elbow — no
real cluster structure. Reading the exemplars confirmed it: a single cluster
contained app bugs, playback problems, catalogue issues, how-to questions and
feature requests. The clustering grouped messages by surface vocabulary, not by
intent. The clusters were a reading aid; the taxonomy came from reading them,
and that is stated wherever the taxonomy is described.

## Intent classification results

Agent macro-F1 0.716 [0.632, 0.780] (95% paired bootstrap), against k-NN 0.376
and trivial 0.039. The agent roughly doubles the label-driven k-NN baseline and
is ~18x the majority-class trivial baseline. Zero intent parse failures across
150 classifications.

The k-NN 0.376 is an optimistic ceiling, not a fair competitor: it votes over
the golden labels themselves (leave-one-out), so it has seen human labels for
near-identical messages while the LLM classifier has seen none, and the
taxonomy was defined from the same embedding space it votes in.

Intra-annotator agreement is kappa 0.862 (raw 44/50 on a blind re-label after
an overnight gap). This is the ceiling: the agent's 0.716 should be read
against 0.86, not against 1.0, because even the annotator cannot resolve the
taxonomy's four hard boundaries consistently.

## Intent failure modes

40 intent disagreements between agent and gold. Split by hand into: ~3-4 where
the model's label is defensible and the gold label is soft (not relabelled, to
avoid inflating the score against the test set); the rest genuine model errors
or genuinely-ambiguous boundary cases. Five recurring modes:

1. Payment/money keyword capture (~6). Any message mentioning money, payment or
   premium is pulled toward subscription_query or billing_dispute regardless of
   the real problem. Examples: 93962 (app-update complaint -> subscription),
   2752846 (payment UI frozen, an app bug -> subscription), 2141762, 1882069.
2. Shuffle/algorithm complaints -> playback_library (~4). Feature complaints
   about how shuffle behaves are read as "music will not play". Examples:
   2731505, 1757064, 2902821, 1659869.
3. Catalogue vs personal-library confusion (~5). content_issue and
   playback_library swapped in both directions. Examples: 2908680, 1672172,
   1480468, 925215, 408916. This is one of the four boundaries the
   intra-annotator re-label also disagreed on, so it is genuinely hard, not
   just a model failure.
4. Vague/short messages -> confident wrong guess (~4). "I need help", "what's
   going on", "why does Spotify mobile suck" get concrete labels the text does
   not support. Examples: 1336506, 2919313, 2277222, 2037685.
5. Annotator-model boundary disagreement (~3-4). Cases where the model is
   defensibly right and the gold label is soft. Examples: 654600, 1476578,
   223039, 1877698. Quantified by the intra-annotator kappa of 0.862 rather
   than treated as pure model error.

## Escalation results

The escalation rule is effectively an intent lookup. Scored against the 150
gold escalate labels: the intent family alone scores F1 0.735, but the
combined rule scores only 0.602 — adding the other two families made it worse.
The similarity family scores F1 0.074 (17 fires, precision 0.118, so ~2
correct) and the text family scores 0.103 (2 fires).

The cause is a methodological error stated plainly: `SIMILARITY_FLOOR` was set
from the similarity distribution (just above its 10th percentile), never
validated against whether low similarity actually predicts needing a human. It
does not. Low retrieval similarity and "needs escalation" are close to
uncorrelated on this set.

The agent escalated 46/150 (30.7%) against 37 hand-labelled (24.7%), so it
over-escalates by ~6 points.

## Judge results

Judge means on the agent's 150 replies: groundedness 4.80, correctness 4.97,
tone 4.91. Correctness has almost no variance — nearly every reply scores 5 —
so that axis cannot discriminate between systems and should not be read as
evidence of quality. Tone is nearly as saturated. Only groundedness (range 2-5)
does real work.

The judge is not deterministic at temperature 0. Across 20 replies judged
twice with byte-identical prompts, groundedness disagreed with itself on 3/20
and tone on 2/20 (mean spread 0.150 and 0.100; correctness 20/20 identical).
Any reported judge mean carries that self-inconsistency.

## Judge-human agreement

n=60, stratified on the judge's groundedness so the low scores are
represented. Cohen's kappa -0.048/-0.014/-0.014 (grounded/correct/tone),
quadratic-weighted -0.012/-0.040/-0.038, raw agreement 68/88/90%. Weighting for
ordinality did not raise the numbers, so the disagreements are not near-misses
concentrated at 5-vs-4 — agreement is genuinely at chance level. The cause is
saturation: both the judge and the human cluster at 4-5, so high raw agreement
is what two ceiling-bound raters produce by chance, and there is no spread for
kappa to reward. The judge is usable as a coarse filter (it does catch the
groundedness-2 cases of invented replies), not as a precise quality meter.