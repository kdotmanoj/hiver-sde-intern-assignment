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

## Add to "Identifiers and text"
 
**Spotify agent signatures are more common than a naive probe suggests.** 40,766
of the SpotifyCares replies carry a trailing `/XX`: 39,851 two-letter and 915
one-letter. An initial probe counting only end-of-text occurrences found 31,465,
missing the 9,752 that sit immediately before a trailing `https://t.co/...`.
 
32 merged replies are signed by more than one agent, meaning a split reply was
finished by a different person than started it.
 
## Add a new section: "The SpotifyCares working set"
 
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
 
## Add to "Dataset-level bias"
 
**A brand's own marketing handle can appear as a customer.** Conversation 2812 is
rooted at `@115888`, a Spotify promotional account whose tweets carry
`inbound=True`. At the schema level there is nothing distinguishing a promo post
from a customer's opening message. The >20-tweet cap removes this particular one,
but the general problem is unbounded: any brand handle flagged inbound will be
read as a customer. Intent clusters and golden-set samples drawn from opening
messages can therefore contain brand marketing copy.