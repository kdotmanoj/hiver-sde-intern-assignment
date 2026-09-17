# Report: SpotifyCares support agent

## 1. Problem framing

"Good" here means two things. When a customer's message resembles something
SpotifyCares has already answered in public, the agent should produce a reply
grounded in that past answer. When it does not, or when the problem needs
account access, the agent should say so and hand the ticket to a person
instead of guessing.

What this project does not attempt:

No fine-tuning. The classifier and generator are prompted models; nothing in
this project trains or adjusts weights.

No multi-turn dialogue management. Half of the conversations in the dataset
are exactly two tweets, one complaint and one reply (median 2, mean 3.4). The
agent is built for that shape: one message in, one reply out. It does not
carry state across turns.

No live integration. The agent reads from and writes to files. It does not
connect to Twitter or to Spotify's own account systems.

No multilingual handling. SpotifyCares' historical replies are 0.07%
non-English; inbound customer messages are higher but still a small
minority, and out of scope.

A retrieval corpus of roughly 1,850 conversations, not the full pool of about
27,400 that are actually eligible. The corpus is capped at that size so the
project stays reproducible offline with no API key; this is a real limitation
on reply quality, not a modeling choice, and it is the first item in section
6.

SpotifyCares was chosen as the brand ahead of larger candidates such as
AmazonHelp and AppleSupport specifically because its public replies contain
resolvable troubleshooting content to ground a generated reply in, not just
redirects to DM.

## 2. What I built

Six pipeline stages turn 2.8 million raw tweets into an evaluated agent:
ingest, threads, conversations, sample, taxonomy, agent. The corpus is read
with DuckDB rather than a line-based parser, split into a reply tree by
walking parent pointers rather than sorting by id, cut into individual
conversations by splitting broadcast threads at each inbound reply, narrowed
down to a working set of SpotifyCares conversations with a real
(non-deflection) reply to ground on, read by hand to define nine intent
classes because clustering did not separate them on its own, and finally
handed to an agent that classifies, retrieves, replies and decides whether to
escalate.

- ingest: 2,811,774 tweets (`wc -l` reports 3,002,524; the difference is
  tweet text containing raw newlines), 1,537,843 inbound and 1,273,931
  outbound, across 108 brand handles.
- threads: 798,197 threads reconstructed from the parent column, zero cycles,
  zero contradictions between the parent column and the response_tweet_id
  cross-check.
- conversations: 813,446 conversations after splitting broadcast threads at
  each inbound reply; 8,952 tweets left unassigned as brand-to-brand traffic.
- sample: 28,380 SpotifyCares conversations, 28,326 after dropping oversized
  threads, 27,425 with a substantive first reply; the retrieval corpus is
  ~1,849, drawn from the 2,000-row taxonomy sample minus the 150 golden ids.
- taxonomy: nine intent classes, defined by reading cluster exemplars by hand
  after k-means silhouette scores of 0.037 to 0.045 across k=5 to 12 showed
  no real cluster structure.
- agent: four steps, classify the intent, retrieve the k=3 nearest past
  conversations, generate a reply grounded in them, decide whether to
  escalate, evaluated against two baselines, a trivial canned-reply baseline
  and a k-NN label-voting baseline.

## 3. Results vs baselines

Intent classification, macro-F1 over 150 golden examples:

| system | macro-F1 |
|---|---|
| agent | 0.716 (95% paired bootstrap CI [0.632, 0.780]) |
| k-NN (label voting, leave-one-out) | 0.376 |
| trivial (majority class) | 0.039 |

Escalation, F1 against the 150 gold escalate labels, by signal family:

- intent family alone: 0.735
- combined rule (intent, similarity and text together): 0.602
- similarity family alone: 0.074
- text family alone: 0.103

The combined rule scores below intent alone (0.602 vs 0.735), because the
similarity and text families add noise rather than signal.

The agent escalated 46 of 150 cases (30.7%) against 37 hand-labelled (24.7%).

Judge means on the agent's 150 replies, 1 to 5 scale:

- groundedness: 4.80
- correctness: 4.97
- tone: 4.91

Judge-human agreement, n=60, Cohen's kappa: groundedness -0.048, correctness
-0.014, tone -0.014. All three are near zero because both raters cluster at
4-5, so high raw agreement (68/88/90%) is chance-level.

## 4. Failure analysis

Forty intent disagreements between the agent and the gold labels. Five
recurring modes account for most of them.

1. Payment and money keyword capture (about 6 cases). Any message mentioning
   money, payment or premium is pulled toward subscription_query or
   billing_dispute regardless of the real problem. Examples: 93962 (an
   app-update complaint read as subscription), 2752846 (a frozen payment UI,
   an app bug, read as subscription), 2141762, 1882069.
2. Shuffle and algorithm complaints read as playback_library (about 4 cases).
   Feature complaints about how shuffle behaves are read as "music will not
   play." Examples: 2731505, 1757064, 2902821, 1659869.
3. Catalogue vs personal-library confusion (about 5 cases). content_issue and
   playback_library are swapped in both directions. Examples: 2908680,
   1672172, 1480468, 925215, 408916. This is one of the four boundaries the
   intra-annotator re-label also disagreed on, so it is a genuinely hard
   boundary, not just a model error.
4. Vague or short messages get a confident wrong guess (about 4 cases).
   Messages like "I need help," "what's going on," "why does Spotify mobile
   suck" get concrete labels the text does not support. Examples: 1336506,
   2919313, 2277222, 2037685.
5. Annotator-model boundary disagreement (about 3 to 4 cases). Cases where
   the model's label is defensible and the gold label is soft. Examples:
   654600, 1476578, 223039, 1877698. Quantified by the intra-annotator kappa
   of 0.862 rather than treated as pure model error.

## 5. What is misleading about my headline number?

- Annotator kappa ceiling is 0.862, so the agent's 0.716 reads against 0.86,
  not against 1.0.
- About 3 to 4 of the 40 intent "errors" are soft-gold cases, not model
  error, and were not relabelled, to avoid inflating the score against the
  test set.
- The judge saturates on two of three axes (correctness 4.97, tone 4.91) and
  shows no human agreement above chance on any axis, so judge means are not
  evidence of reply quality on their own.
- Survivorship bias: unanswered customers are absent from the corpus (for
  example, all 49,517 inbound tweets mentioning @AmericanAir are assigned to
  a conversation), so every number here is conditioned on the conversation
  having been engaged in the first place.
- Only one annotator. The 0.862 kappa is intra-annotator agreement, not
  inter-annotator agreement.
- n=150 golden examples gives a wide confidence interval on the headline
  number (0.632 to 0.780).
- The retrieval corpus is about 1,849 of roughly 27,400 eligible
  conversations, about 7% of what is available.
- The similarity escalation threshold (SIMILARITY_FLOOR = 0.60) was set from
  the similarity distribution itself, not validated against the escalate
  labels.
- The golden set is drawn from the same 2,000-row pool whose cluster
  exemplars were read to define the taxonomy, so it is not fully independent
  of it.
- All three systems are judged for groundedness against the same context the
  agent itself retrieved, which favors the agent since its reply was
  generated from exactly that context.
- The k-NN baseline is doubly favoured: it votes on the labels it is scored
  against (leave-one-out over the golden set), and the taxonomy was built in
  the same embedding space it votes in. The real gap between the agent and a
  fair baseline is wider than 0.716 vs 0.376 suggests.

## 6. One more week

- Expand the retrieval corpus from about 1,849 conversations to the full
  pool of roughly 27,400.
- Re-derive or drop the similarity escalation signal; validate a threshold
  against the escalate labels instead of the similarity distribution
  (currently F1 0.074).
- Fix payment and money keyword capture in the classifier prompt, the
  largest single failure mode (about 6 of 40 disagreements).
- Add a second annotator to get an inter-annotator kappa alongside the
  current intra-annotator 0.862.
- Redesign the correctness and tone judge rubrics with harder anchors to
  break the saturation at 4.97 and 4.91.
