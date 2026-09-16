# Labelling guide -- SpotifyCares intent taxonomy

Written before any labelling, from reading 160 cluster exemplars across k=8.
Nine intents. Every opening message gets exactly one.

**Core rule:** label what the customer *wants done*, not what they are talking
about. Two messages that would get the same kind of reply from Spotify have the
same intent.

A note on where this came from: the k-means clusters did **not** map to intents.
Cluster 1 alone contained app bugs, playback problems, catalogue issues, how-to
questions and feature requests. Silhouette scores were 0.037 to 0.045 across
k=5..12 with no elbow, so the clustering grouped messages by surface vocabulary
(device names, "please add", billing words) rather than by what the customer
wanted. The clusters were a reading aid. The taxonomy came from reading them.

---

## The intents

### 1. playback_library

Content the user expects is missing, wrong, or will not play **on their account
or device**, while the catalogue itself is fine.

- songs in their library will not play
- downloads vanished after being offline
- playlists disappeared or were replaced
- library not listing songs they added
- a song stops every time they hit play

Examples: *"my songs in my library are not playing at all"*, *"all my playlists
disappeared and were replaced by a 'my music' playlist"*, *"after 30 days out to
sea, I lose all songs"*, *"thanks for randomly undownloading all my albums"*.

**Not this:** app crashes with nothing missing -> `app_bug`. Song removed from
Spotify entirely -> `content_issue`.

**Edge:** *"Spotify has removed all my tracks it appears"* -> playback_library.
"My tracks" means their library, not the catalogue, despite the wording.

### 2. app_bug

The app misbehaves **regardless of content**. Nothing is missing; the software
is broken. Includes service outages.

- crashes, freezes, hangs, failed installs
- blank screens, missing buttons, UI not responding
- says offline when online
- device-specific breakage (volume control, Chromecast, share on Pixel)
- platform outages (#SpotifyDown)
- generic "your app is buggy" complaints

Examples: *"my app on my Galaxy S7 Edge always freezes"*, *"my app on android
insists I'm offline when I'm not"*, *"your latest update is crashing during
install"*, *"error 404 trying to login on the app but website works"*.

**Not this:** asking how a feature works -> `how_to_question`. Asking for support
on a device that has none yet -> `feature_request`.

**Tiebreak:** if a reinstall would plausibly fix it and no content is missing,
app_bug.

**Edge:** *"why is my Spotify messing up"* -> app_bug. A vague problem report is
still a problem report, not `other`. Escalation handles "too vague to answer";
the intent label should not.

**Edge — the doesn't-exist vs exists-and-broken test:** *"app crashes on my
Pixel"* is app_bug. *"when will you support iPhone X"* is feature_request.

### 3. content_issue

Spotify's **catalogue** is wrong. Affects everyone, resolved by the content team
rather than support.

- song or album removed from the platform
- wrong version uploaded (live instead of studio)
- a track file itself is corrupted
- an official playlist discontinued or has the wrong songs on it
- release dates: an album is out but not on Spotify yet
- requests to add a specific artist, album or podcast
- regional catalogue gaps ("only one album by this artist in the UK")

Examples: *"a song you had on spotify is no longer there"*, *"Track 8 seems to be
the live version instead of studio"*, *"did you discontinue the ElectroNow
playlist?"*, *"me waiting for #reputation to be available, it's been a week"*,
*"please put travelin soldier back"*.

**Not this:** content missing only for them -> `playback_library`. Wanting a new
product capability -> `feature_request`.

**Edge:** *"It's skipping at the end, this track is broken"* -> content_issue.
The uploaded file is faulty, so every listener hears it.

**Edge:** *"could we get [album] added back?"* -> content_issue, not
feature_request. Catalogue content, not a product capability.

### 4. account_access

Cannot get into an account, or cannot manage who is on it and what it is called.

- cannot log in by any method
- account hacked or compromised
- family plan invite not working, member will not add
- username or display-name changes
- Facebook linking problems, email tied to a dead account
- deactivating or deleting an account
- country/region on the account is wrong

Examples: *"my premium account has been hacked, email has changed"*, *"I just
want to add family to my account, it's not letting me"*, *"can you help me change
my username?"*, *"my account was created in the USA, I moved to Thailand and
can't log back in"*.

**Not this:** logged in fine but charged wrongly -> `billing_dispute`. Asking
what plans exist -> `subscription_query`.

**Edge:** hacked account -> account_access even if money was taken. Getting back
in comes first.

### 5. billing_dispute

Money has **already been taken** wrongly, or they paid and did not get what they
paid for.

- charged unexpectedly, twice, or six times
- charged the wrong amount
- was on student pricing, now charged full price
- paying for premium but still seeing ads
- premium bought but not activated
- refund requests

Examples: *"i was charged unnecessarily??"*, *"I was on ur student plan but now
I'm being charged $9.99"*, *"you have charged me twice a month for three
months"*, *"I have a premium family acct but My Account says Free Acct so I get
ads"*, *"got charged 5 times trying to activate the 0.99 pack"*.

**Not this:** asking about prices before buying -> `subscription_query`. Premium
works but app is broken -> `app_bug`.

**Edge:** *"paying customer but seeing ads"* -> billing_dispute. They paid and
did not receive the thing, even though it looks like a malfunction.

### 6. subscription_query

Questions and requests about getting, paying for, or keeping the service, where
**no charge is being disputed**.

- what deals, student plans or bundles exist
- how to apply a student discount, or why verification failed
- how to pause, cancel, downgrade, switch plan
- payment methods available in their country
- gift cards and how they work across regions
- presale codes and presale access
- asking for a free month or a promo code
- what happens to their music if they stop paying

Examples: *"would I have to cancel my premium to sign up for the student
discount?"*, *"I don't have a credit card, is there any other payment option?"*,
*"can y'all give my broke ass a free month"*, *"I need my jhene aiko pre sale
code"*, *"I bought a £10 gift card for a USA friend but it only works in the
UK"*.

**Not this:** money already taken wrongly -> `billing_dispute`. Cannot log in to
manage it -> `account_access`.

**Edge — the student-discount split.** This is the most common single topic and
it goes two ways:
- cannot get verified / cannot apply it / is it available here -> subscription_query
- was on it and got charged full price, or charged repeatedly -> billing_dispute

### 7. how_to_question

Nothing is broken. They want to know how something works or whether it exists.

- does feature X do Y
- how do I do Z
- is this behaviour normal

Examples: *"does downloading songs with premium take up storage?"*, *"do friends
get notified when you listen to their music?"*, *"is there a way to check how
many songs I have downloaded?"*, *"how do I post songs from Spotify?"*.

**Not this:** it is broken and they want it fixed -> `app_bug` /
`playback_library`. They want a capability that does not exist ->
`feature_request`.

**Edge:** *"How do I remove a playlist that says 'Loading..'?"* -> how_to_question.
There is a stuck playlist, but the ask is how-to. Label by the ask.

**broken beats phrasing.** A message asking "how do I fix X" where X is
broken is X's intent, not how_to_question. The polite framing is the ask, not
the intent. how_to_question is only for cases where nothing is broken.

*"Just purchased the premium family plan but can't invite family members"* is
account_access, not how_to_question — the invite is broken.
*"Is there a way to stop my devices auto-connecting?"* is how_to_question —
nothing is broken, they want to know if a setting exists.

### 8. feature_request

They want Spotify to build, change or extend something that does not exist yet.

- "I wish you would add..."
- UI and naming suggestions
- capabilities that do not exist (sleep timer, alarm, blacklist artists, filter
  playlists by genre)
- **country availability**: when will you launch in India / Israel / Serbia /
  Nigeria / Kenya / Saudi Arabia
- **device and platform support**: iPhone X optimisation, Apple Watch app, iPod
  Classic, a standalone podcast app
- asking when a long-requested feature will finally ship
- complaints about algorithm or shuffle behaviour working as designed

Examples: *"can you please add a sleep timer"*, *"when are you going to launch in
India?"*, *"when will you update Spotify optimized for iPhone X?"*, *"i wish you
could blacklist songs and artists"*, *"why do i always get the exact same shuffle
every single day"*.

**Not this:** wanting an album added -> `content_issue`. Something that used to
work and now does not -> `app_bug`.

**Edge:** country availability is feature_request, not content_issue. The
catalogue is not wrong; the product does not exist there.

### 9. other

No actionable support request.

- non-English messages (cannot be labelled reliably by me)
- pure abuse with no request: *"fuck you why do you make things so hard"*
- content-free continuations: *"DM sent :)"*
- jokes, memes, off-topic, praise with no ask
- **brand marketing copy appearing as a customer message** (see below)
- messages where the intent genuinely cannot be determined

**Not this:** vague but real problem reports -> label the problem. Angry but with
a real complaint underneath -> label the complaint.

**Rule:** `other` is for *no request*, not for *unclear request*.

---

## Cross-cutting rules

**Multi-intent:** label the intent the customer **leads with**. "I was charged
twice and also the app keeps crashing" -> billing_dispute.

**Anger is not an intent.** Rage with a real complaint underneath gets the
complaint's label. Rage with nothing underneath gets `other`. Anger is captured
by the escalation decision, not by the intent.

**Vagueness is not `other`.** A problem report with no detail is still that
problem.

**Label the opening message only**, not the whole conversation. Later turns are
not available at classification time in production.

**When torn, pick the intent whose reply would be more useful to the customer**,
press the `hard` flag, and note the id. Those notes become failure-analysis
material.

---

## Two sub-counts to keep separately

`other` is doing two jobs, and both are worth reporting on their own before
being merged back for metrics.

**Non-English.** Labelled `other` and counted. The repo's language heuristic
reports 0.29% but has function-word lists for es/pt/fr/de/it only, so it cannot
detect Indonesian, Swedish or Dutch — all three of which appear in the
exemplars. The 0.29% is a floor and is not usable as an estimate. The rate
observed while hand-labelling is the honest number.

**Brand marketing copy posing as a customer message.** These are `inbound=True`
rows whose text is a Spotify advertisement, indistinguishable from a customer
message at the schema level. Observed in the exemplars:

- *"Don't let ads interrupt your music. Join Premium for students, just $5.99."*
- *"Premium gives you unlimited skips. Plus you can play anywhere, even without
  wifi. Try it free."*
- *"From Steve's Morning Hair Grooves to Nancy's Slaylist, there's a playlist for
  everyone. Find yours: [link]"*

Roughly 3 in 160 exemplars, so on the order of 2% of the dataset. This is the
same `@115888` marketing-handle problem found in conversation 2812 at the sample
stage, and it turns out not to be rare. Label `other`, count separately, and
report the rate — it is a quantified data-quality finding, not just an
annoyance.

---

## Expected distribution

Rough estimate from the 160 exemplars, recorded before labelling so it can be
compared against the actual golden set afterwards:

| intent | est. share |
|---|---|
| feature_request | ~20% |
| app_bug | ~15% |
| account_access | ~15% |
| billing_dispute | ~12% |
| subscription_query | ~12% |
| content_issue | ~12% |
| playback_library | ~8% |
| other | ~8% |
| how_to_question | ~5% |

No class swallows the data and none sits below about 5%, which is what macro-F1
needs to be meaningful.

---

## The escalate vs auto-handle decision

Every labelled example gets one of these alongside its intent.

**ESCALATE to a human when:**

- The answer needs account-specific data Spotify cannot see from a public
  tweet: actual charges, subscription state, student verification status,
  who is on a family plan, whether a refund was issued.
- Money is genuinely in dispute. A wrong charge needs someone with access to
  the billing system, not a template.
- The account is compromised. Hacked accounts are urgent and need identity
  verification.
- The customer is abusive, or is explicitly cancelling or demanding a refund.
  A passing mention of a competitor ("guess I'll switch to Apple Music") is
  not enough on its own — it appears constantly in ordinary feature requests
  and complaints, and escalating all of them would make escalation
  meaningless. Escalate when the customer states they are leaving or wants
  money back, not when they grumble about alternatives.
- There is legal or safety language: chargebacks, fraud accusations, threats
  of legal action, anything involving a minor.

**AUTO-HANDLE when the reply can be written from Spotify's own past public
replies:**

- General how-to questions with a known answer.
- Feature requests, including country availability and device support. The
  honest reply is "we will pass this on", which needs no account access.
- Known bugs with a public workaround: reinstall, log out and back in, check
  your app version.
- Catalogue questions where the answer is public: this album is not available
  in your region, that playlist was retired.
- Anything where the historical reply Spotify actually sent was itself
  generic.

**Tiebreak:** ask whether a support agent with no system access could write a
useful reply from public information alone. If yes, auto. If they would have
to look something up in an internal tool, escalate.

**Note:** escalation is about whether a public reply can resolve it, not about
how upset the customer is. Anger raises escalation only when it needs a
retention or tone judgement, which is why "threatening to cancel" is on the
escalate list but ordinary frustration is not.