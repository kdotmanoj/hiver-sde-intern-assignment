"""Should a human take this ticket, or can the agent answer it?

One of the three modules CLAUDE.md requires to be plain, explicit code: no
pandas, no sklearn, no clever comprehensions standing in for a rule. Everything
here is a named constant and an `if`, because this is a policy I have to be able
to derive on a whiteboard and defend line by line.

The rule is a flat OR over named signals drawn from three independent families:

  intent      the intent class itself needs account data we cannot see
  similarity  nothing in the retrieval corpus resembles this message
  text        something in the wording demands a human

No weights, no score, no threshold-tuned combination. If any signal fires, the
ticket escalates. That is deliberately the dumbest rule that could work, so that
the evaluation stage measures the SIGNALS rather than measuring a fitted
combiner.

Per-family attribution (see signals_by_family) exists because the combined rule
alone cannot answer the question that matters: do the similarity and text
families add anything over the intent lookup? billing_dispute and
account_access are ~27% of the golden set and nearly all escalate, so intent
alone is expected to carry most of the decision. Expected, then measured.

Source of truth for the policy is data/golden/labelling_guide.md, section "The
escalate vs auto-handle decision". Every branch below cites the clause it
encodes. Nothing here is invented.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# the signal names
# --------------------------------------------------------------------------
#
# Every signal any branch below can emit appears exactly once here, mapped to
# its family. The evaluation stage scores each family alone against the golden
# escalate labels, which it can only do if this mapping is complete -- so
# tests/test_escalate.py asserts that it is. Adding a signal without adding it
# here fails the suite rather than silently vanishing from the report.

INTENT_NEEDS_ACCOUNT_DATA = "intent_needs_account_data"
NO_SIMILAR_PRECEDENT = "no_similar_precedent"
ABUSE_OR_PROFANITY = "abuse_or_profanity"
CANCELLATION_OR_REFUND = "cancellation_or_refund"
LEGAL_OR_SAFETY = "legal_or_safety"
PII_PRESENT = "pii_present"

SIGNAL_FAMILY: dict[str, str] = {
    INTENT_NEEDS_ACCOUNT_DATA: "intent",
    NO_SIMILAR_PRECEDENT: "similarity",
    ABUSE_OR_PROFANITY: "text",
    CANCELLATION_OR_REFUND: "text",
    LEGAL_OR_SAFETY: "text",
    PII_PRESENT: "text",
}

FAMILIES = ("intent", "similarity", "text")


# --------------------------------------------------------------------------
# family 1: the intent class
# --------------------------------------------------------------------------
#
# The guide's tiebreak: "ask whether a support agent with no system access could
# write a useful reply from public information alone. If yes, auto. If they
# would have to look something up in an internal tool, escalate."
#
# Exactly two of the nine intents fail that test by definition:
#
#   billing_dispute   money has already been taken wrongly. Answering needs the
#                     actual charges, which are not in a public tweet.
#   account_access    cannot get in, or the account is compromised. Needs
#                     identity verification and account state.
#
# The other seven can be answered from public information at least sometimes,
# so they contribute nothing here and rely on the text and similarity families.
# Note this is a per-CLASS rule, not a per-message one: it escalates every
# billing_dispute, including the rare one that a template could have handled.
# That is the cost of keeping the rule this simple, and the per-family metrics
# are what will show whether the cost is worth paying.

INTENTS_NEEDING_ACCOUNT_DATA = frozenset({"billing_dispute", "account_access"})


# --------------------------------------------------------------------------
# family 2: retrieval similarity
# --------------------------------------------------------------------------
#
# If the nearest thing in the corpus is this far away, we have no past reply to
# ground a draft in, and generating one anyway is how a support bot invents
# policy. Hand it to a human instead.

# Set from the distribution, not from taste. `make diagnose` measured top-1
# similarity for all 150 golden openings against the 1,850-row corpus:
#
#   p0=0.405  p10=0.588  p25=0.667  p50=0.736  p75=0.792  p90=0.833  p100=0.967
#
# and the escalation rate each candidate floor would produce:
#
#   0.50 ->   6/150  ( 4.0%)
#   0.60 ->  17/150  (11.3%)   <- chosen
#   0.70 ->  53/150  (35.3%)
#   0.80 -> 119/150  (79.3%)
#
# 0.60 sits just above p10, so it fires on roughly the bottom decile of
# retrieval quality -- the cases where the nearest precedent genuinely is not
# close. 0.70 escalates a third of everything and would make this signal, not
# the intent lookup, the dominant term in the rule; 0.50 fires so rarely it
# could not be measured against 150 examples.
SIMILARITY_FLOOR = 0.60


# --------------------------------------------------------------------------
# family 3: the text itself
# --------------------------------------------------------------------------
#
# Each lexicon below is a flat tuple of lowercase phrases, matched against the
# normalized message. Phrases, not stems: "refund" alone appears in plenty of
# ordinary questions ("how do refunds work?"), so the entries are written long
# enough to mean what they say.

# The guide escalates on "the customer is abusive". This is a lexicon, not a
# sentiment model, and it is a blunt one.
#
# KNOWN FALSE POSITIVE, worth reporting rather than hiding: golden id 93962
# ("Screw @115888 for the massive mobile app update ... I can't afford that
# shit") is hand-labelled escalate=false. It carries profanity but the profanity
# is not aimed at anyone -- it is an ordinary frustrated feature_request. This
# lexicon fires on it and is wrong. The guide anticipates exactly this: "anger
# raises escalation only when it needs a retention or tone judgement ... ordinary
# frustration is not". Distinguishing aimed abuse from ambient swearing needs
# more than a word list. Left blunt on purpose so the per-family metrics can
# quantify the damage instead of an untested heuristic hiding it.
# Counts are hits over all 28,326 openings. Following the standard set by
# src/deflection.py -- "every pattern was counted over the full text before
# being accepted" -- phrases that matched nothing were removed rather than left
# in as decoration. 43 openings (0.15%) fire this signal.
ABUSE_PHRASES = (
    "fuck you",  # 23
    "idiots",  # 7
    "fuck off",  # 6
    "you suck",  # 4
    "morons",  # 3
)

# The guide: "the customer ... is explicitly cancelling or demanding a refund".
#
# On competitor mentions, the guide is unusually explicit and unusually strict:
# "A passing mention of a competitor ('guess I'll switch to Apple Music') is not
# enough on its own -- it appears constantly in ordinary feature requests and
# complaints, and escalating all of them would make escalation meaningless."
#
# So there is deliberately NO competitor lexicon in this module. Not a weakened
# one, not one gated behind a verb -- none at all. A customer who is actually
# leaving says so in words that appear below, and one who is merely grumbling
# about alternatives says only the competitor's name. Adding "apple music" here
# in any form would re-introduce the exact false-positive the guide forbids.
# 284 openings (1.00%). Every phrase below fires at least once; counts follow.
CANCELLATION_PHRASES = (
    # explicitly cancelling
    "cancel my subscription",  # 61
    "cancel my premium",  # 53
    "cancel my account",  # 33
    "delete my account",  # 17
    "cancelling my subscription",  # 8
    "cancelling my premium",  # 8
    "canceling my subscription",  # 8
    "unsubscribing",  # 7
    "cancel my membership",  # 6
    "deleting my account",  # 4
    "canceling my premium",  # 3
    "done with spotify",  # 2
    "closing my account",  # 1
    # demanding money back
    "my money back",  # 51
    "want a refund",  # 11
    "refund me",  # 6
    "refund my money",  # 4
    "want my refund",  # 2
    "demand a refund",  # 1
    "give me back my money",  # 1
)

# The guide: "legal or safety language: chargebacks, fraud accusations, threats
# of legal action, anything involving a minor". Escalated regardless of how calm
# the message is -- a politely-worded chargeback notice is still one.
#
# REMOVED AFTER COUNTING: "my son", "my daughter", "my child", "my kid". Those
# four alone produced 89 of the signal's original 101 hits, and on this corpus
# essentially all of them are family-plan mechanics -- "when I try to add my son
# to my Spotify family account it says whoops". That is account_access, not
# safeguarding. The guide's "anything involving a minor" clause means a
# safeguarding concern, and a family-relation noun is not evidence of one. Left
# in, this signal would have escalated ~0.3% of the corpus for having a family.
# The explicit age and status terms below are kept and carry the clause.
#
# 12 openings (0.04%) fire this signal after the removal.
LEGAL_PHRASES = (
    "underage",  # 2
    "12 year old",  # 2
    "chargeback",  # 1
    "this is fraud",  # 1
    "fraudulent charge",  # 1
    "identity theft",  # 1
    "my lawyer",  # 1
    "sue you",  # 1
    "13 year old",  # 1
    "a minor",  # 1
    # Spelling and tense variants of phrases observed above. Zero hits here, and
    # kept anyway because they are the same pattern rather than new guesses:
    # "charge back" is "chargeback" spaced, "suing you" is "sue you" inflected.
    # No "charge-back" entry: normalize() turns every hyphen into a space, so
    # the hyphenated spelling already arrives here as "charge back".
    "charge back",
    "suing you",
)

# Personal data that must not sit in a public tweet a bot is about to reply to.
# Present-tense risk, not policy: if the customer has already posted a card
# number or an account id, a human needs to see that thread.
#
# Written as regexes because "is this a phone number" is a shape question, not a
# vocabulary one. Each is deliberately conservative -- tweets are full of stray
# digits (prices, app version numbers, years, error codes) and a loose pattern
# fires on all of them. Every threshold below was set by running the pattern
# over all 28,326 openings and reading the matches, not by guessing.

# The source corpus arrives with email addresses already replaced by the literal
# token "__email__" -- 130 openings, 0.46%. So the address-shaped regex below
# matches nothing on this dataset and the token is what actually carries the
# signal. Both are checked: the token because it is what this corpus contains,
# the regex because nothing guarantees the next corpus is redacted the same way.
# A redacted email is still evidence the customer posted one.
REDACTED_EMAIL = "__email__"
EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}")

# A run of 13-16 digits, optionally split by spaces or hyphens into groups.
# That is the length range of every major card scheme. The guards stop it
# matching the middle of a longer digit run, e.g. a 20-digit error code.
# Fires on 1 of 28,326 openings, and that one is a joke ("2922922883928 fois").
# A 0.004% false-positive rate on a signal this consequential is worth keeping.
CARD_NUMBER = re.compile(r"(?<![\d.-])(?:\d[ -]?){13,16}(?![\d.-])")

# A long identifier run: a phone number, but in practice mostly Spotify account
# numbers and support ticket references, which are the same kind of thing for
# this signal's purpose -- an identifier a public reply must not quote back.
#
# Two deliberate restrictions, both from reading the corpus matches:
#   1. Dots are NOT separators. Prices ("$14.99") and app versions ("1.0.69.323")
#      use dots; phone numbers use spaces, hyphens and parens. Allowing dots
#      took this from 22 matches to 228, essentially all of them prices.
#   2. At least 10 digits total, counted after matching rather than encoded in
#      the pattern, because expressing "10+ digits with arbitrary separators"
#      as a regex is write-only.
# Result: 22 of 28,326 openings (0.08%). Two are false positives -- Spotify's
# own support number quoted back at them, and a song title ("1-800-273-8255").
PHONE_MIN_DIGITS = 10
PHONE_CANDIDATE = re.compile(r"(?<![\d.-])\+?\d[\d ()\-]{7,}\d(?![\d.-])")


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------
#
# Deliberately NOT src.deflection.normalize. That one masks URLs and strips
# leading @mentions, which is right for template detection and wrong here: an
# @mention is part of the message a human would read, and this module never
# looks at URLs. Keeping its own two-line version means a change to deflection's
# rules cannot silently move the escalation boundary.

# Everything that is not a letter, a digit, @ or + becomes a space. Punctuation
# has to go so that "cancel my subscription!!!" and "refund me." match, but the
# digit separators the PII regexes rely on must survive -- which is why PII is
# matched against the RAW text and only the lexicons use this.
NON_WORD = re.compile(r"[^a-z0-9@+]+")


def normalize(text: str) -> str:
    """Lowercase and collapse punctuation to single spaces, padded with spaces.

    The padding means a phrase test can be a plain `in` against a string that
    always has a space on both sides, so " idiots " cannot match "idiotstagram"
    and no word-boundary regex is needed to say so.
    """
    lowered = str(text).lower()
    collapsed = NON_WORD.sub(" ", lowered)
    return " " + collapsed.strip() + " "


def _contains_any(padded: str, phrases: tuple[str, ...]) -> bool:
    """True if any phrase appears in the normalized text as whole words.

    Written as an explicit loop rather than any(...) so that a debugger stops on
    the phrase that matched.
    """
    for phrase in phrases:
        if " " + phrase + " " in padded:
            return True
    return False


def _has_pii(text: str) -> bool:
    """True if the RAW text contains an email, card number, or long identifier.

    Raw, not normalized: normalization turns "a@b.com" into "a@b com", which the
    EMAIL pattern would then miss, and strips the dots the patterns rely on to
    tell a version number from a phone number. Shape-matching needs the shape.
    """
    lowered = str(text).lower()

    if REDACTED_EMAIL in lowered:
        return True
    if EMAIL.search(lowered):
        return True
    if CARD_NUMBER.search(lowered):
        return True

    # The digit count is the real test; the regex only finds candidates. See
    # PHONE_MIN_DIGITS above for why this is not folded into the pattern.
    for match in PHONE_CANDIDATE.finditer(lowered):
        n_digits = sum(1 for char in match.group(0) if char.isdigit())
        if n_digits >= PHONE_MIN_DIGITS:
            return True
    return False


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------


def escalate(intent: str, top_similarity: float, text: str) -> dict:
    """Decide whether this message needs a human.

    Args:
        intent: one of the nine taxonomy labels, as classified.
        top_similarity: cosine similarity of the single nearest corpus opening.
        text: the customer's opening message, raw.

    Returns:
        {"decision": "escalate" | "auto",
         "reason": str,
         "triggered_signals": [str, ...]}

    Signals are appended in family order -- intent, then similarity, then text --
    so `triggered_signals[0]` is the highest-precedence reason and `reason` can
    name it without a second pass.
    """
    triggered: list[str] = []
    padded = normalize(text)

    # --- family 1: intent ---
    if intent in INTENTS_NEEDING_ACCOUNT_DATA:
        triggered.append(INTENT_NEEDS_ACCOUNT_DATA)

    # --- family 2: similarity ---
    if top_similarity < SIMILARITY_FLOOR:
        triggered.append(NO_SIMILAR_PRECEDENT)

    # --- family 3: text ---
    if _contains_any(padded, ABUSE_PHRASES):
        triggered.append(ABUSE_OR_PROFANITY)

    if _contains_any(padded, CANCELLATION_PHRASES):
        triggered.append(CANCELLATION_OR_REFUND)

    if _contains_any(padded, LEGAL_PHRASES):
        triggered.append(LEGAL_OR_SAFETY)

    if _has_pii(text):
        triggered.append(PII_PRESENT)

    if triggered:
        return {
            "decision": "escalate",
            "reason": REASONS[triggered[0]],
            "triggered_signals": triggered,
        }
    return {
        "decision": "auto",
        "reason": "No escalation signal fired; a public reply can address this.",
        "triggered_signals": [],
    }


# One sentence per signal, used verbatim as `reason` when that signal is the
# highest-precedence one that fired. Kept beside the rule so a new signal
# without a reason is an obvious KeyError rather than a blank field.
REASONS: dict[str, str] = {
    INTENT_NEEDS_ACCOUNT_DATA: (
        "Answering needs account-specific data that is not visible from a public tweet."
    ),
    NO_SIMILAR_PRECEDENT: (
        "No similar past conversation was retrieved, so there is no grounded reply to draft from."
    ),
    ABUSE_OR_PROFANITY: "The message is abusive and needs a human tone judgement.",
    CANCELLATION_OR_REFUND: (
        "The customer is cancelling or demanding money back, which is a retention decision."
    ),
    LEGAL_OR_SAFETY: "The message contains legal or safety language.",
    PII_PRESENT: "The message contains personal data that must not be handled by a public reply.",
}


def signals_by_family(triggered_signals: list[str]) -> dict[str, list[str]]:
    """Split fired signals by family, for per-family scoring in the eval stage.

    Always returns all three families, empty list when one did not fire, so the
    caller can count true negatives per family without special-casing a missing
    key.

    Because `escalate` is a flat OR, "would family F have escalated on its own"
    is exactly "did any signal of F fire" -- so this is a complete derivation of
    the per-family decisions, not a second implementation that could drift out
    of step with the rule above.
    """
    out: dict[str, list[str]] = {family: [] for family in FAMILIES}
    for signal in triggered_signals:
        if signal not in SIGNAL_FAMILY:
            raise KeyError(
                f"signal {signal!r} has no family in SIGNAL_FAMILY. Add it there, "
                f"or the evaluation stage will silently under-report its family."
            )
        out[SIGNAL_FAMILY[signal]].append(signal)
    return out
