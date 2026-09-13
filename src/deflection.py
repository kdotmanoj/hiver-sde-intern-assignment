"""Does a brand reply resolve the problem here, or push it somewhere else?

Moved verbatim out of scripts/brand_survey.py, which is where these patterns
were derived and validated. Nothing about the rule changed in the move; the
survey's output is byte-identical before and after.

It lives in src/ now because src/sample.py needs it to pick each conversation's
first *substantive* brand reply, and a pipeline stage must not import from a
one-off script.

The rule in one sentence: a reply is a pure deflection when it pushes the
customer to another channel AND, once the boilerplate around that push is
stripped, almost nothing is left.

Two measures come out of this module and both are reported, never one alone:

- has_redirect()        any channel-switch phrase, regardless of what else is
                        in the reply. The upper bound.
- is_deflection()       that, narrowed to replies with <= MAX_RESIDUAL_WORDS
                        content words left. The lower bound.

The truth is between them. Reporting only one would make the headline depend
entirely on the threshold.
"""

from __future__ import annotations

import re

import pandas as pd

# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------

# Leading @mentions are routing, not template. "@115712 @sprintcare DM us" and
# "DM us" are the same opener and must collapse to the same prefix.
LEADING_MENTIONS = re.compile(r"^(?:\s*@\w+)+\s*")
URL = re.compile(r"https?://\S+")
WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, drop leading @mentions, mask URLs, collapse whitespace.

    URLs are masked rather than dropped because "use this link <url>" is a
    template whose distinguishing feature IS the link.
    """
    text = LEADING_MENTIONS.sub("", str(text))
    text = URL.sub("<url>", text)
    return WHITESPACE.sub(" ", text).strip().lower()


# --------------------------------------------------------------------------
# deflection
# --------------------------------------------------------------------------
#
# Every pattern below was read off the phase-1 prefix dump and then counted
# over the full reply text before being accepted. None of it is guessed.
#
# The rule, agreed after looking at samples: a reply is a *pure redirect* when
# it pushes the customer to another channel AND, once the boilerplate around
# that push is stripped, almost nothing is left. A redirect phrase on its own
# is not enough -- "which iphone and version of ios are you using? please dm
# us" is a redirect phrase attached to a real diagnostic question.
#
# Channel-switches only. A link to a help article ("here's what you can do to
# work around the issue: <url>") is an answer, not a redirect, and does not
# count -- that template alone is 5.88% of AppleSupport's replies, so the
# choice is worth this much comment.

REDIRECT_PATTERNS = {
    "dm_us": r"\b(?:dm|d\.m\.|pm) (?:us|me)\b"
    r"|\bsend (?:us|me) a (?:dm|d\.m\.|direct message|private message)\b"
    r"|\bdirect message us\b",
    "send_us_details": r"\bsend (?:us|me) (?:your|the|a) \b",
    # AmericanAir's house style: "please dm your record locator". Distinct from
    # dm_us ("dm us"), and missing it undercounts that brand specifically.
    "dm_your_details": r"\b(?:dm|d\.m\.) (?:your|the|me your)\b",
    "follow_and_dm": r"\bfollow (?:and|then) dm\b|\bfollow us (?:and|so)\b",
    "contact_us": r"\b(?:reach out to|contact) (?:us|our|the) \b",
    "call_us": r"\bcall (?:us|our|customer|support)\b|\bgive us a call\b|\b1-8\d\d\b",
    "email_us": r"\b(?:e-?mail (?:us|your)|send (?:us )?an e-?mail)\b",
    "already_dmd": r"\b(?:we|i)(?:'ve| have) (?:just )?(?:sent|dm'?d|replied)\b"
    r"[^.!?]*\b(?:dm|direct message)\b|\bin touch via dm\b",
}

# "feel free to reach out to us again if you change your mind" is a sign-off,
# not a deflection. contact_us is the only pattern loose enough to catch these,
# so it gets an explicit veto list rather than a tighter regex nobody can read.
CONTACT_US_SIGNOFF = re.compile(
    r"\b(?:feel free to|don'?t hesitate to|please do)\s+(?:reach out|contact)"
    r"|\b(?:reach out|contact us)[^.!?]*\bagain\b"
    r"|\bif you (?:need|have)\b[^.!?]*\b(?:reach out|contact us)\b"
)

ANY_REDIRECT = re.compile("|".join(f"(?:{p})" for p in REDIRECT_PATTERNS.values()))

# Stripped in this order to leave only content words. Each group is boilerplate
# that surrounds a redirect without adding information.
BOILERPLATE = [
    # agent signatures: ^mw, -ac, /mc. "^" is unambiguous anywhere; "-" and "/"
    # only at the very end, because they are ordinary punctuation mid-sentence.
    re.compile(r"[\^*][a-z]{1,6}\b"),
    re.compile(r"(?:\s[-/][a-z]{2,6})\s*$"),
    re.compile(r"<url>"),
    # greetings and thanks
    re.compile(
        r"\b(?:hi|hey|hello|good (?:morning|afternoon|evening))\b(?: there)?"
        r"|\bhelp'?s here\b|\bthank(?:s| you)(?: so much)?(?: for [^.!?,]*)?"
    ),
    # apologies
    re.compile(
        r"\boh no\b|\bapologies\b|\bwe apologi[sz]e\b"
        r"|\b(?:we|i)(?:'re| am|'m| are)? ?(?:so |very |really |truly )?sorry\b"
        r"(?: to hear| for| about| that)?"
    ),
    # willingness boilerplate -- says only "we intend to help", never how
    re.compile(
        r"\b(?:we|i)(?:'re| are|'d| would| will|'ll)? ?(?:be )?(?:more than )?"
        r"(?:here to help|happy to (?:help|assist|look)|like to (?:help|assist|look)"
        r"|love to (?:help|assist)|want to help|can definitely|can certainly)\b"
        r"|\bhere to help\b|\blet'?s (?:take a look|look into)\b"
        r"|\btake a (?:closer )?look\b"
    ),
    # purpose clauses trailing the redirect
    re.compile(
        r"\bso (?:that )?(?:we|i|our team) (?:can|may)\b[^.!?]*"
        r"|\band (?:we|our team) will be\b[^.!?]*"
        r"|\band (?:we|our team) will\b[^.!?]*"
        r"|\bso we can\b[^.!?]*"
    ),
]

# Function words left behind by the strippers. They are not content.
FILLER = {
    "a", "about", "an", "and", "any", "are", "as", "at", "be", "but", "by",
    "can", "could", "do", "for", "from", "further", "get", "have", "here",
    "how", "i", "if", "in", "into", "is", "it", "just", "like", "look", "me",
    "more", "not", "of", "on", "once", "or", "our", "out", "over", "please",
    "so", "team", "that", "the", "then", "there", "this", "to", "up", "us",
    "we", "will", "with", "would", "you", "your",
}

MAX_RESIDUAL_WORDS = 3

PUNCTUATION = re.compile(r"[^a-z0-9' ]+")


def residual(text: str) -> str:
    """What is left of a normalized reply after the redirect and its packaging.

    Deliberately plain and sequential: this is the judgement call the whole
    deflection number rests on, so it has to be readable top to bottom.
    """
    for pattern in BOILERPLATE:
        text = pattern.sub(" ", text)
    text = ANY_REDIRECT.sub(" ", text)
    # The redirect regexes end before the object ("send us a dm with your
    # email address"), so strip one more clause of what followed it.
    text = PUNCTUATION.sub(" ", text)
    words = [w for w in text.split() if w not in FILLER]
    return " ".join(words)


def has_redirect(normalized: pd.Series) -> pd.Series:
    """Matches any channel-switch pattern, ignoring how much else is in the reply.

    Reported alongside the pure-deflection rate so the headline number does not
    rest solely on MAX_RESIDUAL_WORDS. This is the upper bound; deflection is
    the lower one, and the truth is somewhere between.
    """
    redirect = normalized.str.contains(ANY_REDIRECT, regex=True, na=False)
    signoff_only = normalized.str.contains(CONTACT_US_SIGNOFF, regex=True, na=False)
    channel = normalized.str.contains(
        "|".join(f"(?:{p})" for k, p in REDIRECT_PATTERNS.items() if k != "contact_us"),
        regex=True,
        na=False,
    )
    # A sign-off "reach out again" only disqualifies when it is the sole hit.
    return redirect & ~(signoff_only & ~channel)


def residual_word_count(normalized: pd.Series) -> pd.Series:
    """Content words left per reply. Over ALL replies, not just redirects --
    it is a measure of how much substance a typical reply carries."""
    return normalized.map(lambda t: len(residual(t).split()))


def is_deflection(normalized: pd.Series) -> pd.Series:
    """Redirects to another channel with (near-)nothing else in the reply."""
    return has_redirect(normalized) & (
        residual_word_count(normalized) <= MAX_RESIDUAL_WORDS
    )
