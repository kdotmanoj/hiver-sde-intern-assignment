"""One-off survey of the top brands, to pick which ones the pipeline targets.

Not a pipeline stage: nothing downstream reads its output, it is not in the
Makefile, and it writes nothing to data/. It exists so the brand choice in
decisions.md is made from the data rather than from a guess.

Two phases, on purpose:

  --prefixes  print the most common opening 6-word prefixes of brand replies,
              per brand. This is the evidence for what the deflection
              templates actually are.
  --report    classify and print the summary table, using the patterns that
              were agreed after reading the phase-1 output.

Brand attribution: a conversation belongs to brand X if X authored at least
one outbound tweet in it. A conversation can therefore belong to more than one
brand (handoffs, or a customer tagging two companies); --report counts those
rather than silently picking one.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.ingest import REPO_ROOT, read_parquet

INTERIM = REPO_ROOT / "data" / "interim"

TOP_N_BRANDS = 8
PREFIX_WORDS = 6
TOP_N_PREFIXES = 20

# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_brand_replies(interim: Path = INTERIM) -> pd.DataFrame:
    """Every outbound tweet, tagged with the conversation it sits in.

    One row per brand reply. `author_id` is the brand; the conversation is the
    unit the survey aggregates over.
    """
    tweets = read_parquet(interim / "tweets.parquet")
    assignment = read_parquet(interim / "conversation_assignment.parquet")
    print(f"survey: {len(tweets):,} tweets, {len(assignment):,} assignment rows")

    cols = assignment[["tweet_id", "conversation_id"]]
    tagged = tweets.merge(cols, on="tweet_id", how="left", validate="one_to_one")
    assert len(tagged) == len(tweets), (len(tagged), len(tweets))

    assigned = tagged[tagged["conversation_id"].notna()]
    replies = assigned[~assigned["inbound"]]
    print(
        f"survey: tweets {len(tweets):,} -> assigned {len(assigned):,} "
        f"(lost {len(tweets) - len(assigned):,}) -> brand replies {len(replies):,}"
    )
    return replies


def top_brands(replies: pd.DataFrame, n: int = TOP_N_BRANDS) -> list[str]:
    """The n brands appearing as an outbound author in the most conversations."""
    per_brand = replies.groupby("author_id", dropna=False)["conversation_id"].nunique()
    ranked = per_brand.sort_values(ascending=False)
    print(f"survey: {len(ranked):,} brands with at least one outbound tweet")
    return list(ranked.head(n).index)


# --------------------------------------------------------------------------
# prefixes (phase 1)
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


def prefix(text: str, words: int = PREFIX_WORDS) -> str:
    return " ".join(normalize(text).split()[:words])


def show_prefixes(replies: pd.DataFrame, brands: list[str]) -> None:
    """Print the most common openers per brand. The input to agreeing patterns."""
    for brand in brands:
        texts = replies.loc[replies["author_id"] == brand, "text"]
        counts = texts.map(prefix).value_counts()
        total = len(texts)
        print(f"\n### {brand} — {total:,} replies, {len(counts):,} distinct prefixes")
        for opener, count in counts.head(TOP_N_PREFIXES).items():
            print(f"  {count:6,}  {100 * count / total:5.2f}%  {opener}")


# --------------------------------------------------------------------------
# deflection (phase 2)
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


# --------------------------------------------------------------------------
# multi-part continuations
# --------------------------------------------------------------------------

# "2/2 <url>", "(1/3)", "1:", "1." -- a part marker at either end of the reply.
MULTIPART = re.compile(
    r"^\s*\(?\d{1,2}\s*(?:/|of)\s*\d{1,2}\)?[\s:.\-]"
    r"|[\s(]\(?\d{1,2}\s*/\s*\d{1,2}\)?\s*$"
    r"|^\s*\(?\d{1,2}\)?\s*[:.]\s"
)


# --------------------------------------------------------------------------
# language (heuristic -- see the footnote printed under the table)
# --------------------------------------------------------------------------

import regex  # noqa: E402  -- only this section needs \p{Latin}

LATIN = regex.compile(r"\p{Script=Latin}")
LETTER = regex.compile(r"\p{L}")

# Two or more of a language's function words, and fewer than two English ones.
FUNCTION_WORDS = {
    "es": {"que", "para", "por", "los", "las", "con", "una", "tu", "te", "cuenta", "gracias", "puedes", "nos"},
    "pt": {"que", "para", "por", "uma", "voce", "você", "nos", "obrigado", "sua", "seu", "com", "mais"},
    "fr": {"que", "pour", "vous", "nous", "votre", "avec", "merci", "une", "est", "sur", "bonjour"},
    "de": {"und", "sie", "wir", "ist", "nicht", "das", "den", "dich", "bitte", "gerne", "danke"},
    "it": {"che", "per", "una", "non", "con", "grazie", "tuo", "ciao", "sono", "puoi"},
}
ENGLISH_WORDS = {"the", "you", "we", "to", "and", "for", "is", "your", "this", "that", "with", "please", "can", "our"}


def is_non_english(normalized: str) -> bool:
    """Approximate. Non-Latin script, or Latin script with foreign function words."""
    letters = LETTER.findall(normalized)
    if len(letters) >= 10:
        latin = sum(1 for c in letters if LATIN.match(c))
        if latin / len(letters) < 0.8:
            return True
    words = set(normalized.split())
    if len(words & ENGLISH_WORDS) >= 2:
        return False
    return any(len(words & fw) >= 2 for fw in FUNCTION_WORDS.values())


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def brands_per_conversation(replies: pd.DataFrame) -> pd.Series:
    """How many distinct brands authored an outbound tweet in each conversation."""
    return replies.groupby("conversation_id", dropna=False)["author_id"].nunique()


def mentioned_conversations(interim: Path, brands: list[str]) -> dict[str, set]:
    """Conversations where a customer addressed @brand.

    The denominator for "did the brand reply at all". It cannot be the
    attributed conversations: under the attribution rule every one of those has
    a brand reply by definition, so the rate would be 100% for all 8 brands.
    """
    tweets = read_parquet(interim / "tweets.parquet")
    assignment = read_parquet(interim / "conversation_assignment.parquet")
    tagged = tweets.merge(
        assignment[["tweet_id", "conversation_id"]],
        on="tweet_id",
        how="left",
        validate="one_to_one",
    )
    inbound = tagged[tagged["inbound"] & tagged["conversation_id"].notna()]
    print(f"survey: {len(inbound):,} assigned inbound tweets scanned for @mentions")

    out = {}
    for brand in brands:
        hit = inbound["text"].str.contains(f"@{brand}", case=False, regex=False, na=False)
        out[brand] = set(inbound.loc[hit, "conversation_id"])
    return out


def build_report(interim: Path = INTERIM) -> None:
    replies = load_brand_replies(interim)
    brands = top_brands(replies)

    per_conv = brands_per_conversation(replies)
    multi = int((per_conv > 1).sum())
    print(
        f"survey: {len(per_conv):,} conversations with a brand reply, "
        f"{multi:,} ({100 * multi / len(per_conv):.2f}%) have more than one brand"
    )

    conversations = read_parquet(interim / "conversations.parquet")
    n_tweets = conversations.set_index("conversation_id")["n_tweets"]
    mentioned = mentioned_conversations(interim, brands)

    top = replies[replies["author_id"].isin(brands)].copy()
    top["normalized"] = top["text"].map(normalize)
    # Computed once and shared: deflection is the redirect mask narrowed by the
    # residual threshold, and both halves are reported separately.
    top["redirect"] = has_redirect(top["normalized"])
    top["residual_words"] = residual_word_count(top["normalized"])
    top["deflection"] = top["redirect"] & (top["residual_words"] <= MAX_RESIDUAL_WORDS)
    top["multipart"] = top["normalized"].str.contains(MULTIPART, regex=True, na=False)
    top["non_english"] = top["normalized"].map(is_non_english)

    rows = []
    for brand in brands:
        sub = top[top["author_id"] == brand]
        convs = set(sub["conversation_id"])
        addressed = mentioned[brand]
        replied = len(addressed & convs)
        rows.append(
            {
                "brand": brand,
                "conversations": len(convs),
                "median tweets/conv": n_tweets.reindex(list(convs)).median(),
                "% addressed w/ reply": 100 * replied / len(addressed) if addressed else float("nan"),
                "redirect-present %": 100 * sub["redirect"].mean(),
                "deflection rate %": 100 * sub["deflection"].mean(),
                "median residual words": sub["residual_words"].median(),
                "non-English %": 100 * sub["non_english"].mean(),
                "multi-part %": 100 * sub["multipart"].mean(),
            }
        )
    # Hand-rolled markdown: to_markdown needs tabulate, and a one-off script
    # is not worth a new dependency.
    headers = list(rows[0])
    print()
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        cells = [
            str(row[h]) if isinstance(row[h], (str, int)) else f"{row[h]:,.2f}"
            for h in headers
        ]
        print("| " + " | ".join(cells) + " |")
    print(
        "\nBrands attributed by authoring >=1 outbound tweet in the conversation;"
        f" {multi:,} conversations have more than one brand and are counted under each."
        "\n'% addressed w/ reply' is NOT a service level and should not be read"
        " as one. Over the attributed conversations it is 100% by construction,"
        " so it is computed over conversations where a customer tweeted @brand"
        " -- but every such inbound tweet in this corpus is already assigned to"
        " a conversation, i.e. twcs.csv only contains threads that got engaged."
        " The unanswered population is not in the data, so the residual few"
        " percent below 100 is just conversations answered by a different brand."
        "\n'redirect-present %' is every reply matching a channel-switch pattern"
        " regardless of residual length: the upper bound. 'deflection rate %' is"
        " that narrowed to <=3 content words left after boilerplate is stripped:"
        " the lower bound. The real rate is between them; neither the headline"
        " nor the comparison between brands should rest on the threshold alone."
        "\nHelp-article links do not count as redirects, by decision."
        "\nTwo known leaks, both biasing deflection DOWNWARD, accepted not fixed:"
        " (a) 'let's work together in dm here: <url>' survives at 4 residual"
        " words, just over the threshold; (b) form-fill redirects ('please fill"
        " in this form: <url>') are not in the pattern set at all."
        "\n'median residual words' is over ALL of a brand's replies, not just"
        " redirects -- how much substance a typical reply carries, which is what"
        " determines whether its replies are groundable."
        "\nnon-English is a script + function-word heuristic, not a language"
        " model. It undercounts Spanish/Portuguese/French. Treat as a floor."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefixes", action="store_true", help="phase 1: show openers")
    parser.add_argument("--report", action="store_true", help="phase 2: summary table")
    parser.add_argument("--residuals", action="store_true", help="debug the purity rule")
    args = parser.parse_args()

    if args.report:
        build_report()
        return

    replies = load_brand_replies()
    brands = top_brands(replies)
    print(f"survey: top {len(brands)} brands by conversation count: {', '.join(brands)}")

    if args.prefixes:
        show_prefixes(replies, brands)
    if args.residuals:
        sample = replies[replies["author_id"].isin(brands)].sample(400, random_state=42)
        normalized = sample["text"].map(normalize)
        flagged = is_deflection(normalized)
        for text in normalized[flagged].head(15):
            print(f"  DEFLECT  {text}\n           -> residual: {residual(text)!r}")
        for text in normalized[~flagged].head(15):
            print(f"  KEEP     {text}\n           -> residual: {residual(text)!r}")


if __name__ == "__main__":
    main()
