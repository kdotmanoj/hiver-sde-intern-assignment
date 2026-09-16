"""Tests for the escalation rule.

Two claims these defend:

1. Each signal fires on what it is for and NOT on the near-miss the labelling
   guide explicitly warns about. Most of the value in this module is in what it
   declines to escalate, so most of the tests below are negative.
2. Per-family attribution is complete. The evaluation stage scores each family
   alone, which it can only do if every signal has a family and every family is
   always reported.

SIMILARITY_FLOOR ships as a loud placeholder (1.01, escalates everything by
design). Every test here pins it explicitly rather than inheriting it, so these
stay green when the real value is chosen and so no test accidentally passes for
the wrong reason.
"""

import pytest

from src import escalate as esc

# Comfortably above any floor these tests set, so "similarity did not fire" is
# never the accidental cause of a result.
HIGH = 0.95
FLOOR = 0.60


@pytest.fixture(autouse=True)
def pinned_floor(monkeypatch):
    monkeypatch.setattr(esc, "SIMILARITY_FLOOR", FLOOR)


def decide(text, intent="how_to_question", similarity=HIGH):
    """escalate() with benign defaults, so each test varies one thing."""
    return esc.escalate(intent=intent, top_similarity=similarity, text=text)


# --------------------------------------------------------------------------
# the baseline: nothing fires
# --------------------------------------------------------------------------


def test_clean_auto_handleable_message_does_not_escalate():
    result = decide("does downloading songs with premium take up storage?")

    assert result["decision"] == "auto"
    assert result["triggered_signals"] == []
    assert "No escalation signal" in result["reason"]


# --------------------------------------------------------------------------
# family 1: intent
# --------------------------------------------------------------------------


@pytest.mark.parametrize("intent", ["billing_dispute", "account_access"])
def test_intents_needing_account_data_escalate_on_intent_alone(intent):
    """High similarity, clean text: the intent class is the only thing firing."""
    result = decide("hi, could you take a look at this for me please", intent=intent)

    assert result["decision"] == "escalate"
    assert result["triggered_signals"] == [esc.INTENT_NEEDS_ACCOUNT_DATA]
    assert "account-specific data" in result["reason"]


@pytest.mark.parametrize(
    "intent",
    [
        "playback_library",
        "app_bug",
        "content_issue",
        "subscription_query",
        "how_to_question",
        "feature_request",
        "other",
    ],
)
def test_the_other_seven_intents_contribute_no_signal(intent):
    result = decide("hi, could you take a look at this for me please", intent=intent)

    assert result["decision"] == "auto"


# --------------------------------------------------------------------------
# family 2: similarity
# --------------------------------------------------------------------------


def test_similarity_below_the_floor_escalates():
    result = decide("something nobody has ever asked before", similarity=FLOOR - 0.01)

    assert result["decision"] == "escalate"
    assert result["triggered_signals"] == [esc.NO_SIMILAR_PRECEDENT]


def test_similarity_exactly_at_the_floor_does_not_escalate():
    """The comparison is strict `<`. Pinned so the boundary cannot drift silently."""
    result = decide("a perfectly ordinary question", similarity=FLOOR)

    assert result["decision"] == "auto"


# --------------------------------------------------------------------------
# family 3, signal: abuse
# --------------------------------------------------------------------------


def test_abuse_escalates():
    result = decide("fuck you spotify, your app is garbage")

    assert result["triggered_signals"] == [esc.ABUSE_OR_PROFANITY]


def test_ordinary_frustration_without_abuse_does_not_escalate():
    """The guide: 'ordinary frustration is not' grounds for escalation."""
    result = decide("this is so annoying, the app has been broken for a week now")

    assert result["decision"] == "auto"


def test_abuse_lexicon_matches_whole_words_only():
    """'idiots' must not fire inside a longer word."""
    result = decide("i follow the band Idiotstagram on spotify")

    assert result["decision"] == "auto"


# --------------------------------------------------------------------------
# family 3, signal: cancellation / refund
# --------------------------------------------------------------------------


def test_explicit_cancellation_escalates():
    result = decide("that's it, i want to cancel my subscription today")

    assert result["triggered_signals"] == [esc.CANCELLATION_OR_REFUND]


def test_explicit_refund_demand_escalates():
    result = decide("i paid for premium and got nothing, i want a refund")

    assert result["triggered_signals"] == [esc.CANCELLATION_OR_REFUND]


def test_passing_competitor_mention_does_not_escalate():
    """The guide's named near-miss, verbatim.

    'A passing mention of a competitor ("guess I'll switch to Apple Music") is
    not enough on its own -- it appears constantly in ordinary feature requests
    and complaints, and escalating all of them would make escalation
    meaningless.'

    This is the single most important negative test in the module: it is why
    there is no competitor lexicon in src/escalate.py at all.
    """
    result = decide(
        "still no sleep timer after all these years, guess i'll switch to apple music",
        intent="feature_request",
    )

    assert result["decision"] == "auto"
    assert result["triggered_signals"] == []


def test_competitor_mention_alongside_real_cancellation_still_escalates():
    """The competitor is irrelevant either way; the cancellation is what fires."""
    result = decide("cancel my premium please, i'm moving to apple music")

    assert result["triggered_signals"] == [esc.CANCELLATION_OR_REFUND]


def test_asking_how_refunds_work_does_not_escalate():
    """'refund' as a topic is not 'refund' as a demand."""
    result = decide("how do refunds work if i cancel halfway through the month?")

    assert result["decision"] == "auto"


# --------------------------------------------------------------------------
# family 3, signal: legal / safety
# --------------------------------------------------------------------------


def test_legal_threat_escalates():
    result = decide("i've spoken to my lawyer about this")

    assert result["triggered_signals"] == [esc.LEGAL_OR_SAFETY]


def test_chargeback_escalates_however_calmly_it_is_put():
    result = decide("i will be raising a chargeback with my bank, thanks")

    assert result["triggered_signals"] == [esc.LEGAL_OR_SAFETY]


def test_hyphenated_chargeback_is_caught_by_normalization():
    """normalize() turns the hyphen into a space, so 'charge back' covers it."""
    result = decide("requesting a charge-back through my bank")

    assert result["triggered_signals"] == [esc.LEGAL_OR_SAFETY]


def test_family_plan_mention_of_a_child_does_not_escalate():
    """The counted false positive that got 'my son'/'my daughter' removed.

    Those four phrases produced 89 of the signal's original 101 hits across the
    corpus, and on this dataset they are family-plan mechanics, not safeguarding.
    """
    result = decide(
        "when i try to add my son to my spotify family account it says whoops something went wrong",
        intent="account_access",
        similarity=HIGH,
    )

    assert esc.LEGAL_OR_SAFETY not in result["triggered_signals"]


def test_explicit_age_still_escalates():
    """Removing the family-relation nouns must not disarm the clause entirely."""
    result = decide("my account says i'm underage and won't let me listen")

    assert esc.LEGAL_OR_SAFETY in result["triggered_signals"]


# --------------------------------------------------------------------------
# family 3, signal: PII
# --------------------------------------------------------------------------


def test_card_number_escalates():
    result = decide("you charged card 4111 1111 1111 1111 twice", intent="app_bug")

    assert esc.PII_PRESENT in result["triggered_signals"]


def test_phone_number_escalates():
    result = decide("call me back on 917-565-3894 please", intent="app_bug")

    assert esc.PII_PRESENT in result["triggered_signals"]


def test_redacted_email_token_escalates():
    """This corpus ships emails pre-redacted to '__email__' (130 openings).

    The token is evidence the customer posted an address, so it counts.
    """
    result = decide("cannot log in as __email__ any more", intent="app_bug")

    assert esc.PII_PRESENT in result["triggered_signals"]


def test_raw_email_escalates_even_though_this_corpus_has_none():
    """The regex is for the next corpus, which may not be redacted."""
    result = decide("my account is under jane.doe@example.com", intent="app_bug")

    assert esc.PII_PRESENT in result["triggered_signals"]


def test_price_is_not_pii():
    """The counted false positive that removed '.' as a phone separator."""
    result = decide("why have you charged me £14.99 on the 18th of november", intent="app_bug")

    assert esc.PII_PRESENT not in result["triggered_signals"]


def test_app_version_number_is_not_pii():
    result = decide("your update 1.0.69.323 is crashing during install", intent="app_bug")

    assert esc.PII_PRESENT not in result["triggered_signals"]


def test_short_digit_run_is_not_pii():
    """Under PHONE_MIN_DIGITS, so an error code or a year does not fire."""
    result = decide("i keep getting error 404 since 2017", intent="app_bug")

    assert esc.PII_PRESENT not in result["triggered_signals"]


# --------------------------------------------------------------------------
# combination
# --------------------------------------------------------------------------


def test_every_firing_signal_is_reported_not_just_the_first():
    result = esc.escalate(
        intent="billing_dispute",
        top_similarity=FLOOR - 0.1,
        text="fuck you, refund me or i'm calling my lawyer",
    )

    assert result["decision"] == "escalate"
    assert set(result["triggered_signals"]) == {
        esc.INTENT_NEEDS_ACCOUNT_DATA,
        esc.NO_SIMILAR_PRECEDENT,
        esc.ABUSE_OR_PROFANITY,
        esc.CANCELLATION_OR_REFUND,
        esc.LEGAL_OR_SAFETY,
    }


def test_reason_names_the_highest_precedence_signal():
    """Signals append in family order, so intent outranks text when both fire."""
    result = esc.escalate(
        intent="billing_dispute",
        top_similarity=HIGH,
        text="refund me now",
    )

    assert result["triggered_signals"][0] == esc.INTENT_NEEDS_ACCOUNT_DATA
    assert result["reason"] == esc.REASONS[esc.INTENT_NEEDS_ACCOUNT_DATA]


# --------------------------------------------------------------------------
# per-family attribution
# --------------------------------------------------------------------------


def test_signal_family_covers_every_signal_the_rule_can_emit():
    """Guards the per-family report against a new signal being added without one.

    Collected from the module's signal constants rather than by re-listing them,
    so adding a constant without a family fails here.
    """
    emittable = {
        value
        for name, value in vars(esc).items()
        if name.isupper() and isinstance(value, str) and value in esc.REASONS
    }

    assert emittable == set(esc.SIGNAL_FAMILY)
    assert set(esc.SIGNAL_FAMILY.values()) == set(esc.FAMILIES)


def test_signals_by_family_always_returns_all_three_families():
    """Empty lists included, so eval can count true negatives without a KeyError."""
    out = esc.signals_by_family([])

    assert out == {"intent": [], "similarity": [], "text": []}


def test_signals_by_family_attributes_each_signal_to_the_right_family():
    result = esc.escalate(
        intent="account_access",
        top_similarity=FLOOR - 0.1,
        text="my account was hacked, refund me",
    )

    families = esc.signals_by_family(result["triggered_signals"])

    assert families["intent"] == [esc.INTENT_NEEDS_ACCOUNT_DATA]
    assert families["similarity"] == [esc.NO_SIMILAR_PRECEDENT]
    assert families["text"] == [esc.CANCELLATION_OR_REFUND]


def test_signals_by_family_rejects_an_unmapped_signal():
    with pytest.raises(KeyError, match="no family"):
        esc.signals_by_family(["a_signal_that_was_never_registered"])


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------


def test_normalize_pads_so_phrase_matches_are_word_bounded():
    assert esc.normalize("Refund me!!") == " refund me "


def test_normalize_collapses_punctuation_and_case():
    assert esc.normalize("CANCEL   my... subscription?!") == " cancel my subscription "
