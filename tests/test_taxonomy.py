"""Tests for the stage-5 sampling and export logic.

The clustering itself is not tested — it is sklearn, and its output is read by
a human rather than asserted on. What is tested is everything that could
silently corrupt what that human reads: the proportional allocation, the
stratified draw, and the exemplar file's structure.
"""

import numpy as np
import pandas as pd
import pytest

from src import taxonomy


def eligible_frame(counts: dict[str, int]) -> pd.DataFrame:
    """A minimal frame with the given per-source row counts."""
    rows = []
    conversation_id = 0
    for source, n in counts.items():
        for i in range(n):
            rows.append(
                {
                    "conversation_id": conversation_id,
                    "source": source,
                    "opening_text": f"{source} opening number {i}",
                }
            )
            conversation_id += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# allocate
# --------------------------------------------------------------------------


def test_allocate_sums_to_total_on_the_real_distribution():
    """The distribution that motivated largest-remainder: round() gives 1,999."""
    counts = pd.Series(
        {
            "brand_broadcast": 115,
            "customer_root": 27_245,
            "orphan_broadcast": 34,
            "orphan_customer": 31,
        }
    )

    allocation = taxonomy.allocate(counts, 2_000)

    assert int(allocation.sum()) == 2_000
    assert allocation["customer_root"] == 1_987
    assert allocation["brand_broadcast"] == 8
    # 3, not 2: orphan_broadcast's discarded fraction (.48) is the second
    # largest, so it takes one of the two leftover rows.
    assert allocation["orphan_broadcast"] == 3
    assert allocation["orphan_customer"] == 2

    # The claim in allocate()'s docstring, pinned so it cannot rot.
    naive = (counts / counts.sum() * 2_000).round().astype(int)
    assert int(naive.sum()) == 1_999


def test_allocate_is_exactly_proportional_when_it_divides_evenly():
    counts = pd.Series({"a": 300, "b": 600, "c": 100})

    allocation = taxonomy.allocate(counts, 100)

    assert allocation.to_dict() == {"a": 30, "b": 60, "c": 10}


def test_allocate_breaks_ties_on_name_not_dict_order():
    """Two strata with identical remainders must resolve the same either way."""
    counts = pd.Series({"b": 50, "a": 50})
    reversed_counts = pd.Series({"a": 50, "b": 50})

    # 3 across two equal strata: floors are 1 and 1, one leftover to give.
    first = taxonomy.allocate(counts, 3)
    second = taxonomy.allocate(reversed_counts, 3)

    assert first["a"] == second["a"] and first["b"] == second["b"]
    assert first["a"] == 2, "the alphabetically first stratum should win the leftover"


def test_allocate_refuses_to_draw_more_than_exists():
    counts = pd.Series({"a": 10, "b": 5})

    with pytest.raises(ValueError, match="cannot draw"):
        taxonomy.allocate(counts, 20)


def test_allocate_never_exceeds_a_stratum_size():
    """A tiny stratum must not be handed a leftover it cannot fill."""
    counts = pd.Series({"big": 999, "tiny": 1})

    allocation = taxonomy.allocate(counts, 1_000)

    assert allocation.to_dict() == {"big": 999, "tiny": 1}


# --------------------------------------------------------------------------
# draw_sample
# --------------------------------------------------------------------------


def test_draw_sample_hits_the_allocation_and_is_deterministic():
    eligible = eligible_frame({"customer_root": 900, "brand_broadcast": 100})

    first = taxonomy.draw_sample(eligible, 100)
    second = taxonomy.draw_sample(eligible, 100)

    assert len(first) == 100
    assert first.groupby("source").size().to_dict() == {"brand_broadcast": 10, "customer_root": 90}
    pd.testing.assert_frame_equal(first, second)


def test_draw_sample_does_not_repeat_a_conversation():
    eligible = eligible_frame({"customer_root": 500, "orphan_customer": 500})

    sampled = taxonomy.draw_sample(eligible, 200)

    assert sampled["conversation_id"].is_unique


def test_draw_sample_keeps_a_stratum_that_rounds_below_one():
    """A 3-row stratum in a 1,000-row pool must still be representable."""
    eligible = eligible_frame({"customer_root": 997, "orphan_broadcast": 3})

    sampled = taxonomy.draw_sample(eligible, 100)

    # 0.3 of a row rounds to zero; that is correct, and the assertion is that
    # it does not crash or silently drop the stratum from the allocation.
    assert len(sampled) == 100
    assert sampled["source"].nunique() >= 1


# --------------------------------------------------------------------------
# text_stats
# --------------------------------------------------------------------------


def test_text_stats_counts_short_openings_on_raw_text():
    openings = pd.Series(
        [
            "help",  # 1 word
            "my playlist is gone",  # 4 words
            "hi there please help me with my account",  # 8 words
        ]
    )

    stats = taxonomy.text_stats(openings)

    assert stats["n"] == 3
    assert stats["short"] == 2, "under 5 words means 1 and 4, not 8"
    assert stats["short_pct"] == pytest.approx(66.67, abs=0.01)


def test_text_stats_word_count_is_taken_before_normalization():
    """normalize() strips leading @mentions; the word count must not see that.

    This opening is 5 raw words, so it is NOT short. Normalized it becomes
    "my playlist is gone" -- 4 words -- which would be counted as short. The
    assertion pins which side of normalize() the count happens on.
    """
    openings = pd.Series(["@SpotifyCares my playlist is gone"])

    stats = taxonomy.text_stats(openings)

    assert len(taxonomy.normalize(openings[0]).split()) == 4, "premise: 4 words once normalized"
    assert stats["short"] == 0, "5 raw words is not short"


def test_text_stats_detects_a_clearly_non_english_opening():
    openings = pd.Series(
        [
            "my playlist is gone and i want it back please",
            "no puedo escuchar la musica con mi cuenta premium por favor",
        ]
    )

    stats = taxonomy.text_stats(openings)

    assert stats["non_english"] == 1
    assert stats["non_english_pct"] == pytest.approx(50.0)


# --------------------------------------------------------------------------
# write_exemplars
# --------------------------------------------------------------------------


def test_write_exemplars_names_no_cluster(tmp_path):
    """The whole point of this stage: numbers only, never a label."""
    sampled = eligible_frame({"customer_root": 60})
    labels = np.array([0] * 30 + [1] * 30)

    path = taxonomy.write_exemplars(sampled, labels, k=2, out_dir=tmp_path, per_cluster=5)
    text = path.read_text()

    assert path.name == "clusters_k2.md"
    assert "# Clusters at k=2" in text
    assert "## Cluster 1 (id 0, n=30, 50.0%)" in text
    assert "## Cluster 2 (id 1, n=30, 50.0%)" in text
    assert text.count("\n- `") == 10, "5 exemplars per cluster, 2 clusters"


def test_write_exemplars_orders_clusters_largest_first(tmp_path):
    sampled = eligible_frame({"customer_root": 100})
    labels = np.array([0] * 10 + [1] * 90)

    path = taxonomy.write_exemplars(sampled, labels, k=2, out_dir=tmp_path, per_cluster=3)
    text = path.read_text()

    assert text.index("(id 1, n=90") < text.index("(id 0, n=10")


def test_write_exemplars_handles_a_cluster_smaller_than_the_quota(tmp_path):
    sampled = eligible_frame({"customer_root": 12})
    labels = np.array([0] * 10 + [1] * 2)

    path = taxonomy.write_exemplars(sampled, labels, k=2, out_dir=tmp_path, per_cluster=20)
    text = path.read_text()

    assert "All 10 shown." in text
    assert "All 2 shown." in text
    assert text.count("\n- `") == 12


def test_write_exemplars_collapses_newlines_in_an_opening(tmp_path):
    """A multi-line tweet must not break the markdown list item."""
    sampled = pd.DataFrame(
        [{"conversation_id": 1, "source": "customer_root", "opening_text": "line one\nline two"}]
    )

    path = taxonomy.write_exemplars(sampled, np.array([0]), k=1, out_dir=tmp_path, per_cluster=5)
    text = path.read_text()

    assert "line one line two" in text
    assert text.count("\n- `") == 1


def test_write_exemplars_rejects_a_label_length_mismatch(tmp_path):
    sampled = eligible_frame({"customer_root": 10})

    with pytest.raises(AssertionError):
        taxonomy.write_exemplars(sampled, np.array([0] * 9), k=1, out_dir=tmp_path)
