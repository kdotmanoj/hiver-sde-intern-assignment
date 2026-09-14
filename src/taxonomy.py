"""Stage 5: cluster customer openings so the intent taxonomy can be read off them.

This module produces EVIDENCE, not labels. It samples 2,000 conversation
openings, embeds them, runs k-means across a range of k, and writes the
exemplars of each cluster to notes/clusters_k{n}.md. Clusters are numbered and
never named. Deciding what the clusters mean, and what the taxonomy is, is not
this module's job and must not become it.

Three things are reported alongside the clustering, because they bound what a
taxonomy built on this sample can claim:

- non-English rate. A floor, not an estimate -- see the note on the import
  below.
- share of openings under 5 words. A one-word opening ("help") carries almost
  no intent signal, and whatever cluster it lands in says more about the
  embedding than about the customer.
- per-stratum counts, so the sample can be checked against the population.

Sampling: 2,000 stratified proportionally across `source`, from the
conversations that have a substantive first reply. The proportional choice
keeps cluster sizes readable as intent frequency; it also means the three
non-customer_root strata contribute ~12 rows between them and are effectively
noise. Conversations with a null first_reply_text are excluded because a
conversation with no substantive brand reply cannot later be used to evaluate
a generated reply against.

Known debt: is_non_english is imported from scripts/brand_survey.py, which
inverts the dependency direction src/deflection.py:7-9 establishes ("a
pipeline stage must not import from a one-off script"). Tolerated because the
language check here is a printed diagnostic, not pipeline logic, and because
extracting it would need a byte-diff run against the survey's committed
0.07% figure. Extract it to src/ the moment anything downstream branches on
it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scripts.brand_survey import is_non_english
from src.deflection import normalize
from src.embed import cache_status, embed
from src.ingest import REPO_ROOT, _display_path, read_parquet

INTERIM = REPO_ROOT / "data" / "interim"
SPOTIFY_PARQUET = INTERIM / "spotify.parquet"
NOTES = REPO_ROOT / "notes"

# Feeds .sample(random_state=SEED) for the stratified draw and for the
# per-cluster exemplar draw, and KMeans(random_state=SEED). Those are the only
# three places anything random happens in this stage.
SEED = 42

N_SAMPLE = 2_000

# k-means is run across this whole range and the diagnostics printed for all of
# it, so the elbow and the silhouette curve are both visible.
K_MIN = 5
K_MAX = 12

# Exemplars are only exported for these k. Eight files is more reading than the
# choice needs; three spans the plausible range.
K_EXPORT = (6, 8, 10)

EXEMPLARS_PER_CLUSTER = 20

# An opening this short is treated as carrying no usable intent signal. Counted
# and reported, NOT dropped -- dropping it would hide how common it is.
SHORT_WORDS = 5


# --------------------------------------------------------------------------
# load + sample
# --------------------------------------------------------------------------


def load_eligible(path: Path = SPOTIFY_PARQUET) -> tuple[pd.DataFrame, int]:
    """Conversations with a substantive first reply. Returns (eligible, n_all)."""
    conversations = read_parquet(path)
    n_all = len(conversations)

    eligible = conversations[conversations["first_reply_text"].notna()]
    print(f"  loaded {n_all:,} conversations -> {len(eligible):,} with a substantive first reply")
    return eligible.reset_index(drop=True), n_all


def allocate(counts: pd.Series, total: int = N_SAMPLE) -> pd.Series:
    """Split `total` across strata in proportion to `counts`, summing exactly.

    Largest-remainder: take the floor of each exact share, then hand the
    leftover one at a time to the strata with the biggest discarded fraction.
    Ties break on the stratum name so the result does not depend on dict order.

    Written out rather than rounded because round() does not sum to `total` --
    for the real distribution (27,245 / 115 / 34 / 31) it gives 1,999.
    """
    if total > int(counts.sum()):
        raise ValueError(f"cannot draw {total:,} from {int(counts.sum()):,} rows")

    exact = counts / counts.sum() * total
    floors = np.floor(exact).astype(int)
    remainder = total - int(floors.sum())

    # Biggest discarded fraction wins the leftovers; stratum name breaks ties,
    # so the result never depends on the order the groupby happened to return.
    fractions = exact - floors
    order = sorted(counts.index, key=lambda source: (-fractions[source], str(source)))

    allocation = floors.copy()
    for source in order[:remainder]:
        allocation[source] += 1

    # A stratum can be allocated more than it holds only if one is tiny and the
    # remainder lands on it; clamp and re-check rather than silently oversample.
    if (allocation > counts).any():
        raise ValueError(f"allocation exceeds stratum size:\n{allocation[allocation > counts]}")
    assert int(allocation.sum()) == total, f"allocated {int(allocation.sum())}, wanted {total}"
    return allocation


def draw_sample(eligible: pd.DataFrame, total: int = N_SAMPLE) -> pd.DataFrame:
    """Stratified proportional sample across `source`, sorted for determinism."""
    # dropna=False is explicit: a null source must show up as its own stratum
    # rather than vanishing from the draw.
    counts = eligible.groupby("source", dropna=False).size().sort_index()
    allocation = allocate(counts, total)

    parts = [
        eligible[eligible["source"] == source].sample(n, random_state=SEED)
        for source, n in allocation.items()
    ]
    sampled = pd.concat(parts).sort_values("conversation_id", kind="stable")

    assert len(sampled) == total, f"sampled {len(sampled):,}, wanted {total:,}"
    # reindex, not sort_index: a stratum allocated 0 rows contributes no group
    # to the result, and comparing without filling it back in would fail.
    drawn = (
        sampled.groupby("source", dropna=False)
        .size()
        .reindex(allocation.index, fill_value=0)
    )
    assert drawn.equals(allocation), f"per-stratum counts drifted:\n{drawn}\nvs\n{allocation}"
    return sampled.reset_index(drop=True)


# --------------------------------------------------------------------------
# text statistics
# --------------------------------------------------------------------------


def text_stats(openings: pd.Series) -> dict:
    """Non-English and short-opening counts for a series of raw opening_text.

    is_non_english expects normalized input (lowercased, mentions dropped,
    URLs masked), so normalize() is applied first. Word count is taken on the
    RAW text, because normalization drops leading @mentions and would make a
    two-word opening look shorter than the customer typed.
    """
    normalized = openings.map(normalize)
    non_english = normalized.map(is_non_english)

    words = openings.str.split().str.len()
    short = words < SHORT_WORDS

    return {
        "n": len(openings),
        "non_english": int(non_english.sum()),
        "non_english_pct": 100.0 * non_english.mean(),
        "short": int(short.sum()),
        "short_pct": 100.0 * short.mean(),
        "median_words": float(words.median()),
    }


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------


def cluster(vectors: np.ndarray, k: int) -> tuple[np.ndarray, float, float]:
    """k-means for one k. Returns (labels, inertia, silhouette).

    n_init=10 means ten restarts from different centroid seeds, keeping the
    lowest-inertia one; with random_state fixed this is deterministic. Vectors
    arrive L2-normalized from src.embed, so euclidean k-means here is
    equivalent to spherical/cosine k-means and the silhouette below is
    measured in the same geometry.
    """
    model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = model.fit_predict(vectors)
    score = silhouette_score(vectors, labels, random_state=SEED)
    return labels, float(model.inertia_), float(score)


def sweep(vectors: np.ndarray, k_min: int = K_MIN, k_max: int = K_MAX) -> dict[int, np.ndarray]:
    """Run and report every k in the range. Returns {k: labels}.

    Cluster sizes are printed next to the scores on purpose: a good silhouette
    over a partition whose smallest cluster holds 4 rows is not a good k, and
    that is invisible from the score alone.
    """
    print(f"  k-means k={k_min}..{k_max} on {vectors.shape[0]:,}x{vectors.shape[1]} vectors")
    print(f"    {'k':>3}  {'inertia':>10}  {'silhouette':>10}  {'min':>5}  {'med':>5}  {'max':>5}")

    labels_by_k = {}
    for k in range(k_min, k_max + 1):
        labels, inertia, score = cluster(vectors, k)
        labels_by_k[k] = labels

        sizes = pd.Series(labels).value_counts()
        print(
            f"    {k:>3}  {inertia:>10.2f}  {score:>10.4f}  "
            f"{int(sizes.min()):>5}  {int(sizes.median()):>5}  {int(sizes.max()):>5}"
        )
    return labels_by_k


# --------------------------------------------------------------------------
# exemplar export
# --------------------------------------------------------------------------


def write_exemplars(
    sampled: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    out_dir: Path = NOTES,
    per_cluster: int = EXEMPLARS_PER_CLUSTER,
) -> Path:
    """Write one markdown file of random exemplars per cluster, for reading.

    Clusters are identified by number only. No theme, no label, no guess at
    what they have in common -- naming these is the reader's job, and a
    suggested name in this file would anchor it.
    """
    assert len(labels) == len(sampled), f"{len(labels)} labels for {len(sampled)} rows"
    frame = sampled.assign(cluster=labels)

    lines = [
        f"# Clusters at k={k}",
        "",
        f"{len(frame):,} conversation openings, {per_cluster} random exemplars per "
        f"cluster (seed {SEED}). Clusters are numbered in size order, largest first.",
        "",
        "Generated by `src/taxonomy.py`. Do not hand-edit -- it is overwritten.",
        "",
    ]

    # Largest first, so the reader meets the clusters that matter most while
    # they still have patience. .index gives cluster ids in descending size.
    for rank, cluster_id in enumerate(frame["cluster"].value_counts().index, start=1):
        members = frame[frame["cluster"] == cluster_id]
        share = 100.0 * len(members) / len(frame)
        take = min(per_cluster, len(members))
        exemplars = members.sample(take, random_state=SEED).sort_values(
            "conversation_id", kind="stable"
        )

        lines.append(f"## Cluster {rank} (id {cluster_id}, n={len(members):,}, {share:.1f}%)")
        lines.append("")
        if take < len(members):
            lines.append(f"{take} of {len(members):,} shown.")
        else:
            lines.append(f"All {take} shown.")
        lines.append("")

        for row in exemplars.itertuples():
            # Collapse newlines: a multi-line opening would break the list item
            # and silently swallow the rest of the tweet.
            text = " ".join(str(row.opening_text).split())
            lines.append(f"- `{row.conversation_id}` [{row.source}] {text}")
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"clusters_k{k}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def _print_summary(
    n_all: int,
    eligible: pd.DataFrame,
    sampled: pd.DataFrame,
    sample_stats: dict,
    population_stats: dict,
    cached: int,
    uncached: int,
) -> None:
    """The stage row-count row, plus the counts that qualify what it means."""
    strata = sampled.groupby("source", dropna=False).size().sort_values(ascending=False)
    strata_text = " / ".join(f"{source} {n:,}" for source, n in strata.items())

    print(
        f"taxonomy: conversations {n_all:,} -> with reply {len(eligible):,} "
        f"(lost {n_all - len(eligible):,}) -> sampled {len(sampled):,}"
    )
    print(f"  strata:      {strata_text}")
    print(
        f"  language:    {sample_stats['non_english']:,} non-English "
        f"({sample_stats['non_english_pct']:.2f}% FLOOR -- heuristic covers "
        f"es/pt/fr/de/it only, misses id/sv) "
        f"[pool {population_stats['non_english_pct']:.2f}%]"
    )
    print(
        f"  length:      {sample_stats['short']:,} openings under {SHORT_WORDS} words "
        f"({sample_stats['short_pct']:.2f}%), median {sample_stats['median_words']:.0f} words "
        f"[pool {population_stats['short_pct']:.2f}%]"
    )
    print(f"  embeddings:  {cached + uncached:,} unique texts ({cached:,} cached, {uncached:,} to compute)")


# --------------------------------------------------------------------------


def build_taxonomy_inputs(
    path: Path = SPOTIFY_PARQUET,
    out_dir: Path = NOTES,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    k_export: tuple[int, ...] = K_EXPORT,
    write: bool = True,
) -> pd.DataFrame:
    """Run the whole stage and report. Returns the sampled frame with labels."""
    eligible, n_all = load_eligible(path)
    sampled = draw_sample(eligible)

    sample_stats = text_stats(sampled["opening_text"])
    population_stats = text_stats(eligible["opening_text"])

    texts = sampled["opening_text"].tolist()
    cached, uncached = cache_status(texts)
    _print_summary(n_all, eligible, sampled, sample_stats, population_stats, cached, uncached)

    vectors = embed(texts)
    labels_by_k = sweep(vectors, k_min, k_max)

    out = sampled.copy()
    for k in k_export:
        if k not in labels_by_k:
            raise ValueError(f"k={k} was requested for export but is outside {k_min}..{k_max}")
        out[f"cluster_k{k}"] = labels_by_k[k]
        if write:
            written = write_exemplars(sampled, labels_by_k[k], k, out_dir)
            print(f"  wrote {_display_path(written)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", type=Path, default=SPOTIFY_PARQUET)
    parser.add_argument("--out", type=Path, default=NOTES, help="directory for clusters_k*.md")
    parser.add_argument("--k-min", type=int, default=K_MIN)
    parser.add_argument("--k-max", type=int, default=K_MAX)
    parser.add_argument(
        "--no-write", action="store_true", help="print the diagnostics without writing markdown"
    )
    args = parser.parse_args()

    build_taxonomy_inputs(
        path=args.parquet,
        out_dir=args.out,
        k_min=args.k_min,
        k_max=args.k_max,
        write=not args.no_write,
    )


if __name__ == "__main__":
    main()
