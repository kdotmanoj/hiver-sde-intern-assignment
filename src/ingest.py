"""Load data/raw/twcs.csv and normalize its types. Stage 1 of the pipeline.

Three things about this file make it hostile to a naive reader:

1. `wc -l` reports 3,002,524 lines but there are only 2,811,774 rows. Tweets
   contain literal newlines inside quoted `text` fields. Nothing here may ever
   split on newlines; duckdb's CSV reader handles the quoting and is the only
   thing allowed to parse the file.
2. `response_tweet_id` is a comma-separated *list* on 222,426 rows, not a
   number. It is read as VARCHAR on purpose; typing it as a number silently
   corrupts every branching row.
3. `created_at` is Twitter's format, `Tue Oct 31 22:10:47 +0000 2017`. It is
   parsed with an explicit format string — inference on 2.8M rows is both slow
   and a correctness risk on the ambiguous ones.

Column types are declared explicitly rather than sampled, so a value in the
last chunk cannot contradict a type guessed from the first.

Staleness: `data/interim/tweets.parquet` is derived data, exactly like the LLM
cache. It does NOT invalidate itself. If twcs.csv changes, or any parsing rule
in this module changes (the date format, a dtype, a new derived column), the
parquet is silently stale and every downstream stage inherits the old answer.
Regenerate it with `make clean && make ingest`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

# src/ingest.py -> src -> repo root. Anchoring here means paths do not depend
# on the directory a script happens to be run from. Same rule as src/llm.py.
REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_CSV = REPO_ROOT / "data" / "raw" / "twcs.csv"
INTERIM_PARQUET = REPO_ROOT / "data" / "interim" / "tweets.parquet"

# Explicit types, no sampling. response_tweet_id is VARCHAR because it holds
# comma-separated lists; created_at is VARCHAR because duckdb cannot parse
# Twitter's format and we want one explicit parse, in pandas, below.
CSV_COLUMNS = {
    "tweet_id": "BIGINT",
    "author_id": "VARCHAR",
    "inbound": "BOOLEAN",
    "created_at": "VARCHAR",
    "text": "VARCHAR",
    "response_tweet_id": "VARCHAR",
    "in_response_to_tweet_id": "BIGINT",
}

# `Tue Oct 31 22:10:47 +0000 2017`
CREATED_AT_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def load_tweets(path: Path | str = RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV, parse timestamps, and verify the frame's invariants."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"raw CSV not found at {path}")

    raw = duckdb.sql(
        "SELECT * FROM read_csv(?, header=true, columns=?)",
        params=[str(path), CSV_COLUMNS],
    ).df()
    n_raw = len(raw)

    parsed = raw.assign(
        created_at=pd.to_datetime(
            raw["created_at"], format=CREATED_AT_FORMAT, utc=True, errors="coerce"
        ),
        # Pinned to nullable Int64 rather than left as float64-with-NaN. duckdb
        # hands back float64 from the CSV but Int64 from the parquet, and a
        # dtype that depends on which loader ran is a bug waiting to happen.
        # Int64 also keeps ids out of floats entirely.
        in_response_to_tweet_id=raw["in_response_to_tweet_id"].astype("Int64"),
    )

    # errors="coerce" above turns a bad timestamp into NaT rather than an
    # exception, so we can report *which* rows failed instead of just the first.
    unparseable = parsed["created_at"].isna()
    n_unparseable = int(unparseable.sum())
    if n_unparseable:
        sample = raw.loc[unparseable, ["tweet_id", "created_at"]].head(5)
        raise ValueError(
            f"{n_unparseable} rows have a created_at that does not match "
            f"{CREATED_AT_FORMAT!r}. First few:\n{sample.to_string(index=False)}"
        )

    assert len(parsed) == n_raw, f"date parsing changed row count: {n_raw} -> {len(parsed)}"
    assert parsed["tweet_id"].notna().all(), "tweet_id has nulls; it is the graph's key"
    assert parsed["tweet_id"].is_unique, (
        "tweet_id is not unique. The whole thread reconstruction assumes "
        "tweet_id -> parent is a function; fix this before going further."
    )

    _print_summary(parsed, n_raw=n_raw, n_unparseable=n_unparseable)
    return parsed


def _print_summary(df: pd.DataFrame, n_raw: int, n_unparseable: int) -> None:
    """The one-line-per-stage row-count row, plus the counts that shape stage 2."""
    null_parent = df["in_response_to_tweet_id"].isna()
    # dropna=False is explicit: a null `inbound` must show up as its own bucket
    # rather than vanishing from the split.
    by_inbound = df.loc[null_parent].groupby("inbound", dropna=False).size()
    n_inbound = int(by_inbound.get(True, 0))
    n_outbound = int(by_inbound.get(False, 0))
    n_branching = int(df["response_tweet_id"].str.contains(",", na=False).sum())

    print(
        f"ingest: raw rows {n_raw:,} -> typed {n_raw:,} -> "
        f"dated {len(df):,} ({n_unparseable:,} unparseable, lost {n_raw - len(df):,})"
    )
    print(
        f"  null parent: {int(null_parent.sum()):,} total = "
        f"{n_inbound:,} inbound (customer roots) + "
        f"{n_outbound:,} outbound (not roots)"
    )
    print(f"  branching:   {n_branching:,} rows with a comma-separated response_tweet_id")


# --------------------------------------------------------------------------
# derived parquet
# --------------------------------------------------------------------------


def write_parquet(df: pd.DataFrame, path: Path | str = INTERIM_PARQUET) -> Path:
    """Persist the normalized frame. Written via duckdb so pyarrow is not needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duckdb.sql("COPY (SELECT * FROM df) TO ? (FORMAT parquet)", params=[str(path)])
    return path


def read_parquet(path: Path | str = INTERIM_PARQUET) -> pd.DataFrame:
    """Read back what write_parquet wrote. Assumes it is fresh — see module docstring."""
    df = duckdb.sql("SELECT * FROM read_parquet(?)", params=[str(path)]).df()
    if "in_response_to_tweet_id" in df.columns:
        # Same dtype pin as load_tweets, so callers cannot tell which loader ran.
        df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype("Int64")
    return df


def load_cached_or_raw(
    parquet: Path | str = INTERIM_PARQUET, csv: Path | str = RAW_CSV
) -> pd.DataFrame:
    """Prefer the derived parquet, fall back to parsing the CSV.

    Staleness is not detected here; this trusts the parquet. `make clean`
    removes it when the CSV or the parsing rules change.
    """
    parquet = Path(parquet)
    if parquet.exists():
        df = read_parquet(parquet)
        print(f"ingest: loaded {len(df):,} rows from cached {parquet.name}")
        return df
    return load_tweets(csv)


def _display_path(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise (e.g. a --out in /tmp)."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    parser.add_argument("--out", type=Path, default=INTERIM_PARQUET)
    parser.add_argument(
        "--no-write", action="store_true", help="print the summary without writing parquet"
    )
    args = parser.parse_args()

    df = load_tweets(args.csv)
    if not args.no_write:
        out = write_parquet(df, args.out)
        print(f"  wrote {_display_path(out)} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
