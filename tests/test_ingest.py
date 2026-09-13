"""Tests for the raw CSV loader.

The central claim: the row count is the number of CSV *records*, not the
number of lines in the file. Tweets contain literal newlines inside quoted
fields, which is why the full file has 2,811,774 rows but 3,002,524 lines.
The fixture below reproduces that in miniature, so the test fails if anything
ever starts splitting on newlines.

No test touches the real 516MB file.
"""

import pandas as pd
import pytest

from src.ingest import load_tweets, read_parquet, write_parquet

# Row 2's text spans three lines. Row 3 has a comma-separated response_tweet_id
# (the branching case) and a null parent. Row 4 replies to a LOWER id, because
# tweet ids are not chronological.
CSV = '''tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
1,105834,True,Tue Oct 31 22:10:47 +0000 2017,@AppleSupport help,2,
2,AppleSupport,False,Tue Oct 31 22:12:00 +0000 2017,"Hi there!
Please DM us,
we can help.",3,1
3,105835,True,Tue Oct 31 22:15:30 +0000 2017,thanks,"4,5",2
4,AppleSupport,False,Wed Nov 01 09:00:00 +0000 2017,you are welcome,,3
'''


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "twcs_mini.csv"
    path.write_text(CSV, encoding="utf-8")
    return path


def test_row_count_is_records_not_lines(csv_path):
    """4 records, but 6 newline-terminated lines after the header."""
    assert len(CSV.splitlines()) - 1 == 6, "fixture must actually contain embedded newlines"

    df = load_tweets(csv_path)

    assert len(df) == 4
    assert list(df["tweet_id"]) == [1, 2, 3, 4]


def test_embedded_newlines_survive_in_the_text(csv_path):
    df = load_tweets(csv_path)
    text = df.loc[df["tweet_id"] == 2, "text"].iloc[0]

    assert text == "Hi there!\nPlease DM us,\nwe can help."
    assert "\n" in text, "the newline is part of the tweet, not a record separator"


def test_created_at_parses_to_the_expected_instant(csv_path):
    df = load_tweets(csv_path)
    first = df.loc[df["tweet_id"] == 1, "created_at"].iloc[0]

    assert first == pd.Timestamp("2017-10-31 22:10:47", tz="UTC")
    # tz-aware and UTC is the claim; the resolution is pandas' choice (pandas 3
    # returns microseconds here) and nothing downstream depends on it.
    assert isinstance(df["created_at"].dtype, pd.DatetimeTZDtype)
    assert str(df["created_at"].dtype.tz) == "UTC"
    assert df["created_at"].notna().all()

    # The last row is a different month and day-of-week, so a format string
    # that only happened to fit October would fail here.
    last = df.loc[df["tweet_id"] == 4, "created_at"].iloc[0]
    assert last == pd.Timestamp("2017-11-01 09:00:00", tz="UTC")


def test_response_tweet_id_stays_a_string(csv_path):
    """Typing it as a number would corrupt every branching row into NaN."""
    df = load_tweets(csv_path)
    branching = df.loc[df["tweet_id"] == 3, "response_tweet_id"].iloc[0]

    assert branching == "4,5"
    assert df["response_tweet_id"].dtype == object or str(df["response_tweet_id"].dtype) == "str"


def test_types_and_nulls(csv_path):
    df = load_tweets(csv_path)

    assert df["tweet_id"].dtype == "int64"
    assert df["inbound"].tolist() == [True, False, True, False]
    # Row 1 is a root: its parent is null, not 0 and not the string "".
    assert pd.isna(df.loc[df["tweet_id"] == 1, "in_response_to_tweet_id"].iloc[0])
    assert df.loc[df["tweet_id"] == 4, "in_response_to_tweet_id"].iloc[0] == 3


def test_duplicate_tweet_id_fails_loudly(tmp_path):
    """tweet_id uniqueness is what makes tweet_id -> parent a function."""
    path = tmp_path / "dupes.csv"
    path.write_text(
        CSV + "1,105834,True,Tue Oct 31 22:10:47 +0000 2017,dupe,,\n", encoding="utf-8"
    )

    with pytest.raises(AssertionError, match="not unique"):
        load_tweets(path)


def test_bad_timestamp_names_the_offending_rows(tmp_path):
    path = tmp_path / "baddate.csv"
    path.write_text(
        "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
        "1,105834,True,2017-10-31T22:10:47Z,iso not twitter,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        load_tweets(path)


def test_parquet_round_trip(csv_path, tmp_path):
    """The derived parquet must return the same rows and the same instants."""
    df = load_tweets(csv_path)
    out = write_parquet(df, tmp_path / "tweets.parquet")
    back = read_parquet(out)

    assert len(back) == len(df)
    assert list(back["tweet_id"]) == list(df["tweet_id"])
    assert back.loc[back["tweet_id"] == 2, "text"].iloc[0] == df.loc[1, "text"]
    assert back["created_at"].iloc[0] == df["created_at"].iloc[0]
