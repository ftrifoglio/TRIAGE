"""
Download and preprocess Yelp Round 12 (2018) reviews.

Recipe:
    raw reviews -> dedupe (user, item) -> (20, 10)-core -> adjacency-list output.

Expected stats (LightGCN, He et al. 2020, Table 2):
    31,668 users, 38,048 items, 1,561,406 interactions.

Usage:
    uv run python scripts/yelp_2018.py
"""

import json

import pandas as pd

from common.preprocessing.download import dataset_dir, kaggle_download_file_if_missing
from common.preprocessing.io import save_adjacency
from common.preprocessing.kcore import kcore
from common.preprocessing.logging import (
    log_final_vs_expected,
    log_pr_vs_reference,
    log_recall_miss_diagnostic,
    log_stage,
    log_test_split_parity,
)

KAGGLE_DATASET = "yelp-dataset/yelp-dataset"
KAGGLE_VERSION = 7
REVIEW_FILENAME = "yelp_academic_dataset_review.json"
EXPECTED = {"users": 31668, "items": 38048, "interactions": 1_561_406}


def main() -> None:
    """Download and preprocess Yelp Round 12 (2018) reviews."""
    out = dataset_dir("yelp_2018/triage_splits")
    review_json = out / REVIEW_FILENAME
    ref = out.parent / "lightgcn_splits"

    print("[1/5] download raw data")
    kaggle_download_file_if_missing(
        KAGGLE_DATASET, KAGGLE_VERSION, REVIEW_FILENAME, out
    )

    print("\n[2/5] load raw")
    records: list[tuple[str, str, str]] = []
    with open(review_json, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            records.append((d["user_id"], d["business_id"], d["date"]))
    df = pd.DataFrame.from_records(records, columns=["user", "item", "timestamp"])
    log_stage("raw reviews", df)
    ts = pd.to_datetime(df["timestamp"])
    print(f"  timestamp range: {ts.min()} -> {ts.max()}")

    print("\n[3/5] dedupe + (20, 10)-core")
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["user", "item"], keep="last")
        .reset_index(drop=True)
    )
    log_stage("after dedupe", df)

    print("\n[4/5] (20, 10)-core")
    df = kcore(df, ku=20, ki=10, verbose=True)

    log_final_vs_expected(df, EXPECTED)
    log_pr_vs_reference(df, ref)

    print("\n[5/5] save adjacency lists")
    save_adjacency(df, out, test_ratio=0.2, split_strategy="temporal")
    print(f"  wrote user_list.txt, item_list.txt, train.txt, test.txt -> {out}")
    log_test_split_parity(out, ref)
    log_recall_miss_diagnostic(df, ref, test_ratio=0.2)


if __name__ == "__main__":
    main()


# File refactored by claude-opus-4-8 using prompt: Add pretty print statements to
# `scripts/amazon_books_2014.py`, `scripts/gowalla.py`, `scripts/yelp_2018.py`,
# `src/common/preprocessing/logging.py` and `src/common/preprocessing/download.py`. I
# want to see details of data provenance and vintage. Each step (download, load, dedupe,
# k-core, output) should be clearly identifiable, and each step should report only
# useful information. If a reference split and a diagnostic cannot be run, it should say
# so. Use plain `print` calls inside the existing functions; don't add new functions,
# constants or imports.
