"""
Download and preprocess the Gowalla check-ins dataset.

Recipe:
    raw check-ins -> dedupe (user, item) -> 10-core -> adjacency-list output.

Expected stats (NGCF, Wang et al. 2019, Table 1):
    29,858 users, 40,981 items, 1,027,370 interactions.

Usage:
    uv run python scripts/gowalla.py
"""

import numpy as np
import pandas as pd

from common.preprocessing.download import dataset_dir, download_if_missing
from common.preprocessing.io import save_adjacency
from common.preprocessing.kcore import kcore
from common.preprocessing.logging import (
    log_final_vs_expected,
    log_pr_vs_reference,
    log_recall_miss_diagnostic,
    log_stage,
    log_test_split_parity,
)

URL = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
EXPECTED = {"users": 29858, "items": 40981, "interactions": 1_027_370}


def main() -> None:
    """Download and preprocess the Gowalla check-ins dataset."""
    out = dataset_dir("gowalla/triage_splits")
    raw = out / "loc-gowalla_totalCheckins.txt.gz"
    ref = out.parent / "ngcf_splits"

    print("[1/5] download raw data")
    download_if_missing(URL, raw)

    print("\n[2/5] load raw")
    df = pd.read_csv(
        raw,
        sep="\t",
        header=None,
        names=["user", "timestamp", "lat", "lon", "item"],
        usecols=["user", "timestamp", "item"],
        dtype={"user": np.int32, "item": np.int32},
    )
    log_stage("raw check-ins", df)
    ts = pd.to_datetime(df["timestamp"])
    print(f"  timestamp range: {ts.min()} -> {ts.max()}")

    print("\n[3/5] dedupe")
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["user", "item"], keep="last")
        .reset_index(drop=True)
    )
    log_stage("after dedupe", df)

    print("\n[4/5] 10-core")
    df = kcore(df, ku=10, ki=10, verbose=True)

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
