"""Download public source files and build one frozen chronological protocol."""

from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import config

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
URLS = {
    "Video_Games.csv.gz": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz",
    "meta_Video_Games.jsonl.gz": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz",
}
EXPECTED_SHA256 = {
    "Video_Games.csv.gz": "a2bde5f3b945960d161538c200dd87845e6ee471b46da96410dde61613c6901c",
    "meta_Video_Games.jsonl.gz": "ced9696b0874121327a438bfe9d3d01aba45da158e58f94f4920352d75fc1fc9",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_data():
    """Atomic downloads; existing complete files are reused, hashes are recorded."""
    RAW.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for filename, url in URLS.items():
        path = RAW / filename
        if not path.exists():
            print(f"Downloading {filename}", flush=True)
            temporary = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        hashes[filename] = sha256(path)
        if hashes[filename] != EXPECTED_SHA256[filename]:
            raise ValueError(f"Source checksum mismatch for {filename}; verify the source version before continuing")
    return hashes


def stable_order(values, namespace):
    return sorted(values, key=lambda x: (hashlib.sha256(f"{config.SEED}:{namespace}:{x}".encode()).digest(), x))


def _evaluation_users(frame):
    eligible = frame.loc[frame.rating >= config.RELEVANT_RATING, "user_idx"].unique().tolist()
    return np.array(stable_order(eligible, "evaluation")[: config.MAX_EVALUATION_USERS], dtype=np.int64)


def prepare_data():
    hashes = download_data()
    original = pd.read_csv(RAW / "Video_Games.csv.gz", dtype={"user_id": str, "parent_asin": str})
    missing = {str(k): int(v) for k, v in original.isna().sum().items()}
    exact_duplicates = int(original.duplicated().sum())
    clean = original.dropna(subset=["user_id", "parent_asin", "rating", "timestamp"]).copy()
    clean = clean[(clean.rating.between(1, 5)) & (clean.timestamp > 0)]
    clean["timestamp"] = clean.timestamp.astype("int64")
    # Recommend previously unseen products; a pair's first review defines its event.
    clean = clean.sort_values(["timestamp", "user_id", "parent_asin"], kind="stable")
    pair_duplicates = int(clean.duplicated(["user_id", "parent_asin"]).sum())
    clean = clean.drop_duplicates(["user_id", "parent_asin"], keep="first")
    train_cutoff = int(clean.timestamp.quantile(config.TRAIN_FRACTION, interpolation="lower"))
    validation_cutoff = int(clean.timestamp.quantile(config.VALIDATION_END_FRACTION, interpolation="lower"))
    history = clean[clean.timestamp < train_cutoff]
    # Select the catalog and cohort solely from pre-validation observations.
    popularity = history.groupby("parent_asin").size().rename("count").reset_index()
    popularity = popularity.sort_values(["count", "parent_asin"], ascending=[False, True])
    item_ids = sorted(popularity.parent_asin.head(config.MAX_ITEMS).tolist())
    catalog_history = history[history.parent_asin.isin(item_ids)]
    activity = catalog_history.groupby("user_id").size()
    user_ids = sorted(stable_order(activity[activity >= config.MIN_TRAIN_INTERACTIONS].index.tolist(), "cohort")[: config.MAX_USERS])
    selected = clean[clean.user_id.isin(user_ids) & clean.parent_asin.isin(item_ids)].copy()
    # Remove unobserved selected items after user subsampling; never use holdout activity.
    item_ids = sorted(selected.loc[selected.timestamp < train_cutoff, "parent_asin"].unique())
    selected = selected[selected.parent_asin.isin(item_ids)].copy()
    user_map = {value: idx for idx, value in enumerate(user_ids)}
    item_map = {value: idx for idx, value in enumerate(item_ids)}
    selected["user_idx"] = selected.user_id.map(user_map).astype("int32")
    selected["item_idx"] = selected.parent_asin.map(item_map).astype("int32")
    selected = selected.reset_index(drop=True)
    train = selected[selected.timestamp < train_cutoff].copy()
    valid = selected[(selected.timestamp >= train_cutoff) & (selected.timestamp < validation_cutoff)].copy()
    test = selected[selected.timestamp >= validation_cutoff].copy()
    assert 100_000 <= len(selected) <= 2_000_000, f"Interaction constraint failed: {len(selected)}"
    assert len(user_ids) >= 1000 and len(item_ids) >= 1000
    assert train.timestamp.max() < valid.timestamp.min() <= valid.timestamp.max() < test.timestamp.min()
    assert not selected.duplicated(["user_idx", "item_idx"]).any()

    requested = set(item_ids)
    rows = {}
    with gzip.open(RAW / "meta_Video_Games.jsonl.gz", "rt", encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            key = item.get("parent_asin")
            if key in requested:
                details = item.get("details") or {}
                rows[key] = {
                    "parent_asin": key,
                    "title": item.get("title") or "",
                    "categories": item.get("categories") or [],
                    "store": item.get("store") or "",
                    "brand": str(details.get("Brand", "")) if isinstance(details, dict) else "",
                    "description_present": bool(item.get("description")),
                }
    metadata = pd.DataFrame([{"parent_asin": key, **rows.get(key, {})} for key in item_ids])
    metadata = metadata.reindex(columns=["parent_asin", "title", "categories", "store", "brand", "description_present"])
    metadata["item_idx"] = np.arange(len(item_ids))
    coverage = {field: int(metadata[field].map(lambda x: bool(x) if not isinstance(x, float) else False).sum()) for field in ["title", "categories", "brand", "store", "description_present"]}
    cold_valid = clean[(clean.timestamp >= train_cutoff) & (clean.timestamp < validation_cutoff)]
    cold_test = clean[clean.timestamp >= validation_cutoff]
    summary = {
        "source_rows": len(original), "source_users": int(original.user_id.nunique()),
        "source_items": int(original.parent_asin.nunique()), "missing": missing,
        "exact_duplicates": exact_duplicates, "repeated_pairs": pair_duplicates,
        "clean_rows_before_selection": len(clean), "selected_rows": len(selected),
        "users": len(user_ids), "items": len(item_ids),
        "train_rows": len(train), "validation_rows": len(valid), "test_rows": len(test),
        "train_cutoff_ms": train_cutoff, "validation_cutoff_ms": validation_cutoff,
        "train_cutoff_utc": str(pd.to_datetime(train_cutoff, unit="ms", utc=True)),
        "validation_cutoff_utc": str(pd.to_datetime(validation_cutoff, unit="ms", utc=True)),
        "sparsity": float(1 - len(selected) / (len(user_ids) * len(item_ids))),
        "metadata_coverage": coverage, "sha256": hashes,
        "excluded_validation_rows": len(cold_valid) - len(valid),
        "excluded_test_rows": len(cold_test) - len(test),
        "validation_users_evaluated": len(_evaluation_users(valid)),
        "test_users_evaluated": len(_evaluation_users(test)),
        "seed": config.SEED,
    }
    return dict(interactions=selected, train=train, validation=valid, test=test,
                metadata=metadata, n_users=len(user_ids), n_items=len(item_ids),
                validation_users=_evaluation_users(valid), test_users=_evaluation_users(test),
                train_cutoff_ms=train_cutoff, validation_cutoff_ms=validation_cutoff, summary=summary)
