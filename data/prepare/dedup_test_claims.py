"""
Check claim overlap between train and test splits and (optionally) remove
overlapping test instances.

A test claim is a duplicate if its normalized text exactly matches a training
claim, and a near-duplicate if its cosine similarity to any training claim
(all-MiniLM-L6-v2 embeddings) is >= --threshold.

Usage:
  python -m data.prepare.dedup_test_claims --data_source ezstance \
      --train data/ez-stance/subtaskB/claim/politic/raw_train_all_onecol.csv \
      --test  data/ez-stance/subtaskB/claim/politic/raw_test_all_onecol.csv \
      --out   data/ez-stance/subtaskB/claim/politic/test_dedup.csv
"""
import argparse
import re

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CLAIM_COL = {"stanceosaurus": "claim", "ezstance": "Target 1"}


def normalize(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", str(text).lower().strip())
    return re.sub(r"\s+", " ", text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", required=True, choices=list(CLAIM_COL))
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", default=None, help="write the test set without (near-)duplicate claims")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--encoder", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    col = CLAIM_COL[args.data_source]
    train_df, test_df = pd.read_csv(args.train), pd.read_csv(args.test)
    train_claims = train_df[col].dropna().astype(str).str.strip().unique().tolist()
    test_claims = test_df[col].dropna().astype(str).str.strip().unique().tolist()

    train_norm = {normalize(c) for c in train_claims}
    exact = {c for c in test_claims if normalize(c) in train_norm}

    model = SentenceTransformer(args.encoder)
    sims = cosine_similarity(model.encode(test_claims, batch_size=64), model.encode(train_claims, batch_size=64))
    near = {c for c, s in zip(test_claims, sims.max(axis=1)) if s >= args.threshold} - exact

    drop = test_df[col].astype(str).str.strip().isin(exact | near)
    print(f"test claims: {len(test_claims)} | exact duplicates: {len(exact)} | "
          f"near duplicates (>= {args.threshold}): {len(near)}")
    print(f"test instances: {len(test_df)} | removed: {drop.sum()} ({drop.mean() * 100:.2f}%)")
    for c in sorted(near):
        print(f"  near-dup: {c}")

    if args.out:
        test_df[~drop].to_csv(args.out, index=False)
        print(f"saved {(~drop).sum()} test instances to {args.out}")


if __name__ == "__main__":
    main()
