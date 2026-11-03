"""
Stage 1 of the inference pipeline: score every (claim, tweet) pair in the corpus with
the claim matcher and keep pairs with match probability above the threshold.

Usage:
  python -m pipeline.match_corpus --matcher models/claim_matcher \
      --claims data/votestance/claims.csv --corpus data/votestance/corpus.csv \
      --out results/election/claim_tweet_matches.csv
  # shard across GPUs with --claim_start / --claim_end (inclusive claim_id range)
"""
import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

from pipeline.claim_matcher import ClaimMatcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matcher", required=True)
    parser.add_argument("--claims", default="data/votestance/claims.csv")
    parser.add_argument("--corpus", required=True, help="csv from pipeline.preprocess_tweets")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=None, help="default: threshold saved with the matcher")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--claim_start", type=int, default=None)
    parser.add_argument("--claim_end", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = ClaimMatcher.load(args.matcher, device)
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", 0.6)

    claims = pd.read_csv(args.claims)
    if args.claim_start is not None:
        claims = claims[claims["claim_id"] >= args.claim_start]
    if args.claim_end is not None:
        claims = claims[claims["claim_id"] <= args.claim_end]
    corpus = pd.read_csv(args.corpus, usecols=["tweet_id", "tweet_text"], low_memory=False).dropna()
    tweet_ids, tweets = corpus["tweet_id"].tolist(), corpus["tweet_text"].tolist()
    print(f"{len(claims)} claims x {len(tweets)} tweets, threshold={threshold}")

    rows = []
    for claim_id, claim in tqdm(zip(claims["claim_id"], claims["claim"]), total=len(claims), desc="claims"):
        probs = model.predict_proba([claim] * len(tweets), tweets, batch_size=args.batch_size)
        rows.extend((claim_id, tid, p) for tid, p in zip(tweet_ids, probs) if p > threshold)
        print(f"claim {claim_id}: {sum(1 for r in rows if r[0] == claim_id)} matches")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows, columns=["claim_id", "tweet_id", "match_prob"]).to_csv(args.out, index=False)
    print(f"saved {len(rows)} matched pairs to {args.out}")


if __name__ == "__main__":
    main()
