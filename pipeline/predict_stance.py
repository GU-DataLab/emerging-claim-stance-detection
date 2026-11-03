"""
Stage 2 of the inference pipeline: classify the stance of every matched pair with a
trained stance model (our model fine-tuned on VoteStance).

Usage:
  python -m pipeline.predict_stance --model models/votestance/ours_seed42 \
      --matches results/election/claim_tweet_matches.csv \
      --claims data/votestance/claims.csv --corpus data/votestance/corpus.csv \
      --out results/election/stance_predictions.csv
"""
import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

from stance.modeling import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="checkpoint saved by stance.train")
    parser.add_argument("--matches", required=True)
    parser.add_argument("--claims", default="data/votestance/claims.csv")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device=device)

    claims = pd.read_csv(args.claims)[["claim_id", "claim"]]
    corpus = pd.read_csv(args.corpus, usecols=["tweet_id", "tweet_text"], low_memory=False)
    pairs = (pd.read_csv(args.matches)
             .merge(claims, on="claim_id", how="inner")
             .merge(corpus, on="tweet_id", how="inner")
             .dropna(subset=["claim", "tweet_text"])
             .reset_index(drop=True))
    print(f"{len(pairs)} matched pairs")

    preds = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(pairs), args.batch_size), desc="stance"):
            batch = pairs.iloc[i:i + args.batch_size]
            logits = model(claims=batch["claim"].tolist(), posts=batch["tweet_text"].tolist()).logits
            preds.extend(model.config.id2label[p] for p in logits.argmax(dim=1).cpu().tolist())
    pairs["stance_label"] = preds

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pairs[["claim_id", "tweet_id", "match_prob", "stance_label"]].to_csv(args.out, index=False)
    print(pairs["stance_label"].value_counts().to_dict())
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
