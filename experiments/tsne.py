"""
t-SNE of test-set representations.

Takes the .npz written by `stance.train --embeddings_file ...` (joint representation r
for our model, mean-pooled PLM output for the baseline) and plots a random sample of
N test instances per gold class.

Usage:
  python -m experiments.tsne --embeddings results/tsne/ours.npz --out figures/tsne_ours.pdf
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from stance.data import ID2LABEL

COLORS = {0: "red", 1: "green", 2: "blue"}  # Refute, Support, Neutral


def sample_per_class(embeddings, labels, n, rng):
    idx = []
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        idx.extend(rng.choice(members, size=min(n, len(members)), replace=False))
    idx = np.array(idx)
    return embeddings[idx], labels[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per_class", type=int, default=300)
    parser.add_argument("--perplexity", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.embeddings)
    rng = np.random.default_rng(args.seed)
    emb, labels = sample_per_class(data["embeddings"], data["labels"], args.per_class, rng)

    xy = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed,
              max_iter=1000, learning_rate="auto", init="pca").fit_transform(emb)

    fig, ax = plt.subplots(figsize=(6, 4))
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], c=COLORS[int(c)], label=ID2LABEL[int(c)], alpha=0.6, s=12)
    ax.set_xlabel("t-SNE dimension 1", fontsize=8)
    ax.set_ylabel("t-SNE dimension 2", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=10)
    ax.grid(False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
