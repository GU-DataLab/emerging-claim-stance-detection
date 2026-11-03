"""
Adjusted Classify and Count (ACC) correction of stance prevalence.

The misclassification matrix M[i, j] = P(pred = j | true = i) is estimated from the
stance model's predictions on the VoteStance test set (matched pairs). Corrected
prevalences p solve  min ||M^T p - p_hat||^2  s.t. sum(p) = 1, 0 <= p <= 1,
where p_hat are the raw predicted proportions on the corpus. Bootstrap CIs resample
the test set.

Usage:
  python -m pipeline.acc_correction \
      --test_predictions results/votestance/ours_seed42_test_predictions.csv \
      --corpus_predictions results/election/stance_predictions.csv \
      --claims data/votestance/claims.csv --out results/election/acc_results.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix, f1_score

CLASSES = ["Support", "Refute", "Neutral"]


def misclassification_matrix(cm):
    rows = cm.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1
    return cm / rows


def acc_correct(M, counts):
    p_hat = counts / counts.sum()
    res = minimize(
        lambda p: np.sum((M.T @ p - p_hat) ** 2),
        x0=p_hat,
        method="SLSQP",
        constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1}],
        bounds=[(0.0, 1.0)] * len(CLASSES),
    )
    return res.x


def bootstrap(y_true, y_pred, counts, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    est = np.zeros((n, len(CLASSES)))
    for b in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        cm = confusion_matrix(y_true[idx], y_pred[idx], labels=CLASSES).astype(float)
        est[b] = acc_correct(misclassification_matrix(cm), counts)
    return np.percentile(est, 2.5, axis=0), np.percentile(est, 97.5, axis=0)


def counts_of(labels):
    return np.array([(labels == c).sum() for c in CLASSES], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_predictions", required=True,
                        help="csv with columns label, pred (stance.train --predictions_file on VoteStance test)")
    parser.add_argument("--corpus_predictions", required=True, help="csv from pipeline.predict_stance")
    parser.add_argument("--claims", default="data/votestance/claims.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    test = pd.read_csv(args.test_predictions)
    y_true, y_pred = test["label"].to_numpy(), test["pred"].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).astype(float)
    M = misclassification_matrix(cm)
    print(f"test macro-F1 of the stance model: {f1_score(y_true, y_pred, labels=CLASSES, average='macro'):.4f}")
    print("misclassification matrix (rows = true, cols = predicted):", CLASSES)
    print(M.round(3))

    corpus = pd.read_csv(args.corpus_predictions)
    corpus = corpus[corpus["stance_label"].isin(CLASSES)]
    corpus = corpus.merge(pd.read_csv(args.claims)[["claim_id", "topic"]], on="claim_id", how="left")

    def summarize(labels):
        counts = counts_of(labels)
        return {"n": int(counts.sum()), "counts": counts.tolist(),
                "raw": (counts / counts.sum()).tolist(), "corrected": acc_correct(M, counts).tolist()}

    overall = summarize(corpus["stance_label"].to_numpy())
    lo, hi = bootstrap(y_true, y_pred, np.array(overall["counts"]))
    overall["corrected_ci95"] = [lo.tolist(), hi.tolist()]
    topics = {t: summarize(g["stance_label"].to_numpy()) for t, g in corpus.groupby("topic")}

    print(f"\nmatched pairs: {overall['n']} ({corpus['tweet_id'].nunique()} unique tweets)")
    print(f"{'':10s}{'count':>8s}{'raw':>8s}{'corrected':>11s}{'95% CI':>18s}")
    for i, c in enumerate(CLASSES):
        print(f"{c:10s}{overall['counts'][i]:8.0f}{overall['raw'][i] * 100:7.1f}%{overall['corrected'][i] * 100:10.1f}%"
              f"   [{lo[i] * 100:5.1f}, {hi[i] * 100:5.1f}]")
    for t, r in sorted(topics.items(), key=lambda kv: -kv[1]["n"]):
        print(f"{t} (n={r['n']}): " + ", ".join(f"{c} {v * 100:.0f}%" for c, v in zip(CLASSES, r["corrected"])))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"classes": CLASSES, "confusion_matrix": cm.tolist(), "misclassification_matrix": M.tolist(),
                   "unique_tweets": int(corpus["tweet_id"].nunique()), "overall": overall, "topics": topics},
                  f, indent=2)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
