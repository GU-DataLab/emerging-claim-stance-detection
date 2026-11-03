"""
Summarize the JSON-lines written by `stance.train --output_file`.

1. Groups runs by (data source, encoder, variant, test file) and reports the mean and
   standard deviation of macro-F1 and per-class F1 (x100) over seeds.
2. Significance: for each (data source, encoder), compares every variant against the
   control variant (default: baseline) with a one-tailed Welch's t-test
   (H1: variant > control). The samples are all runs of a variant, i.e. the seeds
   (single test file) or the per-domain scores (leave-one-domain-out).

Usage:
  python -m experiments.aggregate_results results/stanceosaurus.jsonl
  python -m experiments.aggregate_results results/ezstance.jsonl --control baseline
"""
import argparse
import json

import pandas as pd
from scipy.stats import ttest_ind

METRICS = ["Support", "Refute", "Neutral", "macro_f1"]


def load_runs(files):
    rows = []
    for path in files:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "test_macro_f1" not in r:
                    continue
                rows.append({
                    "data": r["data_source"],
                    "encoder": r["model_name"].split("/")[-1],
                    "variant": r["variant"],
                    "test": "/".join(r["test_path"].split("/")[-2:]),
                    "seed": r["seed"],
                    "macro_f1": 100 * r["test_macro_f1"],
                    **{k: 100 * v for k, v in r["test_f1_per_class"].items()},
                })
    return pd.DataFrame(rows)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def significance(df, control):
    """One-tailed Welch's t-test of each variant against `control` within (data, encoder)."""
    rows = []
    for (data, encoder), group in df.groupby(["data", "encoder"]):
        base = group[group["variant"] == control]
        if base.empty:
            continue
        for variant, runs in group[group["variant"] != control].groupby("variant"):
            row = {"data": data, "encoder": encoder, "variant": variant, "vs": control,
                   "n": len(runs), "n_control": len(base)}
            for m in METRICS:
                p = ttest_ind(runs[m], base[m], equal_var=False, alternative="greater").pvalue
                row[f"{m}_delta"] = round(runs[m].mean() - base[m].mean(), 2)
                row[f"{m}_p"] = f"{p:.4f}{stars(p)}"
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--control", default="baseline", help="variant the others are tested against")
    args = parser.parse_args()

    df = load_runs(args.files)
    keys = ["data", "encoder", "variant", "test"]
    summary = df.groupby(keys)[METRICS].agg(["mean", "std"]).round(2)
    summary["n_seeds"] = df.groupby(keys)["seed"].nunique()
    print(summary.to_string())

    sig = significance(df, args.control)
    if not sig.empty:
        print(f"\none-tailed Welch's t-test vs. {args.control} (* p<0.05, ** p<0.01, *** p<0.001)")
        print(sig.to_string(index=False))


if __name__ == "__main__":
    main()
