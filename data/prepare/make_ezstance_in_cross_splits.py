"""
In-domain vs. cross-domain splits for EZ-STANCE.

For each domain D, the original subtaskB/claim test file of D is split by claim
(80/10/10, claim-disjoint, stratified by each claim's majority stance) into
train_in_domain / val_in_domain / test. The cross-domain train/val sets are
sampled from the other seven domains (D's original train/val files) to the
same size as the in-domain sets, so both settings have the same training budget.

Usage:
  python -m data.prepare.make_ezstance_in_cross_splits \
      --source_dir data/ez-stance/subtaskB/claim --target_dir data/ez-stance/in_vs_cross
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


def claim_disjoint_split(df, val_frac=0.1, test_frac=0.1, claim_col="Target 1", stance_col="Stance 1"):
    claims = df[claim_col].unique()
    majority = {c: df.loc[df[claim_col] == c, stance_col].value_counts().index[0] for c in claims}
    claims_df = pd.DataFrame({"claim": claims, "majority_stance": [majority[c] for c in claims]})

    train_c, rest_c = train_test_split(
        claims_df, test_size=val_frac + test_frac, stratify=claims_df["majority_stance"], random_state=SEED
    )
    val_c, test_c = train_test_split(
        rest_c, test_size=test_frac / (val_frac + test_frac), stratify=rest_c["majority_stance"], random_state=SEED
    )
    pick = lambda c: df[df[claim_col].isin(c["claim"])].copy()
    return pick(train_c), pick(val_c), pick(test_c)


def sample_cross_domain(source_dir, domain, split, n):
    """Sample n examples from the other domains, roughly balanced across domains."""
    df = pd.read_csv(os.path.join(source_dir, domain, f"raw_{split}_all_onecol.csv"))
    other = df[df["Domain"] != domain]
    if len(other) <= n:
        print(f"warning: only {len(other)} cross-domain samples available for {domain}/{split}")
        return other
    per_domain = n // other["Domain"].nunique() + 1
    return (
        other.groupby("Domain", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), per_domain), random_state=SEED))
        .sample(n=n, random_state=SEED)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="data/ez-stance/subtaskB/claim")
    parser.add_argument("--target_dir", default="data/ez-stance/in_vs_cross")
    args = parser.parse_args()

    domains = sorted(d for d in os.listdir(args.source_dir) if os.path.isdir(os.path.join(args.source_dir, d)))
    for domain in domains:
        out = os.path.join(args.target_dir, domain)
        os.makedirs(out, exist_ok=True)

        test_full = pd.read_csv(os.path.join(args.source_dir, domain, "raw_test_all_onecol.csv"))
        train_in, val_in, test = claim_disjoint_split(test_full)
        train_cross = sample_cross_domain(args.source_dir, domain, "train", len(train_in))
        val_cross = sample_cross_domain(args.source_dir, domain, "val", len(val_in))

        for name, df in [("train_in_domain", train_in), ("val_in_domain", val_in), ("test", test),
                         ("train_cross_domain", train_cross), ("val_cross_domain", val_cross)]:
            df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
        print(f"{domain}: in-domain train/val/test = {len(train_in)}/{len(val_in)}/{len(test)}, "
              f"cross-domain train/val = {len(train_cross)}/{len(val_cross)}")


if __name__ == "__main__":
    main()
