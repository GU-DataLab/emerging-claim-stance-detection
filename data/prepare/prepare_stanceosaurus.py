"""
Build the 3-way Stanceosaurus splits from the original release
(https://github.com/JonathanQZheng/Stanceosaurus).

Only root tweets (no replies) are used. Labels are mapped to 3 classes:
  supporting / discussing+leaning supporting -> supporting
  refuting   / discussing+leaning refuting   -> refuting
  querying, irrelevant, discussing (no leaning) -> other
The original claim-disjoint train/dev/test split is kept.

Usage:
  python -m data.prepare.prepare_stanceosaurus --raw_dir path/to/Stanceosaurus/data --out_dir data/stanceosaurus
  -> data/stanceosaurus/{train,dev,test}_3.csv
"""
import argparse
import glob
import json
import os

import pandas as pd

from stance.text import clean_tweet


def load_root_tweets(raw_dir: str) -> pd.DataFrame:
    rows = []
    for split in ["train", "dev", "test"]:
        split_dir = os.path.join(raw_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for source in sorted(os.listdir(split_dir)):
            for path in sorted(glob.glob(os.path.join(split_dir, source, "*.json"))):
                with open(path, encoding="utf-8") as f:
                    items = json.load(f)
                for item in items:
                    if "claim" not in item or "root_tweet" not in item:
                        continue
                    root = item["root_tweet"]
                    rows.append({
                        "split": split,
                        "source": item.get("source", source),
                        "claim_id": item.get("id"),
                        "claim": item["claim"],
                        "tweet_id": root.get("id"),
                        "tweet_text": root.get("text"),
                        "stance": root.get("stance", "").lower(),
                        "leaning": root.get("leaning", "").lower(),
                    })
    return pd.DataFrame(rows)


def to_3way(stance: str, leaning: str):
    if stance in ("supporting", "refuting"):
        return stance
    if stance == "discussing":
        return leaning if leaning in ("supporting", "refuting") else "other"
    if stance in ("querying", "irrelevant"):
        return "other"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="Stanceosaurus data dir containing train/ dev/ test/")
    parser.add_argument("--out_dir", default="data/stanceosaurus")
    args = parser.parse_args()

    df = load_root_tweets(args.raw_dir)
    if df.empty:
        raise SystemExit(f"no Stanceosaurus tweets found in {args.raw_dir}: expected train/ dev/ test/ "
                         "folders from the Stanceosaurus release's data/ directory (see data/README.md)")
    print(f"loaded {len(df)} root tweets")

    cleaned = df["tweet_text"].apply(lambda t: clean_tweet(t) if pd.notna(t) else None)
    df["cleaned_tweet_text"] = cleaned.apply(lambda r: r[0] if r else None)
    df["norm_text"] = cleaned.apply(lambda r: r[1] if r else None)
    df = df.dropna(subset=["cleaned_tweet_text", "norm_text"])
    n = len(df)
    df = df.drop_duplicates(subset=["norm_text"])
    print(f"removed {n - len(df)} duplicate tweets")

    df["label"] = [to_3way(s, l) for s, l in zip(df["stance"], df["leaning"])]
    df = df.dropna(subset=["label"])

    os.makedirs(args.out_dir, exist_ok=True)
    for split in ["train", "dev", "test"]:
        part = df[df["split"] == split]
        path = os.path.join(args.out_dir, f"{split}_3.csv")
        part.to_csv(path, index=False)
        print(f"{split}: {len(part)} pairs, {part['claim_id'].nunique()} claims, "
              f"{part['label'].value_counts().to_dict()} -> {path}")


if __name__ == "__main__":
    main()
