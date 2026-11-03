"""
Attach tweet text to the VoteStance files, which are distributed with tweet IDs only.

Fetch the tweets for the IDs in data/votestance/{train,test,easy_negatives}.csv
(and/or corpus_tweet_ids.txt) with the X API into JSON-lines files, then:

  python -m data.prepare.build_votestance --tweets_dir raw_tweets/ --out_dir data/votestance/with_text

Each tweet is cleaned with stance.text.clean_tweet (the same cleaning used during
annotation). Rows whose tweet can no longer be retrieved are dropped and reported.
"""
import argparse
import os

import pandas as pd

from stance.text import clean_tweet, read_tweets

FILES = ["train", "test", "easy_negatives"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tweets_dir", required=True)
    parser.add_argument("--data_dir", default="data/votestance")
    parser.add_argument("--out_dir", default="data/votestance/with_text")
    args = parser.parse_args()

    raw = read_tweets(args.tweets_dir)
    cleaned = raw["text"].apply(lambda t: (clean_tweet(t) or (None,))[0])
    text = dict(zip(raw["tweet_id"], cleaned))

    os.makedirs(args.out_dir, exist_ok=True)
    for name in FILES:
        df = pd.read_csv(os.path.join(args.data_dir, f"{name}.csv"), dtype={"tweet_id": "Int64"})
        df["tweet"] = df["tweet_id"].map(text)
        missing = df["tweet"].isna()
        df = df[~missing]
        df.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
        print(f"{name}: {len(df)} pairs written, {missing.sum()} dropped (tweet unavailable)")


if __name__ == "__main__":
    main()
