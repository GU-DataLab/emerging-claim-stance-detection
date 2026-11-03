"""
Clean the election tweet corpus.

Input: a directory of JSON-lines files with X API v2 tweet objects
(fetched for `data/votestance/corpus_tweet_ids.txt`). Non-English tweets (API `lang`
field), retweets and tweets with fewer than 5 non-hashtag words are dropped;
mentions/URLs/emojis are removed; near-duplicates (same normalized text) are removed.

Usage:
  python -m pipeline.preprocess_tweets --tweets_dir raw_tweets/ --out data/votestance/corpus.csv
"""
import argparse
import os
import re

from stance.text import clean_tweet, read_tweets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tweets_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--keep_all_languages", action="store_true", help="do not filter on lang == 'en'")
    parser.add_argument("--exclude_user_regex", default=None, help="drop tweets from matching usernames")
    args = parser.parse_args()

    df = read_tweets(args.tweets_dir)
    print(f"loaded {len(df)} tweets")

    if not args.keep_all_languages:
        df = df[df["lang"] == "en"]
        print(f"{len(df)} English tweets")
    if args.exclude_user_regex and "user_username" in df.columns:
        df = df[~df["user_username"].str.match(re.compile(args.exclude_user_regex, re.IGNORECASE))]

    cleaned = df["text"].apply(clean_tweet)
    df = df[cleaned.notna()].copy()
    df["tweet_text"] = cleaned[cleaned.notna()].apply(lambda r: r[0])
    df["norm_text"] = cleaned[cleaned.notna()].apply(lambda r: r[1])
    df = df.drop_duplicates(subset=["norm_text"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"saved {len(df)} tweets to {args.out}")


if __name__ == "__main__":
    main()
