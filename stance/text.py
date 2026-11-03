"""Tweet loading and cleaning used for Stanceosaurus preprocessing and the election tweet corpus."""
import glob
import json
import os
import re

import pandas as pd

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002500-\U00002BEF"
    "\U00002702-\U000027B0"
    "\U0001F926-\U0001F937"
    "\U00010000-\U0010FFFF"
    "‍"
    "♀-♂"
    "☀-⭕"
    "⏏"
    "⏩"
    "⌚"
    "️"
    "〰"
    "]+",
    flags=re.UNICODE,
)


def clean_tweet(text: str, min_words: int = 5):
    """
    Returns (clean_text, dedup_key), or None if the tweet should be dropped
    (retweet, or fewer than `min_words` non-hashtag words).
    Mentions are removed, URLs replaced by "HTTPS", emojis stripped.
    """
    if text.strip().startswith("RT @"):
        return None
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "HTTPS", text, flags=re.MULTILINE)
    text = EMOJI_PATTERN.sub("", text)
    text = text.replace("&amp;", "&")
    text = re.sub(r'[^A-Za-z0-9\s.,!?\'"-#&]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len([w for w in text.split() if not w.startswith("#")]) < min_words:
        return None

    # kept as in the original preprocessing: the key is lowercased first, so this
    # substitution is a no-op and URLs remain as "https" in the key
    key = text.lower()
    key = re.sub(r"HTTPS", "", key)
    key = re.sub(r"#\w+", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    return text, key


def read_tweets(tweets_dir: str) -> pd.DataFrame:
    """Read JSON-lines files of X API v2 tweet objects into a flat dataframe."""
    tweets = []
    for path in sorted(glob.glob(os.path.join(tweets_dir, "*.json*"))):
        with open(path) as f:
            for line in f:
                try:
                    tweets.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not tweets:
        raise SystemExit(f"no tweets found in {tweets_dir}: expected JSON-lines files of X API v2 "
                         "tweet objects (see data/README.md)")
    df = pd.json_normalize(tweets, sep="_").rename(columns={"id": "tweet_id"})
    df["tweet_id"] = df["tweet_id"].astype("int64")
    return df
