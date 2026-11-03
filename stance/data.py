"""
Dataset loaders. Every loader returns dataframes with columns
`claim, post, label` (label in STANCE_LABELS) and wraps them in DataLoaders.
"""
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

STANCE_LABELS = {"Refute": 0, "Support": 1, "Neutral": 2}
ID2LABEL = {v: k for k, v in STANCE_LABELS.items()}

# dataset-specific label names -> unified labels
LABEL_MAP = {
    "favor": "Support", "supporting": "Support", "support": "Support",
    "against": "Refute", "refuting": "Refute", "refute": "Refute",
    "none": "Neutral", "other": "Neutral", "neutral": "Neutral",
}


def standardize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].astype(str).str.lower().map(LABEL_MAP)
    unmapped = df["label"].isna().sum()
    if unmapped:
        print(f"warning: dropping {unmapped} rows with unknown labels")
        df = df.dropna(subset=["label"])
    df["label_id"] = df["label"].map(STANCE_LABELS).astype(int)
    return df.reset_index(drop=True)


class StanceDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["claim"], row["post"], int(row["label_id"])


def collate_fn(batch):
    claims, posts, labels = zip(*batch)
    return {"claims": list(claims), "posts": list(posts), "labels": torch.tensor(labels, dtype=torch.long)}


# --- per-dataset readers -------------------------------------------------------

def read_stanceosaurus(path: str) -> pd.DataFrame:
    """CSV produced by data/prepare/prepare_stanceosaurus.py"""
    df = pd.read_csv(path)
    df = df.rename(columns={"cleaned_tweet_text": "post"})[["claim", "post", "label"]]
    return standardize_labels(df.dropna())


def read_ezstance(path: str) -> pd.DataFrame:
    """Original EZ-STANCE csv (subtaskB/claim) or splits from make_ezstance_in_cross_splits.py"""
    df = pd.read_csv(path)
    df = df.rename(columns={"Target 1": "claim", "Text": "post", "Stance 1": "label"})[["claim", "post", "label"]]
    return standardize_labels(df.dropna())


def read_votestance(path: str) -> pd.DataFrame:
    """VoteStance csv; stance is only defined for pairs annotated as `Match`."""
    df = pd.read_csv(path)
    df = df[df["match_label"].str.lower() == "match"]
    df = df.rename(columns={"tweet": "post", "stance_label": "label"})[["claim", "post", "label"]]
    return standardize_labels(df.dropna())


def load_splits(data_source: str, train_path: str, dev_path: str = None, test_path: str = None):
    """Return (train_df, dev_df, test_df)."""
    if data_source == "stanceosaurus":
        reader = read_stanceosaurus
    elif data_source == "ezstance":
        reader = read_ezstance
    elif data_source == "votestance":
        # VoteStance has no official dev split: hold out 15% of the (matched) training
        # pairs, stratified by stance, with a fixed split seed.
        train = read_votestance(train_path)
        train, dev = train_test_split(train, test_size=0.15, random_state=42, stratify=train["label"])
        test = read_votestance(test_path) if test_path else None
        return train.reset_index(drop=True), dev.reset_index(drop=True), test
    else:
        raise ValueError(f"unknown data source: {data_source}")

    train = reader(train_path)
    dev = reader(dev_path) if dev_path else None
    test = reader(test_path) if test_path else None
    return train, dev, test


def make_loader(df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(StanceDataset(df), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def describe(name: str, df: pd.DataFrame):
    if df is not None:
        print(f"{name}: {len(df)} pairs, {df['claim'].nunique()} claims, labels={df['label'].value_counts().to_dict()}")
