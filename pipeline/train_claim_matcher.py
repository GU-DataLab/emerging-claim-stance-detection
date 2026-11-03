"""
Train and evaluate the claim-matching model.

1. Add easy negatives (low-similarity claim-tweet pairs) to the VoteStance training pairs.
2. Contrastive warm-up of the encoder with in-batch InfoNCE on matched pairs (1 epoch).
3. Train the MLP matcher with class-weighted BCE.
   - Leave-one-claim-out cross-validation reports out-of-fold F1 over a grid of
     thresholds (used to choose --threshold).
   - A final model is trained on all training pairs, evaluated on the test set and saved.
4. Cosine-similarity baseline (pretrained encoder, same threshold grid).

Usage:
  python -m pipeline.train_claim_matcher --save_dir models/claim_matcher --threshold 0.6
"""
import argparse
import copy
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from pipeline.claim_matcher import ClaimMatcher, info_nce_loss, mean_pooling

THRESHOLDS = np.round(np.arange(0.3, 0.91, 0.05), 2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pairs(train_path, negatives_path=None):
    df = pd.read_csv(train_path)
    df["y"] = (df["match_label"] == "Match").astype(int)
    df = df[["claim_id", "claim", "tweet", "y"]]
    if negatives_path:
        neg = pd.read_csv(negatives_path).dropna(subset=["claim_id"])
        neg = neg.assign(y=0, claim_id=neg["claim_id"].astype(int))[["claim_id", "claim", "tweet", "y"]]
        df = pd.concat([df, neg], ignore_index=True)
    return df


def pair_loader(df, batch_size, shuffle):
    rows = list(zip(df["claim"], df["tweet"], df["y"]))
    collate = lambda b: (list(x[0] for x in b), list(x[1] for x in b), torch.tensor([x[2] for x in b], dtype=torch.float))
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def scores(y, prob, threshold):
    pred = (np.asarray(prob) > threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f, "macro_f1": f1_score(y, pred, average="macro", zero_division=0)}


def contrastive_warmup(model_name, pos_df, args, device):
    """Fine-tune the encoder with InfoNCE on matched pairs; returns its state dict."""
    encoder = AutoModel.from_pretrained(model_name).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    loader = DataLoader(list(zip(pos_df["claim"], pos_df["tweet"])), batch_size=args.cl_batch_size, shuffle=True)
    opt = AdamW(encoder.parameters(), lr=args.cl_lr)
    encoder.train()
    for _ in range(args.cl_epochs):
        for claims, tweets in tqdm(loader, desc="infonce"):
            enc = lambda x: tok(list(x), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            c, t = enc(claims), enc(tweets)
            loss = info_nce_loss(mean_pooling(encoder(**c).last_hidden_state, c["attention_mask"]),
                                 mean_pooling(encoder(**t).last_hidden_state, t["attention_mask"]),
                                 args.cl_temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return copy.deepcopy(encoder.state_dict())


def fit(train_df, args, device, encoder_state, select_df=None):
    """Train a matcher. Early stopping on F1 of `select_df` (held-out claim) or, if None, the training data."""
    model = ClaimMatcher(args.model_name, args.dropout, args.finetune_encoder).to(device)
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state)
    groups = [{"params": model.classifier.parameters(), "lr": args.lr_classifier}]
    if args.finetune_encoder:
        groups.append({"params": model.encoder.parameters(), "lr": args.lr_encoder})
    opt = AdamW(groups, weight_decay=0.01)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    y = train_df["y"].to_numpy()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y == 0).sum() / (y == 1).sum()], device=device))
    loader = pair_loader(train_df, args.batch_size, shuffle=True)

    best_f1, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        train_probs, train_y = [], []
        for claims, tweets, labels in tqdm(loader, leave=False, desc=f"epoch {epoch + 1}"):
            labels = labels.to(device)
            logits = model(claims, tweets)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
            train_y.extend(labels.cpu().tolist())

        if select_df is not None:
            f1 = scores(select_df["y"], model.predict_proba(select_df["claim"].tolist(), select_df["tweet"].tolist()),
                        args.threshold)["f1"]
        else:
            f1 = scores(train_y, train_probs, args.threshold)["f1"]
        sched.step(f1)
        if f1 > best_f1 + 1e-3:
            best_f1, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model


def cross_validate(df, args, device, encoder_state):
    """Leave-one-claim-out; returns out-of-fold match probabilities."""
    oof = np.zeros(len(df))
    for claim_id in df["claim_id"].unique():
        held = df["claim_id"] == claim_id
        print(f"cv fold: held-out claim {claim_id}")
        model = fit(df[~held], args, device, encoder_state, select_df=df[held])
        oof[held.to_numpy()] = model.predict_proba(df.loc[held, "claim"].tolist(), df.loc[held, "tweet"].tolist())
    return oof


def cosine_baseline(model_name, df, device):
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model_name, device=str(device))
    c = st.encode(df["claim"].tolist(), convert_to_tensor=True)
    t = st.encode(df["tweet"].tolist(), convert_to_tensor=True)
    return torch.nn.functional.cosine_similarity(c, t, dim=1).cpu().numpy()


def threshold_table(y, prob):
    return {float(th): scores(y, prob, th) for th in THRESHOLDS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/votestance/with_text/train.csv")
    parser.add_argument("--negatives_path", default="data/votestance/with_text/easy_negatives.csv")
    parser.add_argument("--test_path", default="data/votestance/with_text/test.csv")
    parser.add_argument("--model_name", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--save_dir", default="models/claim_matcher")
    parser.add_argument("--results_file", default="results/claim_matching.json")
    parser.add_argument("--threshold", type=float, default=0.6, help="match-probability threshold")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--lr_classifier", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--finetune_encoder", action="store_true")
    parser.add_argument("--no_infonce", action="store_true", help="skip the InfoNCE warm-up")
    parser.add_argument("--cl_epochs", type=int, default=1)
    parser.add_argument("--cl_batch_size", type=int, default=16)
    parser.add_argument("--cl_lr", type=float, default=1e-4)
    parser.add_argument("--cl_temperature", type=float, default=0.07)
    parser.add_argument("--skip_cv", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_pairs(args.train_path, args.negatives_path)
    test = load_pairs(args.test_path)
    print(f"train pairs: {len(df)} ({df['y'].sum()} matches), test pairs: {len(test)} ({test['y'].sum()} matches)")

    encoder_state = None if args.no_infonce else contrastive_warmup(args.model_name, df[df["y"] == 1], args, device)
    results = {"args": vars(args)}

    if not args.skip_cv:
        oof = cross_validate(df, args, device, encoder_state)
        results["cv_out_of_fold"] = threshold_table(df["y"], oof)
        best = max(results["cv_out_of_fold"].items(), key=lambda kv: kv[1]["macro_f1"])
        print(f"cv: best threshold {best[0]} (macro-F1 {best[1]['macro_f1']:.4f})")

    model = fit(df, args, device, encoder_state)
    test_prob = model.predict_proba(test["claim"].tolist(), test["tweet"].tolist())
    results["test"] = scores(test["y"], test_prob, args.threshold)
    results["cosine_baseline_test"] = threshold_table(test["y"], cosine_baseline(args.model_name, test, device))
    print(f"test @ {args.threshold}: {results['test']}")

    model.save(args.save_dir, extra={"threshold": args.threshold})
    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved model to {args.save_dir} and results to {args.results_file}")


if __name__ == "__main__":
    main()
