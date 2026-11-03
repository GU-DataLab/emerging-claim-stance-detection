"""
Train and evaluate a stance detection model.

Variants:
  ours       two-way cross-attention + supervised contrastive loss
  no_xattn   w/o XAttn  (mean-pooled PLM + contrastive loss)
  no_cl      w/o CL     (two-way cross-attention, cross-entropy only)
  baseline   fine-tuned PLM (mean-pooled PLM, cross-entropy only)

Example:
  python -m stance.train --data_source stanceosaurus --variant ours \
      --model_name vinai/bertweet-large \
      --train_path data/stanceosaurus/train_3.csv --dev_path data/stanceosaurus/dev_3.csv \
      --test_path data/stanceosaurus/test_3.csv --seed 42 --output_file results/stanceosaurus.jsonl
"""
import argparse
import copy
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from pytorch_metric_learning import losses
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from tqdm import tqdm

from stance.data import ID2LABEL, STANCE_LABELS, describe, load_splits, make_loader
from stance.modeling import BaselineModel, CrossAttentionModel, StanceConfig, load_model

VARIANTS = {
    # name: (cross-attention, contrastive loss)
    "ours": (True, True),
    "no_xattn": (False, True),
    "no_cl": (True, False),
    "baseline": (False, False),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inverse_frequency_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = len(labels) / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float, device=device)


def train_epoch(model, loader, optimizer, ce_loss, cl_loss, alpha, beta, device):
    model.train()
    totals = {"loss": 0.0, "ce": 0.0, "cl": 0.0}
    for batch in tqdm(loader, leave=False, desc="train"):
        labels = batch["labels"].to(device)
        out = model(claims=batch["claims"], posts=batch["posts"])
        ce = ce_loss(out.logits, labels)
        if cl_loss is not None:
            cl = cl_loss(out.projections, labels)
            loss = alpha * ce + beta * cl
            totals["cl"] += cl.item()
        else:
            loss = ce
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        totals["loss"] += loss.item()
        totals["ce"] += ce.item()
    return {k: v / len(loader) for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, return_embeddings=False):
    model.eval()
    preds, labels, embs = [], [], []
    for batch in loader:
        out = model(claims=batch["claims"], posts=batch["posts"])
        preds.extend(out.logits.argmax(dim=1).cpu().tolist())
        labels.extend(batch["labels"].tolist())
        if return_embeddings:
            embs.append(out.embeddings.float().cpu())
    result = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "preds": preds,
        "labels": labels,
    }
    if return_embeddings:
        result["embeddings"] = torch.cat(embs).numpy()
    return result


def build_model(args, device):
    use_xattn, use_cl = VARIANTS[args.variant]
    config = StanceConfig(
        model_name=args.model_name,
        num_classes=len(STANCE_LABELS),
        label2id=STANCE_LABELS,
        id2label=ID2LABEL,
        cross_attn_type="two-way",
        num_heads=args.num_heads,
        pool_hidden=args.pool_hidden,
        use_cl=use_cl,
    )
    cls = CrossAttentionModel if use_xattn else BaselineModel
    return cls.from_plm(config).to(device)


def train(args, model, train_loader, dev_loader, device):
    optimizer = AdamW(
        [
            {"params": model.plm.parameters(), "lr": args.lr_plm},
            {"params": model.head_parameters(), "lr": args.lr_head},
        ],
        weight_decay=0.01,
    )
    cl_loss = losses.SupConLoss(temperature=args.temperature) if model.config.use_cl else None
    # warm-up epoch uses inverse-frequency class weights; later epochs use plain CE
    train_labels = train_loader.dataset.df["label_id"].to_numpy()
    weighted_ce = nn.CrossEntropyLoss(weight=inverse_frequency_weights(train_labels, len(STANCE_LABELS), device))
    plain_ce = nn.CrossEntropyLoss()

    best_f1, best_epoch, best_state, bad_epochs = -1.0, -1, None, 0
    for epoch in range(args.epochs):
        warmup = epoch == 0
        # freeze the PLM during the first epoch so the new layers can warm up
        for p in model.plm.parameters():
            p.requires_grad = not warmup
        stats = train_epoch(
            model, train_loader, optimizer, weighted_ce if warmup else plain_ce,
            cl_loss, args.cl_alpha, args.cl_beta, device,
        )
        dev = evaluate(model, dev_loader, device)
        print(f"epoch {epoch + 1}/{args.epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in stats.items())
              + f", dev_macro_f1={dev['macro_f1']:.4f}")

        if dev["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = dev["macro_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    print(f"best dev macro-F1 {best_f1:.4f} at epoch {best_epoch + 1}")
    return best_f1, best_epoch + 1


def main():
    parser = argparse.ArgumentParser(description="train / evaluate a claim-based stance detection model")
    parser.add_argument("--data_source", required=True, choices=["stanceosaurus", "ezstance", "votestance"])
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--dev_path", default=None, help="not used for votestance (dev is split from train)")
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--variant", default="ours", choices=list(VARIANTS))
    parser.add_argument("--model_name", default="vinai/bertweet-large")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr_plm", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--cl_alpha", type=float, default=0.7, help="weight of the cross-entropy loss")
    parser.add_argument("--cl_beta", type=float, default=0.3, help="weight of the contrastive loss")
    parser.add_argument("--temperature", type=float, default=0.1, help="SupCon temperature")
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--pool_hidden", action="store_true",
                        help="add the mean-pooled PLM output to the cross-attention representation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_model_path", default=None)
    parser.add_argument("--output_file", default=None, help="append a JSON line with the run results")
    parser.add_argument("--predictions_file", default=None, help="write test predictions (csv)")
    parser.add_argument("--embeddings_file", default=None, help="save test embeddings + labels (.npz) for t-SNE")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--load_model_path", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, dev_df, test_df = load_splits(args.data_source, args.train_path, args.dev_path, args.test_path)
    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        describe(name, df)
    train_loader = make_loader(train_df, args.batch_size, shuffle=True)
    dev_loader = make_loader(dev_df, args.batch_size, shuffle=False)

    if args.eval_only:
        model = load_model(args.load_model_path, device=device)
        best_dev_f1, best_epoch = evaluate(model, dev_loader, device)["macro_f1"], None
    else:
        model = build_model(args, device)
        best_dev_f1, best_epoch = train(args, model, train_loader, dev_loader, device)
        if args.save_model_path:
            model.save_pretrained(args.save_model_path)
            print(f"saved model to {args.save_model_path}")

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **{k: v for k, v in vars(args).items() if not k.endswith("_file")},
        "best_epoch": best_epoch,
        "dev_macro_f1": best_dev_f1,
    }

    if test_df is not None:
        test = evaluate(model, make_loader(test_df, args.batch_size, shuffle=False), device,
                        return_embeddings=args.embeddings_file is not None)
        names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
        report = classification_report(test["labels"], test["preds"], labels=list(range(len(names))),
                                       target_names=names, digits=4, output_dict=True, zero_division=0)
        print(classification_report(test["labels"], test["preds"], labels=list(range(len(names))),
                                    target_names=names, digits=4, zero_division=0))
        record.update({
            "test_accuracy": test["accuracy"],
            "test_macro_f1": test["macro_f1"],
            "test_f1_per_class": {n: report[n]["f1-score"] for n in names},
        })
        if args.predictions_file:
            os.makedirs(os.path.dirname(args.predictions_file) or ".", exist_ok=True)
            out = test_df[["claim", "post", "label"]].copy()
            out["pred"] = [ID2LABEL[p] for p in test["preds"]]
            out.to_csv(args.predictions_file, index=False)
        if args.embeddings_file:
            os.makedirs(os.path.dirname(args.embeddings_file) or ".", exist_ok=True)
            np.savez(args.embeddings_file, embeddings=test["embeddings"],
                     labels=np.array(test["labels"]), preds=np.array(test["preds"]))

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
