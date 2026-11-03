"""Bi-encoder claim-matching model: SBERT encoder + MLP over [c, t, |c-t|, c*t]."""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class ClaimMatcher(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", dropout=0.3, finetune_encoder=True):
        super().__init__()
        self.model_name = model_name
        self.dropout = dropout
        self.finetune_encoder = finetune_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = finetune_encoder
        hid = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid * 4, hid),
            nn.LayerNorm(hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, texts, max_length=128):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return mean_pooling(self.encoder(**inputs).last_hidden_state, inputs["attention_mask"])

    def forward(self, claims, tweets):
        c, t = self.encode(claims), self.encode(tweets)
        features = torch.cat([c, t, torch.abs(c - t), c * t], dim=1)
        return self.classifier(features).squeeze(-1)  # logits

    @torch.no_grad()
    def predict_proba(self, claims, tweets, batch_size=64):
        self.eval()
        probs = []
        for i in range(0, len(claims), batch_size):
            probs.extend(torch.sigmoid(self(claims[i:i + batch_size], tweets[i:i + batch_size])).cpu().tolist())
        return probs

    def save(self, save_dir, extra=None):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "model_state_dict.pt"))
        self.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        cfg = {"model_name": self.model_name, "dropout": self.dropout,
               "finetune_encoder": self.finetune_encoder, **(extra or {})}
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, save_dir, device="cpu"):
        with open(os.path.join(save_dir, "config.json")) as f:
            cfg = json.load(f)
        model = cls(cfg["model_name"], cfg["dropout"], cfg["finetune_encoder"])
        model.load_state_dict(torch.load(os.path.join(save_dir, "model_state_dict.pt"), map_location=device))
        return model.to(device).eval(), cfg


def info_nce_loss(claim_emb, tweet_emb, temperature=0.07):
    """In-batch InfoNCE: each claim's matched tweet is the positive, other tweets are negatives."""
    logits = F.normalize(claim_emb, dim=1) @ F.normalize(tweet_emb, dim=1).T / temperature
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
