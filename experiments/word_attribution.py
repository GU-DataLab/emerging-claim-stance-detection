"""
Word attributions for trained stance models.

Runs Layer Integrated Gradients (transformers-interpret) for the predicted class of
every test instance, averages each token's attribution over the test set, and
writes the top-k positive and negative tokens.

Usage:
  python -m experiments.word_attribution \
      --model_path models/in_vs_cross/cross_domain_politic \
      --data_source ezstance --test_path data/ez-stance/in_vs_cross/politic/test.csv \
      --output_file results/attributions/cross_domain_politic.json
"""
import argparse
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers_interpret import SequenceClassificationExplainer

from stance.data import read_ezstance, read_stanceosaurus, read_votestance
from stance.modeling import load_model

READERS = {"ezstance": read_ezstance, "stanceosaurus": read_stanceosaurus, "votestance": read_votestance}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_source", choices=list(READERS), default="ezstance")
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_path, device=device)
    tok = model.tokenizer
    explainer = SequenceClassificationExplainer(model, tok)
    special = set(tok.all_special_tokens)

    test_df = READERS[args.data_source](args.test_path)
    scores = defaultdict(list)
    for claim, post in tqdm(zip(test_df["claim"], test_df["post"]), total=len(test_df)):
        try:
            attributions = explainer(text=f"{claim} {tok.sep_token} {post}")
        except Exception as e:
            print(f"skipping instance: {e}")
            continue
        for word, score in attributions:
            if word in special or len(word.strip()) < 2:
                continue
            scores[word].append(score)

    avg = sorted(((w, sum(s) / len(s)) for w, s in scores.items()), key=lambda x: x[1], reverse=True)
    result = {
        "model_path": args.model_path,
        "test_path": args.test_path,
        "top_positive": [(w, s) for w, s in avg if s > 0][: args.k],
        "top_negative": [(w, s) for w, s in reversed(avg) if s < 0][: args.k],
    }
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=2)
    print("top positive:", [w for w, _ in result["top_positive"][:10]])
    print("top negative:", [w for w, _ in result["top_negative"][:10]])


if __name__ == "__main__":
    main()
