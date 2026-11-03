#!/usr/bin/env bash
set -euo pipefail
DATA=data/stanceosaurus
for variant in baseline ours; do
  python -m stance.train --data_source stanceosaurus --variant $variant --model_name vinai/bertweet-large \
    --train_path $DATA/train_3.csv --dev_path $DATA/dev_3.csv --test_path $DATA/test_3_dedup.csv \
    --seed 314159 --embeddings_file results/tsne/${variant}.npz --output_file results/tsne/runs.jsonl
  python -m experiments.tsne --embeddings results/tsne/${variant}.npz --out figures/tsne_${variant}.pdf
done
