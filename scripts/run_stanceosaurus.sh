#!/usr/bin/env bash
# Stanceosaurus: our model, ablations and fine-tuned baseline
set -euo pipefail
DATA=data/stanceosaurus
OUT=results/stanceosaurus.jsonl
SEEDS=(4242 679 6866 777 314159)
ENCODERS=(vinai/bertweet-large facebook/bart-large-mnli)
VARIANTS=(baseline ours no_xattn no_cl)

for enc in "${ENCODERS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      python -m stance.train --data_source stanceosaurus --variant "$variant" --model_name "$enc" \
        --train_path $DATA/train_3.csv --dev_path $DATA/dev_3.csv --test_path $DATA/test_3_dedup.csv \
        --seed "$seed" --output_file $OUT
    done
  done
done
python -m experiments.aggregate_results $OUT
