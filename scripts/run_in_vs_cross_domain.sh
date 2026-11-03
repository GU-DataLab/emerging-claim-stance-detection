#!/usr/bin/env bash
# EZ-STANCE in-domain vs cross-domain training with a fixed training budget 
set -euo pipefail
DATA=data/ez-stance/in_vs_cross
OUT=results/in_vs_cross_domain.jsonl
DOMAINS=(covid19_domain world_event_domain education_and_culture_domain consumption_and_entertainment_domain
         sports_domain rights_domain environmental_protection_domain politic)

for d in "${DOMAINS[@]}"; do
  for setting in in_domain cross_domain; do
    python -m stance.train --data_source ezstance --variant ours --model_name vinai/bertweet-large \
      --train_path $DATA/$d/train_${setting}.csv --dev_path $DATA/$d/val_${setting}.csv \
      --test_path $DATA/$d/test.csv --epochs 5 --seed 42 \
      --save_model_path models/in_vs_cross/${setting}_${d} --output_file $OUT
  done
done
python -m experiments.aggregate_results $OUT
