#!/usr/bin/env bash
# EZ-STANCE claim subset, leave-one-domain-out: our model, ablations and fine-tuned baseline
set -euo pipefail
DATA=data/ez-stance/subtaskB/claim
OUT=results/ezstance.jsonl
DOMAINS=(consumption_and_entertainment_domain covid19_domain world_event_domain sports_domain
         rights_domain politic education_and_culture_domain environmental_protection_domain)
ENCODERS=(vinai/bertweet-large facebook/bart-large-mnli)
VARIANTS=(baseline ours no_xattn no_cl)

for enc in "${ENCODERS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    for d in "${DOMAINS[@]}"; do
      python -m stance.train --data_source ezstance --variant "$variant" --model_name "$enc" \
        --train_path $DATA/$d/raw_train_all_onecol.csv --dev_path $DATA/$d/raw_val_all_onecol.csv \
        --test_path $DATA/$d/test_dedup.csv --seed 42 --output_file $OUT
    done
  done
done
python -m experiments.aggregate_results $OUT
