#!/usr/bin/env bash
# Top attributed words of the in-domain vs cross-domain models (run run_in_vs_cross_domain.sh first).
set -euo pipefail
DATA=data/ez-stance/in_vs_cross
for d in politic rights_domain sports_domain; do
  for setting in in_domain cross_domain; do
    python -m experiments.word_attribution --model_path models/in_vs_cross/${setting}_${d} \
      --data_source ezstance --test_path $DATA/$d/test.csv \
      --output_file results/attributions/${setting}_${d}.json
  done
done
