#!/usr/bin/env bash
# VoteStance stance detection (matched pairs): BERTweet baseline vs our model
set -euo pipefail
DATA=data/votestance/with_text
OUT=results/votestance.jsonl
SEEDS=(42 1234 666 777 314159)
for seed in "${SEEDS[@]}"; do
  python -m stance.train --data_source votestance --variant baseline --model_name vinai/bertweet-large \
    --train_path $DATA/train.csv --test_path $DATA/test.csv --batch_size 8 --seed "$seed" --output_file $OUT
  python -m stance.train --data_source votestance --variant ours --model_name vinai/bertweet-large \
    --train_path $DATA/train.csv --test_path $DATA/test.csv --batch_size 8 --pool_hidden --seed "$seed" \
    --save_model_path models/votestance/ours_seed${seed} \
    --predictions_file results/votestance/ours_seed${seed}_test_predictions.csv --output_file $OUT
done
python -m experiments.aggregate_results $OUT
