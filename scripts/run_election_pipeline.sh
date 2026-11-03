#!/usr/bin/env bash
# Two-stage pipeline on the 2024 election corpus: claim matching -> stance detection ->
# ACC-corrected stance prevalence overall and per claim topic.
set -euo pipefail
CORPUS=${CORPUS:-data/votestance/corpus.csv}
SEED=${SEED:-42}                                   # which run_votestance.sh checkpoint to use
STANCE_MODEL=models/votestance/ours_seed${SEED}
OUT=results/election

python -m pipeline.train_claim_matcher --finetune_encoder --threshold 0.6 \
  --save_dir models/claim_matcher --results_file $OUT/claim_matching.json
python -m pipeline.match_corpus --matcher models/claim_matcher --corpus "$CORPUS" \
  --out $OUT/claim_tweet_matches.csv
python -m pipeline.predict_stance --model $STANCE_MODEL --matches $OUT/claim_tweet_matches.csv \
  --corpus "$CORPUS" --out $OUT/stance_predictions.csv
python -m pipeline.acc_correction \
  --test_predictions results/votestance/ours_seed${SEED}_test_predictions.csv \
  --corpus_predictions $OUT/stance_predictions.csv --out $OUT/acc_results.json
python -m pipeline.plot_topics --acc $OUT/acc_results.json --out figures/stance_by_topic.pdf
