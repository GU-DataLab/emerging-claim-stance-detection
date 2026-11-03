#!/usr/bin/env bash
# Build the benchmark splits
set -euo pipefail
STANCEOSAURUS_RAW=${STANCEOSAURUS_RAW:-data/stanceosaurus}
THRESHOLD=${THRESHOLD:-0.90}

SO=data/stanceosaurus
python -m data.prepare.prepare_stanceosaurus --raw_dir "$STANCEOSAURUS_RAW" --out_dir $SO
python -m data.prepare.dedup_test_claims --data_source stanceosaurus --threshold $THRESHOLD \
  --train $SO/train_3.csv --test $SO/test_3.csv --out $SO/test_3_dedup.csv

EZ=data/ez-stance/subtaskB/claim
[ -d $EZ ] || { echo "missing $EZ: see data/README.md" >&2; exit 1; }
for d in $(ls $EZ); do
  python -m data.prepare.dedup_test_claims --data_source ezstance --threshold $THRESHOLD \
    --train $EZ/$d/raw_train_all_onecol.csv --test $EZ/$d/raw_test_all_onecol.csv --out $EZ/$d/test_dedup.csv
done

python -m data.prepare.make_ezstance_in_cross_splits --source_dir $EZ --target_dir data/ez-stance/in_vs_cross
