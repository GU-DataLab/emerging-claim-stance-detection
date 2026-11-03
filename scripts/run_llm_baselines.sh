#!/usr/bin/env bash
# Zero-/5-shot LLM baselines on Stanceosaurus: GPT-4o (OpenAI API) and Llama-3.1-8B (ollama).
set -euo pipefail
DATA=data/stanceosaurus
COMMON="--data_source stanceosaurus --train_path $DATA/train_3.csv --test_path $DATA/test_3_dedup.csv"

for mode in zero-shot few-shot; do
  python -m experiments.llm_baseline --backend openai --model_id gpt-4o-2024-08-06 $COMMON \
    --mode $mode --k 5 --output_file results/llm/gpt-4o_${mode}.json
  python -m experiments.llm_baseline --backend ollama --model_id llama3.1:8b-instruct-fp16 $COMMON \
    --mode $mode --k 5 --output_file results/llm/llama3.1-8b_${mode}.json
done
