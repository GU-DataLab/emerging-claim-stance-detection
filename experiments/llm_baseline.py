"""
Zero-/few-shot LLM stance classification.

Backends:
  openai  e.g. --model_id gpt-4o-2024-08-06   (needs OPENAI_API_KEY)
  ollama  e.g. --model_id llama3.1:8b-instruct-fp16   (needs a running ollama server)

In few-shot mode, k labeled examples are sampled once from the training set
(with --seed) and used for every test instance.

Usage:
  python -m experiments.llm_baseline --backend openai --model_id gpt-4o-2024-08-06 \
      --data_source stanceosaurus --train_path data/stanceosaurus/train_3.csv \
      --test_path data/stanceosaurus/test_3.csv --mode few-shot --k 5 \
      --output_file results/llm/gpt4o_5shot.json
"""
import argparse
import json
import os
import random

from pydantic import BaseModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from stance.data import load_splits

LABELS = ["Refute", "Support", "Neutral"]
FALLBACK = "Neutral"  # used when the response cannot be parsed after all retries
RETRY_TEMPERATURES = [0.0, 0.3, 0.7]

SYSTEM_PROMPT = """# Identity

You are an expert in analyzing the stance of a social media post towards a claim. The stance can be one of the following categories:
- Support: The post supports/agrees with/affirms the claim's content
- Refute: The post refutes/disagrees with/contradicts the claim's content
- Neutral: The post is neither supporting nor refuting the claim or is unrelated to the claim

# Instructions

* Analyze the text and the relation between the post and claim to determine the stance
* Output your response as a JSON object with the stance label
* Your response should only contain one of: "Support", "Refute", or "Neutral"

# Examples

{few_shot_examples}

# Task

Determine the stance of the given post towards the claim."""

USER_PROMPT = """Claim: {claim}

Post: {post}"""


class StanceResponse(BaseModel):
    stance: str


def format_examples(examples):
    out = ""
    for i, ex in enumerate(examples, 1):
        out += (
            f'<claim_post_pair id="example-{i}">\nClaim: {ex["claim"]}\nPost: {ex["post"]}\n</claim_post_pair>\n\n'
            f'<assistant_response id="example-{i}">\n{{"stance": "{ex["label"]}"}}\n</assistant_response>\n\n'
        )
    return out


def make_client(backend):
    if backend == "openai":
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        client = OpenAI()

        def call(model_id, system, user, temperature):
            resp = client.responses.parse(
                model=model_id,
                input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                text_format=StanceResponse,
                temperature=temperature,
            )
            return resp.output_parsed.stance
    else:
        from ollama import chat

        def call(model_id, system, user, temperature):
            resp = chat(
                model=model_id,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=StanceResponse.model_json_schema(),
                options={"temperature": temperature},
            )
            return json.loads(resp.message.content).get("stance")

    return call


def classify(call, model_id, system, claim, post):
    """Returns (label, parsed_ok). Retries with increasing temperature on errors."""
    user = USER_PROMPT.format(claim=claim, post=post)
    for temperature in RETRY_TEMPERATURES:
        try:
            stance = call(model_id, system, user, temperature)
        except Exception as e:
            print(f"error: {e}")
            continue
        return (stance, True) if stance in LABELS else (FALLBACK, False)
    return FALLBACK, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "ollama"], required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--data_source", choices=["stanceosaurus", "ezstance", "votestance"], required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--mode", choices=["zero-shot", "few-shot"], default="zero-shot")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    train_df, _, test_df = load_splits(args.data_source, args.train_path, test_path=args.test_path)

    examples = ""
    if args.mode == "few-shot":
        rows = train_df.sample(n=args.k, random_state=args.seed)
        examples = format_examples(rows[["claim", "post", "label"]].to_dict("records"))
    system = SYSTEM_PROMPT.format(few_shot_examples=examples)

    call = make_client(args.backend)
    preds, parsed = [], 0
    for claim, post in tqdm(zip(test_df["claim"], test_df["post"]), total=len(test_df)):
        label, ok = classify(call, args.model_id, system, claim, post)
        preds.append(label)
        parsed += ok

    gold = test_df["label"].tolist()
    report = classification_report(gold, preds, labels=LABELS, output_dict=True, zero_division=0)
    results = {
        **vars(args),
        "accuracy": accuracy_score(gold, preds),
        "macro_f1": f1_score(gold, preds, labels=LABELS, average="macro", zero_division=0),
        "f1_per_class": {l: report[l]["f1-score"] for l in LABELS},
        "parse_success_rate": parsed / len(gold),
        "confusion_matrix": confusion_matrix(gold, preds, labels=LABELS).tolist(),
        "few_shot_examples": examples,
    }
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(classification_report(gold, preds, labels=LABELS, digits=4, zero_division=0))
    print(f"saved to {args.output_file}")


if __name__ == "__main__":
    main()
