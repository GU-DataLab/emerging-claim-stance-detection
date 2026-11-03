"""
Stacked bar chart of ACC-corrected stance proportions per claim topic.

Usage:
  python -m pipeline.plot_topics --acc results/election/acc_results.json --out figures/stance_by_topic.pdf
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"Support": "#2E8B57", "Refute": "#DC143C", "Neutral": "#565656"}
SHORT = {
    "Voter Eligibility and Identification": "Voter\nEligibility",
    "Election Security and Integrity": "Election\nSecurity",
    "Mail Voting and Ballot Counting": "Mail\nVoting",
    "Electoral Procedures and Rules": "Electoral\nProcedures",
    "Election Interference and Disinformation": "Election\nInterference",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acc", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.acc) as f:
        acc = json.load(f)
    classes, topics = acc["classes"], acc["topics"]
    order = sorted(topics, key=lambda t: topics[t]["n"], reverse=True)

    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.33, 2.8))
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for i, c in enumerate(classes):
        vals = np.array([topics[t]["corrected"][i] for t in order])
        bars = ax.bar(x, vals, bottom=bottom, label=c, color=COLORS[c], alpha=0.85, width=0.65)
        for b, v, y0 in zip(bars, vals, bottom):
            if v > 0.08:
                ax.text(b.get_x() + b.get_width() / 2, y0 + v / 2, f"{v * 100:.0f}%",
                        ha="center", va="center", fontsize=6.5, fontweight="bold", color="white")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"{SHORT.get(t, t)}\n(n={topics[t]['n']})" for t in order], fontsize=5)
    ax.set_ylabel("Proportion", fontsize=6.5, fontweight="bold", labelpad=3)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.tick_params(axis="y", labelsize=5)
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc="lower center", ncol=3, fontsize=6.5,
              handlelength=1.0, handletextpad=0.3, columnspacing=0.8)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=600, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
