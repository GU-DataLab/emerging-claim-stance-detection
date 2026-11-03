# Emerging-Claim Stance Detection

[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-red?logo=acm&logoColor=white)](https://dl.acm.org/doi/10.1145/3799682.3840919) ![CIKM 2026](https://img.shields.io/badge/CIKM-2026-blue) ![DOI](https://img.shields.io/badge/DOI-10.1145%2F3799682.3840919-informational) ![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow)

Code and data for ***Emerging-Claim Stance Detection for Understanding Opinions in Voting Discourse*** (CIKM 2026).

The model classifies the stance (Support / Refute / Neutral) of a social media post toward a claim, including claims not seen during training. It combines:

- a pretrained encoder,
- two-way claim–post cross-attention,
- a supervised contrastive objective.

The repository also contains:

- a two-stage inference pipeline (claim matching, then stance detection) for measuring stance toward verified claims in a large tweet corpus,
- **VoteStance**, an annotated dataset of 2024 U.S. election voting claims and tweets.

## 📁 Repository structure

```
├── stance/          # model, data loaders, training
├── experiments/     # main results, ablations, analysis
├── pipeline/        # real-world inference application
├── data/
│   ├── prepare/     # preprocessing
│   └── votestance/  # 🗳️ VoteStance dataset (tweet IDs)
└── scripts/         
```

## ⚙️ Setup

Install the packages in `requirements.txt`, then follow [data/README.md](data/README.md) to download the benchmarks and fill in the VoteStance tweet text.

## 🚀 Experiments

```bash
bash scripts/run_stanceosaurus.sh        # 🧪 Stanceosaurus: our model, ablations, baselines
bash scripts/run_ezstance.sh             # 🧪 EZ-STANCE: leave-one-domain-out
bash scripts/run_votestance.sh           # 🗳️ VoteStance
bash scripts/run_llm_baselines.sh        # 🤖 GPT-4o and Llama-3.1-8B
bash scripts/run_election_pipeline.sh    # 📊 claim matching + stance on the 2024 election corpus
```

📈 Summarize results with significance tests:

```bash
python -m experiments.aggregate_results results/stanceosaurus.jsonl
```

## 📝 Citation

```bibtex
@inproceedings{ahmed2026emerging,
  title     = {Emerging-Claim Stance Detection for Understanding Opinions in Voting Discourse},
  author    = {Ahmed, Mohamed and Singh, Lisa and Paik, Sejin},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  doi       = {10.1145/3799682.3840919}
}
```

## 📄 License

Code: [MIT](LICENSE). VoteStance annotations: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); tweets remain subject to the X Terms of Service.
