## Download the benchmark datasets:

- **Stanceosaurus** (Zheng et al., 2022) [link](https://github.com/JonathanQZheng/Stanceosaurus): copy the `train/`, `dev/` and `test/` folders from its `data/` directory into `data/stanceosaurus/`
- **EZ-STANCE** (Zhao and Caragea, 2024) [link](https://github.com/chenyez/EZ-STANCE): put the claim subset of subtask B at `data/ez-stance/subtaskB/claim/<domain>/raw_{train,val,test}_all_onecol.csv`

Then run `scripts/prepare_data.sh` to build the train/test splits.

## 🗳️ VoteStance

Claim–tweet pairs about voting claims in the 2024 U.S. election, collected Oct 23 – Nov 6, 2024. Tweets are released **as IDs only**.

| file | rows | content |
|---|---|---|
| `train.csv` | 500 | claim, tweet ID, match label, stance label |
| `test.csv` | 150 | claim, tweet ID, match label, stance label |
| `claims.csv` | 62 | fact-checked claims with topic and category |
| `easy_negatives.csv` | 500 | extra negative pairs for the claim matcher |
| `corpus_tweet_ids.txt` | 109,235 | election tweet corpus |

- `match_label`: `Match` / `No Match`
- `stance_label`: `Support` / `Refute` / `Neutral` (only for matched pairs)

### 🔄 Filling in the tweet text

1. Fetch the tweets for the IDs with the X API and save them as JSON-lines files in `raw_tweets/`.
2. Run:
   ```bash
   python -m data.prepare.build_votestance --tweets_dir raw_tweets/ --out_dir data/votestance/with_text
   python -m pipeline.preprocess_tweets --tweets_dir raw_tweets/ --out data/votestance/corpus.csv
   ```

> Note: Tweets deleted since collection can't be recovered, so the rebuilt files may have fewer rows.

The VoteStance annotations are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); tweets remain subject to the X Terms of Service.

The annotation study was approved by the Georgetown University IRB (STUDY00005807).
