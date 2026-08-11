# NCAA March Madness Bracket Simulator

An XGBoost-based model that predicts NCAA Men's Division I Basketball
Tournament outcomes and generates an interactive HTML bracket showing the
probability of every team reaching each round.

## How it works

1. **Feature engineering** — loads historical tournament results, seeds, and
   Massey Ordinals (KenPom) rankings, then derives per-team advanced metrics
   from regular-season box scores for every season on record: offensive/
   defensive/net rating, effective FG%, turnover rate, offensive rebound
   rate, free throw rate, three-point attempt rate, three-point percentage,
   and strength of schedule.
2. **Model training** — trains an `XGBClassifier` on the difference in these
   metrics between two teams to predict which one wins, holding out the most
   recent season as a blind test set.
3. **Live bracket pull** — maps the current year's bracket matchups to each
   team's most recent stats and ranking.
4. **Monte Carlo simulation** — simulates the entire bracket 100,000 times
   using the trained model's win probabilities and tallies how often each
   team reaches each round.
5. **HTML output** — writes `interactive_bracket.html`, a hoverable bracket
   showing the top 10 most likely teams to occupy each slot, with their
   simulated probabilities.

## Setup

```bash
pip install -r requirements.txt
```

## Data

This project uses the Kaggle **["March Machine Learning Mania"](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)**
dataset. Per Kaggle's competition rules, this raw data is not redistributed
in this repository — download it yourself and place the following files in
the project root:

- `MTeams.csv`
- `MNCAATourneyCompactResults.csv`
- `MNCAATourneySeeds.csv`
- `MRegularSeasonDetailedResults.csv`
- `MMasseyOrdinals.csv`

You'll need a free Kaggle account to download the data, either via the
competition's "Data" tab or the [Kaggle CLI](https://github.com/Kaggle/kaggle-api).

## Usage

Update the `matchups_east` / `matchups_west` / `matchups_south` /
`matchups_midwest` lists in [model.py](model.py) with the current year's
bracket, then run:

```bash
python model.py
```

This prints the model's blind-test accuracy and a debug table of ratings for
the top seeds, then writes `interactive_bracket.html` — open it in a browser
to explore the simulated bracket.

## License

[MIT](LICENSE) — see the LICENSE file. Note this covers the code only; the
Kaggle dataset it consumes is licensed separately by Kaggle/its providers.
