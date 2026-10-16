# EPL Match Predictor

This project predicts Premier League results: home win, draw, or away win. It works in two steps, and it only ever uses information that was available before kickoff.

## How it works

**Step 1: Dixon-Coles.** Think of this as a scout who only cares about goals. It estimates how many goals each team is likely to score, then works out how likely every possible scoreline is.

**Step 2: LightGBM.** Think of this as the manager. It takes the scout's goal estimates and adds recent form, Elo ratings, xG differences, and rest days. Then it gives the final chances for a home win, a draw, and an away win.

## No peeking at the future

The easiest way to fool yourself with a sports model is to let future results leak into training. So when I predict a match on date T, everything the models use (team ratings, Elo, rolling averages) comes only from matches played before T. The backtests follow the same rule and move forward through time instead of shuffling the data.

## Getting started

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download the data and build the main dataset:

```bash
python -m src.data_loader
```

Build the features:

```bash
python -m src.features
```

Train the models and run the backtest:

```bash
python -m src.train
```

Predict the upcoming gameweek:

```bash
python -m src.predict
```