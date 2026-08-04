# Spotter Freight Rate Prediction

Predict posted rates for truckload freight. Training data is 48,000 labeled loads
from January–October 2025 (`data/train-test.csv`); the goal is to predict rates for
12,000 unlabeled loads in November–December (`data/validation.csv`), plus a fixed
Lexington → Fort Wayne lane for every day of December.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status

Starting with a time-based holdout (train on Jan–Aug, evaluate on Sep–Oct — the
hidden test set is future months, so a random split would leak) and simple
$/mile baselines to establish the bar:

```bash
python src/benchmark.py
```
