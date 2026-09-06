# Premier League Rank Analysis

A small data-science project that explores which season statistics are associated with Premier League finishing position, then compares equivalent PyTorch and TensorFlow regression models.

The included dataset contains 20 team-season records. This is intentionally an educational comparison: its small sample means the models should not be treated as production-quality forecasts.

## What it does

- Cleans the supplied CSV export, including its byte-order mark and empty trailing rows.
- Produces `correlation_plot.png`, showing statistics most associated with a better final rank.
- Trains matching PyTorch and TensorFlow regressors on a held-out subset and reports mean absolute error.
- Optionally fetches current standings from football-data.org using an environment variable, never a hard-coded key.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## Run and test

```bash
# Run the correlation analysis and create the chart
python prem_analysis.py

# Run the PyTorch/TensorFlow comparison
python proj.py

# Run the lightweight regression tests
python -m unittest discover -s tests -v
```

To try the optional live-standing client, obtain a football-data.org token and set it only in your shell:

```bash
export FOOTBALL_API_KEY="your-token"
python main_engine.py
```

## Repository hygiene

The `.gitignore` excludes local environments, generated output, and credential files. Before publishing, rotate any API keys that may previously have been committed or shared; removing a key from a file does not revoke it.
