# Premier League Rank Analysis

A data-science project that explores which season statistics are associated with Premier League finishing position, then compares equivalent PyTorch and TensorFlow models on a real early-season forecasting problem.

The forecasting dataset contains 80 club-season records: three completed seasons from 2023–24 through 2025–26 plus the current 2026–27 season. It is intentionally an educational comparison, not a betting or production forecasting system.

## What it does

- Cleans the supplied CSV export, including its byte-order mark and empty trailing rows.
- Produces `correlation_plot.png`, showing statistics most associated with a better final rank.
- Builds comparable Gameweek 5 snapshots from match results, shots, and shots on target.
- Uses 2023–24 and 2024–25 to predict the completed 2025–26 season, providing temporal validation with no random season leakage.
- Retrains on all three completed seasons and produces an ensemble forecast for the current 2026–27 season.
- Converts continuous model scores into a unique predicted table from positions 1–20.
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

# Validate against 2025–26 and forecast the current 2026–27 table
python proj.py

# Run the lightweight regression tests
python -m unittest discover -s tests -v
```

The forecast output contains:

- `snapshot_rank`: the real table after five matches.
- `pytorch_rank` and `tensorflow_rank`: each framework's forecast.
- `predicted_rank`: the ensemble forecast produced by averaging both model scores and ranking the clubs.

## Season data

`data/season_snapshots.csv` is derived from public match-result files supplied by [football-data.co.uk](https://www.football-data.co.uk/englandm.php). The current season does not contain a final-rank target because that outcome is not known yet.

Refresh the season data with:

```bash
python scripts/update_season_data.py
```

This requires internet access. The committed snapshot lets the model and tests run offline.

To try the optional live-standing client, obtain a football-data.org token and set it only in your shell:

```bash
export FOOTBALL_API_KEY="your-token"
python main_engine.py
```

## Repository hygiene

The `.gitignore` excludes local environments, generated output, and credential files. Before publishing, rotate any API keys that may previously have been committed or shared; removing a key from a file does not revoke it.
