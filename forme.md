# Premier League Rank Analysis: Interview Guide

## Project summary

This is a Python data-science project that analyzes Premier League team statistics and compares two neural-network implementations for predicting a team's final league rank.

**Interview introduction:**

> I built an end-to-end Premier League forecasting project that combines exploratory analysis with season-aware machine learning. It cleans football data, identifies statistics associated with finishing position, validates equivalent PyTorch and TensorFlow models on a later unseen season, and forecasts the current table from the same Gameweek 5 cutoff.

This is an educational ML project, not a production football-prediction system. The forecasting data contains 60 completed club-season records and 20 current-season forecast records, so uncertainty remains high.

## Technology stack

| Area | Technology | Purpose |
|---|---|---|
| Language | Python | All project logic |
| Data processing | Pandas, NumPy | Load, clean, and transform data |
| Visualization | Matplotlib | Generate the correlation chart |
| ML utilities | scikit-learn | Train/test split, feature scaling, and MAE |
| Deep learning | PyTorch | First neural-network implementation |
| Deep learning | TensorFlow/Keras | Second equivalent implementation |
| Optional API | requests + football-data.org | Fetch live Premier League standings |
| Testing | Python `unittest` | Data-pipeline regression tests |
| Dependencies | `requirements.txt` | Reproducible installation |
| Security | Environment variables + `.gitignore` | Keep API keys out of GitHub |

The current project is a command-line data-analysis project. It does not include a frontend, database, deployment pipeline, or web application.

## Project files

- `prem_analysis.py`: data cleaning, correlation analysis, and chart generation.
- `proj.py`: temporal validation and the PyTorch/TensorFlow current-season forecast.
- `data/season_snapshots.csv`: completed 2023–24 through 2025–26 and current 2026–27 Gameweek 5 records.
- `scripts/update_season_data.py`: reproducibly rebuilds season snapshots from match results.
- `main_engine.py`: optional live standings API client.
- `premstats.csv`: primary dataset used by the scripts.
- `premstats.xlsx`: duplicate/reference dataset; the scripts use the CSV.
- `tests/test_project.py`: automated tests.
- `requirements.txt`: Python dependencies.
- `README.md`: setup and usage documentation.

## Data pipeline

The dataset contains team-level season statistics such as final rank, goals, expected goals (`xg`), shots, possession, touches, dribbles, and progressive carries.

The raw CSV had practical data-quality issues:

- A UTF-8 byte-order mark corrupted the `Team` column name.
- Empty `Unnamed` export columns were included.
- Twenty-three blank rows appeared after the 20 complete team records.

The loader handles those issues by:

1. Reading the CSV with `utf-8-sig` encoding.
2. Removing columns that start with `Unnamed`.
3. Removing entirely blank rows.
4. Converting non-team columns to numeric values.
5. Keeping rows with a valid team name and final rank.

**Good interview point:** I did not assume the source data was clean. I built preprocessing for common CSV-export issues and wrote tests to protect that behavior.

## Exploratory analysis

The analysis script calculates Pearson correlations between numeric team statistics and final league rank.

Because rank 1 is best:

- A negative correlation means higher values are associated with a better finishing position.
- A positive correlation means higher values are associated with a worse finishing position.
- A value close to zero means there is little linear association in this dataset.

The script creates `correlation_plot.png`, showing the ten statistics most negatively associated with final rank. In the included data, goals, expected goals, shots, shots on target, attacking penalty-area touches, and progressive actions are among the strongest associations.

**Correct interpretation:**

> The analysis shows association, not causation. For example, scoring more goals is strongly associated with a better rank, but correlation alone does not prove that any single feature causes league success.

## Season-aware forecasting pipeline

Every club is measured after its first five matches. The model uses these per-game features:

```text
points per game
goal difference per game
shot difference per game
shots-on-target difference per game
```

The target is the club's eventual final rank for completed seasons. The current season intentionally has no target.

Workflow:

1. Download season match results from football-data.co.uk.
2. Build a league table after every club has played five matches.
3. Build the completed final table and attach final ranks as targets.
4. Train on the 2023–24 and 2024–25 Gameweek 5 snapshots.
5. Validate on the later, completely unseen 2025–26 season.
6. Report MAE for the raw Gameweek 5 table, PyTorch, TensorFlow, and their ensemble.
7. Retrain on all three completed seasons.
8. Forecast 2026–27 from its Gameweek 5 snapshot.
9. Convert continuous predictions into unique positions from 1–20.

Fitting the scaler only on historical training records prevents feature-scaling leakage. Keeping the validation season later than the training season is also more realistic than randomly mixing clubs from the same season across train and test sets.

## Neural-network comparison

Both frameworks use comparable model structures:

```text
4 input features
→ Dense layer: 16 neurons + ReLU
→ Dense layer: 8 neurons + ReLU
→ Output layer: 1 predicted rank
```

Both models use:

- Regression output because final rank is numeric.
- Mean Squared Error loss during training.
- Adam optimizer.
- 300 epochs.
- Fixed random seeds for more reproducible results.

The evaluation metric is MAE, reported in league places. An MAE of `4.00`, for example, means predictions were off by four final positions on average across the held-out season.

## Testing and demo

From the project folder:

```bash
source ml_env/bin/activate

# Run automated tests
python -m unittest discover -s tests -v

# Generate correlation analysis and chart
python prem_analysis.py

# Run the PyTorch/TensorFlow comparison
python proj.py
```

For a short live demo:

1. Run the automated tests.
2. Run `python prem_analysis.py`.
3. Open `correlation_plot.png` and explain the results.
4. Run `python proj.py`.
5. Explain the 2025–26 temporal-validation MAE and the 2026–27 forecast.
6. State the limitations honestly.

The test suite verifies:

- The loader removes blank rows and export-artifact columns.
- The analysis produces correlations and writes a chart.
- The season file contains three completed seasons and one target-free current season.
- All clubs are compared at the same five-match cutoff.
- Temporal splitting keeps the newer completed season out of training.
- Continuous model scores convert into unique league positions.
- Match aggregation calculates points and goal difference correctly.

The tests intentionally do not train neural networks or call external APIs, keeping them quick and independent of network access or credentials.

## Optional live-data client

`main_engine.py` can fetch current standings from football-data.org:

```bash
export FOOTBALL_API_KEY="your-new-token"
python main_engine.py
```

The API token is read from an environment variable, not hard-coded into the repository. `.gitignore` excludes secret files and local environments.

The live API module currently fetches standings only. It does not automatically create full model input because live standings do not necessarily provide all historical features, such as xG, progressive carries, and detailed touch statistics.

## Limitations

- Only three completed seasons are available, giving 60 labeled club-season records.
- Temporal validation has only one season, so results may change substantially with more history.
- Neural networks may not be the best model type for such a small tabular dataset.
- The data is observational, so correlation is not causation.
- Rank is modeled as a continuous value even though final rank is an ordered integer.
- A live prediction system would need a richer data source for all model features.
- Five matches is an extremely early and noisy point in a season.
- Promoted teams have no Premier League history in the training file.

## Interview questions and answers

### What was the purpose of this project?

> The goal was to build a reproducible mini data-science workflow: clean raw football data, use exploratory analysis to identify meaningful relationships, and compare the same regression problem in PyTorch and TensorFlow.

### What was the hardest part?

> The initial challenge was data quality rather than model architecture. The CSV had a byte-order mark in the team column, empty export columns, and trailing blank records. I made the loader resilient to those issues, then added tests so the cleanup behavior is verified rather than assumed.

### Why use neural networks here?

> The purpose was primarily to compare equivalent implementations in PyTorch and TensorFlow. With such a small tabular dataset, I would not claim a neural network is the optimal production choice. Simpler baselines may generalize better, which is why benchmarking and cross-validation would be the next steps.

### How did you avoid data leakage?

> I use time-aware validation. The model trains on 2023–24 and 2024–25, then validates on the later 2025–26 season. `StandardScaler` is fitted only on historical training seasons, so the validation distribution does not enter training. The current season has no final-rank target and is used only after evaluation.

### How did you evaluate the models?

> I evaluate the models on an entire later season using Mean Absolute Error in league positions. I also compare them with the actual Gameweek 5 table as a transparent baseline. That shows whether the neural networks add value instead of reporting model scores in isolation.

### What would you improve next?

> I would add five to ten more historical seasons and use rolling-origin validation, where each season is predicted only from earlier seasons. I would compare the neural networks with ordinal regression, random forest, and gradient boosting, add strength-of-schedule features, and produce prediction intervals rather than only point estimates.

## Résumé description

> Built a season-aware Premier League forecasting project using Pandas, Matplotlib, scikit-learn, PyTorch, and TensorFlow. Engineered comparable Gameweek 5 features from match-level data, implemented temporal validation against a later season, and generated a current-season ensemble table. Added reproducible data refresh, automated pipeline tests, and secure environment-based API handling.

## Presentation guidance

Emphasize the data-cleaning pipeline, reproducibility, testing, security practices, and honest evaluation. Do not present the model as production-ready forecasting: the small dataset makes it best suited as an educational comparison project.
