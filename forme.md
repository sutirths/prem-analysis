# Premier League Rank Analysis: Interview Guide

## Project summary

This is a Python data-science project that analyzes Premier League team statistics and compares two neural-network implementations for predicting a team's final league rank.

**Interview introduction:**

> I built an end-to-end exploratory data-analysis and machine-learning comparison project using Premier League team statistics. It cleans a messy CSV export, identifies statistics associated with finishing position, visualizes the findings, and compares equivalent PyTorch and TensorFlow regression models on a held-out test set.

This is an educational ML project, not a production football-prediction system, because the supplied dataset contains only 20 complete team records.

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
- `proj.py`: PyTorch and TensorFlow model comparison.
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

## Machine-learning pipeline

The regression models use these nine features:

```text
goals
xg
shots
shots_on_target
Poss
Total Touches
Successful Dribbles
TotDist Carried
PrgC
```

The target variable is final `rank`.

Workflow:

1. Clean the source data.
2. Remove records with missing required features.
3. Split data into 80% training and 20% testing records.
4. Standardize features with scikit-learn's `StandardScaler`.
5. Fit the scaler on training data only.
6. Transform both training and test data with that fitted scaler.
7. Train equivalent PyTorch and TensorFlow regression networks.
8. Evaluate both models on held-out teams with Mean Absolute Error (MAE).

Fitting the scaler only on training data prevents data leakage: the model does not use information from the test set while learning feature scaling.

## Neural-network comparison

Both frameworks use comparable model structures:

```text
9 input features
→ Dense layer: 16 neurons + ReLU
→ Dense layer: 8 neurons + ReLU
→ Output layer: 1 predicted rank
```

Both models use:

- Regression output because final rank is numeric.
- Mean Squared Error loss during training.
- Adam optimizer.
- 200 epochs.
- Fixed random seeds for more reproducible results.

The evaluation metric is MAE, reported in league places. An MAE of `4.76`, for example, means predictions were off by about 4.76 final league positions on average for the held-out teams.

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
5. Explain the held-out predictions and MAE values.
6. State the limitations honestly.

The test suite verifies:

- The loader removes blank rows and export-artifact columns.
- The analysis produces correlations and writes a chart.
- Model-preparation code produces valid feature matrices and matching held-out team labels.

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

- The dataset has only 20 complete records.
- The held-out test set is approximately four teams, which is too small for strong performance claims.
- Neural networks may not be the best model type for such a small tabular dataset.
- The data is observational, so correlation is not causation.
- Rank is modeled as a continuous value even though final rank is an ordered integer.
- A live prediction system would need a richer data source for all model features.
- The project uses one random split; cross-validation would give more stable evaluation.

## Interview questions and answers

### What was the purpose of this project?

> The goal was to build a reproducible mini data-science workflow: clean raw football data, use exploratory analysis to identify meaningful relationships, and compare the same regression problem in PyTorch and TensorFlow.

### What was the hardest part?

> The initial challenge was data quality rather than model architecture. The CSV had a byte-order mark in the team column, empty export columns, and trailing blank records. I made the loader resilient to those issues, then added tests so the cleanup behavior is verified rather than assumed.

### Why use neural networks here?

> The purpose was primarily to compare equivalent implementations in PyTorch and TensorFlow. With such a small tabular dataset, I would not claim a neural network is the optimal production choice. Simpler baselines may generalize better, which is why benchmarking and cross-validation would be the next steps.

### How did you avoid data leakage?

> I split the records before scaling and fit `StandardScaler` only on the training partition. I then used that fitted scaler to transform the test partition, so no test-data distribution information enters training.

### How did you evaluate the models?

> I evaluated predictions on held-out teams using Mean Absolute Error, expressed in league positions. Because the dataset is small, I present it as a framework comparison and educational result rather than a reliable real-world forecast.

### What would you improve next?

> I would collect multiple seasons of data, add cross-validation, and compare neural networks with simpler baselines such as linear regression, random forest, and gradient boosting. I would add per-match feature engineering, experiment tracking, and a richer data source for live features. Finally, I would expose the analysis through a dashboard or API.

## Résumé description

> Built a Python Premier League analytics project using Pandas, Matplotlib, scikit-learn, PyTorch, and TensorFlow. Developed a robust CSV-cleaning pipeline, correlation-based exploratory analysis, reproducible train/test evaluation, and comparative neural-network regression models. Added automated tests, dependency documentation, and environment-based API credential handling.

## Presentation guidance

Emphasize the data-cleaning pipeline, reproducibility, testing, security practices, and honest evaluation. Do not present the model as production-ready forecasting: the small dataset makes it best suited as an educational comparison project.
