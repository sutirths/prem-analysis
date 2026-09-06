What this project is
This is a Python data-science project that analyzes Premier League team statistics and compares two neural-network implementations for predicting a team’s final league rank.
The strongest way to describe it is:
“I built an end-to-end exploratory data-analysis and machine-learning comparison project using Premier League team statistics. It cleans a messy CSV export, identifies statistics associated with finishing position, visualizes the findings, and compares equivalent PyTorch and TensorFlow regression models on a held-out test set.”

It is an educational ML project—not a production football-prediction system—because the supplied dataset contains only 20 complete team records.
The stack
Area	Technology	Purpose
Language	Python	All project logic
Data processing	Pandas, NumPy	Load, clean, transform data
Visualization	Matplotlib	Generate the correlation chart
Traditional ML utilities	scikit-learn	Train/test split, feature scaling, MAE
Deep learning	PyTorch	First neural-network implementation
Deep learning	TensorFlow/Keras	Second equivalent implementation
Optional API	requests + football-data.org	Fetch live Premier League standings
Testing	Python unittest	Regression tests for the data pipeline
Dependency setup	requirements.txt	Reproducible installation
Security	Environment variables + .gitignore	Prevent API keys entering GitHub


There is no frontend, database, deployment pipeline, or web application in the current version. It is a command-line data-analysis project.
Project files
- [prem_analysis.py](/Users/sutirthsamudrala/prem/prem_analysis.py) — data cleaning, correlation analysis, and chart generation.
- [proj.py](/Users/sutirthsamudrala/prem/proj.py) — PyTorch and TensorFlow model comparison.
- [main_engine.py](/Users/sutirthsamudrala/prem/main_engine.py) — optional live standings API client.
- [premstats.csv](/Users/sutirthsamudrala/prem/premstats.csv) — primary dataset used by the scripts.
- [tests/test_project.py](/Users/sutirthsamudrala/prem/tests/test_project.py) — automated tests.
- [requirements.txt](/Users/sutirthsamudrala/prem/requirements.txt) — dependencies.
- [README.md](/Users/sutirthsamudrala/prem/README.md) — setup and usage documentation.
premstats.xlsx is a duplicate/reference dataset; the Python code uses the CSV.
Data pipeline
The project starts with a CSV export containing team-level season statistics such as:
- Final rank
- Goals
- Expected goals (xg)
- Shots and shots on target
- Possession
- Touches
- Successful dribbles
- Progressive carries
- Carries into the penalty area
The source CSV had real-world data-quality issues:
- A UTF-8 byte-order mark corrupted the Team column name.
- It had empty Unnamed export columns.
- It contained 23 blank rows after the 20 complete records.
The loader fixes those issues by:
1. Reading with utf-8-sig encoding.
2. Removing columns beginning with Unnamed.
3. Dropping completely blank rows.
4. Converting non-team columns to numeric values.
5. Keeping only complete team/rank records.
That is a useful interview point: you did not assume the dataset was clean; you built preprocessing that handles common CSV-export problems.
Analysis process
The analysis script calculates Pearson correlations between each numeric statistic and final league rank.
Because rank 1 is best:
- Negative correlation: higher values tend to be associated with a better finish.
- Positive correlation: higher values tend to be associated with a worse finish.
- A value near zero: little linear relationship in this dataset.
The chart displays the ten factors most negatively correlated with rank. In this dataset, goals, expected goals, shots, shots on target, attacking penalty-area touches, and progressive actions are among the strongest associations.
A good way to explain the conclusion:
“The analysis shows association, not causation. For example, scoring more goals is strongly associated with a better rank, but correlation alone does not prove that any single feature causes league success.”

Machine-learning pipeline
The model uses nine features:
goals
xg
shots
shots_on_target
Poss
Total Touches
Successful Dribbles
TotDist Carried
PrgC
The target is final rank.
The workflow is:
1. Clean the data.
2. Remove teams with missing required features.
3. Split the data into training and testing sets: 80% train, 20% test.
4. Standardize the numeric features with StandardScaler.
5. Fit the scaler only on training data, then transform test data.
That last point matters because it prevents data leakage: the model does not use information from the test set to calculate feature scaling.
Why two frameworks?
The project implements a similar neural network in both PyTorch and TensorFlow/Keras:
9 input features
→ Dense layer: 16 neurons + ReLU
→ Dense layer: 8 neurons + ReLU
→ Output layer: 1 predicted rank
Both models use:
- Regression output, because rank is a numeric target.
- Mean Squared Error loss during training.
- Adam optimizer.
- 200 epochs.
- A fixed random seed for more reproducible results.
The test evaluation metric is Mean Absolute Error (MAE), reported in league places.
For example, an MAE of 4.76 means the predictions were off by about 4.76 league positions on average for the held-out teams.
How to demo it
source ml_env/bin/activate

# Verify the project
python -m unittest discover -s tests -v

# Generate the analysis and chart
python prem_analysis.py

# Compare the two ML frameworks
python proj.py
During a demo:
1. Run the automated tests.
2. Run prem_analysis.py.
3. Open correlation_plot.png.
4. Explain the correlation result.
5. Run proj.py.
6. Explain the held-out predictions and MAE scores.
7. State the limitations openly.
Testing
There are three automated tests:
- Confirms the loader removes blank rows and export-artifact columns.
- Confirms the analysis returns correlations and writes a chart.
- Confirms model-preparation code creates valid feature matrices and matching test-team labels.
The tests do not currently train the neural networks or call the external API. That is intentional: unit tests should be quick and not depend on network access or API keys.
Optional live API feature
main_engine.py can call football-data.org and return current standings:
export FOOTBALL_API_KEY="your-new-token"
python main_engine.py
The key point for an interviewer:
“I moved credentials out of the codebase and into environment variables, and .gitignore prevents secret files from being committed.”

The live API module currently fetches standings only. It does not yet automatically transform live API data into the full nine-feature model input, because the API response does not necessarily contain all historical features such as xG, progressive carries, and detailed touches.
That is an important limitation to say honestly.
Limitations
Be direct about these:
- The dataset has only 20 complete records.
- A test set of roughly four teams is too small to make strong performance claims.
- Neural networks are not necessarily the best model for such a small tabular dataset.
- The data is observational, so correlation is not causation.
- The model predicts continuous rank values, even though rank is ultimately an ordered integer.
- A live-data prediction pipeline needs a richer data source for all model features.
- The project uses one random split; cross-validation would give a more stable evaluation.
Strong interview answer: “What would you improve next?”
“First, I would collect multiple seasons of data, which would turn this from a 20-row demonstration into a meaningful time-series or panel-data problem. I would add cross-validation and compare the neural networks against simpler baselines such as linear regression, random forest, and gradient boosting. I would track experiments, add feature engineering such as per-match rates, and use a richer football API or dataset for live feature collection. Finally, I would package the workflow into a dashboard or API so a user could explore team predictions interactively.”

Strong interview answer: “Why use neural networks here?”
“The purpose was primarily to compare equivalent implementations in PyTorch and TensorFlow. With such a small tabular dataset, I would not claim a neural network is the optimal production choice. A simpler baseline may generalize better, which is exactly why benchmark comparison and cross-validation would be the next step.”

Strong interview answer: “What was the hardest part?”
“The initial challenge was data quality rather than model architecture. The CSV had a byte-order mark in the team column, empty export columns, and trailing blank records. I made the loader resilient to those issues, then added tests so the cleanup behavior is verified rather than assumed.”

What to put on your résumé
Built a Python Premier League analytics project using Pandas, Matplotlib, scikit-learn, PyTorch, and TensorFlow. Developed a robust CSV-cleaning pipeline, correlation-based exploratory analysis, reproducible train/test evaluation, and comparative neural-network regression models. Added automated tests, dependency documentation, and environment-based API credential handling.

The most important presentation choice: emphasize the clean data pipeline, reproducibility, testing, honest evaluation, and your awareness of the project’s limitations.
