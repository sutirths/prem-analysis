"""Forecast Premier League rank from comparable early-season snapshots.

Completed seasons provide Gameweek 5 features and the eventual final rank.
The current season provides the same features but no target, so it is used only
for forecasting. PyTorch and TensorFlow implement the same small regressor.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path(__file__).parent / "data" / "season_snapshots.csv"

RATE_FEATURES = [
    "points_per_game",
    "goal_difference_per_game",
    "shots_difference_per_game",
    "shots_on_target_difference_per_game",
]

REQUIRED_COLUMNS = {
    "season", "season_status", "team", "snapshot_rank", "played", "points",
    "goals_for", "goals_against", "shots_for", "shots_against",
    "shots_on_target_for", "shots_on_target_against", "xg_for", "xg_against",
    "final_rank",
}


def load_season_data(filepath: str | Path = DATA_FILE) -> pd.DataFrame:
    """Load and validate completed-season targets and current-season features."""
    dataframe = pd.read_csv(filepath)
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Season data is missing columns: {', '.join(sorted(missing))}")
    if dataframe.duplicated(["season", "team"]).any():
        raise ValueError("Each club may appear only once per season.")
    if (dataframe["played"] <= 0).any():
        raise ValueError("Every snapshot must contain at least one played match.")

    completed = dataframe["season_status"].eq("complete")
    current = dataframe["season_status"].eq("current")
    if not completed.any() or not current.any():
        raise ValueError("The dataset needs completed training seasons and one current season.")
    if dataframe.loc[completed, "final_rank"].isna().any():
        raise ValueError("Completed seasons must include final ranks.")
    if dataframe.loc[current, "final_rank"].notna().any():
        raise ValueError("The unfinished current season must not include final ranks.")
    if dataframe.loc[current, "season"].nunique() != 1:
        raise ValueError("Exactly one season may be marked current.")
    return dataframe


def add_rate_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert cumulative snapshot totals to comparable per-match features."""
    featured = dataframe.copy()
    played = featured["played"]
    featured["points_per_game"] = featured["points"] / played
    featured["goal_difference_per_game"] = (
        featured["goals_for"] - featured["goals_against"]
    ) / played
    featured["shots_difference_per_game"] = (
        featured["shots_for"] - featured["shots_against"]
    ) / played
    featured["shots_on_target_difference_per_game"] = (
        featured["shots_on_target_for"] - featured["shots_on_target_against"]
    ) / played
    return featured


def split_seasons(dataframe: pd.DataFrame):
    """Use the newest completed season for honest temporal validation."""
    featured = add_rate_features(dataframe)
    completed = sorted(featured.loc[featured["season_status"].eq("complete"), "season"].unique())
    if len(completed) < 2:
        raise ValueError("At least two completed seasons are required for temporal validation.")

    validation_season = completed[-1]
    train = featured[featured["season"].isin(completed[:-1])].copy()
    validation = featured[featured["season"].eq(validation_season)].copy()
    current = featured[featured["season_status"].eq("current")].copy()
    return train, validation, current, validation_season


def _scale(train: pd.DataFrame, *others: pd.DataFrame):
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[RATE_FEATURES])
    return train_x, [scaler.transform(frame[RATE_FEATURES]) for frame in others]


def train_pytorch(x_train: np.ndarray, y_train: np.ndarray, x_predict: np.ndarray) -> np.ndarray:
    """Train and run the PyTorch rank regressor."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    loss_function = nn.MSELoss()
    features = torch.tensor(x_train, dtype=torch.float32)
    targets = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    model.train()
    for _ in range(300):
        optimizer.zero_grad()
        loss = loss_function(model(features), targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x_predict, dtype=torch.float32)).numpy().ravel()


def train_tensorflow(x_train: np.ndarray, y_train: np.ndarray, x_predict: np.ndarray) -> np.ndarray:
    """Train and run the equivalent TensorFlow/Keras rank regressor."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=300, verbose=0)
    return model.predict(x_predict, verbose=0).ravel()


def scores_to_positions(scores: np.ndarray) -> np.ndarray:
    """Convert continuous model scores into unique positions from 1 to N."""
    order = np.argsort(scores, kind="stable")
    positions = np.empty(len(scores), dtype=int)
    positions[order] = np.arange(1, len(scores) + 1)
    return positions


def _predict_both(train: pd.DataFrame, predict: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_train, [x_predict] = _scale(train, predict)
    y_train = train["final_rank"].to_numpy(dtype=float)
    return (
        train_pytorch(x_train, y_train, x_predict),
        train_tensorflow(x_train, y_train, x_predict),
    )


def evaluate_previous_season(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate on a later season that neither model saw during training."""
    pytorch_scores, tensorflow_scores = _predict_both(train, validation)
    result = validation[["team", "snapshot_rank", "final_rank"]].copy()
    result["pytorch_rank"] = scores_to_positions(pytorch_scores)
    result["tensorflow_rank"] = scores_to_positions(tensorflow_scores)
    result["ensemble_rank"] = scores_to_positions((pytorch_scores + tensorflow_scores) / 2)
    actual = result["final_rank"]
    metrics = {
        "gameweek_5_baseline": mean_absolute_error(actual, result["snapshot_rank"]),
        "pytorch": mean_absolute_error(actual, result["pytorch_rank"]),
        "tensorflow": mean_absolute_error(actual, result["tensorflow_rank"]),
        "ensemble": mean_absolute_error(actual, result["ensemble_rank"]),
    }
    return result.sort_values("final_rank"), metrics


def predict_current_season(historical: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Train on all completed seasons and predict the current final table."""
    pytorch_scores, tensorflow_scores = _predict_both(historical, current)
    result = current[["team", "snapshot_rank", "points"]].copy()
    result["pytorch_rank"] = scores_to_positions(pytorch_scores)
    result["tensorflow_rank"] = scores_to_positions(tensorflow_scores)
    result["predicted_rank"] = scores_to_positions((pytorch_scores + tensorflow_scores) / 2)
    return result.sort_values("predicted_rank")


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    dataframe = load_season_data()
    train, validation, current, validation_season = split_seasons(dataframe)

    validation_result, metrics = evaluate_previous_season(train, validation)
    print(f"--- Temporal validation: predicting {validation_season} from Gameweek 5 ---")
    print(validation_result.to_string(index=False, formatters={"final_rank": "{:.0f}".format}))
    print("\nMean absolute error (league positions):")
    for name, value in metrics.items():
        print(f"  {name.replace('_', ' ').title():22} {value:.2f}")

    historical = add_rate_features(dataframe[dataframe["season_status"].eq("complete")])
    current_season = current["season"].iloc[0]
    completed_count = dataframe.loc[dataframe["season_status"].eq("complete"), "season"].nunique()
    forecast = predict_current_season(historical, current)
    print(f"\n--- {current_season} forecast using {completed_count} completed seasons ---")
    print(forecast.to_string(index=False))
    print("\nEarly-season warning: this is a Gameweek 5 experiment, so uncertainty is high.")


if __name__ == "__main__":
    main()
