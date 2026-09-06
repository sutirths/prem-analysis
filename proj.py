"""Compare PyTorch and TensorFlow regressors for Premier League rank."""

import random
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prem_analysis import load_stats

FEATURES = [
    "goals", "xg", "shots", "shots_on_target", "Poss", "Total Touches",
    "Successful Dribbles", "TotDist Carried", "PrgC",
]


def prepare_data(filepath: str | Path, test_size: float = 0.2, random_state: int = 42):
    """Return standardized train/test data and the matching held-out team names."""
    dataframe = load_stats(filepath).dropna(subset=FEATURES + ["rank"])
    if len(dataframe) < 5:
        raise ValueError("At least five complete teams are required to train and test a model.")

    train, test = train_test_split(dataframe, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURES])
    x_test = scaler.transform(test[FEATURES])
    return x_train, x_test, train["rank"].to_numpy(), test["rank"].to_numpy(), test["Team"].to_numpy()


def train_pytorch(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Train a small, reproducible PyTorch regression network."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(x_train.shape[1], 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()
    features = torch.tensor(x_train, dtype=torch.float32)
    targets = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        loss = loss_function(model(features), targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x_test, dtype=torch.float32)).numpy().ravel()


def train_tensorflow(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Train a matching TensorFlow/Keras regression network."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=200, verbose=0)
    return model.predict(x_test, verbose=0).ravel()


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    x_train, x_test, y_train, y_test, teams = prepare_data("premstats.csv")
    pytorch_predictions = train_pytorch(x_train, y_train, x_test)
    tensorflow_predictions = train_tensorflow(x_train, y_train, x_test)

    print("--- Model comparison (held-out teams) ---")
    for team, actual, pytorch, tensorflow in zip(teams, y_test, pytorch_predictions, tensorflow_predictions):
        print(f"{team:16} | Actual: {actual:>4.0f} | PyTorch: {pytorch:>5.1f} | TensorFlow: {tensorflow:>5.1f}")
    print(f"\nPyTorch MAE: {mean_absolute_error(y_test, pytorch_predictions):.2f} league places")
    print(f"TensorFlow MAE: {mean_absolute_error(y_test, tensorflow_predictions):.2f} league places")
    print("Small sample warning: use this as a framework comparison, not a production forecast.")


if __name__ == "__main__":
    main()
