"""Explore which team statistics are associated with Premier League rank."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_stats(filepath: str | Path) -> pd.DataFrame:
    """Load the statistics file and discard empty export columns/rows."""
    dataframe = pd.read_csv(filepath, encoding="utf-8-sig")
    dataframe = dataframe.loc[:, ~dataframe.columns.str.match(r"^Unnamed")]
    dataframe = dataframe.dropna(how="all")

    if "rank" not in dataframe:
        raise ValueError("The dataset must contain a 'rank' column.")

    for column in dataframe.columns:
        if column != "Team":
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    return dataframe.dropna(subset=["Team", "rank"])


def analyze_premier_league_stats(
    filepath: str | Path, output_path: str | Path = "correlation_plot.png"
) -> pd.Series:
    """Return rank correlations and save a chart of the strongest predictors.

    Rank 1 is best, so a negative value means that higher values of a statistic
    are associated with a better finishing position. Correlation is descriptive,
    not evidence that a statistic causes a team's league position.
    """
    dataframe = load_stats(filepath)
    numeric_data = dataframe.select_dtypes(include="number").dropna(axis=1, how="all")
    correlations = numeric_data.corr(numeric_only=True)["rank"].drop("rank").sort_values()

    strongest_factors = correlations.head(10).sort_values()
    figure, axis = plt.subplots(figsize=(12, 8))
    axis.barh(strongest_factors.index, strongest_factors.values, color="teal")
    axis.set_title("Statistics Most Associated with Better Premier League Rank")
    axis.set_xlabel("Correlation with final rank (negative is better)")
    axis.set_ylabel("Statistical factor")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    return correlations


def main() -> None:
    correlations = analyze_premier_league_stats("premstats.csv")
    best_factor = correlations.idxmin()
    worst_factor = correlations.idxmax()
    weakest_factor = correlations.abs().idxmin()

    print("Correlation plot saved to correlation_plot.png")
    print("\n--- Correlation with League Rank ---")
    print(correlations)
    print("\n--- Summary ---")
    print(f"Strongest association with a better rank: {best_factor} ({correlations[best_factor]:.4f})")
    print(f"Strongest association with a worse rank: {worst_factor} ({correlations[worst_factor]:.4f})")
    print(f"Weakest association with rank: {weakest_factor} ({correlations[weakest_factor]:.4f})")
    print("Note: correlation measures association, not causation.")


if __name__ == "__main__":
    main()
