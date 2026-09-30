"""Regression tests for the local data pipeline."""

import tempfile
import unittest
from pathlib import Path

from prem_analysis import analyze_premier_league_stats, load_stats
from proj import RATE_FEATURES, add_rate_features, load_season_data, scores_to_positions, split_seasons
from scripts.update_season_data import build_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "premstats.csv"
SEASON_FILE = PROJECT_ROOT / "data" / "season_snapshots.csv"


class DataPipelineTests(unittest.TestCase):
    def test_loader_removes_export_artifacts_and_blank_rows(self):
        dataframe = load_stats(DATA_FILE)
        self.assertEqual(len(dataframe), 20)
        self.assertIn("Team", dataframe.columns)
        self.assertNotIn("Unnamed: 41", dataframe.columns)
        self.assertTrue(dataframe["rank"].notna().all())

    def test_analysis_returns_factors_and_writes_chart(self):
        with tempfile.TemporaryDirectory() as directory:
            chart = Path(directory) / "chart.png"
            correlations = analyze_premier_league_stats(DATA_FILE, chart)
            self.assertTrue(chart.is_file())
            self.assertNotIn("rank", correlations.index)
            self.assertIn("goals", correlations.index)

    def test_season_dataset_separates_training_targets_from_current_forecast(self):
        dataframe = load_season_data(SEASON_FILE)
        completed = dataframe[dataframe["season_status"].eq("complete")]
        current = dataframe[dataframe["season_status"].eq("current")]

        self.assertEqual(
            sorted(dataframe["season"].unique()), ["2023-24", "2024-25", "2025-26", "2026-27"]
        )
        self.assertEqual(len(completed), 60)
        self.assertEqual(len(current), 20)
        self.assertTrue(completed["final_rank"].notna().all())
        self.assertTrue(current["final_rank"].isna().all())
        self.assertTrue(dataframe["played"].eq(5).all())

    def test_temporal_split_and_rate_features(self):
        dataframe = load_season_data(SEASON_FILE)
        train, validation, current, validation_season = split_seasons(dataframe)

        self.assertEqual(validation_season, "2025-26")
        self.assertEqual(set(train["season"]), {"2023-24", "2024-25"})
        self.assertEqual(set(validation["season"]), {"2025-26"})
        self.assertEqual(set(current["season"]), {"2026-27"})
        self.assertTrue(set(RATE_FEATURES).issubset(add_rate_features(dataframe).columns))

    def test_continuous_scores_become_unique_league_positions(self):
        positions = scores_to_positions([8.2, 1.5, 4.0, 9.7])
        self.assertEqual(positions.tolist(), [3, 1, 2, 4])

    def test_match_aggregation_builds_a_valid_table(self):
        import pandas as pd

        matches = pd.DataFrame(
            [
                {"HomeTeam": "A", "AwayTeam": "B", "FTHG": 2, "FTAG": 0, "HS": 8, "AS": 3, "HST": 4, "AST": 1, "HxG": 1.5, "AxG": 0.2},
                {"HomeTeam": "B", "AwayTeam": "A", "FTHG": 1, "FTAG": 1, "HS": 6, "AS": 5, "HST": 2, "AST": 2, "HxG": 0.8, "AxG": 0.9},
            ]
        )
        table = build_table(matches)
        leader = table.iloc[0]
        self.assertEqual(leader["team"], "A")
        self.assertEqual(leader["points"], 4)
        self.assertEqual(leader["goal_difference"], 2)


if __name__ == "__main__":
    unittest.main()
