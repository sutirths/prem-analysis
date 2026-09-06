"""Regression tests for the local data pipeline."""

import tempfile
import unittest
from pathlib import Path

from prem_analysis import analyze_premier_league_stats, load_stats
from proj import FEATURES, prepare_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "premstats.csv"


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

    def test_model_inputs_have_all_expected_features(self):
        x_train, x_test, y_train, y_test, teams = prepare_data(DATA_FILE)
        self.assertEqual(x_train.shape[1], len(FEATURES))
        self.assertEqual(len(x_test), len(y_test))
        self.assertEqual(len(teams), len(y_test))
        self.assertGreater(len(y_train), len(y_test))


if __name__ == "__main__":
    unittest.main()
