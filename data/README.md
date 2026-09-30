# Season data

`season_snapshots.csv` contains one row per club after its first five league matches.

- Completed 2023–24, 2024–25, and 2025–26 seasons include `final_rank` as the training target.
- Current 2026–27 rows intentionally leave `final_rank` blank because the season is unfinished.
- Model features are derived from results, shots, and shots on target. The source provides xG for 2026–27 only, so historical xG is stored as missing and excluded from modeling rather than being replaced with zero.

Source files come from [football-data.co.uk](https://www.football-data.co.uk/englandm.php):

- `https://www.football-data.co.uk/mmz4281/2324/E0.csv`
- `https://www.football-data.co.uk/mmz4281/2425/E0.csv`
- `https://www.football-data.co.uk/mmz4281/2526/E0.csv`
- `https://www.football-data.co.uk/mmz4281/2627/E0.csv`

To refresh the current snapshot after the first five matches have been played:

```bash
python scripts/update_season_data.py
```

The forecast intentionally uses the same five-match cutoff for every season. Mixing a five-match historical snapshot with a later current table would make the comparison inconsistent.
