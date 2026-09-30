"""Build season snapshots from public football-data.co.uk match results.

The model compares clubs at the same point in each season: after each club has
played five league matches. Completed seasons also receive their final rank,
while the current season deliberately leaves that target blank.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import requests


SEASON_URLS = {
    "2023-24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2024-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "2025-26": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "2026-27": "https://www.football-data.co.uk/mmz4281/2627/E0.csv",
}

OUTPUT_COLUMNS = [
    "season",
    "season_status",
    "team",
    "snapshot_rank",
    "played",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "goal_difference",
    "points",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "xg_for",
    "xg_against",
    "final_rank",
    "source_url",
]


def _empty_record(has_xg: bool) -> dict[str, float]:
    return {
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "points": 0,
        "shots_for": 0,
        "shots_against": 0,
        "shots_on_target_for": 0,
        "shots_on_target_against": 0,
        "xg_for": 0.0 if has_xg else float("nan"),
        "xg_against": 0.0 if has_xg else float("nan"),
    }


def _update_team(
    record: dict[str, float],
    goals_for: float,
    goals_against: float,
    shots_for: float,
    shots_against: float,
    shots_on_target_for: float,
    shots_on_target_against: float,
    xg_for: float,
    xg_against: float,
) -> None:
    record["played"] += 1
    record["goals_for"] += goals_for
    record["goals_against"] += goals_against
    record["shots_for"] += shots_for
    record["shots_against"] += shots_against
    record["shots_on_target_for"] += shots_on_target_for
    record["shots_on_target_against"] += shots_on_target_against
    record["xg_for"] += xg_for
    record["xg_against"] += xg_against

    if goals_for > goals_against:
        record["wins"] += 1
        record["points"] += 3
    elif goals_for == goals_against:
        record["draws"] += 1
        record["points"] += 1
    else:
        record["losses"] += 1


def build_table(matches: pd.DataFrame, match_limit: int | None = None) -> pd.DataFrame:
    """Aggregate results into a league table, optionally stopping at N games/team."""
    required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST"}
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError(f"Match data is missing columns: {', '.join(sorted(missing))}")

    standings: dict[str, dict[str, float]] = {}
    has_xg = {"HxG", "AxG"}.issubset(matches.columns)

    for row in matches.itertuples(index=False):
        home = row.HomeTeam
        away = row.AwayTeam
        standings.setdefault(home, _empty_record(has_xg))
        standings.setdefault(away, _empty_record(has_xg))

        if match_limit is not None and (
            standings[home]["played"] >= match_limit or standings[away]["played"] >= match_limit
        ):
            continue

        home_xg = float(row.HxG) if has_xg and pd.notna(row.HxG) else 0.0
        away_xg = float(row.AxG) if has_xg and pd.notna(row.AxG) else 0.0
        _update_team(
            standings[home], row.FTHG, row.FTAG, row.HS, row.AS, row.HST, row.AST, home_xg, away_xg
        )
        _update_team(
            standings[away], row.FTAG, row.FTHG, row.AS, row.HS, row.AST, row.HST, away_xg, home_xg
        )

    table = pd.DataFrame.from_dict(standings, orient="index").rename_axis("team").reset_index()
    table["goal_difference"] = table["goals_for"] - table["goals_against"]
    table = table.sort_values(
        ["points", "goal_difference", "goals_for", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    table["rank"] = range(1, len(table) + 1)
    return table


def build_season_snapshot(
    matches: pd.DataFrame,
    season: str,
    current_season: str,
    source_url: str,
    snapshot_matches: int = 5,
) -> pd.DataFrame:
    """Create one fixed-matchweek feature snapshot and optional final targets."""
    snapshot = build_table(matches, match_limit=snapshot_matches)
    if len(snapshot) != 20 or not snapshot["played"].eq(snapshot_matches).all():
        raise ValueError(f"{season} does not yet contain {snapshot_matches} matches for all 20 clubs.")

    snapshot = snapshot.rename(columns={"rank": "snapshot_rank"})
    snapshot.insert(0, "season", season)
    is_current = season == current_season
    snapshot.insert(1, "season_status", "current" if is_current else "complete")

    if is_current:
        snapshot["final_rank"] = pd.NA
    else:
        final_table = build_table(matches)[["team", "rank"]].rename(columns={"rank": "final_rank"})
        snapshot = snapshot.merge(final_table, on="team", how="left", validate="one_to_one")

    snapshot["source_url"] = source_url
    return snapshot[OUTPUT_COLUMNS]


def build_dataset(
    season_files: Mapping[str, str | Path], current_season: str, snapshot_matches: int = 5
) -> pd.DataFrame:
    """Build a combined chronological dataset from local source files."""
    snapshots = []
    for season, filepath in sorted(season_files.items()):
        matches = pd.read_csv(filepath, encoding="utf-8-sig")
        snapshots.append(
            build_season_snapshot(
                matches,
                season,
                current_season,
                SEASON_URLS.get(season, str(filepath)),
                snapshot_matches,
            )
        )
    return pd.concat(snapshots, ignore_index=True)


def download_sources(destination: Path) -> dict[str, Path]:
    """Download all configured season files and return their local paths."""
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for season, url in SEASON_URLS.items():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path = destination / f"{season}.csv"
        path.write_bytes(response.content)
        paths[season] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/season_snapshots.csv"))
    parser.add_argument("--current-season", default="2026-27")
    parser.add_argument("--snapshot-matches", type=int, default=5)
    parser.add_argument("--source-dir", type=Path, help="Use local season CSVs named 2024-25.csv, etc.")
    args = parser.parse_args()

    if args.source_dir:
        season_files = {season: args.source_dir / f"{season}.csv" for season in SEASON_URLS}
    else:
        season_files = download_sources(Path("data/raw"))

    dataset = build_dataset(season_files, args.current_season, args.snapshot_matches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False, float_format="%.3f")
    print(f"Wrote {len(dataset)} club-season records to {args.output}")


if __name__ == "__main__":
    main()
