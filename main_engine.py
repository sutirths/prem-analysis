"""Optional live Premier League standings client.

Set FOOTBALL_API_KEY in your shell before using this module. It deliberately
does not store credentials in source code.
"""

import os

import pandas as pd
import requests


def fetch_live_standings(api_key: str | None = None) -> pd.DataFrame:
    """Fetch current Premier League standings from football-data.org."""
    token = api_key or os.getenv("FOOTBALL_API_KEY")
    if not token:
        raise RuntimeError("Set FOOTBALL_API_KEY before fetching live standings.")

    response = requests.get(
        "https://api.football-data.org/v4/competitions/PL/standings",
        headers={"X-Auth-Token": token},
        timeout=20,
    )
    response.raise_for_status()
    table = response.json()["standings"][0]["table"]
    return pd.DataFrame(
        {
            "Team": row["team"]["name"],
            "played": row["playedGames"],
            "goals": row["goalsFor"],
            "rank": row["position"],
        }
        for row in table
    )


if __name__ == "__main__":
    print(fetch_live_standings().to_string(index=False))
