from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def load_pbp(
    game_id: Union[int, str], csv_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Load a play-by-play dataframe.

    If ``csv_path`` is provided and points to an existing file, the data is
    loaded from that CSV. Otherwise the function will attempt to pull data from
    the official NBA Stats API using ``nba_api``.

    Parameters
    ----------
    game_id:
        The NBA game id to load. When ``csv_path`` is ``None`` the id is passed
        to :class:`nba_api.stats.endpoints.PlayByPlayV2`.
    csv_path:
        Optional path to a CSV file. If present, the CSV is read instead of
        calling the API.

    Returns
    -------
    pandas.DataFrame
        DataFrame in the format expected by :class:`~nba_parser.pbp.PbP`.
    """

    if csv_path is not None and Path(csv_path).exists():
        return pd.read_csv(csv_path)

    try:
        from nba_api.stats.endpoints import PlayByPlayV2, BoxScoreSummaryV2
    except Exception as exc:  # pragma: no cover - import failure
        raise ImportError(
            "nba_api is required to fetch play-by-play data"
        ) from exc

    game_id_str = str(game_id).zfill(10)

    pbp_endpoint = PlayByPlayV2(game_id=game_id_str, start_period=1, end_period=14)
    df = pbp_endpoint.get_data_frames()[0]

    df.columns = [c.lower() for c in df.columns]

    jump_ball = df[df["eventmsgtype"] == 10].iloc[0]

    if pd.notnull(jump_ball["homedescription"]):
        home_team_abbrev = jump_ball["player1_team_abbreviation"]
        away_team_abbrev = jump_ball["player2_team_abbreviation"]
        home_team_id = jump_ball["player1_team_id"]
        away_team_id = jump_ball["player2_team_id"]
    else:
        home_team_abbrev = jump_ball["player2_team_abbreviation"]
        away_team_abbrev = jump_ball["player1_team_abbreviation"]
        home_team_id = jump_ball["player2_team_id"]
        away_team_id = jump_ball["player1_team_id"]

    df["home_team_abbrev"] = home_team_abbrev
    df["away_team_abbrev"] = away_team_abbrev
    df["home_team_id"] = home_team_id
    df["away_team_id"] = away_team_id

    summary_endpoint = BoxScoreSummaryV2(game_id=game_id_str)
    game_date_str = summary_endpoint.get_data_frames()[0]["GAME_DATE_EST"].iloc[0].split("T")[0]
    df["game_date"] = game_date_str

    season_code = int(game_id_str[3:5])
    if season_code >= 99:
        season = 2000 + season_code - 98
    else:
        season = 2000 + season_code + 1
    df["season"] = season

    return df
