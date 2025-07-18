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
        to :class:`nba_api.stats.endpoints.playbyplayv3.PlayByPlayV3`.
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
        from nba_api.stats.endpoints import playbyplayv3
    except Exception as exc:  # pragma: no cover - import failure
        raise ImportError(
            "nba_api is required to fetch play-by-play data"
        ) from exc

    pbp = playbyplayv3.PlayByPlayV3(str(game_id))
    df = pbp.get_data_frames()[0]

    # standardise column names to match the lower case convention used in tests
    df.columns = [c.lower() for c in df.columns]

    rename_map = {
        "hometeamid": "home_team_id",
        "visitorteamid": "away_team_id",
        "hometeamtricode": "home_team_abbrev",
        "visitorteamtricode": "away_team_abbrev",
    }
    df = df.rename(columns=rename_map)

    # Some columns expected by PbP are not provided directly by the API.
    for col in [
        "event_team",
        "event_type_de",
        "shot_type_de",
        "shot_made",
        "is_block",
        "shot_type",
        "seconds_elapsed",
        "event_length",
        "is_three",
        "points_made",
        "is_o_rebound",
        "is_d_rebound",
        "is_turnover",
        "is_steal",
        "foul_type",
        "is_putback",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    return df
