import pandas as pd
import pytest

import nba_parser as npar


@pytest.fixture(scope="session")
def setup():
    """
    function for test setup and teardown
    """
    files = [
        "21900002.csv",
        "21900025.csv",
        "21900040.csv",
        "21900054.csv",
        "21900074.csv",
        "21900088.csv",
        "21900100.csv",
        "21900126.csv",
        "21900139.csv",
        "21900151.csv",
    ]
    pbp_dfs = [pd.read_csv(f"test/{f}") for f in files]
    for pbp_df in pbp_dfs:
        for col in [
            "game_id",
            "team_id",
            "home_team_id",
            "away_team_id",
            *(f"home_player_{i}_id" for i in range(1, 6)),
            *(f"away_player_{i}_id" for i in range(1, 6)),
        ]:
            if col in pbp_df.columns:
                pbp_df[col] = pd.to_numeric(pbp_df[col], errors="coerce").fillna(0).astype(int)
    pbp_dfs = [npar.PbP(pbp_df) for pbp_df in pbp_dfs]
    pbg_dfs = [pbp_df.playerbygamestats() for pbp_df in pbp_dfs]
    tbg_dfs = [pbp_df.teambygamestats() for pbp_df in pbp_dfs]

    yield pbg_dfs, tbg_dfs, pbp_dfs


def test_rapm_possessions_runs(setup):
    """
    Simple smoke test: rapm_possessions should run across multiple games.
    """

    _, _, pbp_dfs = setup
    rapm_possessions = pd.concat([pbp.rapm_possessions() for pbp in pbp_dfs])

    assert isinstance(rapm_possessions, pd.DataFrame)
    assert not rapm_possessions.empty
