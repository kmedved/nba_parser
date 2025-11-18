import pandas as pd
from nba_parser.pbp import PbP


def test_player_box_glossary_basics():
    pbp_df = pd.read_csv("test/20700233.csv")
    pbp_df["season"] = 2008
    pbp = PbP(pbp_df)

    box = pbp.player_box_glossary()

    required_cols = [
        "Game_SingleGame",
        "Team_SingleGame",
        "NbaDotComID",
        "PTS",
        "Minutes",
        "POSS",
        "POSS_OFF",
        "POSS_DEF",
        "TSAttempts",
        "TSpct",
        "PTS_100p",
        "Pace",
    ]
    for col in required_cols:
        assert col in box.columns

    assert ((box["POSS_OFF"] + box["POSS_DEF"]) == box["POSS"]).all()
    assert (box["BLK"] == box["BLK_Team"] + box["BLK_Opp"]).all()

    team_minutes = box.groupby("team_id")["Minutes"].sum()
    game_minutes = pbp_df["seconds_elapsed"].max() / 60.0
    for _, minutes in team_minutes.items():
        assert abs(minutes - game_minutes * 5) < 1.0

    team_points = pbp._point_calc_team()[["team_id", "points_for"]]
    team_points_map = dict(zip(team_points["team_id"], team_points["points_for"]))

    for team_id, pts_for in team_points_map.items():
        team_total_for = box.loc[box["team_id"] == team_id, "OnCourt_Team_Points"].sum()
        assert abs(team_total_for - pts_for * 5) < 1e-6

        opp_points = sum(p for t, p in team_points_map.items() if t != team_id)
        team_total_against = box.loc[box["team_id"] == team_id, "OnCourt_Opp_Points"].sum()
        assert abs(team_total_against - opp_points * 5) < 1e-6

    pbg = pbp.playerbygamestats()
    merged = box.merge(
        pbg[["game_id", "team_id", "player_id", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "points"]],
        on=["game_id", "team_id", "player_id"],
        suffixes=("", "_old"),
    )
    for col in ["fgm", "fga", "tpm", "tpa", "ftm", "fta", "points"]:
        assert (merged[col] == merged[f"{col}_old"]).all()


def test_player_box_glossary_cdn_invariants():
    """
    Validates possession and exposure logic against a CDN-era game file,
    ensuring key invariants (minutes and on-court points) hold true.
    """
    pbp_df = pd.read_csv("test/21900151.csv")
    pbp_df["season"] = 2020
    pbp = PbP(pbp_df)

    box = pbp.player_box_glossary()

    # 1. Validate Minutes Invariant
    team_minutes = box.groupby("team_id")["Minutes"].sum()
    game_minutes = pbp.df["seconds_elapsed"].max() / 60.0
    # A small tolerance is needed for floating point comparisons
    for _, minutes in team_minutes.items():
        assert abs(minutes - game_minutes * 5) < 1.0

    # 2. Validate On-Court Points Invariant
    team_points = pbp._point_calc_team()[["team_id", "points_for"]]
    team_points_map = dict(zip(team_points["team_id"], team_points["points_for"]))

    for team_id, pts_for in team_points_map.items():
        # Check points scored by the player's team while they were on court
        team_total_for = box.loc[box["team_id"] == team_id, "OnCourt_Team_Points"].sum()
        assert abs(team_total_for - pts_for * 5) < 1e-6

        # Check points scored by the opponent while they were on court
        opp_points = sum(p for t, p in team_points_map.items() if t != team_id)
        team_total_against = box.loc[box["team_id"] == team_id, "OnCourt_Opp_Points"].sum()
        assert abs(team_total_against - opp_points * 5) < 1e-6
