import pandas as pd
from nba_parser.pbp import PbP


def test_player_box_glossary_basics():
    pbp_df = pd.read_csv("test/20700233.csv")
    pbp_df["season"] = 2008
    pbp = PbP(pbp_df)

    box = pbp.player_box_glossary()

    assert ((box["POSS_OFF"] + box["POSS_DEF"]) == box["POSS"]).all()
    assert (box["BLK"] == box["BLK_Team"] + box["BLK_Opp"]).all()

    team_minutes = box.groupby("team_id")["Minutes"].sum()
    game_minutes = pbp_df["seconds_elapsed"].max() / 60.0
    for _, minutes in team_minutes.items():
        assert abs(minutes - game_minutes * 5) < 1.0

    team_points = pbp._point_calc_team()[["team_id", "points_for"]]
    for _, row in team_points.iterrows():
        team_total = box.loc[box["team_id"] == row["team_id"], "OnCourt_Team_Points"].sum()
        assert abs(team_total - row["points_for"] * 5) < 1e-6

    pbg = pbp.playerbygamestats()
    merged = box.merge(
        pbg[["game_id", "team_id", "player_id", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "points"]],
        on=["game_id", "team_id", "player_id"],
        suffixes=("", "_old"),
    )
    for col in ["fgm", "fga", "tpm", "tpa", "ftm", "fta", "points"]:
        assert (merged[col] == merged[f"{col}_old"]).all()
