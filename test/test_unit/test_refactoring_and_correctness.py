import pandas as pd
import numpy as np
import pytest
from nba_parser.pbp import PbP
from nba_parser.box_glossary import (
    accumulate_player_counts,
    annotate_events,
    build_player_box,
)


@pytest.fixture(scope="module")
def pbp_v2_game_instance():
    """Provides a parsed PbP object for a V2 game."""
    pbp_df = pd.read_csv("test/20700233.csv")
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
    pbp_df["season"] = 2008
    return PbP(pbp_df)


def test_string_ids_are_normalized():
    """Ensure PbP and downstream box glossary coerce string IDs to ints."""

    pbp_df = pd.read_csv("test/20700233.csv")
    id_cols = [
        "game_id",
        "team_id",
        "home_team_id",
        "away_team_id",
        *(f"home_player_{i}_id" for i in range(1, 6)),
        *(f"away_player_{i}_id" for i in range(1, 6)),
        "player1_id",
        "player2_id",
        "player3_id",
        "player1_team_id",
        "player2_team_id",
        "player3_team_id",
    ]
    for col in id_cols:
        if col in pbp_df.columns:
            pbp_df[col] = pbp_df[col].astype(str)
    pbp_df["season"] = 2008

    box = PbP(pbp_df).player_box_glossary()

    assert pd.api.types.is_integer_dtype(box["game_id"])
    assert pd.api.types.is_integer_dtype(box["team_id"])
    assert pd.api.types.is_integer_dtype(box["player_id"])


def test_decoupling_from_pbg_stats(pbp_v2_game_instance):
    """
    Validates Change 1:
    Ensures box score stats are correct after removing the pbg_stats override.
    Also checks that players with minutes but no stats are now included.
    """
    box = pbp_v2_game_instance.player_box_glossary()
    
    # Check a known player's stats (Carmelo Anthony)
    melo = box[box["player_id"] == 2546]
    assert melo["FGM"].iloc[0] == 7
    assert melo["FGA"].iloc[0] == 15
    assert melo["ThreePM"].iloc[0] == 0
    assert melo["ThreePA"].iloc[0] == 1
    assert melo["FTM"].iloc[0] == 10
    assert melo["FTA"].iloc[0] == 11
    assert melo["PTS"].iloc[0] == 24

    # This test file doesn't have a zero-stat player, but we confirm
    # that the number of players matches the number with non-zero minutes.
    players_in_box = len(box)
    players_with_minutes = len(box[box["Minutes"] > 0])
    assert players_in_box == players_with_minutes


def test_vectorized_helpers_match_row_wise():
    """
    Validates Change 3:
    Ensures the new vectorized helpers for and-one and shot-zone produce
    the same output as the old row-wise logic.
    """
    df = pd.DataFrame({
        "qualifiers": ["['and1']", "['other']", "['and-one']"],
        "shot_distance": [2, 8, 25],
        "area": ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range"],
        "is_fg_attempt": [True, True, False] # Last one is not a shot
    })

    # Old logic for comparison
    from nba_parser.box_glossary import classify_shot_zone
    df["shot_zone_old"] = df.apply(lambda r: classify_shot_zone(r.get("shot_distance"), r.get("area")), axis=1)
    df["is_and_one_old"] = df["qualifiers"].astype(str).str.lower().str.contains(r"and[ -]?one|and1")

    # New vectorized logic
    annotated = annotate_events(df.copy())

    pd.testing.assert_series_equal(annotated["shot_zone"], df["shot_zone_old"], check_names=False)
    pd.testing.assert_series_equal(annotated["is_and_one"], df["is_and_one_old"], check_names=False)


def test_assist_attribution_fix():
    """
    Validates Change 4:
    Ensures assists are not credited on missed shots.
    """
    pbp_events = pd.DataFrame({
        "game_id": ["g1", "g1"],
        "player1_id": [101, 101], # Shooter
        "player1_team_id": [1, 1],
        "player2_id": [202, 202], # Helper on make, defender on miss
        "team_id": [1, 1],
        "is_fg_attempt": [True, True],
        "is_fg_make": [True, False], # One make, one miss
        "assist_id": [202, np.nan], # Assist on make, no assist on miss
    })
    
    counts_df = accumulate_player_counts(pbp_events)
    
    # Player 202 should have exactly 1 assist from the made shot
    assister_stats = counts_df[counts_df["player_id"] == 202]
    assert not assister_stats.empty
    assert assister_stats["AST"].iloc[0] == 1


def test_metadata_passthrough(pbp_v2_game_instance):
    """
    Validates Change 5:
    Ensures that pre-existing metadata columns like 'Starts' are preserved
    and not overwritten with defaults.
    """
    # Create player_meta where a player is manually marked as a starter
    player_meta = pd.DataFrame({
        "player_id": [200784],
        "season": [2008],
        "Starts": [1] # Override default
    })
    
    box = pbp_v2_game_instance.player_box_glossary(player_meta=player_meta)
    
    diawara_row = box[box["player_id"] == 200784]
    
    # The 'Starts' value from player_meta should be preserved
    assert not diawara_row.empty
    assert diawara_row["Starts"].iloc[0] == 1


def test_player_box_glossary_cdn_game():
    """
    Smoke + invariants test on a CDN-era game to ensure the new pipeline
    works for modern data as well.
    """
    pbp_df = pd.read_csv("test/0021900151_cdn.csv")
    # Ensure season column exists; adjust to correct season if needed.
    if "season" not in pbp_df.columns:
        pbp_df["season"] = 2020

    pbp = PbP(pbp_df)
    box = pbp.player_box_glossary()

    # Minutes sanity: each team should be ~game_minutes * 5
    total_minutes = pbp_df["seconds_elapsed"].max() / 60.0
    team_minutes = box.groupby("team_id")["Minutes"].sum()
    for minutes in team_minutes:
        assert abs(minutes - total_minutes * 5.0) < 1.0

    # On-court scoring invariants: same as existing v2 tests
    team_points = pbp._point_calc_team()[["team_id", "points_for"]]
    team_points_map = dict(zip(team_points["team_id"], team_points["points_for"]))

    for team_id, pts_for in team_points_map.items():
        on_for = box.loc[box["team_id"] == team_id, "OnCourt_Team_Points"].sum()
        opp_pts = sum(p for t, p in team_points_map.items() if t != team_id)
        on_against = box.loc[box["team_id"] == team_id, "OnCourt_Opp_Points"].sum()

        assert abs(on_for - pts_for * 5.0) < 1e-6
        assert abs(on_against - opp_pts * 5.0) < 1e-6
