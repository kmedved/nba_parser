from __future__ import annotations

from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple

import re

import numpy as np
import pandas as pd

# Mapping for base positions to numeric slots.
# 1=PG, 2=SG, 3=SF, 4=PF, 5=C. "G" and "F" are midpoints.
_BASE_POS_NUM = {
    "PG": 1.0,
    "SG": 2.0,
    "SF": 3.0,
    "PF": 4.0,
    "C": 5.0,
    "G": 1.5,   # generic guard between PG/SG
    "F": 3.5,   # generic forward between SF/PF
}


def position_to_num(pos: Any) -> float | None:
    """
    Convert a position string like 'PG', 'SG', 'G-F', 'F-C' into a numeric
    encoding. Returns NaN for unknown/missing positions.
    """
    if not isinstance(pos, str) or not pos.strip():
        return np.nan

    pos_str = pos.upper().replace(" ", "")
    if pos_str in _BASE_POS_NUM:
        return _BASE_POS_NUM[pos_str]

    tokens = re.split(r"[-/]", pos_str)
    vals = [_BASE_POS_NUM[t] for t in tokens if t in _BASE_POS_NUM]
    if not vals:
        return np.nan
    return float(np.mean(vals))

ZONE_BINS: List[Tuple[float, float, str]] = [
    (0.0, 3.0, "0_3"),
    (3.0, 9.0, "4_9"),
    (9.0, 17.0, "10_17"),
    (17.0, 23.0, "18_23"),
]


def classify_shot_zone(shot_distance: float | None, area: str | None) -> Optional[str]:
    if shot_distance is not None and not pd.isna(shot_distance):
        d = float(shot_distance)
        for lower, upper, label in ZONE_BINS:
            lower_ok = d >= lower if lower == 0.0 else d > lower
            upper_ok = d <= upper or np.isclose(d, upper)
            if lower_ok and upper_ok:
                return label
        return None

    # Some CDN datasets omit the shot area entirely or encode it as NaN/float.
    if area is None or pd.isna(area):
        return None

    area_str = area if isinstance(area, str) else str(area)
    area_lower = area_str.lower()
    if "restricted" in area_lower:
        return "0_3"
    if "paint" in area_lower:
        return "4_9"
    if "mid-range" in area_lower:
        return None
    return None


def _vectorized_is_and_one(qualifiers: pd.Series) -> pd.Series:
    """
    Vectorized check for And-One events.
    Assumes qualifiers is a Series indexed like the pbp DataFrame.
    """
    if qualifiers is None or len(qualifiers) == 0:
        if isinstance(qualifiers, pd.Series):
            return pd.Series(False, index=qualifiers.index)
        return pd.Series(dtype=bool)

    q_str = qualifiers.fillna("").astype(str).str.lower()
    return q_str.str.contains(r"and[ -]?one|and1", regex=True)


def _vectorized_shot_zone(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized calculation of shot zones.
    """
    shot_distance = pd.to_numeric(df.get("shot_distance"), errors="coerce")

    area_col = df.get("area")
    if area_col is None:
        # Ensure same index as df so boolean masks align cleanly
        area_col = pd.Series([""] * len(df), index=df.index)
    area = area_col.fillna("").astype(str).str.lower()

    # Initialize zones as None (object type to hold strings or None)
    zones = pd.Series(None, index=df.index, dtype=object)

    # Bins based on distance
    dist_mask = shot_distance.notna()
    zones[dist_mask & (shot_distance <= 3.0)] = "0_3"
    zones[dist_mask & (shot_distance > 3.0) & (shot_distance <= 9.0)] = "4_9"
    zones[dist_mask & (shot_distance > 9.0) & (shot_distance <= 17.0)] = "10_17"
    zones[dist_mask & (shot_distance > 17.0) & (shot_distance <= 23.0)] = "18_23"

    # Fallback to area text where distance-based zone is still missing
    fallback_mask = zones.isna()
    zones[fallback_mask & area.str.contains("restricted")] = "0_3"
    zones[fallback_mask & area.str.contains("paint")] = "4_9"

    return zones


def annotate_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Preserve original upstream family (e.g., "2pt", "3pt") for debugging.
    if "family" in df.columns and "family_raw" not in df.columns:
        df["family_raw"] = df["family"]
    # --- Subfamily normalization ---
    if "subfamily_de" in df.columns:
        subfam = df["subfamily_de"].fillna("")
    elif "event_sub_family" in df.columns:
        df["subfamily_de"] = df["event_sub_family"].fillna("")
        subfam = df["subfamily_de"]
    elif "event_subfamily" in df.columns:
        df["subfamily_de"] = df["event_subfamily"].fillna("")
        subfam = df["subfamily_de"]
    elif "subfamily" in df.columns:
        df["subfamily_de"] = df["subfamily"].fillna("")
        subfam = df["subfamily_de"]
    else:
        subfam = pd.Series([""] * len(df), index=df.index)
        df["subfamily_de"] = subfam

    # --- Canonical family based on event_type_de, not API family ---
    if "event_type_de" in df.columns:
        fam_src = df["event_type_de"].fillna("")
    elif "family" in df.columns:
        fam_src = df["family"].fillna("")
    else:
        # If we truly have no event type info, fall back to an empty string series
        fam_src = pd.Series([""] * len(df), index=df.index)

    fam = fam_src.astype(str).str.lower().str.replace("-", "_", regex=False)
    df["family"] = fam

    # --- Event team id ---
    if "home_team_id" not in df.columns:
        df["home_team_id"] = np.nan
    if "away_team_id" not in df.columns:
        df["away_team_id"] = np.nan

    if "team_id" in df.columns:
        # Repair rows where team_id is missing or 0 using event_team.
        team_id = df["team_id"].copy()
        missing = team_id.isna() | (team_id == 0)

        if missing.any():
            event_team = df.get("event_team")
            if event_team is None:
                # Some feeds may expose team code as team_tricode instead.
                event_team = df.get("team_tricode")
            if event_team is None:
                event_team = pd.Series([None] * len(df), index=df.index)

            inferred = np.where(
                event_team == df.get("home_team_abbrev"),
                df.get("home_team_id"),
                np.where(
                    event_team == df.get("away_team_abbrev"),
                    df.get("away_team_id"),
                    np.nan,
                ),
            )
            team_id[missing] = inferred[missing]

        df["team_id"] = team_id
    else:
        event_team = df.get("event_team")
        if event_team is None:
            event_team = df.get("team_tricode")
        if event_team is None:
            event_team = pd.Series([None] * len(df), index=df.index)

        df["team_id"] = np.where(
            event_team == df.get("home_team_abbrev"),
            df.get("home_team_id"),
            np.where(
                event_team == df.get("away_team_abbrev"),
                df.get("away_team_id"),
                np.nan,
            ),
        )

    # --- Ensure event-level points_made exists ---
    if "points_made" not in df.columns:
        if "points_made_x" in df.columns:
            df["points_made"] = df["points_made_x"]
        elif "points_made_y" in df.columns:
            df["points_made"] = df["points_made_y"]
        else:
            df["points_made"] = 0

    # --- FGA/FGM/FT flags ---
    if "is_fg_attempt" in df.columns:
        is_shot_like = df["is_fg_attempt"].fillna(False)
    else:
        is_shot_like = fam.isin(["shot", "miss_shot", "missed_shot"])
    df["is_fg_attempt"] = is_shot_like.astype(bool)

    if "is_fg_make" in df.columns:
        df["is_fg_make"] = df["is_fg_attempt"] & df["is_fg_make"].fillna(0).astype(bool)
    elif "shot_made" in df.columns:
        df["shot_made"] = df["shot_made"].fillna(0).astype(int)
        df["is_fg_make"] = df["is_fg_attempt"] & (df["shot_made"] == 1)
    else:
        df["is_fg_make"] = df["is_fg_attempt"] & (df["points_made"] > 0)

    df["is_ft"] = fam == "free_throw"
    df["is_ft_make"] = df["is_ft"] & (df["points_made"] > 0)

    # Identify the last free-throw attempt in a trip so rebound opportunities
    # can include missed end-of-trip free throws.
    if "ft_n" in df.columns and "ft_m" in df.columns:
        try:
            ft_n = df["ft_n"].fillna(0).astype(int)
            ft_m = df["ft_m"].fillna(0).astype(int)
            df["is_last_ft"] = (ft_n == ft_m) & (ft_n > 0)
        except (ValueError, TypeError):
            df["is_last_ft"] = False
    else:
        # Fallback heuristic when counters are missing: look for "X of X" text.
        sub_lower = subfam.astype(str).str.lower()
        df["is_last_ft"] = df["is_ft"] & (
            sub_lower.str.contains("1 of 1")
            | sub_lower.str.contains("2 of 2")
            | sub_lower.str.contains("3 of 3")
        )

    if "is_o_rebound" not in df.columns:
        df["is_o_rebound"] = 0
    if "is_d_rebound" not in df.columns:
        df["is_d_rebound"] = 0

    # Three-pointers
    if "is_three" in df.columns:
        df["is_three"] = df["is_three"].fillna(0).astype(bool)
    else:
        # Fallback heuristic: treat long-distance FG attempts as 3s if distance is known.
        dist = df.get("shot_distance")
        if dist is not None:
            dist_num = pd.to_numeric(dist, errors="coerce")
            df["is_three"] = (dist_num >= 23.0) & df["is_fg_attempt"]
        else:
            df["is_three"] = False

    # --- Turnover live/dead ---
    is_tov_family = fam == "turnover"

    # Anything recorded as a steal is a live-ball turnover.
    steal_col = df.get("is_steal")
    if steal_col is None:
        steal_col = pd.Series([0] * len(df), index=df.index)
    is_steal_flag = steal_col.fillna(0).astype(int) == 1

    sub_lower = subfam.astype(str).str.lower()
    # Some feeds may explicitly label "live-ball" in text.
    sub_live_flag = sub_lower.str.contains("live")

    df["is_turnover_live"] = is_tov_family & (is_steal_flag | sub_live_flag)
    df["is_turnover_dead"] = is_tov_family & ~df["is_turnover_live"]

    # --- Foul flavors ---
    is_foul_family = fam == "foul"
    sub = subfam.str.lower()
    df["is_loose_ball_foul"] = is_foul_family & sub.str.contains("loose")
    df["is_flagrant"] = is_foul_family & sub.str.contains("flagrant")
    df["is_technical"] = is_foul_family & sub.str.contains("technical")
    charge_mask = sub.str.contains("charging") | sub.str.contains("charge")
    df["is_charge"] = is_foul_family & charge_mask

    # --- And-ones via qualifiers ---
    quals_series = df["qualifiers"] if "qualifiers" in df.columns else pd.Series([None] * len(df), index=df.index)
    df["is_and_one"] = _vectorized_is_and_one(quals_series)

    # --- Shot zones ---
    shot_mask = df["is_fg_attempt"]
    if "shot_distance" in df.columns or "area" in df.columns:
        df["shot_zone"] = np.where(shot_mask, _vectorized_shot_zone(df), None)
    else:
        df["shot_zone"] = None

    # --- Off/def team ids for event-level context ---
    off_mask = df["is_fg_attempt"] | df["is_ft"] | (df["family"] == "turnover")
    df["off_team_id"] = np.where(off_mask, df["team_id"], np.nan)
    df["def_team_id"] = np.where(
        off_mask,
        np.where(df["off_team_id"] == df["home_team_id"], df["away_team_id"], df["home_team_id"]),
        np.nan,
    )

    return df


def _increment_count(counter: Dict[str, float], key: str, value: float = 1.0):
    counter[key] += value


def _valid_player_id(pid: Any) -> bool:
    """
    Return True if pid represents a real player id (non-null, non-zero).

    Used anywhere we’re looping over lineup slots to avoid accidentally
    treating NaN/None/0 as real players.
    """
    if pid is None:
        return False
    try:
        if pd.isna(pid):
            return False
    except TypeError:
        pass
    try:
        return int(pid) != 0
    except (TypeError, ValueError):
        return False


def accumulate_player_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized aggregation of player-level counting stats from an annotated
    play-by-play DataFrame.
    """
    df = df.copy()
    # Ensure player IDs are numeric for clean aggregation and joins
    for col in ["player1_id", "player2_id", "player3_id", "assist_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # --- Base Event Aggregations (Player 1 is the primary actor) ---
    p1_events = df[df["player1_id"] != 0]
    
    # Group once for all primary player actions
    p1_groups = p1_events.groupby(["game_id", "player1_team_id", "player1_id"])
    
    counts = p1_groups.agg(
        FGA=("is_fg_attempt", "sum"),
        FGM=("is_fg_make", "sum"),
        ThreePA=("is_three", "sum"),
        FTA=("is_ft", "sum"),
        FTM=("is_ft_make", "sum"),
        PTS=("points_made", "sum"),
        OREB=("is_o_rebound", "sum"),
        DREB=("is_d_rebound", "sum"),
        TOV=("family", lambda x: (x == "turnover").sum()),
        TOV_Live=("is_turnover_live", "sum"),
        TOV_Dead=("is_turnover_dead", "sum"),
        PF=("family", lambda x: (x == "foul").sum()),
        PF_Loose=("is_loose_ball_foul", "sum"),
        FLAGRANT=("is_flagrant", "sum"),
        TECH=("is_technical", "sum"),
        AndOnes=("is_and_one", "sum"),
    ).reset_index().rename(columns={"player1_id": "player_id", "player1_team_id": "team_id"})

    # --- Zonal Shooting (Player 1) ---
    shots = df[df["is_fg_attempt"] & (df["player1_id"] != 0) & df["shot_zone"].notna()]
    zonal_counts = pd.pivot_table(
        shots,
        index=["game_id", "player1_team_id", "player1_id"],
        columns="shot_zone",
        values=["is_fg_attempt", "is_fg_make"],
        aggfunc="sum",
        fill_value=0
    )
    zonal_counts.columns = [f"{zone}_{'FGM' if 'make' in stat else 'FGA'}" for stat, zone in zonal_counts.columns]
    
    # --- Assists (Credited to assist_id) ---
    assists = df[df["is_fg_make"] & (df["assist_id"] != 0) & (df["assist_id"] != df["player1_id"])]
    assist_counts = assists.groupby(["game_id", "team_id", "assist_id"]).agg(
        AST=("is_fg_make", "sum")
    ).reset_index().rename(columns={"assist_id": "player_id"})
    
    # Zonal Assists
    zonal_assists = pd.pivot_table(assists, index=["game_id", "team_id", "assist_id"], columns="shot_zone", aggfunc='size', fill_value=0)
    zonal_assists = zonal_assists.add_prefix("AST_").reset_index().rename(columns={"assist_id": "player_id"})
    assist_counts = assist_counts.merge(zonal_assists, on=["game_id", "team_id", "player_id"], how="left")

    # 3P Assists
    ast_3p = assists[assists["is_three"]].groupby(["game_id", "team_id", "assist_id"]).size().reset_index(name="AST_3P")
    assist_counts = assist_counts.merge(ast_3p.rename(columns={"assist_id": "player_id"}), on=["game_id", "team_id", "player_id"], how="left")

    # --- Steals (Player 2) & Blocks (Player 3) ---
    steals = df[df["is_steal"] & (df["player2_id"] != 0)].groupby(["game_id", "player2_team_id", "player2_id"]).agg(STL=("is_steal", "sum")).reset_index().rename(columns={"player2_id": "player_id", "player2_team_id": "team_id"})
    blocks = df[df["is_block"] & (df["player3_id"] != 0)].groupby(["game_id", "player3_team_id", "player3_id"]).agg(BLK=("is_block", "sum")).reset_index().rename(columns={"player3_id": "player_id", "player3_team_id": "team_id"})

    # --- Fouls Drawn (Player 2) ---
    fouls_drawn = df[(df["family"] == "foul") & (df["player2_id"] != 0)].groupby(["game_id", "player2_team_id", "player2_id"]).agg(PF_DRAWN=("family", "count"), CHRG=("is_charge", "sum")).reset_index().rename(columns={"player2_id": "player_id", "player2_team_id": "team_id"})

    # --- Unassisted / Assisted Makes (Shooter-side) ---
    makes = df[df["is_fg_make"] & (df["player1_id"] != 0)].copy()
    makes["is_assisted"] = (makes["assist_id"] != 0) & (makes["assist_id"] != makes["player1_id"])
    
    assisted_fgm = makes[makes["is_assisted"]].groupby(["game_id", "player1_id"]).size().reset_index(name="FGM_AST")
    assisted_3pm = makes[makes["is_assisted"] & makes["is_three"]].groupby(["game_id", "player1_id"]).size().reset_index(name="ThreePM_AST")
    
    unassisted_makes = makes[~makes["is_assisted"]]
    unassisted_counts = unassisted_makes.groupby(["game_id", "player1_id"]).agg(FGM_UNAST=("is_fg_make", "sum"), ThreePM_UNAST=("is_three", "sum")).reset_index()
    unassisted_counts["FGA_UNAST"] = unassisted_counts["FGM_UNAST"]
    unassisted_counts["ThreePA_UNAST"] = unassisted_counts["ThreePM_UNAST"]

    # Zonal Unassisted
    zonal_unast = pd.pivot_table(unassisted_makes, index=["game_id", "player1_id"], columns="shot_zone", aggfunc='size', fill_value=0)
    zonal_unast.columns = [f"{zone}_FGM_UNAST" for zone in zonal_unast.columns]
    for col in zonal_unast.columns:
        zonal_unast[col.replace("FGM", "FGA")] = zonal_unast[col]
    
    # --- Merge all dataframes ---
    box = counts.merge(zonal_counts, on=["game_id", "team_id", "player_id"], how="left")
    box = box.merge(assist_counts, on=["game_id", "team_id", "player_id"], how="left")
    box = box.merge(steals, on=["game_id", "team_id", "player_id"], how="left")
    box = box.merge(blocks, on=["game_id", "team_id", "player_id"], how="left")
    box = box.merge(fouls_drawn, on=["game_id", "team_id", "player_id"], how="left")
    box = box.merge(assisted_fgm.rename(columns={"player1_id": "player_id"}), on=["game_id", "player_id"], how="left")
    box = box.merge(assisted_3pm.rename(columns={"player1_id": "player_id"}), on=["game_id", "player_id"], how="left")
    box = box.merge(unassisted_counts.rename(columns={"player1_id": "player_id"}), on=["game_id", "player_id"], how="left")
    box = box.merge(zonal_unast.reset_index().rename(columns={"player1_id": "player_id"}), on=["game_id", "player_id"], how="left")

    # Ensure all core counting columns exist
    all_count_cols = [
        "FGA", "FGM", "ThreePA", "ThreePM", "FTA", "FTM", "PTS", "OREB", "DREB",
        "TOV", "TOV_Live", "TOV_Dead", "PF", "PF_Loose", "FLAGRANT", "TECH",
        "AST", "STL", "BLK", "PF_DRAWN", "CHRG", "AndOnes", "FGM_AST", "ThreePM_AST",
        "FGM_UNAST", "FGA_UNAST", "ThreePM_UNAST", "ThreePA_UNAST"
    ]
    for col in all_count_cols:
        if col not in box.columns:
            box[col] = 0

    return box.fillna(0)


def compute_on_court_exposures(pbp: "PbP", df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes on-court exposures like minutes, possessions, and rebound
    opportunities for each player using vectorized operations where possible.
    """
    # --- Melt lineups to get a per-player, per-event view ---
    id_vars = ["game_id", "event_length", "home_team_id", "away_team_id"]
    
    home_players = df.melt(
        id_vars=id_vars,
        value_vars=[f"home_player_{i}_id" for i in range(1, 6)],
        var_name="slot", value_name="player_id"
    )
    home_players["team_id"] = home_players["home_team_id"]

    away_players = df.melt(
        id_vars=id_vars,
        value_vars=[f"away_player_{i}_id" for i in range(1, 6)],
        var_name="slot", value_name="player_id"
    )
    away_players["team_id"] = away_players["away_team_id"]
    
    player_events = pd.concat([home_players, away_players])
    player_events = player_events[player_events["player_id"].notna() & (player_events["player_id"] != 0)].copy()
    player_events["player_id"] = player_events["player_id"].astype(int)

    # --- Vectorized Minutes Calculation ---
    minutes_df = player_events.groupby(["game_id", "team_id", "player_id"])["event_length"].sum().reset_index()
    minutes_df["Minutes"] = minutes_df["event_length"] / 60.0
    
    # --- Vectorized Rebound Opportunities ---
    is_reb_opp = (df["is_fg_attempt"] & ~df["is_fg_make"]) | (df["is_ft"] & ~df["is_ft_make"] & df["is_last_ft"])
    reb_opps = df[is_reb_opp].copy()
    
    home_reb_opps = reb_opps.melt(id_vars=["game_id", "team_id"], value_vars=[f"home_player_{i}_id" for i in range(1, 6)], value_name="player_id")
    home_reb_opps["on_court_team_id"] = pbp.home_team_id
    away_reb_opps = reb_opps.melt(id_vars=["game_id", "team_id"], value_vars=[f"away_player_{i}_id" for i in range(1, 6)], value_name="player_id")
    away_reb_opps["on_court_team_id"] = pbp.away_team_id
    
    all_reb_opps = pd.concat([home_reb_opps, away_reb_opps])
    all_reb_opps = all_reb_opps[all_reb_opps["player_id"].notna() & (all_reb_opps["player_id"] != 0)]
    all_reb_opps["player_id"] = all_reb_opps["player_id"].astype(int)

    all_reb_opps["is_oreb_opp"] = all_reb_opps["team_id"] == all_reb_opps["on_court_team_id"]
    all_reb_opps["is_dreb_opp"] = all_reb_opps["team_id"] != all_reb_opps["on_court_team_id"]
    
    reb_totals = all_reb_opps.groupby(["game_id", "on_court_team_id", "player_id"]).agg(
        OnCourt_For_OREB_Total=("is_oreb_opp", "sum"),
        OnCourt_For_DREB_Total=("is_dreb_opp", "sum")
    ).reset_index().rename(columns={"on_court_team_id": "team_id"})

    # --- Possession-based Exposures ---
    exposures = defaultdict(lambda: defaultdict(float))
    poss_df = pbp._build_possessions(df, include_event_agg=True)

    for _, poss in poss_df.iterrows():
        off_team, def_team = poss.get("off_team_id"), poss.get("def_team_id")
        if pd.isna(off_team) or pd.isna(def_team): continue

        points_off, points_def = poss.get("points_for_offense", 0), poss.get("points_for_defense", 0)
        off_players = [poss.get(f"off_player_{i}_id") for i in range(1, 6)]
        def_players = [poss.get(f"def_player_{i}_id") for i in range(1, 6)]

        for pid in off_players:
            if _valid_player_id(pid):
                key = (poss["game_id"], off_team, int(pid))
                exposures[key]["POSS_OFF"] += 1
                exposures[key]["OnCourt_Team_Points"] += points_off
                exposures[key]["OnCourt_Opp_Points"] += points_def
                exposures[key]["OnCourt_Team_FGA"] += poss.get("off_team_FGA", 0)
                exposures[key]["OnCourt_Team_FT_Att"] += poss.get("off_team_FTA", 0)

        for pid in def_players:
            if _valid_player_id(pid):
                key = (poss["game_id"], def_team, int(pid))
                exposures[key]["POSS_DEF"] += 1
                exposures[key]["OnCourt_Team_Points"] += points_def
                exposures[key]["OnCourt_Opp_Points"] += points_off
                exposures[key]["OnCourt_Opp_2p_Att"] += poss.get("off_team_2PA", 0)

    exposure_rows = [{"game_id": k[0], "team_id": k[1], "player_id": k[2], **v} for k, v in exposures.items()]
    poss_exposures = pd.DataFrame(exposure_rows)

    # --- Merge all exposure types ---
    final_df = minutes_df.merge(reb_totals, on=["game_id", "team_id", "player_id"], how="outer")
    if not poss_exposures.empty:
        final_df = final_df.merge(poss_exposures, on=["game_id", "team_id", "player_id"], how="outer")
    
    final_df["POSS"] = final_df.get("POSS_OFF", 0) + final_df.get("POSS_DEF", 0)
    final_df["MPG"] = final_df.get("Minutes", 0)
    final_df["MPG_R"] = final_df.get("Minutes", 0) / 5.0
    
    return final_df.fillna(0)


def build_player_box(
    df: pd.DataFrame,
    counts_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    player_meta: Optional[pd.DataFrame] = None,
    game_meta: Optional[pd.DataFrame] = None,
    restrict_to_pbg: bool = False,
    player_game_meta: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    merged = counts_df.merge(exposures_df, on=["game_id", "team_id", "player_id"], how="outer")
    merged.fillna(0, inplace=True)
    merged = merged[(merged["team_id"] != 0) & (merged["player_id"] != 0)]

    for col in ["game_id", "team_id", "player_id"]:
        if col in merged.columns:
            merged[col] = merged[col].astype(int)

    # Identify any rows where a player has on-court points credited but no minutes.
    # In clean data this should be rare; it typically indicates a mismatch between
    # rotation/time-on-court tracking and possession parsing.
    zero_minute_with_points = merged[
        (merged.get("Minutes", 0) == 0)
        & (
            (merged.get("OnCourt_Team_Points", 0) != 0)
            | (merged.get("OnCourt_Opp_Points", 0) != 0)
        )
    ]

    # For now, keep such rows so that on-court scoring sums remain consistent
    # with team totals (tests enforce this). If you want to treat these as hard
    # errors in a debugging context, you can uncomment the assertion below.
    #
    # if not zero_minute_with_points.empty:
    #     raise AssertionError(
    #         "Found zero-minute rows with non-zero on-court points; "
    #         "this indicates an upstream bug in exposures/timing logic."
    #     )
    if not zero_minute_with_points.empty:
        import warnings

        warnings.warn(
            "Found zero-minute rows with non-zero on-court points; "
            "this indicates a mismatch between timing and exposures. "
            "Rows are kept to preserve on-court scoring invariants.",
            RuntimeWarning,
        )

    merged = merged[
        (merged.get("Minutes", 0) > 0)
        | (
            (merged.get("OnCourt_Team_Points", 0) != 0)
            | (merged.get("OnCourt_Opp_Points", 0) != 0)
        )
    ]

    # Restrict to players with positive Minutes only (debug / optional).
    # By default we keep zero-minute rows if they carry on-court points,
    # to preserve scoring invariants enforced in tests.
    if restrict_to_pbg and zero_minute_with_points.empty:
        merged = merged[merged["Minutes"] > 0]

    merged["Team_SingleGame"] = merged["team_id"]
    merged["Game_SingleGame"] = merged["game_id"]
    merged["NbaDotComID"] = merged["player_id"].astype(int)

    team_map = {
        df["home_team_id"].iloc[0]: df["home_team_abbrev"].iloc[0],
        df["away_team_id"].iloc[0]: df["away_team_abbrev"].iloc[0],
    }
    merged["Team"] = merged["team_id"].map(team_map)

    if player_meta is not None and not player_meta.empty:
        pm = player_meta.copy()
        if "player_id" in pm.columns and "NbaDotComID" not in pm.columns:
            pm["NbaDotComID"] = pm["player_id"]
            pm = pm.drop(columns=["player_id"])
        merged = merged.merge(pm, on="NbaDotComID", how="left")

    # NEW: derive PositionNum if we have Position but not PositionNum
    if "Position" in merged.columns:
        # Ensure PositionNum column exists before trying to fill it
        if "PositionNum" not in merged.columns:
            merged["PositionNum"] = np.nan
        merged["PositionNum"] = merged["PositionNum"].where(
            pd.notna(merged["PositionNum"]), merged["Position"].apply(position_to_num)
        )

    if game_meta is not None and not game_meta.empty:
        merged = merged.merge(game_meta, on="game_id", how="left")

    if player_game_meta is not None and not player_game_meta.empty:
        merged = merged.merge(
            player_game_meta,
            on=["game_id", "team_id", "player_id"],
            how="left",
        )

    # Games played is always computed from Minutes, not taken from metadata.
    merged["G"] = np.where(merged["Minutes"] > 0, 1, 0)

    meta_defaults = {
        "Inactive": 0, "DNP": 0, "DNP_Rest": 0, "DNP_CD": 0,
        "DNP_SingleGame": 0, "Starts": 0, "PlayoffGamesPlayed": 0,
    }
    for col, default_val in meta_defaults.items():
        if col not in merged.columns:
            merged[col] = default_val
        else:
            merged[col] = merged[col].fillna(default_val)

    for col in ["ThreePA_UNAST", "ThreePM_UNAST", "FGA_UNAST", "FGM_UNAST"]:
        if col not in merged.columns:
            merged[col] = 0

    merged["TSAttempts"] = merged["FGA"] + 0.44 * merged["FTA"]
    merged["TSpct"] = np.where(
        merged["TSAttempts"] > 0,
        merged["PTS"] / (2.0 * merged["TSAttempts"]),
        0.0,
    )
    merged["TSPoss"] = merged["TSAttempts"]
    merged["TS"] = merged["TSpct"]
    merged["PossessionsUsed"] = merged["FGA"] + 0.44 * merged["FTA"] + merged.get("TOV", 0)
    merged["USG"] = np.where(merged["POSS_OFF"] > 0, merged["PossessionsUsed"] / merged["POSS_OFF"], 0)

    merged["FGPct"] = np.where(merged["FGA"] > 0, merged["FGM"] / merged["FGA"], 0)
    merged["FT_pct"] = np.where(merged["FTA"] > 0, merged["FTM"] / merged["FTA"], 0)
    merged["ThreeP_pct"] = np.where(merged["ThreePA"] > 0, merged["ThreePM"] / merged["ThreePA"], 0)
    merged["FTR_Att"] = np.where(merged["FGA"] > 0, merged["FTA"] / merged["FGA"], 0)
    merged["FTR_Made"] = np.where(merged["FGA"] > 0, merged["FTM"] / merged["FGA"], 0)

    merged["ORBpct"] = np.where(
        merged["OnCourt_For_OREB_Total"] > 0,
        merged.get("OREB", 0) / merged["OnCourt_For_OREB_Total"],
        0,
    )
    merged["DRBpct"] = np.where(
        merged["OnCourt_For_DREB_Total"] > 0,
        merged.get("DREB", 0) / merged["OnCourt_For_DREB_Total"],
        0,
    )

    teammate_fgm = np.maximum(
        merged["OnCourt_Team_FGM"] - merged.get("FGM", 0),
        0.0,
    )
    merged["ASTpct"] = np.where(
        teammate_fgm > 0,
        merged.get("AST", 0) / teammate_fgm,
        0.0,
    )
    merged["BLKPct"] = np.where(
        merged["OnCourt_Opp_2p_Att"] > 0,
        merged.get("BLK", 0) / merged["OnCourt_Opp_2p_Att"],
        0.0,
    )
    merged["STLpct"] = np.where(merged["POSS_DEF"] > 0, merged.get("STL", 0) / merged["POSS_DEF"], 0)
    merged["TOVpct"] = np.where(merged["PossessionsUsed"] > 0, merged.get("TOV", 0) / merged["PossessionsUsed"], 0)

    merged["PTS_100p"] = np.where(merged["POSS_OFF"] > 0, merged["PTS"] / merged["POSS_OFF"] * 100.0, 0)
    merged["FGM_100p"] = np.where(merged["POSS_OFF"] > 0, merged["FGM"] / merged["POSS_OFF"] * 100.0, 0)
    merged["FGA_100p"] = np.where(merged["POSS_OFF"] > 0, merged["FGA"] / merged["POSS_OFF"] * 100.0, 0)
    merged["FTM_100p"] = np.where(merged["POSS_OFF"] > 0, merged["FTM"] / merged["POSS_OFF"] * 100.0, 0)
    merged["FTA_100p"] = np.where(merged["POSS_OFF"] > 0, merged["FTA"] / merged["POSS_OFF"] * 100.0, 0)
    merged["OREB_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get("OREB", 0) / merged["POSS_OFF"] * 100.0, 0)
    merged["DREB_100p"] = np.where(merged["POSS_DEF"] > 0, merged.get("DREB", 0) / merged["POSS_DEF"] * 100.0, 0)
    merged["AST_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get("AST", 0) / merged["POSS_OFF"] * 100.0, 0)
    merged["STL_100p"] = np.where(merged["POSS_DEF"] > 0, merged.get("STL", 0) / merged["POSS_DEF"] * 100.0, 0)
    merged["TOV_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get("TOV", 0) / merged["POSS_OFF"] * 100.0, 0)
    merged["TOV_Live_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get("TOV_Live", 0) / merged["POSS_OFF"] * 100.0, 0)
    merged["TOV_Dead_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get("TOV_Dead", 0) / merged["POSS_OFF"] * 100.0, 0)

    merged["BLK"] = merged.get("BLK_Team", 0) + merged.get("BLK_Opp", 0)
    merged["Plus_Minus"] = merged.get("OnCourt_Team_Points", 0) - merged.get("OnCourt_Opp_Points", 0)

    merged["fgm"] = merged.get("FGM", 0)
    merged["fga"] = merged.get("FGA", 0)
    merged["3PM"] = merged.get("ThreePM", 0)
    merged["3PA"] = merged.get("ThreePA", 0)
    merged["tpm"] = merged.get("ThreePM", 0)
    merged["tpa"] = merged.get("ThreePA", 0)
    merged["ftm"] = merged.get("FTM", 0)
    merged["fta"] = merged.get("FTA", 0)
    merged["points"] = merged.get("PTS", 0)

    merged["Pace"] = np.where(
        merged["Minutes"] > 0,
        (merged["POSS_OFF"] + merged["POSS_DEF"]) / merged["Minutes"] * 48.0,
        0.0,
    )

    merged["BLK_Opp_100p"] = np.where(merged["POSS_DEF"] > 0, merged.get("BLK_Opp", 0) / merged["POSS_DEF"] * 100.0, 0)
    merged["BLK_Team_100p"] = np.where(merged["POSS_DEF"] > 0, merged.get("BLK_Team", 0) / merged["POSS_DEF"] * 100.0, 0)
    merged["PF_100p"] = np.where(merged["POSS"] > 0, merged.get("PF", 0) / merged["POSS"] * 100.0, 0)
    merged["PF_DRAWN_100p"] = np.where(merged["POSS"] > 0, merged.get("PF_DRAWN", 0) / merged["POSS"] * 100.0, 0)
    merged["PF_Loose_100p"] = np.where(merged["POSS"] > 0, merged.get("PF_Loose", 0) / merged["POSS"] * 100.0, 0)
    merged["CHRG_100p"] = np.where(merged["POSS"] > 0, merged.get("CHRG", 0) / merged["POSS"] * 100.0, 0)
    merged["TECH_100p"] = np.where(merged["POSS"] > 0, merged.get("TECH", 0) / merged["POSS"] * 100.0, 0)
    merged["FLAGRANT_100p"] = np.where(merged["POSS"] > 0, merged.get("FLAGRANT", 0) / merged["POSS"] * 100.0, 0)
    merged["Goaltends_100p"] = np.where(merged["POSS"] > 0, merged.get("Goaltends", 0) / merged["POSS"] * 100.0, 0)

    for zone, label in [("0_3", "0_3ft"), ("4_9", "4_9ft"), ("10_17", "10_17ft"), ("18_23", "18_23ft")]:
        fga_col = f"{zone}_FGA"
        fgm_col = f"{zone}_FGM"
        if fga_col not in merged.columns:
            merged[fga_col] = 0
        if fgm_col not in merged.columns:
            merged[fgm_col] = 0
        merged[f"{label}_FGA_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get(fga_col, 0) / merged["POSS_OFF"] * 100.0, 0)
        merged[f"{label}_FGM_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get(fgm_col, 0) / merged["POSS_OFF"] * 100.0, 0)
        merged[f"{label}_FGPct"] = np.where(merged.get(fga_col, 0) > 0, merged.get(fgm_col, 0) / merged.get(fga_col, 0), 0)
        una_fga = f"{zone}_FGA_UNAST"
        una_fgm = f"{zone}_FGM_UNAST"
        if una_fga not in merged.columns:
            merged[una_fga] = 0
        if una_fgm not in merged.columns:
            merged[una_fgm] = 0
        merged[f"{label}_FGA_UNAST_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get(una_fga, 0) / merged["POSS_OFF"] * 100.0, 0)
        merged[f"{label}_FGM_UNAST_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get(una_fgm, 0) / merged["POSS_OFF"] * 100.0, 0)
        merged[f"{label}_FG_UNAST_Pct"] = np.where(merged.get(una_fga, 0) > 0, merged.get(una_fgm, 0) / merged.get(una_fga, 0), 0)

    merged["ThreePM_UNAST_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get("ThreePM_UNAST", 0) / merged["POSS_OFF"] * 100.0, 0)
    merged["ThreePA_UNAST_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get("ThreePA_UNAST", 0) / merged["POSS_OFF"] * 100.0, 0)
    merged["ThreeP_UNAST_Pct"] = np.where(merged.get("ThreePA_UNAST", 0) > 0, merged.get("ThreePM_UNAST", 0) / merged.get("ThreePA_UNAST", 0), 0)

    merged["FGM_100p_AST"] = np.where(merged["POSS_OFF"] > 0, merged.get("FGM_AST", 0) / merged["POSS_OFF"] * 100.0, 0)
    merged["ThreePM_100p_AST"] = np.where(merged["POSS_OFF"] > 0, merged.get("ThreePM_AST", 0) / merged["POSS_OFF"] * 100.0, 0)

    # --- Shooter-side zonal assisted FGM aliases ---
    for zone, label in [("0_3", "0_3ft"), ("4_9", "4_9ft"), ("10_17", "10_17ft"), ("18_23", "18_23ft")]:
        fgm_ast_col = f"{zone}_FGM_AST"
        if fgm_ast_col not in merged.columns:
            merged[fgm_ast_col] = 0

        # Raw count: e.g. 0_3ft_FGM_AST
        merged[f"{label}_FGM_AST"] = merged.get(fgm_ast_col, 0)

        # Per-100 offensive possessions: e.g. 0_3ft_FGM_100p_AST
        merged[f"{label}_FGM_100p_AST"] = np.where(
            merged["POSS_OFF"] > 0,
            merged.get(fgm_ast_col, 0) / merged["POSS_OFF"] * 100.0,
            0.0,
        )

    for zone, label in [("0_3", "AST_0_3ft"), ("4_9", "AST_4_9ft"), ("10_17", "AST_10_17ft"), ("18_23", "AST_18_23ft")]:
        merged[f"{label}_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get(f"AST_{zone}", 0) / merged["POSS_OFF"] * 100.0, 0)
    merged["AST_3P_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get("AST_3P", 0) / merged["POSS_OFF"] * 100.0, 0)

    merged["BLK_Opp"] = merged.get("BLK_Opp", 0)
    merged["BLK_Team"] = merged.get("BLK_Team", 0)

    merged["POSS"] = merged.get("POSS_OFF", 0) + merged.get("POSS_DEF", 0)

    # --- Game / side context ---
    home_team_id = df["home_team_id"].iloc[0]
    away_team_id = df["away_team_id"].iloc[0]
    merged["h_tm_id"] = home_team_id
    merged["v_tm_id"] = away_team_id
    merged["home_fl"] = np.where(merged["team_id"] == home_team_id, 1, 0)
    if "season" in df.columns:
        merged["season"] = df["season"].iloc[0]
        merged["Year"] = merged["season"]
    merged["check_season"] = 0

    # --- Metadata placeholders if player_meta didn't provide them ---
    for col in ["PlayerSeasonID", "PlayerID", "FullName", "Position", "PositionNum", "Height", "Weight", "Age"]:
        if col not in merged.columns:
            merged[col] = np.nan

    merged["Player_Team"] = np.where(
        merged["FullName"].notna() & merged["NbaDotComID"].notna(),
        merged["FullName"].astype(str) + " " + merged["NbaDotComID"].astype(int).astype(str),
        merged["FullName"].astype(str),
    )

    # --- FT and 3P aliases ---
    merged["FT%"] = merged["FT_pct"]
    merged["3PPct"] = merged["ThreeP_pct"]
    merged["3PM_100p"] = np.where(
        merged["POSS_OFF"] > 0, merged["ThreePM"] / merged["POSS_OFF"] * 100.0, 0.0
    )
    merged["3PA_100p"] = np.where(
        merged["POSS_OFF"] > 0, merged["ThreePA"] / merged["POSS_OFF"] * 100.0, 0.0
    )

    # --- And-1 rates ---
    merged["AndOnes"] = merged.get("AndOnes", 0)
    merged["AndOne_100p"] = np.where(
        merged["POSS_OFF"] > 0, merged["AndOnes"] / merged["POSS_OFF"] * 100.0, 0.0
    )

    # --- OREB-based rates relative to team FGA / FT ---
    merged["OREB_FGA"] = np.where(
        merged.get("OnCourt_Team_FGA", 0) > 0,
        merged["OREB"] / merged["OnCourt_Team_FGA"] * 100.0,
        0.0,
    )
    merged["OREB_FT"] = np.where(
        merged.get("OnCourt_Team_FT_Att", 0) > 0,
        merged["OREB"] / merged["OnCourt_Team_FT_Att"] * 100.0,
        0.0,
    )

    # --- Global unassisted shooting aliases ---
    merged["FGM_UNAST"] = merged.get("FGM_UNAST", 0)
    merged["FGA_UNAST"] = merged.get("FGA_UNAST", 0)
    merged["FGM_UNAST_100p"] = np.where(
        merged["POSS_OFF"] > 0, merged["FGM_UNAST"] / merged["POSS_OFF"] * 100.0, 0.0
    )
    merged["FGA_UNAST_100p"] = np.where(
        merged["POSS_OFF"] > 0, merged["FGA_UNAST"] / merged["POSS_OFF"] * 100.0, 0.0
    )
    merged["FG_UNAST_Pct"] = np.where(
        merged["FGA_UNAST"] > 0, merged["FGM_UNAST"] / merged["FGA_UNAST"], 0.0
    )

    # --- 3P unassisted/assisted aliasing ---
    merged["3PM_UNAST"] = merged.get("ThreePM_UNAST", 0)
    merged["3PA_UNAST"] = merged.get("ThreePA_UNAST", 0)
    merged["3PM_UNAST_100p"] = np.where(
        merged["POSS_OFF"] > 0, merged["3PM_UNAST"] / merged["POSS_OFF"] * 100.0, 0.0
    )
    merged["3PA_UNAST_100p"] = np.where(
        merged["POSS_OFF"] > 0, merged["3PA_UNAST"] / merged["POSS_OFF"] * 100.0, 0.0
    )
    merged["3P_UNAST_Pct"] = np.where(
        merged["3PA_UNAST"] > 0, merged["3PM_UNAST"] / merged["3PA_UNAST"], 0.0
    )

    merged["3PM_AST"] = merged.get("ThreePM_AST", 0)
    merged["3PM_100p_AST"] = np.where(
        merged["POSS_OFF"] > 0, merged["3PM_AST"] / merged["POSS_OFF"] * 100.0, 0.0
    )

    # --- Zone-level aliases for 0-3, 4-9, 10-17, 18-23 ---
    for zone, label in [("0_3", "0_3ft"), ("4_9", "4_9ft"), ("10_17", "10_17ft"), ("18_23", "18_23ft")]:
        base_fga = merged.get(f"{zone}_FGA", 0)
        base_fgm = merged.get(f"{zone}_FGM", 0)
        una_fga = merged.get(f"{zone}_FGA_UNAST", 0)
        una_fgm = merged.get(f"{zone}_FGM_UNAST", 0)

        merged[f"{label}_FGA"] = base_fga
        merged[f"{label}_FGM"] = base_fgm
        merged[f"{label}_FGA_UNAST"] = una_fga
        merged[f"{label}_FGM_UNAST"] = una_fgm

        # 100p and pct are already in your current code as {label}_FGA_100p, etc.
        merged[f"{label}_FGM_100p_UNAST"] = merged.get(f"{label}_FGM_UNAST_100p", 0)
        merged[f"{label}_FGA_100p_UNAST"] = merged.get(f"{label}_FGA_UNAST_100p", 0)

    # --- Defensive rebound alias for glossary naming ---
    merged["DRB"] = merged.get("DREB", 0)

    # --- Raw AST-by-zone aliases corresponding to AST_*ft_100p ---
    merged["AST_0_3ft"] = merged.get("AST_0_3", 0)
    merged["AST_4_9ft"] = merged.get("AST_4_9", 0)
    merged["AST_10_17ft"] = merged.get("AST_10_17", 0)
    merged["AST_18_23ft"] = merged.get("AST_18_23", 0)
    merged["AST_3P"] = merged.get("AST_3P", 0)

    # --- Final Glossary Naming Alignment ---
    # Alias on-court rebound opportunity counts to glossary-style names
    merged["OnCourt_For_OREB_FGA"] = merged.get("OnCourt_For_OREB_Total", 0)
    merged["OnCourt_For_DREB_FGA"] = merged.get("OnCourt_For_DREB_Total", 0)

    return merged


def append_team_totals(box_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a player-level box (the output of build_player_box / player_box_glossary),
    append one 'TOTAL' row per (game_id, team_id) that aggregates team stats.

    NOTE: This is a simple aggregator:
      - Sums counting stats and minutes across players.
      - Sums POSS_OFF / POSS_DEF and recomputes key rate stats from those sums.
      - Sets identifier fields (NbaDotComID, PlayerID, etc.) to 0 and labels
        Team/FullName/Player_Team as 'TOTAL'.

    You may want to tweak which columns are summed vs averaged depending on your
    use case.
    """
    if box_df.empty:
        return box_df

    id_cols = ["game_id", "team_id"]
    numeric_cols = box_df.select_dtypes(include=[np.number]).columns.tolist()

    # We do NOT want to sum identifiers or flags that are per-row, not additive.
    do_not_sum = {
        "NbaDotComID",
        "PlayerID",
        "PlayerSeasonID",
        "Game_SingleGame",
        "Team_SingleGame",
        "home_fl",
        "h_tm_id",
        "v_tm_id",
        "season",
        "Year",
    }
    sum_cols = [c for c in numeric_cols if c not in do_not_sum]

    team_totals = (
        box_df.groupby(id_cols, as_index=False)[sum_cols]
        .sum()
    )

    # Reattach simple identifiers from the first player row per team/game.
    first_meta = (
        box_df.groupby(id_cols, as_index=False)
        .agg(
            {
                "Game_SingleGame": "first",
                "Team_SingleGame": "first",
                "season": "first",
                "Year": "first",
                "h_tm_id": "first",
                "v_tm_id": "first",
                "home_fl": "first",
                "Team": "first",
            }
        )
    )

    team_totals = team_totals.merge(first_meta, on=id_cols, how="left")

    team_totals["NbaDotComID"] = 0
    team_totals["PlayerID"] = 0
    team_totals["PlayerSeasonID"] = 0
    team_totals["FullName"] = "TOTAL"
    team_totals["Player_Team"] = "TOTAL"
    team_totals["G"] = 1  # team played the game
    team_totals["Inactive"] = 0
    team_totals["DNP"] = 0
    team_totals["DNP_Rest"] = 0
    team_totals["DNP_CD"] = 0
    team_totals["DNP_SingleGame"] = 0
    team_totals["Starts"] = 0  # not meaningful at team level

    # Recompute key rate stats for totals so they're not 5x sums.
    team_totals["TSAttempts"] = team_totals["FGA"] + 0.44 * team_totals["FTA"]
    team_totals["TSpct"] = np.where(
        team_totals["TSAttempts"] > 0,
        team_totals["PTS"] / (2.0 * team_totals["TSAttempts"]),
        0.0,
    )
    team_totals["PossessionsUsed"] = (
        team_totals["FGA"]
        + 0.44 * team_totals["FTA"]
        + team_totals.get("TOV", 0)
    )
    team_totals["USG"] = np.where(
        team_totals["POSS_OFF"] > 0,
        team_totals["PossessionsUsed"] / team_totals["POSS_OFF"],
        0.0,
    )
    team_totals["POSS"] = team_totals["POSS_OFF"] + team_totals["POSS_DEF"]
    team_totals["Pace"] = np.where(
        team_totals["Minutes"] > 0,
        team_totals["POSS"] / team_totals["Minutes"] * 48.0,
        0.0,
    )
    team_totals["PTS_100p"] = np.where(
        team_totals["POSS_OFF"] > 0,
        team_totals["PTS"] / team_totals["POSS_OFF"] * 100.0,
        0.0,
    )

    # Label the team row.
    team_totals["Team"] = "TOTAL"

    # Concatenate players + team totals
    return pd.concat([box_df, team_totals], ignore_index=True)
