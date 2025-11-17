from __future__ import annotations

from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

ZONE_BINS: List[Tuple[float, float, str]] = [
    (0.0, 3.0, "0_3"),
    (3.0, 9.0, "4_9"),
    (9.0, 17.0, "10_17"),
    (17.0, 23.0, "18_23"),
]


def classify_shot_zone(shot_distance: float | None, area: str | None) -> Optional[str]:
    if shot_distance is not None and not pd.isna(shot_distance):
        d = float(shot_distance)
        if 0.0 <= d <= 3.0:
            return "0_3"
        elif 3.0 < d <= 9.0:
            return "4_9"
        elif 9.0 < d <= 17.0:
            return "10_17"
        elif 17.0 < d <= 23.0:
            return "18_23"
        else:
            return None

    if area:
        area_lower = area.lower()
        if "restricted" in area_lower:
            return "0_3"
        if "paint" in area_lower:
            return "4_9"
        if "mid-range" in area_lower:
            return None
    return None


def annotate_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "family" not in df.columns:
        df["family"] = df.get("event_type_de")
    if "team_id" not in df.columns:
        df["team_id"] = np.where(
            df.get("event_team") == df.get("home_team_abbrev"),
            df.get("home_team_id"),
            np.where(
                df.get("event_team") == df.get("away_team_abbrev"),
                df.get("away_team_id"),
                np.nan,
            ),
        )
    if "points_made" not in df.columns and "points_made_y" in df.columns:
        df["points_made"] = df["points_made_y"]
    fam = df["family"].astype(str).str.lower().str.replace("-", "_", regex=False)
    df["family"] = fam

    df["is_fg_attempt"] = fam.isin(["shot", "miss_shot", "missed_shot"])
    df["is_fg_make"] = (fam == "shot") & (df["points_made"] > 0)
    df["is_ft"] = fam == "free_throw"
    df["is_ft_make"] = df["is_ft"] & (df["points_made"] > 0)
    df["is_three"] = df.get("is_three", 0).astype(bool)

    is_tov_family = fam == "turnover"
    subfam = df.get("subfamily_de", "").fillna("") if "subfamily_de" in df.columns else pd.Series([""] * len(df))
    df["is_turnover_live"] = is_tov_family & subfam.str.contains("live", case=False)
    df["is_turnover_dead"] = is_tov_family & ~df["is_turnover_live"]
    df.loc[is_tov_family & ~subfam.astype(bool), "is_turnover_live"] = (
        is_tov_family & (df["is_steal"] == 1)
    )
    df.loc[is_tov_family & ~subfam.astype(bool), "is_turnover_dead"] = (
        is_tov_family & ~df["is_turnover_live"]
    )

    is_foul_family = fam == "foul"
    sub = subfam.str.lower()
    df["is_loose_ball_foul"] = is_foul_family & sub.str.contains("loose")
    df["is_flagrant"] = is_foul_family & sub.str.contains("flagrant")
    df["is_technical"] = is_foul_family & sub.str.contains("technical")
    df["is_charge"] = is_foul_family & sub.str.contains("charging")

    def _is_and_one_row(row: pd.Series) -> bool:
        quals = row.get("qualifiers") or []
        if isinstance(quals, str):
            return False
        quals_lower = [q.lower() for q in quals]
        if "andone" in quals_lower:
            return True
        return False

    df["is_and_one"] = df.apply(_is_and_one_row, axis=1)

    shot_mask = fam.isin(["shot", "miss_shot", "missed_shot"])
    df["shot_zone"] = np.where(
        shot_mask,
        df.apply(lambda r: classify_shot_zone(r.get("shot_distance"), r.get("area")), axis=1),
        None,
    )

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


def accumulate_player_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    counts: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for _, row in df.iterrows():
        player_id = row.get("player1_id")
        team_id = row.get("player1_team_id")
        game_id = row.get("game_id")
        if pd.isna(player_id) or player_id == 0:
            player_id = None
        key = (game_id, team_id, player_id)

        if row.get("is_fg_attempt"):
            _increment_count(counts[key], "FGA")
            if row.get("is_fg_make"):
                _increment_count(counts[key], "FGM")
            if row.get("is_three"):
                _increment_count(counts[key], "ThreePA")
                if row.get("is_fg_make"):
                    _increment_count(counts[key], "ThreePM")
            zone = row.get("shot_zone")
            if zone:
                _increment_count(counts[key], f"{zone}_FGA")
                if row.get("is_fg_make"):
                    _increment_count(counts[key], f"{zone}_FGM")

            assist_id = row.get("assist_id")
            assisted = not (pd.isna(assist_id) or assist_id == 0)
            if assisted:
                _increment_count(counts[key], "FGM_AST", 1 if row.get("is_fg_make") else 0)
                if row.get("is_three") and row.get("is_fg_make"):
                    _increment_count(counts[key], "ThreePM_AST")
                if zone and row.get("is_fg_make"):
                    _increment_count(counts[key], f"{zone}_FGM_AST")
                if assisted:
                    ast_key = (game_id, row.get("team_id"), assist_id)
                    _increment_count(counts[ast_key], "AST")
                    if zone:
                        _increment_count(counts[ast_key], f"AST_{zone}")
                    if row.get("is_three"):
                        _increment_count(counts[ast_key], "AST_3P")
            else:
                _increment_count(counts[key], "FGA_UNAST")
                if row.get("is_fg_make"):
                    _increment_count(counts[key], "FGM_UNAST")
                if row.get("is_three"):
                    _increment_count(counts[key], "ThreePA_UNAST")
                    if row.get("is_fg_make"):
                        _increment_count(counts[key], "ThreePM_UNAST")
                if zone:
                    _increment_count(counts[key], f"{zone}_FGA_UNAST")
                    if row.get("is_fg_make"):
                        _increment_count(counts[key], f"{zone}_FGM_UNAST")

        if row.get("family") == "free_throw":
            _increment_count(counts[key], "FTA")
            if row.get("points_made") > 0:
                _increment_count(counts[key], "FTM")

        if not pd.isna(player_id):
            _increment_count(counts[key], "PTS", row.get("points_made", 0))

        if row.get("is_o_rebound") == 1:
            _increment_count(counts[key], "OREB")
        if row.get("is_d_rebound") == 1:
            _increment_count(counts[key], "DREB")

        if row.get("family") == "turnover" and not pd.isna(player_id):
            _increment_count(counts[key], "TOV")
            if row.get("is_turnover_live"):
                _increment_count(counts[key], "TOV_Live")
            else:
                _increment_count(counts[key], "TOV_Dead")

        if row.get("family") == "foul" and not pd.isna(player_id):
            _increment_count(counts[key], "PF")
            if row.get("is_loose_ball_foul"):
                _increment_count(counts[key], "PF_Loose")
            if row.get("is_flagrant"):
                _increment_count(counts[key], "FLAGRANT")
            if row.get("is_technical"):
                _increment_count(counts[key], "TECH")
            if row.get("is_charge"):
                _increment_count(counts[key], "CHRG")
            fouled = row.get("player2_id")
            fouled_team = row.get("player2_team_id")
            if fouled and not pd.isna(fouled) and fouled_team and not pd.isna(fouled_team):
                foul_key = (game_id, fouled_team, fouled)
                _increment_count(counts[foul_key], "PF_DRAWN")

        if row.get("is_block") == 1:
            blocker = row.get("player3_id")
            block_team = row.get("player3_team_id")
            block_key = (game_id, block_team, blocker)
            if blocker and blocker != 0:
                _increment_count(counts[block_key], "BLK")
            possession_after = row.get("possession_after")
            shooter_team = row.get("team_id")
            if possession_after and possession_after == block_team:
                _increment_count(counts[block_key], "BLK_Team")
            elif possession_after and possession_after == shooter_team:
                _increment_count(counts[block_key], "BLK_Opp")
            else:
                _increment_count(counts[block_key], "BLK_Team")

        if row.get("is_steal") == 1:
            stealer = row.get("player2_id")
            steal_team = row.get("player2_team_id")
            steal_key = (game_id, steal_team, stealer)
            if stealer and stealer != 0:
                _increment_count(counts[steal_key], "STL")

        goaltend_flag = False
        if isinstance(row.get("subfamily_de"), str) and "goaltend" in row.get("subfamily_de").lower():
            goaltend_flag = True
        if goaltend_flag:
            goaltend_player = row.get("player3_id") or row.get("player1_id")
            goaltend_team = row.get("player3_team_id") or row.get("player1_team_id")
            gt_key = (game_id, goaltend_team, goaltend_player)
            _increment_count(counts[gt_key], "Goaltends")

        if row.get("is_fg_make") and row.get("is_and_one") and not pd.isna(player_id):
            _increment_count(counts[key], "AndOnes")

    records: List[Dict[str, Any]] = []
    for (game_id, team_id, player_id), vals in counts.items():
        if player_id is None or player_id == 0:
            continue
        record = {
            "game_id": game_id,
            "team_id": team_id,
            "player_id": player_id,
        }
        record.update(vals)
        records.append(record)

    return pd.DataFrame(records)


def compute_on_court_exposures(pbp: "PbP", df: pd.DataFrame) -> pd.DataFrame:
    exposures: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for _, row in df.iterrows():
        event_length = row.get("event_length", 0)
        if pd.isna(event_length):
            event_length = 0
        home_ids = [row.get(f"home_player_{i}_id") for i in range(1, 6)]
        away_ids = [row.get(f"away_player_{i}_id") for i in range(1, 6)]
        for pid in home_ids:
            if pid and pid != 0:
                key = (row.get("game_id"), row.get("home_team_id"), pid)
                _increment_count(exposures[key], "Minutes", event_length / 60.0)
        for pid in away_ids:
            if pid and pid != 0:
                key = (row.get("game_id"), row.get("away_team_id"), pid)
                _increment_count(exposures[key], "Minutes", event_length / 60.0)

        if row.get("is_block") == 1:
            block_team = row.get("player3_team_id")
            block_ids = home_ids if block_team == row.get("home_team_id") else away_ids
            for pid in block_ids:
                if pid and pid != 0:
                    key = (row.get("game_id"), block_team, pid)
                    _increment_count(exposures[key], "TM_BLK_OnCourt")

        if row.get("is_fg_attempt") and row.get("shot_made") == 0:
            shoot_team = row.get("team_id")
            home_on = home_ids
            away_on = away_ids
            for pid in (home_on if shoot_team == row.get("home_team_id") else away_on):
                if pid and pid != 0:
                    key = (row.get("game_id"), shoot_team, pid)
                    _increment_count(exposures[key], "OnCourt_For_OREB_FGA")
            opp_team = row.get("away_team_id") if shoot_team == row.get("home_team_id") else row.get("home_team_id")
            for pid in (away_on if shoot_team == row.get("home_team_id") else home_on):
                if pid and pid != 0:
                    key = (row.get("game_id"), opp_team, pid)
                    _increment_count(exposures[key], "OnCourt_For_DREB_FGA")

    poss_df = pbp._build_possessions(df, include_event_agg=True)
    for _, poss in poss_df.iterrows():
        off_team = poss.get("off_team_id")
        def_team = poss.get("def_team_id")
        points = poss.get("points_for_offense", 0)
        off_players = [poss.get(f"off_player_{i}_id") for i in range(1, 6)]
        def_players = [poss.get(f"def_player_{i}_id") for i in range(1, 6)]

        for pid in off_players:
            if pid and pid != 0:
                key = (poss.get("game_id"), off_team, pid)
                _increment_count(exposures[key], "POSS_OFF")
                _increment_count(exposures[key], "OnCourt_Team_Points", points)
                _increment_count(exposures[key], "OnCourt_Team_3p_Att", poss.get("off_team_3PA", 0))
                _increment_count(exposures[key], "OnCourt_Team_3p_Made", poss.get("off_team_3PM", 0))
                _increment_count(exposures[key], "OnCourt_Team_FT_Att", poss.get("off_team_FTA", 0))
                _increment_count(exposures[key], "OnCourt_Team_FT_Made", poss.get("off_team_FTM", 0))
                _increment_count(exposures[key], "OnCourt_Team_FGM", poss.get("off_team_FGM", 0))

        for pid in def_players:
            if pid and pid != 0:
                key = (poss.get("game_id"), def_team, pid)
                _increment_count(exposures[key], "POSS_DEF")
                _increment_count(exposures[key], "OnCourt_Opp_Points", points)
                _increment_count(exposures[key], "OnCourt_Opp_3p_Att", poss.get("off_team_3PA", 0))
                _increment_count(exposures[key], "OnCourt_Opp_3p_Made", poss.get("off_team_3PM", 0))
                _increment_count(exposures[key], "OnCourt_Opp_FT_Att", poss.get("off_team_FTA", 0))
                _increment_count(exposures[key], "OnCourt_Opp_FT_Made", poss.get("off_team_FTM", 0))
                _increment_count(exposures[key], "OnCourt_Opp_FGA", poss.get("off_team_FGA", 0))
                _increment_count(exposures[key], "OnCourt_Opp_FGM", poss.get("off_team_FGM", 0))

    exposure_rows: List[Dict[str, Any]] = []
    for (game_id, team_id, player_id), vals in exposures.items():
        vals.setdefault("Minutes", 0)
        vals.setdefault("POSS_OFF", 0)
        vals.setdefault("POSS_DEF", 0)
        vals["POSS"] = vals.get("POSS_OFF", 0) + vals.get("POSS_DEF", 0)
        vals["MPG"] = vals.get("Minutes", 0)
        vals["MPG_R"] = vals.get("MPG", 0) / 5.0
        vals.setdefault("OnCourt_Team_FGM", 0)
        vals.setdefault("OnCourt_Team_Points", 0)
        vals.setdefault("OnCourt_Team_3p_Made", 0)
        vals.setdefault("OnCourt_Team_3p_Att", 0)
        vals.setdefault("OnCourt_Team_FT_Made", 0)
        vals.setdefault("OnCourt_Team_FT_Att", 0)
        vals.setdefault("OnCourt_Opp_Points", 0)
        vals.setdefault("OnCourt_Opp_3p_Made", 0)
        vals.setdefault("OnCourt_Opp_3p_Att", 0)
        vals.setdefault("OnCourt_Opp_FT_Made", 0)
        vals.setdefault("OnCourt_Opp_FT_Att", 0)
        vals.setdefault("OnCourt_For_OREB_FGA", 0)
        vals.setdefault("OnCourt_For_DREB_FGA", 0)
        vals.setdefault("TM_BLK_OnCourt", 0)
        vals.setdefault("OnCourt_Opp_FGM", 0)
        vals.setdefault("OnCourt_Opp_FGA", 0)
        exposure_rows.append({"game_id": game_id, "team_id": team_id, "player_id": player_id, **vals})

    exposure_df = pd.DataFrame(exposure_rows)
    try:
        toc_df = pbp._toc_calc_player()[["player_id", "team_id", "game_id", "toc"]]
        toc_df["Minutes_calc"] = toc_df["toc"] / 60.0
        exposure_df = exposure_df.merge(
            toc_df[["player_id", "team_id", "game_id", "Minutes_calc"]],
            on=["player_id", "team_id", "game_id"],
            how="left",
        )
        exposure_df["Minutes"] = np.where(
            exposure_df["Minutes"] == 0, exposure_df["Minutes_calc"], exposure_df["Minutes"]
        )
        exposure_df.drop(columns=["Minutes_calc"], inplace=True)
    except Exception:
        pass

    # Ensure MPG and MPG_R reflect the final Minutes value
    exposure_df["MPG"] = exposure_df["Minutes"]
    exposure_df["MPG_R"] = exposure_df["MPG"] / 5.0

    return exposure_df


def build_player_box(
    df: pd.DataFrame,
    counts_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    player_meta: Optional[pd.DataFrame] = None,
    game_meta: Optional[pd.DataFrame] = None,
    pbg_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    merged = counts_df.merge(exposures_df, on=["game_id", "team_id", "player_id"], how="outer")
    merged.fillna(0, inplace=True)
    merged = merged[(merged["team_id"] != 0) & (merged["player_id"] != 0)]
    merged = merged[merged.get("Minutes", 0) > 0]

    if pbg_stats is not None:
        pbg_subset = pbg_stats[[
            "game_id",
            "team_id",
            "player_id",
            "fgm",
            "fga",
            "tpm",
            "tpa",
            "ftm",
            "fta",
            "points",
        ]].copy()
        pbg_subset.fillna(0, inplace=True)
        merged = merged.merge(pbg_subset, on=["game_id", "team_id", "player_id"], how="left", suffixes=("", "_pbg"))
        merged = merged[merged["player_id"].isin(pbg_stats["player_id"].unique())]
        for src, dest in [
            ("fgm", "FGM"),
            ("fga", "FGA"),
            ("tpm", "ThreePM"),
            ("tpa", "ThreePA"),
            ("ftm", "FTM"),
            ("fta", "FTA"),
            ("points", "PTS"),
        ]:
            col_src = f"{src}_pbg" if f"{src}_pbg" in merged.columns else src
            merged[dest] = merged[col_src].fillna(merged.get(dest, 0))
        merged.drop(columns=[c for c in merged.columns if c.endswith("_pbg") or c in ["fgm","fga","tpm","tpa","ftm","fta","points"]], inplace=True)

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
        merged = merged.merge(pm, on="NbaDotComID", how="left")

    if game_meta is not None and not game_meta.empty:
        merged = merged.merge(game_meta, on="game_id", how="left")

    merged["G"] = np.where(merged["Minutes"] > 0, 1, 0)
    merged["Inactive"] = 0
    merged["DNP"] = 0
    merged["DNP_Rest"] = 0
    merged["DNP_CD"] = 0
    merged["DNP_SingleGame"] = 0
    merged["Starts"] = 0
    merged["PlayoffGamesPlayed"] = 0

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

    merged["ORBpct"] = np.where(merged["OnCourt_For_OREB_FGA"] > 0, merged.get("OREB", 0) / merged["OnCourt_For_OREB_FGA"], 0)
    merged["DRBpct"] = np.where(merged["OnCourt_For_DREB_FGA"] > 0, merged.get("DREB", 0) / merged["OnCourt_For_DREB_FGA"], 0)

    merged["ASTpct"] = np.where(merged["OnCourt_Team_FGM"] > 0, merged.get("AST", 0) / merged["OnCourt_Team_FGM"], 0)
    merged["BLKPct"] = np.where(merged["OnCourt_Opp_FGA"] > 0, merged.get("BLK", 0) / merged["OnCourt_Opp_FGA"], 0)
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
    merged["tpm"] = merged.get("ThreePM", 0)
    merged["tpa"] = merged.get("ThreePA", 0)
    merged["ftm"] = merged.get("FTM", 0)
    merged["fta"] = merged.get("FTA", 0)
    merged["points"] = merged.get("PTS", 0)

    team_minutes = merged.groupby(["game_id", "team_id"])["Minutes"].transform("sum")
    team_minutes_per_5 = team_minutes / 5.0
    team_poss = merged.groupby(["game_id", "team_id"])["POSS_OFF"].transform("max")
    merged["Pace"] = np.where(team_minutes_per_5 > 0, team_poss * 48.0 / team_minutes_per_5, 0)

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
    for zone, label in [("0_3", "AST_0_3ft"), ("4_9", "AST_4_9ft"), ("10_17", "AST_10_17ft"), ("18_23", "AST_18_23ft")]:
        merged[f"{label}_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get(f"AST_{zone}", 0) / merged["POSS_OFF"] * 100.0, 0)
    merged["AST_3P_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get("AST_3P", 0) / merged["POSS_OFF"] * 100.0, 0)

    merged["BLK_Opp"] = merged.get("BLK_Opp", 0)
    merged["BLK_Team"] = merged.get("BLK_Team", 0)

    merged["POSS"] = merged.get("POSS_OFF", 0) + merged.get("POSS_DEF", 0)

    return merged
