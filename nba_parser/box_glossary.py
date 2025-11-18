from __future__ import annotations

from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple

import re

import numpy as np
import pandas as pd

# Mapping for base positions to numeric slots.
# 1=PG, 2=SG, 3=SF, 4=PF, 5=C. "G" and "F" are midpoints.
_BASE_POS_NUM: Dict[str, float] = {
    "PG": 1.0,
    "SG": 2.0,
    "SF": 3.0,
    "PF": 4.0,
    "C": 5.0,
    "G": 1.5,  # generic guard between PG/SG
    "F": 3.5,  # generic forward between SF/PF
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


def _validate_id_dtypes(df: pd.DataFrame, context: str = ""):
    """Ensure all ID columns are integers."""

    id_cols = [
        "game_id",
        "team_id",
        "player_id",
        "home_team_id",
        "away_team_id",
        "NbaDotComID",
        "PlayerID",
        "h_tm_id",
        "v_tm_id",
    ]
    for col in id_cols:
        if col in df.columns and not pd.api.types.is_integer_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            except Exception:
                import warnings

                warnings.warn(
                    f"{context}: Column '{col}' should be integer, got {df[col].dtype}"
                )


def _coerce_id_scalar(val: Any) -> int:
    """
    Coerce a scalar ID (string/int/float/None) to a plain Python int.

    Non-numeric / missing -> 0.
    """

    coerced = pd.to_numeric(val, errors="coerce")
    if pd.isna(coerced):
        return 0
    return int(coerced)


def classify_shot_zone(shot_distance: float | None, area: str | None) -> Optional[str]:
    """
    Backwards‑compatible single‑shot classifier (kept for any call sites that
    still use it directly). The pipeline should prefer _vectorized_shot_zone.
    """
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


def _qualifier_to_str(val: Any) -> str:
    """
    Normalize a single qualifiers cell to a safe, lowercase string.

    Intended for use anywhere we parse the `qualifiers` column, which can be:
      - None / NaN / pd.NA
      - a scalar string (e.g. "GOALTEND")
      - a list/tuple/dict (e.g. ["and1", "second_chance"])
      - other objects

    Rules:
      - None / NaN / pd.NA -> ""
      - everything else -> str(val).lower()
    """
    if val is None:
        return ""

    # Explicitly treat pandas' scalar NA as empty.
    try:
        import pandas as pd  # local import to avoid circulars in some environments

        if val is pd.NA:
            return ""
    except Exception:
        # If pandas isn't available here for some reason, fall through.
        pass

    # Treat scalar NaN (Python or NumPy float) as empty.
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return ""

    # Lists / tuples / dicts / arrays / strings:
    # use their string repr and lower-case it.
    return str(val).lower()


def _vectorized_is_and_one(qualifiers: pd.Series | None) -> pd.Series:
    """
    Vectorized check for And-One events.

    Accepts a Series whose values may be:
      - strings
      - lists/tuples/dicts
      - None / NaN / pd.NA

    Returns a boolean Series indexed like the input.
    """
    if qualifiers is None:
        # No qualifiers at all: return empty bool Series
        return pd.Series(dtype=bool)

    if len(qualifiers) == 0:
        # Preserve index on empty Series
        return pd.Series(False, index=qualifiers.index, dtype=bool)

    # Normalize every cell to a lowercase string using the shared helper.
    q_str = qualifiers.apply(_qualifier_to_str)

    # Same regex semantics as before.
    return q_str.str.contains(r"and[ -]?one|and1", regex=True)


def _vectorized_shot_zone(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized calculation of shot zones.
    """
    # Handle the case where shot_distance is missing (area-only data).
    raw_distance = df.get("shot_distance")
    if raw_distance is None:
        # No distance information at all: start with an all-NaN Series so
        # distance-based masks simply don't fire.
        shot_distance = pd.Series(np.nan, index=df.index)
    else:
        shot_distance = pd.to_numeric(raw_distance, errors="coerce")

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

    return zones.where(~zones.isna(), None)


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

    # --- V2 API Event Action Type (for robust foul classification) ---
    action_type = df.get("eventmsgactiontype")
    if action_type is not None:
        try:
            action_type = pd.to_numeric(action_type, errors="coerce")
            # Use Int64 (nullable integer type) if possible for robust handling of NaNs
            if action_type.isnull().any():
                try:
                    action_type = action_type.astype("Int64")
                except TypeError:
                    # pandas too old, keep as float
                    pass
        except Exception:
            # If conversion fails completely, set to None
            action_type = None

    # Helper to safely check action type equality or inclusion
    def _check_action_type(val_or_list):
        if action_type is None:
            return pd.Series(False, index=df.index)
        if isinstance(val_or_list, (list, set, tuple)):
            return action_type.isin(val_or_list)
        return action_type == val_or_list

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
        # extra robustness for non-standard inputs:
        is_shot_like |= fam.isin(["2pt", "3pt"])
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

    # Fallback heuristic based on text
    sub_lower = subfam.astype(str).str.lower()
    heuristic_last_ft = df["is_ft"] & (
        sub_lower.str.contains("1 of 1")
        | sub_lower.str.contains("2 of 2")
        | sub_lower.str.contains("3 of 3")
    )

    if "ft_n" in df.columns and "ft_m" in df.columns:
        try:
            ft_n = pd.to_numeric(df["ft_n"], errors="coerce").fillna(0).astype(int)
            ft_m = pd.to_numeric(df["ft_m"], errors="coerce").fillna(0).astype(int)
            df["is_last_ft"] = (ft_n == ft_m) & (ft_n > 0)
        except (ValueError, TypeError):
            # If conversion fails (e.g., bad data), use the heuristic fallback
            df["is_last_ft"] = heuristic_last_ft
    else:
        # Fallback heuristic when counters are missing
        df["is_last_ft"] = heuristic_last_ft

    if "is_o_rebound" not in df.columns:
        df["is_o_rebound"] = 0
    if "is_d_rebound" not in df.columns:
        df["is_d_rebound"] = 0

    # Three-pointers
    if "is_three" in df.columns:
        df["is_three"] = df["is_three"].fillna(False).astype(bool)
    else:
        # Fallback heuristic: treat long-distance FG attempts as 3s if distance is known.
        dist = df.get("shot_distance")
        if dist is not None:
            dist_num = pd.to_numeric(dist, errors="coerce")
            # Conservative threshold; only used when is_three is missing.
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
    # Expanded criteria for live-ball turnovers based on common subfamily descriptions.
    sub_live_flag = (
        sub_lower.str.contains("live")
        | sub_lower.str.contains("bad pass")
        | sub_lower.str.contains("lost ball")
    )

    df["is_turnover_live"] = is_tov_family & (is_steal_flag | sub_live_flag)
    df["is_turnover_dead"] = is_tov_family & ~df["is_turnover_live"]

    # --- Foul flavors ---
    is_foul_family = fam == "foul"
    sub = sub_lower

    # Loose ball foul (v2 type 6)
    df["is_loose_ball_foul"] = is_foul_family & (
        sub.str.contains("loose") | _check_action_type(6)
    )

    # Flagrant fouls (v2 types 11-15, 27-29)
    flagrant_types = {11, 12, 13, 14, 15, 27, 28, 29}
    df["is_flagrant"] = is_foul_family & (
        sub.str.contains("flagrant") | _check_action_type(flagrant_types)
    )

    # Technical fouls (v2 types 16, 18-20, 25, 30).
    technical_types = {16, 18, 19, 20, 25, 30}
    df["is_technical"] = is_foul_family & (
        sub.str.contains("technical") | _check_action_type(technical_types)
    )

    # Charging/Offensive fouls (v2 type 2)
    charge_mask = sub.str.contains(r"\bcharging\b") | sub.str.contains(r"\bcharge\b")
    df["is_charge"] = is_foul_family & (charge_mask | _check_action_type(2))

    # --- And-ones via qualifiers ---
    if "qualifiers" in df.columns:
        quals_series = df["qualifiers"]
    else:
        quals_series = pd.Series([None] * len(df), index=df.index)
    df["is_and_one"] = _vectorized_is_and_one(quals_series)

    # --- Goaltends (Centralized) ---
    goaltend_flag = sub.str.contains("goaltend")

    # CDN feeds may encode goaltends via qualifiers instead of subfamily.
    if "qualifiers" in df.columns:
        # Use the shared normalizer so we don't care if qualifiers is a string,
        # list, dict, NaN, etc.
        quals_str = df["qualifiers"].apply(_qualifier_to_str)
        goaltend_flag = goaltend_flag | quals_str.str.contains("goaltend")

    df["is_goaltend"] = goaltend_flag

    # --- Shot zones ---
    shot_mask = df["is_fg_attempt"]
    if "shot_distance" in df.columns or "area" in df.columns:
        zones = _vectorized_shot_zone(df)
        zones = zones.where(shot_mask, None)
        df["shot_zone"] = zones
    else:
        df["shot_zone"] = None

    # --- Off/def team ids for event-level context ---
    off_mask = df["is_fg_attempt"] | df["is_ft"] | (df["family"] == "turnover")
    df["off_team_id"] = np.where(off_mask, df["team_id"], np.nan)
    df["def_team_id"] = np.where(
        off_mask,
        np.where(
            df["off_team_id"] == df["home_team_id"], df["away_team_id"], df["home_team_id"]
        ),
        np.nan,
    )

    return df


def _increment_count(counter: Dict[str, float], key: str, value: float = 1.0) -> None:
    counter[key] += value


def _valid_player_id(pid: Any) -> bool:
    """
    Return True if pid represents a real player id (non-null, non-zero).

    Used anywhere we’re looping over lineup slots to avoid accidentally
    treating NaN/None/0 as real players.
    """
    if pid is None:
        return False

    # Use pandas.isna for robust checking of NaN/None across types
    if pd.isna(pid):
        return False

    try:
        # Attempt to convert to integer and check if it's non-zero
        return int(pid) != 0
    except (TypeError, ValueError):
        # Handle cases where conversion fails (e.g., strings that are not numbers)
        return False


def accumulate_player_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    counts: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for _, row in df.iterrows():
        player_id_raw = row.get("player1_id")
        player_id = _coerce_id_scalar(player_id_raw) if _valid_player_id(player_id_raw) else None

        # Prefer the normalized team_id from annotate_events (event_team +
        # home/away mapping). Fall back to the raw player1_team_id when the
        # normalized value is missing.
        team_id = row.get("team_id")
        if team_id is None or pd.isna(team_id) or team_id == 0:
            team_id = row.get("player1_team_id")
        team_id = _coerce_id_scalar(team_id)
        game_id = _coerce_id_scalar(row.get("game_id"))

        if not _valid_player_id(player_id):
            player_id = None

        key = (game_id, team_id, player_id)

        if row.get("is_fg_attempt"):
            is_make = bool(row.get("is_fg_make"))
            is_three = bool(row.get("is_three"))
            zone = row.get("shot_zone")

            # Base FGA / FGM / 3PA / 3PM and zonal FGA/FGM
            _increment_count(counts[key], "FGA")
            if is_make:
                _increment_count(counts[key], "FGM")
            if is_three:
                _increment_count(counts[key], "ThreePA")
                if is_make:
                    _increment_count(counts[key], "ThreePM")
            if zone:
                _increment_count(counts[key], f"{zone}_FGA")
                if is_make:
                    _increment_count(counts[key], f"{zone}_FGM")

            # Assist handling:
            #   - Only for made shots.
            #   - Prefer explicit assist_id, fallback to player2_id for v2.
            assist_id = None
            if is_make:
                shooter = row.get("player1_id")
                shooter_team = row.get("player1_team_id")

                # 1) Prefer explicit assist_id (modern CDN / Stats feeds)
                potential_assist = row.get("assist_id")
                if _valid_player_id(potential_assist) and potential_assist != shooter:
                    assist_id = potential_assist

                # 2) Fallback to player2_id (v2) only if same team as shooter
                if assist_id is None and "player2_id" in row.index:
                    p2 = row.get("player2_id")
                    if _valid_player_id(p2) and p2 != shooter:
                        p2_team = row.get("player2_team_id")
                        if p2_team == shooter_team:
                            assist_id = p2

            assisted = is_make and _valid_player_id(assist_id)

            if assisted:
                # Shooter-level assisted makes
                _increment_count(counts[key], "FGM_AST", 1.0)
                if is_three:
                    _increment_count(counts[key], "ThreePM_AST")
                if zone:
                    _increment_count(counts[key], f"{zone}_FGM_AST")

                # Passer-level AST counts (by zone + 3P)
                # The assister is on the same team as the shooter (player1_team_id)
                ast_key = (
                    game_id,
                    _coerce_id_scalar(row.get("player1_team_id")),
                    _coerce_id_scalar(assist_id),
                )
                _increment_count(counts[ast_key], "AST")
                if zone:
                    _increment_count(counts[ast_key], f"AST_{zone}")
                if is_three:
                    _increment_count(counts[ast_key], "AST_3P")

            # Unassisted accounting (for both missed shots and unassisted makes)
            if not assisted:
                _increment_count(counts[key], "FGA_UNAST")
                if is_make:
                    _increment_count(counts[key], "FGM_UNAST")
                if is_three:
                    _increment_count(counts[key], "ThreePA_UNAST")
                    if is_make:
                        _increment_count(counts[key], "ThreePM_UNAST")
                if zone:
                    _increment_count(counts[key], f"{zone}_FGA_UNAST")
                    if is_make:
                        _increment_count(counts[key], f"{zone}_FGM_UNAST")

        if row.get("family") == "free_throw":
            _increment_count(counts[key], "FTA")

            # Mirror legacy logic: prefer shot_made when available, otherwise
            # fall back to points_made > 0.
            shot_made = row.get("shot_made")
            if shot_made is not None and not pd.isna(shot_made):
                if int(shot_made) == 1:
                    _increment_count(counts[key], "FTM")
            elif row.get("points_made", 0) > 0:
                _increment_count(counts[key], "FTM")

        if _valid_player_id(player_id):
            _increment_count(counts[key], "PTS", row.get("points_made", 0))

        if row.get("is_o_rebound") == 1:
            _increment_count(counts[key], "OREB")
        if row.get("is_d_rebound") == 1:
            _increment_count(counts[key], "DREB")

        if row.get("family") == "turnover" and _valid_player_id(player_id):
            _increment_count(counts[key], "TOV")
            if row.get("is_turnover_live"):
                _increment_count(counts[key], "TOV_Live")
            else:
                _increment_count(counts[key], "TOV_Dead")

        if row.get("family") == "foul" and _valid_player_id(player_id):
            # Player 1 commits the foul
            _increment_count(counts[key], "PF")
            if row.get("is_loose_ball_foul"):
                _increment_count(counts[key], "PF_Loose")
            if row.get("is_flagrant"):
                _increment_count(counts[key], "FLAGRANT")
            if row.get("is_technical"):
                _increment_count(counts[key], "TECH")

            fouled = row.get("player2_id")
            fouled_team = row.get("player2_team_id")
            if _valid_player_id(fouled) and not pd.isna(fouled_team):
                foul_key = (
                    game_id,
                    _coerce_id_scalar(fouled_team),
                    _coerce_id_scalar(fouled),
                )
                # Generic foul drawn
                _increment_count(counts[foul_key], "PF_DRAWN")

                # Charges drawn: subset of PF_DRAWN where the foul is a charge
                if row.get("is_charge"):
                    _increment_count(counts[foul_key], "CHRG")

        if row.get("is_block") == 1:
            blocker = row.get("player3_id")
            block_team = row.get("player3_team_id")
            possession_after = row.get("possession_after")
            shooter_team = row.get("team_id")

            if _valid_player_id(blocker) and not pd.isna(block_team):
                block_key = (
                    game_id,
                    _coerce_id_scalar(block_team),
                    _coerce_id_scalar(blocker),
                )
                _increment_count(counts[block_key], "BLK")
                if possession_after and possession_after == block_team:
                    _increment_count(counts[block_key], "BLK_Team")
                elif possession_after and possession_after == shooter_team:
                    _increment_count(counts[block_key], "BLK_Opp")
                else:
                    # If possession_after is unknown, default to BLK_Team (assume defensive recovery).
                    _increment_count(counts[block_key], "BLK_Team")

        if row.get("is_steal") == 1:
            stealer = row.get("player2_id")
            steal_team = row.get("player2_team_id")
            if _valid_player_id(stealer) and not pd.isna(steal_team):
                steal_key = (
                    game_id,
                    _coerce_id_scalar(steal_team),
                    _coerce_id_scalar(stealer),
                )
                _increment_count(counts[steal_key], "STL")

        # Use the centralized is_goaltend flag
        if row.get("is_goaltend"):
            # Goaltends are typically credited to player3 (defensive) or player1 (offensive violation)
            # Determine the responsible player and team.
            goaltend_player = None
            goaltend_team = None

            # Defensive goaltending (player3)
            p3_id = row.get("player3_id")
            if _valid_player_id(p3_id):
                goaltend_player = p3_id
                goaltend_team = row.get("player3_team_id")
            # Offensive goaltending/basket interference (player1)
            elif _valid_player_id(player_id):
                goaltend_player = player_id
                goaltend_team = team_id

            if (
                _valid_player_id(goaltend_player)
                and goaltend_team is not None
                and not pd.isna(goaltend_team)
            ):
                gt_key = (
                    game_id,
                    _coerce_id_scalar(goaltend_team),
                    _coerce_id_scalar(goaltend_player),
                )
                _increment_count(counts[gt_key], "Goaltends")

        if row.get("is_fg_make") and row.get("is_and_one") and _valid_player_id(player_id):
            _increment_count(counts[key], "AndOnes")

    records: List[Dict[str, Any]] = []
    for (game_id, team_id, player_id), vals in counts.items():
        if not _valid_player_id(player_id):
            continue
        record: Dict[str, Any] = {
            "game_id": game_id,
            "team_id": team_id,
            "player_id": player_id,
        }
        record.update(vals)
        records.append(record)

    result_df = pd.DataFrame(records)
    if not result_df.empty:
        num_cols = result_df.select_dtypes(include=["number"]).columns
        result_df[num_cols] = result_df[num_cols].fillna(0)
        _validate_id_dtypes(result_df, context="accumulate_player_counts")
    return result_df


def _build_canonical_team_map(df: pd.DataFrame) -> Dict[int, int]:
    """
    Infer a single canonical team_id for each player from the lineup columns.

    Returns a dict: {player_id: team_id} based on home_player_*_id / away_player_*_id.

    If a player somehow appears for multiple teams, the team with the highest
    appearance count is chosen, and a warning is emitted.
    """

    home_cols = [f"home_player_{i}_id" for i in range(1, 6)]
    away_cols = [f"away_player_{i}_id" for i in range(1, 6)]

    frames: list[pd.DataFrame] = []

    if all(col in df.columns for col in home_cols + ["home_team_id"]):
        home_df = df[["home_team_id"] + home_cols].copy()
        home_df = home_df.melt(
            id_vars="home_team_id",
            value_name="player_id",
        ).drop(columns="variable")
        home_df.rename(columns={"home_team_id": "team_id"}, inplace=True)
        frames.append(home_df)

    if all(col in df.columns for col in away_cols + ["away_team_id"]):
        away_df = df[["away_team_id"] + away_cols].copy()
        away_df = away_df.melt(
            id_vars="away_team_id",
            value_name="player_id",
        ).drop(columns="variable")
        away_df.rename(columns={"away_team_id": "team_id"}, inplace=True)
        frames.append(away_df)

    if not frames:
        return {}

    combined = pd.concat(frames, ignore_index=True)

    # Coerce IDs and drop non-players
    combined["player_id"] = (
        pd.to_numeric(combined["player_id"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    combined["team_id"] = (
        pd.to_numeric(combined["team_id"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    combined = combined[combined["player_id"] != 0]

    if combined.empty:
        return {}

    counts = (
        combined.groupby(["player_id", "team_id"], as_index=False)
        .size()
        .rename(columns={"size": "n_events"})
    )

    idx = counts.groupby("player_id")["n_events"].idxmax()
    canonical = counts.loc[idx, ["player_id", "team_id"]]

    multi_team = counts.groupby("player_id")["team_id"].nunique()
    bad_players = multi_team[multi_team > 1].index.tolist()
    if bad_players:
        import warnings

        warnings.warn(
            f"_build_canonical_team_map: players appear for multiple teams in lineup data: {bad_players}",
            RuntimeWarning,
        )

    return dict(zip(canonical["player_id"], canonical["team_id"]))


def compute_on_court_exposures(pbp: "PbP", df: pd.DataFrame) -> pd.DataFrame:
    exposures: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    canonical_map = _build_canonical_team_map(df)

    def _get_team_for_player(pid_val: Any, fallback_team_val: Any) -> int:
        """
        Resolve the team_id to use for a given player and fallback team label.

        - First prefer the canonical team inferred from lineups.
        - If the player is missing from the map (should be rare), fall back
          to the provided team label (home/away/off/def), coerced to int.
        """

        pid = _coerce_id_scalar(pid_val)
        team = canonical_map.get(pid)
        if team is not None:
            return team
        return _coerce_id_scalar(fallback_team_val)

    for _, row in df.iterrows():
        event_length = row.get("event_length", 0)
        if pd.isna(event_length):
            event_length = 0
        home_ids = [row.get(f"home_player_{i}_id") for i in range(1, 6)]
        away_ids = [row.get(f"away_player_{i}_id") for i in range(1, 6)]
        for pid in home_ids:
            if _valid_player_id(pid):
                key = (
                    _coerce_id_scalar(row.get("game_id")),
                    _get_team_for_player(pid, row.get("home_team_id")),
                    _coerce_id_scalar(pid),
                )
                _increment_count(exposures[key], "Minutes", event_length / 60.0)
        for pid in away_ids:
            if _valid_player_id(pid):
                key = (
                    _coerce_id_scalar(row.get("game_id")),
                    _get_team_for_player(pid, row.get("away_team_id")),
                    _coerce_id_scalar(pid),
                )
                _increment_count(exposures[key], "Minutes", event_length / 60.0)

        if row.get("is_block") == 1:
            block_team = row.get("player3_team_id")
            if pd.isna(block_team) or block_team == 0:
                block_ids = []
            elif block_team == row.get("home_team_id"):
                block_ids = home_ids
            else:
                block_ids = away_ids

            for pid in block_ids:
                if _valid_player_id(pid):
                    key = (
                        _coerce_id_scalar(row.get("game_id")),
                        _get_team_for_player(pid, block_team),
                        _coerce_id_scalar(pid),
                    )
                    _increment_count(exposures[key], "TM_BLK_OnCourt")

        # Rebound opportunities come from missed FGs and missed last free throws.
        is_missed_fg = row.get("is_fg_attempt") and not bool(row.get("is_fg_make"))
        is_missed_last_ft = (
            row.get("is_ft")
            and not bool(row.get("is_ft_make"))
            and row.get("is_last_ft", False)
        )

        if is_missed_fg or is_missed_last_ft:
            shoot_team = row.get("team_id")
            if pd.isna(shoot_team) or shoot_team == 0:
                continue
            home_on = home_ids
            away_on = away_ids

            # Offensive rebound opportunities for the shooting team
            for pid in (home_on if shoot_team == row.get("home_team_id") else away_on):
                if _valid_player_id(pid):
                    key = (
                        _coerce_id_scalar(row.get("game_id")),
                        _get_team_for_player(pid, shoot_team),
                        _coerce_id_scalar(pid),
                    )
                    _increment_count(exposures[key], "OnCourt_For_OREB_Total")

            # Defensive rebound opportunities for the defending team
            opp_team = (
                row.get("away_team_id")
                if shoot_team == row.get("home_team_id")
                else row.get("home_team_id")
            )
            for pid in (away_on if shoot_team == row.get("home_team_id") else home_on):
                if _valid_player_id(pid):
                    key = (
                        _coerce_id_scalar(row.get("game_id")),
                        _get_team_for_player(pid, opp_team),
                        _coerce_id_scalar(pid),
                    )
                    _increment_count(exposures[key], "OnCourt_For_DREB_Total")

    poss_df = pbp._build_possessions(df, include_event_agg=True)
    for _, poss in poss_df.iterrows():
        off_team = poss.get("off_team_id")
        def_team = poss.get("def_team_id")

        # Skip malformed possessions where we can't reliably assign a team.
        if pd.isna(off_team) or pd.isna(def_team) or off_team == 0 or def_team == 0:
            continue
        points = poss.get("points_for_offense", 0)
        def_points = poss.get("points_for_defense", 0)
        off_players = [poss.get(f"off_player_{i}_id") for i in range(1, 6)]
        def_players = [poss.get(f"def_player_{i}_id") for i in range(1, 6)]

        for pid in off_players:
            if _valid_player_id(pid):
                key = (
                    _coerce_id_scalar(poss.get("game_id")),
                    _get_team_for_player(pid, off_team),
                    _coerce_id_scalar(pid),
                )
                _increment_count(exposures[key], "POSS_OFF")
                _increment_count(exposures[key], "OnCourt_Team_Points", points)
                _increment_count(exposures[key], "OnCourt_Opp_Points", def_points)
                _increment_count(
                    exposures[key], "OnCourt_Team_3p_Att", poss.get("off_team_3PA", 0)
                )
                _increment_count(
                    exposures[key], "OnCourt_Team_3p_Made", poss.get("off_team_3PM", 0)
                )
                _increment_count(
                    exposures[key], "OnCourt_Team_FT_Att", poss.get("off_team_FTA", 0)
                )
                _increment_count(
                    exposures[key], "OnCourt_Team_FT_Made", poss.get("off_team_FTM", 0)
                )
                _increment_count(
                    exposures[key], "OnCourt_Team_FGM", poss.get("off_team_FGM", 0)
                )
                _increment_count(
                    exposures[key], "OnCourt_Team_FGA", poss.get("off_team_FGA", 0)
                )

        for pid in def_players:
            if _valid_player_id(pid):
                key = (
                    _coerce_id_scalar(poss.get("game_id")),
                    _get_team_for_player(pid, def_team),
                    _coerce_id_scalar(pid),
                )
                _increment_count(exposures[key], "POSS_DEF")
                _increment_count(exposures[key], "OnCourt_Opp_Points", points)
                _increment_count(exposures[key], "OnCourt_Team_Points", def_points)
                _increment_count(
                    exposures[key], "OnCourt_Opp_3p_Att", poss.get("off_team_3PA", 0)
                )
                _increment_count(
                    exposures[key], "OnCourt_Opp_3p_Made", poss.get("off_team_3PM", 0)
                )
                _increment_count(
                    exposures[key], "OnCourt_Opp_2p_Att", poss.get("off_team_2PA", 0)
                )
                _increment_count(
                    exposures[key], "OnCourt_Opp_FT_Att", poss.get("off_team_FTA", 0)
                )
                _increment_count(
                    exposures[key], "OnCourt_Opp_FT_Made", poss.get("off_team_FTM", 0)
                )
                _increment_count(
                    exposures[key], "OnCourt_Opp_FGA", poss.get("off_team_FGA", 0)
                )
                _increment_count(
                    exposures[key], "OnCourt_Opp_FGM", poss.get("off_team_FGM", 0)
                )

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
        vals.setdefault("OnCourt_Team_FGA", 0)
        vals.setdefault("OnCourt_Opp_Points", 0)
        vals.setdefault("OnCourt_Opp_3p_Made", 0)
        vals.setdefault("OnCourt_Opp_3p_Att", 0)
        vals.setdefault("OnCourt_Opp_FT_Made", 0)
        vals.setdefault("OnCourt_Opp_FT_Att", 0)
        vals.setdefault("TM_BLK_OnCourt", 0)
        vals.setdefault("OnCourt_Opp_FGM", 0)
        vals.setdefault("OnCourt_Opp_FGA", 0)
        vals.setdefault("OnCourt_For_OREB_Total", 0)
        vals.setdefault("OnCourt_For_DREB_Total", 0)
        vals.setdefault("OnCourt_Opp_2p_Att", 0)
        exposure_rows.append({"game_id": game_id, "team_id": team_id, "player_id": player_id, **vals})

    exposure_df = pd.DataFrame(exposure_rows)

    if not exposure_df.empty:
        _validate_id_dtypes(exposure_df, context="compute_on_court_exposures")

        duplicates = exposure_df.groupby(["game_id", "player_id"])["team_id"].nunique()
        bad = duplicates[duplicates > 1]
        if not bad.empty:
            raise AssertionError(
                "compute_on_court_exposures: player(s) appear for multiple teams in one game: "
                f"{list(bad.index)}"
            )

    return exposure_df

def build_player_box(
    df: pd.DataFrame,
    counts_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    player_meta: Optional[pd.DataFrame] = None,
    game_meta: Optional[pd.DataFrame] = None,
    restrict_to_pbg: bool = False,
    player_game_meta: Optional[pd.DataFrame] = None,
    strict_invariants: bool = False,
) -> pd.DataFrame:
    def _coerce_ids(frame: pd.DataFrame) -> pd.DataFrame:
        for col in ["game_id", "team_id", "player_id"]:
            if col in frame.columns:
                frame[col] = (
                    pd.to_numeric(frame[col], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
        return frame

    df = _coerce_ids(df.copy())
    counts_df = _coerce_ids(counts_df.copy())
    exposures_df = _coerce_ids(exposures_df.copy())

    merged = counts_df.merge(exposures_df, on=["game_id", "team_id", "player_id"], how="outer")

    # Only fill NaNs in numeric columns; leave object/string metadata alone.
    num_cols = merged.select_dtypes(include=["number"]).columns
    merged[num_cols] = merged[num_cols].fillna(0)
    merged = merged[(merged["team_id"] != 0) & (merged["player_id"] != 0)]

    _validate_id_dtypes(merged, context="build_player_box::merged")

    # Sanity check: players should not appear for multiple teams in one game.
    dup_team = merged.groupby(["game_id", "player_id"])["team_id"].nunique()
    bad = dup_team[dup_team > 1]
    if strict_invariants and not bad.empty:
        raise AssertionError(
            "Player(s) appear for multiple teams in the same game: "
            f"{list(bad.index)}. This usually indicates inconsistent team_id "
            "between counts and exposures (check player1_team_id vs lineup columns)."
        )
    elif not bad.empty:
        import warnings

        warnings.warn(
            "Player(s) appear for multiple teams in the same game; exposures"
            " may be misaligned with counting stats. This usually indicates"
            " inconsistent team_id between counts and exposures (check"
            " player1_team_id vs lineup columns).",
            RuntimeWarning,
        )

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

    if strict_invariants and not zero_minute_with_points.empty:
        raise AssertionError(
            "Found zero-minute rows with non-zero on-court points; "
            "this indicates an upstream bug in exposures/timing logic."
        )

    # By default, keep such rows so that on-court scoring sums remain consistent
    # with team totals (tests enforce this). When strict_invariants is True the
    # function raises instead of continuing.
    if not zero_minute_with_points.empty:
        import warnings

        warnings.warn(
            "Found zero-minute rows with non-zero on-court points; "
            "this indicates a mismatch between timing and exposures. "
            "Rows are kept to preserve on-court scoring invariants.",
            RuntimeWarning,
        )

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
        for col in ["player_id", "NbaDotComID"]:
            if col in pm.columns:
                pm[col] = pd.to_numeric(pm[col], errors="coerce").fillna(0).astype(int)
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
        gm = game_meta.copy()
        if "game_id" in gm.columns:
            gm["game_id"] = pd.to_numeric(gm["game_id"], errors="coerce").fillna(0).astype(int)
        merged = merged.merge(gm, on="game_id", how="left")

    if player_game_meta is not None and not player_game_meta.empty:
        pgm = player_game_meta.copy()
        if "game_id" in pgm.columns:
            pgm["game_id"] = pd.to_numeric(pgm["game_id"], errors="coerce").fillna(0).astype(int)
        for col in ["team_id", "player_id"]:
            if col in pgm.columns:
                pgm[col] = pd.to_numeric(pgm[col], errors="coerce").fillna(0).astype(int)
        merged = merged.merge(
            pgm,
            on=["game_id", "team_id", "player_id"],
            how="outer",
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

    # Now that we have DNP/Inactive flags, decide which rows to keep.
    # - If restrict_to_pbg is True and there are no timing mismatches,
    #   keep only players with Minutes > 0.
    # - Otherwise, keep:
    #     * players with Minutes > 0
    #     * OR players with non-zero on-court scoring exposure
    #     * OR players flagged as DNP/Inactive (so they show up on the roster).
    if restrict_to_pbg and zero_minute_with_points.empty:
        keep_mask = merged["Minutes"] > 0
    else:
        keep_mask = (
            (merged.get("Minutes", 0) > 0)
            | (merged.get("OnCourt_Team_Points", 0) != 0)
            | (merged.get("OnCourt_Opp_Points", 0) != 0)
            | (merged.get("DNP", 0) != 0)
            | (merged.get("Inactive", 0) != 0)
        )

    merged = merged[keep_mask]

    merged["NbaDotComID"] = merged["player_id"].fillna(0).astype(int)

    required_zero_cols = [
        "ThreePA",
        "ThreePM",
        "TOV",
        "AST",
        "POSS_OFF",
        "POSS_DEF",
        "OnCourt_For_OREB_Total",
        "OnCourt_For_DREB_Total",
        "OnCourt_Opp_2p_Att",
        "OnCourt_Team_FGM",
        "OnCourt_Team_FGA",
        "OnCourt_Team_Points",
        "OnCourt_Opp_Points",
        "OnCourt_Team_FT_Att",
        "OnCourt_Team_FT_Made",
        "OnCourt_Team_3p_Att",
        "OnCourt_Team_3p_Made",
        "Minutes",
        "FTA",
        "FTM",
        "FGA",
        "FGM",
        "POSS",
        "OREB",
        "DREB",
        "PF",
        "BLK",
        "BLK_Team",
        "BLK_Opp",
        "STL",
        "PF_DRAWN",
        "CHRG",
        "Goaltends",
        "AndOnes",
        "TOV_Live",
        "TOV_Dead",
        "FGM_AST",
        "ThreePM_AST",
    ]
    for col in required_zero_cols:
        if col not in merged.columns:
            merged[col] = 0
        else:
            merged[col] = merged[col].fillna(0)

    for col in ["ThreePA_UNAST", "ThreePM_UNAST", "FGA_UNAST", "FGM_UNAST"]:
        if col not in merged.columns:
            merged[col] = 0

    # Build simple derived columns in a batch to reduce fragmentation
    new_cols = {}

    new_cols["TSAttempts"] = merged["FGA"] + 0.44 * merged["FTA"]
    new_cols["TSpct"] = np.where(
        new_cols["TSAttempts"] > 0,
        merged["PTS"] / (2.0 * new_cols["TSAttempts"]),
        0.0,
    )
    new_cols["TSPoss"] = new_cols["TSAttempts"]
    new_cols["TS"] = new_cols["TSpct"]
    new_cols["PossessionsUsed"] = merged["FGA"] + 0.44 * merged["FTA"] + merged.get("TOV", 0)
    new_cols["USG"] = np.where(
        merged["POSS_OFF"] > 0,
        new_cols["PossessionsUsed"] / merged["POSS_OFF"],
        0.0,
    )

    new_cols["FGPct"] = np.where(merged["FGA"] > 0, merged["FGM"] / merged["FGA"], 0.0)
    new_cols["FT_pct"] = np.where(merged["FTA"] > 0, merged["FTM"] / merged["FTA"], 0.0)
    new_cols["ThreeP_pct"] = np.where(merged["ThreePA"] > 0, merged["ThreePM"] / merged["ThreePA"], 0.0)
    new_cols["FTR_Att"] = np.where(merged["FGA"] > 0, merged["FTA"] / merged["FGA"], 0.0)
    new_cols["FTR_Made"] = np.where(merged["FGA"] > 0, merged["FTM"] / merged["FGA"], 0.0)

    new_cols["ORBpct"] = np.where(
        merged["OnCourt_For_OREB_Total"] > 0,
        merged.get("OREB", 0) / merged["OnCourt_For_OREB_Total"],
        0.0,
    )
    new_cols["DRBpct"] = np.where(
        merged["OnCourt_For_DREB_Total"] > 0,
        merged.get("DREB", 0) / merged["OnCourt_For_DREB_Total"],
        0.0,
    )

    # Attach them all at once
    merged = pd.concat(
        [merged, pd.DataFrame(new_cols, index=merged.index)],
        axis=1,
    )

    # Alias rebound opportunity totals to glossary column names
    merged["OnCourt_For_OREB_FGA"] = merged.get("OnCourt_For_OREB_Total", 0)
    merged["OnCourt_For_DREB_FGA"] = merged.get("OnCourt_For_DREB_Total", 0)

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

    zone_cols = {}

    for zone, label in [("0_3", "0_3ft"), ("4_9", "4_9ft"), ("10_17", "10_17ft"), ("18_23", "18_23ft")]:
        fga_col = f"{zone}_FGA"
        fgm_col = f"{zone}_FGM"
        if fga_col not in merged.columns:
            merged[fga_col] = 0
        if fgm_col not in merged.columns:
            merged[fgm_col] = 0

        # Build all derived names
        fga = merged[fga_col]
        fgm = merged[fgm_col]

        zone_cols[f"{label}_FGA_100p"] = np.where(merged["POSS_OFF"] > 0, fga / merged["POSS_OFF"] * 100.0, 0.0)
        zone_cols[f"{label}_FGM_100p"] = np.where(merged["POSS_OFF"] > 0, fgm / merged["POSS_OFF"] * 100.0, 0.0)
        zone_cols[f"{label}_FGPct"] = np.where(fga > 0, fgm / fga, 0.0)

        una_fga = merged.get(f"{zone}_FGA_UNAST")
        una_fgm = merged.get(f"{zone}_FGM_UNAST")

        if una_fga is None:
            una_fga = pd.Series(0, index=merged.index)
        if una_fgm is None:
            una_fgm = pd.Series(0, index=merged.index)

        merged[f"{zone}_FGA_UNAST"] = una_fga
        merged[f"{zone}_FGM_UNAST"] = una_fgm

        zone_cols[f"{label}_FGA_UNAST_100p"] = np.where(
            merged["POSS_OFF"] > 0,
            una_fga / merged["POSS_OFF"] * 100.0,
            0.0,
        )
        zone_cols[f"{label}_FGM_UNAST_100p"] = np.where(
            merged["POSS_OFF"] > 0,
            una_fgm / merged["POSS_OFF"] * 100.0,
            0.0,
        )
        zone_cols[f"{label}_FG_UNAST_Pct"] = np.where(una_fga > 0, una_fgm / una_fga, 0.0)

    # Attach zone-derived columns in one shot
    merged = pd.concat(
        [merged, pd.DataFrame(zone_cols, index=merged.index)],
        axis=1,
    )

    merged["ThreePM_UNAST_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get("ThreePM_UNAST", 0) / merged["POSS_OFF"] * 100.0, 0)
    merged["ThreePA_UNAST_100p"] = np.where(merged["POSS_OFF"] > 0, merged.get("ThreePA_UNAST", 0) / merged["POSS_OFF"] * 100.0, 0)
    merged["ThreeP_UNAST_Pct"] = np.where(merged.get("ThreePA_UNAST", 0) > 0, merged.get("ThreePM_UNAST", 0) / merged.get("ThreePA_UNAST", 0), 0)

    merged["FGM_100p_AST"] = np.where(merged["POSS_OFF"] > 0, merged.get("FGM_AST", 0) / merged["POSS_OFF"] * 100.0, 0)
    merged["ThreePM_100p_AST"] = np.where(merged["POSS_OFF"] > 0, merged.get("ThreePM_AST", 0) / merged["POSS_OFF"] * 100.0, 0)

    # --- Shooter-side zonal assisted FGM aliases ---
    ast_zone_cols = {}
    for zone, label in [("0_3", "0_3ft"), ("4_9", "4_9ft"), ("10_17", "10_17ft"), ("18_23", "18_23ft")]:
        fgm_ast_col = f"{zone}_FGM_AST"
        if fgm_ast_col not in merged.columns:
            merged[fgm_ast_col] = 0

        # Raw count: e.g. 0_3ft_FGM_AST
        ast_zone_cols[f"{label}_FGM_AST"] = merged.get(fgm_ast_col, 0)

        # Per-100 offensive possessions: e.g. 0_3ft_FGM_100p_AST
        ast_zone_cols[f"{label}_FGM_100p_AST"] = np.where(
            merged["POSS_OFF"] > 0,
            merged.get(fgm_ast_col, 0) / merged["POSS_OFF"] * 100.0,
            0.0,
        )

    for zone, label in [("0_3", "AST_0_3ft"), ("4_9", "AST_4_9ft"), ("10_17", "AST_10_17ft"), ("18_23", "AST_18_23ft")]:
        ast_zone_cols[f"{label}_100p"] = np.where(
            merged["POSS_OFF"] > 0,
            merged.get(f"AST_{zone}", 0) / merged["POSS_OFF"] * 100.0,
            0.0,
        )

    merged = pd.concat(
        [merged, pd.DataFrame(ast_zone_cols, index=merged.index)],
        axis=1,
    )
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
    # Alias on-court rebound opportunity counts to glossary-style names,
    # without clobbering any existing columns if they appear upstream.
    if "OnCourt_For_OREB_FGA" not in merged.columns:
        merged["OnCourt_For_OREB_FGA"] = merged.get("OnCourt_For_OREB_Total", 0)
    if "OnCourt_For_DREB_FGA" not in merged.columns:
        merged["OnCourt_For_DREB_FGA"] = merged.get("OnCourt_For_DREB_Total", 0)

    merged = merged.copy()
    _validate_id_dtypes(merged, context="build_player_box::output")
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

    box_df = box_df.copy()
    _validate_id_dtypes(box_df, context="append_team_totals::input")

    id_cols = ["game_id", "team_id"]
    numeric_cols = box_df.select_dtypes(include=[np.number]).columns.tolist()

    # We do NOT want to sum identifiers or flags that are per-row, not additive.
    do_not_sum = {
        "NbaDotComID",
        "PlayerID",
        "player_id",
        "PlayerSeasonID",
        "Game_SingleGame",
        "Team_SingleGame",
        "home_fl",
        "h_tm_id",
        "v_tm_id",
        "season",
        "Year",
        "game_id",
        "team_id",
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
    team_totals["player_id"] = 0
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

    # Use offensive possessions for team pace so we don't double-count.
    team_totals["Pace"] = np.where(
        team_totals["Minutes"] > 0,
        team_totals["POSS_OFF"] / team_totals["Minutes"] * 48.0,
        0.0,
    )

    # Recompute team-level shooting percentages (avoid summed player rates).
    team_totals["FGPct"] = np.where(
        team_totals["FGA"] > 0,
        team_totals["FGM"] / team_totals["FGA"],
        0.0,
    )
    team_totals["FT_pct"] = np.where(
        team_totals["FTA"] > 0,
        team_totals["FTM"] / team_totals["FTA"],
        0.0,
    )
    team_totals["FT%"] = team_totals["FT_pct"]

    if "ThreePM" in team_totals.columns and "ThreePA" in team_totals.columns:
        team_totals["ThreeP_pct"] = np.where(
            team_totals["ThreePA"] > 0,
            team_totals["ThreePM"] / team_totals["ThreePA"],
            0.0,
        )
        team_totals["3PPct"] = team_totals["ThreeP_pct"]

    team_totals["PTS_100p"] = np.where(
        team_totals["POSS_OFF"] > 0,
        team_totals["PTS"] / team_totals["POSS_OFF"] * 100.0,
        0.0,
    )

    # Label the team row.
    team_totals["Team"] = "TOTAL"

    # Concatenate players + team totals
    merged = pd.concat([box_df, team_totals], ignore_index=True)
    _validate_id_dtypes(merged, context="append_team_totals::output")
    return merged
