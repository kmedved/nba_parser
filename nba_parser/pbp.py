from datetime import datetime
import math
import numpy as np
import pandas as pd
from .box_glossary import (
    annotate_events,
    accumulate_player_counts,
    compute_on_court_exposures,
    build_player_box,
    append_team_totals,
    _coerce_id_scalar,
)


def normalize_game_id(game_id):
    """
    Accept int or string game IDs of length 8 or 10 and return a canonical int.
    Examples:
        "0022200001" -> 22200001
        "22200001"   -> 22200001
        22200001     -> 22200001
    """
    if game_id is None:
        return None
    return int(str(game_id))


def format_game_id(game_id):
    """
    Format a normalized game_id as a 10-digit string with leading zeros.
    Example:
        22200001 -> "0022200001"
    """
    if game_id is None:
        return None
    return f"{int(game_id):010d}"

# NOTE:
# The legacy *_calc_player and playerbygamestats() methods implement the original
# per-player stat calculations based on the v2 pbp format. New code should prefer
# player_box_glossary(), which uses annotate_events / accumulate_player_counts /
# compute_on_court_exposures in nba_parser.box_glossary. The legacy methods are kept
# for backwards compatibility and for the existing test suite.


class PbP:
    """
    This class represents one game of of an NBA play by play dataframe. I am
    building methods on top of this class to streamline the calculation of
    stats from the play by player and then insertion into a database of the
    users choosing
    """

    def __init__(self, pbp_df):
        self.df = pbp_df

        id_columns = [
            "game_id",
            "team_id",
            "home_team_id",
            "away_team_id",
            "player1_team_id",
            "player2_team_id",
            "player3_team_id",
            "player1_id",
            "player2_id",
            "player3_id",
            "assist_id",
            *(f"home_player_{i}_id" for i in range(1, 6)),
            *(f"away_player_{i}_id" for i in range(1, 6)),
        ]

        for col in id_columns:
            if col in self.df.columns:
                self.df[col] = (
                    pd.to_numeric(self.df[col], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )

        # Enforce single-game input; many invariants assume one game_id.
        game_ids = self.df["game_id"].unique()
        if len(game_ids) != 1:
            raise ValueError(
                f"PbP expects a single-game DataFrame, but found game_ids={game_ids}"
            )
        # Normalize 8- or 10-character IDs; store as int internally
        normalized_game_id = normalize_game_id(game_ids[0])
        self.game_id = normalized_game_id
        self.df["game_id"] = normalized_game_id
        self.home_team = self.df["home_team_abbrev"].unique()[0]
        self.away_team = self.df["away_team_abbrev"].unique()[0]
        self.home_team_id = int(self.df["home_team_id"].unique()[0])
        self.away_team_id = int(self.df["away_team_id"].unique()[0])
        self.season = self.df["season"].unique()[0]

        # Handle PbP classes created from imported CSV files versus those
        # created by nba_scraper that handles game_date as a proper datetime.
        if self.df["game_date"].dtypes == "O":
            raw = str(pbp_df["game_date"].unique()[0])
            parsed = None
            for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
                try:
                    parsed = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                # Fallback: let pandas try to infer.
                parsed = pd.to_datetime(raw)
            self.game_date = parsed
            self.df["game_date"] = pd.to_datetime(self.df["game_date"])
        else:
            self.game_date = pbp_df["game_date"].unique()[0]

        # change column types to fit my database at a later time on insert

        self.df["scoremargin"] = self.df["scoremargin"].astype(str)

        # calculating home and away possesions to later aggregate for players
        # and teams

        # NOTE: The home_possession / away_possession flags below implement the
        # original v2-era heuristics for detecting possession boundaries. New
        # possession-based analysis (RAPM, on/off exposures, etc.) should prefer
        # the _build_possessions() output as the canonical representation of
        # possessions. These columns are kept for backwards compatibility.

        # Prefer canonical possession_after when available (CDN + modern schema),
        # fall back to legacy text-based heuristics otherwise.
        if "possession_after" in self.df.columns and self.df["possession_after"].notna().any():
            # Start with zeros.
            self.df["home_possession"] = 0
            self.df["away_possession"] = 0

            home_id = self.home_team_id
            away_id = self.away_team_id

            pos_raw = self.df["possession_after"]

            # Normalize: treat only home/away IDs as valid possession owners.
            # Forward-fill to cover sequences of events where possession doesn't change.
            pos_team = pos_raw.where(pos_raw.isin([home_id, away_id]))
            pos_team = pos_team.ffill()

            # For any leading NaNs (pre-jump-ball) that are still NaN after ffill,
            # backfill them with the first valid possession owner so those events
            # belong to the first possession.
            pos_team = pos_team.bfill()

            # At this point, pos_team is constant over stretches where the same
            # team has the ball. The end of each stretch is a possession boundary.
            is_last_in_stretch = (pos_team != pos_team.shift(-1)) | pos_team.isna()

            # Mark the final event of each possession for home vs away.
            home_pos_idx = is_last_in_stretch & (pos_team == home_id)
            away_pos_idx = is_last_in_stretch & (pos_team == away_id)

            self.df.loc[home_pos_idx, "home_possession"] = 1
            self.df.loc[away_pos_idx, "away_possession"] = 1

        else:
            # --- LEGACY HEURISTIC POSSESSION FLAGS (existing v2-style code path) ---

            # calculating made shot possessions
            self.df["home_possession"] = np.where(
                (self.df.event_team == self.df.home_team_abbrev)
                & (self.df.event_type_de == "shot"),
                1,
                0,
            )

            # calculating turnover possessions
            self.df["home_possession"] = np.where(
                (self.df.event_team == self.df.home_team_abbrev)
                & (self.df.event_type_de == "turnover"),
                1,
                self.df["home_possession"],
            )

            # calculating defensive rebound possessions
            self.df["home_possession"] = np.where(
                (
                    (self.df.event_team == self.df.away_team_abbrev)
                    & (self.df.is_d_rebound == 1)
                )
                | (
                    (self.df.event_type_de == "rebound")
                    & (self.df.is_d_rebound == 0)
                    & (self.df.is_o_rebound == 0)
                    & (self.df.event_team == self.df.away_team_abbrev)
                    & (self.df.event_type_de.shift(1) != "free-throw")
                ),
                1,
                self.df["home_possession"],
            )

            # Determine if it's the last free throw using structured data if available
            is_last_ft = pd.Series(False, index=self.df.index)

            # Ensure descriptions are treated as strings and handle NaNs
            homedesc = self.df.get("homedescription", pd.Series(index=self.df.index)).fillna("").astype(str)
            visitordesc = self.df.get("visitordescription", pd.Series(index=self.df.index)).fillna("").astype(str)

            # Heuristic fallback using string matching
            string_match_last_ft = (
                homedesc.str.contains("Free Throw 1 of 1")
                | homedesc.str.contains("Free Throw 2 of 2")
                | homedesc.str.contains("Free Throw 3 of 3")
                | visitordesc.str.contains("Free Throw 1 of 1")
                | visitordesc.str.contains("Free Throw 2 of 2")
                | visitordesc.str.contains("Free Throw 3 of 3")
            )

            if "ft_n" in self.df.columns and "ft_m" in self.df.columns:
                try:
                    ft_n = self.df["ft_n"].fillna(0).astype(int)
                    ft_m = self.df["ft_m"].fillna(0).astype(int)
                    structured_last_ft = (ft_n == ft_m) & (ft_n > 0)
                    # Use structured data if valid, otherwise fallback to string match
                    is_last_ft = np.where(
                        (ft_n > 0) & (ft_m > 0),
                        structured_last_ft,
                        string_match_last_ft,
                    )
                except (ValueError, TypeError):
                    # Fallback if structured data is malformed
                    is_last_ft = string_match_last_ft
            else:
                # Fallback if columns are missing
                is_last_ft = string_match_last_ft

            # calculating final free throw possessions (Home)
            self.df["home_possession"] = np.where(
                (self.df.event_team == self.df.home_team_abbrev)
                & (self.df.event_type_de == "free-throw")
                & is_last_ft,
                1,
                self.df["home_possession"],
            )

            # calculating made shot possessions (Away)
            self.df["away_possession"] = np.where(
                (self.df.event_team == self.df.away_team_abbrev)
                & (self.df.event_type_de == "shot"),
                1,
                0,
            )

            # calculating turnover possessions (Away)
            self.df["away_possession"] = np.where(
                (self.df.event_team == self.df.away_team_abbrev)
                & (self.df.event_type_de == "turnover"),
                1,
                self.df["away_possession"],
            )

            # calculating defensive rebound possessions (Away)
            self.df["away_possession"] = np.where(
                (
                    (self.df.event_team == self.df.home_team_abbrev)
                    & (self.df.is_d_rebound == 1)
                )
                | (
                    (self.df.event_type_de == "rebound")
                    & (self.df.is_d_rebound == 0)
                    & (self.df.is_o_rebound == 0)
                    & (self.df.event_team == self.df.home_team_abbrev)
                    & (self.df.event_type_de.shift(1) != "free-throw")
                ),
                1,
                self.df["away_possession"],
            )

            # calculating final free throw possessions (Away)
            self.df["away_possession"] = np.where(
                (self.df.event_team == self.df.away_team_abbrev)
                & (self.df.event_type_de == "free-throw")
                & is_last_ft,
                1,
                self.df["away_possession"],
            )

    def _point_calc_player(self):
        """
        LEGACY: v2-based stat calculation. Kept for backwards compatibility and tests.
        New code should prefer player_box_glossary() and the box_glossary helpers.

        method calculates simple shooting stats like field goals, three points,
        and free throws made and attempted.
        """
        self.df["fgm"] = np.where(
            (self.df["shot_made"] == 1) & (self.df["event_type_de"] == "shot"), 1, 0
        )
        self.df["fga"] = np.where(
            self.df["event_type_de"].str.contains("shot|missed_shot", regex=True), 1, 0
        )
        self.df["tpm"] = np.where(
            (self.df["shot_made"] == 1) & (self.df["is_three"] == 1), 1, 0
        )
        self.df["tpa"] = np.where(self.df["is_three"] == 1, 1, 0)
        self.df["ftm"] = np.where(
            (self.df["shot_made"] == 1)
            & (self.df["event_type_de"].str.contains("free-throw")),
            1,
            0,
        )
        self.df["fta"] = np.where(
            self.df["event_type_de"].str.contains("free-throw"), 1, 0
        )

        player_points_df = (
            self.df.groupby(["player1_id", "game_date", "game_id", "player1_team_id"])[
                ["fgm", "fga", "tpm", "tpa", "ftm", "fta", "points_made"]
            ]
            .sum()
            .reset_index()
        )
        player_points_df["player1_team_id"] = player_points_df[
            "player1_team_id"
        ].astype(int)
        player_points_df.rename(
            columns={
                "player1_id": "player_id",
                "player1_team_id": "team_id",
                "points_made": "points",
            },
            inplace=True,
        )

        return player_points_df

    def _assist_calc_player(self):
        """
        LEGACY: v2-based stat calculation. Kept for backwards compatibility and tests.
        New code should prefer player_box_glossary() and the box_glossary helpers.

        method to calculat players assist totals from a game play by play
        """
        assists = self.df[
            (self.df["event_type_de"] == "shot") & (self.df["shot_made"] == 1)
        ]

        assists = (
            assists.groupby(["player2_id", "game_id", "game_date", "player2_team_id"])[
                ["eventnum"]
            ]
            .count()
            .reset_index()
        )

        assists["player2_team_id"] = assists["player2_team_id"].astype(int)
        assists.rename(
            columns={
                "player2_id": "player_id",
                "player2_team_id": "team_id",
                "eventnum": "ast",
            },
            inplace=True,
        )

        return assists

    def _rebound_calc_player(self):
        """
        LEGACY: v2-based stat calculation. Kept for backwards compatibility and tests.
        New code should prefer player_box_glossary() and the box_glossary helpers.

        function to calculate player's offensive and defensive rebound totals
        """
        rebounds = (
            self.df.groupby(["player1_id", "game_id", "game_date"])[
                ["is_o_rebound", "is_d_rebound"]
            ]
            .sum()
            .reset_index()
        )

        rebounds.rename(
            columns={
                "player1_id": "player_id",
                "is_o_rebound": "oreb",
                "is_d_rebound": "dreb",
            },
            inplace=True,
        )

        return rebounds

    def _turnover_calc_player(self):
        """
        LEGACY: v2-based stat calculation. Kept for backwards compatibility and tests.
        New code should prefer player_box_glossary() and the box_glossary helpers.

        function to calculate player's turnover totals
        """
        turnovers = (
            self.df.groupby(["player1_id", "game_id", "game_date", "player1_team_id"])[
                ["is_turnover"]
            ]
            .sum()
            .reset_index()
        )

        turnovers["player1_team_id"] = turnovers["player1_team_id"].astype(int)
        turnovers.rename(
            columns={
                "player1_id": "player_id",
                "player1_team_id": "team_id",
                "is_turnover": "tov",
            },
            inplace=True,
        )

        return turnovers

    def _foul_calc_player(self):
        """
        LEGACY: v2-only foul counting logic using eventmsgactiontype codes.
        This is used by playerbygamestats() for backwards compatibility and tests.

        New foul classification for the glossary comes from annotate_events()
        (family == 'foul' plus subfamily flags) in box_glossary.py.

        method to calculate players personal fouls in a game
        """
        fouls = self.df[
            (self.df["event_type_de"] == "foul")
            & (
                self.df["eventmsgactiontype"].isin(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 26, 27, 28]
                )
            )
        ]
        fouls = (
            fouls.groupby(["player1_id", "game_id", "game_date", "player1_team_id"])[
                "eventnum"
            ]
            .count()
            .reset_index()
        )
        fouls["player1_team_id"] = fouls["player1_team_id"].astype(int)
        fouls.rename(
            columns={
                "player1_id": "player_id",
                "player1_team_id": "team_id",
                "eventnum": "pf",
            },
            inplace=True,
        )

        return fouls

    def _steal_calc_player(self):
        """
        LEGACY: v2-based stat calculation. Kept for backwards compatibility and tests.
        New code should prefer player_box_glossary() and the box_glossary helpers.

        function to calculate player's steal totals
        """
        steals = (
            self.df.groupby(["player2_id", "game_id", "game_date", "player2_team_id"])[
                ["is_steal"]
            ]
            .sum()
            .reset_index()
        )

        steals["player2_team_id"] = steals["player2_team_id"].astype(int)
        steals.rename(
            columns={
                "player2_id": "player_id",
                "player2_team_id": "team_id",
                "is_steal": "stl",
            },
            inplace=True,
        )
        return steals

    def _block_calc_player(self):
        """
        LEGACY: v2-based stat calculation. Kept for backwards compatibility and tests.
        New code should prefer player_box_glossary() and the box_glossary helpers.

        function to calculate player blocks and return a dataframe with players
        and blocked shots stats along with key columns to join to other dataframes
        """
        blocks = self.df[self.df["event_type_de"] != "jump-ball"]
        blocks = (
            blocks.groupby(["player3_id", "game_id", "game_date", "player3_team_id"])[
                ["is_block"]
            ]
            .sum()
            .reset_index()
        )
        blocks["player3_team_id"] = blocks["player3_team_id"].astype(int)
        blocks.rename(
            columns={
                "player3_id": "player_id",
                "player3_team_id": "team_id",
                "is_block": "blk",
            },
            inplace=True,
        )

        return blocks

    def _melt_lineup(
        self, df, player_cols, value_name="player_id", extra_id_vars=None
    ):
        if extra_id_vars is None:
            extra_id_vars = []
        melted = df[player_cols + extra_id_vars].melt(
            id_vars=extra_id_vars,
            value_vars=player_cols,
            var_name="slot",
            value_name=value_name,
        )
        return melted

    def _plus_minus_calc_player(self):
        """
        LEGACY: v2-based on/off plus-minus calculation.

        This method computes per-player plus/minus by:
          1. Tagging each scoring event with home_plus/home_minus/away_plus/away_minus.
          2. Aggregating non-free-throw events by on-court players.
          3. Handling free-throw events by joining fouls to the on-court lineups.
          4. Summing the two contributions.

        The implementation has been refactored to use melt-style reshaping instead
        of ten nearly-identical groupby calls.

        Players with id == 0 (placeholders / bench) are ignored.
        """

        df = self.df.copy()

        # Step 1: tag each event with team-level plus/minus contributions
        df["home_plus"] = np.where(
            df["event_team"] == df["home_team_abbrev"],
            df["points_made"],
            0,
        )
        df["home_minus"] = np.where(
            df["event_team"] != df["home_team_abbrev"],
            df["points_made"],
            0,
        )
        df["away_plus"] = np.where(
            df["event_team"] != df["home_team_abbrev"],
            df["points_made"],
            0,
        )
        df["away_minus"] = np.where(
            df["event_team"] == df["home_team_abbrev"],
            df["points_made"],
            0,
        )

        # -------------------------
        # Non-free-throw events
        # -------------------------
        no_ft_df = df[df["event_type_de"] != "free-throw"].copy()

        # Home players on non-FT events
        home_cols = [f"home_player_{i}_id" for i in range(1, 6)]
        home_pm = self._melt_lineup(
            no_ft_df,
            home_cols,
            extra_id_vars=["home_plus", "home_minus", "game_id", "game_date", "home_team_id"],
        )
        home_pm.rename(
            columns={
                "home_team_id": "team_id",
                "home_plus": "plus",
                "home_minus": "minus",
            },
            inplace=True,
        )

        # Away players on non-FT events
        away_cols = [f"away_player_{i}_id" for i in range(1, 6)]
        away_pm = self._melt_lineup(
            no_ft_df,
            away_cols,
            extra_id_vars=["away_plus", "away_minus", "game_id", "game_date", "away_team_id"],
        )
        away_pm.rename(
            columns={
                "away_team_id": "team_id",
                "away_plus": "plus",
                "away_minus": "minus",
            },
            inplace=True,
        )

        non_ft_pm = pd.concat([home_pm, away_pm], ignore_index=True)
        non_ft_pm = non_ft_pm[
            (~non_ft_pm["player_id"].isna()) & (non_ft_pm["player_id"] != 0)
        ]

        non_ft_pm = (
            non_ft_pm.groupby(["player_id", "game_id", "game_date", "team_id"], as_index=False)[["plus", "minus"]]
            .sum()
        )

        # -------------------------
        # Free-throw events
        # -------------------------

        # foul_df: who was on the court at the time of the foul
        foul_df = df[df["event_type_de"] == "foul"][
            [
                "period",
                "seconds_elapsed",
                "pctimestring",
                "home_player_1_id",
                "home_player_2_id",
                "home_player_3_id",
                "home_player_4_id",
                "home_player_5_id",
                "away_player_1_id",
                "away_player_2_id",
                "away_player_3_id",
                "away_player_4_id",
                "away_player_5_id",
            ]
        ].copy()

        # ft_df: FT events with plus/minus deltas and team info
        ft_df = df[df["event_type_de"] == "free-throw"][
            [
                "period",
                "seconds_elapsed",
                "pctimestring",
                "game_id",
                "game_date",
                "home_team_id",
                "away_team_id",
                "home_plus",
                "home_minus",
                "away_plus",
                "away_minus",
            ]
        ].copy()

        # Join FT events to the player lineups present at the foul that led to them
        ft_df = ft_df.merge(
            foul_df,
            on=["period", "seconds_elapsed", "pctimestring"],
            how="inner",
        )

        # Home players on FT events
        home_ft_cols = [f"home_player_{i}_id" for i in range(1, 6)]
        home_ft = self._melt_lineup(
            ft_df,
            home_ft_cols,
            extra_id_vars=["home_plus", "home_minus", "game_id", "game_date", "home_team_id"],
        )
        home_ft.rename(
            columns={
                "home_team_id": "team_id",
                "home_plus": "plus",
                "home_minus": "minus",
            },
            inplace=True,
        )

        # Away players on FT events
        away_ft_cols = [f"away_player_{i}_id" for i in range(1, 6)]
        away_ft = self._melt_lineup(
            ft_df,
            away_ft_cols,
            extra_id_vars=["away_plus", "away_minus", "game_id", "game_date", "away_team_id"],
        )
        away_ft.rename(
            columns={
                "away_team_id": "team_id",
                "away_plus": "plus",
                "away_minus": "minus",
            },
            inplace=True,
        )

        ft_pm = pd.concat([home_ft, away_ft], ignore_index=True)
        ft_pm = ft_pm[
            (~ft_pm["player_id"].isna()) & (ft_pm["player_id"] != 0)
        ]

        ft_pm = (
            ft_pm.groupby(["player_id", "game_id", "game_date", "team_id"], as_index=False)[["plus", "minus"]]
            .sum()
        )

        # -------------------------
        # Combine non-FT and FT contributions
        # -------------------------
        total_pm = pd.concat([non_ft_pm, ft_pm], ignore_index=True)

        total_pm = (
            total_pm.groupby(["player_id", "game_id", "game_date", "team_id"], as_index=False)[["plus", "minus"]]
            .sum()
        )
        total_pm["plus_minus"] = total_pm["plus"] - total_pm["minus"]

        return total_pm

    def _toc_calc_player(self):
        """
        LEGACY: v2-based time-on-court calculation.

        This method calculates a player's time in the game (in seconds) by summing
        event_length across all events where they appear in any of the 5 on-court
        slots (home or away). It then converts that to a MM:SS string.

        The implementation here has been refactored to use melt-style reshaping
        instead of 10 nearly-identical groupby calls.

        Players with id == 0 (placeholders / bench) are ignored.
        """
        df = self.df.copy()

        # Home players: melt home_player_1_id..home_player_5_id with event_length
        home_cols = [f"home_player_{i}_id" for i in range(1, 6)]
        home_toc = self._melt_lineup(
            df,
            home_cols,
            value_name="player_id",
            extra_id_vars=["event_length", "game_id", "game_date", "home_team_id"],
        )
        home_toc.rename(columns={"home_team_id": "team_id"}, inplace=True)

        # Away players: melt away_player_1_id..away_player_5_id with event_length
        away_cols = [f"away_player_{i}_id" for i in range(1, 6)]
        away_toc = self._melt_lineup(
            df,
            away_cols,
            value_name="player_id",
            extra_id_vars=["event_length", "game_id", "game_date", "away_team_id"],
        )
        away_toc.rename(columns={"away_team_id": "team_id"}, inplace=True)

        # Combine, drop bench/empty entries, and aggregate
        all_toc = pd.concat([home_toc, away_toc], ignore_index=True)

        all_toc = all_toc[
            (~all_toc["player_id"].isna()) & (all_toc["player_id"] != 0)
        ]

        toc = (
            all_toc.groupby(["player_id", "team_id", "game_id", "game_date"], as_index=False)["event_length"]
            .sum()
        )
        toc.rename(columns={"event_length": "toc"}, inplace=True)

        toc["toc_string"] = pd.to_datetime(toc["toc"], unit="s").dt.strftime("%M:%S")

        return toc

    def _poss_calc_player(self):
        """
        LEGACY: v2-based stat calculation. Kept for backwards compatibility and tests.
        New code should prefer player_box_glossary() and the box_glossary helpers.

        function to calculate possessions each player participated in
        """

        # calculating player possesions
        player1 = self.df[
            [
                "home_player_1",
                "home_player_1_id",
                "home_possession",
                "game_id",
                "home_team_id",
            ]
        ].rename(
            columns={"home_player_1": "player_name", "home_player_1_id": "player_id"}
        )
        player2 = self.df[
            [
                "home_player_2",
                "home_player_2_id",
                "home_possession",
                "game_id",
                "home_team_id",
            ]
        ].rename(
            columns={"home_player_2": "player_name", "home_player_2_id": "player_id"}
        )
        player3 = self.df[
            [
                "home_player_3",
                "home_player_3_id",
                "home_possession",
                "game_id",
                "home_team_id",
            ]
        ].rename(
            columns={"home_player_3": "player_name", "home_player_3_id": "player_id"}
        )
        player4 = self.df[
            [
                "home_player_4",
                "home_player_4_id",
                "home_possession",
                "game_id",
                "home_team_id",
            ]
        ].rename(
            columns={"home_player_4": "player_name", "home_player_4_id": "player_id"}
        )
        player5 = self.df[
            [
                "home_player_5",
                "home_player_5_id",
                "home_possession",
                "game_id",
                "home_team_id",
            ]
        ].rename(
            columns={"home_player_5": "player_name", "home_player_5_id": "player_id"}
        )
        home_possession_df = pd.concat([player1, player2, player3, player4, player5])
        home_possession_df = (
            home_possession_df.groupby(
                ["player_id", "player_name", "game_id", "home_team_id"]
            )["home_possession"]
            .sum()
            .reset_index()
            .sort_values("home_possession")
        )
        player1 = self.df[
            [
                "away_player_1",
                "away_player_1_id",
                "away_possession",
                "game_id",
                "away_team_id",
            ]
        ].rename(
            columns={"away_player_1": "player_name", "away_player_1_id": "player_id"}
        )
        player2 = self.df[
            [
                "away_player_2",
                "away_player_2_id",
                "away_possession",
                "game_id",
                "away_team_id",
            ]
        ].rename(
            columns={"away_player_2": "player_name", "away_player_2_id": "player_id"}
        )
        player3 = self.df[
            [
                "away_player_3",
                "away_player_3_id",
                "away_possession",
                "game_id",
                "away_team_id",
            ]
        ].rename(
            columns={"away_player_3": "player_name", "away_player_3_id": "player_id"}
        )
        player4 = self.df[
            [
                "away_player_4",
                "away_player_4_id",
                "away_possession",
                "game_id",
                "away_team_id",
            ]
        ].rename(
            columns={"away_player_4": "player_name", "away_player_4_id": "player_id"}
        )
        player5 = self.df[
            [
                "away_player_5",
                "away_player_5_id",
                "away_possession",
                "game_id",
                "away_team_id",
            ]
        ].rename(
            columns={"away_player_5": "player_name", "away_player_5_id": "player_id"}
        )
        away_possession_df = pd.concat([player1, player2, player3, player4, player5])
        away_possession_df = (
            away_possession_df.groupby(
                ["player_id", "player_name", "game_id", "away_team_id"]
            )["away_possession"]
            .sum()
            .reset_index()
            .sort_values("away_possession")
        )

        home_possession_df = home_possession_df.rename(
            columns={"home_team_id": "team_id", "home_possession": "possessions"}
        )
        away_possession_df = away_possession_df.rename(
            columns={"away_team_id": "team_id", "away_possession": "possessions"}
        )
        possession_df = pd.concat([home_possession_df, away_possession_df])

        return possession_df

    def _poss_calc_team(self):
        """
        method to calculate team possession numbers
        """

        row1 = [
            self.df.home_team_id.unique()[0],
            self.df.game_id.unique()[0],
            self.df.home_team_abbrev.unique()[0],
            self.df["home_possession"].sum(),
        ]
        row2 = [
            self.df.away_team_id.unique()[0],
            self.df.game_id.unique()[0],
            self.df.away_team_abbrev.unique()[0],
            self.df["away_possession"].sum(),
        ]
        team_possession_df = pd.DataFrame(
            [row1, row2], columns=["team_id", "game_id", "team_abbrev", "possessions"]
        )

        return team_possession_df

    def _point_calc_team(self):
        """
        method to calculate team field goals, free throws, and three points
        made
        """
        self.df["fg_attempted"] = np.where(
            self.df["event_type_de"].isin(["missed_shot", "shot"]), True, False
        )
        self.df["ft_attempted"] = np.where(
            self.df["event_type_de"] == "free-throw", True, False
        )
        self.df["fg_made"] = np.where(
            (self.df["event_type_de"].isin(["shot"])) & (self.df["points_made"] > 0),
            True,
            False,
        )
        self.df["tp_made"] = np.where(self.df["points_made"] == 3, True, False)
        self.df["ft_made"] = np.where(
            (self.df["event_type_de"] == "free-throw") & (self.df["points_made"] == 1),
            True,
            False,
        )
        valid_team_rows = self.df[
            self.df["player1_team_id"].notna() & (self.df["player1_team_id"] != 0)
        ]

        teams_df = (
            valid_team_rows.groupby(["player1_team_id", "game_id"])[
                [
                    "points_made",
                    "is_three",
                    "fg_attempted",
                    "ft_attempted",
                    "fg_made",
                    "tp_made",
                    "ft_made",
                ]
            ]
            .sum()
            .reset_index()
        )
        teams_df["player1_team_id"] = teams_df["player1_team_id"].astype(int)
        teams_df.rename(
            columns={
                "player1_team_id": "team_id",
                "points_made": "points_for",
                "is_three": "tpa",
                "fg_made": "fgm",
                "fg_attempted": "fga",
                "ft_made": "ftm",
                "ft_attempted": "fta",
                "tp_made": "tpm",
            },
            inplace=True,
        )

        return teams_df

    def _assist_calc_team(self):
        """
        method to sum assists made for each team
        """
        self.df["is_assist"] = np.where(
            (self.df["event_type_de"] == "shot") & (self.df["player2_id"] != 0),
            True,
            False,
        )
        assists_df = (
            self.df.groupby(["player1_team_id", "game_id"])[["is_assist"]]
            .sum()
            .reset_index()
        )
        assists_df.rename(
            columns={"is_assist": "ast", "player1_team_id": "team_id",}, inplace=True,
        )

        return assists_df

    def _rebound_calc_team(self):
        """
        method to calculate team offensive and deffensive rebound totals
        """
        rebounds_df = (
            self.df.groupby(["player1_team_id", "game_id"])[
                ["is_d_rebound", "is_o_rebound",]
            ]
            .sum()
            .reset_index()
        )
        rebounds_df["player1_team_id"] = rebounds_df["player1_team_id"].astype(int)
        rebounds_df.rename(
            columns={
                "player1_team_id": "team_id",
                "is_d_rebound": "dreb",
                "is_o_rebound": "oreb",
            },
            inplace=True,
        )

        return rebounds_df

    def _turnover_calc_team(self):
        turnovers_df = (
            self.df.groupby(["player1_team_id", "game_id"])[["is_turnover"]]
            .sum()
            .reset_index()
        )
        turnovers_df["player1_team_id"] = turnovers_df["player1_team_id"].astype(int)
        turnovers_df.rename(
            columns={"player1_team_id": "team_id", "is_turnover": "tov",}, inplace=True,
        )

        return turnovers_df

    def _foul_calc_team(self):
        """
        method to calculate team personal fouls taken in a game
        """

        fouls = self.df[
            (self.df["event_type_de"] == "foul")
            & (
                self.df["eventmsgactiontype"].isin(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 26, 27, 28]
                )
            )
        ]
        fouls = (
            fouls.groupby(["game_id", "player1_team_id"])["eventnum"]
            .count()
            .reset_index()
        )
        fouls["player1_team_id"] = fouls["player1_team_id"].astype(int)
        fouls.rename(
            columns={"player1_team_id": "team_id", "eventnum": "pf",}, inplace=True,
        )

        fouls = fouls.merge(fouls, on="game_id", suffixes=["", "_opponent"])

        fouls = fouls[fouls["team_id"] != fouls["team_id_opponent"]]
        fouls.rename(
            columns={"pf_opponent": "fouls_drawn",}, inplace=True,
        )

        return fouls[["team_id", "game_id", "pf", "fouls_drawn"]]

    def _steal_calc_team(self):
        """
        method to calculate team steals in a game
        """

        steals_df = (
            self.df.groupby(["player2_team_id", "game_id"])[["is_steal"]]
            .sum()
            .reset_index()
        )
        steals_df["player2_team_id"] = steals_df["player2_team_id"].astype(int)
        steals_df.rename(
            columns={"player2_team_id": "team_id", "is_steal": "stl",}, inplace=True,
        )

        return steals_df

    def _block_calc_team(self):
        """
        method to calculate team blocks
        """
        blocks_df = (
            self.df.groupby(["player3_team_id", "game_id"])[["is_block"]]
            .sum()
            .reset_index()
        )
        blocks_df["player3_team_id"] = blocks_df["player3_team_id"].astype(int)
        blocks_df.rename(
            columns={"player3_team_id": "team_id", "is_block": "blk",}, inplace=True,
        )

        blocks_df = blocks_df.merge(blocks_df, on="game_id", suffixes=["", "_opponent"])

        blocks_df = blocks_df[blocks_df["team_id"] != blocks_df["team_id_opponent"]]
        blocks_df.rename(
            columns={"blk_opponent": "shots_blocked",}, inplace=True,
        )

        return blocks_df[["team_id", "game_id", "blk", "shots_blocked"]]

    def _plus_minus_team(self):
        """
        method to calculate team score differential
        """
        plus_minus_df = (
            self.df.groupby(["player1_team_id", "game_id"])[["points_made",]]
            .sum()
            .reset_index()
        )
        plus_minus_df["player1_team_id"] = plus_minus_df["player1_team_id"].astype(int)
        plus_minus_df.rename(
            columns={"player1_team_id": "team_id", "points_made": "points_for",},
            inplace=True,
        )
        plus_minus_df = plus_minus_df.merge(
            plus_minus_df, on="game_id", suffixes=["", "_opponent"]
        )

        plus_minus_df = plus_minus_df[
            plus_minus_df["team_id"] != plus_minus_df["team_id_opponent"]
        ]

        plus_minus_df["plus_minus"] = (
            plus_minus_df["points_for"] - plus_minus_df["points_for_opponent"]
        )
        plus_minus_df.rename(
            columns={"points_for_opponent": "points_against",}, inplace=True,
        )

        return plus_minus_df[["team_id", "game_id", "points_against", "plus_minus"]]

    @staticmethod
    def _get_off_def_teams(last_event: pd.Series) -> tuple[str, str]:
        """
        Determine offensive and defensive team abbreviations for a possession
        based on the last event in that possession.

        Returns (off_abbrev, def_abbrev), where each is a team abbreviation
        (home_team_abbrev or away_team_abbrev). In ambiguous cases, falls back
        to treating the home team as the offense.
        """
        home_abbrev = last_event.get("home_team_abbrev")
        away_abbrev = last_event.get("away_team_abbrev")
        ev_team = last_event.get("event_team")

        # Prefer annotated 'family' if available, otherwise use raw event_type_de.
        ev_type = last_event.get("family") or last_event.get("event_type_de")
        if isinstance(ev_type, str):
            ev_type = ev_type.replace("-", "_").lower()
        else:
            ev_type = ""

        # Default offense/defense mapping mirrors existing _build_possessions logic:
        # - shot / free_throw / turnover: offense is event_team
        # - rebound: depends on OREB/DREB flags
        # - fallback: treat event_team as offense
        if ev_type in ("shot", "miss_shot", "missed_shot", "free_throw", "turnover"):
            off_abbrev = ev_team
        elif ev_type == "rebound":
            # Check rebound type flags (Critical Fix)
            is_d_rebound = last_event.get("is_d_rebound") == 1
            is_o_rebound = last_event.get("is_o_rebound") == 1

            if is_o_rebound:
                # Offensive rebound: the rebounding team is offense.
                off_abbrev = ev_team
            elif is_d_rebound:
                # Defensive rebound: the rebounding team is defense, the other team was offense.
                if ev_team == home_abbrev:
                    off_abbrev = away_abbrev
                elif ev_team == away_abbrev:
                    off_abbrev = home_abbrev
                else:
                    # If we can't match, fall back to event team
                    off_abbrev = ev_team
            else:
                # Ambiguous/Team rebound: Treat similar to DREB for determining who *had* the ball.
                if ev_team == home_abbrev:
                    off_abbrev = away_abbrev
                elif ev_team == away_abbrev:
                    off_abbrev = home_abbrev
                else:
                    off_abbrev = ev_team
        else:
            off_abbrev = ev_team

        # Derive defense as the other team, with fallbacks.
        if off_abbrev == home_abbrev:
            def_abbrev = away_abbrev
        elif off_abbrev == away_abbrev:
            def_abbrev = home_abbrev
        else:
            # Fallback: if off_abbrev doesn't match either home/away, try event team.
            if ev_team == home_abbrev:
                off_abbrev = home_abbrev
                def_abbrev = away_abbrev
            elif ev_team == away_abbrev:
                off_abbrev = away_abbrev
                def_abbrev = home_abbrev
            else:
                # Final fallback: default to home offense
                off_abbrev = home_abbrev
                def_abbrev = away_abbrev

        return off_abbrev, def_abbrev

    @staticmethod
    def parse_possessions(poss_list: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[int]]:
        """
        Parse each possession segment and create one row for offense team
        and defense team lineups.

        Inputs:
            poss_list   - list of dataframes each one representing one possession

        Outputs:
            parsed_list  - list of single-row dataframes, one per possession
            used_indices - list of integer indices into poss_list corresponding to parsed_list
        """
        parsed_list: list[pd.DataFrame] = []
        used_indices: list[int] = []

        # Define metadata columns to carry through
        metadata_cols = [
            "points_made",
            "home_team_abbrev",
            "event_team",
            "away_team_abbrev",
            "home_team_id",
            "away_team_id",
            "game_id",
            "game_date",
            "season",
        ]

        # Explicit list of player columns so we do not rely on column order
        player_cols = [
            "home_player_1", "home_player_1_id",
            "home_player_2", "home_player_2_id",
            "home_player_3", "home_player_3_id",
            "home_player_4", "home_player_4_id",
            "home_player_5", "home_player_5_id",
            "away_player_1", "away_player_1_id",
            "away_player_2", "away_player_2_id",
            "away_player_3", "away_player_3_id",
            "away_player_4", "away_player_4_id",
            "away_player_5", "away_player_5_id",
        ]

        # Dynamically generate column mappings for renaming
        home_to_off = {c: c.replace("home_", "off_") for c in player_cols if "home_" in c}
        away_to_def = {c: c.replace("away_", "def_") for c in player_cols if "away_" in c}
        home_to_def = {c: c.replace("home_", "def_") for c in player_cols if "home_" in c}
        away_to_off = {c: c.replace("away_", "off_") for c in player_cols if "away_" in c}

        # Combine player and metadata columns for selection
        selected_cols = player_cols + metadata_cols

        # Canonical possession-ending families (after normalization)
        valid_end_families = {
            "rebound",
            "turnover",
            "shot",
            "missed_shot",
            "miss_shot",
            "free_throw",
        }

        for seg_idx, df in enumerate(poss_list):
            if df.empty:
                continue

            # Normalize event "family" for robust CDN/v2 handling.
            if "family" in df.columns:
                fam = df["family"].fillna("").astype(str).str.lower()
            else:
                et = df.get("event_type_de")
                if et is None:
                    fam = pd.Series([""] * len(df), index=df.index)
                else:
                    fam = (
                        et.fillna("")
                        .astype(str)
                        .str.lower()
                        .str.replace("-", "_", regex=False)
                    )

            last_fam = fam.iloc[-1]
            if last_fam not in valid_end_families:
                end_idx = fam[fam.isin(valid_end_families)].index
                if len(end_idx) == 0:
                    # No obvious possession-ending event in this segment; skip it for
                    # possession-level RAPM/on‑court scoring while leaving the events
                    # available for timing-based exposures.
                    continue
                df = df.loc[: end_idx[-1]].copy()
                fam = fam.loc[df.index]

            # Fail fast with a clear error if the input doesn't have the
            # expected player columns (e.g., incompatible nba_scraper version).
            missing = [c for c in player_cols if c not in df.columns]
            if missing:
                raise KeyError(
                    "parse_possessions expected player columns "
                    f"{player_cols}, but these are missing: {missing}"
                )

            last_event = df.iloc[-1]

            # Determine offense/defense using centralized helper
            off_abbrev, _ = PbP._get_off_def_teams(last_event)

            def append_possession(off_abbrev_inner: str) -> None:
                home_abbrev = last_event.get("home_team_abbrev")
                away_abbrev = last_event.get("away_team_abbrev")

                if off_abbrev_inner not in (home_abbrev, away_abbrev):
                    # Fallback: assume home is on offense to preserve row count
                    off_abbrev_inner = home_abbrev

                # Create a DataFrame from the last event's relevant columns
                # Handle potential missing columns gracefully
                data = {col: last_event.get(col) for col in selected_cols}
                row_df = pd.DataFrame([data])

                # Rename 'event_team' to 'event_team_abbrev' for consistency if needed
                if "event_team" in row_df.columns and "event_team_abbrev" not in row_df.columns:
                    row_df = row_df.rename(columns={"event_team": "event_team_abbrev"})

                if off_abbrev_inner == home_abbrev:
                    # Home team on offense
                    row_df = row_df.rename(columns={**home_to_off, **away_to_def})
                else:
                    # Away team on offense
                    row_df = row_df.rename(columns={**away_to_off, **home_to_def})

                parsed_list.append(row_df)

            append_possession(off_abbrev)

            # Track which segment produced this possession row
            used_indices.append(seg_idx)

        return parsed_list, used_indices

    def _build_possessions(self, df: pd.DataFrame, include_event_agg: bool = False):
        """
        Internal helper used by rapm_possessions() and compute_on_court_exposures().

        Returns a DataFrame with one row per possession. When include_event_agg is
        True, possession-level shooting aggregates for the offense and defense are
        added.
        """

        # Annotate events once for the whole game
        pbp_df = annotate_events(df.copy())

        # Ensure 0..N index so that label-based indexing from poss_index matches positional iloc
        pbp_df = pbp_df.reset_index(drop=True)

        poss_index = pbp_df[(pbp_df.home_possession == 1) | (pbp_df.away_possession == 1)].index
        shift_dfs = []
        past_index = -1

        for i in poss_index:
            # Slice events between possession markers, skipping empty segments
            seg = pbp_df.iloc[past_index + 1 : i + 1, :].reset_index(drop=True)
            if not seg.empty:
                shift_dfs.append(seg)
            past_index = i

        # Capture any remaining events after the last possession marker
        if past_index < len(pbp_df) - 1:
            seg = pbp_df.iloc[past_index + 1 :].reset_index(drop=True)
            if not seg.empty:
                shift_dfs.append(seg)

        # --- Normalize segments so parse_possessions and event_aggs see the same slice ---
        # Use the canonical 'family' column from annotate_events so we don't depend
        # on v2-style event_type_de strings.
        valid_end_families = {
            "rebound",
            "turnover",
            "shot",
            "missed_shot",
            "miss_shot",
            "free_throw",
        }

        normalized_segments: list[pd.DataFrame] = []
        for seg in shift_dfs:
            if seg.empty:
                normalized_segments.append(seg)
                continue

            # Prefer the canonical family column when available; otherwise fall back
            # to a normalized event_type_de.
            if "family" in seg.columns:
                fam = seg["family"].fillna("").astype(str).str.lower()
            else:
                et = seg.get("event_type_de")
                if et is None:
                    fam = pd.Series([""] * len(seg), index=seg.index)
                else:
                    fam = (
                        et.fillna("")
                        .astype(str)
                        .str.lower()
                        .str.replace("-", "_", regex=False)
                    )

            last_fam = fam.iloc[-1]
            if last_fam in valid_end_families:
                normalized_segments.append(seg)
                continue

            mask = fam.isin(valid_end_families)
            if not mask.any():
                # No obvious possession-ending event in this stretch:
                # treat as non-possession for RAPM/on‑court scoring.
                normalized_segments.append(seg.iloc[0:0])
                continue

            last_idx = mask[mask].index[-1]
            normalized_segments.append(seg.loc[: last_idx].copy())

        # Use normalized segments everywhere downstream
        parsed_possessions, used_indices = self.parse_possessions(normalized_segments)

        # If no possessions were detected, return an empty DataFrame
        if not parsed_possessions:
            return pd.DataFrame()

        event_aggs = []
        for poss_df in normalized_segments:
            poss_events = poss_df  # already annotated via pbp_df
            if poss_events.empty:
                event_aggs.append(
                    {
                        "off_team_id": np.nan,
                        "off_team_abbrev": "",
                        "def_team_id": np.nan,
                        "def_team_abbrev": "",
                        "off_team_FGA": 0,
                        "off_team_FGM": 0,
                        "off_team_3PA": 0,
                        "off_team_3PM": 0,
                        "off_team_2PA": 0,
                        "off_team_FTA": 0,
                        "off_team_FTM": 0,
                        "def_team_FGA": 0,
                        "def_team_FGM": 0,
                        "def_team_3PA": 0,
                        "def_team_3PM": 0,
                        "def_team_2PA": 0,
                        "def_team_FTA": 0,
                        "def_team_FTM": 0,
                        "points_for_offense": 0,
                        "points_for_defense": 0,
                    }
                )
                continue

            last_event = poss_events.iloc[-1]
            
            home_abbrev = last_event.get("home_team_abbrev")
            away_abbrev = last_event.get("away_team_abbrev")

            # Centralized determination
            off_abbrev, def_abbrev = self._get_off_def_teams(last_event)

            off_team_id = (
                last_event.get("home_team_id")
                if off_abbrev == home_abbrev
                else last_event.get("away_team_id")
            )
            def_team_id = (
                last_event.get("away_team_id")
                if off_abbrev == home_abbrev
                else last_event.get("home_team_id")
            )

            off_mask = poss_events.get("team_id") == off_team_id
            def_mask = poss_events.get("team_id") == def_team_id

            off_fga = poss_events.loc[off_mask, "is_fg_attempt"].sum()
            off_fgm = poss_events.loc[off_mask, "is_fg_make"].sum()
            off_3pa_mask = (
                poss_events.loc[off_mask, "is_fg_attempt"].astype(bool)
                & poss_events.loc[off_mask, "is_three"].astype(bool)
            )
            off_3pm_mask = (
                poss_events.loc[off_mask, "is_fg_make"].astype(bool)
                & poss_events.loc[off_mask, "is_three"].astype(bool)
            )
            off_2pa_mask = (
                poss_events.loc[off_mask, "is_fg_attempt"].astype(bool)
                & ~poss_events.loc[off_mask, "is_three"].astype(bool)
            )

            def_fga = poss_events.loc[def_mask, "is_fg_attempt"].sum()
            def_fgm = poss_events.loc[def_mask, "is_fg_make"].sum()
            def_3pa_mask = (
                poss_events.loc[def_mask, "is_fg_attempt"].astype(bool)
                & poss_events.loc[def_mask, "is_three"].astype(bool)
            )
            def_3pm_mask = (
                poss_events.loc[def_mask, "is_fg_make"].astype(bool)
                & poss_events.loc[def_mask, "is_three"].astype(bool)
            )
            def_2pa_mask = (
                poss_events.loc[def_mask, "is_fg_attempt"].astype(bool)
                & ~poss_events.loc[def_mask, "is_three"].astype(bool)
            )

            off_points = poss_events.loc[off_mask, "points_made"].sum()
            def_points = poss_events.loc[def_mask, "points_made"].sum()

            event_aggs.append(
                {
                    "off_team_id": off_team_id,
                    "off_team_abbrev": off_abbrev,
                    "def_team_id": def_team_id,
                    "def_team_abbrev": def_abbrev,
                    "off_team_FGA": off_fga,
                    "off_team_FGM": off_fgm,
                    "off_team_3PA": off_3pa_mask.sum(),
                    "off_team_3PM": off_3pm_mask.sum(),
                    "off_team_2PA": off_2pa_mask.sum(),
                    "off_team_FTA": poss_events.loc[off_mask, "is_ft"].sum(),
                    "off_team_FTM": poss_events.loc[off_mask, "is_ft_make"].sum(),
                    "def_team_FGA": def_fga,
                    "def_team_FGM": def_fgm,
                    "def_team_3PA": def_3pa_mask.sum(),
                    "def_team_3PM": def_3pm_mask.sum(),
                    "def_team_2PA": def_2pa_mask.sum(),
                    "def_team_FTA": poss_events.loc[def_mask, "is_ft"].sum(),
                    "def_team_FTM": poss_events.loc[def_mask, "is_ft_make"].sum(),
                    "points_for_offense": off_points,
                    "points_for_defense": def_points,
                }
            )

        poss_df = pd.concat(parsed_possessions, sort=True).reset_index(drop=True)

        if event_aggs:
            agg_df = pd.DataFrame(event_aggs)

            # Robustly align aggregates to the parsed possessions using the returned indices
            agg_df = agg_df.iloc[used_indices].reset_index(drop=True)

            # Attach team IDs and abbreviations inferred in the event_aggs loop
            poss_df["off_team_id"] = agg_df["off_team_id"].values
            poss_df["def_team_id"] = agg_df["def_team_id"].values
            poss_df["off_team_abbrev"] = agg_df["off_team_abbrev"].values
            poss_df["def_team_abbrev"] = agg_df["def_team_abbrev"].values

            poss_df["points_for_offense"] = agg_df["points_for_offense"].values
            poss_df["points_for_defense"] = agg_df["points_for_defense"].values

            if include_event_agg:
                for col in [
                    "off_team_FGA",
                    "off_team_FGM",
                    "off_team_3PA",
                    "off_team_3PM",
                    "off_team_2PA",
                    "off_team_FTA",
                    "off_team_FTM",
                    "def_team_FGA",
                    "def_team_FGM",
                    "def_team_3PA",
                    "def_team_3PM",
                    "def_team_2PA",
                    "def_team_FTA",
                    "def_team_FTM",
                ]:
                    poss_df[col] = agg_df[col].values
        else:
            # Fallback when no events were parsed
            poss_df["off_team_id"] = np.nan
            poss_df["def_team_id"] = np.nan
            poss_df["off_team_abbrev"] = ""
            poss_df["def_team_abbrev"] = ""
            poss_df["points_for_offense"] = 0
            poss_df["points_for_defense"] = 0

        # Backwards-compatibility: expose scoring in a single points column too.
        poss_df["points_made"] = poss_df["points_for_offense"]

        return poss_df

    def rapm_possessions(self):
        """
        Extract out all the RAPM possessions as a DataFrame.

        This uses the same event-level scoring logic as the on-court glossary:
        - points_for_offense: total offensive points scored in the possession.
        - points_for_defense: total points scored by the opponent in the possession.

        For backward compatibility, a 'points_made' column is also provided and
        is set equal to points_for_offense by _build_possessions.
        """
        pbp_df = self.df.copy()
        poss_df = self._build_possessions(pbp_df, include_event_agg=True)
        return poss_df

    def _compute_starters(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per (game_id, team_id, player_id)
        marking whether the player started the game (Starts = 1).

        Assumes this PbP instance is a single game.
        """
        df = self.df.copy()

        # Sort by game clock so we consistently pick the earliest event
        sort_cols = [c for c in ["period", "seconds_elapsed", "eventnum"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)

        lineup_cols = [
            *(f"home_player_{i}_id" for i in range(1, 6)),
            *(f"away_player_{i}_id" for i in range(1, 6)),
        ]

        mask = pd.Series(True, index=df.index)
        for col in lineup_cols:
            if col in df.columns:
                mask &= df[col].notna() & (df[col] != 0)

        lineup_df = df[mask] if not df.empty else df
        first_row = lineup_df.iloc[0] if not lineup_df.empty else df.iloc[0]

        home_ids = [first_row.get(f"home_player_{i}_id") for i in range(1, 6)]
        away_ids = [first_row.get(f"away_player_{i}_id") for i in range(1, 6)]

        starters = []
        game_id = _coerce_id_scalar(first_row.get("game_id"))
        home_team_id = _coerce_id_scalar(first_row.get("home_team_id"))
        away_team_id = _coerce_id_scalar(first_row.get("away_team_id"))

        for pid in home_ids:
            if pid and pid != 0 and not pd.isna(pid):
                starters.append(
                    {
                        "game_id": game_id,
                        "team_id": home_team_id,
                        "player_id": _coerce_id_scalar(pid),
                        "Starts": 1,
                    }
                )
        for pid in away_ids:
            if pid and pid != 0 and not pd.isna(pid):
                starters.append(
                    {
                        "game_id": game_id,
                        "team_id": away_team_id,
                        "player_id": _coerce_id_scalar(pid),
                        "Starts": 1,
                    }
                )

        starters_df = pd.DataFrame(starters)
        if not starters_df.empty:
            for col in ["game_id", "team_id", "player_id"]:
                if col in starters_df.columns:
                    starters_df[col] = (
                        pd.to_numeric(starters_df[col], errors="coerce")
                        .fillna(0)
                        .astype(int)
                    )

        return starters_df

    def player_box_glossary(
        self,
        player_meta: pd.DataFrame | None = None,
        game_meta: pd.DataFrame | None = None,
        player_game_meta: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build a per-player, per-game box aligned to an external glossary.

        Assumes this PbP instance represents a single game.

        - Inputs:
            player_meta: optional DataFrame with identity/biographical fields
                keyed by NBA.com personId (player_id / NbaDotComID).
            game_meta: optional DataFrame with game-level fields keyed by game_id.

        - Output:
            DataFrame with one row per (game_id, team_id, player_id),
            including raw counts and derived rates (TS, USG, ORB%, DRB%, AST%, BLK%, etc.)
            plus on/off stats.
        """
        df = self.df.copy()

        df = annotate_events(df)
        counts_df = accumulate_player_counts(df)
        exposures_df = compute_on_court_exposures(self, df)

        # Defensively filter player_meta to the current season to avoid duplicate rows
        pm = player_meta
        if pm is not None and not pm.empty:
            pm = pm.copy()
            # If player_meta includes a 'season' or 'Year' column, trim to this game's season
            if "season" in pm.columns:
                pm = pm[pm["season"] == self.season]
            elif "Year" in pm.columns:
                pm = pm[pm["Year"] == self.season]

            # Ensure unique row per player identifier
            if "NbaDotComID" in pm.columns:
                pm = pm.drop_duplicates(subset=["NbaDotComID"])
            elif "player_id" in pm.columns:
                pm = pm.drop_duplicates(subset=["player_id"])

        box_df = build_player_box(
            df=df,
            counts_df=counts_df,
            exposures_df=exposures_df,
            player_meta=pm,
            game_meta=game_meta,
            player_game_meta=player_game_meta,
        )

        # Normalize potential merge artifacts from player_meta
        if "player_id_x" in box_df.columns:
            box_df["player_id"] = box_df["player_id_x"]
            box_df.drop(columns=[c for c in ["player_id_x", "player_id_y"] if c in box_df.columns], inplace=True)

        # Fill Starts from starting lineups
        starters_df = self._compute_starters()
        if not starters_df.empty:
            # Preserve original Starts column if it exists, then merge
            if "Starts" in box_df.columns:
                box_df.rename(columns={"Starts": "Starts_x"}, inplace=True)
            box_df = box_df.merge(
                starters_df, on=["game_id", "team_id", "player_id"], how="left"
            )
            # Coalesce merged 'Starts' with original, fill NaNs with 0
            box_df["Starts"] = box_df["Starts"].fillna(box_df.get("Starts_x", 0)).fillna(0).astype(int)
            box_df.drop(columns=[c for c in box_df.columns if c.endswith("_x") or c == 'Starts_y'], inplace=True)

        # Sanity check: on-court points must match team totals.
        self._check_on_court_points_consistency(box_df)
        self._check_on_court_minutes_consistency(box_df)

        return box_df

    def player_box_glossary_with_totals(
        self,
        player_meta: pd.DataFrame | None = None,
        game_meta: pd.DataFrame | None = None,
        player_game_meta: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Convenience wrapper: build the per-player glossary box and append one
        'TOTAL' row per (game_id, team_id).
        """
        box = self.player_box_glossary(
            player_meta=player_meta,
            game_meta=game_meta,
            player_game_meta=player_game_meta,
        )
        return append_team_totals(box)

    def _check_on_court_points_consistency(self, box: pd.DataFrame, tol: float = 1e-6) -> None:
        """
        Internal helper: verify that summed OnCourt_Team_Points / OnCourt_Opp_Points
        per team match the scoreboard totals times 5 for this game.

        Raises AssertionError if an invariant is violated.
        """
        team_points = self._point_calc_team()[["team_id", "points_for"]]

        # Build a simple mapping team_id -> points_for for this game.
        team_points_map = dict(zip(team_points["team_id"], team_points["points_for"]))

        for team_id, points_for in team_points_map.items():
            expected_for = points_for * 5.0
            actual_for = box.loc[box["team_id"] == team_id, "OnCourt_Team_Points"].sum()

            if abs(actual_for - expected_for) > tol:
                raise AssertionError(
                    f"OnCourt_Team_Points inconsistency for team {team_id}: "
                    f"actual={actual_for}, expected={expected_for}"
                )

            # Opponent points are the sum of all other teams' points_for.
            opponent_points = sum(
                p for t, p in team_points_map.items() if t != team_id
            )
            expected_against = opponent_points * 5.0
            actual_against = box.loc[box["team_id"] == team_id, "OnCourt_Opp_Points"].sum()

            if abs(actual_against - expected_against) > tol:
                raise AssertionError(
                    f"OnCourt_Opp_Points inconsistency for team {team_id}: "
                    f"actual={actual_against}, expected={expected_against}"
                )

    def _check_on_court_minutes_consistency(self, box: pd.DataFrame, tol: float = 1e-6) -> None:
        """
        Internal helper: verify that summed Minutes per team equal total game
        duration multiplied by 5.

        Raises AssertionError if an invariant is violated.
        """

        if "event_length" not in self.df.columns or self.df.empty:
            return

        durations = self.df.copy()
        durations["event_length"] = durations["event_length"].fillna(0).astype(float)
        game_minutes = durations.groupby("game_id")["event_length"].sum() / 60.0

        for game_id, total_minutes in game_minutes.items():
            expected_minutes = total_minutes * 5.0
            team_minutes = box.loc[box["game_id"] == game_id].groupby("team_id")["Minutes"].sum()

            for team_id, actual_minutes in team_minutes.items():
                if abs(actual_minutes - expected_minutes) > tol:
                    raise AssertionError(
                        f"OnCourt Minutes inconsistency for team {team_id} in game {game_id}: "
                        f"actual={actual_minutes}, expected={expected_minutes}"
                    )

    def playerbygamestats(self):
        """
        LEGACY: v2-based player stat calculation. Kept for backwards compatibility
        and tests. New code should prefer player_box_glossary() plus
        accumulate_player_counts()/compute_on_court_exposures().

        this function combines all playerbygamestats and returns a dataframe
        containing them
        """
        points = self._point_calc_player()
        blocks = self._block_calc_player()
        assists = self._assist_calc_player()
        rebounds = self._rebound_calc_player()
        turnovers = self._turnover_calc_player()
        fouls = self._foul_calc_player()
        steals = self._steal_calc_player()
        plus_minus = self._plus_minus_calc_player()
        toc = self._toc_calc_player()
        poss = self._poss_calc_player()

        pbg = toc.merge(
            points, how="left", on=["player_id", "team_id", "game_date", "game_id"]
        )
        pbg = pbg.merge(
            blocks, how="left", on=["player_id", "team_id", "game_date", "game_id"]
        )
        pbg = pbg.merge(
            assists, how="left", on=["player_id", "team_id", "game_date", "game_id"]
        )
        pbg = pbg.merge(rebounds, how="left", on=["player_id", "game_date", "game_id"])
        pbg = pbg.merge(
            turnovers, how="left", on=["player_id", "team_id", "game_date", "game_id"]
        )
        pbg = pbg.merge(
            fouls, how="left", on=["player_id", "team_id", "game_date", "game_id"]
        )
        pbg = pbg.merge(
            steals, how="left", on=["player_id", "team_id", "game_date", "game_id"]
        )
        pbg = pbg.merge(
            plus_minus, how="left", on=["player_id", "team_id", "game_date", "game_id"]
        )
        pbg = pbg.merge(poss, how="left", on=["player_id", "team_id", "game_id"])

        pbg["blk"] = pbg["blk"].fillna(0).astype(int)
        pbg["ast"] = pbg["ast"].fillna(0).astype(int)
        pbg["dreb"] = pbg["dreb"].fillna(0).astype(int)
        pbg["oreb"] = pbg["oreb"].fillna(0).astype(int)
        pbg["tov"] = pbg["tov"].fillna(0).astype(int)
        pbg["pf"] = pbg["pf"].fillna(0).astype(int)
        pbg["stl"] = pbg["stl"].fillna(0).astype(int)
        pbg["fgm"] = pbg["fgm"].fillna(0).astype(int)
        pbg["fga"] = pbg["fga"].fillna(0).astype(int)
        pbg["tpm"] = pbg["tpm"].fillna(0).astype(int)
        pbg["tpa"] = pbg["tpa"].fillna(0).astype(int)
        pbg["ftm"] = pbg["ftm"].fillna(0).astype(int)
        pbg["fta"] = pbg["fta"].fillna(0).astype(int)
        pbg["points"] = pbg["points"].fillna(0).astype(int)
        pbg["is_home"] = np.where(pbg["team_id"] == self.home_team_id, 1, 0)
        pbg["team_abbrev"] = np.where(
            self.home_team_id == pbg["team_id"], self.home_team, self.away_team
        )
        pbg["opponent"] = np.where(
            pbg["team_id"] == self.home_team_id, self.away_team_id, self.home_team_id
        )
        pbg["opponent_abbrev"] = np.where(
            pbg["team_id"] == self.home_team_id, self.away_team, self.home_team
        )
        pbg["season"] = self.season
        pbg["player_id"] = pbg["player_id"].astype(int)
        pbg = pbg[pbg["toc"] > 0]

        return pbg

    def teambygamestats(self):
        """
        main team stats calc hook
        """

        points = self._point_calc_team()
        blocks = self._block_calc_team()
        assists = self._assist_calc_team()
        rebounds = self._rebound_calc_team()
        turnovers = self._turnover_calc_team()
        fouls = self._foul_calc_team()
        steals = self._steal_calc_team()
        plus_minus = self._plus_minus_team()
        poss = self._poss_calc_team()

        tbg = points.merge(blocks, how="left", on=["team_id", "game_id"])

        tbg = tbg.merge(assists, how="left", on=["team_id", "game_id"])
        tbg = tbg.merge(rebounds, how="left", on=["team_id", "game_id"])
        tbg = tbg.merge(turnovers, how="left", on=["team_id", "game_id"])
        tbg = tbg.merge(fouls, how="left", on=["team_id", "game_id"])
        tbg = tbg.merge(steals, how="left", on=["team_id", "game_id"])
        tbg = tbg.merge(plus_minus, how="left", on=["team_id", "game_id"])
        tbg = tbg.merge(poss, how="left", on=["team_id", "game_id"])
        tbg["game_date"] = self.df["game_date"].unique()[0]
        tbg["season"] = self.df["season"].unique()[0]
        tbg["toc"] = self.df["seconds_elapsed"].max()
        tbg[
            "toc_string"
        ] = f"{math.floor(self.df['seconds_elapsed'].max()/60)}:{self.df['seconds_elapsed'].max()%60}0"
        tbg["is_home"] = np.where(
            tbg["team_id"] == self.df["home_team_id"].unique()[0], 1, 0
        )
        tbg["is_win"] = np.where(tbg["points_for"] > tbg["points_against"], 1, 0)

        tbg["blk"] = tbg["blk"].fillna(0).astype(int)
        tbg["ast"] = tbg["ast"].fillna(0).astype(int)
        tbg["dreb"] = tbg["dreb"].fillna(0).astype(int)
        tbg["oreb"] = tbg["oreb"].fillna(0).astype(int)
        tbg["tov"] = tbg["tov"].fillna(0).astype(int)
        tbg["pf"] = tbg["pf"].fillna(0).astype(int)
        tbg["stl"] = tbg["stl"].fillna(0).astype(int)
        tbg["fgm"] = tbg["fgm"].fillna(0).astype(int)
        tbg["fga"] = tbg["fga"].fillna(0).astype(int)
        tbg["tpm"] = tbg["tpm"].fillna(0).astype(int)
        tbg["tpa"] = tbg["tpa"].fillna(0).astype(int)
        tbg["ftm"] = tbg["ftm"].fillna(0).astype(int)
        tbg["fta"] = tbg["fta"].fillna(0).astype(int)
        tbg["opponent"] = np.where(
            tbg["team_id"] == self.home_team_id, self.away_team_id, self.home_team_id
        )
        tbg["opponent_abbrev"] = np.where(
            tbg["team_id"] == self.home_team_id, self.away_team, self.home_team
        )

        return tbg
