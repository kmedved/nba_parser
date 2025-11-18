from datetime import datetime
import math
import numpy as np
import pandas as pd
from .box_glossary import (
    annotate_events,
    accumulate_player_counts,
    compute_on_court_exposures,
    build_player_box,
)

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
        self.home_team = pbp_df["home_team_abbrev"].unique()[0]
        self.away_team = pbp_df["away_team_abbrev"].unique()[0]
        self.home_team_id = pbp_df["home_team_id"].unique()[0]
        self.away_team_id = pbp_df["away_team_id"].unique()[0]
        self.season = pbp_df["season"].unique()[0]

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
        # calculating final free throw possessions
        self.df["home_possession"] = np.where(
            (self.df.event_team == self.df.home_team_abbrev)
            & (
                (self.df.homedescription.str.contains("Free Throw 2 of 2"))
                | (self.df.homedescription.str.contains("Free Throw 3 of 3"))
            ),
            1,
            self.df["home_possession"],
        )
        # calculating made shot possessions
        self.df["away_possession"] = np.where(
            (self.df.event_team == self.df.away_team_abbrev)
            & (self.df.event_type_de == "shot"),
            1,
            0,
        )
        # calculating turnover possessions
        self.df["away_possession"] = np.where(
            (self.df.event_team == self.df.away_team_abbrev)
            & (self.df.event_type_de == "turnover"),
            1,
            self.df["away_possession"],
        )
        # calculating defensive rebound possessions
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
        # calculating final free throw possessions
        self.df["away_possession"] = np.where(
            (self.df.event_team == self.df.away_team_abbrev)
            & (
                (self.df.visitordescription.str.contains("Free Throw 2 of 2"))
                | (self.df.visitordescription.str.contains("Free Throw 3 of 3"))
            ),
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
        teams_df = (
            self.df.groupby(["player1_team_id", "game_id"])[
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
    def parse_possessions(poss_list):
        """
        a function to parse each possession and create one row for offense team
        and defense team

        Inputs:
        poss_list   - list of dataframes each one representing one possession

        Outputs:
        parsed_list  - list of dataframes where each list inside represents the players on
                       off and def and points score for each possession
        """
        parsed_list = []

        # Explicit list of player columns so we do not rely on column order
        # in the source DataFrame. This matches the expected order used when
        # constructing off_player_* and def_player_* fields below.
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

        for df in poss_list:
            if df.empty:
                continue

            # Fail fast with a clear error if the input doesn't have the
            # expected player columns (e.g., incompatible nba_scraper version).
            missing = [c for c in player_cols if c not in df.columns]
            if missing:
                raise KeyError(
                    "parse_possessions expected player columns "
                    f"{player_cols}, but these are missing: {missing}"
                )

            if df.loc[df.index[-1], "event_type_de"] in ["rebound", "turnover"]:
                if df.loc[df.index[-1], "event_type_de"] == "turnover":
                    if (
                        df.loc[df.index[-1], "event_team"]
                        == df.loc[df.index[-1], "home_team_abbrev"]
                    ):
                        row_df = pd.concat(
                            [
                                df.loc[df.index[-1], player_cols],
                                df.loc[
                                    df.index[-1],
                                    [
                                        "points_made",
                                        "home_team_abbrev",
                                        "event_team",
                                        "away_team_abbrev",
                                        "home_team_id",
                                        "away_team_id",
                                        "game_id",
                                        "game_date",
                                        "season",
                                    ],
                                ],
                            ]
                        )

                        parsed_list.extend(
                            [
                                pd.DataFrame(
                                    [list(row_df)],
                                    columns=[
                                        "off_player_1",
                                        "off_player_1_id",
                                        "off_player_2",
                                        "off_player_2_id",
                                        "off_player_3",
                                        "off_player_3_id",
                                        "off_player_4",
                                        "off_player_4_id",
                                        "off_player_5",
                                        "off_player_5_id",
                                        "def_player_1",
                                        "def_player_1_id",
                                        "def_player_2",
                                        "def_player_2_id",
                                        "def_player_3",
                                        "def_player_3_id",
                                        "def_player_4",
                                        "def_player_4_id",
                                        "def_player_5",
                                        "def_player_5_id",
                                        "points_made",
                                        "home_team_abbrev",
                                        "event_team_abbrev",
                                        "away_team_abbrev",
                                        "home_team_id",
                                        "away_team_id",
                                        "game_id",
                                        "game_date",
                                        "season",
                                    ],
                                )
                            ]
                        )
                    elif (
                        df.loc[df.index[-1], "event_team"]
                        == df.loc[df.index[-1], "away_team_abbrev"]
                    ):
                        row_df = pd.concat(
                            [
                                df.loc[df.index[-1], player_cols],
                                df.loc[
                                    df.index[-1],
                                    [
                                        "points_made",
                                        "home_team_abbrev",
                                        "event_team",
                                        "away_team_abbrev",
                                        "home_team_id",
                                        "away_team_id",
                                        "game_id",
                                        "game_date",
                                        "season",
                                    ],
                                ],
                            ]
                        )

                        parsed_list.extend(
                            [
                                pd.DataFrame(
                                    [list(row_df)],
                                    columns=[
                                        "def_player_1",
                                        "def_player_1_id",
                                        "def_player_2",
                                        "def_player_2_id",
                                        "def_player_3",
                                        "def_player_3_id",
                                        "def_player_4",
                                        "def_player_4_id",
                                        "def_player_5",
                                        "def_player_5_id",
                                        "off_player_1",
                                        "off_player_1_id",
                                        "off_player_2",
                                        "off_player_2_id",
                                        "off_player_3",
                                        "off_player_3_id",
                                        "off_player_4",
                                        "off_player_4_id",
                                        "off_player_5",
                                        "off_player_5_id",
                                        "points_made",
                                        "home_team_abbrev",
                                        "event_team_abbrev",
                                        "away_team_abbrev",
                                        "home_team_id",
                                        "away_team_id",
                                        "game_id",
                                        "game_date",
                                        "season",
                                    ],
                                )
                            ]
                        )
                if df.loc[df.index[-1], "event_type_de"] == "rebound":
                    if (
                        df.loc[df.index[-1], "event_team"]
                        == df.loc[df.index[-1], "away_team_abbrev"]
                    ):
                        row_df = pd.concat(
                            [
                                df.loc[df.index[-1], player_cols],
                                df.loc[
                                    df.index[-1],
                                    [
                                        "points_made",
                                        "home_team_abbrev",
                                        "event_team",
                                        "away_team_abbrev",
                                        "home_team_id",
                                        "away_team_id",
                                        "game_id",
                                        "game_date",
                                        "season",
                                    ],
                                ],
                            ]
                        )

                        parsed_list.extend(
                            [
                                pd.DataFrame(
                                    [list(row_df)],
                                    columns=[
                                        "off_player_1",
                                        "off_player_1_id",
                                        "off_player_2",
                                        "off_player_2_id",
                                        "off_player_3",
                                        "off_player_3_id",
                                        "off_player_4",
                                        "off_player_4_id",
                                        "off_player_5",
                                        "off_player_5_id",
                                        "def_player_1",
                                        "def_player_1_id",
                                        "def_player_2",
                                        "def_player_2_id",
                                        "def_player_3",
                                        "def_player_3_id",
                                        "def_player_4",
                                        "def_player_4_id",
                                        "def_player_5",
                                        "def_player_5_id",
                                        "points_made",
                                        "home_team_abbrev",
                                        "event_team_abbrev",
                                        "away_team_abbrev",
                                        "home_team_id",
                                        "away_team_id",
                                        "game_id",
                                        "game_date",
                                        "season",
                                    ],
                                )
                            ]
                        )

                    elif (
                        df.loc[df.index[-1], "event_team"]
                        == df.loc[df.index[-1], "home_team_abbrev"]
                    ):
                        row_df = pd.concat(
                            [
                                df.loc[df.index[-1], player_cols],
                                df.loc[
                                    df.index[-1],
                                    [
                                        "points_made",
                                        "home_team_abbrev",
                                        "event_team",
                                        "away_team_abbrev",
                                        "home_team_id",
                                        "away_team_id",
                                        "game_id",
                                        "game_date",
                                        "season",
                                    ],
                                ],
                            ]
                        )

                        parsed_list.extend(
                            [
                                pd.DataFrame(
                                    [list(row_df)],
                                    columns=[
                                        "def_player_1",
                                        "def_player_1_id",
                                        "def_player_2",
                                        "def_player_2_id",
                                        "def_player_3",
                                        "def_player_3_id",
                                        "def_player_4",
                                        "def_player_4_id",
                                        "def_player_5",
                                        "def_player_5_id",
                                        "off_player_1",
                                        "off_player_1_id",
                                        "off_player_2",
                                        "off_player_2_id",
                                        "off_player_3",
                                        "off_player_3_id",
                                        "off_player_4",
                                        "off_player_4_id",
                                        "off_player_5",
                                        "off_player_5_id",
                                        "points_made",
                                        "home_team_abbrev",
                                        "event_team_abbrev",
                                        "away_team_abbrev",
                                        "home_team_id",
                                        "away_team_id",
                                        "game_id",
                                        "game_date",
                                        "season",
                                    ],
                                )
                            ]
                        )

            elif df.loc[df.index[-1], "event_type_de"] in ["shot", "free-throw"]:
                if (
                    df.loc[df.index[-1], "event_team"]
                    == df.loc[df.index[-1], "home_team_abbrev"]
                ):
                    row_df = pd.concat(
                        [
                            df.loc[df.index[-1], player_cols],
                            df.loc[
                                df.index[-1],
                                [
                                    "points_made",
                                    "home_team_abbrev",
                                    "event_team",
                                    "away_team_abbrev",
                                    "home_team_id",
                                    "away_team_id",
                                    "game_id",
                                    "game_date",
                                    "season",
                                ],
                            ],
                        ]
                    )

                    parsed_list.extend(
                        [
                            pd.DataFrame(
                                [list(row_df)],
                                columns=[
                                    "off_player_1",
                                    "off_player_1_id",
                                    "off_player_2",
                                    "off_player_2_id",
                                    "off_player_3",
                                    "off_player_3_id",
                                    "off_player_4",
                                    "off_player_4_id",
                                    "off_player_5",
                                    "off_player_5_id",
                                    "def_player_1",
                                    "def_player_1_id",
                                    "def_player_2",
                                    "def_player_2_id",
                                    "def_player_3",
                                    "def_player_3_id",
                                    "def_player_4",
                                    "def_player_4_id",
                                    "def_player_5",
                                    "def_player_5_id",
                                    "points_made",
                                    "home_team_abbrev",
                                    "event_team_abbrev",
                                    "away_team_abbrev",
                                    "home_team_id",
                                    "away_team_id",
                                    "game_id",
                                    "game_date",
                                    "season",
                                ],
                            )
                        ]
                    )
                elif (
                    df.loc[df.index[-1], "event_team"]
                    == df.loc[df.index[-1], "away_team_abbrev"]
                ):
                    row_df = pd.concat(
                        [
                            df.loc[df.index[-1], player_cols],
                            df.loc[
                                df.index[-1],
                                [
                                    "points_made",
                                    "home_team_abbrev",
                                    "event_team",
                                    "away_team_abbrev",
                                    "home_team_id",
                                    "away_team_id",
                                    "game_id",
                                    "game_date",
                                    "season",
                                ],
                            ],
                        ]
                    )

                    parsed_list.extend(
                        [
                            pd.DataFrame(
                                [list(row_df)],
                                columns=[
                                    "def_player_1",
                                    "def_player_1_id",
                                    "def_player_2",
                                    "def_player_2_id",
                                    "def_player_3",
                                    "def_player_3_id",
                                    "def_player_4",
                                    "def_player_4_id",
                                    "def_player_5",
                                    "def_player_5_id",
                                    "off_player_1",
                                    "off_player_1_id",
                                    "off_player_2",
                                    "off_player_2_id",
                                    "off_player_3",
                                    "off_player_3_id",
                                    "off_player_4",
                                    "off_player_4_id",
                                    "off_player_5",
                                    "off_player_5_id",
                                    "points_made",
                                    "home_team_abbrev",
                                    "event_team_abbrev",
                                    "away_team_abbrev",
                                    "home_team_id",
                                    "away_team_id",
                                    "game_id",
                                    "game_date",
                                    "season",
                                ],
                            )
                        ]
                    )

        return parsed_list

    def _build_possessions(self, df: pd.DataFrame, include_event_agg: bool = False):
        """
        Internal helper used by rapm_possessions() and compute_on_court_exposures().

        Returns a DataFrame with one row per possession. When include_event_agg is
        True, possession-level shooting aggregates for the offense and defense are
        added.
        """

        pbp_df = df.copy()

        poss_index = pbp_df[(pbp_df.home_possession == 1) | (pbp_df.away_possession == 1)].index
        shift_dfs = []
        past_index = 0

        for i in poss_index:
            # Slice events between possession markers, skipping empty segments
            seg = pbp_df.iloc[past_index + 1 : i + 1, :].reset_index(drop=True)
            if not seg.empty:
                shift_dfs.append(seg)
            past_index = i

        parsed_possessions = self.parse_possessions(shift_dfs)

        # If no possessions were detected, return an empty DataFrame
        if not parsed_possessions:
            return pd.DataFrame()

        event_aggs = []
        for poss_df in shift_dfs:
            poss_events = annotate_events(poss_df.copy()) if not poss_df.empty else poss_df
            if poss_events.empty:
                event_aggs.append(
                    {
                        "off_team_FGA": 0,
                        "off_team_FGM": 0,
                        "off_team_3PA": 0,
                        "off_team_3PM": 0,
                        "off_team_FTA": 0,
                        "off_team_FTM": 0,
                        "def_team_FGA": 0,
                        "def_team_FGM": 0,
                        "def_team_3PA": 0,
                        "def_team_3PM": 0,
                        "def_team_FTA": 0,
                        "def_team_FTM": 0,
                        "points_for_offense": 0,
                        "points_for_defense": 0,
                    }
                )
                continue

            last_event = poss_events.iloc[-1]
            off_abbrev = last_event.get("event_team")
            off_team_id = (
                last_event.get("home_team_id")
                if off_abbrev == last_event.get("home_team_abbrev")
                else last_event.get("away_team_id")
            )
            def_team_id = (
                last_event.get("away_team_id")
                if off_abbrev == last_event.get("home_team_abbrev")
                else last_event.get("home_team_id")
            )

            off_mask = poss_events.get("team_id") == off_team_id
            def_mask = poss_events.get("team_id") == def_team_id

            off_fga = poss_events.loc[off_mask, "is_fg_attempt"].sum()
            off_fgm = poss_events.loc[off_mask, "is_fg_make"].sum()
            off_3pa = (
                poss_events.loc[off_mask, "is_fg_attempt"].astype(bool)
                & poss_events.loc[off_mask, "is_three"].astype(bool)
            )
            off_3pm = (
                poss_events.loc[off_mask, "is_fg_make"].astype(bool)
                & poss_events.loc[off_mask, "is_three"].astype(bool)
            )
            def_fga = poss_events.loc[def_mask, "is_fg_attempt"].sum()
            def_fgm = poss_events.loc[def_mask, "is_fg_make"].sum()
            def_3pa = (
                poss_events.loc[def_mask, "is_fg_attempt"].astype(bool)
                & poss_events.loc[def_mask, "is_three"].astype(bool)
            )
            def_3pm = (
                poss_events.loc[def_mask, "is_fg_make"].astype(bool)
                & poss_events.loc[def_mask, "is_three"].astype(bool)
            )

            off_points = poss_events.loc[off_mask, "points_made"].sum()
            def_points = poss_events.loc[def_mask, "points_made"].sum()

            event_aggs.append(
                {
                    "off_team_FGA": off_fga,
                    "off_team_FGM": off_fgm,
                    "off_team_3PA": off_3pa.sum() if hasattr(off_3pa, "sum") else 0,
                    "off_team_3PM": off_3pm.sum() if hasattr(off_3pm, "sum") else 0,
                    "off_team_FTA": poss_events.loc[off_mask, "is_ft"].sum(),
                    "off_team_FTM": poss_events.loc[off_mask, "is_ft_make"].sum(),
                    "def_team_FGA": def_fga,
                    "def_team_FGM": def_fgm,
                    "def_team_3PA": def_3pa.sum() if hasattr(def_3pa, "sum") else 0,
                    "def_team_3PM": def_3pm.sum() if hasattr(def_3pm, "sum") else 0,
                    "def_team_FTA": poss_events.loc[def_mask, "is_ft"].sum(),
                    "def_team_FTM": poss_events.loc[def_mask, "is_ft_make"].sum(),
                    "points_for_offense": off_points,
                    "points_for_defense": def_points,
                }
            )

        poss_df = pd.concat(parsed_possessions, sort=True)

        poss_df["off_team_abbrev"] = poss_df["event_team_abbrev"]
        poss_df["off_team_id"] = np.where(
            poss_df["event_team_abbrev"] == poss_df["home_team_abbrev"],
            poss_df["home_team_id"],
            poss_df["away_team_id"],
        )
        poss_df["def_team_abbrev"] = np.where(
            poss_df["event_team_abbrev"] == poss_df["home_team_abbrev"],
            poss_df["away_team_abbrev"],
            poss_df["home_team_abbrev"],
        )
        poss_df["def_team_id"] = np.where(
            poss_df["event_team_abbrev"] == poss_df["home_team_abbrev"],
            poss_df["away_team_id"],
            poss_df["home_team_id"],
        )
        if event_aggs:
            agg_df = pd.DataFrame(event_aggs)
            poss_df["points_for_offense"] = agg_df["points_for_offense"].values
            poss_df["points_for_defense"] = agg_df["points_for_defense"].values

            if include_event_agg:
                for col in [
                    "off_team_FGA",
                    "off_team_FGM",
                    "off_team_3PA",
                    "off_team_3PM",
                    "off_team_FTA",
                    "off_team_FTM",
                    "def_team_FGA",
                    "def_team_FGM",
                    "def_team_3PA",
                    "def_team_3PM",
                    "def_team_FTA",
                    "def_team_FTM",
                ]:
                    poss_df[col] = agg_df[col].values
        else:
            # Fallback when no events were parsed
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

    def player_box_glossary(
        self,
        player_meta: pd.DataFrame | None = None,
        game_meta: pd.DataFrame | None = None,
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

        box_df = build_player_box(
            df=df,
            counts_df=counts_df,
            exposures_df=exposures_df,
            player_meta=player_meta,
            game_meta=game_meta,
            pbg_stats=self.playerbygamestats(),
        )

        # Sanity check: on-court points must match team totals.
        self._check_on_court_points_consistency(box_df)

        return box_df

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
