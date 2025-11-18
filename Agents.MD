# Repository Guidance for LLM Contributors

- This repository works alongside `nba_scraper` to parse data from both the `playbyplayv2` NBA API endpoint and the CDN endpoint. The formats differ, and support for both is required.
- The goal is to generate player-level box score data, on/off data, and other downstream datasets produced from `nba_scraper`. Possession-level data should be produced for RAPM calculations, but running those calculations is out of scope.
- Parsing must be exact—do not rely on approximations when generating data.
- This project depends on `kmedved/nba_scraper` (the repository owner's fork) for scraping.

## Example `nba_scraper` Usage

```python
# Choose a few CDN-era games (2019+). Adjust as you like.
game_ids = [
    "0021900151",
]

# Scrape play-by-play for these games as a single canonical DataFrame
pbp_df = ns.scrape_game(game_ids, data_format="pandas")
pbp_df = lineup_builder.attach_lineups(pbp_df)
```

This produces a DataFrame similar to:

```
game_id\tperiod\tpctimestring\tseconds_elapsed\tevent_length\taction_number\torder_number\teventnum\ttime_actual\tteam_id\tteam_tricode\tevent_team\tplayer1_id\tplayer1_name\tplayer1_team_id\tplayer2_id\tplayer2_name\tplayer2_team_id\tplayer3_id\tplayer3_name\tplayer3_team_id\thome_team_id\thome_team_abbrev\taway_team_id\taway_team_abbrev\thomedescription\tvisitordescription\tgame_date\tseason\tfamily\tsubfamily\teventmsgtype\teventmsgactiontype\tevent_type_de\tis_three\tshot_made\tpoints_made\tshot_distance\tx\ty\tside\tarea\tarea_detail\tassist_id\tblock_id\tsteal_id\tstyle_flags\tqualifiers\tis_o_rebound\tis_d_rebound\tteam_rebound\tlinked_shot_action_number\tpossession_after\tscore_home\tscore_away\tscoremargin\tis_turnover\tis_steal\tis_block\tft_n\tft_m\thome_player_1_id\taway_player_1_id\thome_player_2_id\taway_player_2_id\thome_player_3_id\taway_player_3_id\thome_player_4_id\taway_player_4_id\thome_player_5_id\taway_player_5_id\thome_player_1\thome_player_2\thome_player_3\thome_player_4\thome_player_5\taway_player_1\taway_player_2\taway_player_3\taway_player_4\taway_player_5
21900151\t1\t12:00\t0\t0\t2\t20000\t2\t2019-11-13T02:10:31.6Z\t0\t\t\t0\t\t0\t0\t\t0\t0\t\t0\t1610612756\tPHX\t1610612747\tLAL\tPeriod Start\t\t11/13/2019\t2019\tperiod\tstart\t12\t0\tperiod\t0\t\t0\t\t\t\t\t0\t0\t0\t[]\t[]\t0\t0\t0\t\t0\t0\t0\t0\t0\t0\t0\t\t\t203382\t202340\t203967\t201580\t201937\t203076\t1626162\t201980\t1626164\t2544\tBaynes\tSaric\tRubio\tOubre Jr.\tBooker\tBradley\tMcGee\tDavis\tGreen\tJames
21900151\t1\t11:57\t3\t3\t4\t40000\t4\t2019-11-13T02:10:34.1Z\t1610612756\tPHX\tPHX\t203382\tBaynes\t1610612756\t201580\tMcGee\t1610612747\t1626162\tOubre Jr.\t1610612756\t1610612756\tPHX\t1610612747\tLAL\tJump Ball: Baynes vs. McGee tip to Oubre Jr.\t\t11/13/2019\t2019\tjumpball\trecovered\t10\t0\tjump-ball\t0\t\t0\t\t\t\t\t0\t0\t0\t[]\t[]\t0\t0\t0\t\t1610612756\t0\t0\t0\t0\t0\t0\t\t\t203382\t202340\t203967\t201580\t201937\t203076\t1626162\t201980\t1626164\t2544\tBaynes\tSaric\tRubio\tOubre Jr.\tBooker\tBradley\tMcGee\tDavis\tGreen\tJames
21900151\t1\t11:40\t20\t17\t7\t70000\t7\t2019-11-13T02:10:51.1Z\t1610612756\tPHX\tPHX\t203967\tSaric\t1610612756\t1626164\tBooker\t1610612756\t0\t\t0\t1610612756\tPHX\t1610612747\tLAL\tSaric jumpshot: Made (2 PTS) assist: Booker (1 AST)\t\t11/13/2019\t2019\t2pt\tjumpshot\t1\t0\tshot\t0\t1\t2\t14.07\t96.40275953\t78.01011029\tright\t\t\t\t1626164\t0\t0\t[]\t[]\t0\t0\t0\t\t1610612747\t2\t0\t2\t0\t0\t0\t\t\t203382\t202340\t203967\t201580\t201937\t203076\t1626162\t201980\t1626164\t2544\tBaynes\tSaric\tRubio\tOubre Jr.\tBooker\tBradley\tMcGee\tDavis\tGreen\tJames
```
