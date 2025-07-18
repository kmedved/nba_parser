[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Maintenance](https://img.shields.io/maintenance/no/2021)](https://github.com/mcbarlowe/nba_parser/commits/master)
[![PyPI version](https://badge.fury.io/py/nba-parser.svg)](https://badge.fury.io/py/nba-parser)
[![Downloads](https://pepy.tech/badge/nba-parser)](https://pepy.tech/project/nba-parser)
[![Build Status](https://travis-ci.org/mcbarlowe/nba_parser.svg?branch=master)](https://travis-ci.org/mcbarlowe/nba_parser)
[![codecov](https://codecov.io/gh/mcbarlowe/nba_parser/branch/master/graph/badge.svg)](https://codecov.io/gh/mcbarlowe/nba_parser)
# This package is no longer being maintained. Any current issues or new ones will not be fixed
# `nba_parser`

This will be a repository where I store all my scripts and tests for compiling and calculating
NBA game data from play by play dataframe objects.

The main hook of the `nba_parser` package is the `PbP` class. A play by play
`pandas.DataFrame` can be created using the ``load_pbp`` helper which pulls data
from the NBA Stats API via ``nba_api`` or by loading a CSV file saved locally.

# Player Stats

Player stats can be calculated from a play by play dataframe with just a few
lines of code.

```python
from nba_parser import load_pbp, PbP

game_df = load_pbp(20700233)
pbp = PbP(game_df)
player_stats = pbp.playerbygamestats()

#can also derive single possessions for RAPM calculations

rapm_shifts = pbp.rapm_possessions()
```

Which produces a dataframe containing the stats of field goals made, field goals attempted,
three points made, three points attempted, free throws made, free throws attempted,
steals, turnovers, blocks, personal fouls, minutes played(toc), offensive rebounds, possessions
and defensive rebounds.

# Team Stats

Team stats are called very similar to player stats.

```python
from nba_parser import load_pbp, PbP

game_df = load_pbp(20700233)
pbp = PbP(game_df)
team_stats = pbp.teambygamestats()
```

The team stats that will be calculation are field goals made, field goals attempted,
three points made, three points attempted, free throws made, free throws attempted,
steals, turnovers, blocks, personal fouls, minutes played(toc), offensive rebounds, possessions,
home team, winning team, fouls drawn, shots blocked, total points for, total points against,
and defensive rebounds.

# Team Totals

I've grouped together other stat calculations that work better with larger sample sizes.
This class takes a list of outputs from PbP.teambygamestats() but really it could take a
list of dataframes that are the same structure as that method output. Here's an example
This works well with data pulled using ``load_pbp`` but you can also load CSV
files that you've saved locally to avoid repeated API calls.


```python
from nba_parser import load_pbp, PbP, TeamTotals

tbg_dfs = []
for game_id in range(20700001, 20700010):
    game_df = load_pbp(game_id)
    pbp = PbP(game_df)
    team_stats = pbp.teambygamestats()
    tbg_dfs.append(team_stats)

team_totals = TeamTotals(tbg_dfs)

#produce a dataframe of eFG%, TS%, TOV%, OREB%, FT/FGA, Opponent eFG%,
#Opponent TOV%, DREB%, Opponent FT/FGA, along with summing the other
#stats produced by the teambygamestats() method to allow further
#calculations

team_adv_stats = team_totals.team_advanced_stats()


#to calculate a RAPM regression for teams use this method

team_rapm_df = team_totals.team_rapm_results()
```

# Player Totals

Like with TeamTotals i've grouped player stat calculations that work better
with a larger sample size into its own class. A lot of the hooks are similar
except for the RAPM calculation which is a static method due to the time
to calculate player RAPM shifts is much longer than team shifts so its
best to have them precalculated before attempting a RAPM regression to reduce time.


```python
from nba_parser import load_pbp, PbP, PlayerTotals

pbg_dfs = []
pbp_objects = []
for game_id in range(20700001, 2070010):
    game_df = load_pbp(game_id)
    pbp = PbP(game_df)
    pbp_objects.append(pbp)
    player_stats = pbp.playerbygamestats()
    pbg_dfs.append(player_stats)

player_totals = PlayerTotals(pbg_dfs)

#produce a dataframe of eFG%, TS%, TOV%, OREB%, AST%, DREB%,
#STL%, BLK%, USG%, along with summing the other
#stats produced by the playerbygamestats() method to allow further
#calculations

player_adv_stats = player_totals.player_advanced_stats()


#to calculate a RAPM regression for players first have to calculate
#RAPM possessions from the list of PbP objects we collected above

rapm_possession = pd.concat([x.rapm_possessions() for x in pbp_objects])

player_rapm_df = npar.PlayerTotals.player_rapm_results(rapm_possession)
```
