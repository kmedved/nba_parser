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

## Local Test Data
- Sample scraped play-by-play data lives under `test/` for quick experiments and regression checks.
- `test/0021900151_cdn.csv` holds a CDN-era reference game, while `test/0021900151_v2.csv` contains the same game from the `playbyplayv2` API.
- Additional season games (e.g., `test/21900*.csv` and `test/21100736.csv`) are available for broader coverage.

Use these fixtures to validate parsers against both CDN and `playbyplayv2` formats and to confirm that downstream box score and possession outputs remain exact.

## Repository File Overview
Use this section as the master, up-to-date map of what lives in the repository. Update it any time files are added, renamed, or significantly changed.

### Top-level project files
- `README.md`: Project landing page with usage examples for `PbP`, `TeamTotals`, and `PlayerTotals`, plus status badges indicating the package is unmaintained.
- `.travis.yml`: Legacy CI configuration that installs `requirements.txt` and runs `pytest` coverage against `test/test_unit/` (tests no longer present here).
- `.gitignore`: Ignores Jupyter checkpoints, Python caches, distribution artifacts, coverage output, and temporary “Untitled.*” notebooks.
- `requirements.txt`: Pinned runtime and dev dependencies including pandas, numpy, scikit-learn, SQLAlchemy, pytest/coverage tools, and the `nba-scraper` dependency.
- `setup.py`: Minimal packaging metadata for the `nba_parser` module (GPLv3), reading the README as the long description and requiring pandas/numpy.
- `Agents.MD` (this file): Contributor guidance plus this repository map.

### Python package (`nba_parser/`)
- `__init__.py`: Exposes the `PbP` class as the package’s public entry point.
- `pbp.py`: Core implementation of the `PbP` class that ingests a play-by-play DataFrame and derives rich statistics. It normalizes game metadata, flags possessions, computes player-level stats (scoring, assists, rebounds, turnovers, fouls, steals, blocks, plus/minus, time-on-court), builds possession logs, and exposes methods for player/game/lineup outputs. Relies on helper functions from `box_glossary.py` for event annotation and box construction.
- `box_glossary.py`: Helper utilities for enriching play-by-play data. Provides shot-zone classification, event annotation (`annotate_events`), per-player counting (`accumulate_player_counts`), on-court exposure calculations, and `build_player_box` for assembling box-score style outputs with many rate and zone splits.

### Tests and data fixtures (`test/`)
- `__init__.py`: Empty initializer to make `test` a package.
- CSV fixtures: Sample scraped play-by-play datasets used for local validation and regression checks. Files include CDN-era and `playbyplayv2` examples (`0021900151_cdn.csv`, `0021900151_v2.csv`) and numerous other season games (`21900002.csv`, `21900025.csv`, `21900040.csv`, `21900054.csv`, `21900074.csv`, `21900088.csv`, `21900100.csv`, `21900126.csv`, `21900139.csv`, `21900151.csv`, `21100736.csv`, `20700233.csv`).
