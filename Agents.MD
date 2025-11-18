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

# `nba_scraper` agent overview

This document summarizes how the package ingests NBA data, normalizes it to the canonical schema used by `nba_parser`, and which knobs influence the pipeline. It is intended for LLM agents working on the `nba_parser` fork so they can understand the expectations of incoming data.

## What the package does

`nba_scraper` fetches or loads NBA play-by-play data and emits a pandas dataframe (or CSV) where each row is a canonical action enriched with lineup context and structured flags. The package supports two upstream APIs:

* **CDN live data** (2019–present) – primary source for new games.
* **Legacy v2 JSON archives** – backfilled data for older seasons.

Regardless of source, the output dataframe preserves compatibility with the downstream `nba_parser` possession engine.

## Primary entry points

* `nba_scraper.nba_scraper.scrape_game`, `.scrape_season`, `.scrape_date_range`, `.scrape_from_files` – user-facing helpers that wrap the unified parsing pipeline and optionally write CSVs. These functions resolve game IDs, default to CDN scraping, and surface the `data_format`/`data_dir` knobs from the README.
* `nba_scraper.io_sources.parse_any` – low-level router used by the helpers and tests. Callers pass both the payload reference and a `SourceKind` describing how to load it. This function orchestrates parsing, lineup enrichment, optional shot-coordinate backfill, and box-score validation.

## Source routing expectations

`parse_any` expects different inputs per `SourceKind`:

* `CDN_REMOTE`: a game ID string (with or without leading zeros). The helper fetches play-by-play and box score JSON from `nba_scraper.cdn_client`, then calls the CDN parser.
* `CDN_LOCAL`: a tuple of paths or dicts `(pbp, box)`. Both are required because lineup reconstruction depends on box score metadata.
* `V2_LOCAL`: path to a single legacy v2 JSON file.
* `V2_DICT`: a preloaded v2 JSON dict (useful for tests or caching layers).

When `NBA_SCRAPER_BACKFILL_COORDS=1`, CDN pathways attempt to load or fetch a shot chart and merge the coordinate dataframe into the parsed play-by-play. When `NBA_SCRAPER_VALIDATE_BOX=1`, the router compares PbP-derived team totals against the official box score and logs mismatches.

## Parsing flow and normalization

1. **CDN parsing (`cdn_parser.parse_actions_to_rows`)** – Normalizes liveData play-by-play events using box score metadata when available. The parser computes canonical descriptors, event families/subfamilies, free-throw trip counters, synthesized coordinates (when missing), and qualifier flags (steals, blocks, goaltends, etc.).
2. **Legacy v2 parsing (`v2_parser.parse_actions_to_rows`)** – Mirrors the CDN output schema for archived JSON while inferring modern metadata (family/subfamily, free-throw trip structure, synthetic coordinates) so downstream consumers see the same columns.
3. **Lineup enrichment (`lineup_builder.attach_lineups`)** – Seeds starting lineups from the box score, processes substitution semantics for CDN and v2 payloads, and backfills on-court `home_player_*`/`away_player_*` and `*_id` columns for every event.
4. **Coordinate backfill (`coords_backfill.backfill_coords_with_shotchart`)** – If a shot chart dataframe is available, it overrides synthesized coordinates with official values and cleans style flags.
5. **Box score validation (`boxscore_validation.log_team_boxscore_mismatches`)** – Optional consistency check that recomputes team totals from the canonical dataframe and logs differences versus the box score.

All parsers share helpers in `parser_utils.py` for team backfilling, coordinate presets, and possession inference, as well as `schema.py` for column ordering and label lookups. Mapping helpers in `nba_scraper.mapping` (see below) are invoked by both parsers to align descriptors and event codes.

## Canonical dataframe shape

Downstream consumers, including `nba_parser`, expect the dataframe to expose the legacy numeric fields plus enriched metadata and lineup context:

* **Identity and timing** – `game_id`, `eventnum` (CDN action number / v2 eventnum), `period`, `pctimestring`, `seconds_elapsed`, `eventmsgtype`, `eventmsgactiontype`.
* **Descriptors and labels** – `event_type_de`, `descriptor`, `subType`, `event_family`, `event_sub_family`, `qualifiers`.
* **Teams and players** – acting and secondary `player*_id`/`player*_team_id`, `event_team` abbreviation, and on-court lineup columns `home_player_1…5`/`away_player_1…5` with matching ID fields.
* **Scoring context** – `score`, `scoremargin`, `points_made`, `points_made_team`, `is_turnover`, `is_block`, `is_steal`, plus `ft_n`/`ft_m` counters for free throws.
* **Shot coordinates** – `x`, `y`, and `shot_distance` from the feed or shot chart backfill.
* **Game metadata** – `game_date`, `season`, and home/away team identifiers carried forward from the feed or box score.

The tests in `tests/test_integration.py`, `tests/test_functional.py`, and `tests/test_with_parser.py` validate that these columns remain present and compatible with `nba_parser`.

## Descriptor mapping and overrides

Both parsers consult the mapping utilities under `nba_scraper/mapping`:

* `descriptor_norm.py` normalizes raw descriptor strings (including style flags) so they can be keyed consistently.
* `event_codebook.py` maps descriptor signatures to legacy `eventmsgtype`/`eventmsgactiontype` codes and free-throw trip counters.
* `loader.py` optionally loads a YAML mapping file (see `mapping_template.yml`) when the environment variable `NBA_SCRAPER_MAP` points to a curated override. This allows downstream agents to remap specific signatures without code changes.

If no mapping file is provided the parsers fall back to built-in defaults while still emitting canonical families and descriptors.

## Data quality assumptions

* Never fabricate or guess fields: missing values remain `None`/empty and are only synthesized by deterministic helpers (e.g., coordinate presets for known descriptor patterns).
* Lineup reconstruction depends on a valid box score for CDN flows. For v2 archives the lineup builder infers substitutions directly from play-by-play events.
* Shot chart backfill is best-effort: the pipeline silently continues when the chart is unavailable or empty.

## How `nba_parser` should integrate

`nba_parser` consumers should treat the canonical dataframe as a drop-in equivalent to the legacy v2 format but with richer metadata. Typical integration:

1. Call `io_sources.parse_any` (or a higher-level helper) to obtain the dataframe for a game, date range, or season.
2. If needed, merge additional analytics columns before handing the dataframe to `nba_parser.PbP`.
3. Use `PbP` methods (`playerbygamestats`, `teambygamestats`, possession slicing) as usual; the free-throw counters, lineup fields, and descriptors are aligned to the expectations of `nba_parser` tests.

Because `nba_scraper` already normalizes both CDN and v2 feeds, no extra preprocessing is required on the `nba_parser` side beyond standard Pandas handling.
