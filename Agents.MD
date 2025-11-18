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
