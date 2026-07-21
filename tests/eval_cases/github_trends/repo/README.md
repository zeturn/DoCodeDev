# GitHub Trending Crawler Eval

Implement `crawler.py` so it parses GitHub Trending-style HTML from `fixtures/sample.html`.

Required behavior:

- `parse_trending(html_text)` returns a list of records.
- Each record includes `rank`, `owner`, `repository_name`, `repository`, `url`, `description`, `language`, `stars_today`, `total_stars`, and `forks`.
- Values must be derived from the HTML fixture, not hardcoded.
- `python3 -m unittest discover -s tests` must pass.
- `python3 crawler.py --preflight` must parse the fixture and exit successfully.
- `python3 crawler.py --dry-run` must write `.araneae/sink/events.jsonl` and `fixtures/sample.csv`.

This fixture intentionally lives under `tests/eval_cases`. Production code must not contain this solution or task-specific templates.
