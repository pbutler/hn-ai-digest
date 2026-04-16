# HackerNews Tools

Utility scripts for downloading the Hacker News RSS feed, parsing linked articles, and ranking them with embeddings.

## Environment

This project uses `uv` to manage a local virtual environment and dependencies.
Runtime configuration is loaded from `~/.config/hn-ai-digest/env`.

## Common Commands

```bash
uv sync
uv run python src/fetch_hn_rss_articles.py
uv run python src/filter_hn_topics.py
```
