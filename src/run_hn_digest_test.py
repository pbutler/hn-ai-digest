#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import re
import sqlite3
import ssl
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path
from typing import Any

HN_RSS_URL = "https://news.ycombinator.com/rss"
BASE_DIR = Path("/Users/pbutler/Documents/HackerNews")
OUTPUT_PATH = BASE_DIR / "output" / "hn-ai-24h-digest.xml"
DB_PATH = BASE_DIR / "output" / "hn-ai-articles.sqlite3"
USER_AGENT = "hn-ai-daily-rss-digest/2.0"
FEED_TIMEOUT_SECONDS = 15
ARTICLE_TIMEOUT_SECONDS = 12
DEFUDDLE_TIMEOUT_SECONDS = 20
MAX_FETCH_WORKERS = 8
TOPIC_KEYWORDS = [
    "gpu",
    "cuda",
    "rocm",
    "metal performance shaders",
    "shader",
    "parallel computing",
    "llm",
    "large language model",
    "language model",
    "transformer",
    "transformers",
    "attention mechanism",
    "generative ai",
    "genai",
    "artificial intelligence",
    "deep learning",
    "machine learning",
    "neural network",
    "multimodal model",
    "diffusion model",
    "inference",
    "fine-tuning",
    "rag",
    "embedding model",
    "openai",
    "anthropic",
    "gemini",
    "claude",
    "gpt-",
    "pytorch",
    "jax",
    "mlx",
    "tensor",
]


def fetch_url(url: str, timeout: int = ARTICLE_TIMEOUT_SECONDS) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        content_type = response.headers.get("content-type", "")
        charset_match = re.search(r"charset=([\w\-]+)", content_type, re.I)
        encoding = charset_match.group(1) if charset_match else "utf-8"
        return response.read().decode(encoding, errors="replace")


def parse_pubdate(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style.*?>.*?</style>", " ", raw)
    raw = re.sub(r"(?is)<(br|p|div|li|h1|h2|h3|h4|h5|h6|tr|blockquote)[^>]*>", "\n", raw)
    raw = re.sub(r"(?is)<[^>]+>", " ", raw)
    return normalize_whitespace(html.unescape(raw))


def summarize_text(text: str, max_len: int = 2200) -> str:
    if len(text) <= max_len:
        return text
    clipped = text[:max_len]
    boundary = clipped.rfind(" ")
    if boundary > 400:
        clipped = clipped[:boundary]
    return clipped + "..."


def stable_guid(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def ensure_database(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guid TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            hn_title TEXT NOT NULL,
            hn_comments_url TEXT,
            hn_pub_date TEXT NOT NULL,
            hn_pub_ts INTEGER NOT NULL,
            rss_guid TEXT,
            parser_used TEXT,
            fetched_at TEXT NOT NULL,
            article_title TEXT,
            article_byline TEXT,
            article_excerpt TEXT,
            article_text TEXT,
            article_html TEXT,
            matched_keywords TEXT,
            is_relevant INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_guid ON articles(guid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_pub_ts ON articles(hn_pub_ts DESC)")
    conn.commit()


def run_defuddle(url: str) -> dict[str, str] | None:
    try:
        result = subprocess.run(
            ["defuddle", "parse", "-j", url],
            check=False,
            capture_output=True,
            text=True,
            timeout=DEFUDDLE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    stdout = result.stdout.strip()
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return {"text": normalize_whitespace(stdout), "html": "", "title": "", "byline": "", "excerpt": ""}

    if isinstance(parsed, dict):
        return {
            "title": normalize_whitespace(str(parsed.get("title", ""))),
            "byline": normalize_whitespace(str(parsed.get("byline", ""))),
            "excerpt": normalize_whitespace(str(parsed.get("excerpt", ""))),
            "text": normalize_whitespace(
                str(
                    parsed.get("textContent")
                    or parsed.get("text")
                    or parsed.get("content")
                    or parsed.get("article")
                    or ""
                )
            ),
            "html": str(parsed.get("contentHtml") or parsed.get("html") or ""),
        }

    return {"text": normalize_whitespace(stdout), "html": "", "title": "", "byline": "", "excerpt": ""}


def parse_article(url: str) -> dict[str, str]:
    defuddled = run_defuddle(url)
    if defuddled:
        defuddled["parser_used"] = "defuddle"
        return defuddled

    article_html = fetch_url(url, timeout=ARTICLE_TIMEOUT_SECONDS)
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", article_html)
    title = strip_html(title_match.group(1)) if title_match else ""
    text = summarize_text(strip_html(article_html))
    return {
        "title": title,
        "byline": "",
        "excerpt": summarize_text(text, max_len=500),
        "text": text,
        "html": "",
        "parser_used": "html-fallback",
    }


def matched_keywords(title: str, url: str, article_text: str, excerpt: str) -> list[str]:
    haystack = " ".join([title, url, article_text, excerpt]).lower()
    matches: list[str] = []
    for keyword in TOPIC_KEYWORDS:
        if keyword in haystack:
            matches.append(keyword)

    # Keep short acronyms strict.
    if re.search(r"\bai\b", haystack):
        matches.append("ai")
    if re.search(r"\bml\b", haystack):
        matches.append("ml")
    if re.search(r"\bgpu\b", haystack):
        matches.append("gpu")
    if re.search(r"\bllms?\b", haystack):
        matches.append("llm")

    return sorted(set(matches))


def build_article_record(item: dict[str, Any], now: datetime) -> tuple[Any, ...]:
    url = item["link"]
    guid = stable_guid(url)
    try:
        article = parse_article(url)
    except Exception as exc:
        article = {
            "title": "",
            "byline": "",
            "excerpt": "",
            "text": "",
            "html": "",
            "parser_used": f"parse-error:{type(exc).__name__}",
        }
    keywords = matched_keywords(
        item["title"],
        url,
        article.get("text", ""),
        article.get("excerpt", ""),
    )
    return (
        guid,
        url,
        item["title"],
        item.get("comments_url", ""),
        item["pubDate"],
        int(item["pub_dt"].timestamp()),
        item.get("rss_guid", ""),
        article.get("parser_used", ""),
        now.isoformat(),
        article.get("title", ""),
        article.get("byline", ""),
        article.get("excerpt", ""),
        article.get("text", ""),
        article.get("html", ""),
        json.dumps(keywords, ensure_ascii=True),
        1 if keywords else 0,
    )


def upsert_article(conn: sqlite3.Connection, record: tuple[Any, ...]) -> None:
    conn.execute(
        """
        INSERT INTO articles (
            guid,
            url,
            hn_title,
            hn_comments_url,
            hn_pub_date,
            hn_pub_ts,
            rss_guid,
            parser_used,
            fetched_at,
            article_title,
            article_byline,
            article_excerpt,
            article_text,
            article_html,
            matched_keywords,
            is_relevant
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            guid=excluded.guid,
            hn_title=excluded.hn_title,
            hn_comments_url=excluded.hn_comments_url,
            hn_pub_date=excluded.hn_pub_date,
            hn_pub_ts=excluded.hn_pub_ts,
            rss_guid=excluded.rss_guid,
            parser_used=excluded.parser_used,
            fetched_at=excluded.fetched_at,
            article_title=excluded.article_title,
            article_byline=excluded.article_byline,
            article_excerpt=excluded.article_excerpt,
            article_text=excluded.article_text,
            article_html=excluded.article_html,
            matched_keywords=excluded.matched_keywords,
            is_relevant=excluded.is_relevant
        """,
        record,
    )


def load_feed_items() -> list[dict[str, Any]]:
    rss_xml = fetch_url(HN_RSS_URL, timeout=FEED_TIMEOUT_SECONDS)
    root = ET.fromstring(rss_xml)
    items: list[dict[str, Any]] = []

    for node in root.findall("./channel/item"):
        title = normalize_whitespace(node.findtext("title") or "")
        link = normalize_whitespace(node.findtext("link") or "")
        pub_raw = normalize_whitespace(node.findtext("pubDate") or "")
        pub_dt = parse_pubdate(pub_raw)
        if not title or not link or not pub_dt:
            continue

        comments_url = ""
        for child in list(node):
            if child.tag.endswith("comments") and child.text:
                comments_url = normalize_whitespace(child.text)
                break

        items.append(
            {
                "title": title,
                "link": link,
                "pubDate": pub_raw,
                "pub_dt": pub_dt,
                "rss_guid": normalize_whitespace(node.findtext("guid") or ""),
                "comments_url": comments_url,
            }
        )

    return items


def build_digest_xml(conn: sqlite3.Connection, now: datetime, window_hours: int) -> str:
    cutoff_ts = int((now - timedelta(hours=window_hours)).timestamp())
    now_ts = int(now.timestamp())
    rows = conn.execute(
        """
        SELECT guid, url, hn_title, hn_pub_date, article_title, article_excerpt, article_text, matched_keywords
        FROM articles
        WHERE is_relevant = 1 AND hn_pub_ts >= ? AND hn_pub_ts <= ?
        ORDER BY hn_pub_ts DESC, guid DESC
        """,
        (cutoff_ts, now_ts),
    ).fetchall()

    seen: set[str] = set()
    items: list[sqlite3.Row] = []
    for row in rows:
        if row["guid"] in seen or row["url"] in seen:
            continue
        seen.add(row["guid"])
        seen.add(row["url"])
        items.append(row)

    channel_title = "Hacker News AI 24h Digest"
    channel_link = "https://news.ycombinator.com/rss"
    channel_desc = (
        "Hacker News stories from the last 24 hours related to GPU programming, LLMs, "
        "Generative AI, deep learning, transformer models, and adjacent topics."
    )
    managing_editor = "HN AI 24h Morning RSS Digest"

    rss_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0">',
        "  <channel>",
        f"    <title>{html.escape(channel_title)}</title>",
        f"    <link>{html.escape(channel_link)}</link>",
        f"    <description>{html.escape(channel_desc)}</description>",
        f"    <managingEditor>{html.escape(managing_editor)}</managingEditor>",
        f"    <lastBuildDate>{format_datetime(now)}</lastBuildDate>",
    ]

    for row in items:
        keywords = json.loads(row["matched_keywords"] or "[]")
        summary_source = row["article_excerpt"] or row["article_text"] or row["article_title"] or row["hn_title"]
        summary = summarize_text(normalize_whitespace(summary_source), max_len=800)
        if keywords:
            summary = f"[topics: {', '.join(keywords[:8])}] {summary}"

        rss_lines.extend(
            [
                "    <item>",
                f"      <title>{html.escape(row['hn_title'])}</title>",
                f"      <link>{html.escape(row['url'])}</link>",
                f"      <guid isPermaLink=\"false\">{row['guid']}</guid>",
                f"      <pubDate>{html.escape(row['hn_pub_date'])}</pubDate>",
                f"      <description>{html.escape(summary)}</description>",
                "    </item>",
            ]
        )

    rss_lines.extend(["  </channel>", "</rss>"])
    return "\n".join(rss_lines) + "\n"


def write_digest(path: Path, xml_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml_text, encoding="utf-8")


def run(window_hours: int = 24) -> tuple[int, Path, Path]:
    now = datetime.now(timezone.utc)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        ensure_database(conn)
        feed_error: Exception | None = None
        try:
            feed_items = load_feed_items()
        except Exception as exc:
            feed_items = []
            feed_error = exc

        records: list[tuple[Any, ...]] = []
        if feed_items:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as executor:
                for record in executor.map(lambda item: build_article_record(item, now), feed_items):
                    records.append(record)

        for record in records:
            upsert_article(conn, record)
        conn.commit()

        if feed_error is not None:
            existing_count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            if existing_count == 0:
                raise RuntimeError(f"could not download Hacker News RSS and database is empty: {feed_error}") from feed_error

        rss_xml = build_digest_xml(conn, now, window_hours)

    write_digest(OUTPUT_PATH, rss_xml)
    root = ET.fromstring(rss_xml)
    count = len(root.findall("./channel/item"))
    return count, OUTPUT_PATH, DB_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the HN AI 24h digest RSS feed")
    parser.add_argument("--window-hours", type=int, default=24)
    args = parser.parse_args()

    try:
        count, output_path, db_path = run(window_hours=args.window_hours)
    except Exception as exc:
        print(f"Digest build failed: {exc}", file=sys.stderr)
        return 1

    print(f"Generated digest with {count} items: {output_path}")
    print(f"Article database: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
