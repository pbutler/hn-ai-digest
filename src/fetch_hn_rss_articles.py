#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from loguru import logger

try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover
    DocumentConverter = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

try:
    import zvec
except ImportError:  # pragma: no cover
    zvec = None  # type: ignore[assignment]

HN_RSS_URL = "https://news.ycombinator.com/rss"
BASE_DIR = Path("/Users/pbutler/Documents/HackerNews")
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_SUMMARY_MODEL = "gpt-5-mini"
USER_AGENT = "hn-rss-article-ingester/2.0 (+https://news.ycombinator.com/)"
RSS_TIMEOUT_SECONDS = 20
ARTICLE_TIMEOUT_SECONDS = 20
DEFUDDLE_TIMEOUT_SECONDS = 45
MAX_WORKERS = 8
EMBED_BATCH_SIZE = 32
MAX_DOCUMENT_CHARS = 12000


def configure_logging(verbose: bool) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


@dataclass(frozen=True)
class FeedItem:
    rss_guid: str
    title: str
    link: str
    comments_url: str
    pub_date: str
    pub_ts: int | None
    description: str


def require_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("Missing dependency: install the 'openai' Python package")
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")
    logger.debug("Loaded OpenAI API key from environment")
    return OpenAI(api_key=api_key)


def require_docling() -> type[DocumentConverter]:
    if DocumentConverter is None:
        raise RuntimeError("Missing dependency: install the 'docling' Python package")
    return DocumentConverter


def require_zvec() -> None:
    if zvec is None:
        raise RuntimeError("Missing dependency: install the 'zvec' Python package")


def require_defuddle() -> str:
    executable = shutil.which("defuddle")
    if not executable:
        raise RuntimeError("defuddle executable not found in PATH")
    logger.debug("Using defuddle executable at {}", executable)
    return executable


def chunked[T](items: list[T], size: int) -> Iterable[list[T]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def doc_id_for_url(article_url: str) -> str:
    return sha256_text(article_url)


def zvec_path_for_model(model: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-") or "embedding-model"
    return BASE_DIR / "output" / f"hn-rss-articles-{slug}.zvec"


def fetch_url(url: str, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        encoding = response.headers.get_content_charset() or "utf-8"
        payload = response.read()
    return payload.decode(encoding, errors="replace")


def parse_pub_date(value: str) -> int | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp())


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").split())


def is_probable_pdf_url(url: str) -> bool:
    return bool(re.search(r"\.pdf(?:$|[?#])", url, re.I))


def is_pdf_content_type(content_type: str) -> bool:
    return "application/pdf" in content_type.lower()


def load_feed_items(rss_url: str) -> list[FeedItem]:
    logger.info("Downloading Hacker News RSS feed from {}", rss_url)
    rss_text = fetch_url(rss_url, timeout=RSS_TIMEOUT_SECONDS)
    root = ET.fromstring(rss_text)
    items: list[FeedItem] = []

    for node in root.findall("./channel/item"):
        title = normalize_text(node.findtext("title"))
        link = normalize_text(node.findtext("link"))
        if not title or not link:
            continue

        comments_url = ""
        for child in list(node):
            if child.tag.endswith("comments") and child.text:
                comments_url = normalize_text(child.text)
                break

        items.append(
            FeedItem(
                rss_guid=normalize_text(node.findtext("guid")) or link,
                title=title,
                link=link,
                comments_url=comments_url,
                pub_date=normalize_text(node.findtext("pubDate")),
                pub_ts=parse_pub_date(normalize_text(node.findtext("pubDate"))),
                description=normalize_text(node.findtext("description")),
            )
        )

    logger.info("Loaded {} RSS items", len(items))
    return items


def summarize_hn_comments_with_web_search(
    client: OpenAI,
    model: str,
    item: FeedItem,
) -> str:
    if not item.comments_url:
        return ""
    logger.info("Summarizing Hacker News discussion via web search for {}", item.comments_url)
    response = client.responses.create(
        model=model,
        tools=[
            {
                "type": "web_search",
                "filters": {"allowed_domains": ["news.ycombinator.com"]},
            }
        ],
        tool_choice="auto",
        instructions=(
            "Find the Hacker News discussion thread for the provided comments URL and summarize that discussion. "
            "Focus on the main arguments, notable technical insights, criticism, and any consensus. "
            "Use only information from the Hacker News discussion page. "
            "If the thread cannot be accessed or has no meaningful comments, return a short statement saying that."
        ),
        input=(
            f"Article title: {item.title}\n"
            f"Article URL: {item.link}\n"
            f"Hacker News comments URL: {item.comments_url}\n\n"
            "Retrieve the comments from that Hacker News thread and summarize the conversation for internal article notes."
        ),
    )
    return normalize_text(response.output_text)


def convert_pdf_to_markdown(url: str) -> str:
    converter_cls = require_docling()
    logger.info("Converting PDF to markdown with docling for {}", url)
    converter = converter_cls()
    result = converter.convert(url)
    markdown = result.document.export_to_markdown()
    return markdown.strip()


def parse_article_with_defuddle(defuddle_bin: str, url: str) -> dict[str, Any]:
    result = subprocess.run(
        [defuddle_bin, "parse", "-j", "-m", url],
        check=False,
        capture_output=True,
        text=True,
        timeout=DEFUDDLE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown defuddle error"
        raise RuntimeError(stderr)

    raw_output = result.stdout.strip()
    if not raw_output:
        raise RuntimeError("defuddle returned empty output")

    parsed = json.loads(raw_output)
    if not isinstance(parsed, dict):
        raise RuntimeError("defuddle returned a non-object JSON payload")
    return parsed


def build_record(
    client: OpenAI,
    summary_model: str,
    defuddle_bin: str,
    item: FeedItem,
) -> dict[str, Any]:
    logger.debug("Fetching article for {}", item.link)
    fetched_at = datetime.now(timezone.utc).isoformat()
    fetch_error = ""
    raw_html = ""
    html_status = 0
    html_content_type = ""
    defuddle_json = ""
    article_title = ""
    article_author = ""
    article_excerpt = ""
    article_text = ""
    article_html = ""
    is_pdf = is_probable_pdf_url(item.link)

    try:
        request = urllib.request.Request(item.link, headers={"User-Agent": USER_AGENT})
        context = ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=ARTICLE_TIMEOUT_SECONDS, context=context) as response:
            html_status = int(response.status)
            html_content_type = response.headers.get("content-type", "")
            if is_pdf_content_type(html_content_type):
                is_pdf = True
                response.read()
            else:
                encoding = response.headers.get_content_charset() or "utf-8"
                raw_html = response.read().decode(encoding, errors="replace")
    except Exception as exc:
        fetch_error = f"download error: {type(exc).__name__}: {exc}"

    try:
        if is_pdf:
            article_text = normalize_text(convert_pdf_to_markdown(item.link))
            article_title = item.title
            article_excerpt = article_text[:500]
            logger.info("Stored docling markdown for PDF {}", item.link)
        else:
            parsed = parse_article_with_defuddle(defuddle_bin, item.link)
            defuddle_json = json.dumps(parsed, ensure_ascii=True, sort_keys=True)
            article_title = normalize_text(str(parsed.get("title", "")))
            article_author = normalize_text(str(parsed.get("author") or parsed.get("byline") or ""))
            article_excerpt = normalize_text(str(parsed.get("excerpt", "")))
            article_text = normalize_text(
                str(
                    parsed.get("markdown")
                    or parsed.get("contentMarkdown")
                    or parsed.get("markdownContent")
                    or parsed.get("textContent")
                    or parsed.get("text")
                    or parsed.get("content")
                    or parsed.get("article")
                    or ""
                )
            )
            article_html = str(parsed.get("contentHtml") or parsed.get("html") or "")
    except Exception as exc:
        parser_name = "docling" if is_pdf else "defuddle"
        message = f"{parser_name} error: {type(exc).__name__}: {exc}"
        fetch_error = f"{fetch_error}; {message}".strip("; ") if fetch_error else message
        logger.warning("{} failed for {}: {}", parser_name.capitalize(), item.link, message)
        try:
            summary = summarize_hn_comments_with_web_search(client, summary_model, item)
            if summary:
                article_title = article_title or f"Hacker News discussion: {item.title}"
                article_excerpt = summary[:500]
                article_text = summary
                logger.info("Stored Hacker News summary fallback for {}", item.link)
            else:
                logger.warning("No Hacker News summary available for fallback on {}", item.link)
        except Exception as summary_exc:
            summary_message = (
                f"HN-summary error: {type(summary_exc).__name__}: {summary_exc}"
            )
            fetch_error = f"{fetch_error}; {summary_message}".strip("; ")
            logger.warning("Hacker News summary fallback failed for {}: {}", item.link, summary_message)

    if fetch_error:
        logger.warning("Recorded article with errors for {}: {}", item.link, fetch_error)
    else:
        logger.debug("Parsed article successfully for {}", item.link)

    return {
        "rss_guid": item.rss_guid,
        "article_url": item.link,
        "rss_title": item.title,
        "comments_url": item.comments_url,
        "pub_date": item.pub_date,
        "pub_ts": int(item.pub_ts or 0),
        "rss_description": item.description,
        "fetched_at": fetched_at,
        "html_status": html_status,
        "html_content_type": html_content_type,
        "raw_html": raw_html,
        "defuddle_json": defuddle_json,
        "article_title": article_title,
        "article_author": article_author,
        "article_excerpt": article_excerpt,
        "article_text": article_text,
        "article_html": article_html,
        "fetch_error": fetch_error,
    }


def build_document_text(record: dict[str, Any]) -> str:
    parts = [
        str(record["rss_title"]),
        str(record["article_title"]),
        str(record["rss_description"]),
        str(record["article_excerpt"]),
        str(record["article_text"]),
    ]
    text = "\n\n".join(part for part in parts if part)
    return text[:MAX_DOCUMENT_CHARS]


def embed_texts(client: OpenAI, model: str, inputs: list[str]) -> list[list[float]]:
    logger.info("Requesting {} embeddings from model {}", len(inputs), model)
    response = client.embeddings.create(model=model, input=inputs, encoding_format="float")
    return [list(item.embedding) for item in response.data]


def open_or_create_collection(path: Path, dimension: int):
    require_zvec()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        logger.info("Opening existing Zvec collection at {}", path)
        return zvec.open(str(path))

    schema = zvec.CollectionSchema(
        name="hn_rss_articles",
        fields=[
            zvec.FieldSchema("article_url", zvec.DataType.STRING),
            zvec.FieldSchema("rss_guid", zvec.DataType.STRING),
            zvec.FieldSchema("rss_title", zvec.DataType.STRING),
            zvec.FieldSchema("comments_url", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("pub_date", zvec.DataType.STRING),
            zvec.FieldSchema("pub_ts", zvec.DataType.INT64, index_param=zvec.InvertIndexParam()),
            zvec.FieldSchema("rss_description", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("fetched_at", zvec.DataType.STRING),
            zvec.FieldSchema("html_status", zvec.DataType.INT64),
            zvec.FieldSchema("html_content_type", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("raw_html", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("defuddle_json", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("article_title", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("article_author", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("article_excerpt", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("article_text", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("article_html", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("fetch_error", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("content_hash", zvec.DataType.STRING),
            zvec.FieldSchema("embed_model", zvec.DataType.STRING),
        ],
        vectors=[
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension,
                index_param=zvec.FlatIndexParam(zvec.MetricType.COSINE),
            )
        ],
    )
    logger.info("Creating Zvec collection at {} with dimension {}", path, dimension)
    return zvec.create_and_open(str(path), schema)


def has_successful_data(existing) -> bool:
    if existing is None:
        return False
    article_text = normalize_text(str(existing.fields.get("article_text", "")))
    return bool(article_text)


def split_items_by_existing_success(collection, items: list[FeedItem]) -> tuple[list[FeedItem], int]:
    to_process: list[FeedItem] = []
    skipped = 0

    for batch in chunked(items, EMBED_BATCH_SIZE):
        ids = [doc_id_for_url(item.link) for item in batch]
        existing_docs = collection.fetch(ids)
        for item in batch:
            existing = existing_docs.get(doc_id_for_url(item.link))
            if has_successful_data(existing):
                skipped += 1
                logger.info("Skipping previously successful article {}", item.link)
            else:
                to_process.append(item)

    return to_process, skipped


def upsert_records_to_collection(collection, client: OpenAI, model: str, records: list[dict[str, Any]]) -> None:
    logger.info("Preparing {} records for Zvec upsert", len(records))
    docs_to_embed: list[tuple[dict[str, Any], str, str, str]] = []

    for batch in chunked(records, EMBED_BATCH_SIZE):
        ids = [doc_id_for_url(str(record["article_url"])) for record in batch]
        existing_docs = collection.fetch(ids)

        for record in batch:
            article_url = str(record["article_url"])
            document_text = build_document_text(record)
            content_hash = sha256_text(document_text)
            doc_id = doc_id_for_url(article_url)
            existing = existing_docs.get(doc_id)

            if (
                existing is not None
                and existing.fields.get("content_hash") == content_hash
                and existing.fields.get("embed_model") == model
            ):
                continue

            docs_to_embed.append((record, doc_id, document_text, content_hash))

    logger.info("Embedding and upserting {} changed/new records", len(docs_to_embed))
    for batch in chunked(docs_to_embed, EMBED_BATCH_SIZE):
        texts = [document_text for _, _, document_text, _ in batch]
        embeddings = embed_texts(client, model, texts)
        docs = []
        for (record, doc_id, _, content_hash), embedding in zip(batch, embeddings):
            docs.append(
                zvec.Doc(
                    id=doc_id,
                    vectors={"embedding": embedding},
                    fields={
                        "article_url": str(record["article_url"]),
                        "rss_guid": str(record["rss_guid"]),
                        "rss_title": str(record["rss_title"]),
                        "comments_url": str(record["comments_url"]),
                        "pub_date": str(record["pub_date"]),
                        "pub_ts": int(record["pub_ts"]),
                        "rss_description": str(record["rss_description"]),
                        "fetched_at": str(record["fetched_at"]),
                        "html_status": int(record["html_status"]),
                        "html_content_type": str(record["html_content_type"]),
                        "raw_html": str(record["raw_html"]),
                        "defuddle_json": str(record["defuddle_json"]),
                        "article_title": str(record["article_title"]),
                        "article_author": str(record["article_author"]),
                        "article_excerpt": str(record["article_excerpt"]),
                        "article_text": str(record["article_text"]),
                        "article_html": str(record["article_html"]),
                        "fetch_error": str(record["fetch_error"]),
                        "content_hash": content_hash,
                        "embed_model": model,
                    },
                )
            )
        collection.upsert(docs)
        logger.info("Upserted batch of {} records into Zvec", len(docs))

    collection.flush()
    logger.info("Flushed Zvec collection to disk")


def run(
    rss_url: str,
    collection_path: Path,
    model: str,
    summary_model: str,
    workers: int,
) -> tuple[int, int, Path]:
    logger.info(
        "Starting RSS ingest rss_url={} collection={} embed_model={} summary_model={} workers={}",
        rss_url,
        collection_path,
        model,
        summary_model,
        workers,
    )
    client = require_openai_client()
    defuddle_bin = require_defuddle()
    items = load_feed_items(rss_url)
    query_embedding = embed_texts(client, model, ["seed collection creation"])[0]
    collection = open_or_create_collection(collection_path, len(query_embedding))
    items_to_process, skipped = split_items_by_existing_success(collection, items)
    logger.info(
        "Found {} total RSS items, skipping {} already successful items, processing {}",
        len(items),
        skipped,
        len(items_to_process),
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(
            executor.map(
                lambda item: build_record(client, summary_model, defuddle_bin, item),
                items_to_process,
            )
        )
    logger.info("Built {} article records", len(records))

    upsert_records_to_collection(collection, client, model, records)

    failures = sum(1 for record in records if record["fetch_error"])
    logger.info(
        "Ingest complete: {} records processed, {} skipped, {} with errors",
        len(records),
        skipped,
        failures,
    )
    return len(records), failures, collection_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the Hacker News RSS feed, parse each linked article, embed it, and store everything directly in Zvec."
    )
    parser.add_argument("--rss-url", default=HN_RSS_URL)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--summary-model", default=DEFAULT_SUMMARY_MODEL)
    parser.add_argument("--collection")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    collection_path = (
        Path(args.collection).expanduser().resolve()
        if args.collection
        else zvec_path_for_model(args.model)
    )
    count, failures, stored_path = run(
        args.rss_url,
        collection_path,
        args.model,
        args.summary_model,
        max(args.workers, 1),
    )
    print(f"Stored {count} Hacker News items in {stored_path}")
    if failures:
        print(f"{failures} items recorded download/parse errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
