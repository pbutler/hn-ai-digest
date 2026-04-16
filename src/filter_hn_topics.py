#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from markdown import markdown as markdown_to_html

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

try:
    import zvec
except ImportError:  # pragma: no cover
    zvec = None  # type: ignore[assignment]

BASE_DIR = Path("/Users/pbutler/Documents/HackerNews")
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_LIMIT = 15

INTEREST_PROFILE = """
Articles about GPU programming, CUDA, ROCm, Metal shaders, parallel computing,
LLMs, large language models, generative AI, artificial intelligence, deep learning,
transformer-based models, attention mechanisms, neural networks, multimodal systems,
model inference, training, fine-tuning, embeddings, retrieval-augmented generation,
AI infrastructure, model tooling, PyTorch, JAX, MLX, diffusion models, and adjacent topics.
"""


def configure_logging(verbose: bool) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def require_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("Missing dependency: install the 'openai' Python package")
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")
    logger.debug("Loaded OpenAI API key from environment")
    return OpenAI(api_key=api_key)


def require_zvec() -> None:
    if zvec is None:
        raise RuntimeError("Missing dependency: install the 'zvec' Python package")


def zvec_path_for_model(model: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-") or "embedding-model"
    return BASE_DIR / "output" / f"hn-rss-articles-{slug}.zvec"


def embed_texts(client: OpenAI, model: str, inputs: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=inputs, encoding_format="float")
    return [list(item.embedding) for item in response.data]


def open_collection(path: Path):
    require_zvec()
    if not path.exists():
        raise RuntimeError(f"Zvec collection does not exist: {path}")
    logger.info("Opening Zvec collection at {}", path)
    return zvec.open(str(path))


def query_top_articles(collection, query_embedding: list[float], cutoff_ts: int, limit: int):
    logger.info("Querying Zvec for top {} articles with cutoff {}", limit, cutoff_ts)
    return collection.query(
        zvec.VectorQuery("embedding", vector=query_embedding),
        topk=limit,
        filter=f"pub_ts >= {cutoff_ts}",
        output_fields=[
            "article_url",
            "rss_guid",
            "rss_title",
            "article_title",
            "comments_url",
            "pub_date",
            "article_author",
            "article_excerpt",
            "article_text",
            "fetch_error",
            "embed_model",
        ],
    )


def normalize_results(docs) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for doc in docs:
        fields = doc.fields
        article_excerpt = str(fields.get("article_excerpt", ""))
        article_text = str(fields.get("article_text", ""))
        if len(article_text.strip()) > len(article_excerpt.strip()):
            article_excerpt = article_text
        results.append(
            {
                "rss_guid": fields.get("rss_guid", ""),
                "rss_title": fields.get("rss_title", ""),
                "article_title": fields.get("article_title", ""),
                "article_url": fields.get("article_url", ""),
                "comments_url": fields.get("comments_url", ""),
                "pub_date": fields.get("pub_date", ""),
                "article_author": fields.get("article_author", ""),
                "article_excerpt": article_excerpt,
                "article_text": article_text,
                "semantic_score": round(float(doc.score or 0.0), 6),
                "fetch_error": fields.get("fetch_error", ""),
                "embed_model": fields.get("embed_model", ""),
            }
        )
    return results


def build_rss_xml(results: list[dict[str, object]], hours: int, model: str) -> str:
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = f"Hacker News Semantic Filter ({model})"
    ET.SubElement(channel, "link").text = "https://news.ycombinator.com/rss"
    ET.SubElement(channel, "description").text = (
        f"Top semantically relevant Hacker News articles from the last {hours} hours."
    )
    ET.SubElement(channel, "lastBuildDate").text = format_datetime(datetime.now(timezone.utc))

    for item in results:
        node = ET.SubElement(channel, "item")
        ET.SubElement(node, "title").text = str(item["rss_title"])
        ET.SubElement(node, "link").text = str(item["article_url"])
        ET.SubElement(node, "guid").text = str(item["article_url"])
        ET.SubElement(node, "pubDate").text = str(item["pub_date"])
        description = str(item.get("article_excerpt") or item.get("article_text") or "")
        if item.get("semantic_score") is not None:
            description = f"[score: {float(item['semantic_score']):.6f}] {description}".strip()
        description_html = markdown_to_html(description, output_format="html")
        ET.SubElement(node, "description").text = description_html

    return ET.tostring(rss, encoding="utf-8", xml_declaration=True).decode("utf-8")


def print_text(results: list[dict[str, object]], hours: int, model: str, zvec_path: Path) -> None:
    if not results:
        print(f"No matching articles found in the last {hours} hours.")
        return

    print(
        f"Top {len(results)} semantically relevant articles from the last {hours} hours "
        f"using {model} via Zvec:"
    )
    print(f"Zvec collection: {zvec_path}")
    for index, item in enumerate(results, start=1):
        print()
        print(f"{index}. {item['rss_title']}")
        if item["article_title"] and item["article_title"] != item["rss_title"]:
            print(f"   Article title: {item['article_title']}")
        print(f"   Relevance: {item['semantic_score']:.6f}")
        print(f"   Published: {item['pub_date']}")
        print(f"   URL: {item['article_url']}")
        if item["comments_url"]:
            print(f"   Comments: {item['comments_url']}")
        if item["article_excerpt"]:
            print(f"   Excerpt: {str(item['article_excerpt'])[:280]}")
        if item["fetch_error"]:
            print(f"   Note: {item['fetch_error']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank stored Hacker News articles from a Zvec collection with OpenAI embeddings."
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--collection")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--rss", action="store_true")
    parser.add_argument("--rss-output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    logger.info(
        "Starting semantic search hours={} limit={} model={}",
        max(args.hours, 1),
        max(args.limit, 1),
        args.model,
    )
    client = require_openai_client()
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=max(args.hours, 1))).timestamp())
    collection_path = (
        Path(args.collection).expanduser().resolve()
        if args.collection
        else zvec_path_for_model(args.model)
    )

    logger.info("Embedding semantic query with model {}", args.model)
    query_embedding = embed_texts(client, args.model, [INTEREST_PROFILE])[0]
    collection = open_collection(collection_path)
    results = normalize_results(
        query_top_articles(collection, query_embedding, cutoff_ts, max(args.limit, 1))
    )
    logger.info("Semantic search returned {} results", len(results))

    if args.json and args.rss:
        raise RuntimeError("Use only one of --json or --rss")

    if args.json:
        print(json.dumps(results, ensure_ascii=True, indent=2))
    elif args.rss:
        rss_xml = build_rss_xml(results, max(args.hours, 1), args.model)
        if args.rss_output:
            output_path = Path(args.rss_output).expanduser().resolve()
            output_path.write_text(rss_xml, encoding="utf-8")
            logger.info("Wrote RSS output to {}", output_path)
            print(output_path)
        else:
            print(rss_xml)
    else:
        print_text(results, max(args.hours, 1), args.model, collection_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
