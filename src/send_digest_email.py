#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from dotenv import load_dotenv
from markdown import markdown as markdown_to_html

DEFAULT_INPUT_PATH = "-"
DEFAULT_TO_EMAIL = "pjbutler@gmail.com"
HTTP_USER_AGENT = "hn-ai-digest-bot/2.0 (+https://news.ycombinator.com/)"
CONFIG_DIR = Path.home() / ".config" / "hn-ai-digest"
ENV_FILE = CONFIG_DIR / "env"
MAX_EMAIL_EXCERPT_CHARS = 1000


def infer_input_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    return "xml"


def infer_input_format_from_name(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".json":
        return "json"
    return "xml"


def load_xml_digest(path: Path) -> tuple[list[dict[str, str]], str]:
    tree = ET.parse(path)
    root = tree.getroot()
    items: list[dict[str, str]] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        description = (item.findtext("description") or "").strip()
        if title and link:
            items.append(
                {
                    "title": title,
                    "link": link,
                    "pubDate": pub,
                    "description": description,
                    "comments_url": "",
                    "article_title": "",
                    "article_excerpt": description,
                    "semantic_score": "",
                }
            )
    return items, path.read_text(encoding="utf-8")


def load_json_digest(path: Path) -> tuple[list[dict[str, str]], str]:
    raw_items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list):
        raise RuntimeError("JSON digest must be a list of result objects")

    items: list[dict[str, str]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("rss_title") or "").strip()
        link = str(raw.get("article_url") or "").strip()
        if not title or not link:
            continue
        items.append(
            {
                "title": title,
                "link": link,
                "pubDate": str(raw.get("pub_date") or "").strip(),
                "description": str(raw.get("article_excerpt") or raw.get("article_text") or "").strip(),
                "comments_url": str(raw.get("comments_url") or "").strip(),
                "article_title": str(raw.get("article_title") or "").strip(),
                "article_excerpt": str(raw.get("article_excerpt") or "").strip(),
                "semantic_score": str(raw.get("semantic_score") or "").strip(),
            }
        )
    return items, path.read_text(encoding="utf-8")


def load_digest(path: Path, input_format: str) -> tuple[list[dict[str, str]], str]:
    if input_format == "json":
        return load_json_digest(path)
    if input_format == "xml":
        return load_xml_digest(path)
    raise RuntimeError(f"Unsupported input format: {input_format}")


def load_xml_digest_text(xml_text: str) -> tuple[list[dict[str, str]], str]:
    root = ET.fromstring(xml_text)
    items: list[dict[str, str]] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        description = (item.findtext("description") or "").strip()
        if title and link:
            items.append(
                {
                    "title": title,
                    "link": link,
                    "pubDate": pub,
                    "description": description,
                    "comments_url": "",
                    "article_title": "",
                    "article_excerpt": description,
                    "semantic_score": "",
                }
            )
    return items, xml_text


def load_json_digest_text(json_text: str) -> tuple[list[dict[str, str]], str]:
    raw_items = json.loads(json_text)
    if not isinstance(raw_items, list):
        raise RuntimeError("JSON digest must be a list of result objects")

    items: list[dict[str, str]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("rss_title") or "").strip()
        link = str(raw.get("article_url") or "").strip()
        if not title or not link:
            continue
        items.append(
            {
                "title": title,
                "link": link,
                "pubDate": str(raw.get("pub_date") or "").strip(),
                "description": str(raw.get("article_excerpt") or raw.get("article_text") or "").strip(),
                "comments_url": str(raw.get("comments_url") or "").strip(),
                "article_title": str(raw.get("article_title") or "").strip(),
                "article_excerpt": str(raw.get("article_excerpt") or "").strip(),
                "semantic_score": str(raw.get("semantic_score") or "").strip(),
            }
        )
    return items, json_text


def load_digest_text(text: str, input_format: str) -> tuple[list[dict[str, str]], str]:
    if input_format == "json":
        return load_json_digest_text(text)
    if input_format == "xml":
        return load_xml_digest_text(text)
    raise RuntimeError(f"Unsupported input format: {input_format}")


def format_subject(now: datetime, count: int) -> str:
    return f"HN AI 24h Digest - {now.strftime('%Y-%m-%d')} ({count} items)"


def truncate_excerpt(text: str, limit: int = MAX_EMAIL_EXCERPT_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    split_at = clipped.rfind(" ")
    if split_at > max(200, limit // 2):
        clipped = clipped[:split_at]
    return clipped.rstrip() + "..."


def constrain_email_images(html_text: str) -> str:
    return html_text.replace(
        "<img ",
        '<img style="max-width:100%;height:auto;display:block;margin:12px 0;" ',
    )


def format_text_body(items: list[dict[str, str]]) -> str:
    lines = [
        "Hacker News AI/LLM/ML 24-hour digest",
        "",
        f"Item count: {len(items)}",
        "",
    ]
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {item['title']}")
        if item.get("article_title") and item["article_title"] != item["title"]:
            lines.append(f"   Article title: {item['article_title']}")
        lines.append(f"   Published: {item.get('pubDate', '')}")
        lines.append(f"   URL: {item['link']}")
        if item.get("semantic_score"):
            lines.append(f"   Score: {item['semantic_score']}")
        if item.get("comments_url"):
            lines.append(f"   Comments: {item['comments_url']}")
        if item.get("description"):
            lines.append(f"   Excerpt: {truncate_excerpt(item['description'])}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_item_html(item: dict[str, str]) -> str:
    title_html = html.escape(item["title"])
    link_html = html.escape(item["link"])
    article_title = item.get("article_title", "").strip()
    pub_date = html.escape(item.get("pubDate", ""))
    score = html.escape(item.get("semantic_score", ""))
    comments_url = item.get("comments_url", "").strip()
    excerpt = truncate_excerpt(item.get("description", ""))
    excerpt_html = markdown_to_html(excerpt, output_format="html") if excerpt else ""
    excerpt_html = constrain_email_images(excerpt_html)

    parts = [
        '<article style="margin:0 0 28px 0;padding:0 0 24px 0;border-bottom:1px solid #ddd;">',
        f'<h2 style="margin:0 0 8px 0;font-size:20px;"><a href="{link_html}">{title_html}</a></h2>',
    ]
    if article_title and article_title != item["title"]:
        parts.append(
            f'<div style="margin:0 0 8px 0;color:#444;font-size:14px;">Article title: {html.escape(article_title)}</div>'
        )
    meta_bits = [bit for bit in [pub_date, f"score {score}" if score else ""] if bit]
    if meta_bits:
        parts.append(
            f'<div style="margin:0 0 10px 0;color:#666;font-size:13px;">{" | ".join(meta_bits)}</div>'
        )
    if comments_url:
        parts.append(
            f'<div style="margin:0 0 10px 0;font-size:13px;"><a href="{html.escape(comments_url)}">Hacker News discussion</a></div>'
        )
    if excerpt_html:
        parts.append(f'<div style="font-size:15px;line-height:1.5;">{excerpt_html}</div>')
    parts.append("</article>")
    return "\n".join(parts)


def format_html_body(items: list[dict[str, str]], subject: str) -> str:
    rendered_items = "\n".join(render_item_html(item) for item in items)
    return f"""\
<!doctype html>
<html>
  <body style="margin:0;padding:24px;background:#f6f6f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#111;">
    <div style="max-width:900px;margin:0 auto;background:#ffffff;border:1px solid #e5e5e5;padding:32px;">
      <h1 style="margin:0 0 8px 0;font-size:28px;">{html.escape(subject)}</h1>
      <p style="margin:0 0 28px 0;color:#555;">Semantic digest of recent Hacker News articles relevant to AI, GPUs, and adjacent topics.</p>
      {rendered_items}
    </div>
  </body>
</html>
"""


def send_via_resend(
    api_key: str, from_email: str, to_email: str, subject: str, text_body: str, html_body: str
) -> None:
    payload = {
        "from": from_email,
        "to": [to_email],
        "subject": subject,
        "text": text_body,
        "html": html_body,
    }
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": HTTP_USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context()) as resp:
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"Resend API failed with status {resp.status}")
    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Resend API HTTP {e.code}: {details}") from e


def send_via_sendgrid(
    api_key: str, from_email: str, to_email: str, subject: str, text_body: str, html_body: str
) -> None:
    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": from_email},
        "subject": subject,
        "content": [
            {"type": "text/plain", "value": text_body},
            {"type": "text/html", "value": html_body},
        ],
    }
    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context()) as resp:
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"SendGrid API failed with status {resp.status}")
    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SendGrid API HTTP {e.code}: {details}") from e


def validate_items_window(items: list[dict[str, str]]) -> None:
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - 24 * 3600
    stale = 0
    for item in items:
        raw = item.get("pubDate", "")
        if not raw:
            continue
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt.timestamp() < cutoff:
                stale += 1
        except Exception:
            continue
    if stale > 0:
        print(f"Warning: {stale} items have pubDate older than 24 hours", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Send HN digest email from XML or JSON input")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--format", choices=["auto", "xml", "json"], default="auto")
    parser.add_argument("--to", default=DEFAULT_TO_EMAIL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ENV_FILE)

    if args.input == "-":
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            print("Digest input missing on stdin", file=sys.stderr)
            return 2
        input_format = "json" if args.format == "auto" else args.format
        items, _ = load_digest_text(raw_input, input_format)
    else:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            print(f"Digest input missing: {input_path}", file=sys.stderr)
            return 2
        input_format = (
            infer_input_format_from_name(args.input) if args.format == "auto" else args.format
        )
        items, _ = load_digest(input_path, input_format)

    if not items:
        print("Digest contains 0 items; refusing to send empty digest", file=sys.stderr)
        return 3

    validate_items_window(items)

    now = datetime.now(timezone.utc)
    subject = format_subject(now, len(items))
    text_body = format_text_body(items)
    html_body = format_html_body(items, subject)

    provider = os.getenv("DIGEST_EMAIL_PROVIDER", "resend").strip().lower()
    from_email = os.getenv("DIGEST_FROM_EMAIL", "HN Digest <onboarding@resend.dev>")

    if args.dry_run:
        print(f"Dry run: provider={provider} to={args.to} subject={subject} format={input_format}")
        print("Text preview:")
        print("\n".join(text_body.splitlines()[:20]))
        print("\nHTML preview:")
        print("\n".join(html_body.splitlines()[:20]))
        return 0

    if provider == "resend":
        api_key = os.getenv("RESEND_API_KEY", "")
        if not api_key:
            print("Missing RESEND_API_KEY", file=sys.stderr)
            return 4
        send_via_resend(api_key, from_email, args.to, subject, text_body, html_body)
    elif provider == "sendgrid":
        api_key = os.getenv("SENDGRID_API_KEY", "")
        if not api_key:
            print("Missing SENDGRID_API_KEY", file=sys.stderr)
            return 5
        send_via_sendgrid(api_key, from_email, args.to, subject, text_body, html_body)
    else:
        print("Unsupported DIGEST_EMAIL_PROVIDER. Use 'resend' or 'sendgrid'.", file=sys.stderr)
        return 6

    print(f"Email sent via {provider} to {args.to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
