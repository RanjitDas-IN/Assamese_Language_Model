#!/usr/bin/env python3
# scrapping_Xahitya_org.py
#
# Usage:
#   python scrapping_Xahitya_org.py --out xahitya_dump
#   python scrapping_Xahitya_org.py --method auto --delay 1.0 --max-posts 0
#
# Notes:
# - Uses a Firefox-like user agent by default.
# - Tries WordPress REST API first.
# - Falls back to HTML crawling if the REST API is blocked or unavailable.
# - Writes:
#     1) output/articles.jsonl
#     2) output/corpus.txt
# - Shows tqdm progress bars for discovery, fetching, and saving.

#python scripts/web_scrapping/scrapping_Xahitya_org.py --out xahitya_dump --method auto --delay 1.0

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


BASE_URL = "https://xahitya.org/"
DOMAIN = urlparse(BASE_URL).netloc

FIREFOX_UA = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0"
)

EXCLUDE_PATH_RE = re.compile(
    r"("
    r"/wp-admin/|/wp-login\.php|/wp-json/|/xmlrpc\.php|"
    r"/feed/?$|/comments/|/page/\d+/?$|"
    r"/tag/|/author/|/category/|/search/|"
    r"/privacy-policy|/about-us/?$|"
    r"\.jpg$|\.jpeg$|\.png$|\.gif$|\.webp$|\.pdf$|\.zip$"
    r")",
    re.IGNORECASE,
)

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\uFEFF]")
WHITESPACE_RE = re.compile(r"[ \t]+")
MANY_BLANK_LINES_RE = re.compile(r"\n{3,}")


@dataclass
class Post:
    url: str
    title: str
    date: str
    text: str
    source: str = "unknown"


def make_session(timeout_retries: int = 4) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": FIREFOX_UA,
            "Accept-Language": "en-US,en;q=0.9,as;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    retries = Retry(
        total=timeout_retries,
        connect=timeout_retries,
        read=timeout_retries,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def normalize_url(url: str) -> str:
    p = urlparse(url)
    scheme = p.scheme or "https"
    netloc = p.netloc.lower()
    path = re.sub(r"/{2,}", "/", p.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    # Remove fragments and keep query only when needed
    return urlunparse((scheme, netloc, path, "", p.query, ""))


def same_domain(url: str) -> bool:
    try:
        return urlparse(url).netloc.lower() == DOMAIN
    except Exception:
        return False


def safe_get(session: requests.Session, url: str, delay: float = 0.0, timeout: int = 30) -> Optional[requests.Response]:
    if delay > 0:
        time.sleep(delay)
    try:
        resp = session.get(url, timeout=timeout)
        return resp
    except requests.RequestException:
        return None


def is_probably_content_url(url: str) -> bool:
    p = urlparse(url)
    path = p.path.lower()

    if not same_domain(url):
        return False
    if EXCLUDE_PATH_RE.search(path):
        return False
    if path in ("", "/"):
        return False

    # Favor article-like permalinks, but still allow unknown slugs.
    return True


def clean_text(raw: str) -> str:
    if not raw:
        return ""

    text = unicodedata.normalize("NFKC", raw)
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = MANY_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def extract_title_date_text_from_html(html: str, url: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(html, "html.parser")

    # Prefer article / entry-content if present
    article = (
        soup.select_one("article")
        or soup.select_one("div.entry-content")
        or soup.select_one("main")
        or soup.body
    )

    title = ""
    date = ""

    # title
    h1 = soup.select_one("h1.entry-title") or soup.select_one("h1")
    if h1:
        title = h1.get_text(" ", strip=True)

    # date
    time_el = soup.select_one("time.entry-date") or soup.find("time")
    if time_el:
        date = time_el.get_text(" ", strip=True)

    # content extraction
    parts: list[str] = []

    if article:
        for tag in article.find_all(["script", "style", "noscript", "iframe"]):
            tag.decompose()

        # Prefer block-like text nodes to keep paragraphs readable
        for el in article.find_all(["p", "h2", "h3", "li", "blockquote", "pre"]):
            txt = el.get_text(" ", strip=True)
            if txt:
                parts.append(txt)

        if not parts:
            txt = article.get_text("\n", strip=True)
            if txt:
                parts = [line.strip() for line in txt.split("\n") if line.strip()]

    text = clean_text("\n\n".join(parts))

    # Fallbacks if markup is odd
    if not title:
        title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()
        if title.endswith(" - সাহিত্য ডট অৰ্গ"):
            title = title[: -len(" - সাহিত্য ডট অৰ্গ")].strip()

    return clean_text(title), clean_text(date), text


def discover_internal_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        absolute = normalize_url(urljoin(base_url, href))
        if same_domain(absolute):
            links.append(absolute)
    return links


def discover_seed_urls(session: requests.Session, delay: float = 0.0) -> list[str]:
    """
    Pull homepage links and keep the category-ish / content-ish internal URLs.
    This is mostly for HTML fallback and for discovery if REST API fails.
    """
    resp = safe_get(session, BASE_URL, delay=delay)
    if not resp or resp.status_code >= 400:
        return [normalize_url(BASE_URL)]

    html = resp.text
    links = discover_internal_links(html, BASE_URL)

    seeds = set([normalize_url(BASE_URL)])
    for link in links:
        p = urlparse(link)
        path = p.path.lower()

        # Keep navigation categories and archive/list pages.
        if (
            path in ("/", "")
            or path.startswith("/page/")
            or path.startswith("/20")  # year-based archives
            or "stories-all-type" in path
            or path.endswith((
                "/about-us",
                "/author",
            ))
        ):
            seeds.add(link)

        # Keep likely category or content URLs discovered from the homepage
        if is_probably_content_url(link):
            # Skip obviously non-content pages
            if not any(x in path for x in ["/wp-content/", "/tag/", "/search/"]):
                seeds.add(link)

    return sorted(seeds)


def fetch_wp_rest_posts(
    session: requests.Session,
    delay: float = 0.0,
    max_posts: int = 0,
) -> list[Post]:
    """
    Pull posts via WordPress REST API:
    /wp-json/wp/v2/posts
    """
    endpoint = urljoin(BASE_URL, "/wp-json/wp/v2/posts")
    posts: list[Post] = []

    first = safe_get(
        session,
        f"{endpoint}?per_page=100&page=1&_fields=link,title,date,content",
        delay=delay,
    )
    if not first or first.status_code >= 400:
        return []

    total_pages = int(first.headers.get("X-WP-TotalPages", "1") or "1")
    page_num = 1

    pbar = tqdm(total=total_pages, desc="REST pages", unit="page")
    while True:
        if page_num == 1:
            resp = first
        else:
            resp = safe_get(
                session,
                f"{endpoint}?per_page=100&page={page_num}&_fields=link,title,date,content",
                delay=delay,
            )

        if not resp or resp.status_code >= 400:
            break

        try:
            data = resp.json()
        except Exception:
            break

        if not isinstance(data, list) or not data:
            break

        fetch_bar = tqdm(data, desc=f"Parsing REST posts p{page_num}", unit="post", leave=False)
        for item in fetch_bar:
            link = normalize_url(item.get("link", ""))
            title_html = item.get("title", {}).get("rendered", "")
            date = item.get("date", "")
            content_html = item.get("content", {}).get("rendered", "")

            title = BeautifulSoup(title_html, "html.parser").get_text(" ", strip=True)
            _, _, text = extract_title_date_text_from_html(
                f"<html><body>{content_html}</body></html>", link
            )

            if text:
                posts.append(Post(url=link, title=title, date=date, text=text, source="wp-rest"))

            if max_posts and len(posts) >= max_posts:
                fetch_bar.close()
                pbar.update(1)
                pbar.close()
                return posts

        pbar.update(1)
        page_num += 1

        if page_num > total_pages:
            break

    pbar.close()
    return posts


def crawl_html_for_article_urls(
    session: requests.Session,
    seeds: Iterable[str],
    delay: float = 0.0,
    max_pages: int = 0,
) -> list[str]:
    """
    Crawl list/archive/category pages and collect article URLs.
    """
    queue = deque(normalize_url(u) for u in seeds)
    seen_pages: set[str] = set()
    article_urls: set[str] = set()

    pbar = tqdm(desc="Crawling list/archive pages", unit="page")
    while queue:
        page_url = queue.popleft()
        if page_url in seen_pages:
            continue
        seen_pages.add(page_url)

        # Only keep list-like pages for discovery
        resp = safe_get(session, page_url, delay=delay)
        pbar.update(1)

        if not resp or resp.status_code >= 400:
            continue

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        # 1) Article links, usually inside h2/h3 / entry-title on WordPress pages
        for a in soup.select("h1 a[href], h2 a[href], h3 a[href], .entry-title a[href], .post-title a[href]"):
            href = a.get("href", "").strip()
            if not href:
                continue
            abs_url = normalize_url(urljoin(page_url, href))
            if is_probably_content_url(abs_url):
                article_urls.add(abs_url)

        # 2) Fallback: any internal link that looks like a post
        for a in soup.select("a[href]"):
            href = a.get("href", "").strip()
            if not href:
                continue
            abs_url = normalize_url(urljoin(page_url, href))
            if is_probably_content_url(abs_url):
                path = urlparse(abs_url).path.lower()
                if not any(x in path for x in ["/category/", "/tag/", "/author/", "/page/"]):
                    # avoid turning every menu item into a content target
                    if re.search(r"/\d{4}/\d{2}/", path) or len(path.strip("/").split("/")) >= 1:
                        article_urls.add(abs_url)

        # 3) Pagination / older posts
        next_candidates = []

        # rel=next is ideal when present
        for a in soup.select('a[rel="next"][href]'):
            next_candidates.append(a.get("href", "").strip())

        # WordPress older-posts links / localized text
        for a in soup.select("a[href]"):
            txt = a.get_text(" ", strip=True)
            if txt and ("আগৰ" in txt or "older" in txt.lower() or "previous" in txt.lower()):
                next_candidates.append(a.get("href", "").strip())

        for href in next_candidates:
            if not href:
                continue
            abs_url = normalize_url(urljoin(page_url, href))
            if same_domain(abs_url) and abs_url not in seen_pages:
                queue.append(abs_url)

        if max_pages and len(seen_pages) >= max_pages:
            break

    pbar.close()
    return sorted(article_urls)


def fetch_html_posts(
    session: requests.Session,
    article_urls: Iterable[str],
    delay: float = 0.0,
    max_posts: int = 0,
) -> list[Post]:
    posts: list[Post] = []
    seen_hashes: set[str] = set()

    article_urls = list(dict.fromkeys(normalize_url(u) for u in article_urls))
    pbar = tqdm(article_urls, desc="Fetching article pages", unit="post")
    for url in pbar:
        resp = safe_get(session, url, delay=delay)
        if not resp or resp.status_code >= 400:
            continue

        title, date, text = extract_title_date_text_from_html(resp.text, url)
        if not text:
            continue

        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        posts.append(Post(url=url, title=title, date=date, text=text, source="html"))
        pbar.set_postfix_str(title[:24])

        if max_posts and len(posts) >= max_posts:
            break

    return posts


def write_outputs(posts: list[Post], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "articles.jsonl"
    txt_path = out_dir / "corpus.txt"

    # JSONL
    save_jsonl = tqdm(posts, desc="Writing JSONL", unit="post")
    with jsonl_path.open("w", encoding="utf-8") as f_jsonl:
        for post in save_jsonl:
            f_jsonl.write(
                json.dumps(
                    {
                        "url": post.url,
                        "title": post.title,
                        "date": post.date,
                        "source": post.source,
                        "text": post.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Plain text corpus
    save_txt = tqdm(posts, desc="Writing TXT", unit="post")
    with txt_path.open("w", encoding="utf-8") as f_txt:
        for post in save_txt:
            block = []
            if post.title:
                block.append(post.title)
            if post.date:
                block.append(post.date)
            if post.url:
                block.append(post.url)
            block.append(post.text)
            f_txt.write("\n".join(block).strip() + "\n\n" + ("=" * 80) + "\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape xahitya.org into JSONL and TXT.")
    parser.add_argument("--out", default="xahitya_dump", help="Output folder")
    parser.add_argument(
        "--method",
        choices=["auto", "api", "html"],
        default="auto",
        help="Scraping method",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.6,
        help="Delay between requests in seconds",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=0,
        help="Stop after this many posts (0 = no limit)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Max HTML discovery pages to crawl (0 = no limit)",
    )
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        help="Extra seed URL(s) for HTML crawl. Can be repeated.",
    )
    args = parser.parse_args()

    session = make_session()
    out_dir = Path(args.out)

    posts: list[Post] = []

    if args.method in ("auto", "api"):
        tqdm.write("[i] Trying WordPress REST API first...")
        posts = fetch_wp_rest_posts(
            session=session,
            delay=args.delay,
            max_posts=args.max_posts,
        )

        if posts:
            tqdm.write(f"[✓] REST API worked: {len(posts)} posts fetched.")
        elif args.method == "api":
            tqdm.write("[x] REST API failed or returned no posts.")
            return 1

    if not posts and args.method in ("auto", "html"):
        tqdm.write("[i] Falling back to HTML crawl...")

        seeds = discover_seed_urls(session, delay=args.delay)
        if args.seed:
            seeds.extend(args.seed)

        # Keep only same-domain seeds
        seeds = sorted(set(normalize_url(u) for u in seeds if same_domain(u)))

        tqdm.write(f"[i] Seed URLs discovered: {len(seeds)}")

        article_urls = crawl_html_for_article_urls(
            session=session,
            seeds=seeds,
            delay=args.delay,
            max_pages=args.max_pages,
        )

        tqdm.write(f"[i] Article URLs discovered: {len(article_urls)}")

        posts = fetch_html_posts(
            session=session,
            article_urls=article_urls,
            delay=args.delay,
            max_posts=args.max_posts,
        )

        tqdm.write(f"[✓] HTML crawl finished: {len(posts)} posts fetched.")

    if not posts:
        tqdm.write("[x] No posts collected.")
        return 1

    # Final dedupe by normalized text hash
    deduped: list[Post] = []
    seen: set[str] = set()
    dedupe_bar = tqdm(posts, desc="Deduplicating", unit="post")
    for post in dedupe_bar:
        key = hashlib.sha1(clean_text(post.text).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(post)

    tqdm.write(f"[i] Unique posts: {len(deduped)}")

    write_outputs(deduped, out_dir)
    tqdm.write(f"[✓] Saved to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




