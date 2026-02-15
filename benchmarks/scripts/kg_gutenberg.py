"""
Gutenberg Text Manager — Search, download, and cache Project Gutenberg texts.

Uses the Gutendex API (https://gutendex.com) for catalog searches and
downloads plain text directly from gutenberg.org. Caches texts locally
for reuse by the KG extraction pipeline.

Output convention: --download and --random print the book ID to stdout
(one line, just the integer). Logging goes to stderr. This lets the
Makefile capture the ID: $(shell $(KG_GUTENBERG) --random)
"""

import argparse
import json
import logging
import random
import sys
import time
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEXTS_DIR = PROJECT_ROOT / "benchmarks" / "texts"
CATALOG_CACHE = TEXTS_DIR / "gutendex_catalog.json"
CATALOG_CACHE_TTL = 86400  # 24 hours


# ── Gutendex API ──────────────────────────────────────────────────────


def search_gutendex(query, topic=None):
    """Search the Gutendex catalog API, returning a list of book dicts.

    Results are cached for 24h in benchmarks/texts/gutendex_catalog.json
    keyed by (query, topic) to avoid repeated API calls.
    """
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing catalog cache
    catalog = {}
    if CATALOG_CACHE.exists():
        try:
            catalog = json.loads(CATALOG_CACHE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            catalog = {}

    cache_key = f"{query}|{topic or ''}"
    cached = catalog.get(cache_key)
    if cached and time.time() - cached.get("timestamp", 0) < CATALOG_CACHE_TTL:
        log.info("Using cached catalog results for %r (topic=%s)", query, topic)
        return cached["results"]

    # Build API URL
    params = f"search={urllib.request.quote(query)}"
    if topic:
        params += f"&topic={urllib.request.quote(topic)}"
    url = f"https://gutendex.com/books?{params}"

    log.info("Searching Gutendex: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "muninn-kg/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = data.get("results", [])

    # Cache results
    catalog[cache_key] = {"timestamp": time.time(), "results": results}
    CATALOG_CACHE.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    log.info("Found %d results, cached to %s", len(results), CATALOG_CACHE)

    return results


def format_book_info(book):
    """Format a Gutendex book dict for display."""
    authors = ", ".join(a["name"] for a in book.get("authors", []))
    subjects = "; ".join(book.get("subjects", [])[:3])
    languages = ", ".join(book.get("languages", []))
    book_id = book["id"]
    title = book.get("title", "Unknown")
    return f"  [{book_id:>5d}] {title}\n         by {authors} | lang={languages}\n         {subjects}"


# ── Text download ─────────────────────────────────────────────────────


def download_gutenberg_text(gutenberg_id):
    """Fetch plain text from Project Gutenberg, strip boilerplate, and cache locally.

    Returns the path to the cached text file. Replicates the logic from
    benchmark_vss.py:download_gutenberg().
    """
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TEXTS_DIR / f"gutenberg_{gutenberg_id}.txt"

    if cache_path.exists():
        log.info("Gutenberg #%d: cached at %s", gutenberg_id, cache_path)
        return cache_path

    url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
    log.info("Downloading Gutenberg #%d from %s...", gutenberg_id, url)

    req = urllib.request.Request(url, headers={"User-Agent": "muninn-kg/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw_text = resp.read().decode("utf-8-sig")

    # Strip Gutenberg boilerplate (header and footer)
    start_markers = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"]
    end_markers = ["*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG"]

    start_idx = 0
    for marker in start_markers:
        idx = raw_text.find(marker)
        if idx != -1:
            start_idx = raw_text.index("\n", idx) + 1
            break

    end_idx = len(raw_text)
    for marker in end_markers:
        idx = raw_text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    clean_text = raw_text[start_idx:end_idx].strip()

    cache_path.write_text(clean_text, encoding="utf-8")
    log.info("Gutenberg #%d: saved %d chars to %s", gutenberg_id, len(clean_text), cache_path)
    return cache_path


# ── Random book selection ─────────────────────────────────────────────


def pick_random_book(topic="economics", exclude_ids=None):
    """Search catalog for a topic, filter for English + plain-text available,
    exclude already-processed IDs, and pick a random book.

    Returns a Gutendex book dict, or None if no candidates found.
    """
    if exclude_ids is None:
        exclude_ids = set()

    results = search_gutendex(topic, topic=topic)

    candidates = []
    for book in results:
        book_id = book["id"]
        if book_id in exclude_ids:
            continue
        # Must be English
        if "en" not in book.get("languages", []):
            continue
        # Must have plain text available
        formats = book.get("formats", {})
        has_text = any("text/plain" in fmt for fmt in formats)
        if not has_text:
            continue
        candidates.append(book)

    if not candidates:
        log.warning("No candidate books found for topic=%r (excluding %s)", topic, exclude_ids)
        return None

    chosen = random.choice(candidates)
    log.info("Randomly selected: [%d] %s", chosen["id"], chosen.get("title", "Unknown"))
    return chosen


def list_cached_texts():
    """List all cached Gutenberg text files."""
    if not TEXTS_DIR.exists():
        return []
    return sorted(TEXTS_DIR.glob("gutenberg_*.txt"))


def get_cached_book_ids():
    """Return set of Gutenberg IDs for which we already have cached text."""
    ids = set()
    for path in list_cached_texts():
        # Parse ID from filename: gutenberg_3300.txt -> 3300
        stem = path.stem  # gutenberg_3300
        parts = stem.split("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            ids.add(int(parts[1]))
    return ids


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Gutenberg Text Manager for KG pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--search", type=str, help="Search Gutendex API, print results")
    group.add_argument("--download", type=int, metavar="ID", help="Download + cache a specific book (prints ID to stdout)")
    group.add_argument("--random", action="store_true", help="Pick a random book, download it (prints ID to stdout)")
    group.add_argument("--list-cached", action="store_true", help="List cached text files")
    parser.add_argument("--topic", type=str, default="economics", help="Topic filter for --search/--random (default: economics)")
    args = parser.parse_args()

    # Logging to stderr so stdout is clean for piping
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if args.search:
        results = search_gutendex(args.search, topic=args.topic)
        if not results:
            log.warning("No results found for %r", args.search)
            sys.exit(1)
        for book in results:
            print(format_book_info(book))

    elif args.download is not None:
        path = download_gutenberg_text(args.download)
        log.info("Text file: %s (%d bytes)", path, path.stat().st_size)
        # Print just the ID to stdout for Makefile capture
        print(args.download)

    elif args.random:
        existing = get_cached_book_ids()
        book = pick_random_book(topic=args.topic, exclude_ids=existing)
        if book is None:
            log.error("Could not find a random book for topic=%r", args.topic)
            sys.exit(1)
        path = download_gutenberg_text(book["id"])
        log.info("Text file: %s (%d bytes)", path, path.stat().st_size)
        # Print just the ID to stdout for Makefile capture
        print(book["id"])

    elif args.list_cached:
        cached = list_cached_texts()
        if not cached:
            log.info("No cached texts found in %s", TEXTS_DIR)
        else:
            for path in cached:
                size_kb = path.stat().st_size / 1024
                print(f"  {path.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
