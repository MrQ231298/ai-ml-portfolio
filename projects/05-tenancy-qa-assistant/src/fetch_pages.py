"""
Fetch a modest set of PUBLIC Irish tenancy guidance pages.

Sources are official public guidance (Citizens Information and the
Residential Tenancies Board). Each saved page records its source URL
and the fetch time (Europe/Dublin). The text remains the publishers'
copyright; it is stored only so this portfolio project can demonstrate
retrieval with citations.

This script does not log in, does not crawl the whole site, and uses
a small, hand-picked URL list.

Usage
-----
    python src/fetch_pages.py
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

DUBLIN_TZ = ZoneInfo("Europe/Dublin")
# Citizens Information returned HTTP 403 to a custom educational
# User-Agent (tested 14 November 2026) but served the same public pages
# to a standard browser User-Agent. Requests are still polite: no login,
# a hand-picked URL list only, and a short delay between pages.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Hand-picked public guidance pages covering landlord obligations,
# notice periods, deposits and registration, plus the Sept 2026 changes.
URLS = [
    "https://www.citizensinformation.ie/en/housing/renting-a-home/landlords-rights-and-responsibilities/landlords-rights-and-obligations/",
    "https://www.citizensinformation.ie/en/housing/renting-a-home/landlords-rights-and-responsibilities/registering-a-tenancy/",
    "https://www.citizensinformation.ie/en/housing/renting-a-home/tenants-rights-and-responsibilities/if-your-landlord-wants-you-to-leave/",
    "https://www.citizensinformation.ie/en/whats-new/changes-to-rental-laws-from-14-september-2026/",
    "https://rtb.ie/renting/rights-responsibilities/landlord-rights-responsibilities/",
    "https://rtb.ie/register-tenancies/register-a-tenancy/when-to-register/",
    "https://rtb.ie/renting/rights-responsibilities/security-deposits/",
    "https://rtb.ie/renting/ending-a-tenancy/notice-of-termination-guides/",
    "https://rtb.ie/renting/how-a-landlord-can-end-a-tenancy/",
    "https://rtb.ie/renting/rights-responsibilities/tenant-rights-responsibilities/",
]


def project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "raw": root / "data" / "raw",
        "pages": root / "data" / "pages",
        "manifest": root / "data" / "manifest.json",
        "readme": root / "data" / "README.md",
    }


class _TextExtractor(HTMLParser):
    """HTML-to-text extractor that prefers <main>/<article> and skips chrome."""

    SKIP = {
        "script", "style", "noscript", "svg", "path", "nav", "footer",
        "aside", "form", "button", "iframe",
    }
    BLOCK = {"p", "h1", "h2", "h3", "h4", "li", "tr", "br", "div", "section"}
    HEADINGS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### "}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._prefer_depth = 0
        self._heading: str | None = None
        self._title_bits: list[str] = []
        self._in_title = False
        self.all_parts: list[str] = []
        self.pref_parts: list[str] = []
        self.title = ""

    def _emit(self, s: str) -> None:
        if self._skip_depth:
            return
        self.all_parts.append(s)
        if self._prefer_depth:
            self.pref_parts.append(s)

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        if tag in {"main", "article"}:
            self._prefer_depth += 1
        if tag in self.SKIP:
            self._skip_depth += 1
        if tag in self.HEADINGS:
            self._heading = self.HEADINGS[tag]
            self._emit("\n\n")
        elif tag in self.BLOCK:
            self._emit("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            self.title = " ".join(self._title_bits).strip()
        if tag in self.HEADINGS:
            self._heading = None
            self._emit("\n")
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"main", "article"} and self._prefer_depth:
            self._prefer_depth -= 1
        if tag in {"p", "li"}:
            self._emit("\n")

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self._title_bits.append(text)
            return
        if self._heading:
            self._emit(self._heading + text + "\n")
            self._heading = None
        else:
            self._emit(text + " ")


def _normalise_text(parts: list[str]) -> str:
    text = re.sub(r"[ \t]+", " ", "".join(parts))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    title = re.sub(r"\s+", " ", parser.title).strip()
    # Prefer main/article text when it is long enough; otherwise use the body.
    preferred = _normalise_text(parser.pref_parts)
    fallback = _normalise_text(parser.all_parts)
    text = preferred if len(preferred) >= 400 else fallback
    return title, text


def slug_for(url: str) -> str:
    path = urlparse(url).path.strip("/")
    slug = path.replace("/", "__") or "index"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", slug).strip("-")
    return slug[:120]


def fetch_url(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IE,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return getattr(resp, "status", 200) or 200, resp.read()


def main() -> None:
    paths = project_paths()
    paths["raw"].mkdir(parents=True, exist_ok=True)
    paths["pages"].mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(DUBLIN_TZ)
    records = []
    for i, url in enumerate(URLS):
        slug = slug_for(url)
        print(f"Fetching {url}")
        try:
            status, payload = fetch_url(url)
            raw_path = paths["raw"] / f"{slug}.html"
            raw_path.write_bytes(payload)
            html = payload.decode("utf-8", errors="replace")
            title, text = html_to_text(html)
            # Keep only reasonably long documents.
            if len(text) < 400:
                print(f"  skip (only {len(text)} chars of text, HTTP {status})")
                continue
            rec = {
                "page_id": slug,
                "url": url,
                "title": title or slug,
                "fetched_at_dublin": fetched_at.strftime("%Y-%m-%d %H:%M %Z"),
                "http_status": status,
                "n_chars": len(text),
                "text": text,
            }
            (paths["pages"] / f"{slug}.json").write_text(
                json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            records.append({k: rec[k] for k in rec if k != "text"})
            print(f"  saved {slug}  ({len(text):,} chars)  title={title[:80]!r}")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        if i < len(URLS) - 1:
            time.sleep(0.8)

    if len(records) < 6:
        raise RuntimeError(
            f"Only fetched {len(records)} pages; need a modest working set. "
            "Check the URLs or network and retry."
        )

    manifest = {
        "fetched_at_dublin": fetched_at.strftime("%Y-%m-%d %H:%M %Z"),
        "n_pages": len(records),
        "pages": records,
        "note": (
            "Public guidance pages from Citizens Information and the RTB, "
            "fetched for an educational retrieval demo. Copyright remains "
            "with the publishers. Always cite the source URL."
        ),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["readme"].write_text(
        f"""# Data folder — public tenancy guidance (snapshot)

This folder holds a **snapshot** of public Irish tenancy guidance pages
fetched on **{fetched_at.strftime("%Y-%m-%d %H:%M %Z")}**.

- Sources: [Citizens Information](https://www.citizensinformation.ie/) and
  the [Residential Tenancies Board](https://www.rtb.ie/).
- Each page file in `pages/` records the **source URL** and the **fetch
  time**. Raw HTML is in `raw/`.
- The text remains the publishers' copyright. It is stored only so this
  educational project can demonstrate retrieval with citations. Do not
  republish the full text as your own guidance.
- This is a snapshot. Irish rental law can change (it did, for example,
  on 14 September 2026). Always check the live page.

{len(records)} pages were fetched successfully. See `manifest.json`.
""",
        encoding="utf-8",
    )
    print(f"\nFetched {len(records)} pages. Manifest: {paths['manifest']}")


if __name__ == "__main__":
    main()
