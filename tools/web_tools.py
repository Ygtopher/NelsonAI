"""
Nelson AI — Web Tools
Gives Nelson the ability to search the internet and fetch web pages.
Uses DuckDuckGo (free, no API key needed) and Wikipedia.
"""

import re
import time
import requests
from typing import Optional
from urllib.parse import quote_plus, urlparse
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "rw,en;q=0.9",
}
TIMEOUT = 10  # seconds


# ─────────────────────────────────────────────────────────────────
# DuckDuckGo Search (no API key required)
# ─────────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo.
    Returns list of {title, url, snippet} dicts.
    """
    results = []

    # Try duckduckgo-search library first
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except ImportError:
        pass  # Fall through to manual scrape
    except Exception as e:
        print(f"    [web_search] DDGS error: {e}")

    # Fallback: DuckDuckGo HTML scrape
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        soup = BeautifulSoup(resp.text, "lxml")

        for result in soup.select(".result")[:max_results]:
            title_el   = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            url_el     = result.select_one(".result__url")

            if title_el:
                results.append({
                    "title":   title_el.get_text(strip=True),
                    "url":     url_el.get_text(strip=True) if url_el else "",
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                })
    except Exception as e:
        results.append({"title": "Error", "url": "", "snippet": f"Search failed: {e}"})

    return results


# ─────────────────────────────────────────────────────────────────
# URL Fetcher — extracts readable text from any webpage
# ─────────────────────────────────────────────────────────────────

def fetch_page(url: str, max_chars: int = 3000) -> str:
    """
    Fetch a webpage and return its clean text content.
    Strips HTML, ads, navigation, etc.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "iframe", "noscript", "meta",
                          "link", "button", "advertisement"]):
            tag.decompose()

        # Get main content — try article/main first, then body
        main = (
            soup.find("article") or
            soup.find("main") or
            soup.find(id=re.compile(r"content|article|main", re.I)) or
            soup.find(class_=re.compile(r"content|article|post|entry", re.I)) or
            soup.body
        )

        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 30]
        text = "\n".join(lines)

        return text[:max_chars]

    except Exception as e:
        return f"Failed to fetch {url}: {e}"


# ─────────────────────────────────────────────────────────────────
# Wikipedia (Kinyarwanda + English)
# ─────────────────────────────────────────────────────────────────

def wikipedia_search(query: str, lang: str = "rw", max_chars: int = 2000) -> str:
    """
    Search Wikipedia (Kinyarwanda by default, fallback to English).
    Returns a text summary.
    """
    for language in [lang, "en"]:
        try:
            # Search for article
            search_url = (
                f"https://{language}.wikipedia.org/w/api.php"
                f"?action=query&list=search&srsearch={quote_plus(query)}"
                f"&format=json&srlimit=1"
            )
            resp = requests.get(search_url, headers=HEADERS, timeout=TIMEOUT)
            data = resp.json()
            hits = data.get("query", {}).get("search", [])

            if not hits:
                continue

            # Fetch article summary
            title = hits[0]["title"]
            summary_url = (
                f"https://{language}.wikipedia.org/w/api.php"
                f"?action=query&prop=extracts&exintro&explaintext"
                f"&titles={quote_plus(title)}&format=json"
            )
            resp2 = requests.get(summary_url, headers=HEADERS, timeout=TIMEOUT)
            pages = resp2.json().get("query", {}).get("pages", {})
            page = next(iter(pages.values()))
            extract = page.get("extract", "")

            if extract:
                lang_label = "Kinyarwanda Wikipedia" if language == "rw" else "English Wikipedia"
                return f"[{lang_label}: {title}]\n{extract[:max_chars]}"

        except Exception:
            continue

    return f"Wikipedia: No results found for '{query}'"


# ─────────────────────────────────────────────────────────────────
# Kinyarwanda News Scraper (for self-evolution data)
# ─────────────────────────────────────────────────────────────────

NEWS_SOURCES = [
    {
        "name": "Igihe",
        "url": "https://igihe.com",
        "article_selector": "article a, .article-title a, h2 a",
        "lang": "rw",
    },
    {
        "name": "RNA News",
        "url": "https://rnanews.com",
        "article_selector": "article a, .post-title a, h2 a",
        "lang": "rw",
    },
    {
        "name": "KT Press",
        "url": "https://www.ktpress.rw",
        "article_selector": "article a, .entry-title a",
        "lang": "rw",
    },
]


def scrape_news_articles(max_per_source: int = 5) -> list[dict]:
    """
    Scrape fresh Kinyarwanda news articles.
    Returns list of {source, url, title, text} dicts.
    """
    articles = []

    for source in NEWS_SOURCES:
        try:
            resp = requests.get(source["url"], headers=HEADERS, timeout=TIMEOUT)
            soup = BeautifulSoup(resp.text, "lxml")

            links = set()
            for a in soup.select(source["article_selector"]):
                href = a.get("href", "")
                if href and not href.startswith("#"):
                    # Make absolute URL
                    if href.startswith("http"):
                        links.add(href)
                    else:
                        base = f"{urlparse(source['url']).scheme}://{urlparse(source['url']).netloc}"
                        links.add(base + href)

            for url in list(links)[:max_per_source]:
                try:
                    text = fetch_page(url, max_chars=5000)
                    if len(text) > 200:  # Minimum content check
                        articles.append({
                            "source": source["name"],
                            "url":    url,
                            "text":   text,
                        })
                    time.sleep(0.5)  # Be polite to servers
                except Exception:
                    continue

        except Exception as e:
            print(f"    [scraper] {source['name']} failed: {e}")
            continue

    return articles


# ─────────────────────────────────────────────────────────────────
# Tool Dispatcher — used by chat.py
# ─────────────────────────────────────────────────────────────────

def run_tool(tool_name: str, query: str) -> str:
    """
    Execute a named tool and return formatted result string.
    Used by the chat loop when Nelson requests a tool.
    """
    print(f"\n    🔍 Nelson is searching: '{query}' ...")

    if tool_name == "search":
        results = web_search(query, max_results=4)
        if not results:
            return "Ntibyabonetse ibisubizo (No results found)."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            if r["snippet"]:
                lines.append(f"   {r['snippet'][:200]}")
            if r["url"]:
                lines.append(f"   Source: {r['url']}")
        return "\n".join(lines)

    elif tool_name == "fetch":
        return fetch_page(query, max_chars=2500)

    elif tool_name == "wikipedia":
        return wikipedia_search(query, lang="rw")

    else:
        return f"Unknown tool: {tool_name}"


if __name__ == "__main__":
    print("Testing web tools ...")
    print("\n[Search] 'u Rwanda':")
    results = web_search("u Rwanda ubutegetsi", max_results=3)
    for r in results:
        print(f"  • {r['title']}: {r['snippet'][:100]}")

    print("\n[Wikipedia] 'Rwanda':")
    print(wikipedia_search("Rwanda", lang="rw")[:500])
