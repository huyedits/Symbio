"""Keyless web access: layered search backends and readable page fetching."""

import html as html_lib
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


def _http_get(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(1_000_000).decode("utf-8", errors="replace")


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "header", "footer", "nav"}

    def __init__(self):
        super().__init__()
        self.chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self.chunks.append(data.strip())


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return re.sub(r"\s+", " ", " ".join(parser.chunks)).strip()


def _search_duckduckgo(query: str, max_results: int, timeout: int = 15) -> list[tuple[str, str, str]]:
    doc = _http_get("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query),
                    timeout=timeout)
    titles = re.findall(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', doc, re.DOTALL
    )
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', doc, re.DOTALL)
    results = []
    for i, (href, title_html) in enumerate(titles[:max_results]):
        url = href
        if "uddg=" in href:
            url = urllib.parse.unquote(href.split("uddg=", 1)[1].split("&", 1)[0])
        snippet = html_to_text(snippets[i]) if i < len(snippets) else ""
        results.append((html_to_text(title_html), url, snippet))
    return results


def _search_ddg_api(query: str, max_results: int, timeout: int = 15) -> list[tuple[str, str, str]]:
    """DuckDuckGo instant-answer API: great for factual queries, no key needed."""
    doc = _http_get(
        "https://api.duckduckgo.com/?format=json&no_html=1&q="
        + urllib.parse.quote_plus(query),
        timeout=timeout,
    )
    data = json.loads(doc)
    results = []
    if data.get("AbstractText"):
        results.append((
            data.get("Heading", query),
            data.get("AbstractURL", ""),
            data["AbstractText"],
        ))
    for topic in data.get("RelatedTopics", []):
        if len(results) >= max_results:
            break
        if isinstance(topic, dict) and topic.get("Text") and topic.get("FirstURL"):
            results.append((topic["Text"][:80], topic["FirstURL"], topic["Text"]))
    return results


def _search_google_news(query: str, max_results: int, timeout: int = 15) -> list[tuple[str, str, str]]:
    """Google News RSS: reliable for news and current events, no key needed."""
    doc = _http_get(
        "https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q="
        + urllib.parse.quote_plus(query),
        timeout=timeout,
    )
    results = []
    for item in re.findall(r"<item>(.*?)</item>", doc, re.DOTALL)[:max_results]:
        t = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
        s = re.search(r"<source[^>]*>(.*?)</source>", item, re.DOTALL)
        d = re.search(r"<pubDate>(.*?)</pubDate>", item, re.DOTALL)
        if not t:
            continue
        title = html_to_text(html_lib.unescape(t.group(1)))
        # Google News links are enormous redirect URLs — noise that the model
        # tends to echo back; the source name is what a reader needs.
        source = html_to_text(html_lib.unescape(s.group(1))) if s else ""
        results.append((
            title,
            f"via {source}" if source else "",
            f"Published {d.group(1)}" if d else "",
        ))
    return results


def web_search(query: str, config: dict[str, Any], max_results: int | None = None) -> tuple[bool, str]:
    """Search the web without API keys. Backends in order: DuckDuckGo HTML
    (full web results, sometimes bot-challenged), DuckDuckGo instant-answer
    API (facts), Google News RSS (news/current events)."""
    query = query.strip()
    if not query:
        return False, "Empty query."
    if max_results is None:
        max_results = int(config["web"]["search_results"])

    timeout = int(config["web"]["http_timeout"])
    results: list[tuple[str, str, str]] = []
    last_err = "No results found."
    for backend in (_search_duckduckgo, _search_ddg_api, _search_google_news):
        try:
            results = backend(query, max_results, timeout=timeout)
        except Exception as e:
            last_err = f"Search failed: {e}"
            continue
        if results:
            break
    if not results:
        return False, last_err

    lines = [
        f"{i + 1}. {title}\n   {url}\n   {snippet}"
        for i, (title, url, snippet) in enumerate(results)
    ]
    out = "\n".join(lines)
    max_len = config["agent"]["max_output_len"]
    if len(out) > max_len:
        out = out[:max_len] + "\n... (truncated)"
    return True, out


def read_page(url: str, config: dict[str, Any]) -> tuple[bool, str]:
    """Fetch a web page and return its readable text."""
    url = url.strip()
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        return False, f"Only http/https URLs can be read, got: {url!r}"
    try:
        text = html_to_text(_http_get(url, timeout=int(config["web"]["http_timeout"])))
    except Exception as e:
        return False, f"Could not read {url}: {e}"
    if not text:
        return False, f"No readable text found at {url}."
    max_len = int(config["agent"].get("max_output_len", 4000))
    if len(text) > max_len:
        text = text[:max_len] + "\n... (truncated)"
    return True, text
