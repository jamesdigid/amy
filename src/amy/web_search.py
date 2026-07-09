from __future__ import annotations

from dataclasses import dataclass
import html
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen
import re


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: str = ""


class WebSearcher(Protocol):
    def search(self, query: str, limit: int = 4) -> list[SearchResult]: ...


class DuckDuckGoWebSearch:
    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self._timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 4) -> list[SearchResult]:
        if not query.strip():
            return []

        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            },
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            html_text = response.read().decode("utf-8", errors="ignore")

        return self._parse_results(html_text, limit)

    def _parse_results(self, html_text: str, limit: int) -> list[SearchResult]:
        parser = _DuckDuckGoResultParser()
        parser.feed(html_text)
        results: list[SearchResult] = []
        for item in parser.results[:limit]:
            title = self._clean_text(item.get("title", ""))
            url = self._resolve_url(item.get("url", ""))
            snippet = self._clean_text(item.get("snippet", ""))
            if title and url:
                content = self._fetch_page_text(url)
                results.append(SearchResult(title=title, url=url, snippet=snippet, content=content))
        return results

    def _resolve_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.netloc.endswith("duckduckgo.com"):
            query = parse_qs(parsed.query)
            if "uddg" in query and query["uddg"]:
                return unquote(query["uddg"][0])
        return url

    def _clean_text(self, value: str) -> str:
        without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value))
        return re.sub(r"\s+", " ", without_tags).strip()

    def _fetch_page_text(self, url: str, limit: int = 4000) -> str:
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return ""
                body = response.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""

        extractor = _VisibleTextExtractor()
        extractor.feed(body)
        text = self._clean_text(extractor.get_text())
        return text[:limit]


class _DuckDuckGoResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._current_field: str | None = None
        self._capture_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        class_name = attr_map.get("class", "")

        if tag == "a" and "result__a" in class_name:
            self._current = {
                "title": "",
                "url": attr_map.get("href", ""),
                "snippet": "",
            }
            self._current_field = "title"
            self._capture_depth = 1
            return

        if self._current is not None and ("result__snippet" in class_name or "result__title" in class_name):
            self._current_field = "snippet"
            self._capture_depth += 1
            return

        if self._current is not None and self._current_field is not None:
            self._capture_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._current is None or self._current_field is None:
            return

        self._capture_depth -= 1
        if self._capture_depth <= 0:
            self.results.append(self._current)
            self._current = None
            self._current_field = None
            self._capture_depth = 0

    def handle_data(self, data: str) -> None:
        if self._current is None or self._current_field is None:
            return
        current_value = self._current.get(self._current_field, "")
        self._current[self._current_field] = f"{current_value}{data}"


class _VisibleTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = re.sub(r"\s+", " ", html.unescape(data)).strip()
        if text:
            self._parts.append(text)

    def get_text(self) -> str:
        return " ".join(self._parts)
