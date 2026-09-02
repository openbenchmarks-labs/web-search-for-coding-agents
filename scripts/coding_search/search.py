"""Pluggable web-search backends.

HTTP call + hit shape for Parallel, Firecrawl, Exa, Linkup, Tavily, and Brave
(LLM Context). Vendors are locked to search-only vs search-fetch boards in
SEARCH_ONLY_BACKENDS / SEARCH_FETCH_BACKENDS.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Protocol

import requests

PARALLEL_SEARCH_URL = "https://api.parallel.ai/v1/search"
PARALLEL_EXTRACT_URL = "https://api.parallel.ai/v1/extract"
FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"
EXA_SEARCH_URL = "https://api.exa.ai/search"
EXA_CONTENTS_URL = "https://api.exa.ai/contents"
LINKUP_SEARCH_URL = "https://api.linkup.so/v1/search"
LINKUP_FETCH_URL = "https://api.linkup.so/v1/fetch"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
BRAVE_LLM_CONTEXT_URL = "https://api.search.brave.com/res/v1/llm/context"
BRAVE_SNIPPET_CHARS = 8000
HEADER_REDACT_TOKENS = (
    "authorization",
    "api-key",
    "api_key",
    "apikey",
    "x-api-key",
    "token",
    "secret",
    "password",
    "cookie",
)
RAW_META_KEYS = ("request", "response")
MAX_RAW_CHARS = 1_000_000
DEFAULT_MAX_RESULTS = 8
DEFAULT_MAX_FETCH_CHARS = 16000
FETCH_TIMEOUT_S = 90


class SearchHit(dict):
    """url / title / snippet."""


class SearchBackend(Protocol):
    name: str

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        ...

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        ...


class ParallelBasic:
    name = "parallel_basic"
    mode = "basic"
    last_meta: dict[str, Any] | None = None

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        key = os.environ.get("PARALLEL_API_KEY")
        if not key:
            raise RuntimeError(f"PARALLEL_API_KEY is required for {self.name}")
        timeout = 90 if self.mode == "advanced" else 45 if self.mode in {"turbo", "fast"} else 60
        payload, meta = vendor_call(self,
            "POST",
            PARALLEL_SEARCH_URL,
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json_body={
                "objective": query,
                "search_queries": [query],
                "mode": self.mode,
                "advanced_settings": {"max_results": max_results},
            },
            timeout=timeout,
        )
        hits = parse_parallel_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        key = os.environ.get("PARALLEL_API_KEY")
        if not key:
            raise RuntimeError(f"PARALLEL_API_KEY is required for {self.name}")
        body: dict[str, Any] = {
            "urls": [url],
            "max_chars_total": DEFAULT_MAX_FETCH_CHARS,
            "advanced_settings": {"full_content": True},
        }
        if objective.strip():
            body["objective"] = objective.strip()
        payload, meta = vendor_call(self,
            "POST",
            PARALLEL_EXTRACT_URL,
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json_body=body,
            timeout=FETCH_TIMEOUT_S,
        )
        page = parse_parallel_extract(payload, url=url)
        flags = _pop_flags(page)
        meta = {**meta, **flags, "extract_id": flags.get("extract_id") or meta.get("extract_id")}
        self.last_meta = meta
        page["_meta"] = meta
        return page


class ParallelTurbo(ParallelBasic):
    name = "parallel_turbo"
    mode = "turbo"


class ParallelFast(ParallelBasic):
    name = "parallel_fast"
    mode = "fast"


class ParallelAdvanced(ParallelBasic):
    name = "parallel_advanced"
    mode = "advanced"


class FirecrawlSearch:
    name = "firecrawl"
    last_meta: dict[str, Any] | None = None

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        key = os.environ.get("FIRECRAWL_API_KEY")
        if not key:
            raise RuntimeError("FIRECRAWL_API_KEY is required for firecrawl")
        payload, meta = vendor_call(self,
            "POST",
            FIRECRAWL_SEARCH_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json_body={"query": query, "limit": max_results},
            timeout=60,
        )
        hits = parse_firecrawl_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del objective
        key = os.environ.get("FIRECRAWL_API_KEY")
        if not key:
            raise RuntimeError("FIRECRAWL_API_KEY is required for firecrawl")
        payload, meta = vendor_call(self,
            "POST",
            FIRECRAWL_SCRAPE_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json_body={"url": url, "formats": ["markdown"]},
            timeout=FETCH_TIMEOUT_S,
        )
        page = parse_firecrawl_scrape(payload, url=url)
        flags = _pop_flags(page)
        self.last_meta = {**meta, **flags}
        page["_meta"] = self.last_meta
        return page


class FirecrawlDevSearch(FirecrawlSearch):
    """Firecrawl Developer index — `firecrawl_developer_search` on the hosted MCP.

    Search hits come from the developer index; fetch inherits FirecrawlSearch's
    /v2/scrape. So the ONLY difference vs the `firecrawl` arm is the search tool:
    developer index vs general web search — a clean A/B on the same tasks.
    """

    name = "fc_dev"

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        from .fc_dev_mcp import developer_search

        hits = developer_search(query, max_results=max_results)
        self.last_meta = _search_meta({}, hits)
        return hits


class ExaSearch:
    """Exa search. Fetch is POST /contents with text=true (get_contents)."""

    name = "exa_auto"
    search_type = "auto"
    last_meta: dict[str, Any] | None = None

    def _key(self) -> str:
        key = os.environ.get("EXA_API_KEY")
        if not key:
            raise RuntimeError(f"EXA_API_KEY is required for {self.name}")
        return key

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        timeout = 120 if self.search_type == "deep" else 30 if self.search_type in {"fast", "instant"} else 45
        payload, meta = vendor_call(self,
            "POST",
            EXA_SEARCH_URL,
            headers={"x-api-key": self._key(), "Content-Type": "application/json"},
            json_body={
                "query": query,
                "type": self.search_type,
                "numResults": max_results,
                "contents": {"highlights": True},
            },
            timeout=timeout,
        )
        hits = parse_exa_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del objective
        payload, meta = vendor_call(self,
            "POST",
            EXA_CONTENTS_URL,
            headers={"x-api-key": self._key(), "Content-Type": "application/json"},
            json_body={"urls": [url], "text": True},
            timeout=FETCH_TIMEOUT_S,
        )
        page = parse_exa_contents(payload, url=url)
        flags = _pop_flags(page)
        self.last_meta = {**meta, **flags}
        page["_meta"] = self.last_meta
        return page


class ExaDeep(ExaSearch):
    name = "exa_deep"
    search_type = "deep"


class ExaFast(ExaSearch):
    name = "exa_fast"
    search_type = "fast"


class ExaInstant(ExaSearch):
    name = "exa_instant"
    search_type = "instant"


class LinkupSearch:
    """Linkup search + fetch.

    Search: depth=fast, outputType=searchResults ($0.005). Query is passed as-is,
    same as Firecrawl / Parallel basic. Not standard (agentic rewrite, same $)
    or deep ($0.05). Not sourcedAnswer (LLM answer on top of search).

    Fetch: mode=standard, renderJs=true. JS-rendered markdown extract.
    """

    name = "linkup_fast"
    depth = "fast"
    output_type = "searchResults"
    fetch_mode = "standard"
    render_js = True
    last_meta: dict[str, Any] | None = None

    def _key(self) -> str:
        key = os.environ.get("LINKUP_API_KEY")
        if not key:
            raise RuntimeError(f"LINKUP_API_KEY is required for {self.name}")
        return key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"}

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        payload, meta = vendor_call(self,
            "POST",
            LINKUP_SEARCH_URL,
            headers=self._headers(),
            json_body={
                "q": query,
                "depth": self.depth,
                "outputType": self.output_type,
                "maxResults": max_results,
            },
            timeout=30,
        )
        hits = parse_linkup_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del objective
        payload, meta = vendor_call(self,
            "POST",
            LINKUP_FETCH_URL,
            headers=self._headers(),
            json_body={
                "url": url,
                "mode": self.fetch_mode,
                "renderJs": self.render_js,
            },
            timeout=FETCH_TIMEOUT_S,
        )
        page = parse_linkup_fetch(payload, url=url)
        flags = _pop_flags(page)
        self.last_meta = {**meta, **flags}
        page["_meta"] = self.last_meta
        return page


class LinkupStandard(LinkupSearch):
    name = "linkup_standard"
    depth = "standard"


class TavilySearch:
    """Tavily search + extract.

    Search: search_depth=fast (1 credit, ~$0.008). Chunked snippets like Exa
    highlights. Not ultra-fast (one NLP summary per URL) or advanced (2 credits).
    include_answer and include_raw_content stay off — the agent writes the
    answer; fetch owns page bodies.

    Fetch: extract_depth=basic, format=markdown (1 credit / 5 URLs, ~$0.0016).
    Not advanced (tables/embeds, 2x credits).
    """

    name = "tavily_fast"
    search_depth = "fast"
    extract_depth = "basic"
    chunks_per_source = 3
    last_meta: dict[str, Any] | None = None

    def _key(self) -> str:
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            raise RuntimeError(f"TAVILY_API_KEY is required for {self.name}")
        return key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"}

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        payload, meta = vendor_call(self,
            "POST",
            TAVILY_SEARCH_URL,
            headers=self._headers(),
            json_body={
                "query": query,
                "search_depth": self.search_depth,
                "max_results": max_results,
                "chunks_per_source": self.chunks_per_source,
                "include_answer": False,
                "include_raw_content": False,
                "topic": "general",
            },
            timeout=45,
        )
        hits = parse_tavily_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        body: dict[str, Any] = {
            "urls": [url],
            "extract_depth": self.extract_depth,
            "format": "markdown",
            "timeout": 60,
        }
        if objective.strip():
            body["query"] = objective.strip()
        payload, meta = vendor_call(self,
            "POST",
            TAVILY_EXTRACT_URL,
            headers=self._headers(),
            json_body=body,
            timeout=FETCH_TIMEOUT_S,
        )
        page = parse_tavily_extract(payload, url=url)
        flags = _pop_flags(page)
        self.last_meta = {**meta, **flags}
        page["_meta"] = self.last_meta
        return page


class TavilyBasic(TavilySearch):
    name = "tavily_basic"
    search_depth = "basic"
    extract_depth = "basic"


class TavilyAdvanced(TavilySearch):
    name = "tavily_advanced"
    search_depth = "advanced"
    extract_depth = "advanced"


class BraveSearch:
    """Brave LLM Context search-only.

    POST /res/v1/llm/context — the agent/RAG endpoint, not Web Search
    (/res/v1/web/search). Returns pre-extracted snippets per URL so search-only
    runs have grounding text without a fetch hop. fetch() is not implemented.
    """

    name = "brave"
    last_meta: dict[str, Any] | None = None

    def _key(self) -> str:
        key = os.environ.get("BRAVE_SEARCH_API_KEY")
        if not key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY is required for brave")
        return key

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        n = max(1, min(int(max_results), 50))
        payload, meta = vendor_call(
            self,
            "POST",
            BRAVE_LLM_CONTEXT_URL,
            headers={
                "X-Subscription-Token": self._key(),
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json_body={
                "q": query,
                "count": n,
                "maximum_number_of_urls": n,
                "maximum_number_of_tokens": min(8192, max(2048, n * 1024)),
                "enable_local": False,
            },
            timeout=30,
        )
        hits = parse_brave_hits(payload, max_results=n)
        self.last_meta = _search_meta(meta, hits)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del url, objective
        raise RuntimeError("brave is LLM Context search-only; web_fetch is not wired")


def parse_parallel_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        excerpts = item.get("excerpts") or []
        snippet = "\n".join(part for part in excerpts if isinstance(part, str))
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": snippet[:1200],
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_firecrawl_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        rows = data.get("web") or data.get("results") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link")
        if not url:
            continue
        url = str(url)
        if url in seen:
            continue
        seen.add(url)
        snippet = item.get("description") or item.get("snippet") or item.get("markdown") or ""
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": str(snippet)[:1200],
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_parallel_extract(
    payload: Any,
    *,
    url: str,
    max_chars: int = DEFAULT_MAX_FETCH_CHARS,
) -> dict[str, str]:
    errors = (payload or {}).get("errors") or []
    results = (payload or {}).get("results") or []
    if errors and not results:
        first = errors[0] if isinstance(errors[0], dict) else {}
        detail = first.get("content") or first.get("error_type") or "extract failed"
        raise RuntimeError(f"{url}: {detail}")
    item = results[0] if results and isinstance(results[0], dict) else {}
    excerpts = item.get("excerpts") or []
    excerpt_text = "\n\n".join(part for part in excerpts if isinstance(part, str))
    raw = str(item.get("full_content") or excerpt_text or "")
    return {
        "url": str(item.get("url") or url),
        "title": str(item.get("title") or ""),
        "content": raw[:max_chars],
        "_truncated": len(raw) > max_chars,
        "_extract_id": str((payload or {}).get("extract_id") or ""),
        "_session_id": str((payload or {}).get("session_id") or ""),
        "_had_full_content": bool(item.get("full_content")),
        "_n_excerpts": len(excerpts),
    }


def parse_firecrawl_scrape(
    payload: Any,
    *,
    url: str,
    max_chars: int = DEFAULT_MAX_FETCH_CHARS,
) -> dict[str, str]:
    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(str(payload.get("error") or "scrape failed"))
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, dict):
        data = payload if isinstance(payload, dict) else {}
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    content = str(data.get("markdown") or data.get("content") or "")
    if not content:
        raise RuntimeError(f"{url}: scrape returned no markdown")
    return {
        "url": str(meta.get("sourceURL") or meta.get("url") or data.get("url") or url),
        "title": str(meta.get("title") or data.get("title") or ""),
        "content": content[:max_chars],
        "_truncated": len(content) > max_chars,
        "_status_code": meta.get("statusCode") or meta.get("status_code"),
    }


def hit_text_stats(hits: list[dict[str, str]]) -> dict[str, int]:
    """Chars and ~tokens of title+url+snippet actually fed back to the model."""
    chars = 0
    for hit in hits:
        chars += len(str(hit.get("url") or ""))
        chars += len(str(hit.get("title") or ""))
        chars += len(str(hit.get("snippet") or ""))
    return {
        "n_hits": len(hits),
        "chars": chars,
        "tokens_est": chars // 4,
    }


def page_text_stats(page: dict[str, str]) -> dict[str, int]:
    return hit_text_stats([{
        "url": str(page.get("url") or ""),
        "title": str(page.get("title") or ""),
        "snippet": str(page.get("content") or ""),
    }])


def parse_exa_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        highlights = item.get("highlights") or []
        if isinstance(highlights, list) and highlights:
            snippet = "\n".join(part for part in highlights if isinstance(part, str))
        else:
            snippet = str(item.get("summary") or item.get("text") or "")
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": snippet[:1200],
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_exa_contents(
    payload: Any,
    *,
    url: str,
    max_chars: int = DEFAULT_MAX_FETCH_CHARS,
) -> dict[str, str]:
    results = (payload or {}).get("results") or []
    item = results[0] if results and isinstance(results[0], dict) else {}
    raw = item.get("text") or item.get("excerpt") or ""
    if isinstance(raw, dict):
        raw = raw.get("text") or ""
    content = str(raw)
    if not content:
        raise RuntimeError(f"{url}: contents returned no text")
    return {
        "url": str(item.get("url") or url),
        "title": str(item.get("title") or ""),
        "content": content[:max_chars],
        "_truncated": len(content) > max_chars,
    }


def parse_linkup_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict) or item.get("type") == "image" or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        hits.append({
            "url": url,
            "title": str(item.get("name") or item.get("title") or ""),
            "snippet": str(item.get("content") or item.get("snippet") or "")[:1200],
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_linkup_fetch(
    payload: Any,
    *,
    url: str,
    max_chars: int = DEFAULT_MAX_FETCH_CHARS,
) -> dict[str, str]:
    data = payload if isinstance(payload, dict) else {}
    content = str(data.get("markdown") or data.get("content") or "")
    if not content:
        raise RuntimeError(f"{url}: fetch returned no markdown")
    return {
        "url": url,
        "title": str(data.get("title") or ""),
        "content": content[:max_chars],
        "_truncated": len(content) > max_chars,
    }


def parse_tavily_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": str(item.get("content") or "")[:1200],
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_brave_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    """Map LLM Context grounding.generic[] to the shared url/title/snippet shape."""
    grounding = payload.get("grounding") if isinstance(payload, dict) else None
    rows = grounding.get("generic") if isinstance(grounding, dict) else None
    if not isinstance(rows, list):
        rows = []
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        chunks = item.get("snippets") or []
        if isinstance(chunks, list):
            snippet = "\n".join(str(part) for part in chunks if part)
        else:
            snippet = str(chunks)
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": snippet[:BRAVE_SNIPPET_CHARS],
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_tavily_extract(
    payload: Any,
    *,
    url: str,
    max_chars: int = DEFAULT_MAX_FETCH_CHARS,
) -> dict[str, str]:
    failed = (payload or {}).get("failed_results") or []
    results = (payload or {}).get("results") or []
    if failed and not results:
        first = failed[0] if isinstance(failed[0], dict) else {}
        detail = first.get("error") or "extract failed"
        raise RuntimeError(f"{url}: {detail}")
    item = results[0] if results and isinstance(results[0], dict) else {}
    content = str(item.get("raw_content") or item.get("content") or "")
    if not content:
        raise RuntimeError(f"{url}: extract returned no content")
    usage = (payload or {}).get("usage") if isinstance((payload or {}).get("usage"), dict) else {}
    return {
        "url": str(item.get("url") or url),
        "title": str(item.get("title") or ""),
        "content": content[:max_chars],
        "_truncated": len(content) > max_chars,
        "_credits": usage.get("credits"),
        "_request_id": str((payload or {}).get("request_id") or ""),
    }


def vendor_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    request_audit = {
        "method": method,
        "url": url,
        "headers": redact_headers(headers),
        "params": redact_payload(params),
        "json": redact_payload(json_body),
    }
    started = time.perf_counter()
    try:
        response = requests.request(
            method,
            url,
            headers=headers,
            json=json_body,
            params=params,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        meta = {
            "endpoint": url,
            "method": method,
            "http_status": None,
            "http_ms": int((time.perf_counter() - started) * 1000),
            "response_bytes": 0,
            "vendor_request_id": "",
            "error": f"{type(exc).__name__}: {exc}",
            "request": request_audit,
            "response": None,
        }
        error = RuntimeError(f"{type(exc).__name__}: {exc}")
        setattr(error, "vendor_meta", meta)
        raise error from exc
    try:
        payload: Any = response.json() if response.content else {}
    except ValueError:
        payload = {"raw": (response.text or "")[:500]}
    meta: dict[str, Any] = {
        "endpoint": url,
        "method": method,
        "http_status": response.status_code,
        "http_ms": int((time.perf_counter() - started) * 1000),
        "response_bytes": len(response.content or b""),
        "vendor_request_id": _header_id(response),
        "request": request_audit,
        "response": {
            "http_status": response.status_code,
            "headers": redact_headers(dict(response.headers)),
            "body": clip_raw(payload),
        },
    }
    if isinstance(payload, dict):
        for key in ("extract_id", "search_id", "id", "session_id", "request_id"):
            if payload.get(key):
                meta[key] = str(payload[key])
    if not response.ok:
        meta["error_body"] = str(payload)[:800]
        error = requests.HTTPError(f"{response.status_code} for {url}", response=response)
        setattr(error, "vendor_meta", meta)
        raise error
    return payload, meta


def redact_headers(headers: dict[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in (headers or {}).items():
        lowered = str(key).lower()
        if any(token in lowered for token in HEADER_REDACT_TOKENS):
            out[str(key)] = "***REDACTED***"
        else:
            out[str(key)] = str(value)
    return out


def redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if any(token in str(key).lower() for token in HEADER_REDACT_TOKENS):
                out[str(key)] = "***REDACTED***"
            else:
                out[str(key)] = redact_payload(item)
        return out
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    return value


def clip_raw(value: Any, *, max_chars: int = MAX_RAW_CHARS) -> Any:
    try:
        encoded = json.dumps(value, default=str)
    except TypeError:
        encoded = str(value)
    if len(encoded) <= max_chars:
        return value
    return {"_truncated": True, "chars": len(encoded), "preview": encoded[:max_chars]}


def public_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    return {key: item for key, item in (meta or {}).items() if key not in RAW_META_KEYS}


def vendor_call(backend: Any, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    try:
        return vendor_request(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        backend.last_meta = dict(getattr(exc, "vendor_meta", None) or {})
        raise


def _header_id(response: Any) -> str:
    headers = getattr(response, "headers", None) or {}
    for key in ("x-request-id", "x-parallel-request-id", "cf-ray", "fly-request-id"):
        value = headers.get(key)
        if value:
            return str(value)
    return ""


def _search_meta(meta: dict[str, Any], hits: list[dict[str, str]]) -> dict[str, Any]:
    urls = [str(hit.get("url") or "") for hit in hits if hit.get("url")]
    return {
        **meta,
        "n_hits": len(hits),
        "hit_urls": urls,
        "empty": len(hits) == 0,
    }


def _pop_flags(page: dict[str, Any]) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    for key in list(page):
        if key.startswith("_") and key != "_meta":
            flags[key[1:]] = page.pop(key)
    return flags


# Canonical ids. Aliases below keep old CLI names working.
SEARCH_ONLY_BACKENDS: tuple[str, ...] = (
    "parallel_turbo",
    "parallel_fast",
    "exa_fast",
    "exa_instant",
    "tavily_fast",
    "brave",
    "linkup_fast",
    "firecrawl",
)
SEARCH_FETCH_BACKENDS: tuple[str, ...] = (
    "fc_dev",
    "parallel_basic",
    "parallel_advanced",
    "exa_auto",
    "exa_deep",
    "tavily_basic",
    "tavily_advanced",
    "linkup_standard",
    "firecrawl",
)
BACKEND_ALIASES = {
    "exa": "exa_auto",
    "tavily": "tavily_fast",
    "linkup": "linkup_fast",
}
# Search-only exclusive rows cannot turn fetch on. Search+fetch exclusive rows
# cannot turn fetch off. firecrawl sits on both splits.
FETCH_FORBIDDEN = frozenset(SEARCH_ONLY_BACKENDS) - {"firecrawl"}
FETCH_REQUIRED = frozenset(SEARCH_FETCH_BACKENDS) - {"firecrawl"}

BACKENDS: dict[str, Callable[[], SearchBackend]] = {
    "parallel_basic": ParallelBasic,
    "parallel_turbo": ParallelTurbo,
    "parallel_fast": ParallelFast,
    "parallel_advanced": ParallelAdvanced,
    "firecrawl": FirecrawlSearch,
    "fc_dev": FirecrawlDevSearch,
    "exa_auto": ExaSearch,
    "exa": ExaSearch,
    "exa_deep": ExaDeep,
    "exa_fast": ExaFast,
    "exa_instant": ExaInstant,
    "linkup_fast": LinkupSearch,
    "linkup": LinkupSearch,
    "linkup_standard": LinkupStandard,
    "tavily_fast": TavilySearch,
    "tavily": TavilySearch,
    "tavily_basic": TavilyBasic,
    "tavily_advanced": TavilyAdvanced,
    "brave": BraveSearch,
}


def resolve_backend(name: str) -> str:
    return BACKEND_ALIASES.get(name, name)


def get_backend(name: str) -> SearchBackend:
    canonical = resolve_backend(name)
    factory = BACKENDS.get(canonical)
    if factory is None:
        known = ", ".join(sorted(set(resolve_backend(item) for item in BACKENDS)))
        raise KeyError(f"unknown search backend {name!r}; wired now: {known}")
    return factory()
