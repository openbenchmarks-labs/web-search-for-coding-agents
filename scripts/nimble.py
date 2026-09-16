"""Shared Nimble request shapes and source-only response parsing."""

from __future__ import annotations

import math
import os
from typing import Any

SEARCH_URL = "https://sdk.nimbleway.com/v2/search"
EXTRACT_URL = "https://sdk.nimbleway.com/v2/extract"
SEARCH_PRICES = {"lite": 0.0011, "standard": 0.005}


def search_body(query: str, depth: str, max_results: int) -> dict[str, Any]:
    if depth not in SEARCH_PRICES:
        raise ValueError(f"Unsupported Nimble search depth: {depth}")
    return {"query": query, "search_depth": depth, "full_content": False,
            "focus": "general", "max_results": max_results}


def search_hits(payload: Any, max_results: int) -> list[dict[str, str]]:
    rows = payload.get("results") if isinstance(payload, dict) else None
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("url"), str):
            continue
        url = row["url"].strip()
        if not url or url in seen:
            continue
        seen.add(url)
        hits.append({"url": url, "title": str(row.get("title") or ""),
                     "snippet": str(row.get("description") or "")})
        if len(hits) >= max_results:
            break
    return hits


def extract_page(payload: Any, url: str, max_chars: int) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("status") not in (None, "success"):
        raise RuntimeError("Nimble Extract returned an unsuccessful response")
    status = payload.get("status_code")
    if status is not None and not 200 <= int(status) < 300:
        raise RuntimeError(f"Nimble Extract upstream HTTP {status}")
    data = payload.get("data") or {}
    text = data.get("markdown") if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Nimble Extract returned no markdown content")
    return {"requested_url": url, "final_url": payload.get("url") or url,
            "title": "", "text": text[:max_chars], "truncated": len(text) > max_chars,
            "fetch_provider": "nimble_extract"}


def extract_price() -> float | None:
    raw = os.getenv("NIMBLE_EXTRACT_USD_PER_REQUEST", "").strip()
    if not raw:
        return None
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError("NIMBLE_EXTRACT_USD_PER_REQUEST must be finite and nonnegative")
    return value
