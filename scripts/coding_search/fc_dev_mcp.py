"""Minimal MCP client for Firecrawl's hosted `firecrawl_developer_search` (the
developer index), so it can be an openbenchmarks search backend.

developer_search is MCP-only (no REST); it lives on https://mcp.firecrawl.dev/<KEY>/v2/mcp
(streamable HTTP, key in the URL path, Cloudflare in front). Result is MARKDOWN:

    ## [web:<url>] (<source>) <title>
    <snippet>

This is a trimmed copy of devdex/harness/mcp_client.py (stdlib-only): a fresh
client per call (handshake each time) — simple and session-expiry-proof at this scale.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

PROTO = "2024-11-05"
UA = "ob-fc-dev/1.0"  # Cloudflare 403s urllib's default UA
_HEADER = re.compile(r"^##\s*\[(?:\w+:)?([^\]]+)\]\s*(?:\([^)]*\)\s*)?(.*)$")


class _McpHttp:
    def __init__(self, url: str) -> None:
        self.url, self.sid = url, None
        self._init()

    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": UA,
            "MCP-Protocol-Version": PROTO,
        }
        if self.sid:
            h["Mcp-Session-Id"] = self.sid
        return h

    def _post(self, body: dict, timeout: int) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=self._headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode(errors="replace")
            if not self.sid:
                self.sid = r.headers.get("mcp-session-id") or r.headers.get("Mcp-Session-Id")
            ctype = r.headers.get("content-type") or ""
        if "event-stream" in ctype:
            frames = [l[5:].strip() for l in raw.splitlines() if l.startswith("data:")]
            return json.loads(frames[-1]) if frames else {}
        return json.loads(raw or "{}")

    def _init(self) -> None:
        self._post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": PROTO, "capabilities": {},
                               "clientInfo": {"name": "ob-fc-dev", "version": "1"}}}, 60)
        try:
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}, 30)
        except Exception:
            pass

    def call_tool(self, name: str, args: dict, timeout: int = 60) -> dict:
        out = self._post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                          "params": {"name": name, "arguments": args or {}}}, timeout)
        if out.get("error"):
            raise RuntimeError(json.dumps(out["error"])[:200])
        return out.get("result") or {}


def _mcp_url() -> str:
    key = os.environ.get("FIRECRAWL_API_KEY") or ""
    if not key:
        raise RuntimeError("FIRECRAWL_API_KEY is required for fc_dev")
    return os.environ.get("FIRECRAWL_MCP_URL", f"https://mcp.firecrawl.dev/{key}/v2/mcp")


def _parse_markdown_hits(text: str, max_results: int) -> list[dict[str, str]]:
    """Parse `## [web:url] (web) Title` blocks + following snippet into hits."""
    hits: list[dict[str, str]] = []
    cur: dict[str, str] | None = None
    snippet: list[str] = []
    for line in (text or "").splitlines():
        m = _HEADER.match(line.strip())
        if m:
            if cur:
                cur["snippet"] = " ".join(snippet).strip()[:500]
                hits.append(cur)
                snippet = []
            url, title = m.group(1).strip(), m.group(2).strip()
            cur = {"url": url, "title": title, "snippet": ""}
        elif cur is not None and line.strip():
            snippet.append(line.strip())
    if cur:
        cur["snippet"] = " ".join(snippet).strip()[:500]
        hits.append(cur)
    return hits[:max_results]


def developer_search(query: str, *, max_results: int = 10) -> list[dict[str, str]]:
    client = _McpHttp(_mcp_url())
    r = client.call_tool("firecrawl_developer_search", {"query": query, "k": max_results})
    text = "".join(b.get("text") or "" for b in (r.get("content") or []) if b.get("type") == "text")
    return _parse_markdown_hits(text, max_results)
