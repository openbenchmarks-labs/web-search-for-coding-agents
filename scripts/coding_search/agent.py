"""GPT-5.6 Sol loop with vendor search/fetch. Gold tokens in queries are flagged, not blocked."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .search import (
    DEFAULT_MAX_RESULTS,
    SearchBackend,
    get_backend,
    hit_text_stats,
    page_text_stats,
    public_meta,
)
from .tracing import agent_span, fetch_span, log_span, permalink, search_span

DEFAULT_MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"

WEB_SEARCH_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": (
        "Search the web. Required before you return an answer or file. "
        "Call it to find current docs. You may call it more than once "
        "with a new query. If you are unsure, search again rather than "
        "returning a wrong file."
    ),
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Focused search query for this hop."}},
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

WEB_FETCH_TOOL = {
    "type": "function",
    "name": "web_fetch",
    "description": (
        "Fetch a specific URL through this vendor's native extract/scrape API. "
        "Use after search when snippets are not enough."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "https URL to fetch."},
            "objective": {
                "type": "string",
                "description": "What you need from the page. Empty string for the full page.",
            },
        },
        "required": ["url", "objective"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class SearchCall:
    query: str
    backend: str
    hits: list[dict[str, str]]
    error: str | None = None
    latency_ms: int = 0
    n_hits: int = 0
    chars: int = 0
    tokens_est: int = 0
    turn: int = 0
    seq: int = 0
    endpoint: str = ""
    http_status: int | None = None
    http_ms: int | None = None
    vendor_request_id: str = ""
    hit_urls: list[str] = field(default_factory=list)
    empty: bool = False
    gold_token: str | None = None
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None

    def hop_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "backend": self.backend,
            "n_hits": self.n_hits,
            "chars": self.chars,
            "tokens_est": self.tokens_est,
            "latency_ms": self.latency_ms,
            "http_ms": self.http_ms,
            "http_status": self.http_status,
            "endpoint": self.endpoint,
            "vendor_request_id": self.vendor_request_id,
            "hit_urls": self.hit_urls,
            "empty": self.empty,
            "error": self.error,
            "gold_token": self.gold_token,
            "turn": self.turn,
            "seq": self.seq,
            "hits": [
                {
                    "url": str(hit.get("url") or ""),
                    "title": str(hit.get("title") or ""),
                    "snippet": str(hit.get("snippet") or "")[:1200],
                }
                for hit in (self.hits or [])
                if isinstance(hit, dict)
            ],
        }


@dataclass
class FetchCall:
    url: str
    backend: str
    title: str = ""
    content: str = ""
    objective: str = ""
    error: str | None = None
    latency_ms: int = 0
    chars: int = 0
    tokens_est: int = 0
    turn: int = 0
    seq: int = 0
    endpoint: str = ""
    http_status: int | None = None
    http_ms: int | None = None
    vendor_request_id: str = ""
    preview: str = ""
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None

    def hop_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "backend": self.backend,
            "title": self.title,
            "objective": self.objective,
            "chars": self.chars,
            "tokens_est": self.tokens_est,
            "latency_ms": self.latency_ms,
            "http_ms": self.http_ms,
            "http_status": self.http_status,
            "endpoint": self.endpoint,
            "vendor_request_id": self.vendor_request_id,
            "error": self.error,
            "turn": self.turn,
            "seq": self.seq,
            "preview": (self.preview or self.content or "")[:800],
        }


@dataclass
class LlmTurn:
    turn: int
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    n_tool_calls: int
    tool_names: list[str] = field(default_factory=list)
    ended: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRun:
    question: str
    answer: str
    model: str
    backend: str
    used_search: bool
    used_fetch: bool = False
    run_id: str = ""
    searches: list[SearchCall] = field(default_factory=list)
    fetches: list[FetchCall] = field(default_factory=list)
    llm_turns: list[LlmTurn] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    permalink: str | None = None
    wall_ms: int = 0
    llm_ms: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    hop_seq: int = 0

    def metrics(self) -> dict[str, Any]:
        return {
            "wall_ms": self.wall_ms,
            "llm_ms": self.llm_ms,
            "search_count": len(self.searches),
            "fetch_count": len(self.fetches),
            "search_ms": sum(call.latency_ms for call in self.searches),
            "fetch_ms": sum(call.latency_ms for call in self.fetches),
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "gold_token_queries": sum(1 for call in self.searches if call.gold_token),
        }


def run_agent(
    question: str,
    *,
    client: Any,
    backend: SearchBackend | None = None,
    backend_name: str = "parallel_basic",
    model: str = DEFAULT_MODEL,
    max_turns: int = 32,
    max_searches: int = 5,
    max_fetches: int = 5,
    max_results: int = DEFAULT_MAX_RESULTS,
    search_fn: Callable[[str], list[dict[str, str]]] | None = None,
    fetch_fn: Callable[[str, str], dict[str, str]] | None = None,
    system: str,
    allow_search: bool = True,
    allow_fetch: bool = True,
    leak_needles: list[str] | None = None,
) -> AgentRun:
    backend = backend or (None if search_fn else get_backend(backend_name))
    provider = backend.name if backend is not None else backend_name
    if not allow_search:
        provider = "no_search"
    run = AgentRun(
        question=question,
        answer="",
        model=model,
        backend=provider,
        used_search=False,
        run_id=uuid.uuid4().hex,
    )
    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    tools = []
    if allow_search:
        tools.append(WEB_SEARCH_TOOL)
    if allow_fetch:
        tools.append(WEB_FETCH_TOOL)
    started = time.perf_counter()
    with agent_span(question, model=model, backend=provider) as root:
        for turn in range(max_turns):
            llm_started = time.perf_counter()
            response = client.responses.create(
                model=model,
                reasoning={"effort": REASONING_EFFORT},
                tools=tools,
                input=conversation,
                max_output_tokens=4000,
                store=False,
            )
            llm_ms = int((time.perf_counter() - llm_started) * 1000)
            prompt_tokens, completion_tokens = _usage(response)
            run.llm_ms += llm_ms
            run.llm_prompt_tokens += prompt_tokens
            run.llm_completion_tokens += completion_tokens
            calls = _function_calls(response)
            run.llm_turns.append(
                LlmTurn(
                    turn=turn,
                    latency_ms=llm_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    n_tool_calls=len(calls),
                    tool_names=[_call_name(call) for call in calls],
                    ended=not calls,
                )
            )
            if not calls:
                run.answer = (getattr(response, "output_text", None) or "").strip()
                run.wall_ms = int((time.perf_counter() - started) * 1000)
                log_span(
                    root,
                    output=run.answer,
                    metadata={
                        "used_search": run.used_search,
                        "used_fetch": run.used_fetch,
                        "search_count": len(run.searches),
                        "fetch_count": len(run.fetches),
                        "notes": run.notes,
                        **run.metrics(),
                    },
                )
                run.permalink = permalink(root)
                return run
            _append_model_output(conversation, response)
            for call in calls:
                name = _call_name(call)
                if name == "web_search":
                    _run_search(
                        call,
                        run=run,
                        conversation=conversation,
                        backend=backend,
                        provider=provider,
                        max_searches=max_searches,
                        max_results=max_results,
                        search_fn=search_fn,
                        turn=turn,
                        leak_needles=leak_needles,
                    )
                elif name == "web_fetch":
                    if not allow_fetch:
                        conversation.append(_tool_output(call, {"error": "web_fetch is not available; search again"}))
                        continue
                    _run_fetch(
                        call,
                        run=run,
                        conversation=conversation,
                        backend=backend,
                        provider=provider,
                        max_fetches=max_fetches,
                        fetch_fn=fetch_fn,
                        turn=turn,
                    )
                else:
                    conversation.append(_tool_output(call, {"error": f"unknown tool {name}"}))
        run.notes.append("hit max turns without a final answer")
        run.wall_ms = int((time.perf_counter() - started) * 1000)
        log_span(root, output=run.answer, metadata={"notes": run.notes, **run.metrics()})
        run.permalink = permalink(root)
        return run


def _next_seq(run: AgentRun) -> int:
    run.hop_seq += 1
    return run.hop_seq


def _hidden_token_in_query(query: str, needles: list[str] | None) -> str | None:
    if not needles:
        return None
    hay = query.casefold()
    for needle in needles:
        token = (needle or "").strip()
        if len(token) < 4:
            continue
        if token.casefold() in hay:
            return token
    return None


def _vendor_meta(backend: SearchBackend | None) -> dict[str, Any]:
    return dict(getattr(backend, "last_meta", None) or {})


def _run_search(
    call: Any,
    *,
    run: AgentRun,
    conversation: list[dict[str, Any]],
    backend: SearchBackend | None,
    provider: str,
    max_searches: int,
    max_results: int,
    search_fn: Callable[[str], list[dict[str, str]]] | None,
    turn: int,
    leak_needles: list[str] | None,
) -> None:
    if len(run.searches) >= max_searches:
        conversation.append(_tool_output(call, {"error": "search budget exhausted; answer now"}))
        run.notes.append("search budget exhausted")
        return
    query = str(_call_args(call).get("query") or "").strip()
    if not query:
        conversation.append(_tool_output(call, {"error": "query is required"}))
        return
    with search_span(query, backend=provider) as tool_span:
        search_started = time.perf_counter()
        meta: dict[str, Any] = {}
        gold_token = _hidden_token_in_query(query, leak_needles)
        if gold_token:
            run.notes.append(f"query contained gold token:{gold_token}")
        try:
            hits = search_fn(query) if search_fn else backend.search(query, max_results=max_results)
            if not search_fn:
                meta = _vendor_meta(backend)
            error = None
        except Exception as exc:  # noqa: BLE001
            hits = []
            error = f"{type(exc).__name__}: {exc}"
            meta = _vendor_meta(backend) or dict(getattr(exc, "vendor_meta", None) or {})
        latency_ms = int((time.perf_counter() - search_started) * 1000)
        stats = hit_text_stats(hits)
        hit_urls = [str(hit.get("url") or "") for hit in hits if hit.get("url")] or list(meta.get("hit_urls") or [])
        hop = SearchCall(
            query=query,
            backend=provider,
            hits=hits,
            error=error,
            latency_ms=latency_ms,
            n_hits=stats["n_hits"],
            chars=stats["chars"],
            tokens_est=stats["tokens_est"],
            turn=turn,
            seq=_next_seq(run),
            endpoint=str(meta.get("endpoint") or ""),
            http_status=meta.get("http_status"),
            http_ms=meta.get("http_ms"),
            vendor_request_id=str(meta.get("vendor_request_id") or ""),
            hit_urls=hit_urls,
            empty=stats["n_hits"] == 0,
            gold_token=gold_token,
            request=meta.get("request") if isinstance(meta.get("request"), dict) else None,
            response=meta.get("response") if isinstance(meta.get("response"), dict) else None,
        )
        log_span(
            tool_span,
            input=query,
            output=[{"url": hit.get("url"), "title": hit.get("title")} for hit in hits],
            metadata={**public_meta(meta), **({"gold_token": gold_token} if gold_token else {}), "n_hits": hop.n_hits, "error": error},
            metrics={"latency_ms": latency_ms, "chars": hop.chars},
        )
    run.used_search = True
    run.searches.append(hop)
    tool_result: dict[str, Any] = {"backend": provider, "query": query, "hits": hits}
    if error:
        tool_result["error"] = error
    conversation.append(_tool_output(call, tool_result))


def _run_fetch(
    call: Any,
    *,
    run: AgentRun,
    conversation: list[dict[str, Any]],
    backend: SearchBackend | None,
    provider: str,
    max_fetches: int,
    fetch_fn: Callable[[str, str], dict[str, str]] | None,
    turn: int,
) -> None:
    if len(run.fetches) >= max_fetches:
        conversation.append(_tool_output(call, {"error": "fetch budget exhausted; answer now"}))
        run.notes.append("fetch budget exhausted")
        return
    args = _call_args(call)
    url = str(args.get("url") or "").strip()
    objective = str(args.get("objective") or "").strip()
    if not url.startswith(("http://", "https://")):
        conversation.append(_tool_output(call, {"error": "url must be http(s)"}))
        return
    with fetch_span(url, backend=provider) as tool_span:
        fetch_started = time.perf_counter()
        meta: dict[str, Any] = {}
        try:
            page = fetch_fn(url, objective) if fetch_fn else backend.fetch(url, objective=objective)
            meta = dict(page.pop("_meta", None) or {})
            if not meta and not fetch_fn:
                meta = _vendor_meta(backend)
            error = None
        except Exception as exc:  # noqa: BLE001
            page = {"url": url, "title": "", "content": ""}
            error = f"{type(exc).__name__}: {exc}"
            meta = _vendor_meta(backend) or dict(getattr(exc, "vendor_meta", None) or {})
        latency_ms = int((time.perf_counter() - fetch_started) * 1000)
        stats = page_text_stats(page)
        content = str(page.get("content") or "")
        hop = FetchCall(
            url=str(page.get("url") or url),
            backend=provider,
            title=str(page.get("title") or ""),
            content=content,
            objective=objective,
            error=error,
            latency_ms=latency_ms,
            chars=stats["chars"],
            tokens_est=stats["tokens_est"],
            turn=turn,
            seq=_next_seq(run),
            endpoint=str(meta.get("endpoint") or ""),
            http_status=meta.get("http_status"),
            http_ms=meta.get("http_ms"),
            vendor_request_id=str(meta.get("vendor_request_id") or ""),
            preview=content[:800],
            request=meta.get("request") if isinstance(meta.get("request"), dict) else None,
            response=meta.get("response") if isinstance(meta.get("response"), dict) else None,
        )
        log_span(
            tool_span,
            input=url,
            output=hop.preview,
            metadata={**public_meta(meta), "title": hop.title, "error": error},
            metrics={"latency_ms": latency_ms, "chars": hop.chars},
        )
    run.used_fetch = True
    run.fetches.append(hop)
    fetch_result: dict[str, Any] = {
        "backend": provider,
        "url": page.get("url") or url,
        "title": page.get("title") or "",
        "content": page.get("content") or "",
    }
    if error:
        fetch_result["error"] = error
    conversation.append(_tool_output(call, fetch_result))


def _function_calls(response: Any) -> list[Any]:
    return [item for item in (getattr(response, "output", None) or []) if _item_type(item) == "function_call"]


def _item_type(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("type") or "")
    return str(getattr(item, "type", "") or "")


def _call_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("name") or "")
    return str(getattr(call, "name", "") or "")


def _call_id(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("call_id") or call.get("id") or "")
    return str(getattr(call, "call_id", None) or getattr(call, "id", "") or "")


def _call_args(call: Any) -> dict[str, Any]:
    raw = call.get("arguments") if isinstance(call, dict) else getattr(call, "arguments", "")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _append_model_output(conversation: list[dict[str, Any]], response: Any) -> None:
    for item in getattr(response, "output", None) or []:
        if _item_type(item) in {"reasoning", "message"}:
            continue
        conversation.append(_item_as_input(item))


def _item_as_input(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    payload = {
        "type": getattr(item, "type", None),
        "id": getattr(item, "id", None),
        "call_id": getattr(item, "call_id", None),
        "name": getattr(item, "name", None),
        "arguments": getattr(item, "arguments", None),
        "status": getattr(item, "status", None),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _tool_output(call: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": _call_id(call), "output": json.dumps(payload, ensure_ascii=False)}


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)
