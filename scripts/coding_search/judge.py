"""Needles judge: compile + gold strings + grounded # source: URL.

A cite is grounded if the URL appeared in search/fetch hops, or if the needle
itself showed up in retrieved hit snippets / fetch bodies. Search queries are
not retrieved text: putting a gold token in the query and citing a URL that
never came back is still ungrounded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .tasks import CodingTask

FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", re.S)
COMMENT_URL = re.compile(r"https?://[^\s#]+", re.I)


@dataclass
class JudgeScore:
    passed: bool
    compiled: bool
    output: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "compiled": self.compiled,
            "output": self.output,
            "notes": list(self.notes),
        }


def extract_code(answer: str, outfile: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""
    fences = [block.strip() for block in FENCE.findall(text) if block.strip()]
    if fences:
        return fences[-1]
    if text.startswith("```"):
        return ""
    if outfile.endswith(".py") and ("import " in text or "def " in text):
        return text
    return text


def patch_needles(task: CodingTask) -> list[str]:
    found: list[str] = []
    for item in list(task.required) + list(task.query_forbidden):
        token = (item or "").strip()
        if len(token) < 4 or token in found:
            continue
        found.append(token)
    return found


def comment_source_urls(source: str) -> list[str]:
    found: list[str] = []
    for line in (source or "").splitlines():
        if not line.lstrip().startswith("#"):
            continue
        for match in COMMENT_URL.findall(line):
            url = match.rstrip(").,;\"'")
            if url not in found:
                found.append(url)
    return found


def needle_source_urls(source: str, needles: list[str]) -> dict[str, str | None]:
    cited: dict[str, str | None] = {needle: None for needle in needles}
    last_url: str | None = None
    for line in (source or "").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            urls = comment_source_urls(line)
            if urls:
                last_url = urls[-1]
            continue
        for needle in needles:
            if needle in line and last_url:
                cited[needle] = last_url
    return cited


def normalize_cite_url(url: str) -> str:
    raw = (url or "").strip().rstrip(").,;\"'")
    host_path = raw.split("://", 1)[-1]
    host_path = host_path.split("#", 1)[0].split("?", 1)[0]
    host_path = host_path.lower().removeprefix("www.")
    return host_path.rstrip("/") or host_path


def url_was_observed(cited: str, observed: list[str]) -> bool:
    key = normalize_cite_url(cited)
    if not key:
        return False
    for item in observed:
        other = normalize_cite_url(item)
        if not other:
            continue
        if key == other or key.startswith(other) or other.startswith(key):
            return True
    return False


def hop_urls(run: Any) -> list[str]:
    found: list[str] = []
    for search in _searches(run):
        for url in list(getattr(search, "hit_urls", None) or (search.get("hit_urls") if isinstance(search, dict) else None) or []):
            if url and url not in found:
                found.append(url)
        for hit in _hits(search):
            url = str((hit or {}).get("url") or "")
            if url and url not in found:
                found.append(url)
    for fetch in _fetches(run):
        url = str(_fetch_url(fetch) or "")
        if url and url not in found:
            found.append(url)
    return found


def hop_retrieved_text(run: Any) -> str:
    """Title + snippet + fetch body shown to the model. Does not include queries."""
    parts: list[str] = []
    for search in _searches(run):
        for hit in _hits(search):
            if not isinstance(hit, dict):
                continue
            parts.append(str(hit.get("title") or ""))
            parts.append(str(hit.get("snippet") or hit.get("text") or hit.get("content") or ""))
    for fetch in _fetches(run):
        parts.append(_fetch_title(fetch))
        parts.append(_fetch_body(fetch))
    return "\n".join(part for part in parts if part)


def needle_was_retrieved(needle: str, retrieved_text: str | None) -> bool:
    token = (needle or "").strip()
    if not token or retrieved_text is None:
        return False
    return token in retrieved_text


def _searches(run: Any) -> list[Any]:
    if run is None:
        return []
    if isinstance(run, dict):
        return list(run.get("searches") or run.get("search_hops") or [])
    return list(getattr(run, "searches", None) or [])


def _fetches(run: Any) -> list[Any]:
    if run is None:
        return []
    if isinstance(run, dict):
        return list(run.get("fetches") or run.get("fetch_hops") or [])
    return list(getattr(run, "fetches", None) or [])


def _hits(search: Any) -> list[Any]:
    if isinstance(search, dict):
        return list(search.get("hits") or [])
    return list(getattr(search, "hits", None) or [])


def _fetch_url(fetch: Any) -> str:
    if isinstance(fetch, dict):
        return str(fetch.get("url") or "")
    return str(getattr(fetch, "url", "") or "")


def _fetch_title(fetch: Any) -> str:
    if isinstance(fetch, dict):
        return str(fetch.get("title") or "")
    return str(getattr(fetch, "title", "") or "")


def _fetch_body(fetch: Any) -> str:
    if isinstance(fetch, dict):
        return str(fetch.get("content") or fetch.get("preview") or "")
    return str(getattr(fetch, "content", "") or getattr(fetch, "preview", "") or "")


def score_needles(
    source: str,
    task: CodingTask,
    *,
    observed_urls: list[str] | None = None,
    retrieved_text: str | None = None,
) -> JudgeScore:
    text = source or ""
    if not text.strip():
        return JudgeScore(False, False, "", ["empty submission"])
    compiled = True
    notes: list[str] = []
    try:
        compile(text, task.outfile or "submission.py", "exec")
    except SyntaxError as exc:
        compiled = False
        notes.append(f"{exc.__class__.__name__}: {exc}")
    needles = patch_needles(task)
    missing = [needle for needle in needles if needle not in text]
    notes.extend(f"missing:{needle}" for needle in missing)
    check_grounding = observed_urls is not None or retrieved_text is not None
    if task.cite_urls and compiled and not missing and check_grounding:
        cited = needle_source_urls(text, needles)
        for needle, url in cited.items():
            if not url:
                notes.append(f"uncited:{needle}")
                continue
            url_ok = observed_urls is not None and url_was_observed(url, observed_urls or [])
            text_ok = needle_was_retrieved(needle, retrieved_text)
            if not url_ok and not text_ok:
                notes.append(f"ungrounded:{needle}")
    passed = compiled and not missing and not any(
        note.startswith(("uncited:", "ungrounded:")) for note in notes
    )
    output = "NEEDLES_OK" if passed else "NEEDLES_FAIL " + "; ".join(notes)
    return JudgeScore(passed, compiled, output, notes)
