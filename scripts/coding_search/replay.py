"""Persist the judge's exact inputs and re-score saved runs offline.

The needles judge is deterministic given (source, task fields, observed URLs,
retrieved text). Hop summaries clip snippets and fetch bodies for size, so they
cannot reproduce a verdict that was grounded past the clip. Each result row
therefore carries `judge_evidence` (the exact URLs and text the judge saw, plus a
sha256) and `judge_task` (the task fields the judge reads). `replay` re-scores
rows from that evidence with no API keys or network calls.

Rows written before `judge_evidence` existed are replayed from their hop
summaries and labelled `hop_summaries`: a mismatch there shows where the old
summaries cannot reproduce the live verdict.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from .judge import hop_retrieved_text, hop_urls, score_needles
from .tasks import CodingTask

EVIDENCE_VERSION = 1
JUDGE_TASK_FIELDS = ("outfile", "required", "query_forbidden", "cite_urls")


def text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def code_rev() -> str:
    """Git commit of the harness (judge + adapters) that produced a run."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def judge_evidence(run: Any) -> dict[str, Any]:
    """Exact grounding inputs passed to score_needles for this run."""
    retrieved = hop_retrieved_text(run)
    return {
        "version": EVIDENCE_VERSION,
        "code_rev": code_rev(),
        "observed_urls": hop_urls(run),
        "retrieved_text": retrieved,
        "sha256": text_sha256(retrieved),
    }


def judge_task(task: CodingTask) -> dict[str, Any]:
    return {
        "outfile": task.outfile,
        "required": list(task.required),
        "query_forbidden": list(task.query_forbidden),
        "cite_urls": bool(task.cite_urls),
    }


def task_from_row(row: dict[str, Any], tasks_by_id: dict[str, CodingTask] | None = None) -> CodingTask | None:
    fields = row.get("judge_task")
    if isinstance(fields, dict):
        return CodingTask(
            id=str(row.get("id") or ""),
            repo="",
            outfile=str(fields.get("outfile") or "submission.py"),
            prompt="",
            required=list(fields.get("required") or []),
            query_forbidden=list(fields.get("query_forbidden") or []),
            cite_urls=bool(fields.get("cite_urls", True)),
        )
    if tasks_by_id:
        return tasks_by_id.get(str(row.get("id") or ""))
    return None


def replay_row(row: dict[str, Any], tasks_by_id: dict[str, CodingTask] | None = None) -> dict[str, Any]:
    """Re-score one saved row. Returns a report dict; never raises on bad rows."""
    report: dict[str, Any] = {
        "id": row.get("id"),
        "backend": row.get("backend"),
        "live_passed": row.get("passed"),
        "replay_passed": None,
        "evidence_source": None,
        "status": "skipped",
        "reason": "",
    }
    if row.get("error"):
        report["reason"] = "run errored; nothing to replay"
        return report
    if "source" not in row:
        report["reason"] = "row has no submission source"
        return report
    task = task_from_row(row, tasks_by_id)
    if task is None:
        report["reason"] = "no judge_task in row and task not found in DATASETS_ROOT"
        return report

    evidence = row.get("judge_evidence")
    if isinstance(evidence, dict):
        retrieved = str(evidence.get("retrieved_text") or "")
        expected = evidence.get("sha256")
        if expected and text_sha256(retrieved) != expected:
            report.update(status="corrupt", reason="judge_evidence sha256 mismatch")
            return report
        observed = list(evidence.get("observed_urls") or [])
        report["evidence_source"] = "judge_evidence"
    else:
        hops = {"search_hops": row.get("search_hops") or [], "fetch_hops": row.get("fetch_hops") or []}
        observed = hop_urls(hops)
        retrieved = hop_retrieved_text(hops)
        report["evidence_source"] = "hop_summaries"

    score = score_needles(row.get("source") or "", task, observed_urls=observed, retrieved_text=retrieved)
    live_notes = list((row.get("score") or {}).get("notes") or [])
    report["replay_passed"] = score.passed
    report["replay_notes"] = score.notes
    same = score.passed == row.get("passed") and score.notes == live_notes
    report["status"] = "match" if same else "mismatch"
    if not same:
        report["live_notes"] = live_notes
    return report


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return list(payload["results"])
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"{path}: expected an eval summary, a run.json row or a list of rows")


def replay_paths(paths: list[Path], *, tasks_by_id: dict[str, CodingTask] | None = None) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        for row in load_rows(path):
            report = replay_row(row, tasks_by_id)
            report["file"] = str(path)
            reports.append(report)
    return reports


def print_reports(reports: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"match": 0, "mismatch": 0, "corrupt": 0, "skipped": 0}
    for report in reports:
        counts[report["status"]] = counts.get(report["status"], 0) + 1
        live = "pass" if report["live_passed"] else "fail"
        replay = "-" if report["replay_passed"] is None else ("pass" if report["replay_passed"] else "fail")
        line = f"{report['status']:8} {report['id']} {report['backend']} live={live} replay={replay}"
        if report.get("evidence_source"):
            line += f" from={report['evidence_source']}"
        if report.get("reason"):
            line += f" ({report['reason']})"
        elif report["status"] == "mismatch":
            line += f" live_notes={report.get('live_notes')} replay_notes={report.get('replay_notes')}"
        print(line, flush=True)
    print(
        "replayed={n} match={match} mismatch={mismatch} corrupt={corrupt} skipped={skipped}".format(
            n=counts["match"] + counts["mismatch"], **counts
        ),
        flush=True,
    )
    return counts
