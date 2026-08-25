"""Write redacted vendor request/response hops to disk."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# UTC, filesystem-safe, lexicographically sortable. Example: 2026-08-24T16-29-16Z
RUN_STAMP_FMT = "%Y-%m-%dT%H-%M-%SZ"
RUN_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$")


def new_run_stamp() -> str:
    return datetime.now(timezone.utc).strftime(RUN_STAMP_FMT)


def split_label(allow_fetch: bool, split: str = "") -> str:
    name = (split or "").strip().lower().replace("_", "-")
    if name in {"search-fetch", "search-only"}:
        return name
    return "search-fetch" if allow_fetch else "search-only"


def vendor_slug(backend: str) -> str:
    return backend.replace("_", "-")


def vendor_rel(
    *,
    dataset: str,
    split: str,
    backend: str,
    allow_fetch: bool,
    run_stamp: str = "",
) -> Path:
    rel = Path(dataset or "adhoc") / split_label(allow_fetch, split) / vendor_slug(backend)
    stamp = (run_stamp or "").strip()
    if stamp:
        rel = rel / stamp
    return rel


def list_run_stamps(
    raw_dir: Path,
    *,
    dataset: str,
    split: str,
    backend: str,
    allow_fetch: bool,
) -> list[str]:
    parent = raw_dir / vendor_rel(dataset=dataset, split=split, backend=backend, allow_fetch=allow_fetch)
    if not parent.is_dir():
        return []
    return sorted(name for name in os.listdir(parent) if RUN_STAMP_RE.match(name) and (parent / name).is_dir())


def parse_run_stamp(raw: str) -> str:
    stamp = (raw or "").strip()
    if not stamp:
        return ""
    if not RUN_STAMP_RE.match(stamp):
        raise SystemExit(f"invalid --run-stamp {stamp!r}; expected {RUN_STAMP_FMT} e.g. 2026-08-24T16-29-16Z")
    return stamp


def resolve_run_stamp(
    raw_dir: Path,
    *,
    dataset: str,
    split: str,
    backend: str,
    allow_fetch: bool,
    run_stamp: str = "",
) -> str:
    stamp = parse_run_stamp(run_stamp)
    if stamp:
        return stamp
    found = list_run_stamps(raw_dir, dataset=dataset, split=split, backend=backend, allow_fetch=allow_fetch)
    if found:
        return found[-1]
    return ""


def arm_dir(
    root: Path,
    *,
    dataset: str,
    split: str,
    backend: str,
    allow_fetch: bool,
    task_id: str,
    run_stamp: str = "",
) -> Path:
    return (
        root
        / vendor_rel(
            dataset=dataset,
            split=split,
            backend=backend,
            allow_fetch=allow_fetch,
            run_stamp=run_stamp,
        )
        / task_id
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def write_hop_pair(hops_dir: Path, stem: str, request: Any, response: Any) -> dict[str, str]:
    req_path = hops_dir / f"{stem}.request.json"
    res_path = hops_dir / f"{stem}.response.json"
    write_json(req_path, request)
    write_json(res_path, response)
    return {"request": str(req_path), "response": str(res_path)}


def write_arm_raw(
    root: Path,
    *,
    dataset: str,
    split: str,
    backend: str,
    allow_fetch: bool,
    task_id: str,
    run: Any,
    row: dict[str, Any],
    run_stamp: str = "",
) -> Path:
    dest = arm_dir(
        root,
        dataset=dataset,
        split=split,
        backend=backend,
        allow_fetch=allow_fetch,
        task_id=task_id,
        run_stamp=run_stamp,
    )
    hops_dir = dest / "hops"
    hops_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {"searches": [], "fetches": []}
    for index, call in enumerate(getattr(run, "searches", None) or [], start=1):
        files["searches"].append(
            write_hop_pair(
                hops_dir,
                f"search-{index:02d}",
                getattr(call, "request", None),
                getattr(call, "response", None),
            )
        )
    for index, call in enumerate(getattr(run, "fetches", None) or [], start=1):
        files["fetches"].append(
            write_hop_pair(
                hops_dir,
                f"fetch-{index:02d}",
                getattr(call, "request", None),
                getattr(call, "response", None),
            )
        )
    summary = {key: value for key, value in row.items() if key not in {"source"}}
    summary["raw_files"] = files
    write_json(dest / "run.json", summary)
    notes = list(getattr(run, "notes", None) or [])
    if notes:
        write_json(dest / "notes.json", notes)
    return dest
