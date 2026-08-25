"""Load locked SAP / ServiceNow / Workday tickets from the datasets repo."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .env import datasets_root


@dataclass
class CodingTask:
    id: str
    repo: str
    outfile: str
    prompt: str
    required: list[str]
    query_forbidden: list[str] = field(default_factory=list)
    hops: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)
    cite_urls: bool = True
    max_turns: int = 32
    max_searches: int = 5
    max_fetches: int = 5
    kind: str = "patch"
    task_dir: Path = field(default_factory=Path)

    def leak_needles(self) -> list[str]:
        """Gold path tokens. Logged when a search query contains them; not withheld."""
        return list(self.query_forbidden)

    def gold_source(self) -> str:
        return (self.task_dir / "gold" / self.outfile).read_text()

    def starter_source(self) -> str:
        return (self.task_dir / "starter" / self.outfile).read_text()


def catalog_path(root: Path | None = None) -> Path:
    base = root or datasets_root()
    tasks = base / "web-search" / "coding" / "tasks"
    for name in ("catalog.json", "catalog.yaml"):
        path = tasks / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"no catalog under {tasks}")


def load_catalog(root: Path | None = None) -> list[dict[str, Any]]:
    path = catalog_path(root)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
    else:
        import yaml

        payload = yaml.safe_load(path.read_text())
    defaults = dict(payload.get("defaults") or {})
    items = []
    for row in payload.get("items") or []:
        items.append({**defaults, **row})
    return items


def load_tasks(*, root: Path | None = None, ids: list[str] | None = None) -> list[CodingTask]:
    base = root or datasets_root()
    tasks_dir = base / "web-search" / "coding" / "tasks"
    wanted = set(ids or [])
    out: list[CodingTask] = []
    for row in load_catalog(base):
        if wanted and row["id"] not in wanted:
            continue
        dest = tasks_dir / row["id"]
        prompt_path = dest / "prompt.txt"
        out.append(
            CodingTask(
                id=row["id"],
                repo=str(row.get("repo") or ""),
                outfile=str(row["outfile"]),
                prompt=prompt_path.read_text() if prompt_path.is_file() else str(row.get("prompt") or ""),
                required=list(row.get("required") or []),
                query_forbidden=list(row.get("query_forbidden") or []),
                hops=list(row.get("hops") or []),
                docs=list(row.get("docs") or []),
                cite_urls=bool(row.get("cite_urls", True)),
                max_turns=int(row.get("max_turns") or 32),
                max_searches=int(row.get("max_searches") or 5),
                max_fetches=int(row.get("max_fetches") or 5),
                kind=str(row.get("kind") or "patch"),
                task_dir=dest,
            )
        )
    if wanted and len(out) != len(wanted):
        missing = wanted - {item.id for item in out}
        raise KeyError(f"unknown task ids: {sorted(missing)}")
    return out
