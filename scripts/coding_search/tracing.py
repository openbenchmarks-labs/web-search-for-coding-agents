"""Braintrust tracing for coding-search. No-ops when BRAINTRUST_API_KEY is unset."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import urljoin

import requests

DEFAULT_PROJECT_ID = "e07637e6-dbca-4813-8847-1d243d84cdbb"
DEFAULT_API_URL = "https://api.braintrust.dev"
DEFAULT_APP_URL = "https://www.braintrust.dev"

# Locked coding-search boards. Project name is what Braintrust shows;
# env id is filled after the first create-or-get.
DATASETS: dict[str, dict[str, Any]] = {
    "private": {
        "project_name": "web-coding-private",
        "env_key": "BRAINTRUST_PROJECT_ID_PRIVATE",
        "snapshot": "web-search/coding/2026-08-21/snapshot-2",
        "n": 100,
    },
    "public": {
        "project_name": "web-coding-public",
        "env_key": "BRAINTRUST_PROJECT_ID_PUBLIC",
        "snapshot": "web-search/coding/2026-08-20/snapshot-1",
        "n": 60,
    },
}
SPLITS = ("search-fetch", "search-only")

_EXPERIMENT: Any | None = None


def tracing_enabled() -> bool:
    return bool(os.getenv("BRAINTRUST_API_KEY"))


def parse_dataset(raw: str) -> str:
    name = (raw or "").strip().lower()
    if name not in DATASETS:
        allowed = ", ".join(DATASETS)
        raise SystemExit(f"unknown dataset {raw!r}; expected {allowed}")
    return name


def parse_split(raw: str) -> str:
    name = (raw or "").strip().lower().replace("_", "-")
    aliases = {"fetch": "search-fetch", "search": "search-only"}
    name = aliases.get(name, name)
    if name not in SPLITS:
        raise SystemExit(f"unknown split {raw!r}; expected {', '.join(SPLITS)}")
    return name


def vendor_slug(backend: str) -> str:
    return backend.replace("_", "-")


def experiment_name(*, split: str, backends: tuple[str, ...]) -> str:
    vendors = "+".join(vendor_slug(item) for item in backends)
    return f"{split}/{vendors}"


def experiment_tags(*, dataset: str, split: str, backends: tuple[str, ...], n: int) -> list[str]:
    spec = DATASETS[dataset]
    snapshot_tag = spec["snapshot"].replace("/", "-")
    tags = ["coding-search", dataset, split, f"n{n}", snapshot_tag]
    tags.extend(vendor_slug(item) for item in backends)
    return tags


def ensure_project(name: str) -> str:
    """Create-or-get a Braintrust project. Returns its id. Never prints the API key."""
    api_key = os.getenv("BRAINTRUST_API_KEY") or ""
    if not api_key:
        raise RuntimeError("BRAINTRUST_API_KEY is required to create projects")
    api = os.getenv("BRAINTRUST_API_URL", DEFAULT_API_URL).rstrip("/")
    body: dict[str, Any] = {"name": name}
    org = os.getenv("BRAINTRUST_ORG")
    if org:
        body["org_name"] = org
    response = requests.post(
        f"{api}/v1/project",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    project_id = payload.get("id")
    if not project_id:
        raise RuntimeError(f"Braintrust project {name!r} returned no id")
    return str(project_id)


def resolve_project(dataset: str) -> tuple[str, dict[str, Any]]:
    spec = DATASETS[parse_dataset(dataset)]
    project_id = (os.getenv(str(spec["env_key"])) or "").strip()
    if not project_id:
        project_id = ensure_project(str(spec["project_name"]))
        os.environ[str(spec["env_key"])] = project_id
    return project_id, spec


def current_experiment() -> Any | None:
    return _EXPERIMENT


def experiment_url(logger: Any | None) -> str | None:
    if logger is None:
        return None
    try:
        summary = logger.summarize(summarize_scores=False)
        if summary.experiment_url:
            return str(summary.experiment_url)
    except Exception:  # noqa: BLE001
        pass
    try:
        experiment_id = logger.id
        project_id = logger.project.id
    except Exception:  # noqa: BLE001
        return None
    app = os.getenv("BRAINTRUST_APP_URL", DEFAULT_APP_URL).rstrip("/")
    org = os.getenv("BRAINTRUST_ORG") or "Openbenchmarks"
    return f"{app}/app/{org}/p/{project_id}/experiments/{experiment_id}"


def init_tracing(
    *,
    experiment: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    project_id: str | None = None,
) -> Any | None:
    global _EXPERIMENT
    _EXPERIMENT = None
    if not tracing_enabled():
        return None
    os.environ.setdefault("BRAINTRUST_APP_URL", DEFAULT_APP_URL)
    os.environ.setdefault("BRAINTRUST_API_URL", DEFAULT_API_URL)
    os.environ.setdefault("BRAINTRUST_PROJECT_ID", DEFAULT_PROJECT_ID)
    project_id = (project_id or os.getenv("BRAINTRUST_PROJECT_ID") or DEFAULT_PROJECT_ID).strip()
    api_key = os.getenv("BRAINTRUST_API_KEY")
    app_url = os.getenv("BRAINTRUST_APP_URL", DEFAULT_APP_URL)
    name = (experiment or "").strip() or os.getenv("CODING_SEARCH_EXPERIMENT_ID") or ""
    if name:
        from braintrust import init

        _EXPERIMENT = init(
            project_id=project_id,
            experiment=name,
            description=description,
            metadata=metadata,
            tags=tags,
            api_key=api_key,
            app_url=app_url,
            set_current=True,
        )
        _ = _EXPERIMENT.id
        return _EXPERIMENT
    from braintrust import init_logger

    return init_logger(
        project_id=project_id,
        api_key=api_key,
        app_url=app_url,
        set_current=True,
        async_flush=False,
    )


def wrap_client(client: Any) -> Any:
    if not tracing_enabled():
        return client
    from braintrust import wrap_openai

    return wrap_openai(client)


@contextmanager
def agent_span(question: str, *, model: str, backend: str) -> Iterator[Any]:
    if not tracing_enabled():
        yield None
        return
    from braintrust import start_span

    with start_span(
        name="coding_search_agent",
        type="task",
        input=question,
        metadata={"model": model, "backend": backend},
    ) as span:
        yield span


@contextmanager
def search_span(query: str, *, backend: str) -> Iterator[Any]:
    if not tracing_enabled():
        yield None
        return
    from braintrust import start_span

    with start_span(
        name="web_search",
        type="tool",
        input=query,
        metadata={"backend": backend},
    ) as span:
        yield span


def log_span(span: Any, **fields: Any) -> None:
    if span is None:
        return
    span.log(**{key: value for key, value in fields.items() if value is not None})


@contextmanager
def fetch_span(url: str, *, backend: str) -> Iterator[Any]:
    if not tracing_enabled():
        yield None
        return
    from braintrust import start_span

    with start_span(
        name="web_fetch",
        type="tool",
        input=url,
        metadata={"backend": backend},
    ) as span:
        yield span


@contextmanager
def eval_span(*, task_id: str, backend: str, model: str, experiment_id: str) -> Iterator[Any]:
    if not tracing_enabled():
        yield None
        return
    fields = dict(
        name=task_id,
        type="eval",
        input={"task_id": task_id, "backend": backend},
        metadata={
            "task_id": task_id,
            "backend": backend,
            "model": model,
            "experiment_id": experiment_id,
        },
    )
    if _EXPERIMENT is not None:
        with _EXPERIMENT.start_span(**fields) as span:
            yield span
        return
    from braintrust import start_span

    with start_span(**fields) as span:
        yield span


@contextmanager
def judge_span(*, task_id: str, backend: str) -> Iterator[Any]:
    if not tracing_enabled():
        yield None
        return
    from braintrust import start_span

    with start_span(
        name="judge",
        type="score",
        input={"task_id": task_id, "backend": backend},
        metadata={"task_id": task_id, "backend": backend},
    ) as span:
        yield span


def permalink(span: Any) -> str | None:
    if span is None:
        return None
    try:
        link = span.permalink()
    except Exception:  # noqa: BLE001
        link = None
    if link:
        return str(link)
    project_id = os.getenv("BRAINTRUST_PROJECT_ID", DEFAULT_PROJECT_ID)
    return urljoin(os.getenv("BRAINTRUST_APP_URL", DEFAULT_APP_URL).rstrip("/") + "/", f"app/p/{project_id}")


def flush_tracing(logger: Any | None) -> None:
    if logger is None:
        return
    logger.flush()
