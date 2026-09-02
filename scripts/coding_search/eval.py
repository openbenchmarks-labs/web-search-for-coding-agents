"""Run locked coding-search tickets against Parallel, Firecrawl, or Exa."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import DEFAULT_MODEL, run_agent
from .env import datasets_root, llm_transport, load_environment, make_llm_client
from .judge import extract_code, hop_retrieved_text, hop_urls, score_needles
from .rawlog import new_run_stamp, parse_run_stamp, write_arm_raw
from .search import (
    BACKENDS,
    FETCH_FORBIDDEN,
    FETCH_REQUIRED,
    SEARCH_FETCH_BACKENDS,
    SEARCH_ONLY_BACKENDS,
    get_backend,
    resolve_backend,
)
from .tasks import load_tasks
from .tracing import (
    DATASETS,
    eval_span,
    experiment_name,
    experiment_tags,
    experiment_url,
    flush_tracing,
    init_tracing,
    judge_span,
    log_span,
    parse_dataset,
    parse_split,
    resolve_project,
    ensure_project,
    tracing_enabled,
    vendor_slug,
    wrap_client,
)

PATCH_SYSTEM = """You edit an existing production script. Make the smallest change that satisfies the ticket.
Do not rewrite the file from scratch. Use the latest public APIs. Do not invent symbols that are not in the current docs.
You have web_search and web_fetch. You must call web_search before you return the file. Do not rely on memory for params, paths, or versions. Search for the current docs, then fetch a specific URL when snippets are not enough.
For every API version or field you add, put a `# source: https://...` comment on the line above with the search or fetch URL where you saw it. Do not use a name you cannot cite from a result you received.
Do not return until you are sure the path, version, and field names match current docs. If you are unsure, search or fetch again. Using more of the search budget is better than returning a wrong file. Only stop early when you are sure. Return only the full updated file."""
PATCH_SYSTEM_SEARCH_ONLY = """You edit an existing production script. Make the smallest change that satisfies the ticket.
Do not rewrite the file from scratch. Use the latest public APIs. Do not invent symbols that are not in the current docs.
You have web_search. You do not have web_fetch. You must call web_search before you return the file. Do not rely on memory for params, paths, or versions. Search for the current docs. If snippets are not enough, search again with a more specific query.
For every API version or field you add, put a `# source: https://...` comment on the line above with the search-result URL where you saw it. Do not use a name you cannot cite from a result you received.
Do not return until you are sure the path, version, and field names match current docs. If you are unsure, search again. Using more of the search budget is better than returning a wrong file. Only stop early when you are sure. Return only the full updated file."""


def parse_arms(raw: str, *, split: str = "") -> tuple[str, ...]:
    text = (raw or "all").strip()
    if text == "all":
        if split == "search-only":
            return SEARCH_ONLY_BACKENDS
        if split == "search-fetch":
            return SEARCH_FETCH_BACKENDS
        raise SystemExit("--backend all requires --split search-only or search-fetch")
    arms = tuple(resolve_backend(item.strip()) for item in text.split(",") if item.strip())
    unknown = [item for item in arms if item not in BACKENDS]
    if unknown:
        known = ", ".join(sorted(set(resolve_backend(name) for name in BACKENDS)))
        raise SystemExit(f"unknown backend {unknown}; expected {known}")
    if split == "search-only":
        allowed = set(SEARCH_ONLY_BACKENDS)
        label = "search-only"
    elif split == "search-fetch":
        allowed = set(SEARCH_FETCH_BACKENDS)
        label = "search-fetch"
    else:
        allowed = None
        label = ""
    if allowed is not None:
        wrong = [item for item in arms if item not in allowed]
        if wrong:
            raise SystemExit(
                f"{wrong} is not on the {label} matrix; "
                f"use {', '.join(sorted(allowed))}"
            )
    return arms


def parse_fetch_modes(raw: str) -> tuple[bool, ...]:
    if raw == "on":
        return (True,)
    if raw == "off":
        return (False,)
    if raw == "both":
        return (True, False)
    raise SystemExit(f"unknown fetch-mode {raw!r}")


def arm_label(backend: str, allow_fetch: bool) -> str:
    return f"{backend}+fetch" if allow_fetch else f"{backend}+search"


def run_arm(
    task,
    *,
    client: Any,
    backend: str,
    allow_fetch: bool,
    model: str,
    max_turns: int,
    max_searches: int,
    max_fetches: int,
    experiment_id: str = "",
    raw_dir: Path | None = None,
    dataset: str = "",
    split: str = "",
    run_stamp: str = "",
) -> dict[str, Any]:
    label = arm_label(backend, allow_fetch)
    with eval_span(task_id=task.id, backend=label, model=model, experiment_id=experiment_id) as arm:
        run = run_agent(
            task.prompt,
            client=client,
            backend_name=backend,
            model=model,
            max_turns=max_turns or task.max_turns,
            max_searches=max_searches or task.max_searches,
            max_fetches=0 if not allow_fetch else (max_fetches or task.max_fetches),
            system=PATCH_SYSTEM if allow_fetch else PATCH_SYSTEM_SEARCH_ONLY,
            allow_search=True,
            allow_fetch=allow_fetch,
            leak_needles=task.leak_needles(),
        )
        source = extract_code(run.answer, task.outfile)
        with judge_span(task_id=task.id, backend=label) as scored:
            score = score_needles(
                source,
                task,
                observed_urls=hop_urls(run),
                retrieved_text=hop_retrieved_text(run),
            )
            log_span(scored, output=score.to_dict(), scores={"passed": float(score.passed)})
        log_span(
            arm,
            output=source,
            scores={"passed": float(score.passed)},
            metadata={"notes": run.notes, "permalink": run.permalink, **run.metrics()},
        )
    result = {
        "id": task.id,
        "backend": label,
        "search_backend": backend,
        "allow_fetch": allow_fetch,
        "used_search": run.used_search,
        "used_fetch": run.used_fetch,
        "search_count": len(run.searches),
        "fetch_count": len(run.fetches),
        "queries": [call.query for call in run.searches],
        "fetch_urls": [call.url for call in run.fetches],
        "search_hops": [call.hop_dict() for call in run.searches],
        "fetch_hops": [call.hop_dict() for call in run.fetches],
        "passed": score.passed,
        "score": score.to_dict(),
        "metrics": run.metrics(),
        "source": source,
        "notes": run.notes,
        "braintrust_permalink": run.permalink,
        "query_contained_gold_token": any(bool(call.gold_token) for call in run.searches),
        "error": None,
    }
    if raw_dir is not None:
        dest = write_arm_raw(
            raw_dir,
            dataset=dataset,
            split=split,
            backend=backend,
            allow_fetch=allow_fetch,
            task_id=task.id,
            run=run,
            row=result,
            run_stamp=run_stamp,
        )
        result["raw_dir"] = str(dest)
    return result


def failed_arm(task, *, backend: str, allow_fetch: bool, exc: BaseException) -> dict[str, Any]:
    label = arm_label(backend, allow_fetch)
    return {
        "id": task.id,
        "backend": label,
        "search_backend": backend,
        "allow_fetch": allow_fetch,
        "used_search": False,
        "used_fetch": False,
        "search_count": 0,
        "fetch_count": 0,
        "queries": [],
        "fetch_urls": [],
        "search_hops": [],
        "fetch_hops": [],
        "passed": False,
        "score": {"passed": False, "compiled": False, "output": "RUN_ERROR", "notes": [str(exc)]},
        "metrics": {},
        "source": "",
        "notes": [f"{type(exc).__name__}: {exc}"],
        "braintrust_permalink": None,
        "query_contained_gold_token": False,
        "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
    }


def selftest() -> int:
    load_environment()
    tasks = load_tasks()
    if not tasks:
        print("FAIL no tasks loaded")
        return 1
    failed = 0
    for task in tasks:
        gold = score_needles(task.gold_source(), task)
        starter = score_needles(task.starter_source(), task)
        if not gold.passed:
            print(f"FAIL {task.id} gold {gold.output}")
            failed += 1
        if starter.passed:
            print(f"FAIL {task.id} starter unexpectedly passed")
            failed += 1
    print(f"judge gold/starter ok={len(tasks) - failed} failed={failed}")
    names = tuple(dict.fromkeys((*SEARCH_ONLY_BACKENDS, *SEARCH_FETCH_BACKENDS)))
    for name in names:
        backend = get_backend(name)
        print(f"runner {name} constructed name={backend.name}")
    live = []
    probes = (
        ("parallel_basic", "PARALLEL_API_KEY"),
        ("parallel_turbo", "PARALLEL_API_KEY"),
        ("firecrawl", "FIRECRAWL_API_KEY"),
        ("exa_auto", "EXA_API_KEY"),
        ("exa_fast", "EXA_API_KEY"),
        ("linkup_fast", "LINKUP_API_KEY"),
        ("tavily_fast", "TAVILY_API_KEY"),
        ("brave", "BRAVE_SEARCH_API_KEY"),
        ("you", "YDC_API_KEY"),
        ("tinyfish", "TINYFISH_API_KEY"),
        ("perplexity_low", "PERPLEXITY_API_KEY"),
        ("perplexity_high", "PERPLEXITY_API_KEY"),
    )
    for name, key in probes:
        if name == "you" and not (
            os.environ.get("YDC_API_KEY")
            or os.environ.get("YOU_API_KEY")
            or os.environ.get("YOU_KEY")
        ):
            print(f"skip live {name} (no YDC_API_KEY)")
            continue
        if name == "tinyfish" and not (
            os.environ.get("TINYFISH_API_KEY") or os.environ.get("TINYFISH_KEY")
        ):
            print(f"skip live {name} (no TINYFISH_API_KEY)")
            continue
        if name in ("perplexity_low", "perplexity_high") and not (
            os.environ.get("PERPLEXITY_API_KEY") or os.environ.get("PERPLEXITY_API")
        ):
            print(f"skip live {name} (no PERPLEXITY_API_KEY)")
            continue
        if not os.environ.get(key) and name not in (
            "you",
            "tinyfish",
            "perplexity_low",
            "perplexity_high",
        ):
            print(f"skip live {name} (no {key})")
            continue
        hits = get_backend(name).search("site:example.com example domain", max_results=1)
        live.append((name, len(hits)))
        print(f"live {name} hits={len(hits)}")
    print(
        f"datasets={datasets_root()} tasks={len(tasks)} live={live} "
        f"llm={llm_transport()} braintrust={'on' if tracing_enabled() else 'off'}"
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="Load datasets + judge gold/starter + construct runners.")
    sub.add_parser("ensure-projects", help="Create or fetch Braintrust projects web-coding-private and web-coding-public.")
    run = sub.add_parser("run", help="Score one or more tickets.")
    run.add_argument("--ids", default="")
    run.add_argument(
        "--backend",
        default="all",
        help=(
            "Vendor id, comma list, or all (default). all requires --split. "
            "search-only: parallel_turbo, parallel_fast, exa_fast, exa_instant, "
            "tavily_fast, brave, linkup_fast, firecrawl, you, tinyfish, "
            "perplexity_low. "
            "search-fetch: parallel_basic, parallel_advanced, exa_auto, exa_deep, "
            "tavily_basic, tavily_advanced, linkup_standard, firecrawl, you, "
            "tinyfish, perplexity_high. "
            "Aliases: exa=exa_auto, tavily=tavily_fast, linkup=linkup_fast."
        ),
    )
    run.add_argument("--fetch-mode", default=None, choices=("on", "off", "both"))
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--max-turns", type=int, default=32)
    run.add_argument("--max-searches", type=int, default=5)
    run.add_argument("--max-fetches", type=int, default=5)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--dataset", default="", choices=("", "private", "public"), help="Braintrust project: web-coding-private or web-coding-public.")
    run.add_argument(
        "--split",
        default="",
        help="search-only (no fetch) or search-fetch. Names the experiment {split}/{vendor} and locks fetch on/off.",
    )
    run.add_argument("--experiment", default="", help="Braintrust experiment name. Default: {split}/{vendor} when --dataset/--split are set.")
    run.add_argument("--out", type=Path, default=Path("/tmp/coding-search-eval.json"))
    run.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("runs/web-coding"),
        help="Write redacted vendor hops under {dataset}/{split}/{vendor}/{run_stamp}/{task_id}/. Use none to skip.",
    )
    run.add_argument(
        "--run-stamp",
        default="",
        help="UTC folder name for this dump (default: now, e.g. 2026-08-24T16-29-16Z). Distinguishes reruns.",
    )
    args = parser.parse_args(argv)
    load_environment()
    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "ensure-projects":
        for key, spec in DATASETS.items():
            project_id = ensure_project(str(spec["project_name"]))
            os.environ[str(spec["env_key"])] = project_id
            print(f"{key} project={spec['project_name']} id={project_id}")
        return 0
    ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    tasks = load_tasks(ids=ids or None)
    dataset = parse_dataset(args.dataset) if args.dataset else ""
    split = parse_split(args.split) if args.split else ""
    backends = parse_arms(args.backend, split=split)
    if args.fetch_mode is None:
        if split == "search-only":
            args.fetch_mode = "off"
        elif split == "search-fetch":
            args.fetch_mode = "on"
        elif backends and all(name in FETCH_FORBIDDEN for name in backends):
            args.fetch_mode = "off"
        elif backends and all(name in FETCH_REQUIRED for name in backends):
            args.fetch_mode = "on"
        else:
            args.fetch_mode = "both"
    fetch_modes = parse_fetch_modes(args.fetch_mode)
    if split == "search-only" and True in fetch_modes:
        raise SystemExit("search-only has no fetch; omit --fetch-mode or use --fetch-mode off")
    if split == "search-fetch" and False in fetch_modes:
        raise SystemExit("search-fetch requires fetch; omit --fetch-mode or use --fetch-mode on")
    fetch_on = [name for name in backends if name in FETCH_FORBIDDEN]
    if fetch_on and True in fetch_modes:
        raise SystemExit(
            f"{', '.join(fetch_on)} is search-only; use --split search-only (no fetch)"
        )
    fetch_off = [name for name in backends if name in FETCH_REQUIRED]
    if fetch_off and False in fetch_modes:
        raise SystemExit(
            f"{', '.join(fetch_off)} is search+fetch; use --split search-fetch"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_stamp = parse_run_stamp(args.run_stamp) or new_run_stamp()
    spec = DATASETS[dataset] if dataset else None
    project_id = None
    if dataset:
        project_id, spec = resolve_project(dataset)
    experiment_id = (
        args.experiment.strip()
        or os.environ.get("CODING_SEARCH_EXPERIMENT_ID")
        or (experiment_name(split=split, backends=backends) if split else "")
        or f"coding-search-{'-'.join(backends)}-{stamp}"
    )
    n_tasks = spec["n"] if spec else len(tasks)
    tags = (
        experiment_tags(dataset=dataset, split=split or "adhoc", backends=backends, n=int(n_tasks))
        if dataset
        else ["coding-search", *backends]
    )
    metadata: dict[str, Any] = {
        "backend": list(backends),
        "fetch_mode": args.fetch_mode,
        "model": args.model,
        "n_tasks": len(tasks),
        "concurrency": args.concurrency,
        "max_searches": args.max_searches,
        "max_fetches": args.max_fetches,
        "vendors": [vendor_slug(item) for item in backends],
        "run_stamp": run_stamp,
    }
    if dataset and spec:
        metadata.update(
            {
                "dataset": dataset,
                "split": split or "adhoc",
                "snapshot": spec["snapshot"],
                "n": spec["n"],
                "project": spec["project_name"],
            }
        )
    logger = init_tracing(
        experiment=experiment_id,
        description=f"coding-search {','.join(backends)} fetch={args.fetch_mode} n={len(tasks)}",
        metadata=metadata,
        tags=tags,
        project_id=project_id,
    )
    client = wrap_client(make_llm_client(timeout=180, max_retries=1))
    jobs = [
        (task, backend, allow_fetch)
        for task in tasks
        for backend in backends
        for allow_fetch in fetch_modes
    ]
    workers = max(1, args.concurrency)
    print(
        f"braintrust={'on' if logger else 'off'} experiment={experiment_id} "
        f"dataset={dataset or '-'} split={split or '-'} project={spec['project_name'] if spec else '-'} "
        f"llm={llm_transport()} model={args.model} n={len(jobs)} concurrency={workers}",
        flush=True,
    )
    if logger is not None:
        try:
            print(f"experiment_id={logger.id}", flush=True)
        except Exception:  # noqa: BLE001
            pass

    raw_dir = None if str(args.raw_dir) in {"", "-", "none"} else args.raw_dir
    if raw_dir is not None:
        print(f"raw_dir={raw_dir.resolve()} run_stamp={run_stamp}", flush=True)

    def _one(job: tuple) -> dict[str, Any]:
        task, backend, allow_fetch = job
        try:
            return run_arm(
                task,
                client=client,
                backend=backend,
                allow_fetch=allow_fetch,
                model=args.model,
                max_turns=args.max_turns,
                max_searches=args.max_searches,
                max_fetches=args.max_fetches,
                experiment_id=experiment_id,
                raw_dir=raw_dir,
                dataset=dataset,
                split=split,
                run_stamp=run_stamp,
            )
        except Exception as exc:  # noqa: BLE001
            return failed_arm(task, backend=backend, allow_fetch=allow_fetch, exc=exc)

    results: list[dict[str, Any]] = []
    try:
        if workers == 1:
            for job in jobs:
                row = _one(job)
                results.append(row)
                print(
                    row["id"],
                    row["backend"],
                    "pass" if row["passed"] else "fail",
                    row.get("braintrust_permalink") or row.get("error") or "",
                    flush=True,
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_one, job): job for job in jobs}
                for fut in as_completed(futs):
                    row = fut.result()
                    results.append(row)
                    print(
                        row["id"],
                        row["backend"],
                        "pass" if row["passed"] else "fail",
                        row.get("braintrust_permalink") or row.get("error") or "",
                        flush=True,
                    )
    finally:
        flush_tracing(logger)
    results.sort(key=lambda row: (row["id"], row["backend"]))
    passed = sum(1 for row in results if row["passed"])
    summary = {
        "n": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "experiment": experiment_id,
        "experiment_url": experiment_url(logger),
        "backend": list(backends),
        "fetch_mode": args.fetch_mode,
        "model": args.model,
        "concurrency": workers,
        "raw_dir": str(raw_dir) if raw_dir else None,
        "run_stamp": run_stamp if raw_dir else None,
        "results": results,
    }
    if logger is not None and hasattr(logger, "summarize"):
        try:
            printed = logger.summarize()
            summary["experiment_url"] = printed.experiment_url or summary["experiment_url"]
            print(printed, flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"braintrust summarize failed: {exc}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"passed={passed}/{len(results)} wrote {args.out}", flush=True)
    if summary.get("experiment_url"):
        print(summary["experiment_url"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
