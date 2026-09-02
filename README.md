# Web Search for Coding Agents

Harness, vendor runners, and needles judge for the
[Web Search for Coding Agents](https://openbenchmarks.com/web-search-for-coding-agents)
benchmark.

Published by **[OpenBenchmarks Labs](https://openbenchmarks.com)**.

This repo is **open code only**. It does not include the scored private
task set, run dumps, or leaderboard snapshots.

**Datasets.** The live boards are scored on a held-out **100-ticket**
private set that is not distributed, so vendors and models cannot train
and fit to the benchmark. A **30-ticket** public set (format only) is on
Hugging Face as
[`openbenchmarks/OB-Code-Websearch`](https://huggingface.co/datasets/openbenchmarks/OB-Code-Websearch).
Use the public rows to inspect the format; scores on those rows are not
comparable to the boards. Task completion is mean ± SD of 3 runs on the
100 private tickets.

## What is here

| path | purpose |
|---|---|
| `scripts/coding_search/search.py` | Vendor HTTP runners (search + fetch) |
| `scripts/coding_search/judge.py` | Compile + gold tokens + grounded `# source:` URL |
| `scripts/coding_search/agent.py` | LLM loop with `web_search` / `web_fetch` |
| `scripts/coding_search/eval.py` | CLI: `selftest`, `run`, Braintrust tracing |
| `scripts/coding_search/tasks.py` | Load a local ticket catalog |
| `scripts/coding_search/rawlog.py` | Redacted hop dumps |
| `scripts/coding_search/tracing.py` | Optional Braintrust spans |

A ticket passes only if the file compiles, every gold token is present,
each has a `# source:` URL, and that URL (or the token itself) appeared
in that run's search or fetch results.

## Boards

**search-only** (no `web_fetch`): Parallel turbo/fast, Exa fast/instant,
Tavily fast, Brave LLM Context, Linkup fast, Firecrawl, You, TinyFish,
Perplexity low.

**search & fetch**: Parallel basic/advanced, Exa auto/deep, Tavily
basic/advanced, Linkup standard, Firecrawl search + scrape, You, TinyFish,
Perplexity high.

The model (`gpt-5.6-sol`), ticket, and budgets stay fixed (32 turns, 5
searches, 5 fetches on search & fetch). Only the search/fetch vendor
changes. Parallel maps `site:` hosts into `source_policy.include_domains`
(the query string is stripped of those operators); other vendors receive
the agent query unchanged.

## Run

Point `DATASETS_ROOT` at a ticket tree with
`web-search/coding/tasks/catalog.json` (and per-task `prompt.txt`,
`starter/`, `gold/`).

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env

DATASETS_ROOT=/path/to/tickets PYTHONPATH=scripts python -m coding_search selftest

DATASETS_ROOT=/path/to/tickets PYTHONPATH=scripts python -m coding_search run \
  --dataset public --split search-only --backend firecrawl \
  --ids hard_sn_price \
  --out /tmp/coding-search-eval.json
```

`--backend all` requires `--split`. Aliases: `exa` → `exa_auto`,
`tavily` → `tavily_fast`, `linkup` → `linkup_fast`.

Keys: `OPENAI_API_KEY` (or Azure NEXTGEN), plus the vendor you run
(`PARALLEL_API_KEY`, `FIRECRAWL_API_KEY`, `EXA_API_KEY`, `LINKUP_API_KEY`,
`TAVILY_API_KEY`, `BRAVE_SEARCH_API_KEY`, `YDC_API_KEY` or `YOU_API_KEY`,
`TINYFISH_API_KEY`, `PERPLEXITY_API_KEY`). Optional:
`BRAINTRUST_API_KEY`.

## License

MIT.
