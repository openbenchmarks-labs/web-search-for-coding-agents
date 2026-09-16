# Web Search API Benchmark for Coding Agents and Developers

Open, independent benchmark of the best web search APIs for coding agents: Exa,
Parallel, Perplexity, Firecrawl, Tavily, Linkup, Brave, You, Nimble, and TinyFish.
Scored on grounded task completion against held-out enterprise documentation
tickets. Open source code + open data. Search-only and search & fetch are ranked
separately. The model, the tickets, and the budgets stay fixed; only the search
vendor changes.

**Live leaderboard:** https://openbenchmarks.com/web-search-for-coding-agents
**Public dataset:** [`openbenchmarks/OB-Code-Websearch`](https://huggingface.co/datasets/openbenchmarks/OB-Code-Websearch)

Published and maintained by **[OpenBenchmarks Labs](https://openbenchmarks.com)**.

This repo is the open harness, vendor runners, and judge behind that page. It is
**open code only**. It does not include the scored private task set, run dumps,
or leaderboard snapshots.

## Which web search API is most accurate for coding agents?

**Exa (deep)** leads search & fetch at 83.0% task completion; **Perplexity (low)**
leads search-only at 77.3%. Task completion is mean ± SD over 3 runs on 100
held-out tickets, model fixed at `gpt-5.6-sol`.

### Search & fetch

| # | Vendor | Configuration | Task completion | Avg search | Median tokens |
| --- | --- | --- | --- | --- | --- |
| 1 | Exa deep | `type=deep` | 83.0% ± 1.0 | 3.97s | 23,660 |
| 2 | Exa auto | `type=auto` | 81.7% ± 1.1 | 1.19s | 27,433 |
| 3 | TinyFish | Default | 79.0% ± 2.0 | 1.32s | 12,844 |
| 4 | Perplexity (high) | `search_context_size=high` | 77.7% ± 1.5 | 991ms | 20,062 |
| 5 | Parallel advanced | `mode=advanced` | 77.0% ± 1.0 | 3.11s | 27,092 |
| 6 | Parallel basic | `mode=basic` | 76.0% ± 0.0 | 1.59s | 32,809 |
| 7 | Firecrawl | Default | 76.0% ± 1.0 | 2.81s | 17,379 |
| 8 | Nimble | `search_depth=lite` | 60.3% ± 2.5 | 1.93s | 16,828 |
| 9 | Tavily advanced | `search_depth=advanced` | 60.0% ± 2.0 | 3.41s | 26,269 |
| 10 | Tavily basic | `search_depth=basic` | 59.0% ± 1.7 | 1.50s | 27,405 |
| 11 | You | `extraction_mode=highlights` | 55.0% ± 1.7 | 638ms | 42,806 |
| 12 | You | `extraction_mode=highlights` · `knowledge=core` | 54.0% ± 2.0 | 678ms | 37,453 |
| 13 | Linkup standard | `depth=standard` | 48.3% ± 4.9 | 2.00s | 57,791 |
| 14 | Nimble | `search_depth=standard` | 45.0% ± 1.0 | 905ms | 28,586 |

### Search only

| # | Vendor | Configuration | Task completion | Avg search | Median tokens |
| --- | --- | --- | --- | --- | --- |
| 1 | Perplexity (low) | `search_context_size=low` | 77.3% ± 2.1 | 957ms | 8,765 |
| 2 | Firecrawl | Default | 70.3% ± 1.5 | 2.87s | 7,456 |
| 3 | Parallel fast | `mode=fast` | 66.7% ± 1.5 | 953ms | 12,460 |
| 4 | Exa fast | `type=fast` | 66.3% ± 1.5 | 626ms | 22,344 |
| 5 | Parallel turbo | `mode=turbo` | 64.7% ± 2.1 | 333ms | 14,130 |
| 6 | Exa instant | `type=instant` | 61.3% ± 2.9 | 447ms | 22,423 |
| 7 | TinyFish | Default | 59.3% ± 1.5 | 2.15s | 7,469 |
| 8 | Tavily fast | `search_depth=fast` | 47.3% ± 2.5 | 282ms | 23,922 |
| 9 | Nimble | `search_depth=lite` | 46.7% ± 3.8 | 3.15s | 8,712 |
| 10 | Linkup fast | `depth=fast` | 43.3% ± 1.1 | 1.39s | 24,057 |
| 11 | Brave (LLM Context) | LLM Context | 43.0% ± 2.0 | 547ms | 21,000 |
| 12 | Nimble | `search_depth=standard` | 42.3% ± 2.5 | 878ms | 14,858 |
| 13 | You | `extraction_mode=highlights` | 41.3% ± 3.1 | 596ms | 20,333 |
| 14 | You | `extraction_mode=highlights` · `knowledge=core` | 38.3% ± 3.2 | 677ms | 20,786 |

Snapshot updated 2026-09-15. The
[live board](https://openbenchmarks.com/web-search-for-coding-agents) is the
source of truth; re-read it before quoting these numbers.

The two boards are ranked separately because search-result quality and
page-retrieval quality are different products. Read the board that matches
whether your agent can open pages.

## Which web search API is fastest for coding agents?

**Tavily fast** at 282ms and **Parallel turbo** at 333ms lead search-only average
search time; **You** with `extraction_mode=highlights` at 638ms leads search & fetch. Parallel turbo is the better
trade at the fast end: it holds 64.7% completion against Tavily fast's 47.3%.

Full ranking: https://openbenchmarks.com/web-search-for-coding-agents/fastest-search-api

## Which web search API is most token-efficient for coding agents?

**Firecrawl** at 7,456 median tokens per ticket leads search-only, with TinyFish
close behind at 7,469 and Perplexity (low) at 8,765, and Perplexity carries the
top completion score at that token budget. On search & fetch, **TinyFish** leads
at 12,844. Linkup standard costs 57,791, roughly 4.5× TinyFish, for the lowest
completion on the board.

This measures median LLM tokens per ticket, not search-API list price.

Full ranking: https://openbenchmarks.com/web-search-for-coding-agents/most-token-efficient-search-api

## What is scored

Each ticket is an enterprise documentation task: the agent must find the right
behavior in a vendor's docs and write code against it. A ticket passes only if

- the file compiles,
- every gold token is present,
- each gold token carries a `# source:` URL, and
- that URL, or the token itself, appeared in that run's search or fetch results.

The last condition is what makes the score a retrieval measurement rather than a
model measurement: the agent has to have actually found the answer, not recalled
it.

The model (`gpt-5.6-sol`), tickets, and budgets stay fixed at 32 turns, 5
searches, and 5 fetches on search & fetch. Only the search/fetch vendor changes.
Parallel maps `site:` hosts into `source_policy.include_domains` and strips those
operators from the query string; every other vendor receives the agent query
unchanged.

Popular, highly regular docs like Stripe are deliberately excluded: the model
could infer the documentation slug from the ticket title and reach the right
answer without retrieving anything. If a capable model can reliably solve a task
without searching, the task does not belong in a search benchmark.

## Datasets

The live boards are scored on a held-out **100-ticket** private set that is not
distributed, so vendors and models cannot fit to the benchmark. A **30-ticket**
public set (format only) is on Hugging Face as
[`openbenchmarks/OB-Code-Websearch`](https://huggingface.co/datasets/openbenchmarks/OB-Code-Websearch).
Use the public rows to inspect the format; scores on those rows are not
comparable to the boards.

## Reproduce

Point `DATASETS_ROOT` at a ticket tree containing
`web-search/coding/tasks/catalog.json` and per-task `prompt.txt`, `starter/`,
and `gold/`.

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

`--backend all` requires `--split`. Aliases: `exa` → `exa_auto`, `tavily` →
`tavily_fast`, `linkup` → `linkup_fast`.

Keys: `OPENAI_API_KEY` (or Azure NEXTGEN), plus the vendor you run
(`PARALLEL_API_KEY`, `FIRECRAWL_API_KEY`, `EXA_API_KEY`, `LINKUP_API_KEY`,
`TAVILY_API_KEY`, `BRAVE_SEARCH_API_KEY`, `YDC_API_KEY` or `YOU_API_KEY`,
`TINYFISH_API_KEY`, `PERPLEXITY_API_KEY`). Optional: `BRAINTRUST_API_KEY`.

## Repository map

| path | purpose |
|---|---|
| `scripts/coding_search/search.py` | Vendor HTTP runners (search + fetch) |
| `scripts/coding_search/judge.py` | Compile + gold tokens + grounded `# source:` URL |
| `scripts/coding_search/agent.py` | LLM loop with `web_search` / `web_fetch` |
| `scripts/coding_search/eval.py` | CLI: `selftest`, `run`, Braintrust tracing |
| `scripts/coding_search/tasks.py` | Load a local ticket catalog |
| `scripts/coding_search/rawlog.py` | Redacted hop dumps |
| `scripts/coding_search/tracing.py` | Optional Braintrust spans |

## Related benchmarks

This is the hard-retrieval task of the OpenBenchmarks web search benchmark. The
same vendors are measured on two other jobs:

- **Factual lookup.** 300 company-news questions, scored on extracted-answer
  accuracy: https://openbenchmarks.com/company-news
- **Multi-hop search.** Multi-constraint company discovery, scored on F1:
  https://openbenchmarks.com/multi-turn-company-search
- **Methodology and all three boards:** https://openbenchmarks.com/web-search
- **Agent-readable index:** https://openbenchmarks.com/llms.txt

## Changelog

- **2026-09-15.** Added Nimble lite and standard to both existing tables with three-repeat means and standard deviations. Search uses `focus=general` and `full_content=false`; search + fetch calls `POST /v2/extract` with `formats=[markdown]`. Added You highlights and highlights + `knowledge=core` to both modes, using a separate You Contents call for fetch. Published 100-task, three-repeat results, including standard deviations; replaced the older plain You rows.

- **2026-09-09.** Re-evaluated TinyFish after updates were rolled out to their
  GA Fetch endpoint. Search & fetch completion moves to 79.0% ± 2.0 (3rd);
  median tokens 12,844; avg search 1.32s.

## License

MIT.
