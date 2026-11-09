# Usage insights

An opt-in record of how a chat uses its context window and its tools, and a report built
from it. It is meant for practical experiments: work on a real project as usual, then read
how the context grew and shrank, what the model was doing when it compacted or curated,
which tools it used and which ones failed.

Code: `ctxagent/insights.py` (recorder), `ctxagent/insights_report.py` (report),
`ctxagent/insights_import.py` (events rebuilt from older chats). Tests: `tests/test_insights.py`.

## Turning it on

**Settings > Insights > Record usage insights.** It is off by default. The switch is an app
preference (`insights` in `settings.json`) and applies to every chat from the next request
on, delegated child agents included. Turning it off stops recording; what was recorded stays.

Files: `<data folder>/insights/<chat id>.jsonl`, one JSON object per line. The data folder is
`%LOCALAPPDATA%\CTXAgent` for the installed app and `<repo>/.ctxagent` from a checkout
(Settings > About shows it; Settings > Insights has an Open button).

**What is kept.** Tool arguments as paths, flags and numbers; file contents, edit strings and
summaries only as sizes (characters and lines). Shell commands as their first 240 characters.
Short excerpts (600 characters) of your messages, the answers, the text that came with a
compact or curate call, and compaction summaries, so a moment can be read later. The end of
the output of a failing command (300 characters). Nothing leaves the computer.

## What is recorded

| event | when | main fields |
|---|---|---|
| `turn_start` | a message is sent | `turn`, `model`, `effort`, `window`, `self_steered`, `context` (estimate), `user` (excerpt) |
| `request` | every model request | `prompt_tokens`, `uncached`, `cache_read`, `cache_write`, `output`, `reasoning`, `cost`, `context`, `finish`, `tool_calls`, `ms`, `cache_break` (expected cache-break reasons) |
| `tool` | every tool result | `name`, `category`, `args` (summary), `ok`, `error_class`, `error`, `ms`, `tokens`, `chars`, `mutates`, `segment`, `repeated`; run_command adds `exit`, `program`, `tail`; searches `empty`; write_file `overwrote` |
| `context` | compact_context or curate_context (applied, dry run or held), an engine compaction, a review hint | `source`, `applied`, `dry_run`, `held`, `before`, `after`, `freed`, `rewrite_tokens`, `breakeven_requests`, `since` (tool batches, answers and requests since the last agent compact/curate), `task`, `call_text`, `last_answer`, `summary`, `dropped`, `tiers`, `hint` |
| `turn_end` | the turn ends | `outcome` (final, error, cancelled, stopped), `answer` (excerpt), `tools`, `tool_errors`, the turn's `requests`, tokens and `cost` |

Tool categories: edit, inspect, search, shell, context, web, agents, research, other
(`insights.CATEGORIES`). Error classes come from the result text (`insights.RESULT_PATTERNS`,
shared with `tools_scripts/scan_tool_failures.py`): `result.edit_not_found`,
`result.path_not_found`, `result.bad_arguments`, `result.command_timeout`, and so on. A
non-zero exit of run_command is `info.command_nonzero_exit`, counted apart from failures,
because a failing test run is a result, not a broken tool call.

## The report

**Settings > Insights > Build and open report** writes `report.html` and `report.md` into
the insights folder and opens the HTML page in the default browser. From a shell:

```
python -m ctxagent.insights_report                                  # this computer's insights
python -m ctxagent.insights_report --chat ID --md r.md --json r.json
python -m ctxagent.insights_report --model opus --html opus.html     # only Opus requests and calls
python -m ctxagent.insights_report --from-db PATH\ctxagent.db --html old.html   # older chats, see below
```

Sections:

- **Tokens.** Composition of all input and output (uncached, cache read, cache write,
  output, reasoning) and a table per model: requests, cost, cache-read share, tool calls and
  failures. Per turn: median and p90 of cost, requests and tool calls.
- **Context window over time.** One chart per chat: tokens sent per request, with markers
  for compact_context (▼), curate_context (●), engine compactions (■) and review hints (○).
  Hover the line for a request's tokens, cache-read share and cost; hover a marker for what
  the chat was doing: the task, the text that came with the call, tokens before and after,
  the rewrite and its break-even, and tool batches since the previous compact or curate.
- **Compaction and curation.** Per source: applied, dry runs, held, tokens freed, the context
  before, the rewrite, and the tool batches since the previous one. How often a review hint
  was followed by a compact or curate within five requests. A table of every agent
  compact/curate call with its moment.
- **Tools.** Per category and per tool: calls, failures and their main class, median
  duration, mean result tokens, how many results were later dropped by curate_context, and
  repeated identical failures.
- **Editing.** Failed edits and whether a later edit of the same file fixed them within eight
  tool calls; failure rate after reading the file, on a file the chat wrote itself, and with
  neither; a missed `old_string` by what happened to the file since the last read; write_file
  creating versus overwriting; edit sizes; the files edited most.
- **Inspection.** read_file: partial reads, result size, repeat reads, identical re-reads
  (same file and range with no edit in between), reads of a file after editing it; searches
  without a match; result size and later drops per tool.
- **Shell commands.** Non-zero exits and timeouts per program (the program is found past
  `cd`, variable assignments and here-strings), why commands failed (the end of the output:
  PowerShell parser or not-a-command errors, also in German; Python exceptions; failing tests;
  GDScript errors; ...), and how many commands only read or searched files.
- **Per model.** The tools that fail most (at least five calls) with their classes and an
  example, and the tools that work (≤ 2 % failures).

**Older chats.** `--from-db` rebuilds the events from a chat database (read-only: the database
is copied first): requests from the usage ledger, tool calls and results from the history
including archived segments, context events from the compaction log, turns from the user's
messages. It cannot rebuild tool durations, dry runs or review hints, and it reads the error
class from the result text.

## A practical experiment

1. Turn recording on and work on the project as usual, for example with the Opus 5.5
   starter (self-steered, 1M).
2. Build the report now and then. Worth reading:
   - the context chart of a long chat: does the model compact at task boundaries, how far
     does the context grow before it does, how much does each curate free;
   - the moments table: what the task was when it compacted or curated;
   - Editing and Shell commands for the code work: misses, recoveries, failing commands;
   - Inspection: how much of the reading goes through read_file versus the shell, and how
     much of it is dropped again.
3. To compare models, filter with `--model`, or run the same kind of task with another
   starter and compare the per-model tables.

## Opus 5.5: which tools work and which are hard (2026-11-05)

Data: the developer's CTXAgent chat databases, read-only: the live chats of a checkout
(among them a long session on a game project), the installed app, and 187 benchmark
sessions (Windows and WSL), rebuilt with `--from-db`:
220 chats, 28,243 tool calls, 27,230 requests. Opus 5.5 (OpenRouter and direct API):
47 live turns with 1,911 tool calls ($135), 2,219 benchmark turns with 3,745 tool calls
($204). Every event is attributed to the model of the request before it, so the turns of
other models in a chat that switched model are not counted.

**No tool is hard for Opus by failure rate.** 0.8 % of its live tool calls and 0.16 % of its
benchmark calls returned an error; only Astra (0.05 %) and Muse (0 %) were lower in the benchmarks.

| | Opus live | Opus bench | Sol bench | Fable bench | Sonnet bench | Haiku bench |
|---|---|---|---|---|---|---|
| tool calls failed | 0.8 % | 0.16 % | 0.34 % | 0.22 % | 0.7 % | 0.8 % |
| edit_file calls / failed | 173 / 0 % | 112 / 0 % | 753 / 0.3 % | 597 / 0 % | 39 / 0 % | 470 / 2.3 % |
| write_file overwrote an existing file | 17 % | 70 % | 66 % | 67 % | 73 % | 61 % |
| run_command share of tool calls | 38 % | 75 % | 33 % | 34 % | 48 % | 39 % |
| run_command non-zero exit | 6.4 % | 1.7 % | 3.8 % | 3.9 % | 4.3 % | 4.7 % |
| reading/searching done in the shell | 30 % | 79 % | 0.6 % | 12 % | 30 % | 10 % |
| compact / curate calls | 6 / 55 | 35 / 10 | 97 / 58 | 32 / 83 | 8 / 0 | 1 / 18 |
| tokens freed per compact / per curate | 106k / 9.5k | 35k / 8.5k | 83k / 0.9k | 51k / 8.2k | 53k / – | 67k / 13.5k |

What works:

- **Editing.** edit_file: 285 calls, none failed (live and benchmark); prepare_edit_plan and
  apply_edit_plan: 26 live calls, none failed. Opus reads a file before editing it in most
  cases and edits files it wrote itself without re-reading; neither failed. In live work it
  creates far more files than it overwrites (39 created, 8 overwritten); in the benchmarks
  70 % of its write_file calls overwrite a file it is iterating on.
- **Reading.** read_file, focus_file, recall and grep_search all below 3 % failures; the
  grep_search misses are paths that do not exist or lie outside the working directories.
- **Context.** compact_context frees about 106k tokens per call in live chats. In the long
  game chat (1,152 requests, window 500k) the engine masked results until the context
  reached 417k; after the first compaction Opus kept it between about 20k and 200k with 47
  curates and 6 compactions.

Where it struggles:

- **The Windows shell.** 6.4 % of its live commands exit non-zero, against 1.7 % in the Linux
  benchmark containers. By program: Get-ChildItem 18 %, git 13 %, Python 13 %, Select-String
  3 %. By the end of the output: PowerShell errors (16, mostly ObjectNotFound for a path that
  does not exist), Python exceptions (5, among them `UnicodeEncodeError` from the cp1252
  console encoding when printing non-ASCII text), failing tests (3, a result rather than a
  broken call), `rg`, which is not installed (2), a parser error in a long one-liner and an
  execution-policy refusal (1 each). PowerShell reports these in German on this computer.
- **It reads through the shell.** 30 % of its live reading and searching (79 % in the
  benchmarks) goes through Select-String, Get-Content or cat instead of read_file and
  grep_search; no other model comes close (Sol 0.6 %, Fable 12 %). Those results are smaller
  (582 tokens on average live, against 1,282 for read_file and the other inspection tools),
  but the engine cannot tie them to a file: a later edit does not mark them stale.
- **Small curates.** Live, a curate freed 9.5k tokens on average, against 106k per
  compaction. With a cache write costing 25 times a cache read on Opus, a curate that frees
  10k tokens behind a long suffix pays back slowly.
- **Paths outside the working directories.** summarize_file and grep_search on the Godot
  user-data folder in AppData were refused (`path_outside`); adding that folder as a working
  directory would let them work.

Limits: imported chats carry no tool durations, dry runs or review hints. Error classes come
from result texts. The live sample is small (47 turns) and mostly one long session, so its
percentages are rough. Benchmark sessions are short scripted tasks, mostly on Linux.
