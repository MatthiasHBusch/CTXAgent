# CTXAgent: developer guide

Installing the app: [README.md](README.md).

A local agent with a GUI — multiple chats, selectable tools, per-chat working
directories — built around a **context compaction engine** that you can watch,
tune and measure while it runs.

* [RESEARCH.md](RESEARCH.md) — how the shipping agents and the literature do compaction, and what this design takes from each.
* [MODELS.md](MODELS.md) — which model should do the compaction work, measured rather than assumed.
* [FREEZE_REVIEW.md](FREEZE_REVIEW.md) — the review before freezing (removed on 2026-11-10 with the frozen prefix), bounded result delivery, and the history of the linked review cycle.
* [DOCUMENT_WORKSPACE.md](DOCUMENT_WORKSPACE.md) — explicit current document views, automatic refresh, late-bound format tool catalogs (PDF/DOCX/XLSX/OBJ/STL), reviewed edit plans, and safe read batching.
* [AGENTS_AND_BROWSING.md](AGENTS_AND_BROWSING.md) — parallel persistent child agents, inherited context controls, browsing, cost aggregation and Flex fallback safeguards.
* [DESKTOP_APP.md](DESKTOP_APP.md) — the CTXAgent desktop app: native window, installer, data and log locations, importing old chats.
* [SETTINGS.md](SETTINGS.md) — the Settings menu: API keys in the OS credential store, chat profiles, the summariser defaulting to the chat model.
* [INSIGHTS.md](INSIGHTS.md) — opt-in usage insights (context window over time, compaction moments, tool and token statistics), the report, and an analysis of how Opus 5.5 uses the tools.

---

## Live browser research

Existing `web_search`, `web_open` and `web_find` remain available. For deep
research use the **Context > Browser research** workspace or the new
`browser_*` tools: open bounded live page selections, render screenshots,
archive manual notes or summary-model notes, then close with a compact handoff.
Full source bodies/pixels stay outside chat history; idle tabs can be closed
without losing their cached snapshots or research notes. Summaries use the
configured summary model/pricing and are billed; unsuccessful summaries leave
tabs open. Browser and document views share a bounded live-context budget.
See [AGENTS_AND_BROWSING.md](AGENTS_AND_BROWSING.md#live-browser-research-workspace)
for tools, limits, provenance and browser safety caveats.

## Quick start

**Desktop app (CTXAgent).** `.\run.cmd --desktop` (or `python -m ctxagent.desktop`)
opens the same UI in a native window. To build the Windows installer run
`powershell -ExecutionPolicy Bypass -File packaging\build.ps1`. Where the
installed app keeps its data and logs, how to import the chats of a checkout
(`--import-data`), and the known limits are in [DESKTOP_APP.md](DESKTOP_APP.md).

From a checkout:

```powershell
pip install -r requirements.txt
```

Put your OpenRouter key in a `.env` file next to this README:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

To call Claude directly through the Anthropic API instead, add
`ANTHROPIC_API_KEY=sk-ant-...` and pick a bare model id such as
`claude-opus-5-5` or `claude-sonnet-5-5` (ids with a provider prefix, like
`anthropic/claude-sonnet-5`, keep going through OpenRouter). A user-scoped key
(`sk-ant-usr-...`) also needs `ANTHROPIC_WORKSPACE_ID=wrkspc_...`. The direct
route (`ctxagent/anthropic_api.py`) keeps the same cache markers, adds a moving
cache breakpoint on the conversation tail, never sends `temperature` (current
models reject it), asks the API to drop thinking blocks that compaction has
invalidated instead of failing with a 400, and enables server-side refusal
fallbacks on Opus 5.5 / Opus 5 / Sonnet 5.5 / Fable 5.1. See `.env.example`.

In **Settings > API key / provider** each chat picks the key it uses. The model
suggestions follow the choice, and switching maps the chat and compaction models
to their counterparts (`anthropic/claude-opus-5.5` <-> `claude-opus-5-5`), or to
a default (`claude-sonnet-5-5` for compaction) when there is none. Anthropic has
no Flex tier: on the direct route only Standard pricing is offered. Its Batch
API (-50%) is asynchronous and does not fit an interactive agent loop.

Then:

```powershell
.\run.ps1
```

`run.ps1` and `run.cmd` use `.venv\Scripts\python.exe` when it exists, else
`python` from PATH. Where PowerShell only runs signed scripts (execution policy
`AllSigned`), start with `run.cmd` instead; it does the same thing.

It opens `http://127.0.0.1:8765`. Add a working directory in the **Tools** tab
before asking it to touch files — the file and shell tools are confined to the
directories you list, and refuse anything outside them.

Tests and measurement:

```bash
python -m tests.test_compaction   # 42 engine-invariant tests
```

```bash
python -m tests.test_readers      # 116 reader / outline / fold / channel tests
```

```bash
python -m tests.bench             # billed-cost benchmark (~4 min, no network)
```

```bash
python -m tests.model_eval        # which model to compact with (~$0.11)
```

---

## The compaction engine

Two mechanisms: the cache-stable split (free, always) and deep compaction (one LLM
call, rare). Between deep compactions the engine never changes a message, so the
provider's prompt cache keeps hitting on the whole history. In self-steered chats
the model also manages its own context (see below).

### Cache points (free, always)

Rewriting an old message to save 2,000 tokens re-writes the provider's cache for
every token after it; that is the constraint the rest of the design bends around.
Anthropic caches only where a request puts a breakpoint (at most 4 per request), and
an entry lives 5 minutes from the last request that read or wrote it. So every
request marks the static system prompt, two **rungs** in the history and the tail
(`cache_policy.plan_rungs`); marking a rung again keeps its entry warm. A breakpoint
on a message writes an entry there at no extra cost
([Anthropic, prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)),
so the rungs are placed where they help:

* the **near rung** about 50k tokens before the end of the history (in a history
  below 100k in its middle), and
* the **far rung** halfway between the start and the near rung's place. Half of the
  whole history would coincide with the near rung at 100k and stay within 50k of it
  up to 200k.

A rung stays on its message while it is within a tolerance of that place (the near
one up to 75k before the end, the far one within a quarter of its place), so
that one entry serves many requests. A change (a curate replaced an item, a live view
changed) is re-written from the newest rung before it. That request marks the last
unchanged message, which is the near rung from then on: the next change behind it
re-writes only what follows. A request that appends more messages than a breakpoint's
20-block look-back reaches has the previous request's tail as its near rung.

So replacing items behind the near rung re-writes 50 to 75k tokens, items between the
rungs what follows the far rung (about half of the history plus 25k), and an item before
the far rung everything after the system prompt. Until 2026-11-11 both rungs followed the end of the history in 20k
steps, and a curate that reached further back than about 40k had nothing to read from.

Live views (the operation catalog, each open document, each browser tab) are messages
of their own. They go into the history behind the previous request's end when they
appear or change, and stay there while they are unchanged; the message before them is
one of the two rungs as long as they are open. There a view is read from the cache,
but a new version re-writes the views and the history that came after them. A view
that changes again within two requests of its last change (a document being edited,
with or without a test run in between) is therefore sent at the **end** of the
request, behind the tail mark, until it has been unchanged for two requests
(`cache_policy.restless_views`): uncached, but a new version costs only itself. The
canvas image is always sent there.
Until 2026-11-10 everything older than the newest 50k tokens was frozen instead and
could not be curated.

### Deep compaction (one LLM call, rare)

Triggers at the configured share of the context window (0.9 of a 700k window: at 630k
provider tokens), at the latest when the prompt would leave less than the requested output
plus 13k of headroom, subject to cooldown and minimum gain. Sizes are local token estimates
scaled by the provider's measured count (token calibration), so the trigger and the Context
tab count the tokens the provider bills.
The **target fraction remaining** (default 0.10) applies to the *whole resulting
prompt*: system instructions, tool schema, protected content, summary and recent
verbatim work. It is not a fraction removed and not merely a tail-size setting.
For example, 90k targets at most 9k, when protected content permits.

The planner reserves a strict summary budget before choosing recent whole tool
groups. The summariser receives that text budget; oversized responses are bounded
locally without a second paid request. Originals are archived before replacement.
Reports expose target tokens, achieved fraction, summary clipping and any
unavoidable floor due to system/tools, explicit pins, recent user messages or the
newest indivisible tool group. A failed summary leaves the context intact.

Tool calls, their results and attached images stay together. Explicit user pins
survive deep compaction; a recalled result does not stay protected, and recall
continues to work from the archive. Proven legacy recall pins are migrated using
recorded recall calls; unknown legacy pins are kept conservatively. Recent-user
protection is transient, not a permanent pin added to every user message.

### Self-steered context management

With **Engine › Self-steered context management** on, the model gets
`compact_context` (replace finished work by its own handoff summary) and
`curate_context` (replace items it no longer needs, counted back from the newest, by
its own short text), and a short "Managing your own context" section in the system
prompt. When a review is due (every N tool batches, or under pressure) the engine
appends `[harness: context review due; cache breakpoints after item 6 = grep_search
result (a.py:1 foo) and after item 10 = ...]` to the newest tool result: the two rungs
of the next request, as curate references with a few words of the item, so the model
sees which items are re-written together. `compact_context(summary, until)` and
`curate_context` are no entries of the tool selection: they come and go with the
switch. Deep compaction stays automatic as the safety net.

### Removed: the incremental tiers

Until 2026-11-05 the engine also rewrote results between deep compactions:
supersession and dedup of re-read files, age-tiered masking and disk offload, and
significance-ranked condensation by a cheap model. Self-steered models now curate
and compact their context themselves, and deep compaction covers the rest, so these
tiers were removed together with the guidance experiment switches. Older chats keep segments that were masked, superseded
or condensed; recall still restores their originals, and settings of the removed
tiers are ignored when a chat loads.

### Nothing is lost, only moved

Every removal writes the original to `.ctxagent/blobs/` first. The model gets a
`recall(segment_id=...)` tool; calling it restores exact text. Deep compaction can
archive it again. Use the Context panel's explicit **Pin** for permanent verbatim
protection.

---

## The GUI

| | |
|---|---|
| **Sidebar** | chats, created and deleted; each keeps its own model, tools, directories and engine config |
| **Context tab** | live window occupancy in provider tokens (in context / free until the deep-compaction limit / above it), what it consists of (history, open documents and pages, tool definitions), tokens saved, cache hit rate, and every segment with its state and before→after size. Click one to see what is in context now versus the stored original, and restore or pin it. |
| **Tools tab** | per-chat tool toggles and the working-directory picker |
| **Settings tab** | model (autocompleted from OpenRouter's live list), context window, max output, temperature, extra system prompt |
| **Engine tab** | every compaction knob, plus a running log of what each pass did |
| **⇲ button** | force a deep compaction now, with optional focus instructions ("focus on the solver refactor, drop the plotting experiments") |

Compaction events appear inline in the transcript as they happen:

```
Deep compaction — 312.4k → 31.7k tokens (−280.7k)
  deep: 214 msgs (290.1k) → 2.9k summary, kept 18 verbatim (tail 9%); target 31.2k, actual 10%
```

---

## Tools

**Files and shell** — `read_file` · `write_file` · `edit_file` · `list_dir` ·
`glob_search` · `grep_search` · `run_command` · `fetch_url`

**Documents** — `read_document` · `summarize_file` · `focus_file` · `summarize_folder`

**Context** — `recall`

**Collaboration** — `scratchpad_read` · `scratchpad_write`: a text field per chat
under the chat list that you and the agent both write in; Ctrl+Enter hands it to
the agent.

**Canvas** — `canvas_open` · `canvas_close` (also `view_image(path, canvas=true)`),
and while a canvas is open `canvas_draw` · `canvas_script` · `canvas_focus` ·
`canvas_undo` · `canvas_save`: an image the agent shows you in the Canvas tab and
keeps current at the end of its requests.

Messages you send while the agent works are queued and delivered with its next
step. Details: [CANVAS_AND_SCRATCHPAD.md](CANVAS_AND_SCRATCHPAD.md).

### Provider region

Every request can be pinned to providers **headquartered and hosted** in the
EU/EEA/UK/CH/US/CA, with `allow_fallbacks: false` so it is a guarantee rather
than a preference. This is on by default (**Settings → Provider region**).

A model's origin and its serving location are different things, and you cannot
tell them apart from the model id. `z-ai/glm-5.3-flash` is a Chinese model served
from ~20 endpoints, most of them US companies on US hardware.
`qwen/qwen3.5-flash-02-23` looks similar and is served from exactly one place:
Alibaba. So the app asks OpenRouter's provider directory (`headquarters` and
`datacenters` per provider) instead of guessing — 59 of 106 providers qualify.

The Settings tab checks the model you type and tells you before you use it:

```
✓ EU/US available — 20 EU/US endpoint(s) available (cheapest: DeepInfra, US,
  $0.075/M). 7 endpoint(s) excluded by region.

✗ not available in-region — NO EU/US endpoint serves this model; it is available
  only via Alibaba, Makora.
```

A request that cannot be served in-region **fails loudly** (HTTP 404, "no allowed
providers") rather than quietly routing somewhere else.

File and shell access is confined to the chat's working directories; paths
outside them are refused. `run_command` uses PowerShell on Windows, bash elsewhere.

---

## Reading documents without reading them

Three tools, one idea: **look at a file at the level of detail the question
needs, not at full size.** A 4,000-line module costs ~40k tokens to read and
~500 to outline; most of the time the outline answers the question.

### `read_document(path)`

Full text out of any container: **PDF** (PyMuPDF, falls back to pypdf) ·
**DOCX** · **PPTX** · **Jupyter notebooks** · **CSV / TSV / Excel / Parquet** ·
**HTML** · **JSON / JSONL / YAML / TOML / INI / XML** · plain text and source.
Missing optional dependencies produce a message naming the package, never a
traceback.

### `summarize_file(path, level, mode)`

**`mode="static"`** is free, instant, and cannot hallucinate — it reports
structure that is literally in the file. Here **`level` is a structural depth**,
not a size limit: 0 is the barest skeleton, each step adds a layer, 5 is the
full text. An outline costs nothing to compute, so there is nothing to ration —
pick the level by the detail you need.

| level | code | markdown / latex | tabular | pdf | json/yaml |
|---|---|---|---|---|---|
| **0** | docstring, imports, **class names** | title, top-level headings | shape + column names | pages, metadata, top bookmarks | top-level keys |
| **1** | + function/method names | + one more depth | + dtypes, row labels | + full bookmark tree | + depth 2 |
| **2** | + full signatures, bases | + one more depth | + per-column stats | + first lines of each page | + depth 3 |
| **3** | + docstring first lines | + one more depth | + first rows | + first paragraph per page | + depth 4 |
| **4** | + full docstrings | + deepest headings | + more rows | — | + depth 5 |
| **5** | the document's full text | | | | |

Python goes through `ast`, so it is exact. Other languages use a
declaration-form regex pass covering C-like syntax, Ruby, MATLAB, Julia, R,
shell and SQL.

#### LaTeX

`.tex` gets its own reader, because the structure is explicit markup rather than
something to infer — so the outline can be exact *and* tiny. The sample paper in
the test suite outlines to **236 tokens at level 0** and 859 at level 4, with
every section, caption and equation label intact:

| level | adds |
|---|---|
| **0** | `\documentclass`, title, author, top-level sections, counts of figures/tables/equations/citations |
| **1** | every sectioning depth with line numbers, `\usepackage` list, the abstract's opening sentence |
| **2** | **figures and tables** with captions, `\label`s and `\includegraphics` paths; labelled display equations |
| **3** | **the opening sentence of every paragraph**, attributed to its section |
| **4** | a second sentence per paragraph, the full abstract, `\input/\include` children, custom macros |

Braces are matched properly rather than with a greedy regex, so a caption
containing `$\alpha$` and `\textbf{...}` survives intact. **siunitx is expanded**
before commands are stripped — otherwise `\SI{500}{\hour}` becomes "500 ."
and loses the part that mattered. Floats, display maths, code listings and the
bibliography are excluded from paragraph extraction, so the reference list is not
reported as the closing paragraph of your conclusions.

**`mode="llm"`** asks a small model what the document *means*, which structure
cannot tell you. Here **`level` is a token budget** — 100 / 300 / 1,000 / 3,000 /
10,000 / uncapped — because unlike an outline, a generated summary has a length
you choose and a price you pay. The budget sets both the instruction ("at most N
tokens") and the API's `max_tokens`, so an over-eager model is cut off by the
provider rather than by your context window. The prompt is the static
outline **plus** a bounded slice of the body, so a 400-page PDF still summarises
in one cheap call — a 200,000-line file compresses to a 59k-token prompt.

### `focus_file(path, select, level)`

The one you will reach for most. `summarize_file` answers *"what is in here?"*
and `read_document` answers *"show me everything"* — neither answers the question
you usually have: **"show me this one part, and enough of the rest that I know
where it sits."**

It builds a node tree for the document, prints every node as one collapsed line,
and expands what you select to its full content — a folded JSON viewer with one
key open, a file browser with one folder expanded:

```
· class CompactionReport  [L42-43]
· def dynamic_tail_ratio(pressure: float, cfg)  [L50-63]
· class CompactionEngine  [L107-369]  (5 inside)
  · def __init__(self, session_id, cfg, complete_fn)  [L108-113]
  ▼ async def run(self, conv, effective_window, *, tool_overhead=0, ...)  [L116-216]
    116 |     async def run(
    117 |         self,
    ... the whole method, verbatim ...
  ▲ end CompactionEngine.run
  · def _deep(self, conv, effective_window, pressure, ...)  [L230-300]
```

Measured on the test suite's 60-class module: **21,784 tokens → 1,337** for one
method plus the full skeleton. And unlike a bare extract, what comes back shows
the agent what *else* is there, so the next request can name it.

**Selectors** — pass one or several:

| form | example | works on |
|---|---|---|
| name | `run` | anything with a tree |
| dotted path | `CompactionEngine.run`, `server.tls` | code, JSON/YAML/TOML |
| regex | `/^_snap/` | anything with a tree |
| line range | `L120-160` | **any file at all**, tree or not |

Ranking is exact path (5) > dotted suffix (4) > regex (3) > substring (2), so a
precise selector always wins over a loose one. Trees are built per format: Python
via `ast`, other code by declaration regex, markdown/LaTeX by heading depth,
JSON/YAML by nesting, notebooks by cell, PDFs by bookmark, spreadsheets by column.

Two failure modes are handled deliberately rather than left to chance: a selector
that matches nothing returns **the list of addressable names** so the next
attempt succeeds, and a selection larger than `max_tokens` is **refused with its
size** rather than truncated into something that looks complete but is not.

### `summarize_folder(path, level)`

Same dial, applied to a tree:

```
level 0   just the listing — names and sizes, nothing opened
level N   each FILE gets a static outline at level N-1
          each SUBFOLDER is surveyed at level N-2
```

The extra decrement on descent is what keeps this bounded: detail decays twice as
fast going down as going in, so a level-3 survey is rich at the top, names three
levels down, and stops. A token budget backs that up as a backstop on the survey
as a whole; individual static outlines are not budgeted. One
call replaces a dozen `list_dir`/`read_file` round trips when orienting in an
unfamiliar project.

### How this interacts with compaction

A full read, a static outline and an LLM summary of the same file are three
different **channels** of observation; each result records its channel, detail
and the source version (SHA-256). Since the supersession pass was removed
(2026-11-05) no result evicts another; the model drops what it no longer needs
with `curate_context`.

---

## Lending the readers to other agents (MCP)

The readers are useful outside ctxagent. `ctxagent/mcp_server.py` exposes three
of them over MCP, so Claude Code — or any MCP client — gets the same tools:

| tool | what it answers |
|---|---|
| `folder_overview(path, level)` | what is in this tree, with content, at a chosen depth |
| `document_outline(path, level)` | the structure of one file, without its content |
| `focus_file(path, select, level)` | one named part in full, inside the collapsed rest |

Register it for every project:

```bash
claude mcp add --scope user ctxread -- python C:\path\to\CTXAgent\app\ctxagent\mcp_server.py
```

A project-scoped `.mcp.json` is also committed here, which covers this repo
alone. Set `CTXREAD_ROOTS` (os.pathsep-separated) to confine the server to
particular directories; unset, it can read whatever the process can, which is
what the client's own file tools can already reach.

**Why it is worth wiring up.** Orienting in this package, measured:

```
ctxagent/ — 46 .py files
  glob, names only      :    359 tok   (then you still have to guess)
  folder_overview lvl 0 :    362 tok
  folder_overview lvl 1 :  4,946 tok
  folder_overview lvl 2 :  7,055 tok   <- signatures; the useful default
  read every file whole : 90,343 tok
```

A 13x saving, but the size is not really the point. A general coding agent has
only *read the whole file* and *grep for a pattern*, and the gap between them is
where the waste lives: it globs for names, guesses which three files matter,
reads them whole, guesses wrong, repeats. These tools answer the question that
actually precedes a read — *which part do I want?* — and `focus_file` answers it
without discarding the context that tells you what else is there.

This matters more for a client than it does for ctxagent itself. ctxagent can
compact a large read afterwards; a client whose context management is a single
summarize-everything pass cannot. Not reading the tokens is the only lever it
has.

## Layout

```
ctxagent/
  config.py          SessionConfig + every compaction knob
  segments.py        the Segment model; rendering to chat messages
  tokens.py          counting and head/tail truncation
  blobs.py           disk store for archived and dropped content
  llm.py             OpenRouter client, cache-breakpoint injection, usage passthrough
  cache_policy.py    which cache points a request sends (rungs), prefix diagnostics
  tools.py           tool registry, sandbox, resource keys
  agent.py           the agent loop
  server.py          FastAPI + SSE
  compaction/
    engine.py        review hints, deep compaction and budget enforcement
    deep_plan.py     whole-prompt budget and protected tool-group planner
    curate.py        curate_context (the agent replaces items by its own text)
    manual.py        compact_context (agent-written handoff)
    economics.py     cache break-even figures for compact_context
    summarize.py     the 9-section deep summary
  readers/
    __init__.py      kind detection, extract_text, outline dispatch
    fold.py          node trees + folded views (focus_file)
    latex.py         .tex: sections, floats, equations, paragraph openings
    code.py          Python via ast; regex declarations for everything else
    markup.py        markdown, rst, html, plain text
    tabular.py       csv, tsv, excel, parquet
    docs.py          pdf, docx, pptx, notebooks
    structured.py    json, jsonl, yaml, toml, ini, xml
    folder.py        recursive folder overviews
    llm.py           model-generated summaries, level-scaled
web/                 the GUI (no build step)
tests/               engine tests, reader tests, cost benchmark, model eval
```

## Notes

* A new chat starts with **Claude Opus 5.5** (`anthropic/claude-opus-5.5`, or
  `claude-opus-5-5` when only an Anthropic key is set up), reasoning effort
  **xhigh**, context window **700,000**, **unlimited output** (max output 0: the
  model's maximum, 128k for current Claude models, as far as the window has room),
  **self-steered** context management with deep compaction at **90 %** of the
  window, and every tool on. The summariser runs at reasoning effort **high**.
  These are the defaults without a saved template or default profile; with the
  default "New chat copies the open chat", a new chat takes the open chat's settings.
  Deep summaries have a **10,000-token maximum**, a **1,024-token budget floor**
  (capped by a smaller configured maximum), and a dynamic budget within the total
  post-compaction target. These are budgets, not guaranteed summary lengths.
* **Reasoning effort** is set per model in **Settings** (`xhigh` / `high` /
  `medium` / `low` / `off` / provider default; `xhigh` for the chat model only).
  It is not a simple passthrough: reasoning tokens come out of the same
  `max_tokens` budget as the answer, so a high-effort call can consume the budget
  and return an empty string. The client honours your setting first, then steps
  down rather than failing: a chat response that spent the budget on reasoning
  is retried one level lower (xhigh to high), then at low; a summary steps down
  to low and then the provider default. Some endpoints refuse to have reasoning
  disabled at all, which is handled separately. A model that
  does not offer `xhigh` runs it as `high`: OpenRouter maps it to the nearest
  level, and the direct Anthropic path sends `high` to Opus 4.6 and Sonnet 4.6.
  Changing the effort in a running chat invalidates the prompt cache once.
* **Token pricing** is shown live under each model field: input and output rate
  per Mtok, the model's context length, and what a *full* context costs per
  request. That last number is the one that matters — at 700k context a
  $10/Mtok model is **$7 per request** before cache hits, which is precisely
  what the cache points exist to avoid. Both are per chat — an existing chat keeps
  the settings it was created with, so change them in **Settings** if you want
  the new defaults applied to an old chat.
* The agent model and the compaction/summary model are both set in **Settings**;
  the summariser is used for context compaction *and* for
  `summarize_file(mode="llm")`. See [MODELS.md](MODELS.md) for how the default
  was chosen.
* The default context window is 700k even for models that offer 1M: deep
  compaction at 90 % of it keeps a request below 630k. Raise it in **Settings**
  if you'd rather not compact.
* Explicit cache breakpoints are only emitted for `anthropic/*` models, which is
  where OpenRouter requires them; other providers cache the unchanged prefix
  automatically.
* Optional document dependencies: `pymupdf` (PDF), `python-docx`, `python-pptx`,
  `openpyxl`/`pandas` (spreadsheets). Each is probed at call time — without one
  you get a message naming the package, and everything else keeps working.
* A turn persists even if the client disconnects mid-stream, and checkpoints
  after every tool batch: tool calls have already touched the filesystem by then.
* State lives in `.ctxagent/` (SQLite + blobs) next to the app.


## Productivity tools: restart-stage delivery

Stage 1 adds native `view_image`/`render_pdf`, durable managed jobs
(`start_job`, `job_status`, `list_jobs`, `read_job_log`, `cancel_job`), conservative
coverage-aware supersession, archived recall and interruption-safe tool pairing.
The fixed 40-iteration stop is replaced by `max_tool_iterations` (0 = unlimited).
Existing chats require explicitly enabling new tools after restarting the server.

Run the complete offline regression suite in an isolated data directory:

```powershell
python -B -m tests.run_all
node --check web/app.js
```

Working-directory checks are file access guardrails, **not a shell sandbox**.
Managed jobs survive chat Stop/disconnect/server restart; cancel them explicitly.
Settings and Engine controls save on their own when a value is committed (a select, checkbox
or slider at once, a text or number field on Enter or when it loses focus). Saved while the
agent works, a change applies from that turn's next model request; the pane says so.


### Flex pricing

Settings now offers independent **Pricing / service tier** selectors for the main
and compaction/summary models: standard, try Flex with standard-price fallback,
or Flex only. Refresh availability uses metadata only; a selection is saved at once.
Advertised discounts are shown rather than assuming every endpoint is 50% cheaper.
Reported service tier and actual cost govern billing; partial/network failures are
not replayed for fallback. See [FLEX_PRICING.md](FLEX_PRICING.md) for safety boundaries,
verification results and restart instructions (`backend_build: flex-v1`).

### Cache accounting and workflow tools

`CACHE_AUDIT.md` documents cache fixes and measured live smoke results. The UI now
separates lifetime/recent cache rates, authoritative reported cost, and explicitly
labelled input/cache-read/cache-write/output/reasoning tariff estimates. Historical
cost is unallocated where old request details were not retained.

New opt-in tools: `latex_inspect`, `latex_build`, `latex_diagnostics`,
`prepare_edit_plan`, `apply_edit_plan`. Restart and enable them in the chat Tools
panel. LaTeX uses managed jobs, then diagnostics and native PDF review. Edit plans
show a diff before applying hash-checked per-file atomic changes with backups.
Neither TeX execution nor shell commands are sandboxed; multi-file editing is not
a globally atomic transaction.

Optional UI test: install `requirements-dev.txt`, then run
`python -B -m tests.ui_smoke` (uses installed Microsoft Edge; synthetic data only).


### Rewrite cadence and research tools

Review hints default to every 20 completed tool batches, earlier only under
pressure (70 % of the window), never more often than the interval; checkpoints
still happen every batch. Configure in Engine. Until 2026-11-10 a
hint also came after 20k new tokens; in a long game-project session (INSIGHTS.md) the
curates that followed hints within 12 batches freed little for the same rewrite
and lost money, so the token trigger and its setting were removed and the
interval went from 10 to 20. See COMPACTION_CADENCE.md for the offline trade-off
analysis, rather than treating cache-hit percentage alone as the objective.

Optional research/science tools: scholarly_search, doi_lookup, record_evidence,
environment_report, experiment_manifest and finalize_experiment. Source quotation
presence is not evidence of claim entailment; manifests record explicit inputs
and output hashes, not proof of complete reproducibility.

Verify a real backend restart using `python -B -m ctxagent.check_backend`.
The Windows launcher refuses occupied ports; browser refresh alone does not reload
Python code. The UI warns on old backend identity or source changes since startup.


### Chat settings and stream isolation

**New chat** inherits the **complete saved settings of the currently open chat**:
Settings, Engine, Tools, working directories and delegation limits. It starts with
empty history, usage and live-document workspace; later changes are independent.
Unsaved form edits are not captured. If no chat is open, creation uses the saved
new-chat template. **Settings -> Use saved settings, Engine & Tools as new-chat
defaults** replaces that fallback template. Existing chats keep their own
configuration; an explicitly empty tool list means no tools.

**Engine › Self-steered context management** is one switch for the model's own
context tools: on, the chat gets `compact_context`, `curate_context` and the
"Managing your own context" prompt section, and the engine only hints; off, the
engine manages the context with deep compaction alone. The two tools are not in
the Tools list (SETTINGS.md).

Running chats own their stream state and DOM independently. Switching chats
never redirects results, figures, notices, usage or Stop to another chat.
Unsent drafts are isolated per chat for the lifetime of the browser tab; they
are not persisted across browser reloads. Late chat/context/settings responses
are ignored when their originating view is no longer current.

Offline regression suite: `python -B -m tests.run_all`.
Optional browser isolation regression (Playwright + Edge, fully mocked APIs):
`python -B -m tests.chat_ui_smoke`.
