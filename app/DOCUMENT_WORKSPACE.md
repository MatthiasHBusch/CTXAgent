# Explicit live document workspace and tool batching

> **Note (2026-11-05):** the engine's incremental tiers (supersession and dedup, age-tiered
> masking and offload, significance-ranked condensation) were removed; deep compaction is now
> its only automatic compaction (README, "The compaction engine"). Observation coverage is still recorded with each result, but no pass compares it any more. The code it names
> is in the history up to commit 8af424c.

Backend identity: **document-catalog-v1**. This extends the tested linked
review/compact/freeze system; it does not replace historical evidence handling.

## Why explicit views

Document deduplication asks whether an old observation is covered by a newer one.
That is tricky with partial reads, changed files, lossy representations and frozen
history. A workspace instead answers a simpler question: **which current files
and sections do we intentionally want in the next request?**

- `document_open(path, select?, max_tokens=8000)` opens full source/readable text
  or a folded section view. Names, dotted names and `L20-80` are selectors.
- Opening the same canonical path replaces its view, retaining its document id.
- `document_list()` returns ids, selectors, SHA-256 versions, status and costs.
- `document_close(document_id, summary?)` removes the live payload, optionally
  retaining a close note of at most 1000 tokens. No paid summary call is made.
- The Context panel has matching open, refresh, close and enable-tools controls.
  Optional selectors are entered **one per line**, so commas within a name do not
  need an ambiguous splitting convention.

## Context and persistence contract

The workspace is a single **user-role data frame** after the frozen historical
prefix and before the mutable working tail (always after the system prompt). Its header says file contents are data, not
instructions. It is not a system/developer prompt or a substitute for the task.

Live text never becomes a conversation segment. Opening returns a small metadata
receipt, not a second file copy. Closing removes this workspace payload on the
next request; it **does not erase earlier ordinary reads**. Those remain subject
to the existing archive/dedup/compaction policy and are recoverable evidence.

Only ids, paths, selectors and limits persist with the conversation. On reload,
views are pending until refreshed. Before every model request, files are read
from disk and hashed; mtime/size alone are not trusted. Before/after hashes catch
changes during extraction. Permissions and roots are rechecked on refresh.
External changes take effect on the **next request**, not during an in-flight one.

Each open document is a message of its own. Its place in the request follows the
provider cache (README.md, "Cache points"): in the history behind the previous
request's end while it is unchanged, where it is read from the cache; at the end of
the request while it is being edited, where a new version costs only itself.
Workspace tokens contribute to occupancy and compaction pressure but cannot be
silently summarized away. Unavailable files and oversized updates have visible
error frames with **no stale payload** and can recover on a later refresh.

## Limits and representations

Initial limits: 8 open files, 8000 tokens per view by default, up to 20000 per view,
24000 aggregate workspace tokens, 20 MB source-file guard. These are estimated
token budgets, not exact provider tokenizer counts. Opening an oversized or
incomplete view is rejected rather than silently capped; narrow it explicitly.
A failed replacement preserves the existing view specification.

Code, TeX, HTML, structured source, notebooks and CSV/TSV retain raw source text
for full views. PDF/DOCX/PPTX and binary tables use existing readable text
extraction; this is **not visual/layout evidence or a byte-for-byte document
representation**. Use render/view tools separately for figures/layout. ODT/EPUB
need conversion first. Existing parser/extraction limits are guarded against
where detectable; this is an MVP, not a universal lossless document parser.

## Document-scoped editing adapters

Adapter permissions are opt-in. Native schemas stay unchanged when files open or
close: one fixed `document_call(operation, arguments_json)` gateway remains
available while any adapter permission is enabled. It is not in the tool list:
the app adds it whenever a document or canvas operation is on and the route has
no native deferred tools (OpenRouter; on the direct Anthropic API the operations
are deferred tools instead, CANVAS_AND_SCRATCHPAD.md). Operation names and closed
parameter schemas appear only in the generated catalog for compatible, ready
views. `arguments_json` is an object encoded as JSON text; invoke the operations
below through that gateway, not native per-format calls.

The catalog is a live view like the documents and is placed with them (README.md,
"Cache points"), always after the system prompt. No compatible views means
no catalog; it is generated afresh, not persisted as history. Lifecycle tools
remain native. A refused call says which reason applies: an unknown operation, one
that is not enabled in the chat, or nothing open that it works on. Runtime guards recheck id, format, permission and source version;
extra fields, malformed JSON and duplicate keys are refused. An unrelated open
file cannot authorize a closed/incompatible target.

On the direct Anthropic API with a model that supports mid-conversation tool
changes, the operations are native deferred tools instead: declared once with
`defer_loading`, surfaced and removed by `tool_addition` / `tool_removal` system
messages appended to the history, with no gateway and no catalog
([CANVAS_AND_SCRATCHPAD.md](CANVAS_AND_SCRATCHPAD.md#native-dynamic-tools-on-the-direct-anthropic-api)).
Canvas operations are late-bound the same way, while a canvas is open.

This protects the unchanged native-schema/frozen-message prefix, not the whole
request: changing the catalog invalidates its suffix. Provider cache hits and
actual completed-task costs remain measurements. Catalog tokens count toward
pressure/compaction and are displayed separately from native tool overhead.

- `document_prepare_edits(changes)` prepares one multi-file, multi-edit plan for
  1..30 open UTF-8 sources, with up to 100 ordered exact-string edits per file.
- `document_json_update(document_id, expected_version, updates)` replaces existing
  JSON nodes using RFC 6901 pointers. Values are JSON-encoded `value_json` strings.
  No append/delete or duplicate keys/pointers; nonstandard constants and numbers
  that would lose decimal precision on serialization are refused.
- `document_table_update(...)` changes CSV/TSV cells using zero-based **data** row
  indices (header excluded) and exact column names. Requires comma/tab dialects,
  unique nonempty headers and rectangular rows. Cells remain strings, including
  leading zeroes and embedded CR/LF. Quoting may normalize, as shown in the diff.
- `document_apply_edits(plan_id, rollback=false)` applies a reviewed scoped plan
  only while **all** its original target ids remain open and ready. It rechecks
  hashes, journals per-file atomic replacements and supports hash-checked rollback.
  Closing/reopening creates a new id, so old plans do not silently bind a new view.

Preparation does not change sources. Review the complete diff before applying.
JSON formatting/escapes and table quoting may normalize; generic text patches
are preferable when byte-level formatting matters. BOMs and record newlines are
preserved. Structured adapters use a verbatim plan path so cell newlines are not
silently rewritten. Selected views are supported, but can omit changed regions:
the diff must still be reviewed. Apply refreshes live views immediately and its
receipt contains metadata only, never another copy of the document.

### Native-format operations

- **PDF:** `document_pdf_inspect`, `document_pdf_text`, `document_pdf_render`.
  Read/render-only, 1-based pages, bounded output, source-hash-checked image
  receipts; no silent text truncation, OCR, redaction or reflow.
- **DOCX:** `document_docx_inspect`, `document_docx_replace`, apply. Zero-based
  top-level body paragraphs; unique replacements within one direct text run.
  No run-crossing edits, fields, tables, headers or drawings.
- **XLSX:** `document_xlsx_inspect`, `document_xlsx_update`, apply. Exact sheet
  names, existing stored cells and scalar JSON values. Refuse formula/rich-text
  edits and unsafe numeric precision; caches stay stale until Excel recalculates.
- **OBJ:** `document_mesh_inspect`, `document_obj_transform`, apply. Positive
  uniform scale then translation of finite vertices. Faces, normals and material
  lines stay unchanged; external material files are never loaded.
- **Binary STL:** `document_mesh_inspect`. Exact triangle-byte-count validation;
  bounds/counts only. ASCII STL and STL edits are not supported yet.

DOCX/XLSX stage a binary replacement plus change summary for review. Only selected
XML members are rewritten; other member contents are copied unchanged. Run/cell
styles are preserved in fixtures, not a guarantee of unchanged visual layout:
verify in Word/Excel. Signed packages, DTD/entities, unsupported compression and
oversized expanded packages are refused. Native operations use an 8 MiB source
guard, separate from the 20 MB live-view guard.

Mesh summaries are **not visual evidence** or full CAD/topology validation; there
is no 3D renderer yet. PDF text does not prove layout: use rendering. PowerPoint,
Parquet and other binary formats still have extraction-only support.
General filesystem tools are still governed by their existing permissions, not
this suite's open-document guard. This is scoping, not a security sandbox.

Writes remain serial ordering barriers. Windows sharing-conflict replacements
retry briefly with a source-hash recheck before each attempt. Multi-file apply is
**not a global transaction**; the journal makes partial application explicit.
Grouping edits saves model requests without requiring concurrent writes. Opening
or closing a format changes the catalog, not native schemas. The unchanged frozen
prefix is preserved; measure net cost rather than
assuming fewer exposed tools necessarily means a cheaper request.

## Batching: billing versus execution

The existing model protocol already accepts several tools in one response. Each
new model request still processes its input/cache, so grouping genuinely
independent calls reduces repeated reads of the same prefix. Guidance now asks
for structure discovery once, then grouped focused reads, avoiding speculative
calls whose arguments/results depend on earlier operations.

Execution now overlaps up to **four static read-only tools**: file/document/focus
reads, directory listing, glob/grep, static file/folder outlines. Synchronous
parsing runs off the event loop. Tool results commit in the model's original
order. Every other operation is a barrier: edits, shell commands (even apparently
read-only shell scripts), context changes, recalls, jobs and LLM summaries.

Concurrency alone saves latency, **not cache billing**. Batching model calls is
the cost-saving mechanism. Cancellation prevents subsequent serial effects;
already-started reads may finish. Disconnect cleanup cancels/joins asyncio tasks
before releasing the turn lock; underlying read-only worker threads can finish
without committing discarded results. Existing pair repair/checkpoints remain.
New usage records also record `tool_calls` per response, when usage was returned,
so planning can be monitored rather than inferred from concurrency.

## Validation and next experiment

- `python -B -m tests.run_all`: isolated offline unit/integration regressions.
- `python -B -m tests.workspace_ui_smoke`: mocked Edge open/refresh/close controls;
  screenshot `.ctxagent/workspace-ui-smoke.png`.
- `python -B -m tests.workspace_benchmark`: scripted eight-file task plus synthetic
  blocking-read timing; writes `.ctxagent/workspace-benchmark.json`.

The scripted task compares 9 model round trips for one-at-a-time reads with 2 for
one grouped batch plus completion, obtaining identical tool observations. This
is **not live billing, autonomous planning performance or task-quality evidence**.
Real disk speedups may be smaller than synthetic delayed-read speedups.

After restart, verify the backend with `python -B -m ctxagent.check_backend`, then
exercise an actual working document: open selected sections, edit, confirm the
latest version in the next request, close with a note. Compare provider-recorded
cost per completed task, prompt/cache tokens, model request count, tool batch size,
correctness and recalls needed. Do not optimize cache-hit fraction in isolation.

## Startup configuration for this delivery

Nine native-format operation permissions are now queued for this chat
(`82a7f15e4d544984`), bringing its next-startup list and saved new-chat template to
47. All existing chat config rows were verified unchanged while queuing; other
chats are not migrated. The Context enable-tools button includes these operations.
Backup: `.ctxagent/backups/document-catalog-settings-03c6bd3a5cae4f1b9cf93e316e385503.json`.
Running backend still has the prior build. Restart after this turn, then run
`python -B -m ctxagent.check_backend`; expect `document-catalog-v1`.

### Previous edit-suite startup configuration

The four new editing tools are queued for this chat (`82a7f15e4d544984`) on next
startup (38 enabled tools). The saved new-chat template also includes them.
All existing chat configurations, including the active one, were checked unchanged;
other chats are not migrated. Settings backup:
`.ctxagent/backups/document-edit-settings-9e169030c8a8438593502c8a2f3ec28b.json`.
The Context panel's enable-tools button now opts in to lifecycle and editing tools.
The earlier workspace settings backup is retained separately.

## Catalog/native-format delivery validation

Full offline suite, JS syntax and whitespace checks passed in job
`5e01e56626bb4e429064f8f8a474b347`. Ten new dispatcher/format regressions cover
request ordering, unchanged frozen/native prefixes, permission/schema guards,
PDF text/rendering, concurrent read dispatch, Office styles/member preservation,
exact rollback, stale/closed/signed targets, OBJ transforms and STL/XML/ZIP limits.
Mocked Edge workspace smoke passed (`0d538507eb8c477493153eb7fb15c93e`), with
visual screenshot inspection. Final source validation also passed in job
`2f706e9382794df187fa61641fc8c6f1`: full offline suite, JS syntax, whitespace,
workspace UI and the existing cost/chat/Flex UI smokes. The final workspace
screenshot and a source-hash-checked native PDF render were inspected visually.
Demo PDF: `.ctxagent/document-format-demo/scoped-render.pdf` (offline fixture).
No provider-billed savings or autonomous document-task quality claim is made.

## Previous edit-suite delivery validation

Final isolated full suite, JavaScript syntax and whitespace checks passed:
job `1b2ae5ccc1df47f78f50350b949bea2b`. The new document-edit module has
16 regressions covering tool-schema gating, real tool roundtrips, closed/stale
scopes, ordered multi-edits, selected views, rollback, BOM/record/cell newlines,
JSON ambiguity/precision, and Windows replacement retries/concurrent changes.
Adapter receipts are asserted metadata-only, without live document bodies.

Updated mocked Edge workspace smoke and JavaScript syntax checks passed:
job `6dfb346d549644cbbdcba0c0fa0f9161`. Its screenshot was visually checked.
Live source edits and close handoffs were exercised in this chat; specialized
adapter execution remains offline-tested until the next server restart.
No provider-billed savings or autonomous task-quality claim is made.

## Previous delivery validation record

Previous isolated full suite, JavaScript syntax and all four mocked Edge UI suites:
job `7380905ee17e44df9f455d8fc3b7e3be`. The workspace unit/integration module
contains 13 regressions, including stable cache-slot placement, automatic refresh,
metadata-only receipts, PDF/range views, close notes, busy guards and disconnect
cleanup. The workspace screenshot was visually checked.

Latest clean scripted benchmark: job `c01f2ae9014f4041bf3fa21f19411811`, with
planned/result manifests `.ctxagent/workspace-experiment-*-v2.json`. Fixed read
results match and model requests are 9 versus 2. Timings/token estimates can vary
with host load and temporary-path tokenization; these are not paid model runs.
Previous benchmark output was preserved as `workspace-benchmark-v1.json`.
