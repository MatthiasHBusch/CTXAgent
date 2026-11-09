# Agent efficiency audit

> **Note (2026-11-05):** the engine's incremental tiers (supersession and dedup, age-tiered
> masking and offload, significance-ranked condensation) were removed; deep compaction is now
> its only automatic compaction (README, "The compaction engine"). This audit describes the earlier engine, and `tests.efficiency_replay` went with the tiers. The code it names
> is in the history up to commit 8af424c.

## Bottom line

Optimize **cost per successful task**, not cache-hit percentage in isolation. The largest opportunity is preventing unnecessary observations from entering the prompt in the first place. Frequent rewrites can cost more than repeatedly reading a larger cached prompt, especially with a 0.05x cache-read tariff.

This audit read the implementation and read-only saved chat/usage records, added bounded diagnostics, and ran two offline policy sweeps. It did not run paid evaluation generations or replay historical tools. No active chat configuration or saved defaults were changed. Source changes require a backend restart; existing tool selections are not automatically extended.

## What the current context actually contains

The most recently updated saved chat matches this audit session. At one inspection checkpoint its local estimate was **76,783 tokens**, including **4,357 tokens of tool definitions**, against a configured 200,000-token window and 179,000 usable tokens (8,000 output reservation + 13,000 safety). The conversation continues growing; this is a checkpoint, not the final size or an exact provider count. I cannot directly introspect the model's native remaining-token counter.

At that checkpoint:

- Tool results: 56,651 estimated tokens; tool calls/arguments: 15,391.
- Actual frozen prefix: 63,862 estimated tokens, including 55,596 tokens of original tool observations. Ordinary incremental passes cannot reduce these now-frozen bodies.
- A single diagnostic shell result contributed 10,935 estimated tokens. That was unnecessary verbosity in this audit itself: a full JSON usage dump should have been an aggregate report from the outset.
- Full `read_file` results contributed 27,718 estimated tokens; focused readers were substantially smaller in aggregate, although this is not a matched-task comparison.
- Incremental document compaction is **disabled**. Changing rewrite cadence alone therefore cannot compact documents. Deep compaction is enabled, triggered at 90%, with a 10% total-size target.

Local/provider token estimates differ materially: across the inspected recent records, reported/local ratios ranged roughly 0.66–0.96 in this session and 0.82–0.93 in larger sessions. Do not present the local tokenizer as a precise provider token counter. Image and opaque-reasoning estimates further complicate comparison.

## Measured cost and cache evidence

Recorded API totals are distinct from inferred component tariffs. The following input-cost fractions use recorded `upstream_inference_prompt_cost` and matching reported totals, not an assumed decomposition.

| Recorded session | Recent agent requests | Weighted cache hit | Input-side fraction | Locally diagnosed prefix changes |
|---|---:|---:|---:|---:|
| Larger agent-development session (`7f995507…`) | 20 | 75.7% | 96.8% | 17 |
| Paper-work session (`200d5126…`) | 20 | 98.4% | 87.1% | 0 |
| This session, earlier checkpoint | 6 | 64.0% | 83.4% | 0 |

These cohorts differ in workload/model/age. They are not controlled A/B evidence. The current session's low early hit rate was primarily prompt growth/cold writes, not locally observed prefix rewrites. Later, its 15-request weighted hit rate rose to 90.7%.

In the paper session, total recorded summary cost was $0.033 out of $16.534. Optimizing the summarizer's tariff first would miss almost all the opportunity. That does not establish its summaries are accurate: cheap but lossy summaries can cause expensive downstream mistakes.

## New offline sweep

Commands:

```powershell
python -B -m tests.efficiency_replay --chat-id 7f995507e6c84720 --output .ctxagent/efficiency-replay-7f995507-v1.json
python -B -m tests.efficiency_replay --chat-id 200d51266e5f42e6 --output .ctxagent/efficiency-replay-200d5126-v1.json
```

The implementation uses the actual scheduler, including its pressure and new-token overrides. It replays **72** and **43** retained tool batches, respectively. It preserves explicit pins, reconstructs recoverable original bodies, and redirects offload writes to temporary storage. The first trajectory had 44 recoverable originals, the second zero. Earlier deep-compacted history is absent. Aggregate output includes source/config hashes, not chat contents.

Assumptions: ideal unlimited shared-prefix caching; local token counts; static tool schema; 1.25x write tariff; either 0.10x or 0.05x read tariff; no output/reasoning/quality changes. Deep and paid semantic compaction are disabled for the sweep. **These are normalized input-cost comparisons, not provider bill forecasts or evidence of equal task quality.**

| Policy | Development replay: cost vs disabled (.10 / .05 reads) | Final context | Paper replay: cost vs disabled (.10 / .05 reads) | Final context |
|---|---:|---:|---:|---:|
| Incremental off | 1.000 / 1.000 | 173k | 1.000 / 1.000 | 186k |
| On, 10k hot, interval 5 | 1.231 / 1.435 | 158k | 1.017 / 1.032 | 184k |
| On, 50k hot, interval 1 | 1.640 / 2.338 | 120k | 1.739 / 2.482 | 75k |
| On, 50k hot, interval 10 | 1.003 / 1.172 | 127k | 0.965 / 1.201 | 92k |
| On, 80k hot, interval 10 | 0.957 / 1.126 | 121k | 0.930 / 1.185 | 69k |

Important implications:

1. **Do not automatically enable frequent masking to save money.** Every-batch rewriting was expensive on these fixed trajectories.
2. **10k hot is too small for effective delayed reduction of large observations.** Bodies can freeze before they age out of the six-result protection window. In the development replay, pressure forced 43 passes; intervals 5 and 10 became nearly identical.
3. A **50–80k hot budget / interval 10** is a reasonable *attention-oriented trial* if enabling deterministic compaction, not a universal cost optimum. At .10x reads, the best tested rows saved 4–7% of normalized input cost; at .05x reads, even those rows cost 13–19% more over these finite trajectories.
4. Keeping a larger prompt can be temporarily cheaper, but risks attention degradation, long-context tariffs and earlier deep passes. None of those effects were evaluated here.

A simplified break-even model explains the results. For S tokens removed, R surviving tokens invalidated, cache read rate r, write rate w, and one-off compaction cost C:

`future requests > ((w - r) * R + C) / (r * S)`

For example, removing 20k tokens while invalidating 50k surviving tokens takes about 29 future requests at r=.10, w=1.25, C=0, or 60 at r=.05. This is only an economic approximation; token position, cache granularity, quality, eviction and changed context size matter.

## Recommended operating policy

### Apply without a cache reset

- Start repository exploration with `summarize_folder(level=2, token_budget=4000)`.
- Follow with `focus_file(max_tokens=2000–4000)` for named functions/sections. The 24k default is an upper allowance, not a sensible routine target.
- Use `read_file` for exact bounded ranges, not whole modules by habit. Use static outlines rather than paid semantic summaries for navigation.
- Bound grep results and shell output. Report failing tests and the final summary, not successful-test transcripts or entire usage JSON. Keep complete logs in managed jobs and fetch only needed chunks.
- Batch independent reads logically; do not parallelize dependent edits/builds. The current agent loop executes its tool calls serially, so batching saves model round trips but is not backend execution concurrency.
- Prefer `recall` to rerunning expensive commands when evidence was shortened. Prefer a new read when the underlying file changed: a recall is historical evidence, not current state.

### Tune only at a task seam, with measured evaluation

- Keep the current incremental-off setting if input cache reads are very cheap and attention remains adequate. Do not change a working prefix just to improve a hit-rate dashboard.
- If context clutter is the problem, trial deterministic compaction with **50k hot / interval 10 / 20k new-token flush / six recent results protected**. Initially disable paid significance condensation so its separate cost/quality effects are observable. Keep pressure checks; do not delay persistence or tool pairing.
- Consider **75–80% deep trigger** rather than 90% when large observations are possible. A 20–50k tool batch can jump past a 90% threshold. A request-time capacity guard/emergency cooldown override remains a recommended follow-up, not implemented by this audit.
- A 10% deep target is aggressive. Use the new 1,024-token summary floor, a 3,000-token maximum for a trial, and evaluate retained constraints, decisions, identifiers, evidence locators and next steps. Increase the target if summaries repeatedly lose facts; lower prompt size is not worth trajectory divergence.
- Keep system instructions, tool order, model, service tier and provider routing stable within a task. Changing them mid-chat can invalidate a much larger prefix than the schema saving. Choose coding vs paper/science tool profiles when creating a chat; do not dynamically rotate tools each iteration.
- Continue using region-respecting Flex routing where latency/availability permit. Do not sacrifice locality or silently rerun a partially streamed request for a cheaper price.

### Compact earlier; preserve longer

Good candidates for removal at a deliberate boundary: superseded file versions with verified coverage, completed test successes, old build logs, resolved stack traces, redundant directory surveys, old visual evidence, and repeated configuration/usage dumps. Preserve user constraints, rationale, unresolved errors, pending edits/jobs, exact identifiers and evidence locators. Do not truncate a fresh tool result before the model has consumed it.

Large edit/write arguments are also prompt ballast: tool-result masking does not reduce them. Deep compaction of completed call/result groups at a task seam is the safer remedy; arbitrarily rewriting historical calls or opaque reasoning is not.

## Implemented changes

- **`context_report` tool**: read-only bounded report of context makeup, biggest segments/tools/schemas, recent agent-only weighted cache hits, matched reported cost fractions and inactive settings. No message bodies, paths or arguments. No API calls.
- **CLI**: `python -B -m ctxagent.context_report --top 3 --recent 20`. SQLite is opened with `mode=ro`; no session repair, persistence, or network lookup. Existing chats need the new tool enabled after backend restart; new chats include it by default.
- **Correct output headroom**: reserve the full requested output, not at most 20k. A requested 32k output previously under-reserved by 12k. Small windows no longer invent 20k of usable capacity. This is still a local budget estimate, not provider-context validation.
- **Honest frozen stats**: display the actual immutable boundary, not a prospective freeze that has not happened yet.
- **Summary floor**: `summary_min_tokens=1024`, configurable between 128 and 10000 and capped by the configured maximum (for ordinary maxima >=128). If mandatory context prevents the total target, preserve working memory and report the limitation instead of collapsing it to 128 tokens. This is a safety heuristic, not a measured quality guarantee.
- **Shell transcript de-duplication**: long commands execute unchanged but only a short preview is echoed into their result; exact arguments already exist in the tool call. Full stdout/stderr behavior is otherwise unchanged.
- Added an isolated offline test suite and policy-sweep script.

`tail_ratio_min`, `tail_ratio_max`, `max_pressure` and `rehydrate_*` remain legacy config fields exposed by older UI/docs but unused by the current deep planner. Remove/deprecate their UI controls in a separate change rather than suggesting they are active optimizers.

## Follow-up priorities

1. Quality-backed task benchmarks with constraints/fact retention and cost per successful task, not just retained trajectory replay.
2. Economic hysteresis for tiny repeated rewrites under persistent pressure; never suppress emergency capacity protection.
3. Safe request-time capacity guard, including large user inputs and failed/unproductive deep passes.
4. Bounded log/results delivery with lossless disk handles by construction. Several tools currently clip at a character ceiling before conversation-level recall sees the content.
5. Incremental persistence/event-loop profiling. Whole-chat JSON is checkpointed after every tool result for crash safety; large bodies and edit arguments could make CPU/I/O significant. Preserve durability before optimizing this path.
6. Expose per-request cache-miss reasons, request latency and task boundaries in aggregate dashboards. Avoid collecting additional full-content telemetry.
