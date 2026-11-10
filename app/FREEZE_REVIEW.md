# Review before freezing

*Removed on 2026-11-10 together with the frozen prefix: nothing is frozen any more,
curate_context may replace any item, and the cache points each request sends follow the
history (README.md, "Cache points"). The bounded result delivery below still holds. The
rest of this file is the record of how it worked.*

## Current behaviour (since 2026-11-05)

`review_before_freeze` (Engine › Review before freezing) keeps fresh and protected
results out of the frozen, cached prefix: the newest tool batch until the model has
seen it, the newest N observations and the results of the newest N batches, and
pins that can still expire (`compaction/freeze_review.py`). The boundary snaps back
to whole call/result/media groups and never moves backwards.

* Engine-managed chats: the reviewed prefix freezes after every batch. Nothing is
  rewritten before it freezes; the incremental tiers this review once ran (dedup,
  masking/offload, visual retirement, condensation) were removed.
* Self-steered chats: the prefix advances only when the model commits a
  `curate_context` review or calls `compact_context`, and after a deep compaction,
  so the model can still drop results it has not curated yet.

Without the switch, everything older than the working tail freezes after every
batch. The bounded first delivery (`initial_result_max_tokens`, below) is
unchanged. The rest of this file records the October 2026 design, when the review
also cleaned the prefix up before freezing it; the replay and the
`ctxagent.efficiency_profile` activation it mentions were removed with the tiers.

## History (2026-10): linked review → compact → freeze

The previous freeze-first ordering could permanently retain a large observation
before the cadence-controlled cleanup saw it. The new opt-in policy makes
promotion and cleanup one ordered cycle in the engine's control flow:

1. Check pressure and persistent cadence counters after every batch.
2. If deferred, neither rewrite nor extend the immutable boundary. Tool effects
   and each result are still checkpointed immediately.
3. On a due cycle, select the retiring **unfrozen** prefix and snap the boundary
   backwards to whole call/result/media groups.
4. Review that prefix using deterministic dedup, masking/offload and visual
   retirement. Selective semantic condensation remains pressure-gated.
5. Only then freeze the checked prefix. Future incremental cycles never rewrite it.
6. A pressure-triggered/manual deep pass can rebuild history separately. Its new
   summary is checked for emptiness, token budget and net savings before freezing.

Fresh parallel batches survive even when they contain more observations than the
configured protection count. The last six observations, last two visual
observations and all pins remain protected from incremental shortening. Seen
images may freeze unchanged; keeping the newest two images native no longer pins
the entire later mutable suffix. Fresh/unseen image batches still cannot freeze
before their next model request. Deep
compaction retains the newest indivisible group and explicit pins, but can
archive recall-pinned history again. Exact originals remain recallable.

A failed deterministic review does not promote an unchecked prefix or clear its
pending cadence counters. Scheduler state and boundaries survive reloads.
Already-frozen legacy history is **not** silently thawed on migration.

## Avoiding verbose output at the source

The system instructions now prefer overview → focused read, bounded selectors
and short log chunks. `initial_result_max_tokens` adds a last-resort first-delivery
cap. It stores exact text in an immutable blob before substituting an explicitly
**INCOMPLETE** head/tail excerpt and recall handle. Missing middle content must
not be inferred; recall before relying on it. Errors, recalls, pins and native
images are exempt. Archive write failure leaves the original result intact and
does not repeat the tool's effects.

The cap is independent of incremental document compaction. Setting it to zero
turns it off; the overall compaction master switch also disables it. A capped
observation cannot masquerade as a complete replacement in dedup.

## Proposed opt-in profile

| Setting | Value |
|---|---:|
| Incremental document compaction | enabled |
| Review before freeze | enabled |
| Recent working-tail target | 10,000 tokens |
| Review interval | 10 completed tool batches |
| Early new-token flush | 20,000 tokens |
| Initial tool-result cap | 8,000 tokens |
| Protected recent observations | 6 |

Existing model, reasoning, pricing, pressure thresholds, summary limits, user
instructions and deep target are preserved. A batch can contain several tools;
final assistant replies do not increment the batch count. The mutable suffix
can exceed its working-tail target while review is pending or fresh evidence is
protected. This is intentional, not a false promise of a hard context cap.

No paid review is introduced on every cycle. Free deterministic cleanup does the
routine work; LLM passes run only under pressure or an explicit deep request.

## Validation and limits

`tests.test_freeze_review` covers ordering, immutability, whole parallel batches,
recent observations, native visuals, recall/manual pins, failure, restart,
legacy migration, exact recall, incomplete-result coverage, actual agent delivery,
storage failures without repeated effects, and pending-config migration.
The existing offline suites and mocked browser checks remain part of validation.
The Engine UI exposes both new controls and displays the actual frozen boundary,
not a hypothetical future split.

The replay compares 26 policies with semantic/deep calls disabled and ideal
shared-prefix caching, local token estimates and normalized cache tariffs. It
recovers archived segments read-only, excludes synthetic deep summaries when
reconstructing originals, and excludes an unfinished trailing batch. It never
reruns tools, modifies the source chat, sends documents to a model, or predicts
quality, changed agent behavior, output/reasoning cost, provider eviction or
long-context tariffs. Raw aggregate results are local under
`.ctxagent/freeze-review-sweep-archive-v3.json`; earlier v1/v2 files are historical.

The archived-history replays support a 10k tail and a 10–20-batch region. The
initial profile uses 10 as a conservative cadence, not a proven global optimum:
10 was best in v2, while 20 was cheaper on the later v3 snapshot. At interval 10,
v2 estimated 25.2% / 10.9% input savings and v3 25.9% / 9.2%, respectively at
normalized cache-read tariffs .10 / .05 and write tariff 1.25. Compared with the
legacy enabled policy, v3's linked policy was also cheaper at both tariffs.
V3 reduced the final estimate from 156,475 to 84,176 tokens; its normalized ideal
hit rate was 94.2% versus the disabled baseline's 98.4%. This illustrates why a
higher hit percentage is not itself the objective. A 50k tail performed worse
in both snapshots despite keeping more material. The 8k cap
has little measured effect on this particular trace; it is a conservative
safety net rather than the source of the estimated savings. Live tuning should
compare cost **per useful completed task**, evidence recovery and correctness,
not just cache-hit percentage. Old frozen ballast remains until a deliberate
successful deep compaction; migration itself does not remove it.

## Safe activation and restart

The active backend cannot import the new code by refreshing the browser.
`ctxagent.efficiency_profile` is read-only unless explicitly passed `--apply`:

```powershell
python -B -m ctxagent.efficiency_profile --chat-id 82a7f15e4d544984 --apply --new-chat-defaults
```

It queues this chat's patch for startup, separately updates the saved new-chat
template, and backs up the original settings/config under `.ctxagent/backups/`.
It does not change the active chat row, conversation, timestamps or usage, nor
any other existing chat. Startup applies the queued patch transactionally and
idempotently before sessions are loaded. Keep the backup for rollback.

After this turn finishes:

1. Stop the terminal running `python -m ctxagent` with Ctrl+C.
2. Launch `run.ps1` once; do not launch a duplicate server on the occupied port.
3. Hard-refresh the browser with Ctrl+F5.
4. Run `python -B -m ctxagent.check_backend`.

Expect **CURRENT BACKEND**, `backend_build: freeze-review-v1`,
`restart_required: false`, and accounting version 2. Reopen this chat and ask to
try the new system. The next agent turn can inspect the live profile, observe
review/freeze events and exercise bounded delivery/recall with synthetic evidence.
Do not repeat earlier billable cache/Flex smoke tests merely to confirm a restart.
