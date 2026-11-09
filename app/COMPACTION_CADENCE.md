# Compaction cadence: results and policy

> **Note (2026-11-05):** the engine's incremental tiers (supersession and dedup, age-tiered
> masking and offload, significance-ranked condensation) were removed; deep compaction is now
> its only automatic compaction (README, "The compaction engine"). This file describes the earlier cadence, and `tests.cadence_replay` went with the tiers. The code it names
> is in the history up to commit 8af424c.

## Recommendation

Start at **5 completed model/tool batches** between context-rewriting passes, not 10 individual tools. A batch can contain several independent calls but creates only one next model request. The setting is exposed in Engine; 1 restores the previous cadence. Do not optimize cache-hit percentage alone: a huge cached prompt still costs money and may impair attention or trigger expensive long-context tariffs.

No universal optimum exists. It depends on size of new observations, how much compaction removes, how early a rewrite occurs, cache read/write prices and expected remaining turns. If a rewrite removes S tokens but invalidates R surviving tokens, a rough future-use break-even is:

`remaining requests > ((cache_write_price - cache_read_price) * R + compaction_cost) / (cache_read_price * S)`

This deliberately simplified comparison ignores changed model behavior, newly arriving context, eviction, cache granularity and long-context pricing. It explains why many tiny rewrites can lose money, but never rewriting can also be expensive.

## Offline fixed-trajectory replay

Replayed the last 100 recorded tool batches from this chat, reconstructing original bodies where retained. No tools rerun, no LLM calls, no user files modified. Ideal prefix caching, read price 0.1x input and misses/writes 1.25x. Semantic/deep compaction disabled. Token counts are local estimates and irrecoverable historical bodies remain stubs. This is not a quality test or measured provider bill.

| Hot budget | Rewrite interval | Estimated input cost vs interval 1 | Ideal cache hit | Final context estimate |
|---|---:|---:|---:|---:|
| 10k | 1 | 100% | 95.8% | 195k |
| 10k | 2 | 92.7% | 97.0% | 201k |
| 10k | 3 | 90.8% | 97.4% | 205k |
| 10k | 5 | 89.4% | 97.7% | 207k |
| 10k | 10 | 88.2% | 97.9% | 212k |
| 10k | 20 | 87.5% | 98.1% | 214k |
| 50k | 1 | 100% | 77.1% | 87k |
| 50k | 5 | 60.5% | 90.4% | 87k |
| 50k | 10 | 56.3% | 92.9% | 90k |
| 50k | 20 | 60.5% | 95.4% | 114k |

At 10k there is a broad diminishing-return region around **5–10 batches**; 20 is slightly cheaper in this simplified finite replay but retains more context. At 50k, 10 was best among tested intervals. Do not infer that changing your hot budget to 50k is therefore optimal: the counterfactual does not model answer quality or long-run deep-compaction costs. Your 10k budget was left unchanged.

Replay file: `.ctxagent/cadence-replay.json`; script: `python -B -m tests.cadence_replay`. Results were produced before adding scheduling to the engine. The implemented safety overrides can flush earlier than a pure fixed-N replay.

## Implemented policy

Every batch: checkpoint results, repair/preserve tool pairing, calculate pressure, advance immutable frozen boundary. These are NOT delayed.

Rewriting occurs when any condition holds:

- 5 completed tool batches (configurable 1–100), or
- 20k newly added context tokens (configurable), or
- pressure reaches the earliest applicable threshold: 70% safety threshold, configured deep threshold, or semantic-condensation eligibility (currently 60% with a 75% deep trigger), or
- explicit manual compaction.

Final assistant messages do not count as tool batches. Counters and deep cooldown persist across restarts. Frozen messages remain immutable except explicit deep compaction/restore; batching does not permit freezing and later secretly rewriting old context.

One consequence of a small hot zone: delaying rewriting can let some large results become permanently frozen before they are reduced. The replay includes that effect. Safety and quality should take precedence over a higher cache-hit number.

## Cost monitoring

Compare recent measured input + cache-write + cache-read cost per useful result, not lifetime hit rate. If input-side cost is fraction f of the total, an 11% input-side saving can reduce total cost by at most approximately `0.11 * f`, assuming output/reasoning unchanged. If reasoning or output dominates, changing compaction cadence has limited direct upside.

At the start of this turn the running server was still the old process (started before accounting changes), so no trustworthy new cost fractions existed. The new UI detects this mismatch, and `python -B -m ctxagent.check_backend` verifies the server build. Do not interpret missing category values as zero cost.
