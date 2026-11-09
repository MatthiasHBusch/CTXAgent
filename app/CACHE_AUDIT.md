# Cache and cost audit — 2026-09-15

## Findings and changes

1. **Lifetime hit rate masked recent improvements.** The live aggregate at audit start was 727,082 cached / 5,099,372 prompt tokens (14.26%). This includes earlier poor behavior. UI now shows lifetime, last request, and prompt-token-weighted last ten agent requests separately.
2. **Wrong cache-write field.** OpenRouter returns `prompt_tokens_details.cache_write_tokens`; old accounting checked other names. Corrected with backwards-compatible aliases. Cache writes are not free on all models.
3. **Summary costs missing.** Non-streaming compaction/document-summary calls discarded usage. They now record every returned usage object, including a billed empty response that triggers a retry. Summary costs are included in total and shown as a subset.
4. **Missing explicit session affinity.** Added stable per-chat `session_id`, and a separate `:summary` session for auxiliary calls. Provider allowlists/fallbacks remain intact; no silent provider pin that defeats regional policy.
5. **Explicit caching only implemented for Anthropic.** Added documented OpenAI GPT-5.6+ `prompt_cache_breakpoint` hints, leaving automatic caching enabled. No guessed hints for older models.
6. **Breakpoint silently disappeared on empty assistant/tool-call messages.** Walk backwards to a cacheable block. Keep a previous stable anchor when the frozen prefix advances substantially, rather than relying entirely on Anthropic's bounded look-back.
7. **Prospective versus real frozen boundary.** Model request now uses `Conversation.frozen_upto`, not a recalculated boundary that hasn't actually been frozen.
8. **Provider reasoning continuation discarded.** Preserve returned structured reasoning blocks in order across tool round trips. Treat encrypted content as opaque. Do not display it in UI or copy it to usage logs. This avoids protocol/continuity problems; no claim that this alone explains historical misses.
9. **Cumulative SSE usage could be double-counted.** Merge snapshots and emit one final accounting event. Capture model/provider/generation ID for auditability.
10. **Frozen incorrectly labelled cached.** UI now says cache-eligible. Actual provider cache counts remain the source of truth.

## Diagnostics

Each new agent request stores content hashes, message/token estimates, schema hash, actual frozen boundary, prior shared-prefix estimate and elapsed gap. Public diagnostics show possible local invalidation causes: changed model/schema/reasoning settings, prefix rewrite, deep compaction/restore, >5 minute gap, or provider switch. Prefix/token estimates are not provider billing measurements.

No old per-request details existed, so the audit cannot conclusively attribute historical misses to endpoints versus rewrites. A restart, changed tool schema, deep compaction, provider cache expiry/eviction, or edited system message can legitimately cause cold requests.

## Live validation

Three explicitly launched synthetic requests to `openai/gpt-5.6-luna`, no user documents:

| Request | Prompt | Cache read | Cache write | Reported cost |
|---|---:|---:|---:|---:|
| First | 2193 | 0 | 2190 | $0.00055410 |
| Second | 2193 | 2179 | 11 | $0.00005293 |
| Third | 2193 | 2179 | 11 | $0.00005293 |

Warm hit rate: **99.36%**. Warm request cost decreased **90.45%** versus initial write. Total test cost: **$0.00065996**. This validates transport compatibility, not a guaranteed full-agent hit rate. Raw telemetry is in `.ctxagent/cache-smoke.json`; smoke script is opt-in and excluded from offline tests.

## Cost semantics

- Authoritative number: `usage.cost` reported by OpenRouter.
- Categories: uncached input, cache read, cache write, visible output, reasoning. Reasoning is a subset of completion tokens; never add it twice.
- If reasoning count is unavailable, completion stays in `output_unsplit`, not falsely labelled visible output.
- Category dollar values are explicitly **estimates** based on request-time endpoint tariffs, including long-context overrides. When provider has ambiguous service-tier prices, use only agreed rates, or a unique tariff that reconciles with the reported total (labelled inferred). Do not pretend this is provider itemization.
- Keep reconciliation residual, unknown tariff amounts, and missing-cost request counts visible.
- Historical reported total remains `legacy_unallocated_cost`. Old missing summary costs cannot be recovered from retained aggregates.
- New raw usage/pricing/diagnostics are appended to `.ctxagent/usage/<chat-id>.jsonl`. No prompt or reasoning text is logged there. Interrupted streams with no returned usage remain a limitation; generation-ID reconciliation can be added later.

## Tests / visual verification

Offline tests cover cache-block placement, old-anchor retention, prefix mutations, model/schema changes, reasoning round trips, usage snapshot merging, summary retries, pricing overrides/ambiguity, zero/missing counters, historical migration, and no reasoning double charge. Headless Edge UI smoke uses synthetic data with no live chat mutation; its screenshot was visually inspected.

## Sources checked

- https://openrouter.ai/docs/guides/best-practices/prompt-caching
- https://openrouter.ai/docs/guides/guides/usage-accounting
- https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
- Live public model endpoint pricing for the configured agent and summary models.
