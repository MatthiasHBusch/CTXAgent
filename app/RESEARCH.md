# Context Compaction in Agentic Systems — State of the Art (Sept 2026)

> **Note (2026-11-05):** the engine's incremental tiers (supersession and dedup, age-tiered
> masking and offload, significance-ranked condensation) were removed; deep compaction is now
> its only automatic compaction (README, "The compaction engine"). The survey stands; the dedup, masking and significance passes this design took from it are gone. The code it names
> is in the history up to commit 8af424c.

Research notes collected before building `ctxagent`. Sources at the bottom.

---

## 1. Why compaction exists

An agent loop appends to a message list forever. The dominant cost is not the model's
output but the **re-sent prefix**: every turn re-uploads the entire history. Two pressures:

1. **Hard limit** — the context window (200k for Claude-class, 1M for some others).
2. **Soft limit** — attention dilution. Long, noisy contexts measurably *reduce* solve rates
   even when they fit. Slipstream's framing: "context length alone doesn't solve the problem."

And one counter-pressure that most naive implementations get wrong:

3. **Prompt cache continuity.** Providers cache on an exact *prefix* match. Rewriting any
   message invalidates everything after it. TokenPilot states it bluntly: "constant layout
   mutation shatters prompt prefix continuity" — text-pruning schemes routinely cost more in
   cache misses than they save in tokens.

**This third point is the design constraint that matters most**, and it's exactly what the
"only compact inside the last 50k, freeze everything before" requirement encodes.

---

## 2. What the shipping systems actually do

### 2.1 Claude Code — three tiers, escalating cost

| Layer | Cost | Mechanism |
|---|---|---|
| 1. Tool-result trimming | 0 LLM calls | Older tool results replaced by `[Old tool result content cleared]`. Tool *calls* and reasoning preserved. |
| 2. Microcompaction | 0 LLM calls | Keeps a **"hot tail"** of recent tool outputs fully visible; older bodies offloaded to disk, kept in context as a path reference. Applies to Read/Bash/Grep/Glob/Edit/Write/WebSearch/WebFetch. Described internally as "a cache policy plus a persistence format". |
| 3. Auto-compaction | 1 LLM call | Last resort. Triggers at roughly `effective_window − 13,000` tokens. Produces a **9-section structured summary** and rebuilds context from it. |

The 9 sections: user intent · core technical concepts · files of interest · errors and their
resolutions · reasoning chain · summary of user messages · TODOs · current work · next step.
The summarizer is instructed to **quote key phrases verbatim** rather than paraphrase — a
cheap and effective hedge against lossy abstraction.

After compaction it rebuilds with the summary plus recent files, todo list, plan state and
hook outputs. It deliberately reserves headroom for both output and for the compaction call
itself, and it skips compaction on small sessions.

The API-level version of this (`compact_20260112`) exposes the knobs: a `trigger` on
`input_tokens` (min 50,000), optional custom `instructions`, and `pause_after_compaction` so
the client can splice the last N messages back in verbatim after the summary. It also
recommends a `cache_control` breakpoint at the end of the system prompt so the system prompt
survives compaction events in cache.

**Takeaways I'm keeping:** escalating tiers (free before expensive); hot tail; disk offload
with a path stub; quote-don't-paraphrase; preserve recent messages verbatim after a summary.

### 2.2 OpenCode — prune first, summarize only on failure

- **Prune before summarize.** Verbose old tool outputs get replaced with truncation markers.
- **Minimum-gain gate:** pruning only runs if it can free **more than 20,000 tokens**. Small
  cleanups are skipped — they cost cache invalidation and buy nothing.
- **Protected zone:** the most recent **40,000 tokens are untouchable**, a "safety cushion".
  Skill-type tool outputs are never pruned; the last 2 user turns are fully protected.
- **Non-destructive:** marks with a timestamp (`compacted = Date.now()`) instead of deleting,
  so content can be recovered.
- **Nested summaries:** when a new compression range overlaps an old summary, the earlier
  summary is *nested inside* the new one rather than re-summarized flat — information degrades
  through layers instead of being diluted away each pass.
- **Model-driven timing:** exposes a `Compress` tool to the model so the agent can compact at
  a task boundary rather than at a blind token threshold.

**Takeaways I'm keeping:** minimum-gain gate, protected recent zone, non-destructive marking,
nested summaries.

### 2.3 Codex CLI — server-side, opaque, with re-read

- Triggers at `effective_window − 13,000`, where
  `effective_window = context_window − min(max_output_tokens, 20000)`.
  On a 200k model that's ~167k.
- Two-tier: a cheap **session-memory compact** that reads already-structured state and avoids
  an LLM call entirely, and a **server-side compact** (`POST /v1/responses/compact`) returning
  an encrypted opaque blob.
- **Physically deletes all assistant replies and tool messages but preserves every user
  message verbatim.**
- After compaction it **re-reads up to 5 recently edited files, 5,000 tokens each, 50,000
  total.** A v0.118 regression that doubled compaction frequency turned this into a cascading
  re-read loop that blew up token usage — a good cautionary tale about hysteresis.

**Takeaways I'm keeping:** user messages are sacred; post-compaction re-hydration of live
files; and *do not let compaction thrash* (hysteresis + cooldown).

---

## 3. What the research literature says

### 3.1 Observation masking ≈ LLM summarization, at half the cost

"The Complexity Trap" (Lindenbauer et al., arXiv 2508.21433) is the most decision-relevant
paper here. On SWE-bench Verified with SWE-agent:

- Replace older tool observations with a placeholder (`"Previous 8 lines omitted for brevity"`),
  keep the most recent **M** observations in full, preserve **all** reasoning and actions.
- This **halves cost versus the raw agent while matching — sometimes slightly exceeding — the
  solve rate of LLM summarization.**
- A hybrid (mask, then summarize only what's left) cuts a further 7% / 11%.

The conclusion is unintuitive and important: **the expensive, clever thing is not better than
the cheap, dumb thing.** Do the free deterministic work first and hard; call a model only for
what deterministic rules genuinely cannot handle.

A follow-up ("Masking Stale Observations Helps Search Agents — Until It Doesn't") maps the
regime where this breaks: masking hurts when the agent needs to *re-derive* facts from an
observation it can no longer see. That argues for a **recovery path** rather than for masking
less.

### 3.2 TokenPilot — cache-efficient context management

Directly targets the prefix-continuity problem. Dual granularity:

- **Global: Ingestion-Aware Compaction.** A deterministic harness at the *entry point*, shaping
  the sequence layout during cache warm-up rather than retroactively mutating an established
  cache. Splits messages into:
  - **internal** (task prompts, reasoning traces, tool calls) → preserved at maximum utility;
  - **environmental** (tool responses) → deterministic reduction passes.
  - **Prefix stabilization:** replace volatile fields (timestamps, absolute paths, tool
    definitions) with static placeholders to get a **byte-identical prefix from turn one**.
  - **Observation reduction:** rule-based HTML slimming, truncation, **deduplication**. A
    **recovery tool** can pull back a compressed payload on demand, and doing so *permanently
    disables truncation for that content path*.
- **Local: Lifecycle-Aware Eviction.** Segments go `active → completed → evictable`. Reaching
  "completed" does *not* evict; the segment holds its cache slot while residual relevance is
  non-zero. State updates are batched (B=3 turns) instead of turn-by-turn to avoid churn.

Results: **61% cost reduction** on PinchBench at equal-or-better performance (81.0 vs 80.5);
**87%** on Claw-Eval at a small quality cost (60.8 vs 63.4). Cache-read tokens drop 65%;
**79–83% cache hit rate** maintained.

**Takeaways I'm keeping:** the internal/environmental distinction, dedup as a first-class pass,
the recovery tool that self-disables future truncation for that path, batched state updates,
and above all: *measure cache hit rate, not just token count.*

### 3.3 Zone-based progressive pruning (MatClaw)

History is divided into zones by age; compression gets **progressively more aggressive from
newest to oldest**. Recent = untouched, older = truncated, oldest = summarized. Includes
explicit **deduplication of repeated observations**: near-identical outputs across turns are
identified by similarity, one reference retained, duplicates dropped.

### 3.4 Slipstream — compaction is a correctness problem, not just a cost problem

Validates a summary by asking whether the agent can execute the **same trajectory** from the
compacted history as from the full one. Failure modes: information loss, trajectory divergence,
and context dependency (summaries that work until the agent needs to revisit an early decision).
Trajectory alignment as ground truth for compression quality.

---

## 4. Synthesis — the design this implies

Everything above converges on six rules:

1. **Free before expensive.** Deterministic passes (dedup, masking, truncation) recover most of
   the tokens at zero LLM cost and match summarization's quality. Summarize only the residue.
2. **Never rewrite the cached prefix.** Confine all mutation to a bounded recent window; freeze
   everything older byte-identically. This is the single highest-leverage constraint.
3. **Internal ≠ environmental.** User messages, assistant reasoning and tool *calls* are cheap
   and high-signal — keep them. Tool *results* are expensive and mostly stale — compress them.
4. **Lossy, but recoverable.** Offload bodies to disk, leave a stub with a handle, give the model
   a tool to pull it back, and stop truncating that path once it does.
5. **Hysteresis.** Minimum gain before acting (OpenCode's 20k), cooldown between deep passes,
   and never compact right after compacting — Codex's regression shows the failure mode.
6. **Nest, don't re-summarize.** Summaries of summaries degrade. Nest the old inside the new.

Two things I did *not* find anyone shipping, which are in the brief and which I've built:

- **Explicit supersession of re-read files.** Everyone deduplicates "identical" observations by
  similarity. Nobody appears to model the *causal* case: read v1 → edit → read v2 → edit → read
  v3, where v1 and v2 are not near-duplicates of v3 but are nonetheless *definitively obsolete*
  because the agent itself caused the change. That's a much stronger signal than similarity and
  it's cheap to compute from the tool-call graph. See `compaction/dedup.py`.
- **A significance ranking that survives into the summarizer.** Prior work is age-based; a
  segment's *importance* is only implicitly modelled. Scoring it explicitly lets low-significance
  content be dropped hard while a high-significance segment of the same age is kept verbatim.
  See `compaction/significance.py`.

---

## Sources

- [How Claude Code works — Claude Code Docs](https://code.claude.com/docs/en/how-claude-code-works)
- [Compaction — Claude Platform Docs](https://platform.claude.com/docs/en/build-with-claude/compaction)
- [Inside Claude Code's Compaction System — Decode Claude](https://decodeclaude.com/compaction-deep-dive/)
- [Context Compaction Deep Dive: Codex CLI, Claude Code, OpenCode](https://codex.danielvaughan.com/2026/04/14/context-compaction-deep-dive-codex-cli-claude-code-opencode/)
- [Shedding Heavy Memories: Context Compaction in Codex, Claude Code, and OpenCode](https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode)
- [The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization (arXiv 2508.21433)](https://arxiv.org/abs/2508.21433)
- [Masking Stale Observations Helps Search Agents — Until It Doesn't (arXiv 2606.00408)](https://arxiv.org/html/2606.00408v1)
- [TokenPilot: Cache-Efficient Context Management for LLM Agents (arXiv 2606.17016)](https://arxiv.org/html/2606.17016v1)
- [Slipstream: Trajectory-Grounded Compaction Validation (arXiv 2605.08580)](https://arxiv.org/pdf/2605.08580)
- [MatClaw: An Autonomous Code-First LLM Agent (arXiv 2604.02688)](https://arxiv.org/pdf/2604.02688)
- [Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures (arXiv 2604.03515)](https://arxiv.org/pdf/2604.03515)

## Added 2026-10-20

- Li et al., *Self-Compacting Language Model Agents*, arXiv:2606.23525 (2026): model-invoked
  compaction tool + interval-probed rubric + gates/backstop; open-weight models only;
  "unprompted models cannot reliably tell when their own context is rotting". Our benchmark
  (bench/RESULTS_P2.md) shows Opus 5.5 doing it without the rubric and GPT-6.1 Sol not.
