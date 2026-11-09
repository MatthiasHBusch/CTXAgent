# Which model should do the compaction?

Measured, not guessed. Reproduce with `python -m tests.model_eval`.

Compaction is a different job from being the agent: **long input, short output,
strict format, and the only thing that matters is not losing facts.** Models far
apart on agentic benchmarks turn out nearly identical at it, priced up to 50×
apart — so input price dominates, and it is worth measuring once.

---

## Method

Two tasks, run through ctxagent's real production prompts:

| | |
|---|---|
| **A — deep summary** (tier 3) | A 33,246-token synthetic agent trajectory with **15 facts planted** in it (file paths, a dependency version, an error string and its iteration number, a design decision *and its reason*, a tolerance, a test command, a port, an env quirk, an open TODO, a superseded user request, a magic constant, a dataset name, a collaborator's requirement). `max_tokens` is the production value, 10,000 (level 4). |
| **B — condensation** (tier 2, removed 2026-11-05; `tests.model_eval` now runs task A only) | Eight noisy tool outputs with **5 planted specifics**, which also demands **valid JSON** and a compression ratio. |

Scoring is deterministic string matching — no LLM judge, so the numbers are
reproducible and cost nothing to recompute. A fact counts as preserved if any
accepted spelling appears; `"numpy&2.1.3"` means both parts must appear, so
*"Numpy version: 2.1.3"* scores as a hit. That detail matters: the first matcher
was literal and scored phrasing rather than recall.

Every model is checked for **EU/US availability** first and run under the
matching region policy, so the table never compares an in-region model against
one that cannot be used in-region.

---

## Results (2026-09-15, 33k-token input)

**Servable from EU/US providers:**

| model | facts A+B | sec | $/1k compactions |
|---|---|---|---|
| z-ai/glm-5.3-flash | **15+5** | 104 ⚠️ | 7.09 |
| **openai/gpt-5.6-luna** ← default | **15+5** | **13.0** | 10.36 |
| google/gemini-3.1-flash-lite | **15+5** | **5.7** | 12.45 |
| inclusionai/ling-3.0-flash | 14+5 | 10.0 | **0.98** |
| z-ai/glm-4.7-flash | 14+5 | 106 ⚠️ | 4.30 |
| mistralai/mistral-small-3.2-24b | 14+5 | 53 | 3.46 |
| deepseek/deepseek-v3.2 | 14+5 | 25 | 12.12 |
| anthropic/claude-haiku-4.5 | 14+5 | 15 | 48.77 |
| openai/gpt-5-nano | 12+5 | 19 | 2.67 |
| openai/gpt-oss-120b | 12+5 | 41 | 1.79 |

**Not servable from EU/US — excluded by the default region policy:**

| model | facts | sec | $/1k | served only by |
|---|---|---|---|---|
| qwen/qwen3.8-flash | 14+5 | 21 | 7.10 | Alibaba (SG), Makora |
| qwen/qwen3.7-flash | 14+5 | 16 | 1.45 | Alibaba (SG) |
| qwen/qwen3.5-flash-02-23 | 14+5 | 12 | 3.23 | Alibaba (SG) |

Total spend for the sweep: **$0.11**.

---

## On the three suggestions

**`openai/gpt-5.6-luna`** — the pick. Perfect 15/15 + 5/5, 13 s, seven EU/US
endpoints including `azure/eu`. Now the default.

**`z-ai/glm-5.3-flash`** — equally perfect on facts and 30% cheaper, but **104
seconds** against 13. That is not the model thinking; it is queueing on the EU/US
endpoints. Its cheapest in-region options are a DeepInfra fp4 endpoint at ~90%
daily uptime and a Morph endpoint at **11.8%**. Since a deep compaction blocks
the user's turn, that disqualifies it as a default — but it is a good choice if
you are cost-sensitive and compaction happens off the critical path. The
operational picture may well improve; re-run the eval.

**`qwen/qwen3.8-flash`** — cannot be used under your EU/US requirement. It is
served by exactly two providers: **Alibaba (HQ Singapore)** and **Makora**
(region undeclared). A request under the region policy returns
`404 No allowed providers are available`. Same for `qwen3.7-flash` and for
`qwen3.5-flash-02-23`, which was the previous default — that is why it changed.
On quality the Qwen flash models are fine (14/15); the blocker is purely where
they are hosted.

---

## Two findings worth keeping

**1. The reasoning trap.** On a reasoning model, reasoning tokens come out of the
*same* `max_tokens` budget as the answer. Left alone, the budget is consumed
before a single output token is written and the call returns an **empty string** —
which looks exactly like a model that cannot follow the format. This initially
made `qwen3.7-flash` and `gpt-5-nano` look broken; one of them was joint best.

`llm.complete()` asks for `reasoning: {"enabled": false}`, falls back to
`{"effort": "low"}` where reasoning is mandatory (gpt-5-nano returns *"Reasoning
is mandatory for this endpoint"*), then to the provider default, and retries when
a call returns empty content with a burned budget. **If you swap in your own
summariser model, keep that behaviour** — an otherwise-good model will look
broken without it.

**2. `allow_fallbacks: false` was a mistake.** The region guarantee comes from
`provider.only`, which is a **hard filter** — verified directly: a
Singapore-only model returns `404 No allowed providers` under both
`allow_fallbacks: false` *and* `true`. Disabling fallbacks therefore bought no
extra safety while pinning each request to one endpoint:

```
allow_fallbacks: false   glm-5.3-flash, 3 calls -> 1 success, 2x HTTP 429
allow_fallbacks: true    glm-5.3-flash, 3 calls -> 3 successes, 2-7s
                                                  (GMICloud, Friendli, Novita)
```

With fallbacks enabled, `glm-4.7-flash` went from **failing the eval outright**
(persistent 429s) to completing it, and `glm-5.3-flash` halved from 214 s to
104 s. Fallbacks are now on, and they can still only reroute *within* the
allowlist.

---

## Choosing your own

| if you care most about | use | why |
|---|---|---|
| **balance** (default) | `openai/gpt-5.6-luna` | 15/15, 13 s, Azure EU available |
| **cost** | `inclusionai/ling-3.0-flash` | $0.98/1k — 10× cheaper, loses only the buried `0.618` constant |
| **latency** | `google/gemini-3.1-flash-lite` | 5.7 s, also 15/15, 20% dearer |
| **cost, off the critical path** | `z-ai/glm-5.3-flash` | 15/15 at $7.09/1k, but budget ~100 s per compaction |
| **avoiding it entirely** | turn off tiers 2 and 3 | tiers 0 and 1 are free, deterministic, and do most of the work |

Set it in **Settings → Compaction & summary model**; the same model is used for
context compaction *and* for `summarize_file(mode="llm")`. The field checks
region availability as you type.

A caveat the numbers cannot show: one or two runs per model on **one synthetic
trajectory**. That is enough to rule out a 50× price premium and to rank the
field coarsely; it is not a claim that 15.0 beats 14.0. Re-run against your own
traffic before treating the ordering as settled.

---

## Reproducing

```bash
python -m tests.model_eval                    # full field, ~$0.11
python -m tests.model_eval --quick            # the three suggestions + the cost pick
python -m tests.model_eval --repeat 3 --models "a,b"
```

Results, including every generated summary and each model's region verdict, land
in `tests/model_eval_results.json`.
