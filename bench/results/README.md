# bench/results

One JSON object per session (see `bench.run.finish_trial`): per-turn scores, every request's
tokens by class and billed cost as seen on the wire, compaction events detected on the wire,
tool counts and wall time. Later rows with the same `tag` supersede earlier ones. Session
families and arms are described in the paper.

| file | sessions |
|---|---|
| `p2_mixdep.jsonl` | mixed sessions `mixdep:34@{0,1,2}`: Claude Code (Opus 5.5) and Codex (Sol 6.1) at their native window and at 1M |
| `p2_mixdep_ctx_v2.jsonl` | mixed sessions: ctxagent (`ctxagent_live`, 1M) with Opus 5.5 and Sol 6.1 |
| `p3a_baselines200k.jsonl` | mixed sessions: Claude Code (Opus) and Codex (Sol) at an explicit 200k window |
| `p3_sci.jsonl` | clean chains `sci:chat0..4`, two reps: ctxagent at 1M, Claude Code and Codex at 200k, both models |
| `p4_marathon.jsonl` | marathon `marathon:0` (212 turns): ctxagent at 1M, Claude Code and Codex at 200k, both models |
| `p5_models.jsonl` | seven models on the mixed sessions, arm A (`ctxagent_live`) and arm B (`ctxagent_live_noctx`) |
| `p7_opencode.jsonl` | OpenCode 1.19.3 at 200k on the mixed sessions and the marathon, Opus 5.5 and Sol 6.1 |
| `p8_heavy.jsonl` | hard sessions `heavy:{0,1,2}` (SWE-bench Pro, Terminal-Bench 4.0, SciCode-Verified): five models, arm A (`ctxagent_live`) and arm B (`ctxagent_live_noctx`) |
| `p8_harness.jsonl` | hard sessions: Claude Code and OpenCode (Opus 5.5), Codex and OpenCode (Sol 6.1) at an explicit 200k window |
| `prices.json` | OpenRouter list-price snapshot used by `bench/pricing.py` |
