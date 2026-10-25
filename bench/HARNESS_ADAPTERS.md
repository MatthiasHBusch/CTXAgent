# 4.3 New harness adapters: OpenCode, Letta Code, mini-swe-agent

Text for `bench/REDESIGN.md` section 4 (the file is untracked on `main` and edited by another
session, so it is kept here to paste in as 4.3). Status 2026-10-16: adapters written and
verified **offline only**. The live smoke runs were not started, at the request to review the plan
before any paid call. The exact commands are at the end.

All three are pinned under `bench/external/` (gitignored, like `bench/external/cc`), not the
user's global installs. Adapters live in `bench/harness_opencode.py`, `bench/harness_letta.py` and
`bench/harness_miniswe.py`, plus the session driver `bench/miniswe_driver.py`. Each exposes
`run_<name>(a, workdir, prompts, tdir, proxy)` with the same per-turn records as `run_codex`. They
are registered in `bench/run.py` (`HARNESSES`, `harness_version()`).

Per trial the row now also carries:

- `harness_flags`: the argv template, config and env switches, and the window.
- `tools_in_force`: the adapter's own view of the tools offered.
- `wire_tools` (per trial and per turn) and `wire_effort`: the tool names and reasoning setting
  of the main-model requests, as they appeared on the wire. The proxy now logs these for every
  harness.

`BENCH_UPSTREAM` lets the proxy forward to `bench/fake_llm.py` for offline tests.

## Install

| harness | pinned version | install |
|---|---|---|
| OpenCode | `opencode-ai` 1.19.3 (native `opencode.exe`) | `cd bench/external/opencode && npm install` (package.json pins 1.19.3) |
| Letta Code | `@letta-ai/letta-code` 0.34.15 (`node letta.js`) | `cd bench/external/letta && npm install` (pins 0.34.15) |
| mini-swe-agent | `mini-swe-agent` 2.4.6 (+ LiteLLM) | `python -m venv bench/external/miniswe/.venv && bench/external/miniswe/.venv/Scripts/python -m pip install mini-swe-agent==2.4.6` |

## Verified flag table

Verification sources: `--help`, the config schema, the bundled code, and requests captured by a
fake endpoint, on the installed versions.

| capability | OpenCode 1.19.3 | Letta Code 0.34.15 | mini-swe-agent 2.4.6 |
|---|---|---|---|
| headless | `opencode run --format json <prompt>` | `letta -p <prompt> --output-format stream-json` | no headless multi-turn CLI (`mini` is a TUI); `bench/miniswe_driver.py` runs the library in one process |
| multi-turn | `-s <sessionID>` (id from the JSON events); verified live: turn 2 recalled a turn-1 codeword | turn 1 `--agent <id> --new`, then `--conversation <id>`. `--agent` together with `--conversation` is rejected. Verified offline: the model sees 1, 2, 3 user messages | the `mini` TUI path "Agent wants to finish, type new task", which appends `The user added a new task: <text>` to the same message list (InteractiveAgent), reproduced verbatim |
| state isolation | per-trial `XDG_*` dirs plus `HOME`/`USERPROFILE` (otherwise `~/.claude/skills` and `~/.agents/skills` are loaded); `OPENCODE_DISABLE_PROJECT_CONFIG`, `OPENCODE_DISABLE_CLAUDE_CODE`, `OPENCODE_DISABLE_EXTERNAL_SKILLS` | per-trial `HOME`/`USERPROFILE`/`LETTA_LOCAL_BACKEND_DIR` on a **short path** (the local backend git-inits memory dirs; under `bench/.work` this fails with "Filename too long"); `letta backend local`, no Letta account | `MSWEA_GLOBAL_CONFIG_DIR` per trial |
| OpenRouter model | `-m openrouter/<id>`; any OpenRouter id | `letta connect openrouter --api-key <key> --base-url <url>`, `-m openrouter/<id>`. **Only ids in the pinned pi-ai catalogue**: Opus 5.5 and GPT-6.1-sol are in it, Haiku 5.5 is not. Uncatalogued ids are refused, since their wire format is unverified | LiteLLM `openrouter/<id>` |
| proxy | `provider.openrouter.options.baseURL = <proxy>/api/v1` (POST `/api/v1/chat/completions`) | `--base-url`: catalogue api `anthropic-messages` appends `/v1/messages` and needs `<proxy>/api`; `openai-completions` appends `/chat/completions` and needs `<proxy>/api/v1`. The adapter picks the base per model | `api_base = <proxy>/api/v1` |
| reasoning effort | `--variant high` sends `reasoning: {"effort": "high"}` | `letta model set <handle> --reasoning high --agent <id>`. OpenAI: `reasoning: {"effort": "high"}`. Anthropic: `thinking`, effort-mapped where the catalogue has a thinkingLevelMap (Opus 5.5), else `budget_tokens: 16384` (Haiku 4.5, seen on the wire) | `model_kwargs.extra_body.reasoning.effort` |
| output cap (`--max-out`) | `limit.output` (with an explicit window) or `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` (native) | `--model-settings '{"max_tokens": N}'` | `model_kwargs.max_tokens` |
| context window / compaction threshold | `provider.openrouter.models.<id>.limit.context` (needs `output` too) is the basis of native auto-compaction. Native = models.dev: Opus 5.5 1M, GPT-6.1-sol 1.05M. `--window 0` = native | `--model-settings '{"context_window_limit": N}'` drives Letta's compaction (`compactBeforeProviderCall`, emits `event_message` compaction). Native = catalogue: Opus 5.5 1M, GPT-6.1-sol 1.05M | **none**: no window, no compaction; full history every step until the provider rejects it. `--window` is not applied |
| 1M | yes (`--window 1000000`; also native for both benchmark models) | yes (same) | n/a |
| machine-readable output | JSON events (`text`, `tool_use`, `step_finish` with tokens and cost). Per-request usage and compaction come from `opencode export <sid>` after each turn; only the export labels compaction | stream-json (`system/init` with tools, `assistant_message`, `tool_call_message`, `usage_statistics`, `event_message` compaction, `result`) | driver JSON lines (final, commands, per-request usage) plus mini's own trajectory file |
| no permission prompts | `--auto` (approves all that is not denied) | `--yolo` (permission mode `unrestricted`) | DefaultAgent executes directly (confirm mode exists only in the TUI agent) |
| **web off** | `permission: {"webfetch": "deny", "websearch": "deny"}`; both tools are absent from the request (wire) | `--base-tools none` drops the default server-side `web_search` / `fetch_webpage`; `--tools <list>` without `WatchPR` (polls GitHub); the client toolset has no web tool. Verified: the request tools equal the `--tools` list | no web tool exists (single `bash` tool) |
| skills / MCP / plugins off | `permission.skill = deny`, `mcp: {}`, the two disable envs above | `--no-skills`, `Skill` removed from `--tools`, `--no-mods` | none exist |
| tools in force (offline wire) | bash, edit, glob, grep, read, task, todowrite, write | Bash, Monitor, Wake, EnterWorktree, ExitWorktree, SetWorkingDirectory, Edit, TaskStop, Read, Memory, Workflow, Agent, SendAgentMessage, TaskCreate, TaskGet, TaskList, TaskUpdate, Write | bash |
| side calls | title generation skipped via `--title <tag>`; `small_model = openrouter/<--cc-small-model>` | Letta memory reflection (native, every 25 steps by default) | none |

Harness-specific notes and deviations, all recorded in `harness_flags`:

- **Letta bootstrap.** Reasoning effort and window are agent settings, and `--new-agent` creates
  the agent only together with a first prompt. The adapter therefore creates the agent against
  `bench/fake_llm.py` (in-process, free) from an empty directory. It then sets model, effort and
  window, reconnects OpenRouter to the proxy, and opens a NEW conversation for turn 1. The
  bootstrap exchange never enters the benchmark conversation.
- **Letta model coverage.** Only models in the pinned catalogue run. Haiku 5.5 is not in it.
- **mini-swe-agent shell.** On Windows `LocalEnvironment` runs commands with `shell=True`, which is
  cmd.exe, although the shipped prompt promises bash and gives heredoc/sed examples. The driver
  runs each command as Git Bash `bash -c` and never uses WSL's `bash.exe`.
- **mini-swe-agent cost.** `cost_tracking=ignore_errors`, because LiteLLM's local price table lacks
  the new models and would abort. The billed cost comes from the proxy as for every harness. As a
  consequence mini's own `cost_limit` only bites if LiteLLM can price the model, while
  `step_limit = --max-iters` always applies. Both limits are per session, as in `mini`.
- **Window sentinel.** For OpenCode and Letta, `--window 0` means native. Do not pass 0 to
  `claude_code` or `codex`.
- **Equal windows.** For the benchmark models, native already means ~1M in OpenCode and Letta. A
  200k condition comparable to Claude Code's native window needs `--window 200000` explicitly.

Nothing was excluded. All three run multi-turn sessions, use OpenRouter with the benchmark's
model ids (Letta: catalogue ids only), route every request through the proxy, and have web access
off by their own flags or config.

## Offline verification (no cost)

`python -m bench.test_adapters parsers` checks the stream parsers on recorded samples in
`bench/test_data/adapters/`:

- OpenCode resume: same session id, finals "OK" and then "pelican".
- OpenCode tool turn: tool names, and the final taken from the last message only.
- OpenCode compaction turn: 2 compactions and 9 requests, with the two compaction requests labelled.
- Letta: 3 turns in one conversation, Anthropic and OpenAI formats.

All pass.

`python -m bench.test_adapters e2e [--window N]` runs a 3-turn session through the real harness
binary, the real `UsageProxy` (upstream = fake server) and `finish_trial`:

| harness | window | turns | finals parsed (continuation) | proxied requests | billed on every request | web tools on wire | effort on wire |
|---|---|---|---|---|---|---|---|
| opencode | native | 3/3 | user_messages = 1, 2, 3 | 6 | yes (0 missing) | none | `{"effort": "high"}` |
| opencode | 1M | 3/3 | 1, 2, 3 | 6 | yes | none | same; limit.context 1000000, max_tokens 32000 |
| letta | native | 3/3 | 1, 2, 3 | 6 | yes | none | `thinking` enabled, budget 16384 (Haiku 4.5) |
| letta | 1M | 3/3 | 1, 2, 3 | 6 | yes | none | same; effective context_window_limit 1000000, max_tokens 32000 |
| miniswe | n/a | 3/3 | 1, 2, 3 | 6 | yes | none (bash only) | `{"effort": "high"}` |

## Live smoke (NOT run; commands for after review)

Each is one 3-turn session; the Context-Bench judge (gpt-5-mini) is the only extra cost.

```
python -m bench.run trial --model anthropic/claude-haiku-5.5 --harness opencode --task cbseq:3@0 --window 0 --tag smoke_opencode --out bench/results/smoke_new_harnesses.jsonl
python -m bench.run trial --model openai/gpt-5-mini --harness letta --task cbseq:3@0 --window 0 --tag smoke_letta --out bench/results/smoke_new_harnesses.jsonl
python -m bench.run trial --model anthropic/claude-haiku-5.5 --harness miniswe --task cbseq:3@0 --tag smoke_miniswe --out bench/results/smoke_new_harnesses.jsonl
```

Pass criteria:

- `completed_turns == 3`.
- Every `turns[k].final_tail` is non-empty.
- `billed_missing == 0` and `wire_errors == 0`.
- `wire_tools` contains no web tool.
- `wire_effort` shows the high setting.

Paid calls made before the stop request: four OpenCode probes on `anthropic/claude-haiku-5.5`
through the proxy. These were a tool turn, a 2-turn resume and a 30k-limit compaction turn, about
$0.03 billed in total. They are the source of the recorded `opencode_*` samples.
