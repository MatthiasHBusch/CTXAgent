# Settings: API keys, profiles, summariser default

The gear button in the top bar opens the app-level Settings menu (API keys,
Profiles, New chats, Appearance, Insights, About). The pill next to the chat title is the
chat's profile slot. Per-chat Settings / Engine / Tools panels still work as before.

**Insights** (off by default) records how every chat uses its context window and its
tools and builds a report from it; see [INSIGHTS.md](INSIGHTS.md).

## Engine: self-steered context management

One switch at the top of a chat's Engine tab. On, the model manages its own
context: it gets `compact_context` and `curate_context`, the system prompt gets
a short "Managing your own context" section, and the engine appends a one-line
review hint when a review is due. Deep compaction stays automatic. Off, the
engine manages the context with deep compaction alone and the model has neither
tool. The two tools are therefore not in the Tools list. The details in the
self-steered group (hint cadence, prompt variant) apply only while it is on.

The engine compacts automatically in one way only, deep compaction (README, "The
compaction engine"). The incremental tiers (dedup, masking, offload,
condensation) and the guidance experiment switches (cost verdicts, curate and
compact gates, hint policy, absolute deep trigger, the 'economics' and
'hint_only' prompt variants) were removed on 2026-11-05. Their keys in older
chats and profiles are ignored; a chat that used a removed prompt variant gets
the standard text.

### Review hints and curate

A review hint comes every **20** completed tool batches (Engine, "Review hint every N
completed tool batches"), earlier only when the context passes the pressure share of the
window (default 70 %), and never more often than the interval. The hint is one line appended
to the newest tool result, `[harness: context review due; cache breakpoints after item 6 =
grep_search result (a.py:1 foo) and after item 10 = run_command result (BUILD LOG)]`. It names
the item before each cache breakpoint the next request sends in the history (at most two;
`cache_policy.plan_rungs`) by its curate reference and a few words: replacing an item re-writes
everything behind the nearest breakpoint before it, so the items behind one breakpoint are
best replaced in one call. Without a breakpoint in the history it says so; for a model
without explicit cache breakpoints it is `[harness: context review due]`. What a review is and how
`curate_context` works is in the system prompt ("Managing your own context"). The hint on new
tokens since the last review ("Earlier hint: new tokens") was removed on 2026-11-10; the
setting in older chats is ignored. A curate or compact that changes something restarts the
count.

`curate_context(reference1, replacement1, reference2, replacement2, ...)` replaces items by the
model's own text. A reference counts back from the newest item before the call: 1 is the
newest tool result, tool call or answer, 2 the one before, and so on. An answer is an
assistant message without tool calls; text written alongside tool calls, user messages,
summaries and harness messages are not counted (in a live check, Opus 5.5 counted that way in
4 of 4 replies). A replaced result or answer becomes
`[replaced: <text> | recall(segment_id="...") restores the original]`; a replaced tool call
keeps its name and gets the arguments `{"replaced": <text>, "recall_segment_id": ...}`. The
result names what each reference hit, so a miscount shows. A call without pairs changes
nothing. There is no listing and no frozen part any more: any item can be replaced.

Nothing is frozen since 2026-11-10 (the "Recent working tail", "Protected observations" and
"Review before freezing" settings are gone; their keys in older chats are ignored). Which
cache points a request sends is decided per request (`cache_policy.plan_rungs`, see
README.md, "Cache points"), so a replacement re-writes the cache from the newest cache point
before the oldest replaced item: recent items are cheap to replace, old ones cost about as
much as the context after them. `recall` returns the original as a new result and no longer
restores it in place, which would re-write the cache from there.

### Cache keep-alive

Engine, "Keep the prompt cache warm while tools run" (on by default; Claude models, direct or
through OpenRouter). Anthropic's prompt cache lives 5 minutes from the last request that read or
wrote it. While a tool batch runs longer (a build, a test suite, `job_wait`, `agent_wait`), the
app re-sends the last request every 4.5 minutes without output: `max_tokens: 0` on the direct
API, `max_tokens: 1` (non-streamed, same payload and `session_id`, so sticky routing reaches the
provider that holds the cache) on OpenRouter, which has no `max_tokens: 0`. The provider reads
the cached prompt and restarts the timer. A keep-alive costs a cache read of the prompt ($0.20
per million tokens on Opus 5.5: $0.04 at 200k, about $0.53 per hour of waiting); an expired cache
costs a full write ($5 per million: $1.00 at 200k).

After half as many keep-alives as one rewrite costs (half the write/read price ratio: 12 on Opus
5.5, so the cache stays warm for about 58 minutes) the cache is let go, so a wait of any length
costs at most 1.5 rewrites. A keep-alive that wrote more than 20 % of its prompt did not hit the
cache: none more in that batch, and none in the chat after two such misses. If the API refuses a
keep-alive, none more on that route for the rest of the session. Models without a write premium
(OpenAI on OpenRouter) get none. Keep-alives appear in the usage panel ("Cache keep-alives") and
in the insights report, apart from the model requests. A 1-hour cache TTL was considered instead:
on a long game-project session it would have cost $15.72 more (every write at 2x input instead of 1.25x),
keep-alives $21.49 less.

## API keys

| Name | Used for |
|---|---|
| `OPENROUTER_API_KEY` | every model id with a provider prefix (`anthropic/claude-opus-5.5`, `openai/...`) |
| `ANTHROPIC_API_KEY` | bare Claude ids (`claude-opus-5-5`), sent to the Anthropic API directly |
| `ANTHROPIC_WORKSPACE_ID` | only for user-scoped keys (`sk-ant-usr-...`) |

Other names (upper case, digits, `_`) can be stored for future integrations.

**Claude subscriptions.** A Pro, Max or Ultra subscription is not an API key.
Max, Ultra and Team plans include monthly API credits, claimed by linking a
Claude Console organization on claude.ai (Settings > Billing). A normal Console
API key from that organization (`ANTHROPIC_API_KEY`, bare model ids such as
`claude-opus-5-5`) then draws those credits. The OAuth token from
`claude setup-token` belongs to Claude Code: Anthropic does not allow other
apps to offer claude.ai logins, so use a Console API key here instead.

**Where they are stored.** In the OS credential store through `keyring`
(Windows Credential Manager, macOS Keychain, Secret Service), service name
`CTXAgent`, one entry per key name. If no keyring backend works, the fallback is
`<data folder>/credentials.json`, created `0600` in a `0700` folder on POSIX; on
Windows it inherits the per-user ACL of `%LOCALAPPDATA%\CTXAgent`. Only the key
*names* are listed in `<data folder>/keys.json`, because a credential store
cannot be enumerated.

**Lookup order.** Credential store first, then the environment (which
`load_env()` fills from a `.env` next to the app). So `.env` keeps working for
development and the benchmarks, and a key saved in Settings wins over it.
Removing the stored key falls back to the environment value again.

**Guarantees.** Keys are never written into chat configs, profiles,
`settings.json` or exports; never logged; returned by the API only masked
(`sk-or-…a1b2`); never placed in a URL; and not copied into `os.environ`, so
shell commands the agent runs do not inherit them. The settings API only
answers requests addressed to a loopback host from a loopback origin (no DNS
rebinding, no cross-site use).

**Test.** "Test" uses a free metadata request, nothing is generated or billed:
OpenRouter `GET /api/v1/key`, Anthropic `GET /v1/models` (with the workspace
header when set). A typed key can be tested before saving it.

**Delegated agents** run as separate processes and resolve keys the same way
(same OS user, inherited data folder and environment); no key travels on a
command line.

`CTXAGENT_KEYSTORE` = `auto` (default: keyring, else file) | `keyring` | `file` |
`env` (environment only, read-only). The offline test suite sets `env`, so it
never touches the real credential store.

## Profiles

A profile is a named set of chat settings: model, reasoning effort, max output,
context window, temperature, provider policy, pricing mode, enabled tools,
system-prompt addition, the whole compaction config and the delegation limits.
It is stored as a *partial* settings dict; anything it does not set resolves to
the current defaults, so settings added to the app later are carried
automatically. Working directories are per chat and never part of a profile.

- **Built-ins** (read-only, duplicate to edit). Not every model manages its
  context reliably, so the starters come in two groups. Every starter enables
  every tool in the Tools list and uses the chat model as summariser.

  | starter | model id | window | context management |
  |---|---|---|---|
  | Fable 5.1 | `anthropic/claude-fable-5.1` | 1M | self-steered |
  | Opus 5.5 | `anthropic/claude-opus-5.5` | 1M | self-steered |
  | Sol 6.1 | `openai/gpt-6.1-sol` | 1M | self-steered |
  | Astra 6 | `openai/gpt-6-astra` | 1M | self-steered |
  | Sonnet 5.5 | `anthropic/claude-sonnet-5.5` | 1M | self-steered |
  | Terra 6.1 | `openai/gpt-6.1-terra` | 1M | self-steered |
  | Gemini 4 Argon | `google/gemini-4-argon` | 400k | engine |

  The self-steered starters use the compaction settings of the benchmark's
  live configuration (without its frozen prefix, removed on 2026-11-10): review
  hints at most every 25 tool batches, the deep safety net at 90 % of the window. Sonnet and Terra joined them on
  2026-11-05 after the self-steered comparison of the benchmark; Gemini 4 Argon never compacted there and stays on the engine defaults. They use OpenRouter ids; for the direct Anthropic API duplicate one
  and use the bare id (`claude-opus-5-5`). A chat created from a starter that no
  longer exists, such as the former Haiku 5.5 starter, keeps its settings and
  shows the profile as deleted.
- **A new chat** is created from a profile (profile menu, "New chat", or the
  default profile) or, as before, from the open chat's settings. Settings >
  New chats chooses which one the New chat button does, the default profile and
  a default working directory.
- **The chat stores a full snapshot** of its settings plus `profile_id`. Editing
  the chat's settings is always allowed; the slot then shows
  `<profile> (modified)` and lists the differing keys.
- **Slot actions:** reset chat to profile, save chat settings into the profile
  (user profiles only), save as new profile (links the chat), apply another
  profile, new chat from this profile. Reset and apply keep history, usage and
  working directories, and refuse while a turn runs.
- **Changing or deleting a profile never rewrites existing chats.** They keep
  their snapshot; they may then show "(modified)" or "(deleted)".
- **Import / export** as JSON (`format: ctxagent-profiles`): settings only, no
  ids, no keys. Imports are validated completely before anything is written.

Files in the data folder: `profiles.json` (user profiles), `settings.json`
(default profile, default working directory, New-chat behaviour, theme, zoom).

## Summariser default: the chat model

`compaction.summary_model` now defaults to `""`, meaning *same as the chat
model*. The conversation is already in the provider's cache for that model, so a
summariser on the same model can reuse the cached prefix instead of paying for
the whole input again on a second model. While the summariser is the chat model
it also uses the chat's pricing tier (same endpoint, same cache);
`summary_pricing_mode` applies to an explicit summariser only.

`SessionConfig.effective_summary_model()` and
`effective_summary_pricing_mode()` are the single resolvers; compaction, LLM
document summaries, browser summaries and the server's key check all go through
them. Chats that stored an explicit summary model (every chat created before
this change) keep it. To use a separate cheap summariser, pick "Another model"
in a profile or type an id into the chat's summary-model field; see MODELS.md
for measured candidates.

## Endpoints (`ctxagent/settings_api.py`)

```
GET    /api/settings                         product, version, data/logs folders, key store, prefs
PATCH  /api/settings/prefs                   {prefs: {...}}
POST   /api/settings/open-folder             {which: "data"|"logs"|"insights"}
GET    /api/settings/insights                enabled, folder, chats recorded, bytes, report path
POST   /api/settings/insights/report         {open?}: writes report.html / report.md, opens the HTML
GET    /api/settings/keys                    masked list + store info
PUT    /api/settings/keys/{NAME}             {value}
DELETE /api/settings/keys/{NAME}
POST   /api/settings/keys/{NAME}/test        {value?}
GET    /api/settings/profiles                POST {name, description?, config?|from_chat?}
GET|PATCH|DELETE /api/settings/profiles/{id}
POST   /api/settings/profiles/{id}/duplicate {name?}
POST   /api/settings/profiles/export         {ids?}      POST /api/settings/profiles/import {data}
GET    /api/settings/chats/{cid}/profile     profile, modified, differences, missing
POST   /api/settings/chats/{cid}/profile/reset    {profile_id?}
POST   /api/settings/chats/{cid}/profile/save
POST   /api/settings/chats/{cid}/profile/save-as  {name}
POST   /api/chats                            now also accepts {profile_id}
```

Tests: `python -B -m tests.test_settings` (offline, fake keyring, mocked HTTP)
and `python -B -m tests.settings_ui_smoke` (Edge, temporary server and data).
