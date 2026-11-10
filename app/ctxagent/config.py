"""Configuration objects for ctxagent.

Everything tunable about the compaction engine lives in CompactionConfig so the
GUI can expose it and so experiments are reproducible from a single dict.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from . import paths

# Source checkout: <repo> and <repo>/.ctxagent. Installed app: the bundle and
# %LOCALAPPDATA%\CTXAgent (see paths.py). CTXAGENT_DATA overrides the data folder.
APP_DIR = paths.app_root()
DATA_DIR = paths.data_dir()
BLOB_DIR = DATA_DIR / "blobs"
DB_PATH = DATA_DIR / "ctxagent.db"

# Overridable for local logging/accounting proxies; inherited by delegated workers.
OPENROUTER_URL = os.environ.get("CTXAGENT_OPENROUTER_URL") or "https://openrouter.ai/api/v1/chat/completions"
BACKEND_BUILD = 'ctxagent-curate-refs-v1'


def default_tools():
    from .tools import DEFAULT_TOOLS
    return list(DEFAULT_TOOLS)


#: The model a new chat starts with (OpenRouter id; with only an Anthropic key the chat
#: gets the direct-API id, provider_counterpart).
DEFAULT_MODEL = "anthropic/claude-opus-5.5"
#: max_output_tokens = 0: no limit of the app's own. Requests ask for the model's maximum
#: output (model_output_limit), as far as the window has room; the prompt limit keeps this
#: much of the window free for the answer (SessionConfig.output_reserve).
UNLIMITED_OUTPUT_RESERVE = 32_000


def _claude_base(model: str) -> str | None:
    """`claude-opus-5-5` for `claude-opus-5-5`, `claude-opus-5-5-20260101` and
    `anthropic/claude-opus-5.5[:variant]`; None for other providers' models."""
    import re
    m = re.match(r"^(?:anthropic/)?(claude-[a-z0-9.-]+?)(?:-\d{8})?(?::.*)?$", model or "")
    return m[1].replace(".", "-") if m else None


#: Largest max_tokens per Claude model family (the API rejects more). Current models: 128K.
_CLAUDE_OUTPUT_LIMITS = {
    "claude-haiku-4-5": 64_000, "claude-sonnet-4-5": 64_000, "claude-opus-4-5": 64_000,
    "claude-opus-4-1": 32_000, "claude-opus-4": 32_000, "claude-sonnet-4": 64_000,
}
_CLAUDE_128K = ("claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5", "claude-haiku-5",
                "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-6")


def model_output_limit(model: str) -> int | None:
    """The most output tokens `model` produces per response, or None when unknown
    (non-Claude models via OpenRouter: the request then leaves max_tokens out)."""
    base = _claude_base(model)
    if base is None:
        return None
    if base.startswith(_CLAUDE_128K):
        return 128_000
    return _CLAUDE_OUTPUT_LIMITS.get(base, 32_000)


def provider_counterpart(model: str) -> str | None:
    """The same Claude model on the other route: `anthropic/claude-opus-5.5` (OpenRouter)
    <-> `claude-opus-5-5` (Anthropic API). None for non-Claude models."""
    import re
    base = _claude_base(model)
    if base is None:
        return None
    if model.startswith("claude-"):
        return "anthropic/" + re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", base)
    return base


#: The model's own context tools. They are not switched one by one: self-steered context
#: management (CompactionConfig.self_steered) brings both, together with the "Managing your
#: own context" section of the system prompt, and takes both away when it is off.
SELF_STEERED_TOOLS = ("compact_context", "curate_context")


@dataclass
class CompactionConfig:
    """Settings of the compaction engine (compaction/engine.py).

    The engine compacts automatically in one way: deep compaction, which replaces the older
    history by a summary once the context reaches deep_trigger_ratio of the window.
    The dataclass defaults are the engine's; a new chat starts self-steered at 0.9
    (SessionConfig.compaction, NEW_CHAT_COMPACTION).
    Self-steered chats (rewrite_policy='agent') also get compact_context and curate_context
    and review hints; deep compaction stays their safety net.

    Deep compaction keeps the newest tool group and the recent user turns verbatim. Nothing
    is frozen: curate_context may replace any item, and the cache points each request sends
    follow the history (cache_policy.plan_rungs).
    """

    enabled: bool = True
    # Target TOTAL post-compaction context / pre-compaction context, not a saving.
    deep_target_ratio: float = 0.10

    # (The frozen prefix - hot_zone_tokens, review_before_freeze, protected_recent_
    # observations/batches - was removed on 2026-11-10; old settings are ignored.)
    # Lossless first-delivery cap for large tool results; 0 = off. Recall is exempt.
    initial_result_max_tokens: int = 0
    # Review hints (self-steered): due every N completed tool batches, or once the context
    # passes this share of the window; never more often than every N batches. (A trigger on
    # new tokens since the last review, rewrite_max_new_tokens, was removed on 2026-11-10:
    # early curates freed little for the same rewrite; old settings are ignored.)
    rewrite_interval_batches: int = 20
    rewrite_pressure_ratio: float = 0.70
    # Scale the local token estimate by provider-reported prompt tokens (median ratio of
    # the last requests). The local tokenizer undercounts e.g. Claude by up to ~40%, so
    # uncalibrated pressure lets a "200k" session run past its real window.
    calibrate_tokens: bool = True
    # What a rewrite costs (compaction/economics.py): the cache-miss / cache-read price
    # ratio when the chat's own prices are unknown, and the bounds of the expected
    # remaining requests (requests so far) behind the break-even figures of compact/curate.
    rewrite_cache_cost_ratio: float = 20.0
    rewrite_horizon_min: int = 10
    rewrite_horizon_max: int = 200
    # 'auto': the engine manages the context alone (deep compaction).
    # 'agent': self-steered (see `self_steered`): the model compacts and curates with
    # compact_context and curate_context, the engine hints and keeps deep compaction as
    # the safety net.
    rewrite_policy: str = 'auto'

    # Deep compaction keeps the last N user turns verbatim.
    protected_recent_user_turns: int = 2

    # ---- Deep compaction --------------------------------------------------
    deep_enabled: bool = True
    # Fraction of the context window at which a deep pass runs.
    deep_trigger_ratio: float = 0.75
    # Never run two deep passes closer together than this many turns.
    deep_cooldown_turns: int = 6
    # A deep pass must free at least this much or it is skipped.
    deep_min_gain_tokens: int = 20_000

    # ---- Summariser --------------------------------------------------------
    # "" = the chat's own model (SessionConfig.effective_summary_model). The
    # conversation is already cached for that model, so a summariser sharing
    # it can reuse the cached prefix instead of paying for the whole input
    # again on a second model. An explicit id (stored by older chats, or set
    # by the user) is honoured as before. For a separate cheap summariser see
    # MODELS.md (tests/model_eval.py): openai/gpt-5.6-luna 15/15 facts, ~13 s,
    # 7 EU/US endpoints; inclusionai/ling-3.0-flash ~10x cheaper.
    summary_model: str = ""
    #: Reasoning effort for the summariser. "high" by default: compaction is the
    #: one step whose output everything later depends on, and a lost fact is
    #: unrecoverable once the originals are out of the hot zone.
    summary_reasoning: str = "high"
    #: Applies to an explicit summary_model only; a summariser that IS the chat
    #: model uses the chat's pricing tier (same endpoint, same cache).
    summary_pricing_mode: str = "standard"
    summary_max_tokens: int = 10_000
    # Do not crush working memory to 128 tokens merely because fixed pins or
    # the newest tool group already make the requested total target impossible.
    summary_min_tokens: int = 1_024
    # When the summariser is the chat model, the deep summary re-sends the cached
    # conversation prefix with the instruction appended (cache reads) instead of a fresh
    # prompt. Falls back to the fresh prompt when the cache has likely expired.
    summary_share_prefix: bool = True
    # Claude models (direct Anthropic API, or OpenRouter): while a tool batch runs longer than
    # the 5-minute cache lifetime (builds, test suites, job_wait, agent_wait), re-send the last
    # request without output (max_tokens 0 direct, 1 on OpenRouter) shortly before the cache
    # expires. It bills a cache read of the prompt; an expired cache would be written again at
    # 25x that price (Opus 5.5). After half as many keep-alives as one rewrite costs, the cache
    # is let go (agent._keep_cache_warm).
    cache_keepalive: bool = True

    # ---- Guidance -----------------------------------------------------------
    # System-prompt variant of the "Managing your own context" section (agent.py), present
    # only while self-steered: 'default' = BASE_SYSTEM's own text; 'live' = the text the
    # benchmark's frozen ctxagent_live arm ran with.
    guidance: str = 'default'

    @property
    def self_steered(self) -> bool:
        """Self-steered context management: the model manages its context with
        compact_context and curate_context, guided by the system prompt; the engine hints
        when a review is due (rewrite_policy='agent'). Deep compaction stays automatic."""
        return bool(self.enabled) and self.rewrite_policy == 'agent'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CompactionConfig":
        if not d:
            return cls()
        # Keys of removed settings (masking, condensing, the guidance switches) are ignored,
        # so chats and profiles saved before still load.
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        clean = {k: v for k, v in d.items() if k in known}
        if 'deep_target_ratio' in clean:
            value = float(clean['deep_target_ratio'])
            if not 0.01 <= value <= 0.5:
                raise ValueError('deep_target_ratio must be between 0.01 and 0.5')
            clean['deep_target_ratio'] = value
        if 'rewrite_interval_batches' in clean:
            clean['rewrite_interval_batches'] = max(1, min(100, int(clean['rewrite_interval_batches'])))
        if 'rewrite_pressure_ratio' in clean:
            value = float(clean['rewrite_pressure_ratio'])
            if not 0.1 <= value <= 0.95:
                raise ValueError('rewrite_pressure_ratio must be between 0.1 and 0.95')
            clean['rewrite_pressure_ratio'] = value
        for key in ('calibrate_tokens', 'summary_share_prefix', 'cache_keepalive'):
            if key in clean and not isinstance(clean[key], bool):
                raise ValueError(f'{key} must be a boolean')
        if 'rewrite_cache_cost_ratio' in clean:
            clean['rewrite_cache_cost_ratio'] = max(1.0, min(100.0, float(clean['rewrite_cache_cost_ratio'])))
        for key in ('rewrite_horizon_min', 'rewrite_horizon_max'):
            if key in clean:
                clean[key] = max(1, min(10000, int(clean[key])))
        if 'rewrite_policy' in clean:
            if clean['rewrite_policy'] not in ('auto', 'agent'):
                raise ValueError("rewrite_policy must be 'auto' or 'agent'")
        if 'initial_result_max_tokens' in clean:
            value = int(clean['initial_result_max_tokens'])
            if value != 0 and not 1000 <= value <= 100000:
                raise ValueError('initial_result_max_tokens must be 0 or between 1000 and 100000')
            clean['initial_result_max_tokens'] = value
        if 'summary_min_tokens' in clean:
            value = int(clean['summary_min_tokens'])
            if not 128 <= value <= 10_000:
                raise ValueError('summary_min_tokens must be between 128 and 10000')
            clean['summary_min_tokens'] = value
        if 'summary_pricing_mode' in clean:
            from .flex import validate_mode
            clean['summary_pricing_mode'] = validate_mode(clean['summary_pricing_mode'])
        if 'summary_model' in clean:
            value = clean['summary_model']
            if value is None:
                value = ''
            if not isinstance(value, str) or len(value) > 256:
                raise ValueError('summary_model must be a model id (or empty for the chat model)')
            clean['summary_model'] = value.strip()
        if clean.get('guidance') in REMOVED_GUIDANCE:
            clean['guidance'] = 'default'
        if 'guidance' in clean and clean['guidance'] not in GUIDANCE:
            raise ValueError(f"guidance must be one of {', '.join(GUIDANCE)}")
        return cls(**clean)


#: System-prompt variants of the context-management section (CompactionConfig.guidance).
GUIDANCE = ('default', 'live')
#: Variants removed with the guidance experiment switches; chats that chose one get 'default'.
REMOVED_GUIDANCE = ('economics', 'hint_only', 'economics_hint')


#: Compaction of a new chat: self-steered (the model compacts and curates its context,
#: guided by the system prompt), deep compaction at 90 % of the window as the safety net.
NEW_CHAT_COMPACTION = {"rewrite_policy": "agent", "deep_trigger_ratio": 0.9}


def new_chat_compaction() -> "CompactionConfig":
    return CompactionConfig.from_dict(dict(NEW_CHAT_COMPACTION))


@dataclass
class SessionConfig:
    """Per-chat settings the GUI controls. Defaults = a new chat's settings."""

    model: str = DEFAULT_MODEL
    context_window: int = 700_000
    #: 0 = unlimited: the model's maximum output, as far as the window has room.
    max_output_tokens: int = 0
    max_tool_iterations: int = 0  # 0 = unlimited; Stop remains available
    temperature: float = 1.0
    #: "off" | "low" | "medium" | "high" | "xhigh" | "default" (let the provider decide).
    #: "max" is passed through as well but not offered in the GUI.
    reasoning_effort: str = "xhigh"
    system_prompt: str = ""
    workdirs: list[str] = field(default_factory=list)
    enabled_tools: list[str] = field(default_factory=default_tools)
    # "eu_us" pins every request to providers headquartered and hosted in
    # the EU/EEA/UK/CH/US/CA, with fallbacks disabled. "any" lets
    # OpenRouter route freely. See ctxagent/providers.py.
    provider_policy: str = "eu_us"
    pricing_mode: str = "standard"
    compaction: CompactionConfig = field(default_factory=new_chat_compaction)
    agent_max_parallel: int = 4  # inherited root budget; global process cap is 8
    agent_max_depth: int = 2
    agent_timeout_seconds: int = 3600  # includes queue time; supervisor kills tree
    #: Settings profile this chat was created from (ctxagent/profiles.py). Only
    #: a label: the chat keeps its own full snapshot, and "modified" is computed
    #: by comparing that snapshot with the profile. "" = no profile.
    profile_id: str = ""

    def effective_summary_model(self) -> str:
        """The model that runs compaction and LLM summaries for this chat.

        An empty CompactionConfig.summary_model means "same as the chat model".
        Every summariser call site goes through here (agent, engine, browser
        summaries, document tools, the server's key check)."""
        return (self.compaction.summary_model or "").strip() or self.model

    def effective_summary_pricing_mode(self, model: str | None = None) -> str:
        """Pricing tier for a summariser request: the chat's own tier when the
        summariser is the chat model ("same as chat", and `model`, if given, is
        the chat model), else the explicit summary tier."""
        same = not (self.compaction.summary_model or "").strip()
        if same and (model is None or model == self.model):
            return self.pricing_mode
        return self.compaction.summary_pricing_mode

    def active_tools(self) -> list[str]:
        """The tools this chat's requests carry: enabled_tools without the context tools,
        which follow the self-steered setting instead. A context tool already in
        enabled_tools keeps its place (stable tool order, so a stable cached prefix)."""
        on = self.compaction.self_steered
        names = [n for n in self.enabled_tools if n not in SELF_STEERED_TOOLS or on]
        if on:
            names += [n for n in SELF_STEERED_TOOLS if n not in names]
        return names

    def effective_window(self) -> int:
        """Usable window after reserving output + compaction headroom.

        Reserve the full requested output: capping at 20k under-reserves when
        max_output_tokens is 32k (or higher). Do not invent a 20k usable window
        for a small model whose input/output reservations already exhaust it.
        """
        return max(1, self.context_window - self.output_reserve() - 13_000)

    def output_reserve(self) -> int:
        """Window kept free for the answer: the output limit, or with unlimited output
        UNLIMITED_OUTPUT_RESERVE (a request still gets what room there is, request_max_tokens)."""
        return self.max_output_tokens if self.max_output_tokens > 0 else UNLIMITED_OUTPUT_RESERVE

    def request_max_tokens(self, prompt_tokens: int) -> int | None:
        """max_tokens for a request whose prompt has about `prompt_tokens` provider tokens.
        A set limit as it is. Unlimited: the model's maximum output, no more than the window
        has room for (the API rejects prompt + max_tokens beyond the context window), at
        least 4096; None when the model's maximum is unknown (OpenRouter decides)."""
        if self.max_output_tokens > 0:
            return self.max_output_tokens
        ceiling = model_output_limit(self.model)
        if ceiling is None:
            return None
        room = self.context_window - int(prompt_tokens) - 2_000
        return max(4_096, min(ceiling, room))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "max_tool_iterations": self.max_tool_iterations,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "system_prompt": self.system_prompt,
            "workdirs": list(self.workdirs),
            "enabled_tools": list(self.enabled_tools),
            "provider_policy": self.provider_policy,
            "pricing_mode": self.pricing_mode,
            "compaction": self.compaction.to_dict(),
            "agent_max_parallel": self.agent_max_parallel,
            "agent_max_depth": self.agent_max_depth,
            "agent_timeout_seconds": self.agent_timeout_seconds,
            "profile_id": self.profile_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "SessionConfig":
        d = d or {}
        from .flex import validate_mode
        limits = {}
        for key, default, hi in (('agent_max_parallel', 4, 8), ('agent_max_depth', 2, 4), ('agent_timeout_seconds', 3600, 604800)):
            value = d.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= hi:
                raise ValueError(f'{key} must be an integer from 1 to {hi}')
            limits[key] = value
        profile_id = d.get("profile_id") or ""
        if not isinstance(profile_id, str) or len(profile_id) > 64:
            raise ValueError("profile_id must be a short string")
        return cls(
            model=d.get("model", DEFAULT_MODEL),
            context_window=int(d.get("context_window", 700_000)),
            max_output_tokens=max(0, int(d.get("max_output_tokens", 0))),
            max_tool_iterations=max(0, int(d.get("max_tool_iterations", 0))),
            temperature=float(d.get("temperature", 1.0)),
            reasoning_effort=d.get("reasoning_effort") or "xhigh",
            system_prompt=d.get("system_prompt", "") or "",
            workdirs=list(d.get("workdirs") or []),
            enabled_tools=list(d["enabled_tools"]) if d.get("enabled_tools") is not None else default_tools(),
            provider_policy=d.get("provider_policy") or "eu_us",
            pricing_mode=validate_mode(d.get("pricing_mode", "standard")),
            compaction=(CompactionConfig.from_dict(d["compaction"]) if d.get("compaction") is not None
                        else new_chat_compaction()),
            profile_id=profile_id,
            **limits,
        )


def load_env() -> None:
    """Minimal .env loader. From source: <repo>/.env, ./.env, then <data>/.env;
    the installed app reads <data>/.env only. Real environment variables win."""
    for candidate in paths.env_files(DATA_DIR):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BLOB_DIR.mkdir(parents=True, exist_ok=True)


# Provider keys: OS credential store first, then the environment / .env
# (ctxagent/keys.py). Never cache these in module globals or copy them into
# configs; read them where the request is built.

def api_key() -> str:
    from . import keys
    return keys.get("OPENROUTER_API_KEY")


def anthropic_api_key() -> str:
    from . import keys
    return keys.get("ANTHROPIC_API_KEY")


def anthropic_workspace_id() -> str:
    """Needed for user-scoped keys (`sk-ant-usr-...`), which are not bound to a workspace."""
    from . import keys
    return keys.get("ANTHROPIC_WORKSPACE_ID")


def key_problem(model: str) -> str | None:
    """Which key `model` needs and is missing, if any. Bare Claude ids use the
    direct Anthropic API, everything else goes through OpenRouter."""
    if model.startswith("claude-"):
        return None if anthropic_api_key() else "ANTHROPIC_API_KEY not configured (add it in Settings > API keys)"
    return None if api_key() else "OPENROUTER_API_KEY not configured (add it in Settings > API keys)"


def json_default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"not serialisable: {type(o)}")


def dumps(o: Any) -> str:
    return json.dumps(o, default=json_default, ensure_ascii=False)
