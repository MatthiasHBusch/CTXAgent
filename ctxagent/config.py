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

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CTXAGENT_DATA", APP_DIR / ".ctxagent"))
BLOB_DIR = DATA_DIR / "blobs"
DB_PATH = DATA_DIR / "ctxagent.db"

# Overridable for local logging/accounting proxies; inherited by delegated workers.
OPENROUTER_URL = os.environ.get("CTXAGENT_OPENROUTER_URL") or "https://openrouter.ai/api/v1/chat/completions"
BACKEND_BUILD = 'browser-workspace-v1'


def default_tools():
    from .tools import DEFAULT_TOOLS
    return list(DEFAULT_TOOLS)


@dataclass
class CompactionConfig:
    """Tuning knobs for the three-tier compaction engine.

    The two structural invariants:

    * Nothing outside the hot zone is ever rewritten (cache safety).
    * Deep compaction keeps a dynamic tail verbatim (recency safety).
    """

    enabled: bool = True
    # Master switch for incremental document dedup/masking/condensation only.
    document_compaction_enabled: bool = True
    # Target TOTAL post-compaction context / pre-compaction context, not a saving.
    deep_target_ratio: float = 0.10

    # ---- Tier 0/1: the cache-stable split -------------------------------
    # Target recent working tail, not a permission to rewrite frozen history.
    # Linked mode reviews the retiring UNFROZEN prefix before promoting it;
    # deferral/fresh evidence can make the mutable suffix exceed this target.
    # Legacy mode still computes and freezes the split before reductions.
    hot_zone_tokens: int = 50_000
    # Opt-in migration: on a due cycle, review the retiring suffix FIRST,
    # then freeze it once. Legacy chats keep their original policy unless enabled.
    review_before_freeze: bool = False
    # Lossless first-delivery cap; 0 preserves legacy output. Recall is exempt.
    initial_result_max_tokens: int = 0
    # Rewrite cadence counts model tool batches, not individual calls in a batch.
    rewrite_interval_batches: int = 5
    rewrite_max_new_tokens: int = 20_000  # early flush for large observations
    rewrite_pressure_ratio: float = 0.70  # safety override; deep threshold also wins
    # Cost gate for SOFT cycles (batch interval / large new context; never pressure).
    # Any rewrite re-sends everything after the first changed byte uncached. Keep the
    # rewrite only if  freed * horizon >= invalidated * (rewrite_cache_cost_ratio - 1),
    # where horizon = requests so far (Copernican estimate of the remaining ones),
    # clamped to [rewrite_horizon_min, rewrite_horizon_max]. Ratio = cache-miss price /
    # cache-read price (Anthropic write 1.25x / read 0.1x = 12.5..25; OpenAI 1x/0.1x = 10..20).
    # Scale the local token estimate by provider-reported prompt tokens (median ratio of
    # the last requests). The local tokenizer undercounts e.g. Claude by up to ~40%, so
    # uncalibrated pressure lets a "200k" session run past its real window.
    calibrate_tokens: bool = True
    rewrite_cost_gate: bool = True
    rewrite_cache_cost_ratio: float = 20.0
    rewrite_horizon_min: int = 10
    rewrite_horizon_max: int = 200
    # Who decides what an incremental review drops (compaction/curate.py):
    #   'auto'  - the due cycle masks by age rank (legacy behaviour).
    #   'agent' - a due cycle (interval / large new context / soft pressure) only
    #             HINTS; the agent reviews with `curate_context` and names what to
    #             drop. Safety nets stay automatic: pressure >= deep_trigger_ratio,
    #             or no review for `agent_review_overdue_batches` batches.
    rewrite_policy: str = 'auto'
    agent_review_overdue_batches: int = 30

    # Number of most-recent tool results that are never touched at all,
    # regardless of the hot-zone maths. ("keep the last M observations in
    # full" from the observation-masking literature.)
    protected_recent_observations: int = 12
    # ... and every result of the last N tool BATCHES, however many results a
    # batch had. An agent reads a batch's results in the following turn(s);
    # masking them before then forces an immediate recall round trip.
    protected_recent_batches: int = 3

    # Never rewrite the last N user turns.
    protected_recent_user_turns: int = 2

    # ---- Tier 1: deterministic, zero-LLM passes -------------------------
    dedup_enabled: bool = True          # supersede stale reads of the same file
    masking_enabled: bool = True        # age-tiered observation truncation
    offload_enabled: bool = True        # write big bodies to disk, keep a handle

    # A tool result bigger than this is a candidate for offloading.
    offload_min_tokens: int = 1_500
    # Age-tiered truncation budgets (tokens kept) by observation age rank.
    # Index 0 = the (protected_recent+1)-th newest observation, and so on.
    mask_tiers: tuple[int, ...] = (2000, 1500, 1000, 600)
    # Oldest tier keeps this much. Below ~200 a stub carries no usable content
    # (a header line alone often exceeds 60 tokens).
    mask_floor_tokens: int = 300

    # ---- Tier 2: significance-driven selective summarisation ------------
    significance_enabled: bool = True
    # Segments scoring below this inside the hot zone get condensed.
    significance_threshold: float = 0.30
    # Only bother if the pass frees at least this many tokens (hysteresis,
    # cf. OpenCode's 20k minimum gain).
    min_gain_tokens: int = 4_000

    # ---- Tier 3: deep compaction ----------------------------------------
    deep_enabled: bool = True
    # Fraction of the context window at which a deep pass is considered.
    deep_trigger_ratio: float = 0.75
    # Dynamic verbatim tail: at deep_trigger_ratio pressure we keep
    # tail_ratio_max of the history verbatim; at max_pressure we keep
    # tail_ratio_min. => we compact 75%..90% of history, as specified.
    tail_ratio_max: float = 0.25
    tail_ratio_min: float = 0.10
    max_pressure: float = 0.95
    # Never run two deep passes closer together than this many turns.
    deep_cooldown_turns: int = 6
    # A deep pass must free at least this much or it is skipped.
    deep_min_gain_tokens: int = 20_000
    # Re-read live files after a deep pass (Codex-style rehydration).
    rehydrate_files: int = 4
    rehydrate_tokens_per_file: int = 4_000

    # ---- Summariser ------------------------------------------------------
    # Measured, not guessed: tests/model_eval.py and MODELS.md. Compaction is
    # long-input / short-output / high-fidelity work, and a dozen models score
    # identically across a ~50x price range, so the tie-breakers are latency
    # (a deep compaction blocks the user's turn) and region availability.
    # gpt-5.6-luna: 15/15 facts, ~13 s, 7 EU/US endpoints including Azure EU.
    # Cheaper: inclusionai/ling-3.0-flash ($0.98 vs $10.36 per 1000, loses one
    # minor fact). Faster: google/gemini-3.1-flash-lite (~6 s).
    summary_model: str = "openai/gpt-5.6-luna"
    #: Reasoning effort for the summariser. "high" by default: compaction is the
    #: one step whose output everything later depends on, and a lost fact is
    #: unrecoverable once the originals are out of the hot zone.
    summary_reasoning: str = "high"
    summary_pricing_mode: str = "standard"
    summary_max_tokens: int = 10_000
    # Do not crush working memory to 128 tokens merely because fixed pins or
    # the newest tool group already make the requested total target impossible.
    summary_min_tokens: int = 1_024

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mask_tiers"] = list(self.mask_tiers)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CompactionConfig":
        if not d:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        clean = {k: v for k, v in d.items() if k in known}
        if 'deep_target_ratio' in clean:
            value = float(clean['deep_target_ratio'])
            if not 0.01 <= value <= 0.5:
                raise ValueError('deep_target_ratio must be between 0.01 and 0.5')
            clean['deep_target_ratio'] = value
        if "mask_tiers" in clean and clean["mask_tiers"] is not None:
            clean["mask_tiers"] = tuple(int(x) for x in clean["mask_tiers"])
        if 'protected_recent_batches' in clean:
            clean['protected_recent_batches'] = max(0, min(50, int(clean['protected_recent_batches'])))
        if 'rewrite_interval_batches' in clean:
            clean['rewrite_interval_batches'] = max(1, min(100, int(clean['rewrite_interval_batches'])))
        if 'rewrite_max_new_tokens' in clean:
            clean['rewrite_max_new_tokens'] = max(1000, min(1000000, int(clean['rewrite_max_new_tokens'])))
        if 'rewrite_pressure_ratio' in clean:
            value = float(clean['rewrite_pressure_ratio'])
            if not 0.1 <= value <= 0.95:
                raise ValueError('rewrite_pressure_ratio must be between 0.1 and 0.95')
            clean['rewrite_pressure_ratio'] = value
        for key in ('rewrite_cost_gate', 'calibrate_tokens'):
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
        if 'agent_review_overdue_batches' in clean:
            clean['agent_review_overdue_batches'] = max(2, min(500, int(clean['agent_review_overdue_batches'])))
        if 'review_before_freeze' in clean and not isinstance(clean['review_before_freeze'], bool):
            raise ValueError('review_before_freeze must be a boolean')
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
        return cls(**clean)


@dataclass
class SessionConfig:
    """Per-chat settings the GUI controls."""

    model: str = "anthropic/claude-sonnet-5"
    context_window: int = 700_000
    max_output_tokens: int = 32_000
    max_tool_iterations: int = 0  # 0 = unlimited; Stop remains available
    temperature: float = 1.0
    #: "off" | "low" | "medium" | "high" | "default" (let the provider decide).
    reasoning_effort: str = "high"
    system_prompt: str = ""
    workdirs: list[str] = field(default_factory=list)
    enabled_tools: list[str] = field(default_factory=default_tools)
    # "eu_us" pins every request to providers headquartered and hosted in
    # the EU/EEA/UK/CH/US/CA, with fallbacks disabled. "any" lets
    # OpenRouter route freely. See ctxagent/providers.py.
    provider_policy: str = "eu_us"
    pricing_mode: str = "standard"
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    agent_max_parallel: int = 4  # inherited root budget; global process cap is 8
    agent_max_depth: int = 2
    agent_timeout_seconds: int = 3600  # includes queue time; supervisor kills tree

    def effective_window(self) -> int:
        """Usable window after reserving output + compaction headroom.

        Reserve the full requested output: capping at 20k under-reserves when
        max_output_tokens is 32k (or higher). Do not invent a 20k usable window
        for a small model whose input/output reservations already exhaust it.
        """
        reserve_out = max(0, self.max_output_tokens)
        return max(1, self.context_window - reserve_out - 13_000)

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
        return cls(
            model=d.get("model", "anthropic/claude-sonnet-5"),
            context_window=int(d.get("context_window", 700_000)),
            max_output_tokens=int(d.get("max_output_tokens", 32_000)),
            max_tool_iterations=max(0, int(d.get("max_tool_iterations", 0))),
            temperature=float(d.get("temperature", 1.0)),
            reasoning_effort=d.get("reasoning_effort") or "high",
            system_prompt=d.get("system_prompt", "") or "",
            workdirs=list(d.get("workdirs") or []),
            enabled_tools=list(d["enabled_tools"]) if d.get("enabled_tools") is not None else default_tools(),
            provider_policy=d.get("provider_policy") or "eu_us",
            pricing_mode=validate_mode(d.get("pricing_mode", "standard")),
            compaction=CompactionConfig.from_dict(d.get("compaction")),
            **limits,
        )


def load_env() -> None:
    """Minimal .env loader so the key can live next to the app."""
    for candidate in (APP_DIR / ".env", Path.cwd() / ".env"):
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


def api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def anthropic_api_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def anthropic_workspace_id() -> str:
    """Needed for user-scoped keys (`sk-ant-usr-...`), which are not bound to a workspace."""
    return os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()


def key_problem(model: str) -> str | None:
    """Which key `model` needs and is missing, if any. Bare Claude ids use the
    direct Anthropic API, everything else goes through OpenRouter."""
    if model.startswith("claude-"):
        return None if anthropic_api_key() else "ANTHROPIC_API_KEY not configured"
    return None if api_key() else "OPENROUTER_API_KEY not configured"


def json_default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"not serialisable: {type(o)}")


def dumps(o: Any) -> str:
    return json.dumps(o, default=json_default, ensure_ascii=False)
