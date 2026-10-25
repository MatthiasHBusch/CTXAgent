"""Benchmark runner: real harnesses, real models, deterministic or rubric checkers.

  python -m bench.run sweep --models M[,M2] --harnesses ctxagent,claude_code,codex \
      --tasks cb:0,cb:1 | cb-strat:10 | cbseq:10@0 | seqdep:100@0 | lookup,bugfix,audit \
      --windows 200000[,1000000] --reps 3 --out bench/results/x.jsonl [--parallel 3] [--recap] [--no-shell]
  python -m bench.run trial --model M --harness H --task T --tag TAG --out file.jsonl
  python -m bench.run show --task seqdep:20@0        # dry run: print the session script

A *session* is a list of user turns sent to ONE agent session, like a user asking
one question after another. Single-question tasks are 1-turn sessions;
`cbseq:N@S` sends N Context-Bench questions (stratified sample, seed S, shuffled)
sequentially, so the context accumulates and compaction actually happens. With
--recap a final turn asks for all previous answers from memory (retention probe).
`seqdep:N@S` is the dependent session of bench/seqdep.py: filler questions plus
facts, standing rules, derived questions and ephemeral observations whose probes
can only be answered from the conversation (turn kinds and checkers live there).

Harnesses: ctxagent arms (see ARMS), `claude_code` (pinned local install), `codex`.
Context limit (--window) is applied natively: ctxagent `context_window`; Claude
Code `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (100k..200k) or the `[1m]` model suffix
(1M); Codex `model_context_window`. Reasoning effort (--reasoning) is applied to
all three (Claude Code `--effort`). `--no-shell` removes the shell: Claude Code
`--disallowedTools Bash`, ctxagent arms drop run_command and the job tools; Codex
has no other tool, so its cells are skipped with a recorded reason.

Each trial runs in its own process with its own CTXAGENT_DATA / CLAUDE_CONFIG_DIR /
CODEX_HOME, so nothing leaks between trials or into the user's own setup. One JSON
object per trial (session) is appended to --out; raw trajectories stay in
bench/.work/<tag>/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.environ["BENCH_WORK"]) if os.environ.get("BENCH_WORK") else ROOT / "bench" / ".work"
CC_EXE = ROOT / "bench" / "external" / "cc" / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"

BASIC_TOOLS = ["read_file", "write_file", "edit_file", "list_dir", "glob_search", "grep_search", "run_command"]
# Everything that executes a command (the background job tools run shell commands too).
SHELL_TOOLS = ["run_command", "start_job", "job_status", "job_wait", "list_jobs", "read_job_log", "cancel_job"]
# Turn kinds that count towards the session score (fact/rule turns are acknowledgements).
SCORED_KINDS = ("question", "ephemeral", "probe_fact", "probe_derived", "control_derived", "probe_ephemeral", "probe_task")

VANILLA_SYSTEM = """You are a capable software agent working on the user's machine.

Working directories for this chat:
{workdirs}

Use the tools to inspect and change files and to run commands. Read before you edit.
Verify your changes. Be concise."""

# ctxagent arms: system prompt override (None = shipped), tools (None = shipped default),
# compaction on/off, optional CompactionConfig overrides.
ARMS = {
    # The configuration the author actually works with (frozen from a live chat, see
    # bench/snapshot_config.py); only model, context_window and workdirs are set per trial.
    "ctxagent_live": dict(system=None, tools=None, compaction=True, live=True),
    # Phase 5' control: the same frozen config with NOTHING managing the context: engine
    # off (no masking, no review hints, no automatic compaction), the model-facing context
    # tools removed, and the shipped prompt minus its context sections (ctxagent drops
    # them when the engine is off). The context grows to its raw size.
    "ctxagent_live_noctx": dict(system=None, tools=None, compaction=False, live=True,
                                drop_tools=("curate_context", "compact_context", "pin", "recall", "context_report")),
    "ctxagent":dict(system=None, tools=None, compaction=True),
    "ctxagent_nocompact": dict(system=None, tools=None, compaction=False),
    "ctxagent_maskonly": dict(system=None, tools=None, compaction=True, comp={"deep_enabled": False}),
    "ctxagent_basictools": dict(system=None, tools=BASIC_TOOLS, compaction=True),
    "vanilla": dict(system=VANILLA_SYSTEM, tools=BASIC_TOOLS, compaction=False),
    "vanilla_compact": dict(system=VANILLA_SYSTEM, tools=BASIC_TOOLS, compaction=True),
}
# Main grid (REDESIGN.md 4.1): one ctxagent arm, always at 1M. The other arms are appendix
# ablations, available but not part of the planned runs.
MAIN_ARMS = ["ctxagent_live", "claude_code", "codex"]
EXTERNAL = ["claude_code", "codex", "opencode", "letta", "miniswe"]  # the last three: bench/harness_<name>.py
HARNESSES = list(ARMS) + EXTERNAL
CTX_WINDOW = 1_000_000  # ctxagent arms always run at this window; `--windows` does not apply to them
NATIVE = "native"       # a window value: the harness's own default (Claude Code 200k, Codex per model)

# Tool matrix (REDESIGN.md 4.2): web / browsing off everywhere. ctxagent's context tools,
# The document workspace (read_document, document_*, render_pdf, view_image, latex_*) stays
# on: it is part of its context management. agent_* (delegation to child agents) is off since
# 2026-10-24: a child agent is a second model loop whose requests are billed on the wire but
# sit outside the main agent's context and accounting, so sessions that delegated (Sol, Astra)
# were not comparable with the single-agent harnesses; the affected sessions were rerun.
CTX_BANNED_PREFIXES = ("web_", "browser_", "agent_")
CTX_BANNED_TOOLS = ("fetch_url",
                    # not in the review's list, but they query Crossref / scholarly APIs over the network
                    "scholarly_search", "doi_lookup")
CC_DISALLOWED = ["WebFetch", "WebSearch"]
PREAMBLE = ("Work only with the files in the working directory and the tools available to you. Do not "
            "access the internet (no web search, no fetching URLs, no package installation, no git clone "
            "from remote). If something seems to require it, say so and continue without it.")


def ctx_tool_banned(name: str) -> bool:
    return name.startswith(CTX_BANNED_PREFIXES) or name in CTX_BANNED_TOOLS


def parse_window(s) -> int | str:
    """'native' or an int token count."""
    if isinstance(s, int):
        return s
    if str(s).strip().lower() == NATIVE:
        return NATIVE
    return int(s)


def window_slug(w) -> str:
    return "wnative" if w == NATIVE else f"w{int(w) // 1000}k"


def tool_matrix(a) -> dict:
    """What was in force for this trial (recorded on the row)."""
    no_shell = bool(getattr(a, "no_shell", False))
    m = {"web": "off", "shell": "off" if no_shell else "on", "preamble": True, "mcp": "none"}
    if a.harness == "claude_code":
        m["disallowed"] = CC_DISALLOWED + (["Bash"] if no_shell else [])
    elif a.harness == "codex":
        m["web_search"] = "disabled"
    else:
        m["ctx_banned"] = list(CTX_BANNED_PREFIXES) + list(CTX_BANNED_TOOLS) + (SHELL_TOOLS if no_shell else [])
    return m


def comp_overrides(a) -> dict:
    """--comp KEY=VALUE pairs (VALUE parsed as JSON, else kept as string)."""
    out = {}
    for kv in getattr(a, "comp", None) or []:
        k, _, v = kv.partition("=")
        try:
            out[k.strip()] = json.loads(v)
        except ValueError:
            out[k.strip()] = v
    return out

RECAP_PROMPT = ("Final check for this session: without reading files or running any tools, list the final "
                "answer you gave to each previous question in this session, in order, one per line in the "
                "form `Q<number>: <answer>`. If you no longer know an answer, write `Q<number>: unknown`.")


def _dotenv_keys() -> dict:
    """Keys from the repo .env. These OVERRIDE the inherited environment: ctxagent's
    load_env() uses setdefault, so a server started before a key rotation would
    otherwise pass its stale key to every benchmark subprocess."""
    out = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID"):
                    out[k.strip()] = v.strip().strip('"').strip("'")
    return out


# ----------------------------------------------------------------------- tasks
def build_seqdep(a, tdir: Path) -> tuple[Path, list[dict]]:
    """`seqdep:N@S` (bench/seqdep.py)."""
    from bench import contextbench as cb, seqdep
    n, seed = seqdep.parse_spec(a.task)
    data = tdir / "data"
    shutil.copytree(cb.SRC / "files", data, dirs_exist_ok=True)
    session = seqdep.build_session(n, seed, data)
    (tdir / "seqdep_session.txt").write_text(session.describe(), encoding="utf-8")
    turns = []
    for t in session.turns:
        turns.append({"prompt": t.prompt, "kind": t.kind, "ground_truth": t.ground_truth,
                      "meta": dict(t.meta), "pre": (lambda k: lambda: session.before_turn(k))(t.turn),
                      "check": (lambda k: lambda f, h=None: session.check(k, f, h))(t.turn),
                      "needs_history": True,
                      "rules": (lambda k: lambda f: session.rule_adherence(k, f))(t.turn)})
    return data, turns


def build_contextbench(a, tdir: Path) -> tuple[Path, list[dict]]:
    """`cb:ID` (one question) or `cbseq:N@S` (N stratified questions, shuffled, + optional recap)."""
    from bench import contextbench as cb
    if a.task.startswith("cb:"):
        ids = [int(a.task[3:])]
    else:
        n, _, seed = a.task[6:].partition("@")
        ids = cb.stratified(int(n), int(seed or 0))
        random.Random(int(seed or 0)).shuffle(ids)
    turns = []
    for k, qid in enumerate(ids, 1):
        ctx = cb.setup(tdir, qid)
        prompt = ctx["prompt"] if len(ids) == 1 else f"Q{k}: {ctx['prompt']}"
        turns.append({"prompt": prompt, "kind": "question", "check": (lambda c: lambda f: cb.judge(c, f))(ctx),
                      "meta": {k2: v for k2, v in ctx.items() if k2 not in ("prompt", "question")}})
    if getattr(a, "recap", False) and len(ids) > 1:
        ctxs = [cb.setup(tdir, q) for q in ids]
        # check receives the earlier turns as well (see finish_trial): retention, not correctness
        turns.append({"prompt": RECAP_PROMPT, "kind": "recap", "meta": {"qids": ids}, "needs_history": True,
                      "check": lambda f, h=None: recap_check(ctxs, f, [x["final"] for x in (h or [])])})
    return tdir / "data", turns


def build_ledgerkit(a, tdir: Path) -> tuple[Path, list[dict]]:
    """`lookup` / `bugfix` / `audit` on a generated ledgerkit repo (bench/tasks.py)."""
    from bench.fixtures import build_repo
    from bench.tasks import TASKS
    task = TASKS[a.task]
    repo = tdir / "repo"
    truth = build_repo(repo, seed=a.seed)
    pre = task["pre"](repo)
    return repo, [{"prompt": task["prompt"](repo, truth), "kind": "question", "meta": {},
                   "check": lambda f: task["check"](repo, truth, f, pre)}]


# Task-spec prefix -> builder(a, tdir) -> (workdir, turns). Register new families here
# (e.g. "poly:" -> tasks_polyglot.build, "sci:" -> tasks_scicode.build); the longest
# matching prefix wins, a spec without ':' is a ledgerkit task name.
def build_polyglot(a, tdir: Path) -> tuple[Path, list[dict]]:
    """`poly:K@S[:lang]` (bench/tasks_polyglot.py)."""
    from bench import tasks_polyglot
    return tasks_polyglot.build(a, tdir)


def build_polydep(a, tdir: Path) -> tuple[Path, list[dict]]:
    """`polydep:K@S[:lang]` (bench/tasks_polyglot.py + bench/seqdep.py)."""
    from bench import tasks_polyglot
    return tasks_polyglot.build_dep(a, tdir)


def build_scicode(a, tdir: Path) -> tuple[Path, list[dict]]:
    """`sci:ID[+ID..]@S` or `sci:chatN` (bench/tasks_scicode.py)."""
    from bench import tasks_scicode
    return tasks_scicode.build(a, tdir)


def build_mixdep(a, tdir: Path) -> tuple[Path, list[dict]]:
    """`mixdep:K@S[:ids]` (bench/tasks_mix.py: exercises + SciCode dev steps + seqdep edges)."""
    from bench import tasks_mix
    return tasks_mix.build(a, tdir)


def build_marathon(a, tdir: Path) -> tuple[Path, list[dict]]:
    """`marathon:S` (bench/tasks_mix.py: all tracks + all SciCode dev problems in one chat)."""
    from bench import tasks_mix
    return tasks_mix.build_marathon(a, tdir)


def build_hardmix(a, tdir: Path) -> tuple[Path, list[dict]]:
    """`hardmix:K@S` (bench/tasks_hard.py: Go/JS exercises + SciCode TEST problems + recall probes + seqdep edges)."""
    from bench import tasks_hard
    return tasks_hard.build(a, tdir)


def build_heavy(a, tdir: Path) -> tuple[Path, list[dict]]:
    """`heavy:S[:nswe+ntb+nsci]` (bench/tasks_hard.py + bench/tasks_heavy.py: SWE-bench, Terminal-Bench, SciCode-Verified; Docker)."""
    from bench import tasks_hard
    return tasks_hard.build_heavy(a, tdir)


TASK_BUILDERS: dict = {
    "heavy:": build_heavy,
    "hardmix:": build_hardmix,
    "marathon:": build_marathon,
    "mixdep:": build_mixdep,
    "scitest:": build_scicode,
    "sci:": build_scicode,
    "polydep:": build_polydep,
    "poly:": build_polyglot,
    "seqdep:": build_seqdep,
    "cbseq:": build_contextbench,
    "cb:": build_contextbench,
}


def setup_task(a, tdir: Path) -> tuple[Path, list[dict]]:
    """Return (workdir, turns). Each turn: {'prompt', 'kind', 'meta', 'check'} plus optional
    'needs_history' (check(final, history) with history = earlier turns as
    {turn, kind, prompt, final, meta}), 'pre' (callable run before the turn is sent),
    'rules' (final -> rule-adherence dict) and 'ground_truth'."""
    for prefix in sorted(TASK_BUILDERS, key=len, reverse=True):
        if a.task.startswith(prefix):
            return TASK_BUILDERS[prefix](a, tdir)
    if ":" in a.task:
        raise SystemExit(f"unknown task spec {a.task!r}; known prefixes: {sorted(TASK_BUILDERS)}")
    return build_ledgerkit(a, tdir)


def recap_check(ctxs: list[dict], final: str, earlier: list[str]) -> tuple[float, dict]:
    """Retention probe: each recalled answer vs. the agent's OWN earlier final answer to that
    question (agreement, not correctness -- correctness is already scored per turn)."""
    import re
    from bench import contextbench as cb
    lines = {int(m.group(1)): m.group(2).strip() for m in re.finditer(r"Q(\d+)\s*[:.)-]\s*(.+)", final or "")}
    scores = [cb.judge_consistency(ctx["question"], earlier[k - 1] if k - 1 < len(earlier) else "",
                                   lines.get(k, "")) for k, ctx in enumerate(ctxs, 1)]
    return sum(scores) / max(len(scores), 1), {"per_question": scores, "parsed": len(lines),
                                               "unknown": sum(1 for v in lines.values()
                                                              if v.lower().startswith("unknown"))}


# -------------------------------------------------------------------- ctxagent
def _ctx_trace_entry(r: dict) -> dict:
    u = r.get("usage") or {}
    prompt, cr, cw = u.get("prompt_tokens") or 0, u.get("cache_read") or 0, u.get("cache_write") or 0
    return {"source": r.get("source"), "model": r.get("model"), "provider": r.get("provider"),
            "uncached": max(prompt - cr - cw, 0), "cache_write": cw, "cache_read": cr,
            "output": u.get("completion_tokens") or 0, "reasoning": u.get("reasoning"),
            "reported_cost": u.get("cost")}


def _new_turn() -> dict:
    return {"final": "", "finish": None, "errors": [], "tools": [], "compactions": 0, "aborted": None,
            "trace": []}


LIVE_CONFIG = ROOT / "bench" / "configs" / "ctxagent_live.json"


def build_ctx_config(a, workdir: Path):
    """SessionConfig for a ctxagent arm. `ctxagent_live` = frozen live-chat config with
    only model / context_window / workdirs set (CLI --max-out/--max-iters/--reasoning/
    --temperature/--summary-model do NOT apply to it; --comp does, as explicit ablation)."""
    from ctxagent.config import SessionConfig, CompactionConfig
    arm = ARMS[a.harness]
    window = CTX_WINDOW if a.window == NATIVE else int(a.window)
    if arm.get("live"):
        cfg = SessionConfig.from_dict(json.loads(LIVE_CONFIG.read_text(encoding="utf-8"))["config"])
        cfg.model, cfg.context_window, cfg.workdirs = a.model, window, [str(workdir)]
        comp = cfg.compaction
        comp.enabled = bool(arm["compaction"])
    else:
        comp = CompactionConfig(enabled=arm["compaction"])
        if a.summary_model:
            comp.summary_model = a.summary_model
        cfg = SessionConfig(model=a.model, context_window=window, max_output_tokens=a.max_out,
                            max_tool_iterations=a.max_iters, temperature=a.temperature,
                            reasoning_effort=a.reasoning, workdirs=[str(workdir)], compaction=comp)
        if arm["tools"] is not None:
            cfg.enabled_tools = list(arm["tools"])
    # Tool matrix: web / browser / document tools off; shell off under --no-shell. The
    # effective list is recorded in ctx_config.json next to the trial.
    cfg.enabled_tools = [t for t in cfg.enabled_tools if not ctx_tool_banned(t)]
    if arm.get("drop_tools"):
        cfg.enabled_tools = [t for t in cfg.enabled_tools if t not in arm["drop_tools"]]
    if getattr(a, "no_shell", False):
        cfg.enabled_tools = [t for t in cfg.enabled_tools if t not in SHELL_TOOLS]
    # Service tier (ctxagent arms only): `standard` keeps list rates; `prefer_flex` routes to
    # exact /flex endpoint tags where a provider offers them (OpenAI models) and falls back
    # to standard otherwise. Price only, never behaviour; the reported tier is logged per
    # request and `wire_list_cost` in the result row re-prices every run at list rates.
    if getattr(a, "pricing", None):
        cfg.pricing_mode = a.pricing
    for k, v in {**comp_overrides(a), **(arm.get("comp") or {})}.items():
        if not hasattr(comp, k):
            raise SystemExit(f"unknown CompactionConfig field: {k}")
        setattr(comp, k, v)
    return cfg


def _pre(hooks, i: int) -> None:
    """Run the pre-turn hook of turn i (0-based), e.g. plant / remove an ephemeral file."""
    if hooks and i < len(hooks) and hooks[i]:
        hooks[i]()


async def run_ctxagent(a, workdir: Path, prompts: list[str], proxy=None, tdir: Path | None = None,
                       pre_hooks: list | None = None) -> list[dict]:
    if proxy is not None:  # benchmark-side redirect only; inherited by delegated workers too
        os.environ["CTXAGENT_OPENROUTER_URL"] = f"http://127.0.0.1:{proxy.port}/api/v1/chat/completions"
    from ctxagent.config import load_env
    load_env()
    from ctxagent import store, agent as agentmod, llm
    from ctxagent.segments import Conversation
    if proxy is not None:
        llm.OPENROUTER_URL = os.environ["CTXAGENT_OPENROUTER_URL"]

    arm = ARMS[a.harness]
    if arm["system"] is not None:
        agentmod.BASE_SYSTEM = arm["system"]
    cfg = build_ctx_config(a, workdir)
    if tdir is not None:
        eff = cfg.to_dict()
        eff.pop("workdirs", None)
        (tdir / "ctx_config.json").write_text(json.dumps(eff, indent=1, sort_keys=True), encoding="utf-8")
    store.init()
    cid = store.create_chat(a.tag, cfg.to_dict())
    session = agentmod.AgentSession(cid, cfg, Conversation())
    seen: set = set()
    turns = []
    for k_turn, prompt in enumerate(prompts, 1):
        if proxy is not None:
            proxy.turn = k_turn
        _pre(pre_hooks, k_turn - 1)
        st = _new_turn()
        st["chat_id"] = cid
        turns.append(st)

        async def drive():
            async for ev in session.run_turn(prompt):
                k = ev.get("type")
                if k == "final":
                    st["final"], st["finish"] = ev.get("text", ""), ev.get("finish_reason")
                elif k == "error":
                    st["errors"].append(str(ev.get("error"))[:500])
                elif k == "tool_result":
                    st["tools"].append(ev.get("name"))
                elif k == "compaction":
                    st["compactions"] += 1
                elif k == "usage":
                    for r in session.usage.get("recent_requests", []):
                        key = r.get("generation_id") or id(r)
                        if key not in seen:
                            seen.add(key)
                            st["trace"].append(_ctx_trace_entry(r))
                    if (session.usage.get("cost") or 0) > a.max_cost:
                        st["aborted"] = f"cost cap ${a.max_cost} exceeded"
                        return

        try:
            await asyncio.wait_for(drive(), timeout=a.timeout)
        except asyncio.TimeoutError:
            st["aborted"] = f"timeout {a.timeout}s"
        except Exception as exc:  # noqa: BLE001 - record, do not crash the sweep
            st["errors"].append(f"{type(exc).__name__}: {exc}"[:500])
            st["errors"].append(traceback.format_exc()[-1500:])
        # Requests recorded after the last usage event of this turn (e.g. post-turn compaction)
        for r in session.usage.get("recent_requests", []):
            key = r.get("generation_id") or id(r)
            if key not in seen:
                seen.add(key)
                st["trace"].append(_ctx_trace_entry(r))
        try:
            session.persist()
        except Exception:  # noqa: BLE001
            pass
        if st["aborted"] and "cost cap" in st["aborted"]:
            break
    return turns


def _billed_so_far(tdir: Path) -> float:
    """Sum of billed cost in this trial's proxy log (every harness, same code path)."""
    from bench.proxy import read_log, normalize
    try:
        return sum((n.get("billed") or 0) for n in (normalize(r) for r in read_log(tdir / "proxy.jsonl")) if n)
    except FileNotFoundError:
        return 0.0


def _billed_cap_hit(a, tdir: Path | None) -> str | None:
    """`--billed-cap`: abort the session before the next turn once the proxy's billed total
    exceeds the cap. Unlike `--max-cost`, this uses what OpenRouter actually billed: Claude
    Code's `--max-budget-usd` prices OpenRouter model ids with its own table and let a $15
    cap run to $17 in the P2 sweep; Codex has no cap at all."""
    cap = getattr(a, "billed_cap", 0) or 0
    if not cap or tdir is None:
        return None
    b = _billed_so_far(tdir)
    return f"billed cap ${cap} exceeded (billed ${b:.2f})" if b > cap else None


# ----------------------------------------------------------------- claude code
def _cc_model(a) -> str:
    return a.model + "[1m]" if a.window != NATIVE and int(a.window) >= 1_000_000 else a.model


def run_claude_code(a, workdir: Path, prompts: list[str], tdir: Path, proxy=None,
                    pre_hooks: list | None = None) -> list[dict]:
    key = os.environ["OPENROUTER_API_KEY"]
    cfgdir = tdir / "claude_cfg"
    cfgdir.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{proxy.port}/api" if proxy is not None else "https://openrouter.ai/api"
    env = {**os.environ, "CLAUDE_CONFIG_DIR": str(cfgdir), "ANTHROPIC_BASE_URL": base,
           "ANTHROPIC_AUTH_TOKEN": key, "ANTHROPIC_API_KEY": "",
           "ANTHROPIC_DEFAULT_HAIKU_MODEL": a.cc_small_model, "DISABLE_TELEMETRY": "1",
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1"}
    if a.window != NATIVE and int(a.window) < 1_000_000:
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(a.window)  # effective: min(model window, max(100k, v))
    # Pinned local install, native exe: no cmd.exe shim (which would cut the prompt at its
    # first newline), and the user's global Claude Code stays untouched.
    # on Linux (WSL) the pinned Windows exe would run through interop with Windows paths and its
    # own credentials ("OAuth session expired"): use the Linux install on PATH (same pinned version)
    exe = str(CC_EXE) if (os.name == "nt" and CC_EXE.exists()) else (shutil.which("claude") or "claude")
    sid = str(uuid.uuid4())
    cum: dict = {}  # cumulative modelUsage of the session so far (CC restores it on --resume)
    turns = []
    for i, prompt in enumerate(prompts):
        if proxy is not None:
            proxy.turn = i + 1
        _pre(pre_hooks, i)
        st = _new_turn()
        turns.append(st)
        hit = _billed_cap_hit(a, tdir)
        if hit:
            st["aborted"] = hit
            break
        # --strict-mcp-config with no MCP config = no MCP servers; web tools off by name.
        cmd = [exe, "-p", prompt, "--model", _cc_model(a), "--output-format", "stream-json", "--verbose",
               "--dangerously-skip-permissions", "--strict-mcp-config", "--effort", a.reasoning,
               "--max-budget-usd", str(a.max_cost),  # safety cap only (CC's own price table)
               "--disallowedTools", ",".join(CC_DISALLOWED + (["Bash"] if getattr(a, "no_shell", False) else []))]
        cmd += ["--session-id", sid] if i == 0 else ["--resume", sid]
        try:
            p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=a.timeout)
            out = p.stdout
            if p.returncode != 0 and not out.strip():
                st["errors"].append(f"exit {p.returncode}: {p.stderr[-800:]}")
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            st["aborted"] = f"timeout {a.timeout}s"
        with open(tdir / "claude_stream.jsonl", "a", encoding="utf-8") as f:
            f.write(out)
        cum = _parse_cc_stream(a, out, st, cum)
    return turns


def _parse_cc_stream(a, out: str, st: dict, prev_cum: dict | None = None) -> dict:
    """Parse one Claude Code process (= one turn). Per-message usage in the stream is used
    for the INPUT side of each request (context size, cache split); it under-reports output
    tokens (snapshots), and compaction/side calls never appear. The result's `modelUsage`
    is authoritative but CUMULATIVE over the whole session after --resume (verified), so the
    turn's true usage is modelUsage - prev_cum; the gap to the streamed rows is booked as an
    'unstreamed' row. Returns the new cumulative modelUsage."""
    prev_cum = prev_cum or {}
    msgs, model_usage = {}, {}
    for line in out.splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = j.get("type")
        if t == "assistant":
            m = j.get("message") or {}
            mid = m.get("id") or id(m)
            for c in m.get("content") or []:
                if c.get("type") == "tool_use":
                    st["tools"].append(c.get("name"))
            msgs[mid] = (m.get("model") or a.model, m.get("usage") or {})  # last usage per message wins
        elif t == "system" and j.get("subtype") == "compact_boundary":
            st["compactions"] += 1
            st.setdefault("compact_meta", []).append(j.get("compactMetadata") or j.get("compact_metadata"))
        elif t == "result":
            st["final"] = j.get("result") or ""
            st["finish"] = j.get("subtype")
            st["reported_cost"] = j.get("total_cost_usd")  # Claude Code's own price table: NOT used
            st["num_turns"] = j.get("num_turns")
            model_usage = j.get("modelUsage") or {}
    seen_tot: dict = {}
    for model, u in msgs.values():
        row = {"source": "agent", "model": model, "provider": None,
               "uncached": u.get("input_tokens") or 0, "cache_write": u.get("cache_creation_input_tokens") or 0,
               "cache_read": u.get("cache_read_input_tokens") or 0, "output": u.get("output_tokens") or 0,
               "reasoning": None, "reported_cost": None}
        st["trace"].append(row)
        tot = seen_tot.setdefault(model.split("[", 1)[0], [0, 0, 0, 0])
        for i, k in enumerate(("uncached", "cache_write", "cache_read", "output")):
            tot[i] += row[k]
    new_cum = dict(prev_cum)
    for model, mu in model_usage.items():
        tot = [mu.get("inputTokens") or 0, mu.get("cacheCreationInputTokens") or 0,
               mu.get("cacheReadInputTokens") or 0, mu.get("outputTokens") or 0]
        before = prev_cum.get(model, [0, 0, 0, 0])
        turn = [t - b for t, b in zip(tot, before)] if all(t >= b for t, b in zip(tot, before)) else tot
        new_cum[model] = tot
        have = seen_tot.get(model.split("[", 1)[0], [0, 0, 0, 0])
        diff = [max(w - h, 0) for w, h in zip(turn, have)]
        if any(diff):
            st["trace"].append({"source": "unstreamed", "model": model, "provider": None, "uncached": diff[0],
                                "cache_write": diff[1], "cache_read": diff[2], "output": diff[3],
                                "reasoning": None, "reported_cost": None})
    return new_cum


# ----------------------------------------------------------------------- codex
CODEX_CONFIG = """model = "{model}"
model_provider = "openrouter"
model_reasoning_effort = "{reasoning}"
web_search = "disabled"
{window_line}
[model_providers.openrouter]
name = "OpenRouter"
base_url = "{base_url}"
env_key = "OPENROUTER_API_KEY"
wire_api = "responses"
"""


CODEX_KEYS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens",
              "reasoning_output_tokens")


def codex_usage_row(model: str, u: dict, prev: tuple | None) -> tuple[dict, tuple]:
    """One trace row per turn from Codex's turn.completed usage. Codex reports per-turn
    aggregates only (no per-request peak), and after `exec resume` the numbers are the
    CUMULATIVE thread totals (verified: output_tokens 845, 1548, 2034, ... over 11 turns),
    so the per-turn value is the difference to the previous turn."""
    cur = tuple(int(u.get(k) or 0) for k in CODEX_KEYS)
    d = tuple(c - p for c, p in zip(cur, prev)) if prev and all(c >= p for c, p in zip(cur, prev)) else cur
    inp, cr, cw, out, rsn = d
    return ({"source": "agent", "model": model, "provider": None, "uncached": max(inp - cr - cw, 0),
             "cache_write": cw, "cache_read": cr, "output": out, "reasoning": rsn, "reported_cost": None,
             "aggregate": True}, cur)


def run_codex(a, workdir: Path, prompts: list[str], tdir: Path, proxy=None,
              pre_hooks: list | None = None) -> list[dict]:
    home = tdir / "codex_home"
    home.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{proxy.port}/api/v1" if proxy is not None else "https://openrouter.ai/api/v1"
    window_line = "" if a.window == NATIVE else f"model_context_window = {int(a.window)}\n"
    (home / "config.toml").write_text(CODEX_CONFIG.format(model=a.model, reasoning=a.reasoning,
                                                          window_line=window_line, base_url=base), encoding="utf-8")
    env = {**os.environ, "CODEX_HOME": str(home)}
    exe = shutil.which("codex") or "codex"
    thread = None
    prev = None  # Codex reports CUMULATIVE thread usage after `exec resume`: we take deltas
    turns = []
    for i, prompt in enumerate(prompts):
        if proxy is not None:
            proxy.turn = i + 1
        _pre(pre_hooks, i)
        st = _new_turn()
        turns.append(st)
        hit = _billed_cap_hit(a, tdir)
        if hit:
            st["aborted"] = hit
            break
        common = ["--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"]
        if i == 0:
            cmd = [exe, "exec", *common, "-C", str(workdir), "-"]
        elif thread:
            cmd = [exe, "exec", "resume", thread, *common, "-"]
        else:
            st["errors"].append("no thread id from first turn; cannot resume")
            break
        try:
            # Prompt via stdin: codex is an npm .cmd shim on Windows, and cmd.exe truncates
            # command-line arguments at the first newline.
            p = subprocess.run(cmd, cwd=workdir, env=env, input=prompt, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=a.timeout)
            out = p.stdout
            if p.returncode != 0:
                st["errors"].append(f"exit {p.returncode}: {p.stderr[-800:]}")
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            st["aborted"] = f"timeout {a.timeout}s"
        with open(tdir / "codex_stream.jsonl", "a", encoding="utf-8") as f:
            f.write(out)
        for line in out.splitlines():
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = j.get("type")
            item = j.get("item") or {}
            if t == "thread.started":
                thread = thread or j.get("thread_id")
            elif t == "item.completed":
                if item.get("type") == "agent_message":
                    st["final"] = item.get("text") or ""
                elif item.get("type") != "reasoning":
                    st["tools"].append(item.get("type"))
                if "compact" in (item.get("type") or ""):
                    st["compactions"] += 1
            elif t in ("turn.completed", "turn.failed"):
                st["finish"] = t
                row, prev = codex_usage_row(a.model, j.get("usage") or {}, prev)
                st["trace"].append(row)
            elif t == "error":
                st["errors"].append(str(j.get("message"))[:500])
    return turns


def harness_version(h: str) -> str:
    try:
        if h == "claude_code":
            return subprocess.run([str(CC_EXE), "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
        if h == "codex":
            return subprocess.run([shutil.which("codex") or "codex", "--version"], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        if h == "opencode":
            from bench.harness_opencode import oc_version
            return oc_version()
        if h == "letta":
            from bench.harness_letta import letta_version
            return letta_version()
        if h == "miniswe":
            from bench.harness_miniswe import mini_version
            return mini_version()
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "ctxagent"], cwd=ROOT, capture_output=True,
                               text=True, timeout=30).stdout.strip()
        return f"ctxagent@{head}" + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


# ----------------------------------------------------------------------- trial
def _is_main_model(wire_model: str | None, model: str) -> bool:
    base = lambda m: (m or "").lstrip("~").split("[", 1)[0]  # noqa: E731
    return base(wire_model) == base(model)


def _sum(tr, k):
    return sum(t[k] or 0 for t in tr)


def _is_harness_error(st: dict) -> bool:
    """Infrastructure failure, not a task failure: never judged (the rubric would score an
    error message as a 0.5 'refusal'); flagged for re-running instead."""
    final = st["final"] or ""
    no_output = not st["trace"] or all((t["output"] or 0) == 0 for t in st["trace"])
    return final.lstrip().startswith("API Error") or (no_output and bool(st["errors"] or final))


def skip_reason(a) -> str | None:
    """Why this cell cannot run at all (recorded as a row with `skipped`, never scored)."""
    if a.harness == "codex" and getattr(a, "no_shell", False):
        return "no-shell: Codex has no tool besides the shell"
    return None


def write_skipped(a, reason: str) -> int:
    res = {"tag": a.tag, "harness": a.harness, "model": a.model, "task": a.task, "rep": a.rep, "seed": a.seed,
           "window": a.window, "reasoning": a.reasoning, "skipped": reason, "harness_error": False,
           "n_turns": 0, "completed_turns": 0, "score": None, "success": False, "billed_cost": None, "turns": []}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(json.dumps({"tag": a.tag, "skipped": reason}), flush=True)
    return 0


def _fresh_dir(tdir: Path) -> None:
    """Recreate a trial directory. On Windows a previous trial's files can be read-only or
    still open for a moment after its processes died; retry, clear the read-only bit, and
    accept leftovers rather than failing the trial (that cost a Codex session in P2)."""
    import stat
    for attempt in range(5):
        if not tdir.exists():
            break
        def _onerror(fn, path, exc):  # noqa: ARG001
            try:
                os.chmod(path, stat.S_IWRITE)
                fn(path)
            except OSError:
                pass
        shutil.rmtree(tdir, onerror=_onerror)
        if tdir.exists():
            time.sleep(2 * (attempt + 1))
    tdir.mkdir(parents=True, exist_ok=True)


def trial_main(a) -> int:
    os.environ.update(_dotenv_keys())
    try:  # portable Go toolchain for the Go exercises (bench/tasks_polyglot.go_env); harmless otherwise
        from bench import tasks_polyglot
        if tasks_polyglot.track_available("go"):
            os.environ.update(tasks_polyglot.go_env(WORK / a.tag))
    except Exception:  # noqa: BLE001
        pass
    reason = skip_reason(a)
    if reason:
        return write_skipped(a, reason)
    from bench import pricing
    tdir = WORK / a.tag
    _fresh_dir(tdir)
    workdir, turns_spec = setup_task(a, tdir)
    prompts = [t["prompt"] for t in turns_spec]
    prompts[0] = PREAMBLE + "\n\n" + prompts[0]  # identical first-turn preamble for every harness
    hooks = [t.get("pre") for t in turns_spec]
    from bench.proxy import UsageProxy, read_log, normalize
    proxy = None
    if not a.no_proxy:
        proxy = UsageProxy(tdir / "proxy.jsonl", upstream=os.environ.get("BENCH_UPSTREAM", "https://openrouter.ai"),
                           pin=not getattr(a, "no_pin", False))
        proxy.start()
    t0 = time.time()
    try:
        if a.harness == "claude_code":
            turns = run_claude_code(a, workdir, prompts, tdir, proxy, hooks)
        elif a.harness == "codex":
            turns = run_codex(a, workdir, prompts, tdir, proxy, hooks)
        elif a.harness == "opencode":
            from bench.harness_opencode import run_opencode
            turns = run_opencode(a, workdir, prompts, tdir, proxy, hooks)
        elif a.harness == "letta":
            from bench.harness_letta import run_letta
            turns = run_letta(a, workdir, prompts, tdir, proxy, hooks)
        elif a.harness == "miniswe":
            from bench.harness_miniswe import run_miniswe
            turns = run_miniswe(a, workdir, prompts, tdir, proxy, hooks)
        else:
            turns = asyncio.run(run_ctxagent(a, workdir, prompts, proxy, tdir, hooks))
    finally:
        if proxy is not None:
            proxy.stop()
    wall = time.time() - t0
    try:
        return finish_trial(a, tdir, turns_spec, turns, wall)  # scores the turns: heavy checks need the containers
    finally:
        teardown = next((t.get("teardown") for t in turns_spec if t.get("teardown")), None)
        if teardown:  # heavy sessions: remove the task containers (bench/tasks_heavy.py)
            try:
                teardown()
            except Exception as e:  # noqa: BLE001
                print(f"teardown failed: {e!r}", flush=True)


def show_main(a) -> int:
    """Dry run of a task spec: build the session in a scratch directory and print it."""
    import tempfile
    tdir = Path(tempfile.mkdtemp(prefix="bench-show-"))
    a.recap = getattr(a, "recap", False)
    a.seed = getattr(a, "seed", 7)
    workdir, turns = setup_task(a, tdir)
    if (tdir / "seqdep_session.txt").exists():
        print((tdir / "seqdep_session.txt").read_text(encoding="utf-8"))
    else:
        for k, t in enumerate(turns, 1):
            print(f"{k:>4} {t['kind']:18} {str(t.get('ground_truth') or t['meta'].get('ground_truth') or '')[:28]:28} "
                  f"{t['prompt'][:100]!r}")
    print(f"\n{len(turns)} turns, workdir {workdir}")
    shutil.rmtree(tdir, ignore_errors=True)
    return 0


def _wire_list_cost(wire: list[dict], model: str) -> float | None:
    """Sum of per-request wire tokens x list prices (per request model); None if a price is unknown."""
    from bench import pricing
    total = 0.0
    for w in wire:
        try:
            total += pricing.cost(w["model"] or model, w["uncached"], w["cache_write"], w["cache_read"], w["output"])
        except KeyError:
            return None
    return round(total, 6) if wire else None


def finish_trial(a, tdir: Path, turns_spec: list[dict], turns: list[dict], wall: float) -> int:
    """Score the turns and append the result row (shared with bench.salvage_ctxagent)."""
    from bench import pricing
    from bench.proxy import read_log, normalize
    # Per-request rows as seen on the wire (identical method for every harness).
    wire = [n for n in (normalize(r) for r in read_log(tdir / "proxy.jsonl")) if n]
    plog = read_log(tdir / "proxy.jsonl")
    # Upstream failures only; a client hanging up after the full response is benign.
    wire_err = sum(1 for r in plog if r.get("method") != "LOOP" and (
        (r.get("status") or 0) >= 400 or (r.get("error") and not str(r["error"]).startswith("client gone"))))
    wire_client_gone = sum(1 for r in plog if str(r.get("error") or "").startswith("client gone"))
    wire_loop_noise = sum(1 for r in plog if r.get("method") == "LOOP")

    per_turn, trace = [], []
    for k, (spec, st) in enumerate(zip(turns_spec, turns), 1):
        for t in st["trace"]:
            try:
                t["cost"] = pricing.cost(t["model"] or a.model, t["uncached"], t["cache_write"], t["cache_read"],
                                         t["output"])
            except KeyError:
                t["cost"] = None
            t["turn"] = k
        herr = _is_harness_error(st)
        history = [{"turn": j + 1, "kind": s2["kind"], "prompt": s2["prompt"], "final": t2["final"] or "",
                    "meta": s2["meta"]} for j, (s2, t2) in enumerate(zip(turns_spec[:k - 1], turns[:k - 1]))]
        try:
            if herr:
                score, details = 0.0, {"harness_error": (st["final"] or str(st["errors"][:1]))[:300]}
            elif spec.get("needs_history"):
                score, details = spec["check"](st["final"], history)
            else:
                score, details = spec["check"](st["final"])
        except Exception as exc:  # noqa: BLE001
            score, details = 0.0, {"check_error": f"{type(exc).__name__}: {exc}"}
        rules = None
        if spec.get("rules") and not herr:
            try:
                rules = spec["rules"](st["final"]) or None
            except Exception as exc:  # noqa: BLE001
                rules = {"error": f"{type(exc).__name__}: {exc}"}
        agent_tr = [t for t in st["trace"] if t["source"] == "agent"]
        aggregate = any(t.get("aggregate") for t in agent_tr)
        w_k = [w for w in wire if w["turn"] == k]
        w_main = [w for w in w_k if _is_main_model(w["model"], a.model)]
        per_turn.append({
            "turn": k, "kind": spec["kind"], "meta": spec["meta"], "score": score, "harness_error": herr,
            "ground_truth": spec.get("ground_truth"), "rule_adherence": rules,
            "details": details, "finish": st["finish"], "aborted": st["aborted"], "errors": st["errors"][:3],
            "billed": round(sum(w["billed"] or 0 for w in w_k), 6) if w_k else None,
            "wire_requests": len(w_k),
            "wire_peak": max((w["context"] for w in w_main), default=None),
            "wire_end_context": w_main[-1]["context"] if w_main else None,
            "cost": round(sum(t["cost"] or 0 for t in st["trace"]), 6),
            "requests": len(st["trace"]), "n_tools": len(st["tools"]),
            "tool_counts": {n: st["tools"].count(n) for n in sorted(set(st["tools"]))},
            "compactions": st["compactions"], "compact_meta": st.get("compact_meta"),
            # last request's prompt size = context carried at the end of the turn
            "end_context": None if aggregate or not agent_tr else
            agent_tr[-1]["uncached"] + agent_tr[-1]["cache_write"] + agent_tr[-1]["cache_read"],
            "peak_context": None if aggregate else
            max((t["uncached"] + t["cache_write"] + t["cache_read"] for t in agent_tr), default=0),
            "final_tail": (st["final"] or "")[-500:],
            "wire_tools": sorted({t for r in plog if r.get("turn") == k and r.get("method") == "POST"
                                  and _is_main_model(r.get("model"), a.model) for t in r.get("tools") or [] if t}),
        })
        trace += st["trace"]
    q_turns = [t for t in per_turn if t["kind"] in SCORED_KINDS]
    recap = next((t for t in per_turn if t["kind"] == "recap"), None)
    agent_tr = [t for t in trace if t["source"] == "agent"]
    main_wire = [w for w in wire if _is_main_model(w["model"], a.model)]
    hf = tdir / "harness_flags.json"  # external adapters: argv template, config, env switches
    harness_flags = json.loads(hf.read_text(encoding="utf-8")) if hf.exists() else None
    main_reqs = [r for r in plog if r.get("method") == "POST" and _is_main_model(r.get("model"), a.model)]
    first_main = main_wire[0] if main_wire else None
    billed_turn1 = sum(w["billed"] or 0 for w in wire if w["turn"] == 1)
    rule_checks = [v["ok"] for t in per_turn for v in (t["rule_adherence"] or {}).values() if isinstance(v, dict)]
    kinds = sorted({t["kind"] for t in per_turn})
    try:
        from bench import audit
        web_use = audit.scan_trial(tdir, a.harness)
    except Exception as exc:  # noqa: BLE001 - the audit must never lose a trial row
        web_use = [{"where": "audit", "hit": f"audit error: {type(exc).__name__}: {exc}"}]
    wire_peak = max((w["context"] for w in main_wire), default=None)
    res = {
        "tag": a.tag, "harness": a.harness, "model": a.model, "task": a.task, "rep": a.rep, "seed": a.seed,
        "window": a.window, "reasoning": a.reasoning, "max_iters": a.max_iters,
        "no_shell": bool(getattr(a, "no_shell", False)), "provider_pin": not getattr(a, "no_pin", False),
        "tool_matrix": tool_matrix(a), "harness_version": harness_version(a.harness),
        # transcript audit (bench/audit.py): banned tools / network-reaching commands, quoted
        "web_use": web_use or None, "web_use_n": len(web_use),
        # Cache state at session start (REDESIGN.md 5.3): a cached prefix on the very first
        # main-model request means another trial of the same harness warmed the provider cache.
        "warm_start": (first_main["cache_read"] or 0) > 0 if first_main else None,
        "first_cache_read": first_main["cache_read"] if first_main else None,
        "billed_turn1": round(billed_turn1, 6) if wire else None,
        "billed_excl_turn1": round(sum(w["billed"] or 0 for w in wire) - billed_turn1, 6) if wire else None,
        "score_by_kind": {k: sum(t["score"] for t in per_turn if t["kind"] == k) /
                          max(1, sum(1 for t in per_turn if t["kind"] == k)) for k in kinds},
        "n_by_kind": {k: sum(1 for t in per_turn if t["kind"] == k) for k in kinds},
        "rule_adherence": sum(rule_checks) / len(rule_checks) if rule_checks else None,
        "rule_checks": len(rule_checks),
        "comp_overrides": comp_overrides(a) if a.harness in ARMS else None,
        "ctx_config": (json.loads((tdir / "ctx_config.json").read_text(encoding="utf-8"))
                       if (tdir / "ctx_config.json").exists() else None),
        "key_suffix": os.environ.get("OPENROUTER_API_KEY", "")[-4:],
        # how an external harness was run and which tools it offered (per trial and on the wire)
        "harness_flags": harness_flags,
        "tools_in_force": next((st.get("tools_in_force") for st in turns if st.get("tools_in_force")), None),
        "wire_tools": sorted({t for r in main_reqs for t in r.get("tools") or [] if t}),
        "wire_effort": sorted({json.dumps(r["effort"], sort_keys=True) for r in main_reqs if r.get("effort")}),
        # how the harness was run (argv template, config, env switches) and which tools it offered
        "harness_flags": harness_flags,
        "tools_in_force": next((st.get("tools_in_force") for st in turns if st.get("tools_in_force")), None),
        "wire_tools": sorted({t for r in main_reqs for t in r.get("tools") or [] if t}),
        "wire_effort": sorted({json.dumps(r["effort"], sort_keys=True) for r in main_reqs if r.get("effort")}),
        "n_turns": len(per_turn), "completed_turns": len(turns),
        "score": sum(t["score"] for t in q_turns) / max(len(q_turns), 1),
        "success": all(t["score"] >= 1.0 for t in q_turns) and len(turns) == len(turns_spec),
        "harness_error": any(t["harness_error"] for t in per_turn),
        "recap_score": recap["score"] if recap else None,
        "details": q_turns[0]["details"] if len(q_turns) == 1 else None,
        "task_meta": q_turns[0]["meta"] if len(q_turns) == 1 else None,
        "finish": per_turn[-1]["finish"] if per_turn else None,
        "aborted": next((t["aborted"] for t in per_turn if t["aborted"]), None),
        "errors": [e for t in per_turn for e in t["errors"]][:4], "wall_s": round(wall, 1),
        # PRIMARY cost: what OpenRouter billed, per request, captured by the proxy (same for all harnesses)
        "billed_cost": round(sum(w["billed"] or 0 for w in wire), 6) if wire else None,
        "billed_missing": sum(1 for w in wire if w["billed"] is None),
        # the same wire tokens at OpenRouter LIST rates (flex/standard-independent comparison)
        "wire_list_cost": _wire_list_cost(wire, a.model),
        "pricing_mode": getattr(a, "pricing", None) or ("standard" if a.harness in ARMS else None),
        "wire_requests": len(wire), "wire_errors": wire_err, "wire_client_gone": wire_client_gone,
        "wire_loop_noise": wire_loop_noise,
        "wire_peak_context": wire_peak, "peak_over_200k": (wire_peak or 0) > 200_000,
        "wire_tokens": {k: sum(w[k] for w in wire) for k in ("uncached", "cache_write", "cache_read", "output")},
        "wire_by_model": {m: round(sum(w["billed"] or 0 for w in wire if w["model"] == m), 6)
                          for m in sorted({w["model"] or "?" for w in wire})},
        "wire": [{k: w[k] for k in ("turn", "model", "uncached", "cache_write", "cache_read", "output", "billed")}
                 for w in wire],
        # secondary: token counts reported by the harness x list prices (cross-check)
        "cost": round(sum(t["cost"] or 0 for t in trace), 6),
        "cost_by_source": {s: round(sum(t["cost"] or 0 for t in trace if t["source"] == s), 6)
                           for s in sorted({t["source"] for t in trace})},
        # harness-reported totals are cumulative over a resumed session: take the last one
        "reported_cost": max((st.get("reported_cost") or 0) for st in turns) if turns else None,
        "uncached": _sum(trace, "uncached"), "cache_write": _sum(trace, "cache_write"),
        "cache_read": _sum(trace, "cache_read"), "output": _sum(trace, "output"),
        "requests": len(trace), "agent_requests": len(agent_tr),
        "peak_context": None if any(t.get("aggregate") for t in agent_tr) else
        max((t["uncached"] + t["cache_write"] + t["cache_read"] for t in agent_tr), default=0),
        "n_tools": sum(t["n_tools"] for t in per_turn),
        "tool_counts": {n: sum(t["tool_counts"].get(n, 0) for t in per_turn)
                        for n in sorted({n for t in per_turn for n in t["tool_counts"]})},
        "compaction_events": sum(t["compactions"] for t in per_turn),
        "chat_id": turns[0].get("chat_id") if turns else None,
        "turns": per_turn, "trace": trace,
        "final_tail": per_turn[-1]["final_tail"] if per_turn else "",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(json.dumps({k: res[k] for k in ("tag", "score", "score_by_kind", "rule_adherence", "recap_score",
                                          "billed_cost", "billed_excl_turn1", "warm_start", "web_use_n", "cost",
                                          "wire_requests", "wire_errors", "wire_peak_context", "requests",
                                          "peak_context", "n_tools", "compaction_events", "aborted", "wall_s")}),
          flush=True)
    if res["errors"]:
        print("ERRORS:", res["errors"][:2], flush=True)
    return 0


def expand_tasks(spec: str) -> list[str]:
    out = []
    for t in spec.split(","):
        if t.startswith("cb-strat:"):
            from bench import contextbench as cb
            n, _, seed = t[9:].partition("@")
            out += [f"cb:{i}" for i in cb.stratified(int(n), int(seed or 0))]
        else:
            out.append(t)
    return out


def sweep_jobs(a) -> list[dict]:
    """The trial list of a sweep: every (model, window, task, harness, rep) cell, ctxagent
    arms pinned to CTX_WINDOW (duplicates across --windows collapse), order shuffled with
    the sweep seed so that cache warmth and time-of-day do not line up with any factor."""
    jobs, seen = [], set()
    for model in a.models.split(","):
        for window in [parse_window(w) for w in a.windows.split(",")]:
            for task in expand_tasks(a.tasks):
                for h in a.harnesses.split(","):
                    w = CTX_WINDOW if h in ARMS else window
                    for rep in range(a.reps):
                        slug = model.replace("/", "_").replace(":", "_").replace("~", "")
                        tag = (f"{a.prefix}{slug}__{task.replace(':', '-').replace('@', 's')}__{h}"
                               f"__{window_slug(w)}__r{rep}")
                        if tag in seen:
                            continue
                        seen.add(tag)
                        jobs.append({"model": model, "window": w, "task": task, "harness": h, "rep": rep,
                                     "tag": tag, "skip": skip_reason(argparse.Namespace(harness=h, no_shell=a.no_shell))})
    random.Random(a.seed).shuffle(jobs)
    return jobs


def pick_next(pending: list[dict], running: list[dict]) -> dict | None:
    """Next job whose (model, harness) is not running right now; among those, the one whose
    harness has had the fewest starts so far (interleaves harnesses)."""
    busy = {(j["model"], j["harness"]) for j in running}
    starts = {j["harness"]: 0 for j in pending + running}
    for j in running:
        starts[j["harness"]] += 1
    cands = [j for j in pending if (j["model"], j["harness"]) not in busy]
    if not cands:
        return None
    return min(cands, key=lambda j: (starts[j["harness"]], pending.index(j)))


def sweep_main(a) -> int:
    jobs = sweep_jobs(a)
    print(f"{len(jobs)} trials -> {a.out}  (seed {a.seed}, order shuffled, one trial per (model, harness) "
          f"at a time, parallel {a.parallel})", flush=True)
    if a.dry_run:
        for i, j in enumerate(jobs, 1):
            print(f"{i:3} {j['tag']}" + (f"  SKIP: {j['skip']}" if j["skip"] else ""))
        shown = set()
        for j in jobs:
            if j["task"] not in shown and (j["task"].startswith("seqdep:") or j["task"].startswith("cbseq:")):
                shown.add(j["task"])
                print(f"\n--- session script {j['task']} ---")
                show_main(argparse.Namespace(task=j["task"], seed=a.seed + j["rep"], recap=a.recap))
        return 0
    pending, running = list(jobs), []
    common = ["--out", a.out, "--max-out", str(a.max_out), "--max-iters", str(a.max_iters),
              "--reasoning", a.reasoning, "--timeout", str(a.timeout), "--max-cost", str(a.max_cost), "--billed-cap", str(getattr(a, "billed_cap", 0) or 0),
              "--temperature", str(a.temperature), "--cc-small-model", a.cc_small_model]
    if a.summary_model:
        common += ["--summary-model", a.summary_model]
    for kv in a.comp:
        common += ["--comp", kv]
    if getattr(a, "pricing", None):
        common += ["--pricing", a.pricing]
    for flag in ("recap", "no_proxy", "no_shell", "no_pin"):
        if getattr(a, flag):
            common += ["--" + flag.replace("_", "-")]
    while pending or running:
        while pending and len(running) < a.parallel:
            job = pick_next(pending, running)
            if job is None:
                break  # everything left shares a (model, harness) with a running trial
            pending.remove(job)
            model, window, task, h, rep, tag = (job[k] for k in ("model", "window", "task", "harness", "rep", "tag"))
            env = {**os.environ, "CTXAGENT_DATA": str(WORK / tag / "ctxdata"), "PYTHONIOENCODING": "utf-8",
                   "PYTHONDONTWRITEBYTECODE": "1"}
            cmd = [sys.executable, "-B", "-m", "bench.run", "trial", "--model", model, "--harness", h,
                   "--task", task, "--rep", str(rep), "--seed", str(a.seed + rep), "--tag", tag,
                   "--window", str(window), *common]
            # Trial output goes to a FILE: an undrained pipe fills up (4 KB on Windows) and
            # blocks the trial's next write -- which froze whole sessions mid-run.
            (WORK / "_logs").mkdir(parents=True, exist_ok=True)  # not in WORK/tag: trial wipes that
            logf = open(WORK / "_logs" / f"{tag}.log", "w", encoding="utf-8")
            job["proc"], job["log"] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf,
                                                       stderr=subprocess.STDOUT), logf
            running.append(job)
        time.sleep(1)
        for job in list(running):
            p, logf, tag = job["proc"], job["log"], job["tag"]
            if p.poll() is not None:
                logf.close()
                out = (WORK / "_logs" / f"{tag}.log").read_text(encoding="utf-8", errors="replace")
                lines = [l for l in out.splitlines() if l.startswith("{") or l.startswith("ERRORS")]
                print(f"[{tag}] exit={p.returncode} " + (" ".join(lines[-2:]) if lines else out[-800:]), flush=True)
                running.remove(job)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("trial", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--max-out", type=int, default=20_000)
        p.add_argument("--max-iters", type=int, default=100)
        p.add_argument("--reasoning", default="high",
                       help="reasoning effort for ALL harnesses (ctxagent reasoning_effort, Codex "
                            "model_reasoning_effort, Claude Code --effort)")
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--timeout", type=int, default=1500, help="seconds per turn")
        p.add_argument("--billed-cap", type=float, default=0.0,
                       help="abort a session before the next turn once the proxy's billed total exceeds this "
                            "(0 = off); applies to every harness, unlike --max-cost")
        p.add_argument("--max-cost", type=float, default=5.0,
                       help="abort a session above this cost (ctxagent: usage; Claude Code: --max-budget-usd)")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--summary-model", default="")
        p.add_argument("--cc-small-model", default="anthropic/claude-haiku-5.5")
        p.add_argument("--recap", action="store_true", help="append a from-memory recap turn to sessions")
        p.add_argument("--no-proxy", action="store_true", help="talk to OpenRouter directly (no billed-cost log)")
        p.add_argument("--no-shell", action="store_true",
                       help="no shell: Claude Code --disallowedTools Bash, ctxagent without run_command/jobs; "
                            "Codex cells are skipped (recorded)")
        p.add_argument("--no-pin", action="store_true", help="do not pin the OpenRouter provider in the proxy")
        p.add_argument("--comp", action="append", default=[], metavar="KEY=VALUE",
                       help="CompactionConfig override for all ctxagent arms (JSON value), repeatable")
        p.add_argument("--pricing", default=None, choices=["standard", "prefer_flex", "flex_only"],
                       help="ctxagent arms: OpenRouter service tier (default: the arm's config, i.e. standard); "
                            "prefer_flex uses /flex endpoints where offered (OpenAI) at half rates")
    t = sub.choices["trial"]
    t.add_argument("--model", required=True)
    t.add_argument("--harness", required=True, choices=HARNESSES)
    t.add_argument("--task", required=True)
    t.add_argument("--rep", type=int, default=0)
    t.add_argument("--tag", required=True)
    t.add_argument("--window", type=parse_window, default=NATIVE,
                   help="token count or 'native' (ctxagent arms always run at 1M)")
    s = sub.choices["sweep"]
    s.add_argument("--models", required=True)
    s.add_argument("--harnesses", default=",".join(MAIN_ARMS))
    s.add_argument("--tasks", default="cb-strat:10")
    s.add_argument("--windows", default="native,1000000",
                   help="comma list of token counts and/or 'native' (applies to claude_code / codex; "
                        "ctxagent arms are always 1M)")
    s.add_argument("--reps", type=int, default=3)
    s.add_argument("--parallel", type=int, default=3)
    s.add_argument("--prefix", default="")
    s.add_argument("--dry-run", action="store_true",
                   help="print the trial list and the generated session scripts; contact no API")
    sh = sub.add_parser("show", help="dry run of one task spec: print the generated turns")
    sh.add_argument("--task", required=True)
    sh.add_argument("--seed", type=int, default=7)
    sh.add_argument("--recap", action="store_true")
    a = ap.parse_args()
    if a.cmd == "trial":
        return trial_main(a)
    if a.cmd == "show":
        return show_main(a)
    return sweep_main(a)


if __name__ == "__main__":
    raise SystemExit(main())
