"""Letta Code adapter (pinned @letta-ai/letta-code 0.34.15 under bench/external/letta).

Letta Code keeps an *agent* (memory, settings) with any number of *conversations*. One trial =
one fresh agent in a per-trial local backend (no Letta Cloud account, no Letta server), one
conversation, one `letta -p` process per turn:

    setup:   letta backend local
             letta connect openrouter --api-key <key> --base-url <fake>      (bootstrap only)
             letta --new-agent --base-tools none -m openrouter/<model> ... -p bootstrap   (in an empty dir, against bench.fake_llm: free)
             letta model set openrouter/<model> --reasoning <effort> --model-settings '{"max_tokens":..[,"context_window_limit":W]}' --agent <id>
             letta connect openrouter --api-key <key> --base-url <proxy>
    turn 1:  letta --backend local --yolo --output-format stream-json --no-system-info-reminder --no-mods --no-skills --tools <list> --agent <id> --new -p <prompt>
    turn k:  ... --conversation <conversation_id> -p <prompt>

Why the bootstrap: reasoning effort and context window are agent settings that exist only after
the agent does; `--new-agent` creates it only together with a first prompt. The bootstrap
prompt goes to the in-process fake model server, so it costs nothing and never reaches the
benchmark conversation (turn 1 opens a NEW conversation of that agent).

Verified on 0.34.15 (bench/REDESIGN.md section 4.3):
  * OpenRouter is a first-class local provider (`letta connect openrouter`, pi-ai's catalogue);
    `--base-url` replaces https://openrouter.ai/api. Catalogue models with api
    `anthropic-messages` append /v1/messages to the base, `openai-completions` ones append
    /chat/completions, so the base is <proxy>/api resp. <proxy>/api/v1. Models that are not in
    the pinned catalogue are refused (their wire format is not verified).
  * `model set --reasoning high` -> Anthropic: `thinking` (effort-mapped where the catalogue
    has a thinkingLevelMap, else budget 16384); OpenAI: `reasoning: {"effort": "high"}`.
  * context_window_limit drives Letta's own compaction (compactBeforeProviderCall); the native
    value is the catalogue's contextWindow (Opus 5.5: 1M, GPT-6.1-sol: 1.05M).
  * web off: `--base-tools none` drops the server-side `web_search` / `fetch_webpage` base tools
    (defaults); `--tools` with WatchPR removed (it polls GitHub) is the tool list sent to the
    model (verified on the wire). Skill is removed and `--no-skills` set (skills off everywhere).
  * `--yolo` = permission mode `unrestricted` (all tools auto-approved).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LETTA_JS = ROOT / "bench" / "external" / "letta" / "node_modules" / "@letta-ai" / "letta-code" / "letta.js"
DROP_TOOLS = {"WatchPR", "Skill"}  # GitHub polling (network) and skills; plus any web-named tool below
WEB_RE = re.compile(r"web|fetch|browse|search_web", re.I)


def node() -> str:
    return shutil.which("node") or "node"


def letta_version() -> str:
    out = subprocess.run([node(), str(LETTA_JS), "--version"], capture_output=True, text=True, timeout=120).stdout
    return "letta-code " + out.strip().splitlines()[-1] if out.strip() else "letta-code ?"


@lru_cache(maxsize=None)
def catalogue_api(model: str) -> str | None:
    """Wire API of an OpenRouter model in the pinned bundle's catalogue ('anthropic-messages',
    'openai-completions'), None if the model is not in it."""
    s = LETTA_JS.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(re.escape(f'"chat:{model}"'), s):
        seg = s[m.end():m.end() + 600]
        prov, api = re.search(r'provider: "([^"]+)"', seg), re.search(r'api: "([^"]+)"', seg)
        if prov and prov.group(1) == "openrouter" and api:
            return api.group(1)
    return None


def _rmtree(p: Path) -> None:
    def onerror(func, path, _exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    shutil.rmtree(p, onerror=onerror) if p.exists() else None


def letta_home(tdir: Path) -> Path:
    """Short per-trial path: the local backend git-inits memory dirs whose paths exceed MAX_PATH
    under bench/.work (verified: 'Filename too long')."""
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Temp" / "lb"
    return base / hashlib.sha1(str(tdir).encode()).hexdigest()[:10]


def parse_stream(out: str, st: dict) -> dict:
    """One `letta -p --output-format stream-json` process (= one turn)."""
    info: dict = {}
    texts: list[str] = []
    for line in out.splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        t, mt = j.get("type"), j.get("message_type")
        for k in ("agent_id", "conversation_id"):
            if j.get(k) and not info.get(k):
                info[k] = j[k]
        if t == "system" and j.get("subtype") == "init":
            info["tools"] = j.get("tools") or []
            info["conversation_id"] = j.get("conversation_id") or info.get("conversation_id")
            info["model"] = j.get("model")
        elif mt == "assistant_message":
            c = j.get("content")
            texts.append(c if isinstance(c, str) else "".join(p.get("text") or "" for p in c or []
                                                               if isinstance(p, dict)))
        elif mt == "tool_call_message":
            st["tools"].append((j.get("tool_call") or {}).get("name"))
        elif mt == "usage_statistics":
            st["trace"].append({"source": "agent", "model": None, "provider": None,
                                "uncached": j.get("prompt_tokens") or 0,
                                "cache_write": j.get("cache_write_tokens") or 0,
                                "cache_read": j.get("cached_input_tokens") or 0,
                                "output": j.get("completion_tokens") or 0,
                                "reasoning": j.get("reasoning_tokens"), "reported_cost": None,
                                "context_tokens": j.get("context_tokens")})
        elif mt == "event_message" and j.get("event_type") == "compaction":
            st["compactions"] += 1
            st.setdefault("compact_meta", []).append(j.get("event_data"))
        elif mt == "stop_reason":
            st["finish"] = j.get("stop_reason")
        elif t == "result":
            st["final"] = j.get("result") or ""
            st["finish"] = j.get("subtype") if j.get("subtype") != "success" else (st["finish"] or "success")
            info["result_usage"] = j.get("usage")
        elif t == "error":
            st["errors"].append(str(j.get("message"))[:500])
    if not st["final"] and texts:
        st["final"] = texts[-1]
    return info


def run_letta(a, workdir: Path, prompts: list[str], tdir: Path, proxy=None,
              pre_hooks: list | None = None) -> list[dict]:
    from bench.fake_llm import FakeLLM
    from bench.run import _new_turn
    api = catalogue_api(a.model)
    turns = [_new_turn()]
    if api not in ("anthropic-messages", "openai-completions"):
        turns[0]["errors"].append(f"model {a.model} not in Letta Code's pinned OpenRouter catalogue "
                                  f"(api={api}); wire format unverified, not run")
        return turns
    suffix = "/api" if api == "anthropic-messages" else "/api/v1"
    origin = f"http://127.0.0.1:{proxy.port}" if proxy is not None else "https://openrouter.ai"
    home = letta_home(tdir)
    _rmtree(home)
    (home / "boot").mkdir(parents=True)
    key = os.environ["OPENROUTER_API_KEY"]
    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home), "LETTA_LOCAL_BACKEND_DIR": str(home / "lc"),
           "LETTA_CODE_TELEM": "0", "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.longpaths",
           "GIT_CONFIG_VALUE_0": "true"}
    base_cmd = [node(), str(LETTA_JS)]
    handle = f"openrouter/{a.model}"
    log = open(tdir / "letta_setup.log", "a", encoding="utf-8")

    def letta(args, cwd=workdir, timeout=300, secret=False):
        p = subprocess.run(base_cmd + args, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, stdin=subprocess.DEVNULL)
        shown = ["<key>" if secret and x == key else x for x in args]
        log.write(f"$ letta {' '.join(shown)}\nexit={p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-2000:]}\n\n")
        log.flush()
        return p

    flags = ["--backend", "local", "--yolo", "--output-format", "stream-json", "--no-system-info-reminder",
             "--no-mods", "--no-skills"]
    settings = {"max_tokens": a.max_out}
    if a.window > 0:
        settings["context_window_limit"] = a.window
    st0 = turns[0]
    try:
        letta(["backend", "local"])
        with FakeLLM() as fake:
            letta(["connect", "openrouter", "--api-key", key, "--base-url", fake.url + suffix, "--backend", "local"],
                  secret=True)
            boot = letta([*flags, "--new-agent", "--base-tools", "none", "-m", handle, "-p", "bootstrap"],
                         cwd=home / "boot")
        binfo = parse_stream(boot.stdout, _new_turn())
        aid = binfo.get("agent_id")
        if not aid:
            st0["errors"].append(f"bootstrap created no agent: {boot.stdout[-400:]} {boot.stderr[-400:]}")
            return turns
        ms = letta(["model", "set", handle, "--reasoning", a.reasoning, "--model-settings", json.dumps(settings),
                    "--agent", aid, "--backend", "local"])
        got = letta(["model", "get", "--agent", aid, "--backend", "local"])
        try:
            effective = json.loads(got.stdout)
        except ValueError:
            effective = {"raw": got.stdout[-500:]}
        if ms.returncode != 0:
            st0["errors"].append(f"model set failed: {ms.stderr[-400:]}")
            return turns
        letta(["connect", "openrouter", "--api-key", key, "--base-url", origin + suffix, "--backend", "local"],
              secret=True)
    except subprocess.TimeoutExpired as exc:
        st0["errors"].append(f"setup timeout: {exc}")
        return turns
    finally:
        log.close()
    tools = [t for t in binfo.get("tools") or [] if t not in DROP_TOOLS and not WEB_RE.search(t)]
    tool_arg = ["--tools", ",".join(tools)]
    (tdir / "harness_flags.json").write_text(json.dumps({
        "harness": "letta", "version": letta_version(), "agent_id": aid, "model_api": api,
        "base_url": origin + suffix, "argv": [*base_cmd, *flags, *tool_arg,
                                              "--agent <id> --new | --conversation <id>", "-p", "<prompt>"],
        "model_settings_requested": settings, "reasoning": a.reasoning, "effective_model": effective,
        "window": a.window if a.window > 0 else "native (catalogue contextWindow)",
        "tools_offered_by_default": binfo.get("tools"), "tools_in_force": tools,
        "web_off": {"--base-tools": "none (drops web_search, fetch_webpage)", "dropped": sorted(DROP_TOOLS)},
        "skills_off": ["--no-skills", "Skill removed from --tools"],
        "bootstrap": "agent created against bench.fake_llm (no cost); turn 1 opens a new conversation",
    }, indent=1), encoding="utf-8")
    turns = []
    conv = None
    from bench.run import _pre, _billed_cap_hit
    for i, prompt in enumerate(prompts):
        hit = _billed_cap_hit(a, tdir)  # --billed-cap: same check as the vendor adapters
        if hit:
            st = _new_turn(); st["aborted"] = hit; turns.append(st)
            break
        _pre(pre_hooks, i)
        if proxy is not None:
            proxy.turn = i + 1
        st = _new_turn()
        st["tools_in_force"] = tools
        turns.append(st)
        if i > 0 and not conv:
            st["errors"].append("no conversation id from first turn; cannot continue")
            break
        target = ["--agent", aid, "--new"] if i == 0 else ["--conversation", conv]
        cmd = [*base_cmd, *flags, *tool_arg, *target, "-p", prompt]
        try:
            p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=a.timeout, stdin=subprocess.DEVNULL)
            out = p.stdout
            if p.returncode != 0:
                st["errors"].append(f"exit {p.returncode}: {p.stderr[-800:]}")
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            st["aborted"] = f"timeout {a.timeout}s"
        with open(tdir / "letta_stream.jsonl", "a", encoding="utf-8") as f:
            f.write(out)
        info = parse_stream(out, st)
        for t in st["trace"]:
            t["model"] = a.model
        conv = conv or info.get("conversation_id")
        st["session_id"] = conv
    # keep the transcript (local backend JSONL) with the trial, then drop the short-path home
    try:
        shutil.copytree(home / "lc" / "conversations", tdir / "letta_conversations", dirs_exist_ok=True)
    except OSError:
        pass
    try:
        _rmtree(home)
    except OSError:
        pass
    return turns
