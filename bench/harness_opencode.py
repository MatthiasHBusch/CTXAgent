"""OpenCode adapter (pinned opencode-ai 1.19.3 under bench/external/opencode).

One trial = one OpenCode session driven headlessly, one process per turn:

    turn 1:  opencode run --format json -m openrouter/<model> --variant <effort> --auto --title <tag> --dir <workdir> <prompt>
    turn k:  ... -s <sessionID> <prompt>

Verified on 1.19.3 (bench/REDESIGN.md section 4.3):
  * config comes from OPENCODE_CONFIG_CONTENT (inline JSON); all state lives in per-trial XDG dirs;
    HOME/USERPROFILE point into the trial dir as well, so ~/.claude/skills, ~/.agents/skills and
    the user's global opencode config are invisible (`opencode debug agent build` listed them).
  * provider.openrouter.options.baseURL = <proxy>/api/v1: every model request (main model, title
    and compaction calls) goes through the logging proxy as POST /api/v1/chat/completions.
  * `--variant high` sends `reasoning: {"effort": "high"}` (seen on the wire).
  * window: provider.openrouter.models.<id>.limit.context (= the auto-compaction threshold basis)
    with limit.output = --max-out; 1.19.3 requires both keys. --window 0 (native) sets no model
    override (models.dev limits) and caps output via OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX.
  * web off: permission webfetch/websearch = deny removes both tools from the request (wire);
    skill = deny removes `skill` (skills are off in every harness). `--auto` approves everything
    that is not explicitly denied (no permission prompts).
  * `--title` skips the title-generation side call; small_model is still set to --cc-small-model.
  * per-turn stream: text / tool_use / step_finish (tokens + cost per request) parts. Only the
    session export labels compaction (a user message with a `compaction` part, an assistant
    message with mode=compaction), so usage rows and compaction counts come from
    `opencode export <sessionID>` after each turn (local, no model call).
"""
from __future__ import annotations

import json
import shutil
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OC_DIR = ROOT / "bench" / "external" / "opencode"
WEB_DENY = ("webfetch", "websearch")
OFF_DENY = ("skill",)


def oc_exe() -> Path:
    """The pinned OpenCode binary of this platform. On Linux (WSL) the Windows build under
    node_modules must not be picked up through interop: it would run with Windows paths and
    its own config and fail with "Unexpected server error"; the Linux build or the `opencode`
    on PATH (same pinned version) is used instead."""
    if os.name == "nt":
        subs, names = ("opencode-windows-x64", "opencode-windows-x64-baseline"), ("opencode.exe",)
    else:
        subs, names = ("opencode-linux-x64", "opencode-linux-x64-baseline", "opencode-darwin-arm64"), ("opencode",)
    for sub in subs:
        for name in names:
            p = OC_DIR / "node_modules" / sub / "bin" / name
            if p.exists():
                return p
    found = shutil.which("opencode")
    if found and os.name != "nt":
        return Path(found)
    raise FileNotFoundError(f"pinned OpenCode not installed under {OC_DIR} (npm install there)")


def oc_version() -> str:
    out = subprocess.run([str(oc_exe()), "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    return f"opencode {out}"


def oc_config(a, base_url: str) -> dict:
    # a model override needs limit.context AND limit.output (1.19.3 rejects a partial limit), so
    # the native window means: no override; the output cap then comes from
    # OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX (see oc_env). limit.input must be overridden too:
    # OpenCode's overflow check (session/overflow.ts) takes the usable budget from limit.input
    # when models.dev defines one (OpenAI models via OpenRouter: input 922k), and only otherwise
    # from limit.context, so without it the 200k window never triggered a compaction for Sol.
    models = {a.model: {"limit": {"context": a.window, "input": a.window, "output": a.max_out}}} if a.window > 0 else {}
    return {
        "$schema": "https://opencode.ai/config.json",
        "share": "disabled", "autoupdate": False,
        "permission": {"*": "allow", **{t: "deny" for t in WEB_DENY + OFF_DENY}},
        **({"small_model": f"openrouter/{a.cc_small_model}"} if getattr(a, "cc_small_model", None) else {}),
        "mcp": {},
        "provider": {"openrouter": {
            "options": {"baseURL": base_url, "apiKey": "{env:OPENROUTER_API_KEY}"},
            "models": models}},
    }


def oc_env(tdir: Path, cfg: dict, max_out: int) -> dict:
    home = tdir / "opencode_home"
    env = {**os.environ, "OPENCODE_CONFIG_CONTENT": json.dumps(cfg),
           "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": str(max_out),
           "XDG_DATA_HOME": str(home / "data"), "XDG_CONFIG_HOME": str(home / "config"),
           "XDG_CACHE_HOME": str(home / "cache"), "XDG_STATE_HOME": str(home / "state"),
           "HOME": str(home / "home"), "USERPROFILE": str(home / "home"),
           "OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
           "OPENCODE_DISABLE_CLAUDE_CODE": "1", "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1"}
    for d in ("data", "config", "cache", "state", "home"):
        (home / d).mkdir(parents=True, exist_ok=True)
    return env


def tools_in_force(env: dict, workdir: Path) -> list[str]:
    """`opencode debug agent build`: tool map of the build agent minus explicitly denied
    permissions (the map itself is not permission-filtered; the request is, see wire_tools)."""
    try:
        p = subprocess.run([str(oc_exe()), "debug", "agent", "build"], cwd=workdir, env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=120)
        j = json.loads(p.stdout[p.stdout.find("{"):])
    except (subprocess.SubprocessError, ValueError):
        return []
    denied = {r.get("permission") for r in j.get("permission") or []
              if r.get("action") == "deny" and r.get("pattern") == "*"}
    return sorted(t for t, on in (j.get("tools") or {}).items() if on and t not in denied and t != "invalid")


def parse_run_stream(out: str, st: dict) -> str | None:
    """One `opencode run --format json` process (= one turn). Returns the session id."""
    sid, texts_by_msg, last_msg = None, {}, None
    for line in out.splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = sid or j.get("sessionID")
        t, part = j.get("type"), j.get("part") or {}
        if t == "tool_use":
            st["tools"].append(part.get("tool"))
        elif t == "text" and not part.get("synthetic"):
            mid = part.get("messageID")
            texts_by_msg.setdefault(mid, []).append(part.get("text") or "")
            last_msg = mid
        elif t == "step_finish":
            st["finish"] = part.get("reason")
        elif t == "error":
            err = j.get("error") or {}
            st["errors"].append(str((err.get("data") or {}).get("message") or err)[:500])
    if last_msg is not None:
        st["final"] = "".join(texts_by_msg[last_msg]).strip()
    return sid


def export_session(env: dict, workdir: Path, sid: str) -> dict:
    # stdout goes to a file, not a pipe: the Linux (bun) build exits before a pipe is drained and
    # the export was cut at 64 KiB ("Unterminated string" at char 65518); a file is written in full
    out = workdir.parent / "opencode_export.json"
    with open(out, "w", encoding="utf-8") as f:
        subprocess.run([str(oc_exe()), "export", sid], cwd=workdir, env=env, stdout=f, stderr=subprocess.DEVNULL,
                       timeout=300)
    s = out.read_text(encoding="utf-8", errors="replace")
    return json.loads(s[s.find("{"):]) if "{" in s else {}


def usage_rows(messages: list[dict], model: str) -> list[dict]:
    """One trace row per assistant message (= one model request) from an export. OpenCode's
    `output` excludes reasoning; OpenRouter bills both as output, so output = output + reasoning."""
    rows = []
    for m in messages:
        info = m.get("info") or {}
        if info.get("role") != "assistant":
            continue
        tk = info.get("tokens") or {}
        cache = tk.get("cache") or {}
        rows.append({"source": "compaction" if info.get("mode") == "compaction" or info.get("summary") else "agent",
                     "model": info.get("modelID") or model, "provider": info.get("providerID"),
                     "uncached": tk.get("input") or 0, "cache_write": cache.get("write") or 0,
                     "cache_read": cache.get("read") or 0,
                     "output": (tk.get("output") or 0) + (tk.get("reasoning") or 0),
                     "reasoning": tk.get("reasoning"), "reported_cost": info.get("cost")})
    return rows


def n_compactions(messages: list[dict]) -> int:
    return sum(1 for m in messages for p in m.get("parts") or [] if p.get("type") == "compaction")


def run_opencode(a, workdir: Path, prompts: list[str], tdir: Path, proxy=None,
                 pre_hooks: list | None = None) -> list[dict]:
    from bench.run import _new_turn, _pre
    base = f"http://127.0.0.1:{proxy.port}/api/v1" if proxy is not None else "https://openrouter.ai/api/v1"
    cfg = oc_config(a, base)
    env = oc_env(tdir, cfg, a.max_out)
    exe = str(oc_exe())
    tools = tools_in_force(env, workdir)
    common = ["run", "--format", "json", "-m", f"openrouter/{a.model}", "--variant", a.reasoning, "--auto",
              "--title", a.tag, "--dir", str(workdir)]
    (tdir / "harness_flags.json").write_text(json.dumps({
        "harness": "opencode", "version": oc_version(), "argv": [exe, *common, "[-s <sessionID>]", "<prompt>"],
        "config": cfg, "env": {k: env[k] for k in env if k.startswith(("OPENCODE_", "XDG_"))
                               and k != "OPENCODE_CONFIG_CONTENT"},
        "window": a.window if a.window > 0 else "native (models.dev limit.context)",
        "tools_in_force": tools, "web_off": {"permission": {t: "deny" for t in WEB_DENY}},
        "skills_off": {"permission": {"skill": "deny"}, "env": ["OPENCODE_DISABLE_CLAUDE_CODE",
                                                                "OPENCODE_DISABLE_EXTERNAL_SKILLS"]},
    }, indent=1), encoding="utf-8")
    sid, seen_msgs, seen_comp = None, 0, 0
    turns = []
    from bench.run import _billed_cap_hit
    for i, prompt in enumerate(prompts):
        hit = _billed_cap_hit(a, tdir)  # --billed-cap: same check as the vendor adapters
        if hit:
            st = _new_turn(); st["aborted"] = hit; turns.append(st)
            break
        if proxy is not None:
            proxy.turn = i + 1
        st = _new_turn()
        st["tools_in_force"] = tools
        turns.append(st)
        if i > 0 and not sid:
            st["errors"].append("no session id from first turn; cannot continue")
            break
        _pre(pre_hooks, i)  # bench-side setup for this turn (e.g. the ephemeral status file)
        cmd = [exe, *common, *(["-s", sid] if sid else []), prompt]
        try:
            p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=a.timeout, stdin=subprocess.DEVNULL)
            out = p.stdout
            if p.returncode != 0:
                st["errors"].append(f"exit {p.returncode}: {p.stderr[-800:]}")
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            st["aborted"] = f"timeout {a.timeout}s"
        with open(tdir / "opencode_stream.jsonl", "a", encoding="utf-8") as f:
            f.write(out)
        sid = parse_run_stream(out, st) or sid
        if sid:
            try:
                exp = export_session(env, workdir, sid)
            except (subprocess.SubprocessError, ValueError) as exc:
                st["errors"].append(f"export failed: {exc}")
                exp = {}
            msgs = exp.get("messages") or []
            st["trace"] = usage_rows(msgs[seen_msgs:], a.model)
            comp = n_compactions(msgs)
            st["compactions"] = comp - seen_comp
            seen_msgs, seen_comp = len(msgs), comp
            (tdir / "opencode_session.json").write_text(json.dumps(exp, ensure_ascii=False), encoding="utf-8")
        st["session_id"] = sid
    return turns
