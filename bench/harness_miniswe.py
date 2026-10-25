"""mini-swe-agent adapter (pinned mini-swe-agent 2.4.6 in bench/external/miniswe/.venv).

mini-swe-agent has no resume and no headless multi-turn mode, so the session runs in ONE process,
bench/miniswe_driver.py, which reproduces the interactive `mini` follow-up path ("Type new task"
-> `The user added a new task: ...` in the same message list) with the shipped DefaultAgent,
mini.yaml templates and LitellmModel. See the driver's docstring for the two recorded deviations
(Git Bash instead of cmd.exe; api_base -> proxy, cost_tracking=ignore_errors).

Settings (verified on 2.4.6, bench/REDESIGN.md section 4.3):
  * model: LiteLLM id `openrouter/<model>`, api_base <proxy>/api/v1 (POST /api/v1/chat/completions)
  * effort: model_kwargs.extra_body.reasoning.effort; output cap: model_kwargs.max_tokens = --max-out
  * window / compaction: none. mini sends the full history every step and stops when the provider
    rejects it; --window is not applied (recorded as "n/a").
  * tools: the single `bash` tool (nothing to switch off for web; shell use is audited like
    everywhere else). Confirmation prompts do not exist outside the TUI (`mode` is an
    InteractiveAgent setting; DefaultAgent executes directly).
  * limits: step_limit = --max-iters, cost_limit = --max-cost, both per SESSION as in `mini`.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "bench" / "external" / "miniswe" / ".venv"
DRIVER = ROOT / "bench" / "miniswe_driver.py"


def venv_python() -> Path:
    for p in (VENV / "Scripts" / "python.exe", VENV / "bin" / "python"):
        if p.exists():
            return p
    raise FileNotFoundError(f"pinned mini-swe-agent venv missing: {VENV}")


def git_bash() -> str:
    for p in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if Path(p).exists():
            return p
    b = shutil.which("bash")
    if b and "system32" not in b.lower():  # never WSL's bash.exe
        return b
    raise FileNotFoundError("Git Bash not found")


def mini_version() -> str:
    p = subprocess.run([str(venv_python()), "-I", "-c", "import minisweagent,sys; print(minisweagent.__version__)"],
                       capture_output=True, text=True, timeout=120, env={**os.environ, "MSWEA_SILENT_STARTUP": "1"})
    return "mini-swe-agent " + (p.stdout.strip().splitlines() or ["?"])[-1]


def _row(r: dict, model: str) -> dict:
    """chat/completions usage: prompt_tokens includes cached reads and cache writes."""
    cr, cw = r.get("cached") or 0, r.get("cache_write") or 0
    return {"source": "agent", "model": r.get("model") or model, "provider": None,
            "uncached": max((r.get("prompt_tokens") or 0) - cr - cw, 0), "cache_write": cw, "cache_read": cr,
            "output": r.get("completion_tokens") or 0, "reasoning": r.get("reasoning"),
            "reported_cost": r.get("cost")}


def run_miniswe(a, workdir: Path, prompts: list[str], tdir: Path, proxy=None,
                pre_hooks: list | None = None) -> list[dict]:
    from bench.run import _new_turn
    base = f"http://127.0.0.1:{proxy.port}/api/v1" if proxy is not None else "https://openrouter.ai/api/v1"
    spec = {"model": f"openrouter/{a.model}", "api_base": base, "cwd": str(workdir), "bash": git_bash(),
            "model_kwargs": {"max_tokens": a.max_out,
                             "extra_body": {"reasoning": {"effort": a.reasoning}, "usage": {"include": True}}},
            "step_limit": a.max_iters, "cost_limit": a.max_cost, "traj": str(tdir / "miniswe.traj.json")}
    env = {**os.environ, "MSWEA_SILENT_STARTUP": "1", "MSWEA_GLOBAL_CONFIG_DIR": str(tdir / "mini_cfg"),
           "LITELLM_LOCAL_MODEL_COST_MAP": "True", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    cmd = [str(venv_python()), "-X", "utf8", str(DRIVER)]  # not -I: it would drop PYTHONIOENCODING
    p = subprocess.Popen(cmd, cwd=tdir, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(tdir / "miniswe_stderr.log", "w", encoding="utf-8"), text=True,
                         encoding="utf-8", errors="replace")
    lines: queue.Queue = queue.Queue()

    def pump():
        for line in p.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    raw = open(tdir / "miniswe_stream.jsonl", "a", encoding="utf-8")

    def send(obj):
        p.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        p.stdin.flush()

    def next_event(deadline):
        while True:
            try:
                line = lines.get(timeout=max(deadline - time.time(), 0.1))
            except queue.Empty:
                return "timeout"
            if line is None:
                return None
            raw.write(line)
            raw.flush()
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    turns = []
    try:
        send(spec)
        ready = next_event(time.time() + 300)
        if not isinstance(ready, dict) or ready.get("event") != "ready":
            st = _new_turn()
            st["errors"].append(f"driver did not start: {ready}")
            return [st]
        tools = ready.get("tools") or []
        (tdir / "harness_flags.json").write_text(json.dumps({
            "harness": "miniswe", "version": f"mini-swe-agent {ready.get('mini_version')}",
            "argv": cmd, "spec": {k: v for k, v in spec.items()}, "driver": ready,
            "window": "n/a (no window setting, no compaction; full history every step)",
            "tools_in_force": tools, "web_off": "no web tool exists (bash only)",
            "deviations": ["commands run via Git Bash `bash -c` instead of shell=True (cmd.exe)",
                           "api_base -> logging proxy; cost_tracking=ignore_errors (cost from proxy)",
                           "multi-turn = the interactive 'Type new task' path, driven by bench/miniswe_driver.py"],
        }, indent=1, default=str), encoding="utf-8")
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
            send({"prompt": prompt})
            ev = next_event(time.time() + a.timeout)
            if ev == "timeout":
                st["aborted"] = f"timeout {a.timeout}s"
                break
            if ev is None:
                st["errors"].append(f"driver exited (code {p.poll()}); see miniswe_stderr.log")
                break
            st["final"] = ev.get("final") or ""
            st["finish"] = ev.get("exit_status")
            st["tools"] = ["bash"] * len(ev.get("tools") or [])
            st["commands"] = ev.get("tools") or []
            st["trace"] = [_row(r, a.model) for r in ev.get("trace") or []]
            if ev.get("error"):
                st["errors"].append(ev["error"])
                break
        else:
            send({"end": True})
    finally:
        try:
            p.stdin.close()
        except OSError:
            pass
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
        raw.close()
    return turns
