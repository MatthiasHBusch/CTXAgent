"""mini-swe-agent driver: one multi-turn session in ONE process, run with the pinned venv python
(bench/external/miniswe/.venv). Protocol on stdin/stdout, one JSON object per line:

    stdin  line 1:  spec   {"model", "api_base", "cwd", "model_kwargs", "step_limit", "cost_limit",
                            "cmd_timeout", "bash", "traj"}
    stdin  then:    {"prompt": "..."} for each turn, or {"end": true}
    stdout:         {"event": "ready", "tools": [...], "flags": {...}}  once, then one turn record
                    {"event": "turn", "turn", "final", "submission", "exit_status", "tools", "trace",
                     "n_steps", "error", "wall_s"} per turn

Why a driver: mini-swe-agent 2.4.6 has no session resume and no headless multi-turn mode. `mini`
is an interactive TUI; a multi-turn session there is: the agent submits (`echo
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`), the TUI prints "Agent wants to finish. Type new task",
the user types the next question, and InteractiveAgent continues the SAME message list with
`The user added a new task: <text>` (agents/interactive.py, `_check_for_new_task_or_submit`).
SessionAgent below reproduces exactly that path with the shipped DefaultAgent, mini.yaml templates
and LitellmModel (the `mini` default model class), so nothing about prompts, tools or the
observation format is changed. The next prompt is read from stdin at the moment the agent submits,
so the runner can retag the logging proxy's turn before the next model request goes out.

Recorded deviations from `mini` (all written to the trial's harness_flags):
  * Commands run under Git Bash (`bash -c`) instead of `shell=True` (= cmd.exe on Windows): the
    shipped prompt promises bash and gives heredoc/sed examples, which cmd.exe cannot run.
  * `api_base` points LiteLLM at the logging proxy; `cost_tracking=ignore_errors` because cost is
    taken from the proxy (LiteLLM's local price table lacks new models and would abort the run).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# bench/ (this file's dir) holds modules like tasks.py / proxy.py: never let them shadow imports
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # no fetch of the price map at import

import yaml  # noqa: E402
from minisweagent import __version__  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.config import builtin_config_dir  # noqa: E402
from minisweagent.environments.local import LocalEnvironment  # noqa: E402
from minisweagent.exceptions import Submitted, UserInterruption  # noqa: E402
from minisweagent.models import get_model  # noqa: E402
from minisweagent.models.utils.actions_toolcall import BASH_TOOL  # noqa: E402

NEW_TASK = "The user added a new task: {task}"  # verbatim from InteractiveAgent


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def read_msg() -> dict:
    line = sys.stdin.readline()
    return json.loads(line) if line.strip() else {"end": True}


class BashLocalEnvironment(LocalEnvironment):
    """LocalEnvironment, but each action runs as `bash -c <command>`."""

    def __init__(self, *, bash: str, **kwargs):
        super().__init__(**kwargs)
        self.bash = bash

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        try:
            p = subprocess.run([self.bash, "-c", command], cwd=cwd, env=os.environ | self.config.env,
                               timeout=timeout or self.config.timeout, text=True, encoding="utf-8",
                               errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL)
            out = {"output": p.stdout, "returncode": p.returncode, "exception_info": ""}
        except Exception as e:  # noqa: BLE001 - same shape as LocalEnvironment
            raw = getattr(e, "output", None)
            raw = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else (raw or "")
            out = {"output": raw, "returncode": -1,
                   "exception_info": f"An error occurred while executing the command: {e}",
                   "extra": {"exception_type": type(e).__name__, "exception": str(e)}}
        self._check_finished(out)
        return out


class SessionAgent(DefaultAgent):
    """DefaultAgent + the 'type new task' continuation of InteractiveAgent (yolo, no TUI)."""

    def __init__(self, *a, on_submit, **kw):
        super().__init__(*a, **kw)
        self.on_submit = on_submit  # called with the Submitted exception; returns next task or None

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        try:
            for action in actions:
                outputs.append(self.env.execute(action))
        except Submitted as e:
            nxt = self.on_submit(e)
            if nxt is not None:
                # observations first (padded with "action was not executed"), then the new task
                self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))
                raise UserInterruption({"role": "user", "content": NEW_TASK.format(task=nxt),
                                        "extra": {"interrupt_type": "UserNewTask"}}) from None
            raise
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))


def _text(msg: dict) -> str:
    """The assistant's own text (get_content_string would append the rendered tool calls)."""
    c = msg.get("content")
    if isinstance(c, list):
        return "".join(p.get("text") or "" for p in c if isinstance(p, dict) and p.get("type") in (None, "text"))
    return c or ""


def _trace_row(msg: dict, model: str) -> dict | None:
    r = (msg.get("extra") or {}).get("response") or {}
    u = r.get("usage") or {}
    if not u:
        return None
    pd, cd = u.get("prompt_tokens_details") or {}, u.get("completion_tokens_details") or {}
    return {"model": r.get("model") or model, "prompt_tokens": u.get("prompt_tokens") or 0,
            "completion_tokens": u.get("completion_tokens") or 0, "cached": (pd or {}).get("cached_tokens") or 0,
            "cache_write": (pd or {}).get("cache_write_tokens") or 0, "reasoning": (cd or {}).get("reasoning_tokens"),
            "cost": u.get("cost")}


def main() -> int:
    spec = json.loads(sys.stdin.readline())
    cfg = yaml.safe_load((builtin_config_dir / "mini.yaml").read_text(encoding="utf-8"))
    model_cfg = dict(cfg["model"])
    model_cfg["model_kwargs"] = {**(model_cfg.get("model_kwargs") or {}), **(spec.get("model_kwargs") or {}),
                                 "api_base": spec["api_base"]}
    model_cfg.update(model_name=spec["model"], cost_tracking="ignore_errors")
    model = get_model(config=model_cfg)  # LitellmModel; Anthropic models get cache_control like `mini`
    env = BashLocalEnvironment(bash=spec["bash"], cwd=spec["cwd"], env=cfg["environment"].get("env") or {},
                               timeout=int(spec.get("cmd_timeout") or 120))
    agent_cfg = {k: v for k, v in cfg["agent"].items() if k in ("system_template", "instance_template")}

    state = {"turn": 1, "t0": time.time(), "n0": 0}

    def record(extra: dict, err: str | None = None) -> None:
        new = agent.messages[state["n0"]:]
        assistant = [m for m in new if m.get("role") == "assistant"]
        last = _text(assistant[-1]) if assistant else ""
        sub = (extra.get("submission") or "").strip()
        emit({"event": "turn", "turn": state["turn"], "final": (last.strip() + ("\n" + sub if sub else "")).strip(),
              "submission": sub, "exit_status": extra.get("exit_status"),
              "tools": [a.get("command", "") for m in assistant for a in (m.get("extra") or {}).get("actions", [])],
              "n_steps": len(assistant), "trace": [r for r in (_trace_row(m, spec["model"]) for m in assistant) if r],
              "error": err, "wall_s": round(time.time() - state["t0"], 1)})

    def on_submit(e: Submitted):
        record(e.messages[0].get("extra", {}))
        msg = read_msg()
        if msg.get("end"):
            return None
        state.update(turn=state["turn"] + 1, t0=time.time(), n0=len(agent.messages))
        return msg["prompt"]

    agent = SessionAgent(model, env, on_submit=on_submit, step_limit=int(spec.get("step_limit") or 0),
                         cost_limit=float(spec.get("cost_limit") or 0),
                         output_path=Path(spec["traj"]) if spec.get("traj") else None, **agent_cfg)
    emit({"event": "ready", "mini_version": __version__, "tools": [BASH_TOOL["function"]["name"]],
          "model_class": type(model).__name__, "model_kwargs": {k: v for k, v in model_cfg["model_kwargs"].items()
                                                               if k != "api_key"},
          "set_cache_control": getattr(model.config, "set_cache_control", None), "bash": spec["bash"]})
    first = read_msg()
    if first.get("end"):
        return 0
    try:
        extra = agent.run(first["prompt"])
    except Exception as e:  # noqa: BLE001
        record({"exit_status": type(e).__name__}, f"{type(e).__name__}: {e}"[:800])
        return 1
    if extra.get("exit_status") != "Submitted":  # LimitsExceeded / RepeatedFormatError ends the session
        record(extra, f"session ended: {extra.get('exit_status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
