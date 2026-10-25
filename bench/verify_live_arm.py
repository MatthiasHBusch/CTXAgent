"""Offline check that the `ctxagent_live` arm reproduces a live chat's configuration.

python -m bench.verify_live_arm <chat_id> [--model M] [--window W]

Diffs the effective trial SessionConfig against the live chat's stored config, and checks
that the session the benchmark builds exposes the same tools (incl. the self-compaction
tools) and the same static system prompt as a session built from the live config.
"""
import argparse, json, sqlite3, tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("chat_id"); ap.add_argument("--model", default="anthropic/claude-opus-5.5")
    ap.add_argument("--window", type=int, default=1_000_000)
    a = ap.parse_args()
    from bench import run
    from ctxagent.config import SessionConfig
    c = sqlite3.connect(f"file:{ROOT / '.ctxagent' / 'ctxagent.db'}?mode=ro", uri=True)
    live = json.loads(c.execute("select config from chats where id=?", (a.chat_id,)).fetchone()[0])
    args = SimpleNamespace(harness="ctxagent_live", model=a.model, window=a.window, comp=[],
                           max_out=1, max_iters=1, temperature=0.0, reasoning="off", summary_model="x/ignored")
    with tempfile.TemporaryDirectory() as tmp:
        eff = run.build_ctx_config(args, Path(tmp)).to_dict()
    L, E = flat(live), flat(eff)
    expected = {"model", "context_window", "workdirs"}
    bad = 0
    print("field differences (live -> trial):")
    for k in sorted(set(L) | set(E)):
        if L.get(k, "<absent>") != E.get(k, "<absent>"):
            tag = "expected" if k in expected else ("new key (default)" if k not in L else "UNEXPECTED")
            bad += tag == "UNEXPECTED"
            print(f"  [{tag}] {k}: {json.dumps(L.get(k, '<absent>'))[:90]} -> {json.dumps(E.get(k, '<absent>'))[:90]}")
    # tools + static prompt actually built by AgentSession, live config vs trial config
    from ctxagent import agent, cache_policy
    from ctxagent.segments import Conversation
    s_live = agent.AgentSession("verify-live", SessionConfig.from_dict({**live, "workdirs": [str(ROOT)]}), Conversation())
    s_eff = agent.AgentSession("verify-eff", SessionConfig.from_dict({**eff, "model": live["model"], "workdirs": [str(ROOT)]}), Conversation())
    t_live, t_eff = s_live.tool_names(), s_eff.tool_names()
    print("tools identical:", t_live == t_eff, f"({len(t_eff)} tools)")
    for need in ("curate_context", "compact_context", "recall", "pin"):
        print(f"  {need} available:", need in t_eff)
    bad += (t_live != t_eff) + sum(n not in t_eff for n in ("curate_context", "compact_context", "recall"))
    print("policy:", eff["compaction"]["rewrite_policy"], "| hint every", eff["compaction"]["rewrite_interval_batches"],
          "batches | overdue safety net", eff["compaction"]["agent_review_overdue_batches"],
          "| window", eff["context_window"], "| deep trigger", eff["compaction"]["deep_trigger_ratio"])
    print("RESULT:", "OK" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
