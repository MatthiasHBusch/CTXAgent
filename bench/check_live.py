"""Sanity report for ctxagent_live trials (smoke and main runs).

python -m bench.check_live <results.jsonl> [--prefix lv0_]

Per trial: score / retention / billed; effective policy; review hints delivered vs. the
agent's own context-management calls; compaction-log reasons; proxy-billed vs.
ctxagent-accounted cost (equal => no LLM traffic bypassed the proxy); delegated agents.
"""
import argparse, json, sqlite3
from collections import Counter
from pathlib import Path

from bench.run import WORK

HINT = "[harness: context review due"
SELF_TOOLS = ("curate_context", "compact_context", "pin", "recall", "context_report")


def db_report(tdir: Path) -> dict:
    db = tdir / "ctxdata" / "ctxagent.db"
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = c.execute("select id, conversation, compaction_log, usage from chats").fetchall()
    out = {"chats": len(rows)}
    main = rows[0]
    conv, clog, usage = json.loads(main[1]), json.loads(main[2] or "[]"), json.loads(main[3] or "{}")
    segs = {s["id"]: s for s in conv["segments"]}
    for sid, blob in (conv.get("archive") or {}).items():
        p = tdir / "ctxdata" / "blobs" / f"{blob}.txt"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8")); segs.setdefault(s["id"], s)
    segs = sorted(segs.values(), key=lambda s: s["seq"])
    out["hints"] = sum(1 for s in segs if s["kind"] == "tool_result" and HINT in (s.get("original_body") or s["body"] or ""))
    calls = Counter()
    for s in segs:
        if s["kind"] == "tool_call":
            for tc in s.get("tool_calls") or []:
                calls[(tc.get("function") or {}).get("name") or tc.get("name")] += 1
    out["self_calls"] = {k: calls[k] for k in SELF_TOOLS if calls[k]}
    out["agent_calls"] = {k: v for k, v in calls.items() if k and k.startswith("agent_")}
    reasons = Counter()
    for e in clog:
        for t in e.get("tiers") or []:
            reasons[t.get("tier")] += 1
        if (e.get("schedule") or {}).get("reason"):
            reasons["reason:" + e["schedule"]["reason"]] += 1
    out["compaction_log"] = dict(reasons)
    out["accounted_cost"] = round(usage.get("cost") or 0, 4)
    out["accounted_requests"] = usage.get("requests")
    out["agent_tasks"] = c.execute("select count(*) from agent_tasks").fetchone()[0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    for line in Path(a.results).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r["tag"].startswith(a.prefix):
            continue
        cc = (r.get("ctx_config") or {}).get("compaction") or {}
        print("=" * 8, r["tag"])
        print(f"  score={r['score']:.3f} retention={r.get('recap_score')} billed=${r['billed_cost']:.3f} "
              f"wire_req={r['wire_requests']} wire_err={r['wire_errors']} peak={r['wire_peak_context']} "
              f"harness_error={r['harness_error']} aborted={r['aborted']} version={r['harness_version']}")
        print(f"  config: window={(r.get('ctx_config') or {}).get('context_window')} policy={cc.get('rewrite_policy')} "
              f"interval={cc.get('rewrite_interval_batches')} overdue={cc.get('agent_review_overdue_batches')} "
              f"cap={cc.get('initial_result_max_tokens')} summary={cc.get('summary_model')}")
        d = db_report(WORK / r["tag"])
        print(f"  hints={d['hints']} self_calls={d['self_calls']} agent_calls={d['agent_calls']} agent_tasks={d['agent_tasks']}")
        print(f"  compaction_log={d['compaction_log']}")
        same = abs(d["accounted_cost"] - r["billed_cost"]) < 0.01
        print(f"  proxy billed ${r['billed_cost']:.4f} vs ctxagent accounted ${d['accounted_cost']:.4f} "
              f"({'MATCH' if same else 'MISMATCH'}), requests {r['wire_requests']} vs {d['accounted_requests']}")
        if r.get("errors"):
            print("  errors:", r["errors"][:2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
