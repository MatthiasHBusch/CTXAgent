"""Correct multi-turn rows recorded before the cumulative-usage fixes (commit after 6c3fe6a).

Both CLIs report CUMULATIVE session usage after a resume:
  - Codex `turn.completed.usage` after `exec resume` -> per-turn = delta.
  - Claude Code `result.modelUsage` after `--resume` -> per-turn = delta; the streamed
    per-message usage under-reports output, so the gap is booked as 'unstreamed'.
The raw streams in bench/.work/<tag>/ are re-parsed with the fixed parsers; scores are
untouched, token/cost fields are rebuilt. Rows get "usage_fixed": 2. Idempotent.

  python -m bench.fix_usage bench/results/seq_pilot.jsonl [...]
"""
import json
import sys
from types import SimpleNamespace

from bench import pricing
from bench.run import WORK, _new_turn, _parse_cc_stream, codex_usage_row


def _codex_rows(r):
    rows, prev = [], None
    for line in (WORK / r["tag"] / "codex_stream.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if j.get("type") in ("turn.completed", "turn.failed"):
            row, prev = codex_usage_row(r["model"], j.get("usage") or {}, prev)
            rows.append([row])
    return rows


def _cc_rows(r):
    chunks, cur = [], []
    for line in (WORK / r["tag"] / "claude_stream.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            j = {}
        if j.get("type") == "system" and j.get("subtype") == "init" and cur:
            chunks.append(cur)
            cur = []
        cur.append(line)
    if cur:
        chunks.append(cur)
    a, cum, out = SimpleNamespace(model=r["model"]), {}, []
    for c in chunks:
        st = _new_turn()
        cum = _parse_cc_stream(a, "\n".join(c), st, cum)
        out.append(st["trace"])
    return out


def fix_row(r: dict) -> dict:
    if r.get("usage_fixed") == 2 or len(r.get("turns") or []) < 2 or r.get("harness") not in ("codex", "claude_code"):
        return r  # single process (no resume) already reports per-turn usage
    per_turn = _codex_rows(r) if r["harness"] == "codex" else _cc_rows(r)
    if len(per_turn) != len(r["turns"]):
        raise ValueError(f"{r['tag']}: {len(per_turn)} turn chunks vs {len(r['turns'])} turns")
    trace = []
    for k, (rows, turn) in enumerate(zip(per_turn, r["turns"]), 1):
        for t in rows:
            t["turn"] = k
            t["cost"] = pricing.cost(t["model"] or r["model"], t["uncached"], t["cache_write"], t["cache_read"],
                                     t["output"])
        turn["cost"] = round(sum(t["cost"] for t in rows), 6)
        trace += rows
    r["trace"] = trace
    for k in ("uncached", "cache_write", "cache_read", "output"):
        r[k] = sum(t[k] for t in trace)
    r["cost"] = round(sum(t["cost"] for t in trace), 6)
    r["cost_by_source"] = {s: round(sum(t["cost"] for t in trace if t["source"] == s), 6)
                           for s in sorted({t["source"] for t in trace})}
    r["requests"] = len(trace)
    r["usage_fixed"] = 2
    return r


def main(paths):
    for p in paths:
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        before = {r["tag"]: r.get("cost") for r in rows}
        rows = [fix_row(r) for r in rows]
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        for r in rows:
            if r.get("usage_fixed") == 2:
                print(f"{p}: {r['tag'][-45:]} cost {before[r['tag']]} -> {r['cost']}  {r['cost_by_source']}")


if __name__ == "__main__":
    main(sys.argv[1:])
