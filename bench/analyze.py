"""Summaries of result files: per (model, harness) accuracy, cost, tokens; paired comparisons.

  python -m bench.analyze bench/results/pilot_cb.jsonl [more.jsonl ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import mean, median


def load(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def main(paths) -> int:
    rows = load(paths)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r.get("harness") or r.get("arm"))].append(r)
    hdr = f"{'model':28} {'harness':20} {'n':>3} {'err':>3} {'acc':>5} {'$/task':>8} {'$tot':>7} " \
          f"{'req':>5} {'tools':>5} {'peak':>7} {'in_tok':>8} {'cr%':>5} {'comp':>4} {'wall':>5}"
    print(hdr)
    print("-" * len(hdr))
    for (m, h), rs in sorted(groups.items()):
        ok = [r for r in rs if not r.get("harness_error")]
        if not ok:
            print(f"{m:28} {h:20} {len(rs):3} all harness errors")
            continue
        tin = [r["uncached"] + r["cache_write"] + r["cache_read"] for r in ok]
        cr = sum(r["cache_read"] for r in ok) / max(sum(tin), 1)
        peaks = [r["peak_context"] for r in ok if r.get("peak_context")]
        print(f"{m:28} {h:20} {len(rs):3} {len(rs) - len(ok):3} {mean(r['score'] for r in ok):5.2f} "
              f"{mean(r['cost'] for r in ok):8.4f} {sum(r['cost'] for r in ok):7.3f} "
              f"{mean(r['requests'] for r in ok):5.1f} {mean(r['n_tools'] for r in ok):5.1f} "
              f"{(median(peaks) if peaks else float('nan')):7.0f} {mean(tin):8.0f} {cr * 100:5.1f} "
              f"{sum(r['compaction_events'] for r in ok):4} {mean(r['wall_s'] for r in ok):5.0f}")
    # paired per task: same model family across harnesses
    by_task = defaultdict(dict)
    for r in rows:
        if not r.get("harness_error"):
            by_task[(r["model"], r["task"], r.get("rep", 0))][r.get("harness") or r.get("arm")] = r
    pairs = defaultdict(list)
    for (m, t, _), hs in by_task.items():
        names = sorted(hs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairs[(m, a, b)].append((hs[a], hs[b]))
    if pairs:
        print("\npaired (same model, same task):  cost ratio b/a (median), score diffs")
        for (m, a, b), ps in sorted(pairs.items()):
            ratios = [pb["cost"] / pa["cost"] for pa, pb in ps if pa["cost"]]
            wins = sum(pb["score"] > pa["score"] for pa, pb in ps)
            losses = sum(pb["score"] < pa["score"] for pa, pb in ps)
            print(f"  {m:28} {b} vs {a}: n={len(ps)}  cost x{median(ratios):.2f}  "
                  f"{b} better on {wins}, worse on {losses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
