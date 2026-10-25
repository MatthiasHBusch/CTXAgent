"""Reconcile recorded benchmark spend against what OpenRouter billed the benchmark key.

Recorded = sum of uniform token-priced trial costs (+ judge costs where recorded) over all
result files with the current key suffix. Billed = OpenRouter /key usage. Unrecorded
spend (summary-model calls inside ctxagent are recorded; probes and judge calls of the
recap are not) explains a positive residual; a large negative residual means we over-count.
"""
import glob
import json
import os

import httpx

from bench.run import _dotenv_keys

key = _dotenv_keys()["OPENROUTER_API_KEY"]
suffix = key[-4:]
billed = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"},
                   timeout=60).json()["data"]
recorded, judge, n = 0.0, 0.0, 0
by_file = {}
for p in sorted(glob.glob(os.path.join("bench", "results", "*.jsonl"))):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r.get("key_suffix") not in (suffix, None):
            continue
        if r.get("key_suffix") is None and "smoke" in p and "cb_" not in p:
            continue  # pre-fix runs billed to the old key
        c = r.get("cost") or 0
        j = (r.get("details") or {}).get("judge_cost") or 0
        j += sum(((t.get("details") or {}).get("judge_cost") or 0) for t in r.get("turns") or [])
        recorded += c
        judge += j
        n += 1
        by_file[p] = by_file.get(p, 0) + c + j
for p, c in by_file.items():
    print(f"{p:55} ${c:8.4f}")
print(f"\ntrials: {n}   recorded agent cost ${recorded:.4f}   judge ${judge:.4f}   total ${recorded + judge:.4f}")
print(f"OpenRouter billed to key: usage ${billed.get('usage')}   (daily {billed.get('usage_daily')})")
