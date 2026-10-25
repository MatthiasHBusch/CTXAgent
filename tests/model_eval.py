"""Which model should do the compaction work?

Compaction is a *different* job from being the agent: it is long-input,
short-output, instruction-following, and above all it must not lose facts. Price
per token for that shape varies ~50x across models that can all "summarise", so
the question is worth answering empirically rather than by reputation.

Method
------
Two tasks, using ctxagent's real production prompts:

  A. deep summary (tier 3)  -- a ~35k-token synthetic agent trajectory with
     facts planted in it, summarised through `summarize.deep_summary`.
  B. condensation (tier 2)  -- eight noisy tool outputs with planted specifics,
     through `summarize.condense_batch`, which also demands valid JSON.

Scoring is deterministic: each planted fact is a string (or one of several
accepted spellings) that must survive into the summary. No LLM judge, so the
numbers are reproducible and cost nothing to recompute.

Run:  python -m tests.model_eval            (all candidates)
      python -m tests.model_eval --quick    (a short list)
      python -m tests.model_eval --models a,b,c
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ctxagent import llm, providers as provmod, tokens
from ctxagent.compaction import summarize
from ctxagent.config import load_env
from ctxagent.segments import (
    Conversation, K_SYSTEM, K_USER, K_ASSISTANT, K_TOOL_CALL, K_TOOL_RESULT,
)

# ---------------------------------------------------------------------------
# Planted facts: (label, [accepted spellings])
# ---------------------------------------------------------------------------
FACTS: list[tuple[str, list[str]]] = [
    ("file path",        ["src/solver/newton.py"]),
    ("dependency ver",   ["numpy&2.1.3"]),
    ("error string",     ["singular jacobian", "zerodivisionerror"]),
    ("error location",   ["iteration 7", "iteration7", "iteration&7"]),
    ("decision+reason",  ["damped line search", "damped-line-search"]),
    ("tolerance",        ["1e-11"]),
    ("test command",     ["tests/test_newton.py", "pytest&test_newton"]),
    ("port number",      ["8765"]),
    ("env quirk",        ["mkl_num_threads"]),
    ("open todo",        ["damping"]),
    ("superseded ask",   ["bisection"]),
    ("second file",      ["src/report/render.py"]),
    ("magic constant",   ["0.618", "golden"]),
    ("dataset name",     ["beam_deflection_2026.csv", "beam_deflection"]),
    ("collaborator ask", ["si units", "si-units", "si&unit"]),
]

CONDENSE_FACTS: list[tuple[str, list[str]]] = [
    ("exit code",   ["exit=137", "137"]),
    ("oom",         ["out of memory", "oom", "memory"]),
    ("row count",   ["48211", "48,211"]),
    ("failed test", ["test_singular_jacobian"]),
    ("host:port",   ["127.0.0.1:5432", "5432"]),
]


# ---------------------------------------------------------------------------
# Synthetic trajectory
# ---------------------------------------------------------------------------

def build_trajectory() -> Conversation:
    conv = Conversation()
    conv.new("system", K_SYSTEM, "You are a coding agent.", pinned=True)

    conv.new("user", K_USER,
             "I need a root finder for the beam deflection model. Start with "
             "bisection on the residual.")
    conv.new("assistant", K_ASSISTANT,
             "I'll implement bisection in `src/solver/newton.py` first and we can "
             "revisit if convergence is slow.")

    def call(n, name, args, body, **kw):
        cid = f"c{n}"
        conv.new("assistant", K_TOOL_CALL, "", tool_calls=[{
            "id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}])
        conv.new("tool", K_TOOL_RESULT, body, tool_call_id=cid, tool_name=name, **kw)

    call(1, "run_command", {"command": "pip show numpy"},
         "Name: numpy\nVersion: 2.1.3\nLocation: C:\\anaconda3\\Lib\\site-packages\n"
         + "\n".join(f"Requires: pkg_{i}" for i in range(40)))

    call(2, "read_file", {"path": "src/solver/newton.py"},
         "\n".join(f"{i:6d}\t    # bisection scaffolding line {i}" for i in range(900)))

    conv.new("user", K_USER,
             "Actually forget bisection — switch to a Newton iteration. It has to "
             "converge to a tolerance of 1e-11.")
    conv.new("assistant", K_ASSISTANT,
             "Switching to Newton. I'll use a damped line search because the "
             "Jacobian is cheap to form analytically for this residual, and "
             "undamped Newton diverges on the stiff cases.")

    call(3, "edit_file", {"path": "src/solver/newton.py"},
         "Edited src/solver/newton.py: 1 replacement(s).", mutates=True,
         resource_key="file:src/solver/newton.py")

    call(4, "run_command", {"command": "pytest -q tests/test_newton.py"},
         "F" * 60 + "\n"
         "tests/test_newton.py::test_singular_jacobian FAILED\n"
         "E   ZeroDivisionError: singular Jacobian at iteration 7\n"
         + "\n".join(f"  frame {i} in numpy/linalg/_linalg.py" for i in range(80))
         + "\n1 failed, 23 passed in 4.12s")

    conv.new("assistant", K_ASSISTANT,
             "The Jacobian goes singular at iteration 7. I'll fall back to a "
             "pseudo-inverse when the condition number blows up.")

    call(5, "edit_file", {"path": "src/solver/newton.py"},
         "Edited src/solver/newton.py: 2 replacement(s).", mutates=True,
         resource_key="file:src/solver/newton.py")

    call(6, "run_command", {"command": "pytest -q tests/test_newton.py"},
         "\n".join(f"tests/test_newton.py::test_case_{i} PASSED" for i in range(24))
         + "\n24 passed in 3.87s")

    call(7, "run_command", {"command": "python -c 'import mkl'"},
         "NOTE: on this machine the solver only reproduces if MKL_NUM_THREADS=1 "
         "is set; otherwise the last digit drifts.\n"
         + "\n".join(f"thread pool worker {i} idle" for i in range(60)))

    call(8, "read_document", {"path": "data/beam_deflection_2026.csv"},
         "data/beam_deflection_2026.csv  [csv, 48211 rows extracted]\n"
         + "\n".join(f"{i},{i*0.618:.4f},{i*2}" for i in range(400)))

    conv.new("user", K_USER,
             "Jonas from the lab says the report has to be in SI units throughout. "
             "Also please serve the preview on port 8765.")

    call(9, "read_file", {"path": "src/report/render.py"},
         "\n".join(f"{i:6d}\t    render_line({i})  # golden ratio spacing 0.618"
                   for i in range(700)))

    call(10, "edit_file", {"path": "src/report/render.py"},
         "Edited src/report/render.py: 4 replacement(s).", mutates=True,
         resource_key="file:src/report/render.py")

    call(11, "run_command", {"command": "python -m report.serve --port 8765"},
         "serving on http://127.0.0.1:8765\n"
         + "\n".join(f"GET /asset/{i} 200" for i in range(120)))

    conv.new("assistant", K_ASSISTANT,
             "Report renders in SI units and the preview is on port 8765. Still "
             "outstanding: document the `damping` parameter in the docstring.")
    return conv


def build_condense_segments() -> Conversation:
    conv = Conversation()
    conv.new("system", K_SYSTEM, "sys", pinned=True)
    payloads = [
        "Killed\nexit=137\nThe process ran out of memory while loading the mesh.\n"
        + "\n".join(f"alloc chunk {i}" for i in range(150)),
        "Loaded data/beam_deflection_2026.csv: 48211 rows, 12 columns\n"
        + "\n".join(f"col_{i} parsed" for i in range(150)),
        "tests/test_newton.py::test_singular_jacobian FAILED\n"
        + "\n".join(f"  frame {i}" for i in range(150)),
        "psql: could not connect to server at 127.0.0.1:5432\n"
        + "\n".join(f"retry {i}" for i in range(150)),
    ] + ["\n".join(f"uninteresting progress line {j} of run {k}" for j in range(160))
         for k in range(4)]
    segs = []
    for i, body in enumerate(payloads):
        cid = f"x{i}"
        conv.new("assistant", K_TOOL_CALL, "", tool_calls=[{
            "id": cid, "type": "function",
            "function": {"name": "run_command", "arguments": "{}"}}])
        segs.append(conv.new("tool", K_TOOL_RESULT, body, tool_call_id=cid,
                             tool_name="run_command", resource_key=f"cmd:{i}"))
    conv._eval_segs = segs  # type: ignore[attr-defined]
    return conv


def score(text: str, facts) -> tuple[int, list[str]]:
    """Count planted facts that survived.

    A spelling containing "&" requires every "&"-separated part to appear
    somewhere in the text. That matters: a model writing "Numpy version: 2.1.3"
    has preserved the fact perfectly, and scoring it as a miss because it did
    not write the literal string "numpy 2.1.3" measures phrasing, not recall.
    """
    low = (text or "").lower()

    def present(spelling: str) -> bool:
        return all(part.strip() in low for part in spelling.lower().split("&"))

    hits, missed = 0, []
    for label, spellings in facts:
        if any(present(s) for s in spellings):
            hits += 1
        else:
            missed.append(label)
    return hits, missed


# ---------------------------------------------------------------------------

CANDIDATES = [
    # user-suggested: strong benchmark scores at low *input* price, which is the
    # cost that dominates a long-input / short-output workload like compaction
    "z-ai/glm-5.3-flash",
    "qwen/qwen3.8-flash",
    "openai/gpt-5.6-luna",
    # previous field
    "inclusionai/ling-3.0-flash",
    "z-ai/glm-4.7-flash",
    "qwen/qwen3.7-flash",
    "openai/gpt-oss-120b",
    "mistralai/mistral-small-3.2-24b-instruct",
    "openai/gpt-5-nano",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.5-flash-02-23",
    "anthropic/claude-haiku-4.5",
]
QUICK = ["z-ai/glm-5.3-flash", "openai/gpt-5.6-luna",
         "qwen/qwen3.8-flash", "inclusionai/ling-3.0-flash"]

PRICES: dict[str, tuple[float, float]] = {}


async def load_prices() -> None:
    try:
        for m in await llm.list_models():
            p = m.get("pricing") or {}
            PRICES[m["id"]] = (float(p.get("prompt") or 0), float(p.get("completion") or 0))
    except Exception as e:
        print(f"(could not load pricing: {e})")


def cost_of(model: str, tin: int, tout: int) -> float:
    pr, co = PRICES.get(model, (0.0, 0.0))
    return pr * tin + co * tout


async def eval_model(model: str, conv: Conversation, cond: Conversation,
                     policy: str, summary_tokens: int) -> dict:
    victims = [s for s in conv.segments if s.kind != K_SYSTEM]
    in_tokens = sum(s.tokens() for s in victims)
    row = {"model": model, "in_tokens": in_tokens, "policy": policy}

    async def complete(**kw):
        kw.setdefault("provider_policy", policy)
        return await llm.complete(**kw)

    # --- task A: deep summary -------------------------------------------
    t0 = time.time()
    try:
        text = await summarize.deep_summary(
            victims, None, model, complete, max_tokens=summary_tokens)
        row["A_ms"] = int((time.time() - t0) * 1000)
        hits, missed = score(text, FACTS)
        row["A_hits"] = hits
        row["A_missed"] = missed
        row["A_out"] = tokens.count(text)
        row["A_sections"] = sum(1 for i in range(1, 10) if f"## {i}." in text)
        row["A_text"] = text
    except Exception as e:
        row["A_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        row["A_hits"] = 0
        row["A_out"] = 0
        row["A_ms"] = int((time.time() - t0) * 1000)
        row["A_sections"] = 0
        row["A_missed"] = [f[0] for f in FACTS]

    # --- task B: condensation (JSON) -------------------------------------
    segs = [s for s in cond.segments if s.kind == K_TOOL_RESULT]
    for s in segs:  # reset between models
        s.body = s.original_body or s.body
        s.original_body = None
        s.state = "original"
        s.invalidate()
    before = sum(s.tokens() for s in segs)
    t0 = time.time()
    r = await summarize.condense_batch(segs, model, complete)
    row["B_ms"] = int((time.time() - t0) * 1000)
    row["B_ok"] = not r.get("error") and r.get("count", 0) > 0
    row["B_count"] = r.get("count", 0)
    row["B_freed"] = r.get("freed", 0)
    row["B_ratio"] = (before - sum(s.tokens() for s in segs)) / max(1, before)
    merged = "\n".join(s.body for s in segs)
    row["B_hits"], row["B_missed"] = score(merged, CONDENSE_FACTS)

    row["cost"] = cost_of(model, in_tokens + before, row["A_out"] + 400)
    return row


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--summary-tokens", type=int, default=10_000,
                    dest="summary_tokens",
                    help="max_tokens for the deep summary. Default matches "
                         "production (compaction aims at level 4 = 10,000).")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Runs per model; scores are averaged. Compaction "
                         "quality varies run to run, so >1 is worth it "
                         "before committing to a default.")
    ap.add_argument("--out", default="tests/model_eval_results.json")
    args = ap.parse_args()

    load_env()
    await load_prices()

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              or (QUICK if args.quick else CANDIDATES))

    conv = build_trajectory()
    cond = build_condense_segments()
    traj_tokens = sum(s.tokens() for s in conv.segments)
    print("ctxagent — compaction model evaluation")
    print(f"  trajectory: {traj_tokens:,} tokens, {len(conv.segments)} segments, "
          f"{len(FACTS)} planted facts")
    print(f"  condensation: {len([s for s in cond.segments if s.kind == K_TOOL_RESULT])} "
          f"noisy outputs, {len(CONDENSE_FACTS)} planted facts")
    print(f"  models: {len(models)}\n")

    print("  region policy: EU/US providers only where possible\n")
    verdicts: dict[str, dict] = {}
    for m in models:
        verdicts[m] = await provmod.check_model(m, provmod.POLICY_EU_US)

    blocked = [m for m, v in verdicts.items() if not v["ok"]]
    if blocked:
        print("  !! no EU/US endpoint for: " + ", ".join(blocked))
        print("     these run under policy='any' and are flagged in the table.\n")

    rows = []
    for m in models:
        policy = provmod.POLICY_EU_US if verdicts[m]["ok"] else provmod.POLICY_ANY
        runs = []
        for i in range(max(1, args.repeat)):
            print(f"  … {m}" + (f"  (run {i + 1}/{args.repeat})" if args.repeat > 1 else ""),
                  flush=True)
            try:
                runs.append(await eval_model(m, conv, cond, policy,
                                             args.summary_tokens))
            except Exception as e:
                print(f"      failed: {type(e).__name__}: {e}")
        if not runs:
            continue
        base = dict(runs[-1])
        n = len(runs)
        for k in ("A_hits", "B_hits", "A_out", "A_ms", "B_ms", "A_sections",
                  "B_ratio", "cost", "B_count", "B_freed"):
            base[k] = sum(r[k] for r in runs) / n
        base["runs"] = n
        base["A_hits_spread"] = (min(r["A_hits"] for r in runs),
                                 max(r["A_hits"] for r in runs))
        base["B_hits_spread"] = (min(r["B_hits"] for r in runs),
                                 max(r["B_hits"] for r in runs))
        base["A_missed"] = sorted({x for r in runs for x in r["A_missed"]})
        base["B_ok"] = all(r["B_ok"] for r in runs)
        base["eu_us"] = verdicts[m]["ok"]
        base["region_note"] = verdicts[m]["message"]
        rows.append(base)

    # EU/US-servable models first, then by facts preserved, then by price.
    rows.sort(key=lambda r: (not r.get("eu_us"),
                             -(r["A_hits"] + r["B_hits"]), r["cost"]))

    hdr = (f"\n{'model':<34}{'EU/US':>6} {'facts':>9}{'sect':>6}{'JSON':>6}"
           f"{'shrink':>8}{'out':>7}{'sec':>7}{'$/1k runs':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        facts = f"{r['A_hits']:.1f}+{r['B_hits']:.1f}"
        print(f"{r['model']:<34}{('yes' if r.get('eu_us') else 'NO'):>6} {facts:>9}"
              f"{r['A_sections']:>6.0f}{('ok' if r['B_ok'] else 'FAIL'):>6}"
              f"{r['B_ratio']:>8.0%}{r['A_out']:>7.0f}"
              f"{(r['A_ms'] + r['B_ms']) / 1000:>7.1f}"
              f"{r['cost'] * 1000:>11.2f}")
        if r.get("A_error"):
            print(f"{'':<42}  error: {r['A_error']}")
        if r["A_missed"]:
            print(f"{'':<42}  lost: {', '.join(r['A_missed'])}")

    total = sum(r["cost"] for r in rows)
    print(f"\ntotal spend for this run: ${total:.4f}")

    Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"full results (including every summary) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
