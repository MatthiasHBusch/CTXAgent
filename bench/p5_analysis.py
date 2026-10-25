"""Phase 5': self-compaction capability across models.

Arm A = ctxagent_live (model may compact; context tools + prompt sections present)
Arm B = ctxagent_live_noctx (compaction engine off, context tools removed, prompt sections dropped)

Per model, over the same instrument (mixdep:34 seeds 0-2), the readout is:
  success (all scored kinds), question accuracy, memory (facts/ephemerals/probes/rules),
  billed cost at list rates (A, B, ratio A/B), peak and mean prompt tokens, output tokens,
  compactions per session (wire-detected drops), occupancy at compaction (prompt tokens just
  before the drop; absolute and as a fraction of the 1M window), failed *solvable* SciCode
  steps and how many of them are compaction-attributable (a compaction lies between the
  previous step of the same problem and the failed step — the attribution rule of phase 3a),
  context-tool call counts, wall time.  (In mixdep the SciCode steps are interleaved with
  fillers, so "a chain is open at the cut" is true for nearly every compaction and is not
  reported as a column; `session_stats` still returns it as `midchain`.)

Usage: python -m bench.p5_analysis bench/results/p5_models.jsonl [more files...] [--extra p3a_combined.jsonl]
The --extra files supply arm-A rows for models whose A sessions were run earlier (Opus/Sol in
phase 3a, same instrument and prompt); only ctxagent_live rows on mixdep:34 are taken from them.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from bench import analysis, seq_analysis as sa
from bench.tasks_scicode import UNSOLVED_AS_SPECIFIED

A, B = "ctxagent_live", "ctxagent_live_noctx"
CTX_TOOLS = ("compact_context", "curate_context", "pin", "recall", "context_report")
MEMORY_KINDS = ("fact", "ephemeral", "probe_fact", "probe_ephemeral", "rule")


def short(model: str) -> str:
    return model.split("/")[-1]


def open_problems_before(turns: list[dict], c: int) -> set[str]:
    opened, closed = set(), set()
    for t in turns:
        if t["turn"] >= c:
            break
        m = t.get("meta") or {}
        pid = m.get("problem_id")
        if not pid:
            continue
        if (m.get("n_steps") or 0) > 1:
            opened.add(pid)
            if m.get("step_index") == m.get("n_steps"):
                closed.add(pid)
    return opened - closed


def session_stats(r: dict) -> dict:
    turns = r.get("turns") or []
    ws = sa.main_wire(r)
    comps = analysis.compaction_turns(r)
    sc = [t["score"] for t in turns if t.get("score") is not None and not isinstance(t["score"], dict)]
    q = [t["score"] for t in turns if t.get("kind") == "question" and t.get("score") is not None]
    mem = [t["score"] for t in turns if t.get("kind") in MEMORY_KINDS and t.get("score") is not None]
    by_turn_end = {}
    for w in ws:
        by_turn_end[w["turn"]] = sa.ctx(w)  # last request of the turn wins
    occ = []
    midchain = 0
    for c in comps:
        before = by_turn_end.get(c - 1)
        if before is None:
            # first turn or missing: use the largest request context in that turn before the drop
            idx = [i for i, w in enumerate(ws) if w["turn"] == c]
            before = max((sa.ctx(ws[i]) for i in idx), default=None)
        if before is not None:
            occ.append(before)
        if open_problems_before(turns, c):
            midchain += 1
    # compaction-attributable candidates: failed solvable SciCode steps with a compaction between
    # the previous step of the same problem and this step
    last_step_turn: dict[str, int] = {}
    attributable, solvable_fails = [], []
    for t in turns:
        m = t.get("meta") or {}
        pid = m.get("problem_id")
        if not pid:
            continue
        step = str(m.get("step"))
        if t.get("kind") == "question" and t.get("score") is not None and t["score"] < 1 and step not in UNSOLVED_AS_SPECIFIED:
            solvable_fails.append(step)
            prev = last_step_turn.get(pid)
            if prev is not None and any(prev < c <= t["turn"] for c in comps):
                attributable.append(f"{step}@t{t['turn']}")
        last_step_turn[pid] = t["turn"]
    tc = defaultdict(int)
    for t in turns:
        for k, v in (t.get("tool_counts") or {}).items():
            tc[k] += v
    prompt_tokens = [(w.get("uncached") or 0) + (w.get("cache_write") or 0) + (w.get("cache_read") or 0) for w in ws]
    return dict(
        model=short(r["model"]), harness=r["harness"], task=r["task"], seed=r.get("seed"),
        success=statistics.mean(sc) if sc else None, q_acc=statistics.mean(q) if q else None, n_q=len(q),
        memory=statistics.mean(mem) if mem else None, n_mem=len(mem),
        billed=r.get("billed_cost") or 0.0, billed_discounted=r.get("billed_cost_discounted"),
        pricing=r.get("pricing_mode"), requests=len(ws),
        peak=max((sa.ctx(w) for w in ws), default=0),
        mean_prompt=statistics.mean(prompt_tokens) if prompt_tokens else 0,
        output=sum(w.get("output") or 0 for w in ws),
        n_comp=len(comps), comp_turns=comps, occupancy=occ, midchain=midchain,
        solvable_fails=solvable_fails, attributable=attributable,
        ctx_calls={k: tc[k] for k in CTX_TOOLS if tc.get(k)},
        tools=sum(tc.values()), wall_min=(r.get("wall_s") or 0) / 60,
        aborted=bool(r.get("aborted")), harness_error=bool(r.get("harness_error")),
        errors=sum(len(t.get("errors") or []) for t in turns),
    )


def fmt(v, f="{:.3f}"):
    return "-" if v is None else f.format(v)


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--extra", nargs="*", default=[], help="files whose ctxagent_live mixdep:34 rows are added as arm A")
    ap.add_argument("--per-session", action="store_true")
    args = ap.parse_args()
    rows, aside = analysis.load(args.paths)
    if args.extra:
        extra, _ = analysis.load(args.extra)
        have = {(r["model"], r["harness"], r["task"]) for r in rows}
        for r in extra:
            if r["harness"] == A and r["task"].startswith("mixdep:34") and (r["model"], r["harness"], r["task"]) not in have:
                rows.append(r)
    for r in aside:
        print("set aside:", r["tag"], r["_why"])
    stats = [session_stats(r) for r in rows if r["harness"] in (A, B) and r["task"].startswith("mixdep:34")]
    if args.per_session:
        for s in sorted(stats, key=lambda s: (s["model"], s["harness"], s["task"])):
            print(f"{s['model']:>22} {s['harness']:<20} {s['task']:<12} succ {fmt(s['success'])} q {fmt(s['q_acc'])} mem {fmt(s['memory'])} "
                  f"${s['billed']:.2f} peak {s['peak']//1000}k mean {s['mean_prompt']//1000:.0f}k out {s['output']//1000}k "
                  f"comps {s['n_comp']}@{s['comp_turns']} occ {[o//1000 for o in s['occupancy']]}k "
                  f"solvable fails {s['solvable_fails']} attributable {s['attributable']} "
                  f"ctx {s['ctx_calls']} tools {s['tools']} wall {s['wall_min']:.0f} err {s['errors']}"
                  + (" ABORTED" if s['aborted'] else "") + (" HARNESS_ERR" if s['harness_error'] else ""))
        print()
    g = defaultdict(list)
    for s in stats:
        g[(s["model"], s["harness"])].append(s)
    models = sorted({m for m, _ in g})
    hdr = ["model", "arm", "n", "success", "q_acc", "memory", "billed $", "peak k", "mean prompt k", "output k",
           "comps/sess", "occ@comp k [min-max]", "occ % of 1M", "solvable step fails (comp-attributable)",
           "ctx calls", "tools", "wall min", "errors"]
    print(" | ".join(hdr))
    print("|".join("---" for _ in hdr))
    summary = {}
    for m in models:
        for h in (A, B):
            ss = g.get((m, h))
            if not ss:
                continue
            occ = [o for s in ss for o in s["occupancy"]]
            ncomp = sum(s["n_comp"] for s in ss)
            ctx_calls = defaultdict(int)
            for s in ss:
                for k, v in s["ctx_calls"].items():
                    ctx_calls[k] += v
            row = dict(
                n=len(ss), success=mean_or_none([s["success"] for s in ss]), q_acc=mean_or_none([s["q_acc"] for s in ss]),
                memory=mean_or_none([s["memory"] for s in ss]), billed=statistics.mean([s["billed"] for s in ss]),
                peak=statistics.mean([s["peak"] for s in ss]), mean_prompt=statistics.mean([s["mean_prompt"] for s in ss]),
                output=statistics.mean([s["output"] for s in ss]), comps=ncomp / len(ss),
                occ_med=statistics.median(occ) if occ else None, occ_min=min(occ) if occ else None, occ_max=max(occ) if occ else None,
                mid=sum(s["midchain"] for s in ss), ctx_calls=dict(ctx_calls), tools=statistics.mean([s["tools"] for s in ss]),
                sfails=sum(len(s["solvable_fails"]) for s in ss), attrib=[a for s in ss for a in s["attributable"]],
                wall=statistics.mean([s["wall_min"] for s in ss]), errors=sum(s["errors"] for s in ss),
                aborted=sum(1 for s in ss if s["aborted"]),
            )
            summary[(m, h)] = row
            print(" | ".join([
                m, "A" if h == A else "B", str(row["n"]), fmt(row["success"]), fmt(row["q_acc"]), fmt(row["memory"]),
                f"{row['billed']:.2f}", f"{row['peak']/1000:.0f}", f"{row['mean_prompt']/1000:.0f}", f"{row['output']/1000:.0f}",
                f"{row['comps']:.1f}",
                "-" if row["occ_med"] is None else f"{row['occ_med']/1000:.0f} [{row['occ_min']/1000:.0f}-{row['occ_max']/1000:.0f}]",
                "-" if row["occ_med"] is None else f"{100*row['occ_med']/1_000_000:.1f}",
                f"{row['sfails']} ({len(row['attrib'])}: {', '.join(row['attrib']) if row['attrib'] else '-'})",
                str(row["ctx_calls"]) if row["ctx_calls"] else "-", f"{row['tools']:.0f}", f"{row['wall']:.0f}",
                str(row["errors"]) + (f" ({row['aborted']} aborted)" if row["aborted"] else ""),
            ]))
    print("\nA vs B per model (A minus B; cost ratio A/B at list rates):")
    for m in models:
        a, b = summary.get((m, A)), summary.get((m, B))
        if not a or not b:
            continue
        ds = (a["success"] - b["success"]) if a["success"] is not None and b["success"] is not None else None
        dm = (a["memory"] - b["memory"]) if a["memory"] is not None and b["memory"] is not None else None
        print(f"  {m:>22}: Δsuccess {fmt(ds, '{:+.3f}')}  Δmemory {fmt(dm, '{:+.3f}')}  cost A/B {a['billed']/b['billed']:.2f} "
              f"(${a['billed']:.2f} vs ${b['billed']:.2f})  peak A/B {a['peak']/b['peak']:.2f}  comps/sess {a['comps']:.1f}")


if __name__ == "__main__":
    main()
