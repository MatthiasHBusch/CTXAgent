"""Analysis for the redesigned benchmark (REDESIGN.md section 6).

  python -m bench.analysis bench/results/p2_seqdep.jsonl [more.jsonl ...] [--boot 10000] [--seed 0]
                           [--fig-prefix bench/results/fig_p2] [--include-web-use] [--include-errors]

The unit of randomisation is the session (one trial row), and turns inside a session are
correlated, so every confidence interval here is a cluster bootstrap over sessions.
Paired comparisons between arms use the same (model, task, rep, turn) in both arms:
exact McNemar for binary scores, a sign test for graded ones. Retention is P(probe
correct | distance, compaction between establish and probe), the compaction turns being
detected on the wire with `seq_analysis.compactions` (same rule for every harness).
Cost is reported as billed, billed without turn 1 (the cold prefix), marginal cost per
scored turn, and the prefix-amortised cost per task at a reference session length.

Headline numbers exclude trials flagged `web_use` and turns with harness errors; both
are listed separately. `seq_analysis.py` keeps the per-session table.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from statistics import mean

from bench import seq_analysis as sa

PROBE_KINDS = ("probe_fact", "probe_derived", "probe_ephemeral")
SCORED_KINDS = ("question", "ephemeral", "probe_fact", "probe_derived", "control_derived", "probe_ephemeral")
AMORTISE_N = 100  # reference session length for the prefix-amortised cost per task


# ------------------------------------------------------------------ loading
def load(paths, include_web_use: bool = False) -> tuple[list[dict], list[dict]]:
    """(rows used for headline numbers, rows set aside) -- last row per tag wins."""
    best: dict = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    best[r["tag"]] = r
    use, aside = [], []
    for r in best.values():
        if r.get("skipped"):
            aside.append({**r, "_why": f"skipped: {r['skipped']}"})
        elif r.get("web_use") and not include_web_use:
            aside.append({**r, "_why": f"web_use ({len(r['web_use'])} hits)"})
        else:
            use.append(normalize_cost_basis(r))
    return use, aside


FLEX_MODES = ("prefer_flex", "flex_only")
OPENAI_LONG_CONTEXT_TOKENS = 272_000  # OpenAI's long-context pricing tier starts here (2x prompt, 1.5x output)


def _is_openai(model: str) -> bool:
    return (model or "").split("/")[0] == "openai"


def normalize_cost_basis(r: dict) -> dict:
    """Every cost figure on ONE basis: OpenRouter list rates. Sessions run at a discounted
    service tier (`pricing_mode` prefer_flex / flex_only) carry what was billed in
    `billed_cost` / per-request `billed`; here those are replaced by the same tokens at list
    rates (`wire_list_cost` from the trial, per request via bench.pricing), so flex and
    standard sessions, and different models, compare on the price list. The discounted
    figures stay available as `billed_cost_discounted` / `billed_discounted`.

    Only requests that were actually served at the discounted tier are re-priced. The wire
    record has no tier field, so the discount is detected per request from the ratio of what
    was billed to the flat list price: OpenAI's flex tier is exactly 50 % of list, so a ratio
    in [0.4, 0.6] means flex and the list price is 2 x billed (this also keeps any long-prompt
    surcharge the provider applied). OpenAI bills prompts of >= 272k tokens at a long-context
    tier (2x input / cache, 1.5x output) on top of the flex discount, so such a request shows a
    ratio of 0.9-1.0 to the flat table while still being flex-served: it is re-priced to
    2 x billed as well and counted in `n_flex_longctx_requests` (observed in the Sol-B and
    Astra-B sessions of phase 5', all pinned to openai/flex without fallbacks). Below the
    threshold a ratio >= 0.9 means the request was billed at list (possibly above the flat
    table, e.g. Haiku 5.5's 5x tier above 100k prompt tokens) and the billed figure is kept.
    Anything else falls back to the flat table and is counted in `n_price_anomalies`.
    `n_flex_requests` records how many requests were re-priced."""
    if r.get("pricing_mode") not in FLEX_MODES or r.get("cost_basis") == "list":
        return r
    from bench import pricing
    r = dict(r)
    r["billed_cost_discounted"] = r.get("billed_cost")
    wire = []
    n_flex = n_anom = n_long = 0
    for w in r.get("wire") or []:
        w = dict(w)
        try:
            flat = pricing.cost(w.get("model") or r["model"], w["uncached"], w["cache_write"],
                                w["cache_read"], w["output"])
        except KeyError:
            wire.append(w)
            continue
        billed = w.get("billed")
        w["billed_discounted"] = billed
        if billed is None or flat <= 0:
            w["billed"] = flat
        else:
            ratio = billed / flat
            prompt = (w.get("uncached") or 0) + (w.get("cache_write") or 0) + (w.get("cache_read") or 0)
            if 0.4 <= ratio <= 0.6:
                w["billed"] = 2 * billed
                n_flex += 1
            elif 0.85 <= ratio <= 1.1 and prompt >= OPENAI_LONG_CONTEXT_TOKENS and _is_openai(w.get("model") or r["model"]):
                w["billed"] = 2 * billed
                n_flex += 1
                n_long += 1
            elif ratio >= 0.9:
                w["billed"] = billed
            else:
                w["billed"] = flat
                n_anom += 1
        wire.append(w)
    r["wire"] = wire
    r["n_flex_requests"], r["n_price_anomalies"], r["n_flex_longctx_requests"] = n_flex, n_anom, n_long
    if wire:
        r["billed_cost"] = round(sum(w.get("billed") or 0 for w in wire), 6)
        t1 = sum(w.get("billed") or 0 for w in wire if w.get("turn") == 1)
        r["billed_turn1"], r["billed_excl_turn1"] = round(t1, 6), round(r["billed_cost"] - t1, 6)
    elif r.get("wire_list_cost") is not None:
        r["billed_cost"] = r["wire_list_cost"]
    for t in r.get("turns") or []:
        t["billed"] = round(sum(w.get("billed") or 0 for w in wire if w.get("turn") == t.get("turn")), 6) if wire else t.get("billed")
    r["cost_basis"] = "list"
    return r


def arm(r: dict) -> tuple:
    return (r["model"].split("/")[-1], r["harness"], str(r.get("window")))


def scored_turns(r: dict, kinds=SCORED_KINDS) -> list[dict]:
    return [t for t in r["turns"] if t["kind"] in kinds and not t.get("harness_error")]


# --------------------------------------------------------------- statistics
def cluster_bootstrap(clusters: list[list[float]], n: int = 10_000, seed: int = 0,
                      stat=None) -> tuple[float | None, float | None, float | None]:
    """Point estimate and 95 % percentile CI of `stat` (default: pooled mean over all
    observations) when whole clusters (sessions) are resampled with replacement."""
    clusters = [c for c in clusters if c]
    if not clusters:
        return None, None, None
    stat = stat or (lambda cs: sum(sum(c) for c in cs) / sum(len(c) for c in cs))
    point = stat(clusters)
    if len(clusters) < 2:
        return point, None, None
    rng = random.Random(seed)
    vals = []
    k = len(clusters)
    for _ in range(n):
        vals.append(stat([clusters[rng.randrange(k)] for _ in range(k)]))
    vals.sort()
    return point, vals[int(0.025 * n)], vals[min(n - 1, int(0.975 * n))]


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value for k successes in n trials at p = 0.5."""
    if n == 0:
        return 1.0
    pk = math.comb(n, k) / 2 ** n
    return min(1.0, sum(math.comb(n, i) / 2 ** n for i in range(n + 1) if math.comb(n, i) / 2 ** n <= pk + 1e-12))


def mcnemar(pairs: list[tuple[float, float]]) -> dict:
    """Exact McNemar on binary outcomes (score >= 1 counts as success)."""
    b = sum(1 for x, y in pairs if x >= 1 > y)  # a right, b wrong
    c = sum(1 for x, y in pairs if y >= 1 > x)  # b right, a wrong
    return {"n": len(pairs), "a_only": b, "b_only": c, "p": binom_two_sided(min(b, c), b + c)}


def sign_test(pairs: list[tuple[float, float]]) -> dict:
    pos = sum(1 for x, y in pairs if y > x)
    neg = sum(1 for x, y in pairs if y < x)
    return {"n": len(pairs), "b_better": pos, "a_better": neg, "p": binom_two_sided(min(pos, neg), pos + neg)}


# ---------------------------------------------------------------- summaries
def success_by_kind(rows: list[dict], boot: int, seed: int) -> dict:
    """{arm: {kind: (n, point, lo, hi)}} with session-cluster CIs."""
    out: dict = defaultdict(dict)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for t in scored_turns(r):
            by[arm(r)][t["kind"]].append((r["tag"], t["score"]))
        rules = [v["ok"] for t in r["turns"] for v in (t.get("rule_adherence") or {}).values() if isinstance(v, dict)]
        if rules:
            by[arm(r)]["rule_adherence"] += [(r["tag"], float(v)) for v in rules]
    for a, kinds in by.items():
        for kind, obs in kinds.items():
            clusters = defaultdict(list)
            for tag, s in obs:
                clusters[tag].append(s)
            p, lo, hi = cluster_bootstrap(list(clusters.values()), boot, seed)
            out[a][kind] = (len(obs), p, lo, hi)
    return out


def paired(rows: list[dict]) -> dict:
    """Paired tests between every two arms of the same model on the same (task, rep, turn)."""
    cell = defaultdict(dict)  # (model, task, rep, turn, kind) -> {arm: score}
    for r in rows:
        for t in scored_turns(r):
            cell[(r["model"], r["task"], r.get("rep", 0), t["turn"], t["kind"])][arm(r)] = t["score"]
    pairs = defaultdict(list)
    for key, d in cell.items():
        arms = sorted(d)
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                pairs[(a, b, key[4])].append((d[a], d[b]))
    out = {}
    for (a, b, kind), ps in sorted(pairs.items()):
        binary = all(s in (0.0, 1.0) for p in ps for s in p)
        out[(a, b, kind)] = (mcnemar if binary else sign_test)(ps) | {"test": "mcnemar" if binary else "sign",
                                                                     "mean_a": mean(x for x, _ in ps),
                                                                     "mean_b": mean(y for _, y in ps)}
    return out


def compaction_turns(r: dict) -> list[int]:
    ws = sa.main_wire(r)
    return sorted({ws[i]["turn"] for i in sa.compactions(ws)})


def probe_records(rows: list[dict]) -> list[dict]:
    """One record per probe turn with its distance and whether a wire-detected compaction
    fell between the establishing turn and the probe."""
    out = []
    for r in rows:
        comp = compaction_turns(r)
        base_ok = {t["turn"]: t["score"] for t in r["turns"]}
        for t in r["turns"]:
            if t["kind"] not in PROBE_KINDS or t.get("harness_error"):
                continue
            e, d = t["meta"].get("establish_turn"), t["meta"].get("distance")
            if e is None or d is None:
                continue
            between = any(e < c <= t["turn"] for c in comp)
            out.append({"arm": arm(r), "tag": r["tag"], "kind": t["kind"], "distance": d, "between": between,
                        "score": t["score"], "turn": t["turn"], "establish": e,
                        "base_score": base_ok.get(e) if t["kind"] == "probe_derived" else None})
        for t in r["turns"]:  # rule adherence as a probe with its own distance
            for rid, v in (t.get("rule_adherence") or {}).items():
                if isinstance(v, dict) and "distance" in v:
                    e = t["turn"] - v["distance"]
                    out.append({"arm": arm(r), "tag": r["tag"], "kind": "rule_adherence", "distance": v["distance"],
                                "between": any(e < c <= t["turn"] for c in comp), "score": float(v["ok"]),
                                "turn": t["turn"], "establish": e, "base_score": None})
    return out


def retention_curve(rows: list[dict], boot: int, seed: int, distances=(3, 10, 25, 50, 90)) -> dict:
    """{(arm, distance, between): (n, point, lo, hi)}; rule adherence binned to the nearest
    distance level, the three probe kinds pooled (per-kind split in probe_records)."""
    recs = probe_records(rows)
    by = defaultdict(lambda: defaultdict(list))
    for x in recs:
        d = x["distance"] if x["kind"] != "rule_adherence" else min(distances, key=lambda v: abs(v - x["distance"]))
        by[(x["arm"], d, x["between"])][x["tag"]].append(x["score"])
    return {k: (sum(len(c) for c in cl.values()),) + cluster_bootstrap(list(cl.values()), boot, seed)
            for k, cl in by.items()}


def cost_anatomy(rows: list[dict], boot: int, seed: int) -> dict:
    """Per arm: billed cost (with / without turn 1), marginal and amortised cost per scored
    turn, token anatomy, cache hit rate, summary-model spend, warm starts, peaks > 200k."""
    out = {}
    by = defaultdict(list)
    for r in rows:
        by[arm(r)].append(r)
    for a, rs in by.items():
        n_sc = [len(scored_turns(r)) for r in rs]
        billed = [r.get("billed_cost") or 0 for r in rs]
        t1 = [sum(w.get("billed") or 0 for w in r.get("wire") or [] if w.get("turn") == 1) for r in rs]
        excl = [r["billed_excl_turn1"] if r.get("billed_excl_turn1") is not None else b - t
                for r, b, t in zip(rs, billed, t1)]
        marginal = [e / max(n - 1, 1) for e, n in zip(excl, n_sc)]
        solved = [sum(1 for t in scored_turns(r) if t["score"] >= 1) for r in rs]
        tok = {k: sum((r.get("wire_tokens") or {}).get(k, 0) for r in rs)
               for k in ("uncached", "cache_write", "cache_read", "output")}
        inp = tok["uncached"] + tok["cache_write"] + tok["cache_read"]
        main_base = lambda r: sa.base(r["model"])  # noqa: E731
        summary_spend = [sum(v for m, v in (r.get("wire_by_model") or {}).items() if sa.base(m) != main_base(r))
                         for r in rs]
        comp = [len(sa.compactions(sa.main_wire(r))) for r in rs]
        out[a] = {
            "sessions": len(rs), "scored_turns": sum(n_sc),
            "billed": cluster_bootstrap([[b] for b in billed], boot, seed),
            "billed_excl_turn1": cluster_bootstrap([[e] for e in excl], boot, seed),
            "billed_turn1": mean(t1) if t1 else None,
            "marginal_per_turn": cluster_bootstrap([[m] for m in marginal], boot, seed),
            "amortised_per_task": (mean(t1) / AMORTISE_N + mean(marginal)) if rs else None,
            "cost_per_solved": (sum(billed) / sum(solved)) if sum(solved) else None,
            "tokens": tok, "cache_hit": tok["cache_read"] / inp if inp else None,
            "summary_spend": mean(summary_spend) if rs else None,
            "warm_starts": sum(1 for r in rs if r.get("warm_start")),
            "peak_over_200k": sum(1 for r in rs if r.get("peak_over_200k")),
            "compactions_wire": mean(comp) if comp else None,
            "harness_errors": sum(1 for r in rs for t in r["turns"] if t.get("harness_error")),
            "aborted": sum(1 for r in rs if r.get("aborted")),
        }
    return out


# ---------------------------------------------------------------- reporting
def _ci(v) -> str:
    if v is None or v[0] is None:
        return "-"
    p, lo, hi = v[-3], v[-2], v[-1]
    s = f"{p:.3f}" if isinstance(p, float) else str(p)
    return s if lo is None else f"{s} [{lo:.3f}, {hi:.3f}]"


def report(rows: list[dict], aside: list[dict], boot: int, seed: int) -> None:
    print(f"{len(rows)} sessions in headline numbers, {len(aside)} set aside"
          + (": " + ", ".join(f"{r['tag']} ({r['_why']})" for r in aside) if aside else ""))
    print(f"\n== task success by turn kind (pooled mean, 95% cluster-bootstrap CI over sessions, n={boot}) ==")
    for a, kinds in sorted(success_by_kind(rows, boot, seed).items()):
        print(f"{' / '.join(a)}")
        for kind, (n, p, lo, hi) in sorted(kinds.items()):
            print(f"    {kind:18} n={n:4}  {_ci((p, lo, hi))}")
    print("\n== paired tests (same model, task, rep, turn; McNemar exact for binary, sign test otherwise) ==")
    for (a, b, kind), t in paired(rows).items():
        print(f"  {kind:16} {'/'.join(a[1:])} vs {'/'.join(b[1:])} [{a[0]}]: n={t['n']} "
              f"mean {t['mean_a']:.3f} vs {t['mean_b']:.3f}  "
              f"{'a_only' if t['test'] == 'mcnemar' else 'a_better'}={t.get('a_only', t.get('a_better'))} "
              f"{'b_only' if t['test'] == 'mcnemar' else 'b_better'}={t.get('b_only', t.get('b_better'))} p={t['p']:.3f}")
    print("\n== retention: P(probe correct | distance, compaction between) ==")
    rc = retention_curve(rows, boot, seed)
    for (a, d, between), v in sorted(rc.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        print(f"  {' / '.join(a):50} d={d:3} {'comp' if between else 'none':5} n={v[0]:4}  {_ci(v)}")
    print("\n== cost anatomy ==")
    for a, c in sorted(cost_anatomy(rows, boot, seed).items()):
        print(f"{' / '.join(a)}: sessions={c['sessions']} scored_turns={c['scored_turns']}")
        print(f"    billed {_ci(c['billed'])}  excl_turn1 {_ci(c['billed_excl_turn1'])}  "
              f"turn1 {c['billed_turn1']:.4f}" if c['billed_turn1'] is not None else "    billed -")
        print(f"    marginal/turn {_ci(c['marginal_per_turn'])}  amortised/task@{AMORTISE_N} "
              f"{c['amortised_per_task']:.4f}  per solved {c['cost_per_solved'] if c['cost_per_solved'] is None else round(c['cost_per_solved'], 4)}")
        t = c["tokens"]
        print(f"    tokens uncached={t['uncached']} cache_write={t['cache_write']} cache_read={t['cache_read']} "
              f"output={t['output']}  cache_hit={c['cache_hit'] if c['cache_hit'] is None else round(c['cache_hit'], 3)}  "
              f"summary_spend/session={c['summary_spend'] if c['summary_spend'] is None else round(c['summary_spend'], 4)}")
        print(f"    warm_starts={c['warm_starts']} peak>200k={c['peak_over_200k']} compactions(wire)/session="
              f"{c['compactions_wire']} harness_errors={c['harness_errors']} aborted={c['aborted']}")


def plots(rows: list[dict], prefix: str, boot: int, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rc = retention_curve(rows, boot, seed)
    arms = sorted({k[0] for k in rc})
    models = sorted({a[0] for a in arms})
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
    for ax, m in zip(axes[0], models):
        for a in [a for a in arms if a[0] == m]:
            for between, ls in ((False, "-"), (True, "--")):
                pts = sorted((k[1], v) for k, v in rc.items() if k[0] == a and k[2] == between and v[1] is not None)
                if not pts:
                    continue
                xs = [d for d, _ in pts]
                ys = [v[1] for _, v in pts]
                ax.plot(xs, ys, ls, marker="o", label=f"{a[1]} {a[2]} {'comp between' if between else 'no comp'}")
                if all(v[2] is not None for _, v in pts):
                    ax.fill_between(xs, [v[2] for _, v in pts], [v[3] for _, v in pts], alpha=.12)
        ax.set_title(m)
        ax.set_xlabel("dependency distance (turns)")
        ax.set_ylabel("P(probe correct)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{prefix}_retention.png", dpi=130)
    plt.close(fig)
    print("wrote", f"{prefix}_retention.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig-prefix")
    ap.add_argument("--include-web-use", action="store_true")
    a = ap.parse_args()
    rows, aside = load(a.paths, a.include_web_use)
    report(rows, aside, a.boot, a.seed)
    if a.fig_prefix:
        plots(rows, a.fig_prefix, a.boot, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
