"""Analysis of sequential Context-Bench sessions (cbseq:N + recap).

python -m bench.seq_analysis bench/results/seq100.jsonl [--fig-prefix bench/results/fig_seq100]

Per trial: accuracy, retention (recap), billed $, main-model requests, peak context,
compactions detected on the wire (same rule for every harness), cache hit rate.
Retention is split by whether a question was answered before/after the LAST compaction
(i.e. whether the answer survived only through a compaction summary).
"""
import argparse, json, statistics
from collections import defaultdict

DROP_FRAC, DROP_MIN = 0.25, 10_000  # compaction = main-model context falls >25% and >10k


def base(m):
    return m.replace("[1m]", "") if m else m


def main_wire(r):
    """Main-model requests in order (aux calls such as titles/summaries by other models excluded)."""
    return [w for w in r.get("wire") or [] if base(w.get("model")) == base(r["model"])]


def ctx(w):
    return (w.get("uncached") or 0) + (w.get("cache_write") or 0) + (w.get("cache_read") or 0)


def compactions(ws):
    """Indices i where request i's context dropped substantially vs request i-1."""
    out = []
    for i in range(1, len(ws)):
        a, b = ctx(ws[i - 1]), ctx(ws[i])
        if a - b > max(DROP_MIN, DROP_FRAC * a):
            out.append(i)
    return out


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    best = {}
    for r in rows:  # keep the last row per tag (reruns supersede)
        best[r["tag"]] = r
    from bench.analysis import normalize_cost_basis  # flex-tier sessions re-priced at list rates
    out = []
    for r in best.values():  # one plot key per (harness, window); seeds overlay
        r.setdefault("rep", 0)
        out.append(normalize_cost_basis(r))
    return out


def summarize(r):
    ws = main_wire(r)
    comp = compactions(ws)
    comp_turns = sorted({ws[i]["turn"] for i in comp})
    q = [t for t in r["turns"] if t["kind"] == "question" and not t.get("harness_error")]
    rec = next((t for t in r["turns"] if t["kind"] == "recap"), None)
    inp = sum(ctx(w) for w in r.get("wire") or [])
    read = sum(w.get("cache_read") or 0 for w in r.get("wire") or [])
    s = dict(harness=r["harness"], model=r["model"].split("/")[-1], window=r["window"],
             n_q=len(q), acc=statistics.mean(t["score"] for t in q) if q else None,
             retention=r.get("recap_score"), billed=r.get("billed_cost"),
             per_q=(r.get("billed_cost") or 0) / max(1, len(q)),
             main_req=len(ws), all_req=r.get("wire_requests"), peak=max((ctx(w) for w in ws), default=0),
             end=ctx(ws[-1]) if ws else 0, n_comp=len(comp), comp_turns=comp_turns,
             hit=read / inp if inp else None, out_tok=sum(w.get("output") or 0 for w in r.get("wire") or []),
             wall_min=(r.get("wall_s") or 0) / 60, harness_reported_comp=r.get("compaction_events"))
    # retention split around the last compaction before the recap turn
    if rec and rec.get("details") and rec["details"].get("per_question"):
        pq = rec["details"]["per_question"]
        last = max([t for t in comp_turns if t <= rec["turn"]], default=None)
        pre = [v for k, v in enumerate(pq) if last is not None and k + 1 < last]
        post = [v for k, v in enumerate(pq) if last is None or k + 1 >= last]
        s.update(ret_pre=statistics.mean(pre) if pre else None, n_pre=len(pre),
                 ret_post=statistics.mean(post) if post else None, n_post=len(post))
    return s


def fmt(v, f="{:.3f}"):
    return "-" if v is None else (f.format(v) if isinstance(v, float) else str(v))


def table(ss):
    cols = ["model", "harness", "window", "n_q", "acc", "retention", "ret_pre", "n_pre", "ret_post",
            "billed", "per_q", "main_req", "peak", "end", "n_comp", "hit", "out_tok", "wall_min"]
    print(" | ".join(cols))
    for s in sorted(ss, key=lambda s: (s["model"], str(s["window"]), s["harness"])):
        print(" | ".join(fmt(s.get(c)) for c in cols), "  comp@", s["comp_turns"])


def paired(rows):
    by = defaultdict(dict)
    for r in rows:
        for t in r["turns"]:
            if t["kind"] == "question" and not t.get("harness_error"):
                key = t["meta"].get("qid") or t["meta"].get("step") or t["meta"].get("exercise") or t["turn"]
                by[(r["model"], r["window"], key)][r["harness"]] = t["score"]
    wrong = defaultdict(list)
    for (m, w, qid), d in sorted(by.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        for h, sc in d.items():
            if sc < 1:
                wrong[(m.split("/")[-1], h)].append((w // 1000 if isinstance(w, int) else w, qid))
    print("\nwrong answers (window k, qid):")
    for k, v in sorted(wrong.items()):
        print(" ", k, len(v), v)


def plots(rows, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = sorted({r["model"] for r in rows})
    style = {100000: ":", 200000: "--", "native": "--", 1000000: "-"}
    color = {"ctxagent": "tab:blue", "ctxagent_live": "tab:blue", "claude_code": "tab:orange", "codex": "tab:green"}
    for kind in ("context", "billed"):
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
        for ax, m in zip(axes[0], models):
            for r in sorted((r for r in rows if r["model"] == m), key=lambda r: (r["harness"], str(r["window"]))):
                ws = main_wire(r)
                if kind == "context":
                    xs = [w["turn"] for w in ws]; ys = [ctx(w) / 1000 for w in ws]
                else:
                    acc, xs, ys = 0.0, [], []
                    for w in r.get("wire") or []:
                        acc += w.get("billed") or 0; xs.append(w["turn"]); ys.append(acc)
                ax.plot(xs, ys, style.get(r["window"], "-"), color=color.get(r["harness"], "k"), lw=1.2,
                        label=f"{r['harness']} {r['window']//1000}k" if isinstance(r['window'], int) else f"{r['harness']} native")
            ax.set_title(m.split("/")[-1]); ax.set_xlabel("turn (question #)")
            ax.set_ylabel("context tokens (k)" if kind == "context" else "cumulative billed $")
            ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{prefix}_{kind}.png", dpi=130); plt.close(fig)
        print("wrote", f"{prefix}_{kind}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--fig-prefix")
    a = ap.parse_args()
    rows = load(a.path)
    table([summarize(r) for r in rows])
    paired(rows)
    if a.fig_prefix:
        plots(rows, a.fig_prefix)
