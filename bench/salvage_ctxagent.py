"""Rebuild the result row of a ctxagent trial whose sessions completed but whose process
was stopped before scoring (e.g. cancelled during final judging).

python -m bench.salvage_ctxagent <trial_dir> --model M --window W --task cbseq:100@0 --recap
       --out results.jsonl [--comp k=v ...] [--wall-s S]

Answers come from the trial's ctxagent DB (live segments + archived blobs, ordered by seq);
the usage trace from ctxagent's per-request ledger; wire/billed data from proxy.jsonl.
Scoring and row layout are exactly those of a normal trial (run.finish_trial). The row is
marked "salvaged": true. The session itself is not re-run.
"""
import argparse, json, sqlite3, tempfile
from pathlib import Path

from bench import run


def load_segments(tdir: Path) -> list[dict]:
    db = tdir / "ctxdata" / "ctxagent.db"
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = c.execute("select id, conversation from chats").fetchall()
    if len(rows) != 1:
        raise SystemExit(f"expected 1 chat, found {len(rows)}")
    chat_id, conv = rows[0][0], json.loads(rows[0][1])
    by_id = {}
    for sid, blob in (conv.get("archive") or {}).items():
        p = tdir / "ctxdata" / "blobs" / f"{blob}.txt"
        if p.exists():
            seg = json.loads(p.read_text(encoding="utf-8"))
            by_id[seg["id"]] = seg
    for seg in conv["segments"]:
        by_id.setdefault(seg["id"], seg)  # archived copy = pre-compaction original
    return chat_id, sorted(by_id.values(), key=lambda s: s["seq"])


def rebuild_turns(tdir: Path, prompts: list[str]) -> list[dict]:
    import re
    chat_id, segs = load_segments(tdir)
    starts = [s for s in segs if s["kind"] == "user"]  # benchmark prompts are the only user turns
    norm = lambda t: re.sub(r"\S*[\\/]data\b", "<DATA>", t)  # data path differs (temp dir)
    for k, s in enumerate(starts):
        if k >= len(prompts) or norm(s["body"]) != norm(prompts[k]):
            raise SystemExit(f"user turn {k + 1} does not match the expected prompt: {s['body'][:80]!r}")
    ledger = []
    for f in sorted((tdir / "ctxdata" / "usage").glob("*.jsonl")):
        ledger += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    turns = []
    for k, s in enumerate(starts):
        nxt = starts[k + 1]["seq"] if k + 1 < len(starts) else float("inf")
        t_end = starts[k + 1]["created_at"] if k + 1 < len(starts) else float("inf")
        group = [g for g in segs if s["seq"] <= g["seq"] < nxt]
        texts = [g.get("original_body") or g["body"] for g in group if g["kind"] == "assistant_text"]
        st = run._new_turn()
        st["chat_id"] = chat_id
        st["final"] = texts[-1] if texts else ""
        st["finish"] = "stop" if texts else None
        st["tools"] = [g.get("tool_name") for g in group if g["kind"] == "tool_result"]
        st["trace"] = [run._ctx_trace_entry(r) for r in ledger if s["created_at"] <= r["timestamp"] < t_end]
        turns.append(st)
    return turns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tdir"); ap.add_argument("--model", required=True); ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--task", required=True); ap.add_argument("--recap", action="store_true")
    ap.add_argument("--out", required=True); ap.add_argument("--comp", action="append", default=[])
    ap.add_argument("--harness", default="ctxagent"); ap.add_argument("--reasoning", default="high")
    ap.add_argument("--max-iters", type=int, default=100); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rep", type=int, default=0); ap.add_argument("--wall-s", type=float, default=0.0)
    ap.add_argument("--allow-partial", action="store_true", help="score the turns that exist (e.g. no recap)")
    a = ap.parse_args()
    import os
    os.environ.update(run._dotenv_keys())  # judge key (as in trial_main)
    tdir = Path(a.tdir)
    a.tag = tdir.name
    with tempfile.TemporaryDirectory() as tmp:  # task specs (checks, meta); prompts embed the real data dir
        _, spec = run.setup_task(a, Path(tmp))
        prompts = [t["prompt"] for t in spec]
        turns = rebuild_turns(tdir, prompts)
        partial = len(turns) < len(spec)
        if partial:
            if not a.allow_partial:
                raise SystemExit(f"{len(turns)} turns in DB, {len(spec)} expected (use --allow-partial)")
            spec = spec[:len(turns)]
        if not a.wall_s:
            from bench.proxy import read_log
            ts = [r.get("ts") for r in read_log(tdir / "proxy.jsonl") if r.get("ts")]
            a.wall_s = (max(ts) - min(ts)) if ts else 0.0
        out = Path(a.out)
        tmp_out = Path(tmp) / "row.jsonl"
        a.out = str(tmp_out)
        run.finish_trial(a, tdir, spec, turns, a.wall_s)
        row = json.loads(tmp_out.read_text(encoding="utf-8").splitlines()[-1])
    row["salvaged"] = True
    row["salvage_partial"] = partial
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
