"""Freeze the configuration of a live ctxagent chat as the benchmark arm `ctxagent_live`.

python -m bench.snapshot_config <chat_id> [--db .ctxagent/ctxagent.db] [--out bench/configs/ctxagent_live.json]

Only machine-specific fields are dropped (workdirs). At trial time the benchmark sets
exactly three fields: model (the model under test), context_window (--windows), workdirs.
"""
import argparse, json, sqlite3, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("chat_id")
    ap.add_argument("--db", default=str(ROOT / ".ctxagent" / "ctxagent.db"))
    ap.add_argument("--out", default=str(ROOT / "bench" / "configs" / "ctxagent_live.json"))
    a = ap.parse_args()
    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    row = c.execute("select config from chats where id=?", (a.chat_id,)).fetchone()
    if not row:
        raise SystemExit(f"chat {a.chat_id} not found")
    cfg = json.loads(row[0])
    cfg.pop("workdirs", None)
    out = {"_source": {"chat_id": a.chat_id, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "note": "set at trial time: model, context_window, workdirs"},
           "config": cfg}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
