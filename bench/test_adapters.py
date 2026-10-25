"""Offline tests for the OpenCode / Letta Code / mini-swe-agent adapters. No model calls, no cost.

    python -m bench.test_adapters parsers          # stream parsers on recorded samples (bench/test_data/adapters)
    python -m bench.test_adapters e2e [harness..] [--window N]  # full 3-turn session per adapter against bench.fake_llm,
                                                   # through the real UsageProxy and finish_trial

Recorded samples: opencode_* were recorded from opencode 1.19.3 against OpenRouter
(claude-haiku-5.5; a 2-turn resume probe, a tool-use turn and a turn with two auto-compactions
at a 30k limit); letta_* from Letta Code 0.34.15 against bench.fake_llm (3 turns of one
conversation, Anthropic and OpenAI wire formats).

The e2e mode runs the real harness binaries; bench.fake_llm answers every request (tool call on a
fresh user message, then "FAKE ANSWER user_messages=<n>"), so a correct continuation shows
n = 1, 2, 3 in the three finals, and every proxied request carries a fake billed cost.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "test_data" / "adapters"
WEB = {"webfetch", "websearch", "WebFetch", "WebSearch", "web_search", "fetch_webpage", "WatchPR"}


def _turn():
    from bench.run import _new_turn
    return _new_turn()


# ------------------------------------------------------------------ parsers
def test_opencode_resume_stream():
    from bench.harness_opencode import parse_run_stream
    st1, st2 = _turn(), _turn()
    sid1 = parse_run_stream((DATA / "opencode_resume_t1.jsonl").read_text(encoding="utf-8"), st1)
    sid2 = parse_run_stream((DATA / "opencode_resume_t2.jsonl").read_text(encoding="utf-8"), st2)
    assert sid1 and sid1 == sid2, (sid1, sid2)  # -s <sid> continued the same session
    assert st1["final"] == "OK" and st2["final"] == "pelican", (st1["final"], st2["final"])
    assert st1["finish"] == "stop" and not st1["errors"]


def test_opencode_tool_stream():
    from bench.harness_opencode import parse_run_stream
    st = _turn()
    parse_run_stream((DATA / "opencode_tools.jsonl").read_text(encoding="utf-8"), st)
    assert st["tools"] == ["read", "glob", "read"], st["tools"]
    assert st["final"] == "5", st["final"]  # only the LAST message's text, not the interim remark


def test_opencode_compaction_export():
    from bench.harness_opencode import n_compactions, parse_run_stream, usage_rows
    st = _turn()
    parse_run_stream((DATA / "opencode_compaction_t1.jsonl").read_text(encoding="utf-8"), st)
    assert "Continue if you have next steps" not in st["final"]  # synthetic parts are not answers
    exp = json.loads((DATA / "opencode_compaction_export.json").read_text(encoding="utf-8"))
    msgs = exp["messages"]
    rows = usage_rows(msgs, "anthropic/claude-haiku-5.5")
    assert n_compactions(msgs) == 2
    assert [r["source"] for r in rows].count("compaction") == 2 and len(rows) == 9, [r["source"] for r in rows]
    # same request count as the run stream's step_finish events, but only the export labels which
    # of them were compaction requests
    n_steps = sum(1 for l in (DATA / "opencode_compaction_t1.jsonl").read_text(encoding="utf-8").splitlines()
                  if json.loads(l).get("type") == "step_finish")
    assert n_steps == len(rows), n_steps
    first = rows[0]
    assert first["uncached"] == 4 and first["cache_write"] == 13060 and first["output"] == 211 + 121, first


def _letta(name):
    from bench.harness_letta import parse_stream
    out = []
    for k in (1, 2, 3):
        st = _turn()
        info = parse_stream((DATA / f"letta_{name}_turn{k}.jsonl").read_text(encoding="utf-8"), st)
        out.append((st, info))
    return out


def test_letta_streams():
    for name in ("anthropic", "openai"):
        res = _letta(name)
        convs = {info["conversation_id"] for _, info in res}
        assert len(convs) == 1, convs  # all three turns in one conversation
        for k, (st, info) in enumerate(res, 1):
            assert st["final"] == f"FAKE ANSWER user_messages={k}", (name, k, st["final"])
            assert st["tools"] == ["Bash"] and len(st["trace"]) == 2 and not st["errors"], (name, k, st)
            assert st["finish"] == "end_turn"
            assert not (set(info["tools"]) & WEB), info["tools"]


def run_parsers() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    return 0


# ---------------------------------------------------------------------- e2e
def e2e(harness: str, model: str, out: Path, window: int = 0) -> dict:
    """One 3-turn session through the real adapter, the real UsageProxy and finish_trial."""
    from bench import run as R
    from bench.fake_llm import FakeLLM
    from bench.proxy import UsageProxy
    a = argparse.Namespace(model=model, harness=harness, task="offline3", rep=0, seed=7, tag=f"offline_{harness}",
                           window=window, reasoning="high", max_out=32000, max_iters=20, max_cost=1.0, timeout=300,
                           cc_small_model="anthropic/claude-haiku-4.5", temperature=1.0, comp=[], recap=False,
                           out=str(out), summary_model="")
    tdir = R.WORK / a.tag
    shutil.rmtree(tdir, ignore_errors=True)
    workdir = tdir / "data"
    workdir.mkdir(parents=True)
    (workdir / "notes.txt").write_text("the answer is 42\n", encoding="utf-8")
    prompts = [f"Q{k}: read notes.txt and answer." for k in (1, 2, 3)]
    spec = [{"prompt": p, "kind": "question", "meta": {},
             "check": (lambda k: lambda f: (1.0 if f.strip() == f"FAKE ANSWER user_messages={k}" else 0.0,
                                            {"final": f}))(k)} for k, p in enumerate(prompts, 1)]
    os.environ.setdefault("OPENROUTER_API_KEY", "sk-offline-dummy")
    with FakeLLM() as fake:
        proxy = UsageProxy(tdir / "proxy.jsonl", upstream=fake.url)
        proxy.start()
        try:
            fn = {"opencode": "bench.harness_opencode:run_opencode", "letta": "bench.harness_letta:run_letta",
                  "miniswe": "bench.harness_miniswe:run_miniswe"}[harness]
            mod, name = fn.split(":")
            turns = getattr(__import__(mod, fromlist=[name]), name)(a, workdir, prompts, tdir, proxy)
        finally:
            proxy.stop()
    R.finish_trial(a, tdir, spec, turns, 0.0)
    return json.loads(out.read_text(encoding="utf-8").splitlines()[-1])


E2E_MODELS = {"opencode": "anthropic/claude-haiku-5.5", "letta": "anthropic/claude-haiku-4.5",
              "miniswe": "anthropic/claude-haiku-5.5"}


def check_e2e(res: dict) -> list[str]:
    bad = []
    if res["n_turns"] != 3 or res["completed_turns"] != 3:
        bad.append(f"turns {res['completed_turns']}/{res['n_turns']}")
    for t in res["turns"]:
        if t["score"] != 1.0:
            bad.append(f"turn {t['turn']} final {t['details']}")
        if not t["wire_requests"]:
            bad.append(f"turn {t['turn']}: no proxied request")
    if res["billed_missing"] or not res["billed_cost"]:
        bad.append(f"billed_missing={res['billed_missing']} billed={res['billed_cost']}")
    if res["wire_errors"]:
        bad.append(f"wire_errors={res['wire_errors']}")
    if set(res["wire_tools"]) & WEB:
        bad.append(f"web tool on the wire: {set(res['wire_tools']) & WEB}")
    if not res["tools_in_force"] or not res["harness_flags"]:
        bad.append("tools_in_force / harness_flags not recorded")
    return bad


def run_e2e(harnesses: list[str], window: int = 0) -> int:
    out = Path(tempfile.gettempdir()) / "bench_offline_adapters.jsonl"
    out.unlink(missing_ok=True)
    rc = 0
    for h in harnesses:
        res = e2e(h, E2E_MODELS[h], out, window)
        bad = check_e2e(res)
        summary = {k: res[k] for k in ("harness", "harness_version", "completed_turns", "score", "wire_requests",
                                       "billed_cost", "billed_missing", "wire_errors", "wire_tools", "wire_effort",
                                       "tools_in_force", "compaction_events")}
        summary["finals"] = [t["final_tail"] for t in res["turns"]]
        print(json.dumps(summary, ensure_ascii=False))
        print(("FAIL " + "; ".join(bad)) if bad else f"ok  e2e {h}", flush=True)
        rc |= bool(bad)
    print(f"rows: {out}")
    return rc


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["parsers", "e2e"])
    ap.add_argument("harness", nargs="*", default=["opencode", "letta", "miniswe"])
    ap.add_argument("--window", type=int, default=0, help="0 = harness-native window")
    a = ap.parse_args()
    return run_parsers() if a.mode == "parsers" else run_e2e(a.harness, a.window)


if __name__ == "__main__":
    sys.exit(main())
