"""Tests for the compaction engine's invariants and its headline behaviours.

Run:  python -m tests.test_compaction
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ctxagent.config import CompactionConfig
from ctxagent.compaction import CompactionEngine
from ctxagent.compaction.engine import dynamic_tail_ratio, _snap_cut
from ctxagent.segments import (
    Conversation,
    Segment,
    K_SYSTEM,
    K_USER,
    K_ASSISTANT,
    K_TOOL_CALL,
    K_TOOL_RESULT,
    S_SUPERSEDED,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))


async def fake_complete(*, model, messages, max_tokens=2000, temperature=0.0, response_format=None):
    """Stand-in summariser: no network."""
    if response_format:
        import json
        import re

        ids = re.findall(r"### id=(\w+)", messages[-1]["content"])
        return json.dumps({"items": [{"id": i, "text": f"condensed {i}"} for i in ids]})
    return "## 1. User intent\nBuild the thing.\n## 9. Next step\nContinue."


# ---------------------------------------------------------------------------

def build_conv(n_reads: int = 3, big_lines: int = 600, repeated: bool = False) -> Conversation:
    """read(doc) -> edit(doc) -> read(doc) -> edit(doc) -> read(doc)"""
    conv = Conversation()
    conv.new("system", K_SYSTEM, "You are an agent.", pinned=True)
    conv.new("user", K_USER, "Please revise report.md a few times.")
    doc = "file:c:/proj/report.md"
    for i in range(n_reads):
        cid = f"call_read_{i}"
        conv.new("assistant", K_TOOL_CALL, "", tool_calls=[
            {"id": cid, "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "report.md"}'}}])
        body = "\n".join(f"{j:6d}\tline {j} of the report version {0 if repeated else i}" for j in range(big_lines))
        conv.new("tool", K_TOOL_RESULT, body, tool_call_id=cid,
                 tool_name="read_file", resource_key=doc,
                 observation={"version": str(0 if repeated else i), "representation": "source",
                              "coverage": {"kind": "whole"}, "complete": True})
        if i < n_reads - 1 and not repeated:
            eid = f"call_edit_{i}"
            conv.new("assistant", K_TOOL_CALL, "", tool_calls=[
                {"id": eid, "type": "function",
                 "function": {"name": "edit_file", "arguments": '{"path": "report.md"}'}}])
            conv.new("tool", K_TOOL_RESULT, f"Edited report.md: 1 replacement.",
                     tool_call_id=eid, tool_name="edit_file", resource_key=doc, mutates=True)
    conv.new("assistant", K_ASSISTANT, "Done, the report is revised.")
    return conv


def pairing_ok(conv: Conversation) -> bool:
    """Every tool message must follow an assistant message that declared its id."""
    declared: set[str] = set()
    for s in conv.segments:
        if s.kind == K_TOOL_CALL:
            for tc in s.tool_calls or []:
                declared.add(tc.get("id"))
        elif s.kind == K_TOOL_RESULT:
            if s.tool_call_id not in declared:
                return False
    # And every declared call must have a result.
    results = {s.tool_call_id for s in conv.segments if s.kind == K_TOOL_RESULT}
    return declared <= results


# ---------------------------------------------------------------------------

async def test_supersession():
    print("\n[1] supersession of re-read files")
    conv = build_conv(repeated=True)
    cfg = CompactionConfig(rewrite_interval_batches=1, rewrite_cost_gate=False,  # isolate the tier from the cache-cost gate 
        hot_zone_tokens=10_000_000,     # everything hot, isolate the dedup pass
        masking_enabled=False, significance_enabled=False, deep_enabled=False,
        protected_recent_observations=1,
    )
    eng = CompactionEngine("t", cfg, fake_complete)
    before = conv.total_tokens()
    rep = await eng.run(conv, 200_000)
    after = conv.total_tokens()

    reads = [s for s in conv.segments if s.tool_name == "read_file"]
    check("three reads present", len(reads) == 3)
    check("older two superseded", all(s.state == S_SUPERSEDED for s in reads[:-1]),
          f"states={[s.state for s in reads]}")
    check("newest read untouched", reads[-1].state == "original")
    check("stub names the cause",
          "recall" in reads[0].body and "duplicate" in reads[0].note, reads[0].body[:120])
    check("substantial saving", after < before * 0.45, f"{before} -> {after}")
    check("tool pairing intact", pairing_ok(conv))
    check("original recoverable", reads[0].original_body is not None and
          "line 5 of the report version 0" in reads[0].original_body)
    print(f"       {before} -> {after} tokens ({100*(1-after/before):.0f}% removed)")


async def test_frozen_prefix_immutable():
    print("\n[2] frozen prefix is never rewritten (cache safety)")
    conv = build_conv(n_reads=4, big_lines=400)
    cfg = CompactionConfig(rewrite_interval_batches=1, hot_zone_tokens=6_000, deep_enabled=False,
                           significance_enabled=False, protected_recent_observations=1)
    eng = CompactionEngine("t", cfg, fake_complete)
    await eng.run(conv, 200_000)

    hot_start = conv.frozen_upto
    check("a prefix was frozen", hot_start > 0, f"frozen_upto={hot_start}")
    frozen_before = [(s.id, s.body) for s in conv.segments[:hot_start]]

    # More turns, more compaction passes.
    for k in range(3):
        cid = f"c{k}"
        conv.new("assistant", K_TOOL_CALL, "", tool_calls=[
            {"id": cid, "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"report.md"}'}}])
        conv.new("tool", K_TOOL_RESULT,
                 "\n".join(f"{j}\tmore content line {j}" for j in range(400)),
                 tool_call_id=cid, tool_name="read_file", resource_key="file:c:/proj/report.md")
        await eng.run(conv, 200_000)

    now = {s.id: s.body for s in conv.segments}
    unchanged = all(now.get(sid) == body for sid, body in frozen_before)
    check("frozen bodies byte-identical after 3 more passes", unchanged)
    check("frozen boundary is monotone", conv.frozen_upto >= hot_start)
    check("all frozen segments flagged", all(s.frozen for s in conv.segments[:hot_start]))
    check("tool pairing intact", pairing_ok(conv))


async def test_masking_tiers():
    print("\n[3] age-tiered masking")
    conv = Conversation()
    conv.new("system", K_SYSTEM, "sys", pinned=True)
    conv.new("user", K_USER, "go")
    for i in range(10):
        cid = f"m{i}"
        conv.new("assistant", K_TOOL_CALL, "", tool_calls=[
            {"id": cid, "type": "function",
             "function": {"name": "run_command", "arguments": '{"command":"ls"}'}}])
        conv.new("tool", K_TOOL_RESULT,
                 "\n".join(f"output line {j} of run {i}" for j in range(300)),
                 tool_call_id=cid, tool_name="run_command", resource_key=f"cmd:{i}")
    cfg = CompactionConfig(rewrite_interval_batches=1, rewrite_cost_gate=False,  # isolate the tier from the cache-cost gate hot_zone_tokens=10_000_000, dedup_enabled=False,
                           significance_enabled=False, deep_enabled=False,
                           protected_recent_observations=3, protected_recent_batches=0,
                           offload_enabled=False, mask_tiers=(800, 400, 200, 100), mask_floor_tokens=60)
    eng = CompactionEngine("t", cfg, fake_complete)
    before = conv.total_tokens()
    await eng.run(conv, 200_000)
    obs = [s for s in conv.segments if s.kind == K_TOOL_RESULT]
    check("3 newest untouched", all(s.state == "original" for s in obs[-3:]),
          f"{[s.state for s in obs]}")
    check("older ones masked", all(s.state == "masked" for s in obs[:-3]))
    sizes = [s.tokens() for s in obs[:-3]]
    check("budgets shrink with age", sizes == sorted(sizes, reverse=True) or
          all(a >= b - 60 for a, b in zip(reversed(sizes), list(reversed(sizes))[1:])),
          str(sizes))
    print(f"       {before} -> {conv.total_tokens()} tokens; tier sizes {sizes}")


async def test_dynamic_tail():
    print("\n[4] dynamic tail ratio")
    cfg = CompactionConfig(rewrite_interval_batches=1, deep_trigger_ratio=0.75, tail_ratio_max=0.25,
                           tail_ratio_min=0.10, max_pressure=0.95)
    at_trigger = dynamic_tail_ratio(0.75, cfg)
    mid = dynamic_tail_ratio(0.85, cfg)
    at_max = dynamic_tail_ratio(0.95, cfg)
    beyond = dynamic_tail_ratio(1.30, cfg)
    check("keeps 25% at the trigger", abs(at_trigger - 0.25) < 1e-9, str(at_trigger))
    check("keeps 10% at max pressure", abs(at_max - 0.10) < 1e-9, str(at_max))
    check("monotone decreasing", at_trigger > mid > at_max, f"{at_trigger},{mid},{at_max}")
    check("clamped beyond max", abs(beyond - 0.10) < 1e-9, str(beyond))
    check("never compacts everything", at_max > 0)
    print(f"       compacted fraction: {1-at_trigger:.0%} → {1-at_max:.0%}")


async def test_deep_compaction():
    print("\n[5] deep compaction")
    conv = build_conv(n_reads=6, big_lines=500)
    for i in range(4):
        conv.new("user", K_USER, f"Follow-up question {i}")
        conv.new("assistant", K_ASSISTANT, f"Answer {i}. " + "detail " * 200)
    cfg = CompactionConfig(rewrite_interval_batches=1, hot_zone_tokens=8_000, deep_trigger_ratio=0.05,
                           deep_min_gain_tokens=100, deep_cooldown_turns=0,
                           significance_enabled=False)
    eng = CompactionEngine("t", cfg, fake_complete)
    before = conv.total_tokens()
    n_before = len(conv.segments)
    rep = await eng.run(conv, 60_000)
    deep = next((t for t in rep["tiers"] if t["tier"] == "3-deep"), None)

    check("deep pass ran", deep is not None and not deep.get("error"), str(rep["tiers"]))
    check("context shrank", conv.total_tokens() < before, f"{before} -> {conv.total_tokens()}")
    check("summary segment inserted",
          any(s.kind == "summary" for s in conv.segments))
    check("system message survived at head", conv.segments[0].kind == K_SYSTEM)
    check("a verbatim tail survived", deep and deep["kept_verbatim"] > 0)
    check("tool pairing intact after deletion", pairing_ok(conv),
          "orphaned tool result after cut")
    check("cooldown reset", eng.turns_since_deep == 0)

    # A second immediate run must NOT deep-compact again (hysteresis).
    cfg.deep_cooldown_turns = 6
    rep2 = await eng.run(conv, 60_000)
    check("cooldown blocks an immediate second deep pass",
          not any(t["tier"] == "3-deep" for t in rep2["tiers"]))
    print(f"       {n_before} segs / {before} tok -> {len(conv.segments)} segs / {conv.total_tokens()} tok")


async def test_snap_cut():
    print("\n[6] cut-point snapping")
    conv = build_conv(n_reads=3, big_lines=50)
    for idx in range(len(conv.segments)):
        cut = _snap_cut(conv, idx)
        tail = conv.segments[cut:]
        # No tail may begin with a tool result whose call is behind the cut.
        ok = not (tail and tail[0].kind == K_TOOL_RESULT)
        if not ok:
            check(f"cut at {idx} is safe", False, f"snapped to {cut}, orphan result")
            return
    check("every cut point snaps away from orphaned tool results", True)
    # A tool may write other segments while its batch is in flight (document_close appends a
    # user-kind handoff note before its own result). A cut proposed at that note must move
    # past every result of the call before it; this once left an orphan result and the
    # provider rejected every later request (Astra, mixed seed 2, 2026-10-24).
    conv = Conversation()
    conv.new("system", K_SYSTEM, "You are an agent.", pinned=True)
    conv.new("user", K_USER, "Close the document and compact.")
    conv.new("assistant", K_TOOL_CALL, "", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "document_close", "arguments": "{}"}},
        {"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}])
    conv.new("user", K_USER, "Document handoff for report.md (close note, not file evidence): done.")
    conv.new("tool", K_TOOL_RESULT, "Closed report.md", tool_call_id="c1", tool_name="document_close")
    conv.new("tool", K_TOOL_RESULT, "file body", tool_call_id="c2", tool_name="read_file")
    conv.new("assistant", K_ASSISTANT, "ok")
    note_idx = 3
    cut = _snap_cut(conv, note_idx)
    check("a cut at an interleaved handoff note moves past the batch's results", cut == 6, f"cut={cut}")
    msgs = conv.segments[cut:]
    check("no tool result in the tail without its call", not any(x.kind == K_TOOL_RESULT for x in msgs))
    # Second safety net: an orphan result is sent as a user note, not as a tool message.
    conv2 = Conversation()
    conv2.new("system", K_SYSTEM, "You are an agent.", pinned=True)
    conv2.new("tool", K_TOOL_RESULT, "Closed report.md", tool_call_id="gone", tool_name="document_close")
    roles = [m["role"] for m in conv2.to_messages()]
    check("orphan tool result is rendered as a user note", roles == ["system", "user"], str(roles))


async def test_recall_pins():
    print("\n[7] recall pins against re-compaction")
    conv = build_conv(n_reads=3, big_lines=600, repeated=True)
    cfg = CompactionConfig(rewrite_interval_batches=1, rewrite_cost_gate=False,  # isolate the tier from the cache-cost gate hot_zone_tokens=10_000_000, masking_enabled=False,
                           significance_enabled=False, deep_enabled=False,
                           protected_recent_observations=1)
    eng = CompactionEngine("t", cfg, fake_complete)
    await eng.run(conv, 200_000)
    stale = [s for s in conv.segments if s.tool_name == "read_file"][0]
    check("stale read was collapsed", stale.state == S_SUPERSEDED)
    stale.restore()
    restored_tokens = stale.tokens()
    check("restore brings the body back", restored_tokens > 1000, str(restored_tokens))
    check("restored segment is pinned", stale.pinned)
    await eng.run(conv, 200_000)
    check("a second pass leaves the pinned segment alone",
          stale.tokens() == restored_tokens and stale.state == "original")


async def test_significance():
    print("\n[8] significance scoring")
    from ctxagent.compaction import significance

    conv = Conversation()
    conv.new("system", K_SYSTEM, "sys", pinned=True)
    conv.new("user", K_USER, "Fix the solver in solver.py")
    conv.new("assistant", K_ASSISTANT,
             "I'll use a Newton iteration here because the Jacobian is cheap.")
    cid = "x1"
    conv.new("assistant", K_TOOL_CALL, "", tool_calls=[
        {"id": cid, "type": "function",
         "function": {"name": "run_command", "arguments": '{"command":"pytest"}'}}])
    conv.new("tool", K_TOOL_RESULT, "=" * 60 + "\n" + "\n".join(["...."] * 200),
             tool_call_id=cid, tool_name="run_command", resource_key="cmd:noise")
    cid2 = "x2"
    conv.new("assistant", K_TOOL_CALL, "", tool_calls=[
        {"id": cid2, "type": "function",
         "function": {"name": "read_file", "arguments": '{"path":"solver.py"}'}}])
    conv.new("tool", K_TOOL_RESULT, "def newton_solve(jacobian, residual):\n    pass",
             tool_call_id=cid2, tool_name="read_file", resource_key="file:solver.py")
    conv.new("user", K_USER, "Good, now make newton_solve handle a singular jacobian.")

    significance.score_all(conv, 0)
    by_kind = {s.kind: s for s in conv.segments}
    users = [s for s in conv.segments if s.kind == K_USER]
    noise = [s for s in conv.segments if s.resource_key == "cmd:noise"][0]
    useful = [s for s in conv.segments if s.resource_key == "file:solver.py"][0]

    check("user messages score 1.0", all(s.significance == 1.0 for s in users))
    check("tool calls are protected", by_kind[K_TOOL_CALL].significance >= 0.85)
    check("decision-bearing assistant text scores high",
          by_kind[K_ASSISTANT].significance >= 0.5,
          str(by_kind[K_ASSISTANT].significance))
    check("forward-referenced observation beats padded noise",
          useful.significance > noise.significance,
          f"useful={useful.significance:.2f} noise={noise.significance:.2f}")
    print(f"       noise={noise.significance:.2f}  referenced={useful.significance:.2f}")


async def test_condense():
    print("\n[9] tier-2 condensation (stubbed model)")
    from ctxagent.compaction import summarize

    conv = Conversation()
    conv.new("system", K_SYSTEM, "sys", pinned=True)
    segs = []
    for i in range(3):
        cid = f"c{i}"
        conv.new("assistant", K_TOOL_CALL, "", tool_calls=[
            {"id": cid, "type": "function",
             "function": {"name": "run_command", "arguments": "{}"}}])
        s = conv.new("tool", K_TOOL_RESULT, "\n".join(f"chatter {j}" for j in range(400)),
                     tool_call_id=cid, tool_name="run_command", resource_key=f"cmd:{i}")
        segs.append(s)
    before = conv.total_tokens()
    r = await summarize.condense_batch(segs, "fake/model", fake_complete)
    check("condensation freed tokens", r["freed"] > 0, str(r))
    check("all three condensed", r["count"] == 3, str(r))
    check("originals retained", all(s.original_body for s in segs))
    check("tool pairing intact", pairing_ok(conv))
    print(f"       {before} -> {conv.total_tokens()} tokens")


async def test_disabled_baseline():
    print("\n[10] engine off == untouched baseline")
    conv = build_conv()
    before = [s.body for s in conv.segments]
    cfg = CompactionConfig(rewrite_interval_batches=1, enabled=False)
    eng = CompactionEngine("t", cfg, fake_complete)
    await eng.run(conv, 200_000)
    check("nothing changed when disabled", [s.body for s in conv.segments] == before)


async def main():
    print("ctxagent compaction engine — tests")
    for t in (test_supersession, test_frozen_prefix_immutable, test_masking_tiers,
              test_dynamic_tail, test_deep_compaction, test_snap_cut,
              test_recall_pins, test_significance, test_condense,
              test_disabled_baseline):
        await t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
