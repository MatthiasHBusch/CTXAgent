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
from ctxagent.segments import (
    Conversation,
    K_SYSTEM,
    K_USER,
    K_ASSISTANT,
    K_TOOL_CALL,
    K_TOOL_RESULT,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))


async def fake_complete(*, model, messages, max_tokens=2000, temperature=0.0, response_format=None):
    """Stand-in summariser: no network."""
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

async def test_frozen_prefix_immutable():
    print("\n[1] frozen prefix is never rewritten (cache safety)")
    conv = build_conv(n_reads=4, big_lines=400)
    cfg = CompactionConfig(hot_zone_tokens=6_000, deep_enabled=False, protected_recent_observations=1)
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


async def test_no_incremental_rewrites():
    print("\n[2] below the deep trigger nothing is rewritten")
    conv = build_conv(n_reads=4, repeated=True)
    before = [s.body for s in conv.segments]
    cfg = CompactionConfig(hot_zone_tokens=1_000, protected_recent_observations=0, protected_recent_batches=0)
    eng = CompactionEngine("t", cfg, fake_complete)
    for _ in range(3):
        rep = await eng.run(conv, 10_000_000)
    check("bodies unchanged (no dedup, masking or condensation)", [s.body for s in conv.segments] == before)
    check("every segment still original", all(s.state == "original" for s in conv.segments))
    check("no tiers ran", rep["tiers"] == [] and rep["freed"] == 0, str(rep["tiers"]))


async def test_deep_compaction():
    print("\n[3] deep compaction")
    conv = build_conv(n_reads=6, big_lines=500)
    for i in range(4):
        conv.new("user", K_USER, f"Follow-up question {i}")
        conv.new("assistant", K_ASSISTANT, f"Answer {i}. " + "detail " * 200)
    cfg = CompactionConfig(hot_zone_tokens=8_000, deep_trigger_ratio=0.05,
                           deep_min_gain_tokens=100, deep_cooldown_turns=0)
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


async def test_orphan_rendering():
    print("\n[4] an orphaned tool result is sent as a user note")
    conv = Conversation()
    conv.new("system", K_SYSTEM, "You are an agent.", pinned=True)
    conv.new("tool", K_TOOL_RESULT, "Closed report.md", tool_call_id="gone", tool_name="document_close")
    roles = [m["role"] for m in conv.to_messages()]
    check("orphan tool result is rendered as a user note", roles == ["system", "user"], str(roles))


async def test_disabled_baseline():
    print("\n[5] engine off == untouched baseline")
    conv = build_conv()
    before = [s.body for s in conv.segments]
    cfg = CompactionConfig(enabled=False)
    eng = CompactionEngine("t", cfg, fake_complete)
    await eng.run(conv, 200_000)
    check("nothing changed when disabled", [s.body for s in conv.segments] == before)


async def main():
    print("ctxagent compaction engine — tests")
    for t in (test_frozen_prefix_immutable, test_no_incremental_rewrites, test_deep_compaction,
              test_orphan_rendering, test_disabled_baseline):
        await t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
