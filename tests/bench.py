"""Measurement harness: what does the engine actually buy?

Simulates a long agentic coding trajectory and replays it turn by turn under
several configurations, accounting tokens the way a provider bills them:

    billed  = sum over turns of (fresh prefix tokens + cached prefix tokens * 0.1)

Cache reads on Anthropic bill at ~10% of the base input rate, so a scheme that
saves tokens but shatters the cache can easily cost MORE than doing nothing.
That is the whole point of the frozen-prefix design, and this harness is what
makes the claim checkable rather than asserted.

Run:  python -m tests.bench
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

CACHE_READ_RATE = 0.10  # cached input costs ~10% of fresh input


async def fake_complete(*, model, messages, max_tokens=2000, temperature=0.0,
                        response_format=None):
    if response_format:
        import json
        import re
        ids = re.findall(r"### id=(\w+)", messages[-1]["content"])
        return json.dumps({"items": [
            {"id": i, "text": "Ran a command; exit 0; nothing notable."} for i in ids
        ]})
    return (
        "## 1. User intent\nRefactor the solver and keep the tests green.\n"
        "## 2. Key decisions and rationale\nNewton with a damped line search, "
        "because the Jacobian is cheap to form.\n"
        "## 3. Files and resources touched\n`src/solver.py` — rewritten.\n"
        "## 4. Technical facts established\nnumpy 2.1; tests live in `tests/`.\n"
        "## 5. Errors and resolutions\n`ZeroDivisionError` on a singular Jacobian — "
        "fixed with a pseudo-inverse fallback.\n"
        "## 6. Work completed\nSolver rewritten, tests pass.\n"
        "## 7. Outstanding work\nDocument the new options.\n"
        "## 8. Current state\nMid-refactor of the reporting module.\n"
        "## 9. Next step\nUpdate the docstrings.\n"
    )


# ---------------------------------------------------------------------------

FILES = [f"src/module_{i}.py" for i in range(6)]


def file_body(path: str, version: int, lines: int) -> str:
    return "\n".join(
        f"{j:6d}\tdef fn_{j}():  # {path} v{version} — computation step {j}"
        for j in range(lines)
    )


def build_trajectory(turns: int = 60) -> list[dict]:
    """A realistic mix: read a file, edit it, re-read it, run tests, repeat."""
    import random

    rnd = random.Random(7)
    steps: list[dict] = []
    versions = {f: 0 for f in FILES}
    for t in range(turns):
        if t % 7 == 0:
            steps.append({"op": "user", "text": f"Now handle requirement {t // 7 + 1}."})
        f = rnd.choice(FILES)
        steps.append({"op": "read", "file": f, "version": versions[f],
                      "lines": rnd.choice([400, 900, 1500])})
        steps.append({"op": "think", "text":
                      f"I'll change `{f}` because the current form recomputes the "
                      f"Jacobian每 iteration.".replace("每", " every ")})
        steps.append({"op": "edit", "file": f})
        versions[f] += 1
        steps.append({"op": "read", "file": f, "version": versions[f],
                      "lines": rnd.choice([400, 900, 1500])})
        if t % 3 == 0:
            steps.append({"op": "cmd", "lines": rnd.choice([80, 350, 1200])})
    return steps


def apply_step(conv: Conversation, step: dict, n: int) -> None:
    op = step["op"]
    if op == "user":
        conv.new("user", K_USER, step["text"])
    elif op == "think":
        conv.new("assistant", K_ASSISTANT, step["text"])
    elif op == "read":
        cid = f"c{n}"
        conv.new("assistant", K_TOOL_CALL, "", tool_calls=[{
            "id": cid, "type": "function",
            "function": {"name": "read_file",
                         "arguments": f'{{"path": "{step["file"]}"}}'}}])
        conv.new("tool", K_TOOL_RESULT,
                 file_body(step["file"], step["version"], step["lines"]),
                 tool_call_id=cid, tool_name="read_file",
                 resource_key=f"file:{step['file']}")
    elif op == "edit":
        cid = f"c{n}"
        conv.new("assistant", K_TOOL_CALL, "", tool_calls=[{
            "id": cid, "type": "function",
            "function": {"name": "edit_file",
                         "arguments": f'{{"path": "{step["file"]}"}}'}}])
        conv.new("tool", K_TOOL_RESULT, f"Edited {step['file']}: 1 replacement.",
                 tool_call_id=cid, tool_name="edit_file",
                 resource_key=f"file:{step['file']}", mutates=True)
    elif op == "cmd":
        cid = f"c{n}"
        conv.new("assistant", K_TOOL_CALL, "", tool_calls=[{
            "id": cid, "type": "function",
            "function": {"name": "run_command", "arguments": '{"command":"pytest -q"}'}}])
        conv.new("tool", K_TOOL_RESULT,
                 "\n".join(f"test_case_{j} ... ok" for j in range(step["lines"])),
                 tool_call_id=cid, tool_name="run_command", resource_key=f"cmd:pytest:{n}")


# ---------------------------------------------------------------------------

async def run_config(name: str, cfg: CompactionConfig, steps: list[dict],
                     window: int) -> dict:
    conv = Conversation()
    conv.new("system", K_SYSTEM, "You are an agent." * 40, pinned=True)
    eng = CompactionEngine("bench", cfg, fake_complete)

    billed = 0.0
    raw_sent = 0
    peak = 0
    overflow_turns = 0
    prev_prefix_bodies: list[str] = []
    deep_passes = 0
    llm_calls = 0

    for n, step in enumerate(steps):
        apply_step(conv, step, n)
        if step["op"] not in ("read", "cmd", "edit"):
            continue  # only bill at model-call boundaries

        rep = await eng.run(conv, window)
        if rep.get("deep"):
            deep_passes += 1
        llm_calls += sum(
            1 for t in rep.get("tiers", []) if t["tier"] in ("2-condense", "3-deep")
        )

        # How much of this request's prefix is byte-identical to the last one?
        bodies = [s.body for s in conv.segments]
        common = 0
        for a, b in zip(prev_prefix_bodies, bodies):
            if a != b:
                break
            common += 1
        cached_tokens = sum(s.tokens() for s in conv.segments[:common])
        total = conv.total_tokens()
        fresh = total - cached_tokens

        billed += fresh + cached_tokens * CACHE_READ_RATE
        raw_sent += total
        peak = max(peak, total)
        if total > window:
            overflow_turns += 1
        prev_prefix_bodies = bodies

    return {
        "name": name,
        "billed": billed,
        "raw": raw_sent,
        "peak": peak,
        "final": conv.total_tokens(),
        "overflow": overflow_turns,
        "deep": deep_passes,
        "llm_calls": llm_calls,
        "segments": len(conv.segments),
    }


async def main() -> int:
    window = 120_000
    steps = build_trajectory(turns=60)
    print(f"ctxagent compaction benchmark")
    print(f"  simulated trajectory: {len(steps)} steps, window {window:,} tokens")
    print(f"  billing model: fresh tokens + cached tokens x {CACHE_READ_RATE}\n")

    configs = [
        ("no compaction", CompactionConfig(enabled=False)),
        ("dedup only", CompactionConfig(
            masking_enabled=False, significance_enabled=False, deep_enabled=False)),
        ("mask only", CompactionConfig(
            dedup_enabled=False, significance_enabled=False, deep_enabled=False)),
        ("tier 1 (dedup+mask)", CompactionConfig(
            significance_enabled=False, deep_enabled=False)),
        ("full engine", CompactionConfig()),
        ("full, hot zone 10k", CompactionConfig(hot_zone_tokens=10_000)),
        ("full, NO frozen prefix", CompactionConfig(hot_zone_tokens=10_000_000)),
    ]

    rows = []
    for name, cfg in configs:
        rows.append(await run_config(name, cfg, steps, window))

    base = rows[0]
    hdr = (f"{'configuration':<24}{'billed':>13}{'vs base':>9}"
           f"{'peak ctx':>12}{'final ctx':>12}{'over':>6}{'deep':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        rel = r["billed"] / base["billed"]
        print(f"{r['name']:<24}{r['billed']:>13,.0f}{rel:>8.0%}"
              f"{r['peak']:>12,}{r['final']:>12,}{r['overflow']:>6}{r['deep']:>6}")

    print("\nnotes")
    print("  'over'  = turns whose context exceeded the window (would have failed)")
    print("  'deep'  = number of tier-3 summarisation passes")
    print()
    print("  'mask only' looks useless (99%) and is not: masking may act only inside")
    print("  the hot zone, so on a run this long almost everything has already frozen")
    print("  out of its reach. Tier 1 bounds how fast the prefix grows; only tier 3")
    print("  can reclaim what is already frozen. They are complements, not rivals.")
    full = next(r for r in rows if r["name"] == "full engine")
    nofrozen = next(r for r in rows if r["name"] == "full, NO frozen prefix")
    print(f"\n  Frozen prefix is worth {1 - full['billed']/nofrozen['billed']:.0%} of billed cost")
    print(f"  at an identical tier configuration: compacting the whole history")
    print(f"  ({nofrozen['final']:,} final tokens vs {full['final']:,}) sends fewer raw")
    print(f"  tokens but re-bills them at full price every turn.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
