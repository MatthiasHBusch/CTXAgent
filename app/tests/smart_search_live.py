"""Live check of smart_search with the real model (paid: a few cents per case with Haiku 5.5).

    python -m tests.smart_search_live                    # all cases, on this repository
    python -m tests.smart_search_live --case rungs       # one case
    python -m tests.smart_search_live --out report.json  # keep the numbers and the outputs

Each case is a search goal with the files a good search finds (`expect`) and, for some,
folders it has no reason to open (`avoid`). Reported per case:

* recall: expected files with at least one range; extra files read; avoided folders opened
* effort: folders surveyed, files read, model calls
* output: lines returned and their share of the lines of the files read
* cost and tokens as the provider reports them, and the largest prompt, to see how far
  requests stay below Haiku 5.5's price step (100,000 prompt tokens)
* provider tokens / local tokens of the prompts: the factor REQUEST_TOKENS has to allow for

Not part of run_all: it needs an API key and costs money.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctxagent import config, llm, smart_search, tokens

ROOT = Path(__file__).resolve().parent.parent

CASES = {
    "keepalive": {
        "goal": "How the prompt cache is kept warm while a tool runs for a long time: where the keep-alive "
                "request is built and sent, and what decides when it stops.",
        "expect": ["ctxagent/agent.py", "ctxagent/llm.py", "ctxagent/anthropic_api.py"],
        "avoid": ["web", "packaging"],
    },
    "curate": {
        "goal": "I want to change how curate_context counts its references. Find the code that turns a "
                "reference number into an item and replaces it.",
        "expect": ["ctxagent/compaction/curate.py"],
        "avoid": ["web", "packaging", "ctxagent/readers"],
    },
    "new_tool": {
        "goal": "I want to add a new file tool for the model. Find where tools are registered with their "
                "schema, where the default tool list is, and where a tool call is dispatched.",
        "expect": ["ctxagent/tools.py", "ctxagent/agent.py"],
        "avoid": ["web", "packaging"],
    },
    "installer_upgrade": {
        "goal": "How the Windows installer upgrades an installation of the app's former name (PiAgent): "
                "what it removes and what it keeps.",
        "expect": ["packaging/ctxagent.iss"],
        "avoid": ["web", "tests", "ctxagent/compaction"],
    },
    # Plausible but not in this branch (the Linux build lives on desktop-app-linux): nothing to find.
    "absent_plausible": {
        "goal": "Which shared libraries the Linux build leaves to the system instead of bundling, and where "
                "that is decided.",
        "expect": [],
        "avoid": ["web", "tests", "ctxagent/compaction"],
    },
    "context_bar": {
        "goal": "The context bar in the chat window: where the numbers it shows are computed on the server "
                "and where the browser draws it.",
        "expect": ["web/app.js"],
        "avoid": ["packaging"],
    },
    "rungs": {
        "goal": "The function that decides where the cache rungs go.",
        "instructions": "It is in ctxagent/cache_policy.py. Only that file; I do not need tests or callers.",
        "expect": ["ctxagent/cache_policy.py"],
        "avoid": ["web", "packaging", "tests"],
        "max_files": 1,
    },
    "absent": {
        "goal": "Where the Kubernetes deployment and the Helm chart of this project are configured.",
        "expect": [],
        "avoid": ["ctxagent", "tests", "web"],
    },
}


async def run_case(name: str, case: dict, model: str, root: Path) -> dict:
    calls: list[dict] = []

    async def complete(**kw):
        local = tokens.count_messages(kw["messages"])
        record = {"local_prompt": local}
        calls.append(record)

        async def usage(raw, **meta):
            record.update(prompt=int(raw.get("prompt_tokens") or 0),
                          completion=int(raw.get("completion_tokens") or 0),
                          cost=float(raw.get("cost") or 0.0), provider=meta.get("provider"))

        return await llm.complete(usage_callback=usage, session_id="smart-search-live", **kw)

    started = time.monotonic()
    report = await smart_search.search(root, case["goal"], case.get("instructions", ""), complete, model)
    seconds = time.monotonic() - started
    text = smart_search.render(report, root, case["goal"], model)

    def rel(path: Path) -> str:
        return path.relative_to(root).as_posix() if path != root else "."

    surveyed = [rel(p) for p in report.folders_surveyed]
    read = [rel(h.path) for h in report.files]
    with_ranges = [rel(h.path) for h in report.files if h.ranges]
    lines_out = sum(r.end - r.start + 1 for h in report.files for r in h.ranges)
    lines_read = sum(h.total_lines for h in report.files)
    measured = [c for c in calls if c.get("prompt")]
    return {
        "case": name, "seconds": round(seconds, 1),
        "found": [f for f in case["expect"] if f in with_ranges],
        "missed": [f for f in case["expect"] if f not in with_ranges],
        "extra_files_with_ranges": [f for f in with_ranges if f not in case["expect"]],
        "read_without_ranges": [f for f in read if f not in with_ranges],
        "avoided_but_opened": [f for f in surveyed if any(f == a or f.startswith(a + "/") for a in case.get("avoid", []))],
        "folders": surveyed, "files_read": len(read), "calls": report.calls, "errors": report.errors,
        "lines_out": lines_out, "lines_read": lines_read,
        "core_lines": sum(r.end - r.start + 1 for h in report.files for r in h.ranges if r.kind == "core"),
        "files_high": sum(1 for h in report.files if h.relevance == "high"),
        "result_tokens": tokens.count(text),
        "prompt_tokens": sum(c["prompt"] for c in measured),
        "completion_tokens": sum(c["completion"] for c in measured),
        "largest_prompt": max((c["prompt"] for c in measured), default=0),
        "cost": round(sum(c["cost"] for c in measured), 5),
        "provider_per_local_token": round(sum(c["prompt"] for c in measured)
                                          / max(1, sum(c["local_prompt"] for c in measured)), 3),
        "providers": sorted({str(c.get("provider")) for c in measured}),
        "text": text,
    }


def show(r: dict) -> None:
    print(f"\n== {r['case']}: {r['seconds']} s, ${r['cost']:.4f}, {r['calls']} calls, "
          f"{len(r['folders'])} folders, {r['files_read']} files read ({r['files_high']} high)")
    print(f"   found {r['found']}  missed {r['missed']}")
    print(f"   other files with ranges: {r['extra_files_with_ranges']}")
    print(f"   read without ranges: {r['read_without_ranges']}")
    print(f"   folders: {r['folders']}")
    if r["avoided_but_opened"]:
        print(f"   OPENED WITHOUT REASON: {r['avoided_but_opened']}")
    share = 100 * r["lines_out"] / max(1, r["lines_read"])
    print(f"   lines returned: {r['lines_out']} ({r['core_lines']} core) of {r['lines_read']} read ({share:.0f} %), result {r['result_tokens']} tokens")
    print(f"   tokens: {r['prompt_tokens']} in, {r['completion_tokens']} out, largest prompt {r['largest_prompt']}, "
          f"provider/local {r['provider_per_local_token']}, via {r['providers']}")
    for e in r["errors"]:
        print(f"   ERROR {e}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append")
    ap.add_argument("--model", default="")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default="")
    ap.add_argument("--goal", default="", help="an ad-hoc goal instead of the cases")
    ap.add_argument("--effort", default="")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    config.load_env()
    model = args.model or smart_search.default_model("")
    problem = config.key_problem(model)
    if problem:
        print(problem)
        return 2
    print(f"model {model}, root {args.root}")
    results = []
    if args.effort:
        smart_search.EFFORT = args.effort
    if args.goal:
        CASES["adhoc"] = {"goal": args.goal, "expect": []}
        args.case = ["adhoc"]
    for name in (args.case or list(CASES)) * args.repeat:
        case = CASES[name]
        limit = case.get("max_files")
        if limit:
            old, smart_search.MAX_FILES = smart_search.MAX_FILES, limit
        try:
            results.append(await run_case(name, case, model, Path(args.root)))
        finally:
            if limit:
                smart_search.MAX_FILES = old
        show(results[-1])
    print(f"\ntotal ${sum(r['cost'] for r in results):.4f}, "
          f"{sum(r['prompt_tokens'] for r in results)} prompt tokens, "
          f"missed files in {sum(1 for r in results if r['missed'])} of {len(results)} cases")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
