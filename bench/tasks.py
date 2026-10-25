"""Benchmark tasks: prompt + deterministic checker. No LLM judge.

Each checker returns (score in [0, 1], details). A trial succeeds iff score == 1.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

SKIP = {"__init__", "audit_hooks", "money_core"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------- lookup
def lookup_prompt(repo: Path, truth: dict) -> str:
    return (f"The Python package `ledgerkit` is in {repo}. Which module defines the function "
            "`reconcile_ledger`, what is that module's SCHEMA_VERSION, and who is its maintainer "
            "(email)? End your reply with exactly one line of the form\n"
            "ANSWER: <module name without .py> | <schema version> | <maintainer email>")


def lookup_check(repo: Path, truth: dict, final: str, pre: dict) -> tuple[float, dict]:
    mod = truth["reconcile_module"]
    want = [mod, truth["modules"][mod]["version"], truth["modules"][mod]["owner"]]
    line = next((l for l in reversed(final.splitlines()) if l.strip().upper().startswith("ANSWER:")), "")
    got = [p.strip().strip("`*") for p in line.split(":", 1)[-1].split("|")] if line else []
    hits = sum(1 for i, w in enumerate(want) if i < len(got) and got[i].lower().removesuffix(".py") == w.lower())
    return hits / 3, {"want": want, "got": got}


# --------------------------------------------------------------------- bugfix
def bugfix_prompt(repo: Path, truth: dict) -> str:
    return (f"The repository {repo} has failing unit tests "
            "(run them from the repo root with `python -m unittest discover -s tests -t .`). "
            "Fix the library code so that all tests pass. Do not modify anything under tests/. "
            "Finish with a one-paragraph summary of the root causes.")


def bugfix_pre(repo: Path) -> dict:
    return {"test_sha": _sha(repo / "tests" / "test_money.py")}


def bugfix_check(repo: Path, truth: dict, final: str, pre: dict) -> tuple[float, dict]:
    untouched = _sha(repo / "tests" / "test_money.py") == pre["test_sha"]
    r = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                       cwd=repo, capture_output=True, text=True, timeout=120)
    ran = re.search(r"Ran (\d+) test", r.stderr)
    passed = r.returncode == 0 and ran and int(ran.group(1)) == 3
    return float(bool(passed and untouched)), {"tests_pass": bool(passed), "tests_untouched": untouched,
                                               "tail": r.stderr[-300:]}


# ---------------------------------------------------------------------- audit
def audit_prompt(repo: Path, truth: dict) -> str:
    return (f"Audit the Python package `ledgerkit` in {repo}. Write a file AUDIT.md in the repo root "
            "containing:\n"
            "1. A markdown table with columns `module | schema_version | maintainer` and one row for "
            "every module in ledgerkit/ except __init__.py, audit_hooks.py and money_core.py "
            "(module name without .py).\n"
            "2. A section `## Critical` listing, one per line as `- module.function`, every function "
            "that calls emit_audit_event with level='CRITICAL'.\n"
            "Be exact; values are checked automatically.")


def audit_check(repo: Path, truth: dict, final: str, pre: dict) -> tuple[float, dict]:
    p = repo / "AUDIT.md"
    if not p.exists():
        return 0.0, {"error": "AUDIT.md missing"}
    text = p.read_text(encoding="utf-8", errors="replace")
    rows = {}
    for line in text.splitlines():
        cells = [c.strip().strip("`*") for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0].removesuffix(".py") in truth["modules"]:
            rows[cells[0].removesuffix(".py")] = (cells[1], cells[2])
    good = sum(1 for m, v in truth["modules"].items() if rows.get(m) == (v["version"], v["owner"]))
    crit_part = text.split("## Critical", 1)[1] if "## Critical" in text else ""
    listed = set(re.findall(r"^\s*[-*]\s*`?([\w]+\.[\w]+)`?", crit_part, re.M))
    want = set(truth["critical_functions"])
    crit_ok = listed == want
    n = len(truth["modules"])
    score = (good + (n * 0.25 if crit_ok else 0)) / (n * 1.25)
    return score, {"rows_correct": good, "rows_total": n, "critical_ok": crit_ok,
                   "critical_missing": sorted(want - listed), "critical_extra": sorted(listed - want)}


TASKS = {
    "lookup": {"prompt": lookup_prompt, "check": lookup_check, "pre": lambda repo: {}},
    "bugfix": {"prompt": bugfix_prompt, "check": bugfix_check, "pre": bugfix_pre},
    "audit": {"prompt": audit_prompt, "check": audit_check, "pre": lambda repo: {}},
}
