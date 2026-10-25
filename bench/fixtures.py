"""Seeded synthetic repository with planted, machine-checkable ground truth.

The repo looks like a small Python accounting library. Every module carries
facts a task can ask about (schema version, owner, audit levels), wrapped in
enough realistic filler code that reading everything costs real context.
A deterministic seed makes every trial of a task see the identical repo.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

NOUNS = ["ledger", "invoice", "payment", "account", "journal", "voucher", "batch",
         "refund", "settlement", "transfer", "statement", "budget", "tax", "fee",
         "currency", "audit", "report", "period", "vendor", "customer"]
VERBS = ["load", "validate", "normalize", "compute", "merge", "export", "render",
         "reconcile", "archive", "rebuild", "summarize", "split", "apply", "resolve"]
PEOPLE = ["a.keller", "b.osei", "c.lindqvist", "d.moreau", "e.tanaka", "f.haddad",
          "g.novak", "h.ferreira", "i.kowalski", "j.mbeki", "k.ivanova", "l.okafor",
          "m.schulz", "n.rossi", "o.nakamura", "p.dubois", "q.alvarez", "r.svensson"]


def _ident(rng: random.Random) -> str:
    return f"{rng.choice(VERBS)}_{rng.choice(NOUNS)}_{rng.choice(NOUNS)}"


def _filler_function(rng: random.Random, name: str, audit_level: str | None) -> str:
    a, b = rng.choice(NOUNS), rng.choice(NOUNS)
    k = rng.randint(2, 9)
    body = [
        f"def {name}({a}_rows, {b}_index=None, *, strict=False):",
        f'    """{name.replace("_", " ").capitalize()} for {a} rows keyed by {b}."""',
        f"    if {b}_index is None:",
        f"        {b}_index = {{}}",
        "    out = []",
        f"    for pos, row in enumerate({a}_rows):",
        f"        key = row.get('{b}_id', pos)",
        f"        if strict and key not in {b}_index:",
        f"            raise KeyError(f'unknown {b}: {{key}}')",
        f"        weight = {b}_index.get(key, {k})",
        f"        amount = int(row.get('amount_cents', 0)) * weight // {k}",
        f"        out.append({{'key': key, 'amount_cents': amount, 'tag': '{a[:3]}{k}'}})",
    ]
    if audit_level:
        body.append(f"    emit_audit_event('{name}', level='{audit_level}', count=len(out))")
    for i in range(rng.randint(2, 6)):
        body.append(f"    # step {i}: keep {a} totals consistent with {b} snapshot")
        body.append(f"    out = [r for r in out if r['amount_cents'] >= {-rng.randint(0, 500)}]")
    body.append("    return out")
    return "\n".join(body)


def build_repo(dest: Path, seed: int = 7, n_modules: int = 16, funcs_per_module: int = 24) -> dict:
    """Write the repo under `dest` and return its ground truth."""
    rng = random.Random(seed)
    pkg = dest / "ledgerkit"
    tests = dest / "tests"
    pkg.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)

    names = []
    while len(names) < n_modules:
        n = f"{rng.choice(NOUNS)}_{rng.choice(['core', 'io', 'rules', 'store', 'views', 'sync', 'calc'])}"
        if n not in names and n != "money_core":
            names.append(n)
    owners = rng.sample(PEOPLE, n_modules)
    truth = {"modules": {}, "reconcile_module": None, "critical_functions": []}
    reconcile_at = rng.randrange(n_modules)
    critical_at = set(rng.sample(range(n_modules), 3))

    for i, mod in enumerate(names):
        version = f"{rng.randint(2, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 99)}"
        owner = f"{owners[i]}@ledgerkit.example"
        head = [
            f'"""{mod.replace("_", " ").title()} utilities for ledgerkit.',
            "",
            f"Maintainer: {owner}",
            '"""',
            "from .audit_hooks import emit_audit_event",
            "",
            f'SCHEMA_VERSION = "{version}"',
            "",
        ]
        funcs, used = [], set()
        crit_idx = rng.randrange(funcs_per_module) if i in critical_at else -1
        for j in range(funcs_per_module):
            fname = _ident(rng)
            while fname in used or fname == "reconcile_ledger":
                fname = _ident(rng)
            used.add(fname)
            if j == crit_idx:
                level = "CRITICAL"
                truth["critical_functions"].append(f"{mod}.{fname}")
            else:
                level = rng.choice([None, None, "INFO", "WARNING"])
            funcs.append(_filler_function(rng, fname, level))
        if i == reconcile_at:
            funcs.insert(rng.randrange(len(funcs)), _filler_function(rng, "reconcile_ledger", "INFO"))
            truth["reconcile_module"] = mod
        (pkg / f"{mod}.py").write_text("\n".join(head) + "\n\n" + "\n\n\n".join(funcs) + "\n", encoding="utf-8")
        truth["modules"][mod] = {"version": version, "owner": owner}

    (pkg / "audit_hooks.py").write_text(
        "EVENTS = []\n\n\ndef emit_audit_event(name, level='INFO', **fields):\n"
        "    EVENTS.append((name, level, fields))\n", encoding="utf-8")
    # Planted bug for the bugfix task: remainder distribution is wrong.
    (pkg / "money_core.py").write_text('''"""Money helpers. Amounts are integer cents."""


def split_amount(total_cents: int, parts: int) -> list[int]:
    """Split total_cents into `parts` integer shares that sum to total_cents.

    Shares differ by at most one cent; larger shares come first.
    """
    if parts <= 0:
        raise ValueError("parts must be positive")
    base = total_cents // parts
    remainder = total_cents - base * parts
    shares = [base] * parts
    for i in range(remainder - 1):
        shares[i] += 1
    return shares


def apply_fee(amount_cents: int, fee_permille: int) -> int:
    """Return the amount after deducting a fee given in permille, rounded half up."""
    fee = (amount_cents * fee_permille) // 1000
    return amount_cents - fee
''', encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (tests / "__init__.py").write_text("", encoding="utf-8")
    (tests / "test_money.py").write_text('''import unittest
from ledgerkit.money_core import split_amount, apply_fee


class TestMoney(unittest.TestCase):
    def test_split_sums(self):
        for total in (0, 1, 99, 100, 101, 12345):
            for parts in (1, 2, 3, 7):
                self.assertEqual(sum(split_amount(total, parts)), total)

    def test_split_shape(self):
        self.assertEqual(split_amount(100, 3), [34, 33, 33])
        self.assertEqual(split_amount(5, 3), [2, 2, 1])

    def test_fee_rounding(self):
        self.assertEqual(apply_fee(1000, 25), 975)
        self.assertEqual(apply_fee(1999, 5), 1989)   # fee 9.995 -> 10
        self.assertEqual(apply_fee(100, 5), 99)      # fee 0.5 -> 1


if __name__ == "__main__":
    unittest.main()
''', encoding="utf-8")
    (dest / "README.md").write_text("# ledgerkit\n\nRun tests with `python -m unittest discover -s tests -t .`\n",
                                    encoding="utf-8")
    return truth


if __name__ == "__main__":
    import sys, tempfile
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    t = build_repo(d)
    print(d)
    print(json.dumps(t, indent=1)[:800])
