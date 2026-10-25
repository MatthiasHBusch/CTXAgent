"""Context-Bench Filesystem suite (Letta, Apache-2.0), code-agent variant.

Source: github.com/letta-ai/letta-evals, letta-leaderboard/filesystem-agent.
We use the official prompt (filesystem_code.jsonl, `{pwd}` = trial directory),
the official rubric (rubric.txt) and the official judge model (gpt-5-mini,
temperature 0), called through OpenRouter. Every harness gets its own copy of
the 10 data files, so no harness can contaminate another's environment.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import httpx

SRC = Path(__file__).resolve().parent / "external" / "letta-evals" / "letta-leaderboard" / "filesystem-agent"
JUDGE_MODEL = "openai/gpt-5-mini"

# Ambiguous multi-state owner questions (REDESIGN.md F1): the anchor person has addresses in
# more than one state, the question says "the same state as the owner of ...", and the rubric
# accepts one reading only. Every harness at every window got exactly these four wrong while
# giving both readings. Excluded from stratified samples and from SeqDep filler/derived pools.
EXCLUDED = {5, 15, 42, 78}
EXCLUDED_REASON = "ambiguous multi-state owner questions (anchor has addresses in several states)"


def questions() -> list[dict]:
    out = []
    with open(SRC / "datasets" / "filesystem_code.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                q = json.loads(line)
                q["id"] = i
                q["qtype"] = q["agent_args"]["extra"]["question_type"]
                q["difficulty"] = q["agent_args"]["extra"]["difficulty"]
                out.append(q)
    return out


def stratified(n: int, seed: int = 0, exclude: set | None = None) -> list[int]:
    """Deterministic sample proportional to question_type, at least one per type
    while n allows. Returned ids are stable for a given (n, seed). EXCLUDED ids are
    never sampled (pass exclude=set() to get the historical, pre-exclusion samples)."""
    exclude = EXCLUDED if exclude is None else exclude
    qs = [q for q in questions() if q["id"] not in exclude]
    by = defaultdict(list)
    for q in qs:
        by[q["qtype"]].append(q["id"])
    rng = random.Random(seed)
    for ids in by.values():
        rng.shuffle(ids)
    types = sorted(by, key=lambda t: -len(by[t]))
    quota = {t: 0 for t in types}
    for t in types[:n]:
        quota[t] = 1
    while sum(quota.values()) < n:
        t = max(types, key=lambda t: len(by[t]) / len(qs) * n - quota[t])
        quota[t] += 1
    return sorted(i for t in types for i in by[t][:quota[t]])


def setup(workdir: Path, qid: int) -> dict:
    q = questions()[qid]
    data = workdir / "data"
    shutil.copytree(SRC / "files", data, dirs_exist_ok=True)
    prompt = q["input"].replace("{pwd}", str(data))
    return {"prompt": prompt, "qid": qid, "ground_truth": q["ground_truth"], "question": q["input"],
            "qtype": q["qtype"], "difficulty": q["difficulty"],
            "required_files": q["agent_args"]["extra"].get("required_files")}


def judge(ctx: dict, submission: str, api_key: str | None = None) -> tuple[float, dict]:
    rubric = (SRC / "rubric.txt").read_text(encoding="utf-8")
    prompt = (rubric.replace("{input}", ctx["question"]).replace("{ground_truth}", str(ctx["ground_truth"]))
              .replace("{submission}", submission or "(no answer)"))
    key = api_key or os.environ["OPENROUTER_API_KEY"]
    body = {"model": JUDGE_MODEL, "temperature": 0.0, "messages": [{"role": "user", "content": prompt}],
            "usage": {"include": True}}
    last = None
    for _ in range(3):
        try:
            r = httpx.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=180,
                           headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            j = r.json()
            text = j["choices"][0]["message"]["content"] or ""
            m = re.search(r"\b(1\.0|0\.5|0\.0|1|0)\b", text)
            if m:
                return float(m.group(1)), {"judge_raw": text[:200], "judge_cost": (j.get("usage") or {}).get("cost"),
                                           "ground_truth": ctx["ground_truth"]}
            last = f"unparseable judge reply: {text[:200]}"
        except Exception as exc:  # noqa: BLE001 - retried, then recorded
            last = f"{type(exc).__name__}: {exc}"
    return 0.0, {"judge_error": last, "ground_truth": ctx["ground_truth"]}


CONSISTENCY_PROMPT = """You compare two answers to the same question, given by the same assistant at
different times. Decide whether the RECALLED answer states the same final result as the ORIGINAL
answer (same entity, same value; an id vs. the name of that same entity counts as the same if the
original answer makes the correspondence clear). Correctness does not matter, only agreement.

Question: {question}

ORIGINAL answer (end of the assistant's reply):
{original}

RECALLED answer:
{recalled}

Reply with exactly one character: 1 if they agree, 0 if they do not."""


def judge_consistency(question: str, original: str, recalled: str, api_key: str | None = None) -> float:
    """Retention probe: does a from-memory recall match the agent's own earlier answer?"""
    if not recalled.strip() or recalled.strip().lower().startswith("unknown"):
        return 0.0
    prompt = CONSISTENCY_PROMPT.format(question=question, original=original[-2500:], recalled=recalled)
    key = api_key or os.environ["OPENROUTER_API_KEY"]
    body = {"model": JUDGE_MODEL, "temperature": 0.0, "messages": [{"role": "user", "content": prompt}]}
    for _ in range(3):
        try:
            r = httpx.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=180,
                           headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"]["content"] or "").strip()
            if text[:1] in "01":
                return float(text[:1])
        except Exception:  # noqa: BLE001 - retried
            pass
    return 0.0


# ------------------------------------------------------------- derived questions
# Ground truth for derived questions (REDESIGN.md T1b) is computed from the dataset files
# themselves. Records are parsed once; every template is a function of a person id.
FILES = ("people", "addresses", "vehicles", "pets", "bank_accounts", "employments", "credit_cards",
         "insurance_policies", "internet_accounts", "medical_records")
_RECORDS: dict | None = None


def records() -> dict[str, list[dict]]:
    """{file_stem: [record dict]} with `_id` (record id or person id) and `_owner` (person id)."""
    global _RECORDS
    if _RECORDS is not None:
        return _RECORDS
    out = {}
    for stem in FILES:
        recs = []
        for blk in (SRC / "files" / f"{stem}.txt").read_text(encoding="utf-8").split("\n\n"):
            lines = [l for l in blk.strip().splitlines() if l.strip()]
            if not lines or not lines[0].startswith("### "):
                continue
            m = re.match(r"### (.+?) \((?:owner|ID): (pers-\d+)\)", lines[0])
            d = {"_owner": m.group(2) if m else None, "_head": m.group(1) if m else lines[0][4:]}
            for l in lines[1:]:
                for part in (l.split(" | ") if " | " in l else [l]):
                    k, _, v = part.partition(": ")
                    d[k.strip().lower()] = v.strip()
            if stem == "people":
                d["_id"], d["name"] = d["_owner"], d["_head"]
            else:
                d["_id"] = d["_head"]
            recs.append(d)
        out[stem] = recs
    _RECORDS = out
    return out


def person_by_name(name: str) -> dict | None:
    """People names are unique in the dataset (checked: 500 people, 0 duplicates)."""
    return next((p for p in records()["people"] if p["name"].lower() == name.strip().lower()), None)


def owned(stem: str, pid: str) -> list[dict]:
    return [r for r in records()[stem] if r["_owner"] == pid]


def _money(x: float) -> str:
    return f"${x:,.2f}"


# template key -> (question text with {ref}, ground-truth function(pid) -> str, answer type)
DERIVED_TEMPLATES = {
    "n_vehicles": ("How many vehicles are registered to {ref}?", lambda pid: str(len(owned("vehicles", pid))), "int"),
    "n_pets": ("How many pets does {ref} own?", lambda pid: str(len(owned("pets", pid))), "int"),
    "n_cards": ("How many credit cards does {ref} have?", lambda pid: str(len(owned("credit_cards", pid))), "int"),
    "n_accounts": ("How many bank accounts does {ref} have?",
                   lambda pid: str(len(owned("bank_accounts", pid))), "int"),
    "n_policies": ("How many insurance policies does {ref} hold?",
                   lambda pid: str(len(owned("insurance_policies", pid))), "int"),
    "bank_total": ("What is the total bank balance (sum over all accounts, USD) of {ref}?",
                   lambda pid: _money(sum(float(a["balance"]) for a in owned("bank_accounts", pid))), "money"),
    "n_records": ("How many records in total (pets + vehicles + credit cards + bank accounts + insurance "
                  "policies + internet accounts) does {ref} have?",
                  lambda pid: str(sum(len(owned(s, pid)) for s in ("pets", "vehicles", "credit_cards",
                                                                    "bank_accounts", "insurance_policies",
                                                                    "internet_accounts"))), "int"),
}


def derive(template: str, name: str, ref: str) -> dict:
    """Evaluate a derived-question template for the person `name`, phrased with `ref`
    (either an explicit name, the control, or a back-reference such as 'the person you
    identified in Q12'). Returns {question, ground_truth, answer_type, template, pid}."""
    text, fn, kind = DERIVED_TEMPLATES[template]
    p = person_by_name(name)
    if p is None:
        raise KeyError(f"no person named {name!r}")
    return {"question": text.format(ref=ref), "ground_truth": fn(p["_id"]), "answer_type": kind,
            "template": template, "pid": p["_id"], "name": p["name"]}


def name_answer_ids(exclude: set | None = None) -> list[int]:
    """Question ids whose ground truth is a (unique) person name: the base pool for
    derived questions."""
    exclude = EXCLUDED if exclude is None else exclude
    return [q["id"] for q in questions() if q["id"] not in exclude and person_by_name(str(q["ground_truth"]))]


if __name__ == "__main__":
    ids = stratified(10)
    qs = questions()
    for i in ids:
        print(i, qs[i]["qtype"], qs[i]["difficulty"], qs[i]["ground_truth"])
    print("excluded:", sorted(EXCLUDED), "-", EXCLUDED_REASON)
    print("name-answer pool:", len(name_answer_ids()))
    print(derive("bank_total", qs[ids[0]]["ground_truth"], "the person you identified in Q1"))
