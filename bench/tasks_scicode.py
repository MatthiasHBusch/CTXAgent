"""SciCode (scicode-bench/SciCode, pinned clone under bench/external) as dependent chats:
one main problem = one chat, one subproblem per user turn. Later subproblems call the
functions written in earlier turns, so the conversation itself is the specification.

    sci:77@S            one main problem (problem_id 77), all its subproblems in order
    sci:57+63@S         several main problems chained in ONE chat, one file per problem
    sci:chat0 .. chatN  preset pairs from CHATS (5-9 steps each, no special-cased steps)

Setting: SciCode "without background" (the step description and function header only; the
scientist-written background is not shown). The agent writes every function of a problem
into `<workdir>/<problem_name>.py`; after each turn the checker extracts the step's
function, builds SciCode's own test script (the file so far + the step's test cases against
the h5 targets) and runs it in a bare interpreter. Score 1 iff all test cases pass.

Data: eval/data/problems_test.jsonl (65 problems, HuggingFace SciCode1/SciCode),
eval/data/test_data.h5 (targets, Google Drive link in SciCode's README). The dev split
(problems_dev.jsonl, 15 problems) has ground_truth_code and is used by `oracle()`.

Dry run / oracle:  python -m bench.tasks_scicode [problem_id from the dev split]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "bench" / "external" / "SciCode"
DATA = SRC / "eval" / "data"
H5 = DATA / "test_data.h5"
# SciCode-Verified (github.com/flyingwagner/scicode-verified): corrected test split and the
# cleaned targets rebuilt from the original h5 by eval_clean/build_clean_h5.py
VERIFIED = ROOT / "bench" / "external" / "scicode-verified" / "scicode_verified"
H5_VERIFIED = VERIFIED / "test_data_cleaned.h5"


def h5_for(split: str) -> Path:
    return H5_VERIFIED if split == "verified" else H5
H5_LOADER = ROOT / "bench" / "scicode_h5.py"
TEST_TIMEOUT = 600  # seconds per step test script (SciCode uses 1800)

# Steps whose code SciCode supplies instead of asking the model (eval/data/<step>.txt).
GIVEN_STEPS = {"13.6": "13.6.txt", "62.1": "62.1.txt", "76.3": "76.3.txt"}

# Preset chats (test split): pairs of problems with 5-9 subproblems and no given step.
CHATS = [("57", "63"), ("58", "64"), ("59", "65"), ("60", "66"), ("61", "67"),
         ("80", "68"), ("69", "71"), ("72", "73")]
# Preset chats (dev split, every step verified against the reference solutions): the nine
# dev problems with >= 3 steps grouped into five chats of 6-11 subproblems. Problem 10
# (ewald summation, 11 steps) is the one that lost steps to mid-problem compaction in the
# mixed sessions of phase 2; here it runs as a clean chain.
# Dev-split steps no arm has ever passed (0 of 80 attempts over phases 2-3, both models, all
# harnesses, with and without compaction): their h5 targets depend on choices the prompt does
# not fix (RNG call order in the MC trace of 47.4, timestep selection in 78.3 whose return
# shape is not even stated, ...). They stay in the scripts (same work for every arm) and the
# analysis reports main-problem success with and without them.
UNSOLVED_AS_SPECIFIED = {"44.3", "47.4", "70.8", "78.3"}

DEV_CHATS: list[tuple[str, ...]] = [("10",), ("70", "29"), ("47", "49"), ("51", "78"), ("38", "44")]


def source_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=SRC, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def problems(split: str = "test") -> dict[str, dict]:
    p = VERIFIED / "problems_test.jsonl" if split == "verified" else DATA / f"problems_{split}.jsonl"
    if not p.exists():
        raise SystemExit(f"SciCode problems missing: {p} (HuggingFace dataset SciCode1/SciCode, or scicode-verified)")
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["problem_id"]] = r
    return out


def function_name(header: str) -> str:
    m = re.search(r"\b(?:def|class)\s+(\w+)\s*\(", header)
    if not m:
        raise ValueError(f"no function/class name in header: {header[:80]}")
    return m.group(1)


def solution_path(workdir: Path, prob: dict) -> Path:
    return workdir / f"{prob['problem_name']}.py"


# ------------------------------------------------------------------- prompts
def first_prompt(prob: dict, step: dict, path: Path, k: int, n_steps: int) -> str:
    return (f"Q{k}: We start a multi-step scientific coding problem: **{prob['problem_name'].replace('_', ' ')}** "
            f"({n_steps} steps; I will give you one step per message, later steps build on earlier ones).\n\n"
            f"Overall task: {prob['problem_description_main'].strip()}\n\n"
            f"Write all functions of this problem into the file `{path}`. The file must start with exactly "
            f"these imports and nothing else before them:\n```python\n{prob['required_dependencies'].strip()}\n```\n"
            f"Keep every function you add; later steps call the earlier ones by name. Do not put test or "
            f"example code into the file (you may test in a separate scratch file or in the shell).\n\n"
            f"Step 1 of {n_steps}: {step['step_description_prompt'].strip()}\n\n"
            f"Implement exactly this header (keep name, parameters and the docstring):\n"
            f"```python\n{step['function_header'].strip()}\n{step['return_line']}\n```\n"
            f"Reply with a one-line summary when the function is in the file.")


def step_prompt(prob: dict, step: dict, path: Path, k: int, i: int, n_steps: int) -> str:
    return (f"Q{k}: Step {i} of {n_steps} of the {prob['problem_name'].replace('_', ' ')} problem: "
            f"{step['step_description_prompt'].strip()}\n\n"
            f"Add this function to `{path}` (keep the earlier functions; use them where the step needs them):\n"
            f"```python\n{step['function_header'].strip()}\n{step['return_line']}\n```\n"
            f"Reply with a one-line summary when the function is in the file.")


def given_prompt(prob: dict, step: dict, path: Path, k: int, i: int, n_steps: int, code: str) -> str:
    return (f"Q{k}: Step {i} of {n_steps} of the {prob['problem_name'].replace('_', ' ')} problem is given. "
            f"Add the following function to `{path}` verbatim and reply with one line when done:\n"
            f"```python\n{code.strip()}\n```")


# ------------------------------------------------------------------- checking
def test_script(prob: dict, step: dict, solution_code: str, h5: Path = H5) -> str:
    """SciCode's evaluator: code so far + h5 targets + the step's test cases."""
    tests = step["test_cases"]
    # some test cases import scicode.compare.cmp (numpy/scipy/sympy only) from the clone's src/
    lines = [solution_code, "", "import sys", f"sys.path.insert(0, {str(H5_LOADER.parent)!r})",
             f"sys.path.insert(0, {str(SRC / 'src')!r})",
             "from scicode_h5 import process_hdf5_to_tuple",
             f"targets = process_hdf5_to_tuple({step['step_number']!r}, {len(tests)}, {str(h5)!r})", ""]
    for i, t in enumerate(tests):
        lines.append(f"target = targets[{i}]")
        lines.append("")
        lines += t.split("\n")
        lines.append("")
    return "\n".join(lines)


def check_step(prob: dict, step: dict, path: Path, scratch: Path | None = None, h5: Path = H5) -> tuple[float, dict]:
    if not path.exists():
        return 0.0, {"error": f"{path.name} missing", "step": step["step_number"]}
    code = path.read_text(encoding="utf-8", errors="replace")
    name = function_name(step["function_header"])
    if not re.search(rf"^\s*(?:def|class)\s+{re.escape(name)}\s*\(", code, re.M):
        return 0.0, {"error": f"function {name} not in {path.name}", "step": step["step_number"]}
    scratch = scratch or Path(tempfile.mkdtemp(prefix="sci-"))
    scratch.mkdir(parents=True, exist_ok=True)
    script = scratch / f"test_{step['step_number'].replace('.', '_')}.py"
    script.write_text(test_script(prob, step, code, h5), encoding="utf-8")
    try:
        r = subprocess.run([sys.executable, "-I", str(script)], cwd=scratch, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=TEST_TIMEOUT)
        ok = r.returncode == 0
        return float(ok), {"step": step["step_number"], "function": name, "returncode": r.returncode,
                           "n_tests": len(step["test_cases"]), "stderr_tail": (r.stderr or "")[-400:]}
    except subprocess.TimeoutExpired:
        return 0.0, {"step": step["step_number"], "function": name, "timeout": True}


# ------------------------------------------------------------------- sessions
def parse_spec(spec: str) -> tuple[list[str], int, str]:
    """'sci:7@S' | 'sci:3+19@S' | 'sci:chat0' (dev split) | 'scitest:57+63@S' (test split)
    -> (problem ids, seed, split). The dev split is the default: its reference solutions
    pass every step with this checker (bench/.work/_logs/sci_oracle_dev.log), whereas
    the test split has no reference code and several of its tests reject correct
    answers (SciCode-Verified, 2026); e.g. problem 63 failed identically for every
    harness and model in the P1 pilot."""
    prefix, body = spec.split(":", 1)
    split = "test" if prefix == "scitest" else "dev"
    ids, _, seed = body.partition("@")
    if ids.startswith("chat"):
        chats = CHATS if split == "test" else DEV_CHATS
        return list(chats[int(ids[4:])]), int(seed or 0), split
    return ids.split("+"), int(seed or 0), split


def step_factories(workdir: Path, scratch: Path, pid: str, split: str = "dev", k_offset: int = 0) -> list:
    """Filler factories for bench.seqdep: one `f(k) -> Turn` per subproblem of `pid`, in
    order (the caller must keep the order). Lets SciCode steps be interleaved with other
    work in one chat (bench.tasks_polyglot.build_mix)."""
    from bench import seqdep
    prob = problems(split)[pid]
    path = solution_path(workdir, prob)
    n = len(prob["sub_steps"])
    out = []
    for i, step in enumerate(prob["sub_steps"], 1):
        def make(k: int, i=i, step=step):
            meta = {"problem_id": pid, "problem_name": prob["problem_name"], "step": step["step_number"],
                    "step_index": i, "n_steps": n, "function": function_name(step["function_header"]),
                    "source": f"scicode-{split}@{source_commit()}", "role": "filler"}
            given = GIVEN_STEPS.get(step["step_number"])
            if given:
                code = (DATA / given).read_text(encoding="utf-8")
                return seqdep.Turn(k, "given", given_prompt(prob, step, path, k, i, n, code), meta=meta,
                                   check=lambda final: (1.0, {"given": True}))
            prompt = first_prompt(prob, step, path, k, n) if i == 1 else step_prompt(prob, step, path, k, i, n)
            return seqdep.Turn(k, "question", prompt, meta=meta,
                               check=lambda final, p=prob, s=step: check_step(p, s, solution_path(workdir, p), scratch, h5_for(split)))
        out.append(make)
    return out


def build(a, tdir: Path) -> tuple[Path, list[dict]]:
    """run.py TASK_BUILDERS entry point for `sci:` / `scitest:` specs."""
    spec = a if isinstance(a, str) else a.task
    ids, seed, split = parse_spec(spec)
    probs = problems(split)
    workdir = tdir / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    scratch = tdir / "sci_tests"
    turns, k = [], 0
    for pid in ids:
        if pid not in probs:
            raise SystemExit(f"sci: problem {pid} not in the {split} split")
        prob = probs[pid]
        path = solution_path(workdir, prob)
        n = len(prob["sub_steps"])
        for i, step in enumerate(prob["sub_steps"], 1):
            k += 1
            given = GIVEN_STEPS.get(step["step_number"])
            meta = {"problem_id": pid, "problem_name": prob["problem_name"], "step": step["step_number"],
                    "step_index": i, "n_steps": n, "function": function_name(step["function_header"]),
                    "source": f"scicode-{split}@{source_commit()}", "chat_position": ids.index(pid)}
            if given:
                code = (DATA / given).read_text(encoding="utf-8")
                turns.append({"prompt": given_prompt(prob, step, path, k, i, n, code), "kind": "given",
                              "check": lambda final: (1.0, {"given": True}), "meta": meta})
                continue
            prompt = first_prompt(prob, step, path, k, n) if i == 1 else step_prompt(prob, step, path, k, i, n)
            turns.append({"prompt": prompt, "kind": "question",
                          "check": (lambda p, s: lambda final: check_step(p, s, solution_path(workdir, p), scratch))(prob, step),
                          "meta": meta})
    return workdir, turns


def oracle(pid: str, split: str = "dev") -> list[dict]:
    """Checker self-test on a split with ground_truth_code: write the reference functions
    step by step and check each step (expected 1.0 except for known SciCode test defects)."""
    prob = problems(split)[pid]
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        work.mkdir()
        path = solution_path(work, prob)
        code = prob["required_dependencies"].strip() + "\n\n"
        for step in prob["sub_steps"]:
            if step["step_number"] in GIVEN_STEPS:
                code += (DATA / GIVEN_STEPS[step["step_number"]]).read_text(encoding="utf-8") + "\n\n"
                continue
            empty_score, _ = check_step(prob, step, path, Path(tmp) / "t") if path.exists() else (0.0, {})
            code += step["ground_truth_code"].strip() + "\n\n"
            path.write_text(code, encoding="utf-8")
            score, det = check_step(prob, step, path, Path(tmp) / "t")
            out.append({"step": step["step_number"], "before": empty_score, "ref": score,
                        "err": det.get("stderr_tail", "")[-120:] if score < 1 else ""})
    return out


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "10"
    print("problems:", len(problems("test")), "test /", len(problems("dev")), "dev; commit", source_commit())
    for r in oracle(pid):
        flag = "OK " if (r["before"] == 0 and r["ref"] == 1) else "BAD"
        print(flag, r)
    with tempfile.TemporaryDirectory() as tmp:
        _, turns = build("sci:chat0", Path(tmp))
        print(f"\nsci:chat0 -> {len(turns)} turns; first prompt:\n{turns[0]['prompt'][:900]}")
