"""Aider polyglot benchmark (Aider-AI/polyglot-benchmark, pinned clone under bench/external)
as sequential coding sessions: one Exercism exercise per user turn, test-checked.

    poly:K@S          K exercises of the Python track, order shuffled with seed S
    poly:K@S:lang     another track (go, rust, cpp, java, javascript) if its toolchain exists
    polydep:K@S       the same K exercises as filler of a bench.seqdep dependent session
                      (rules, facts, ephemeral observations interleaved; see build_dep)

Each exercise is copied into the session's working directory WITHOUT its `.meta/`
folder (that holds the reference solution). The agent sees the instructions, the stub
and the test file, as on Exercism and in Aider's own benchmark. The checker re-hashes
the test files (they must be untouched) and runs the test file with pytest in a clean
interpreter; score 1 iff every test passes.

Dry run / oracle check (no model):   python -m bench.tasks_polyglot [K] [seed]
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "bench" / "external" / "polyglot-benchmark"
PYTEST_TIMEOUT = 180  # seconds per exercise test run
GO_BIN = ROOT / "bench" / "external" / "go" / "bin"            # portable Go (go.dev zip), no installer
JS_NODE_MODULES = ROOT / "bench" / "external" / "js_shared" / "node_modules"  # union of the track's devDependencies
TRACKS = ("python", "javascript", "go")


def go_env(tdir: Path | None = None) -> dict:
    """Environment for `go test` and for the agent's shell: portable Go on PATH, caches in
    the trial directory, module mode without network (the exercises are stdlib-only)."""
    env = dict(os.environ)
    env["PATH"] = str(GO_BIN) + os.pathsep + env.get("PATH", "")
    base = (tdir or Path(tempfile.gettempdir())) / "go_cache"
    env.update(GOCACHE=str(base / "cache"), GOPATH=str(base / "path"), GOFLAGS="-mod=mod",
               GOTOOLCHAIN="local", GOPROXY="off", GOSUMDB="off")
    return env


def track_available(lang: str) -> bool:
    if lang == "python":
        return True
    if lang == "go":
        return (GO_BIN / "go.exe").exists() or (GO_BIN / "go").exists()
    if lang == "javascript":
        return (JS_NODE_MODULES / ".bin").exists() and shutil.which("node") is not None
    return False

DOC_ORDER = ("introduction.md", "instructions.md", "instructions.append.md", "hints.md")


def source_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=SRC, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# Exercises whose STUB already passes the tests (Exercism "refactoring" exercises ship working
# code; `counter` has no runnable tests): a free point for every arm, so they are left out.
# Found by the checker oracle (python -m bench.tasks_polyglot 39 0 go).
EXCLUDED = {"go": {"counter", "ledger", "markdown"}, "javascript": {"ledger"}, "python": set()}


def exercises(lang: str = "python") -> list[str]:
    d = SRC / lang / "exercises" / "practice"
    if not d.is_dir():
        raise SystemExit(f"polyglot track not found: {d} (clone Aider-AI/polyglot-benchmark into bench/external)")
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / ".meta" / "config.json").exists()
                  and p.name not in EXCLUDED.get(lang, set()))


def select(k: int, seed: int, lang: str = "python") -> list[str]:
    """Deterministic order: shuffle the whole track with the seed, take the first k.
    k > track size raises: a session must not repeat an exercise."""
    names = exercises(lang)
    if k > len(names):
        raise SystemExit(f"poly: track '{lang}' has {len(names)} exercises, {k} requested")
    rng = random.Random(seed)
    rng.shuffle(names)
    return names[:k]


def config(lang: str, name: str) -> dict:
    return json.loads((SRC / lang / "exercises" / "practice" / name / ".meta" / "config.json").read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def setup_exercise(workdir: Path, name: str, lang: str = "python") -> dict:
    """Copy the exercise (minus .meta) to workdir/<name>; return its context."""
    src = SRC / lang / "exercises" / "practice" / name
    dst = workdir / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".meta", "__pycache__", ".pytest_cache"))
    cfg = config(lang, name)
    files = cfg["files"]
    if lang == "javascript":
        # Exercism ships every test but the first as xtest/xit (students un-skip them one by
        # one); the benchmark runs them all, as Aider's does. Shared node_modules via a
        # directory junction so the agent never needs `npm install` (network is banned).
        for t in files["test"]:
            tp = dst / t
            tp.write_text(re.sub(r"\b(xtest|xit|xdescribe)\(", lambda m: m.group(1)[1:] + "(",
                                 tp.read_text(encoding="utf-8")), encoding="utf-8")
        link = dst / "node_modules"
        if not link.exists():
            if os.name == "nt":
                subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(JS_NODE_MODULES)], capture_output=True)
            else:
                os.symlink(JS_NODE_MODULES, link)
    docs = [f for f in DOC_ORDER if (dst / ".docs" / f).exists()]
    return {"name": name, "lang": lang, "dir": dst, "solution": list(files["solution"]),
            "tests": list(files["test"]), "docs": docs,
            "test_sha": {t: _sha(dst / t) for t in files["test"]}}


def prompt(ctx: dict, k: int | None = None) -> str:
    d = ctx["dir"]
    docs = ", ".join(f"`{d / '.docs' / f}`" for f in ctx["docs"])
    sol = ", ".join(f"`{d / s}`" for s in ctx["solution"])
    tests = ", ".join(f"`{d / t}`" for t in ctx["tests"])
    head = f"Q{k}: " if k else ""
    how = {
        "python": f"You can run the tests yourself with `python -m pytest -q {d}`.",
        "javascript": (f"You can run the tests yourself from inside `{d}` with `node_modules\\.bin\\jest --ci ./*` "
                       "(node_modules is already in place; do not run npm install)."),
        "go": f"You can run the tests yourself from inside `{d}` with `go test ./...`.",
    }[ctx["lang"]]
    return (f"{head}Solve the {ctx['lang']} programming exercise in `{d}`.\n"
            f"Read the instructions in {docs} and implement the solution in {sol}. "
            f"Keep the names of the existing functions and classes in the stub, because the tests in {tests} "
            f"import them. Do not modify the test files. {how} "
            f"When the tests pass, reply with a one-line summary of your approach.")


def run_tests(ctx: dict) -> dict:
    """Run the exercise's test files with pytest in a clean interpreter (-I: no cwd/site
    customisation, so the exercise's own files are the only code that runs)."""
    d: Path = ctx["dir"]
    lang = ctx["lang"]
    env = None
    if lang == "python":
        cmd = [sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", *ctx["tests"]]
    elif lang == "javascript":
        jest = JS_NODE_MODULES / ".bin" / ("jest.cmd" if os.name == "nt" else "jest")
        cmd = [str(jest), "--ci", "--silent", "./*"]
    elif lang == "go":
        # full path: on Windows, subprocess resolves the executable with the PARENT's PATH, not env's
        cmd = [str(GO_BIN / ("go.exe" if os.name == "nt" else "go")), "test", "./..."]
        env = go_env(d.parent.parent)
    else:
        raise ValueError(lang)
    try:
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=PYTEST_TIMEOUT, env=env)
        out = (r.stdout or "") + (r.stderr or "")
        lines = [l for l in out.strip().splitlines() if l.strip()]
        if lang == "javascript":
            tail = next((l for l in lines if l.startswith("Tests:")), lines[-1] if lines else "")
        else:
            tail = lines[-1] if lines else ""
        m_pass = re.search(r"(\d+) passed", tail)
        m_fail = re.search(r"(\d+) (?:failed|error)", tail)
        return {"returncode": r.returncode, "passed": int(m_pass.group(1)) if m_pass else 0,
                "failed": int(m_fail.group(1)) if m_fail else (0 if r.returncode == 0 else None),
                "summary": tail[-200:], "timeout": False}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "passed": 0, "failed": None, "summary": "pytest timeout", "timeout": True}


def check(ctx: dict, final: str) -> tuple[float, dict]:
    """Score 1 iff the test files are untouched and every test passes. `final` is unused:
    the artefact, not the chat text, is what counts."""
    d: Path = ctx["dir"]
    untouched = all((d / t).exists() and _sha(d / t) == h for t, h in ctx["test_sha"].items())
    sol_present = all((d / s).exists() for s in ctx["solution"])
    res = run_tests(ctx) if untouched and sol_present else {"summary": "tests modified or solution missing"}
    ok = untouched and sol_present and res.get("returncode") == 0
    return float(ok), {"tests_untouched": untouched, "solution_present": sol_present, **res,
                       "exercise": ctx["name"], "lang": ctx["lang"]}


def parse_spec(spec: str) -> tuple[int, int, str]:
    """'poly:K@S' or 'poly:K@S:lang' -> (K, S, lang)."""
    body = spec.split(":", 1)[1]
    parts = body.split(":")
    k, _, seed = parts[0].partition("@")
    lang = parts[1] if len(parts) > 1 else "python"
    return int(k), int(seed or 0), lang


def build(a, tdir: Path) -> tuple[Path, list[dict]]:
    """run.py TASK_BUILDERS entry point: (workdir, turns) for a `poly:` task spec (`a.task`,
    or a plain spec string). Exercises are set up before the session starts, so the agent
    may look ahead; each turn names its exercise."""
    spec = a if isinstance(a, str) else a.task
    k, seed, lang = parse_spec(spec)
    workdir = tdir / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    names = select(k, seed, lang)
    turns = []
    for i, name in enumerate(names, 1):
        ctx = setup_exercise(workdir, name, lang)
        turns.append({"prompt": prompt(ctx, i if k > 1 else None), "kind": "question",
                      "check": (lambda c: lambda final: check(c, final))(ctx),
                      "meta": {"exercise": name, "lang": lang, "source": f"polyglot@{source_commit()}",
                               "n_tests": None}})
    return workdir, turns


def exercise_turn_factory(workdir: Path, name: str, lang: str):
    """Filler factory for bench.seqdep: `f(k) -> seqdep.Turn` whose checker runs the tests."""
    from bench import seqdep

    def make(k: int):
        ctx = setup_exercise(workdir, name, lang)
        return seqdep.Turn(k, "question", prompt(ctx, k), ground_truth=None,
                           meta={"exercise": name, "lang": lang, "role": "filler"},
                           check=lambda final: check(ctx, final))
    return make


def build_dep(a, tdir: Path) -> tuple[Path, list[dict]]:
    """`polydep:K@S[:lang]`: K exercises as the filler of a bench.seqdep dependent session
    (standing rules, stated facts, ephemeral observations; no derived edges, those need
    Context-Bench filler). The smallest session length n whose filler-slot count equals K
    is used, so every selected exercise appears exactly once."""
    from bench import seqdep
    spec = a if isinstance(a, str) else a.task
    k, seed, lang = parse_spec(spec)
    workdir = tdir / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    names = select(k, seed, lang)
    factories = [exercise_turn_factory(workdir, nm, lang) for nm in names]
    session = None
    for n in range(k + 2, k + 2 + 4 * max(5, k)):
        try:
            session = seqdep.build_session(n, seed, workdir, filler=factories, edge_types=("fact", "ephemeral"))
            break
        except seqdep.FillerMismatch:
            continue
    if session is None:
        raise SystemExit(f"polydep: no session length gives exactly {k} filler slots")
    (tdir / "seqdep_session.txt").write_text(session.describe(), encoding="utf-8")
    turns = []
    for t in session.turns:
        turns.append({"prompt": t.prompt, "kind": t.kind, "ground_truth": t.ground_truth,
                      "meta": dict(t.meta), "pre": (lambda kk: lambda: session.before_turn(kk))(t.turn),
                      "check": (lambda kk: lambda f, h=None: session.check(kk, f, h))(t.turn),
                      "needs_history": True,
                      "rules": (lambda kk: lambda f: session.rule_adherence(kk, f))(t.turn)})
    return workdir, turns


def oracle(tdir: Path, k: int, seed: int, lang: str = "python") -> list[dict]:
    """Checker self-test: the stub must score 0, the reference solution (.meta/example*) 1."""
    workdir, turns = build(f"poly:{k}@{seed}:{lang}", tdir)
    out = []
    for t in turns:
        name = t["meta"]["exercise"]
        ctx = setup_exercise(workdir, name, lang)
        s_stub, d_stub = check(ctx, "")
        cfg = config(lang, name)
        for ex, sol in zip(cfg["files"]["example"], cfg["files"]["solution"]):
            shutil.copy(SRC / lang / "exercises" / "practice" / name / ex, ctx["dir"] / sol)
        s_ref, d_ref = check(ctx, "")
        out.append({"exercise": name, "stub": s_stub, "ref": s_ref, "ref_passed": d_ref.get("passed"),
                    "ref_summary": d_ref.get("summary")})
    return out


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    lang = sys.argv[3] if len(sys.argv) > 3 else "python"
    print("track:", lang, "size:", len(exercises(lang)), "available:", track_available(lang), "commit:", source_commit())
    with tempfile.TemporaryDirectory() as tmp:
        for r in oracle(Path(tmp), k, seed, lang):
            flag = "OK " if (r["stub"] == 0 and r["ref"] == 1) else "BAD"
            print(flag, r)
        _, turns = build(f"poly:{k}@{seed}:{lang}", Path(tmp))
        print("\n--- first prompt ---\n" + turns[0]["prompt"])
