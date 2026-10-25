"""Hard mixed sessions: the harder counterpart of bench.tasks_mix (`mixdep:`), ONE chat of
heavy, dependent work with the standing rules, stated facts and ephemeral observations of
bench.seqdep on top, plus task-derived recall probes.

    hardmix:K@S           K exercises from the Go and JavaScript tracks + SciCode TEST-split
                          problems (>= 3 steps each, HARD_PROBLEMS_PER_SEED per seed), merged
                          in a seeded order that keeps each problem's steps in sequence;
                          seqdep fact / ephemeral edges and rules interleaved; after every
                          finished SciCode problem a recall probe at a seeded distance asks,
                          from memory, for the name of the function written for one of its
                          steps (checked against the step metadata)

The three seeds use DISJOINT task sets: the candidate lists are shuffled once with a fixed
generator and cut into consecutive chunks, seed s taking chunk s. The container-backed
sources (SWE-bench Pro V2, Terminal-Bench 4.0; bench/tasks_heavy.py) form the `heavy:S`
family built by `build_heavy` below (bench/RUN_PLAN.md section 8); each source yields filler
factories `f(k) -> seqdep.Turn`.

Dry run:  python -m bench.run show --task hardmix:6@0
"""
from __future__ import annotations

import os
import random
from pathlib import Path

from bench import seqdep, tasks_polyglot as poly, tasks_scicode as sci

HARD_PROBLEMS_PER_SEED = 8     # SciCode test problems per session (3 seeds x 8 = 24 of 65)
SCI_MIN_STEPS = 3
HARD_TRACKS = ("go", "javascript")
RECALL_DISTANCES = (3, 10, 25, 50)  # turns between a problem's last step and its recall probe
SPLIT = "test"


def parse_spec(spec: str) -> tuple[int, int]:
    body = spec.split(":", 1)[1]
    k, _, seed = body.partition("@")
    return int(k), int(seed or 0)


def _chunk(items: list, seed: int, size: int, salt: str) -> list:
    """Seed-disjoint selection: one fixed shuffle, consecutive chunks of `size`, chunk `seed`."""
    rng = random.Random(f"hardmix:{salt}")
    items = list(items)
    rng.shuffle(items)
    start = (seed * size) % max(len(items), 1)
    out = items[start:start + size]
    if len(out) < size:  # wrap around only when the pool is exhausted (seed >= pool / size)
        out += items[:size - len(out)]
    return out


def choose_problems(seed: int) -> list[str]:
    probs = sci.problems(SPLIT)
    cands = sorted((pid for pid, p in probs.items() if len(p["sub_steps"]) >= SCI_MIN_STEPS), key=int)
    return _chunk(cands, seed, HARD_PROBLEMS_PER_SEED, "sci-test")


def choose_exercises(seed: int, k: int) -> list[tuple[str, str]]:
    """k exercises spread over the available hard tracks, seed-disjoint within each track."""
    tracks = [t for t in HARD_TRACKS if poly.track_available(t)]
    if not tracks:
        raise SystemExit("hardmix: neither the Go nor the JavaScript toolchain is available (bench/external/fetch.*)")
    per = [k // len(tracks) + (1 if i < k % len(tracks) else 0) for i in range(len(tracks))]
    out = []
    for lang, n in zip(tracks, per):
        out += [(lang, nm) for nm in _chunk(poly.exercises(lang), seed, n, f"poly-{lang}")]
    return out


def recall_probe_factory(prob: dict, step: dict, step_index: int):
    """`f(k) -> Turn`: ask, from memory, for the function name of one step of a finished problem."""
    name = sci.function_name(step["function_header"])
    n = len(prob["sub_steps"])
    topic = step["step_description_prompt"].strip().split("\n")[0][:160]

    def make(k: int):
        prompt = (f"Q{k}: A question about work you did earlier in this chat, to be answered from memory "
                  f"(do not open files, do not use tools): in the {prob['problem_name'].replace('_', ' ')} problem, "
                  f"what was the name of the function you wrote for step {step_index} of {n}, the step that asked: "
                  f"\"{topic}\"? Reply with the function name only.")

        def check(final: str):
            hit = name in (final or "")
            return float(hit), {"recall": "function", "expected": name, "problem_id": prob["problem_id"]}
        return seqdep.Turn(k, "probe_task", prompt, ground_truth=name,
                           meta={"problem_id": prob["problem_id"], "step": step["step_number"],
                                 "function": name, "role": "probe_task"}, check=check)
    return make


HEAVY_SCI_MIN, HEAVY_SCI_MAX = 3, 5


def parse_heavy(spec: str) -> tuple[int, int, int, int]:
    """'heavy:S' | 'heavy:S:3+3+3' -> (seed, n_swe, n_tb, n_sci)."""
    body = spec.split(":", 1)[1]
    parts = body.split(":")
    seed = int(parts[0] or 0)
    n = [int(x) for x in parts[1].split("+")] if len(parts) > 1 and parts[1] else [3, 3, 3]
    return seed, n[0], n[1], n[2]


def choose_verified_problems(seed: int, n: int) -> list[str]:
    probs = sci.problems("verified")
    cands = sorted((pid for pid, p in probs.items() if HEAVY_SCI_MIN <= len(p["sub_steps"]) <= HEAVY_SCI_MAX), key=int)
    return _chunk(cands, seed, n, "sci-verified")


def build_heavy(a, tdir: Path) -> tuple[Path, list[dict]]:
    """run.py TASK_BUILDERS entry point for `heavy:` specs: SWE-bench Pro V2 instances,
    Terminal-Bench 4.0 tasks and SciCode-Verified problems in one chat, with the seqdep edges,
    two rules and the SciCode recall probes. Needs Docker (bench/tasks_heavy.py)."""
    from bench import tasks_heavy as heavy
    spec = a if isinstance(a, str) else a.task
    tag = safe_tag(getattr(a, "tag", None) or f"show-{os.getpid()}")
    seed, n_swe, n_tb, n_sci = parse_heavy(spec)
    workdir = tdir / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    scratch = tdir / "sci_tests"
    logdir = tdir / "heavy_logs"
    rng = random.Random(f"heavy:{seed}:{n_swe}+{n_tb}+{n_sci}")
    streams: list[list] = []
    if n_swe:
        streams.append(heavy.swe_factories(seed, n_swe, tag, workdir, logdir))
    if n_tb:
        streams.append(heavy.tb_factories(seed, n_tb, tag, workdir, logdir))
    probs = sci.problems("verified")
    pids = choose_verified_problems(seed, n_sci)
    sci_streams, recalls = [], {}
    for pid in pids:
        steps = sci.step_factories(workdir, scratch, pid, "verified")
        sci_streams.append(steps)
        prob = probs[pid]
        cands = [i for i, st in enumerate(prob["sub_steps"], 1)
                 if not sci.function_name(st["function_header"]).startswith("__")] or [1]
        i = rng.choice(cands)
        recalls[id(steps[-1])] = (recall_probe_factory(prob, prob["sub_steps"][i - 1], i), rng.choice(RECALL_DISTANCES))
    streams += sci_streams
    slots = [i for i, st in enumerate(streams) for _ in st]
    rng.shuffle(slots)
    iters = [iter(st) for st in streams]
    factories = [next(iters[i]) for i in slots]
    by_key = {id(ff): ff for ff in factories}
    for key, (f, d) in sorted(recalls.items(), key=lambda kv: factories.index(by_key[kv[0]])):
        factories.insert(min(factories.index(by_key[key]) + d, len(factories)), f)
    session = None
    for n in range(len(factories) + 2, len(factories) + 2 + 4 * len(factories)):
        try:
            session = seqdep.build_session(n, seed, workdir, filler=factories, edge_types=("fact", "ephemeral"))
            break
        except seqdep.FillerMismatch:
            continue
    if session is None:
        raise SystemExit(f"heavy: no session length gives exactly {len(factories)} filler slots")
    n_steps = sum(len(st) for st in sci_streams)
    desc = (session.describe() + f"\n\nSWE-bench Pro: {heavy.swe_selection(seed)[:n_swe]}; Terminal-Bench 4.0: "
            f"{heavy.tb_selection(seed)[:n_tb]}; SciCode-Verified problems: {pids} ({n_steps} steps, {len(pids)} recall probes); "
            f"containers labelled {heavy.LABEL}={tag}")
    (tdir / "seqdep_session.txt").write_text(desc, encoding="utf-8")
    heavy.register_cleanup(tag)
    turns = []
    for t in session.turns:
        start = t.meta.pop("_start", None)  # container start hook of a heavy task (not serialisable)
        turns.append({"prompt": t.prompt, "kind": t.kind, "ground_truth": t.ground_truth,
                      "meta": dict(t.meta),
                      "pre": (lambda kk, st: lambda: (session.before_turn(kk), st() if st else None))(t.turn, start),
                      "check": (lambda kk: lambda f, h=None: session.check(kk, f, h))(t.turn),
                      "needs_history": True,
                      "rules": (lambda kk: lambda f: session.rule_adherence(kk, f))(t.turn)})
    turns[0]["teardown"] = lambda: heavy.cleanup(tag)
    return workdir, turns


def safe_tag(tag: str) -> str:
    """Trial tag as a container label value: sanitised, never truncated (the two arms of a seed
    differ only after the 40th character; truncating made them share containers)."""
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "-", str(tag))


def build(a, tdir: Path) -> tuple[Path, list[dict]]:
    """run.py TASK_BUILDERS entry point for `hardmix:` specs."""
    spec = a if isinstance(a, str) else a.task
    k, seed = parse_spec(spec)
    workdir = tdir / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    scratch = tdir / "sci_tests"
    rng = random.Random(f"hardmix:{k}:{seed}")
    probs = sci.problems(SPLIT)
    pids = choose_problems(seed)
    exercises = choose_exercises(seed, k)
    # streams: one per exercise track (each in its own folder: exercise names repeat across
    # tracks) and one per SciCode problem (steps in order, then its recall probe at a
    # seeded distance, enforced below)
    streams: list[list] = []
    for lang in HARD_TRACKS:
        names = [nm for lg, nm in exercises if lg == lang]
        if names:
            streams.append([poly.exercise_turn_factory(workdir / lang, nm, lang) for nm in names])
    sci_streams, recalls = [], {}
    for pid in pids:
        steps = sci.step_factories(workdir, scratch, pid, SPLIT)
        sci_streams.append(steps)
        prob = probs[pid]
        # recall target: a step whose function name is a plain identifier (no dunder constructors,
        # which are not a memorable name) -- the smoke of 2026-10-22 lost one probe to `__init__`
        cands = [i for i, st in enumerate(prob["sub_steps"], 1)
                 if not sci.function_name(st["function_header"]).startswith("__")] or [1]
        i = rng.choice(cands)
        recalls[id(steps[-1])] = (recall_probe_factory(prob, prob["sub_steps"][i - 1], i), rng.choice(RECALL_DISTANCES))
    streams += sci_streams
    slots = [i for i, st in enumerate(streams) for _ in st]
    rng.shuffle(slots)
    iters = [iter(st) for st in streams]
    factories = [next(iters[i]) for i in slots]
    # insert each problem's recall probe `d` filler turns after its last step (clamped to the
    # end); the position is looked up at insertion time, so earlier insertions are accounted for
    by_key = {id(ff): ff for ff in factories}
    for key, (f, d) in sorted(recalls.items(), key=lambda kv: factories.index(by_key[kv[0]])):
        at = min(factories.index(by_key[key]) + d, len(factories))
        factories.insert(at, f)
    session = None
    for n in range(len(factories) + 2, len(factories) + 2 + 4 * len(factories)):
        try:
            session = seqdep.build_session(n, seed, workdir, filler=factories, edge_types=("fact", "ephemeral"))
            break
        except seqdep.FillerMismatch:
            continue
    if session is None:
        raise SystemExit(f"hardmix: no session length gives exactly {len(factories)} filler slots")
    n_steps = sum(len(st) for st in sci_streams)
    desc = (session.describe() + f"\n\nSciCode {SPLIT} problems: {pids} ({n_steps} steps, {len(pids)} recall probes); "
            f"exercises: {exercises}")
    (tdir / "seqdep_session.txt").write_text(desc, encoding="utf-8")
    turns = []
    for t in session.turns:
        turns.append({"prompt": t.prompt, "kind": t.kind, "ground_truth": t.ground_truth,
                      "meta": dict(t.meta), "pre": (lambda kk: lambda: session.before_turn(kk))(t.turn),
                      "check": (lambda kk: lambda f, h=None: session.check(kk, f, h))(t.turn),
                      "needs_history": True,
                      "rules": (lambda kk: lambda f: session.rule_adherence(kk, f))(t.turn)})
    return workdir, turns
