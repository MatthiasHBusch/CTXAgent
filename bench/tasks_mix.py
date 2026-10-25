"""Mixed dependent sessions: Exercism exercises (bench.tasks_polyglot) and SciCode
subproblems (bench.tasks_scicode, dev split) interleaved in ONE chat, with the standing
rules, stated facts and ephemeral observations of bench.seqdep on top.

    mixdep:K@S            K Python exercises + SciCode dev problems chosen by seed S
                          (problems with >= 3 steps, added until >= SCI_STEPS_TARGET steps),
                          merged in a seeded random order that keeps each problem's steps
                          in sequence; seqdep edges (fact / ephemeral) and rules interleaved
    mixdep:K@S:10+70      explicit SciCode problem ids instead of the seeded choice
    marathon:S            everything verified in ONE chat: all Python, Go and JavaScript
                          exercises whose toolchain is present, all 15 SciCode dev problems
                          (50 steps), facts/ephemerals at distances up to 200, 2 rules;
                          ~220 turns, roughly 1.5 Mtok of raw history

Why: the P1 pilot (bench/results/p1_pilot.jsonl) showed that 20 exercises grow the
context to ~100k tokens and nobody compacts; 34 exercises plus ~25 SciCode steps reach the
range where the native harnesses compact and ctxagent's own policy has to act, and the
SciCode steps add intrinsically dependent, genuinely hard turns (frontier ~60 % per step).

Dry run:  python -m bench.run show --task mixdep:34@0
"""
from __future__ import annotations

import random
from pathlib import Path

from bench import seqdep, tasks_polyglot as poly, tasks_scicode as sci

SCI_STEPS_TARGET = 24
SCI_MIN_STEPS = 3


def parse_spec(spec: str) -> tuple[int, int, list[str] | None]:
    body = spec.split(":", 1)[1]
    parts = body.split(":")
    k, _, seed = parts[0].partition("@")
    ids = parts[1].split("+") if len(parts) > 1 and parts[1] else None
    return int(k), int(seed or 0), ids


def choose_problems(seed: int, split: str = "dev") -> list[str]:
    probs = sci.problems(split)
    cands = sorted((pid for pid, p in probs.items() if len(p["sub_steps"]) >= SCI_MIN_STEPS), key=int)
    rng = random.Random(f"mixdep-sci:{seed}")
    rng.shuffle(cands)
    out, steps = [], 0
    for pid in cands:
        if steps >= SCI_STEPS_TARGET:
            break
        out.append(pid)
        steps += len(probs[pid]["sub_steps"])
    return out


def build(a, tdir: Path) -> tuple[Path, list[dict]]:
    """run.py TASK_BUILDERS entry point for `mixdep:` specs."""
    spec = a if isinstance(a, str) else a.task
    k, seed, ids = parse_spec(spec)
    ids = ids or choose_problems(seed)
    workdir = tdir / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    scratch = tdir / "sci_tests"
    rng = random.Random(f"mixdep:{k}:{seed}")
    ex = [poly.exercise_turn_factory(workdir, nm, "python") for nm in poly.select(k, seed, "python")]
    steps = []
    for pid in ids:
        steps += sci.step_factories(workdir, scratch, pid, "dev")
    # seeded merge of the two streams, each keeping its internal order
    slots = ["ex"] * len(ex) + ["sci"] * len(steps)
    rng.shuffle(slots)
    ex_it, st_it = iter(ex), iter(steps)
    factories = [next(ex_it) if s == "ex" else next(st_it) for s in slots]
    session = None
    for n in range(len(factories) + 2, len(factories) + 2 + 4 * len(factories)):
        try:
            session = seqdep.build_session(n, seed, workdir, filler=factories, edge_types=("fact", "ephemeral"))
            break
        except seqdep.FillerMismatch:
            continue
    if session is None:
        raise SystemExit(f"mixdep: no session length gives exactly {len(factories)} filler slots")
    desc = session.describe() + f"\n\nSciCode dev problems: {ids} ({len(steps)} steps); exercises: {k}"
    (tdir / "seqdep_session.txt").write_text(desc, encoding="utf-8")
    turns = []
    for t in session.turns:
        turns.append({"prompt": t.prompt, "kind": t.kind, "ground_truth": t.ground_truth,
                      "meta": dict(t.meta), "pre": (lambda kk: lambda: session.before_turn(kk))(t.turn),
                      "check": (lambda kk: lambda f, h=None: session.check(kk, f, h))(t.turn),
                      "needs_history": True,
                      "rules": (lambda kk: lambda f: session.rule_adherence(kk, f))(t.turn)})
    return workdir, turns


MARATHON_DISTANCES = (3, 10, 25, 50, 90, 150, 200)
MARATHON_EDGES_PER_TURN = 0.1  # ~22 edges on ~220 turns; FACT_SLOTS has 20 labels


def build_marathon(a, tdir: Path) -> tuple[Path, list[dict]]:
    """run.py TASK_BUILDERS entry point for `marathon:S`."""
    spec = a if isinstance(a, str) else a.task
    seed = int(spec.split(":", 1)[1] or 0)
    workdir = tdir / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    scratch = tdir / "sci_tests"
    rng = random.Random(f"marathon:{seed}")
    streams = []
    tracks = [t for t in poly.TRACKS if poly.track_available(t)]
    for lang in tracks:
        names = poly.exercises(lang)
        rng.shuffle(names)
        # one folder per track: the same exercise name exists in several tracks, and
        # setup_exercise replaces workdir/<name> (seed-0 run of 2026-10-20 lost 51 exercises
        # to this before the fix)
        streams.append([poly.exercise_turn_factory(workdir / lang, nm, lang) for nm in names])
    probs = sci.problems("dev")
    pids = sorted(probs, key=int)
    rng.shuffle(pids)
    steps = []
    for pid in pids:
        steps += sci.step_factories(workdir, scratch, pid, "dev")
    streams.append(steps)
    # seeded merge of all streams, each keeping its internal order
    slots = [i for i, st in enumerate(streams) for _ in st]
    rng.shuffle(slots)
    iters = [iter(st) for st in streams]
    factories = [next(iters[i]) for i in slots]
    session = None
    for n in range(len(factories) + 2, len(factories) + 2 + 2 * len(factories)):
        try:
            session = seqdep.build_session(n, seed, workdir, distances=MARATHON_DISTANCES,
                                           edges_per_turn=MARATHON_EDGES_PER_TURN, filler=factories,
                                           edge_types=("fact", "ephemeral"))
            break
        except seqdep.FillerMismatch:
            continue
    if session is None:
        raise SystemExit(f"marathon: no session length gives exactly {len(factories)} filler slots")
    desc = (session.describe() + f"\n\ntracks: {tracks} ({[len(st) for st in streams[:-1]]} exercises); "
            f"SciCode dev problems: {pids} ({len(steps)} steps)")
    (tdir / "seqdep_session.txt").write_text(desc, encoding="utf-8")
    turns = []
    for t in session.turns:
        turns.append({"prompt": t.prompt, "kind": t.kind, "ground_truth": t.ground_truth,
                      "meta": dict(t.meta), "pre": (lambda kk: lambda: session.before_turn(kk))(t.turn),
                      "check": (lambda kk: lambda f, h=None: session.check(kk, f, h))(t.turn),
                      "needs_history": True,
                      "rules": (lambda kk: lambda f: session.rule_adherence(kk, f))(t.turn)})
    return workdir, turns
