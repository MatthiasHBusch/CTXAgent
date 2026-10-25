"""SeqDep: dependent-session generator (REDESIGN.md section 3, T1).

One chat of N user turns: Context-Bench filler questions interleaved with dependency
edges whose probe at turn e + d can only be answered from the conversation at turn e.

    python -m bench.seqdep 20 0            # print the session for N=20, seed 0

Edge types (each with a deterministic checker):

  fact      -> probe_fact       the user states a labelled token; the probe asks for it
  rule      -> rule adherence   a standing instruction, checked on EVERY later turn
  question  -> probe_derived    a Context-Bench question with a person-name answer; the
                                probe asks a derived question about "the person you found
                                in Q<e>", ground truth computed from the dataset files
            control_derived     the same template with the name stated explicitly
  ephemeral -> probe_ephemeral  a planted status file with a random run id, read at turn e
                                and deleted before turn e+1 (the runner calls
                                `Session.before_turn(k)` ahead of every turn)

Distances come from DISTANCES; every distance that fits the session (d <= N - 2) is used
at least once. Everything is a deterministic function of (N, seed): the same session
script is sent to every harness, so comparisons are paired on (seed, turn).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from bench import contextbench as cb

DISTANCES = (3, 10, 25, 50, 90)
KINDS = ("question", "fact", "rule", "ephemeral", "probe_fact", "probe_derived", "control_derived",
         "probe_ephemeral", "probe_task")
ANSWER_LINE = "End your reply with one line of the form `ANSWER: <value>`."
FROM_MEMORY = "Answer from this conversation only; do not read files or run any tools for this."

WORDS = ["ORCHID", "BASALT", "CORVID", "DELPHI", "EMBER", "FALCON", "GRANITE", "HALCYON", "IRIS",
         "JASPER", "KESTREL", "LUMEN", "MARLIN", "NIMBUS", "OSPREY", "PUFFIN", "QUARTZ", "RAVEN",
         "SABLE", "TALON", "UMBER", "VERTEX", "WALNUT", "XENON", "YARROW", "ZEPHYR"]

# label -> token generator. One slot is used at most once per session.
FACT_SLOTS: list[tuple[str, Callable[[random.Random], str]]] = [
    ("project codename", lambda r: f"{r.choice(WORDS)}-{r.randint(10, 99)}"),
    ("client reference number", lambda r: f"{r.randint(1000, 9999)}-{r.choice('ABCDEFGH')}"),
    ("cost centre", lambda r: f"CC-{r.randint(100, 999)}"),
    ("build tag", lambda r: f"build-{r.randint(2000, 9999)}{r.choice('abcdef')}"),
    ("meeting room", lambda r: f"{r.choice(['Aurora', 'Baltic', 'Cedar', 'Delta', 'Ember'])} {r.randint(1, 9)}"),
    ("ticket id", lambda r: f"TCK-{r.randint(10000, 99999)}"),
    ("purchase order number", lambda r: f"PO-{r.randint(100000, 999999)}"),
    ("release train", lambda r: f"{r.choice(WORDS).title()} {r.randint(2027, 2029)}.{r.randint(1, 4)}"),
    ("escalation contact", lambda r: f"{r.choice(['Dana', 'Kim', 'Noor', 'Sasha', 'Tomas'])} "
                                     f"{r.choice(['Alder', 'Brook', 'Cole', 'Dunn', 'Frost'])}"),
    ("data room password hint", lambda r: f"{r.choice(WORDS).lower()}{r.randint(10, 99)}"),
    # more slots: mixed sessions (bench/tasks_mix.py) place up to ~15 fact edges
    ("vendor id", lambda r: f"VND-{r.randint(1000, 9999)}"),
    ("contract number", lambda r: f"K-{r.randint(2026, 2029)}-{r.randint(100, 999)}"),
    ("staging server hostname", lambda r: f"{r.choice(WORDS).lower()}-{r.randint(1, 99):02d}.internal"),
    ("API version to target", lambda r: f"v{r.randint(2, 9)}.{r.randint(0, 20)}"),
    ("budget code", lambda r: f"B{r.randint(10, 99)}-{r.choice('KLMNPQ')}{r.randint(100, 999)}"),
    ("incident number", lambda r: f"INC{r.randint(100000, 999999)}"),
    ("feature branch name", lambda r: f"feature/{r.choice(WORDS).lower()}-{r.randint(100, 999)}"),
    ("license key prefix", lambda r: f"{r.choice(WORDS)[:4]}-{r.randint(1000, 9999)}"),
    ("door access code", lambda r: f"{r.randint(1000, 9999)}#"),
    ("reviewer initials", lambda r: f"{r.choice('ABCDEFGHJKLMNPRST')}.{r.choice('ABCDEFGHJKLMNPRST')}.{r.choice('BDFGHKLMNPRSTVW')}."),
]

RULE_KINDS = ("ref_line", "prefix_word")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _answer_line(final: str) -> str | None:
    lines = re.findall(r"^\s*\**`?ANSWER\**:?\**\s*[:\-]?\s*(.+?)\s*`?\**\s*$", final or "", re.M | re.I)
    return lines[-1].strip().strip("`*") if lines else None


def _numbers(s: str) -> list[float]:
    out = []
    for m in re.finditer(r"-?\$?\d[\d,]*(?:\.\d+)?", s or ""):
        try:
            out.append(float(m.group(0).replace("$", "").replace(",", "")))
        except ValueError:
            pass
    return out


def check_token(token: str, final: str) -> tuple[float, dict]:
    """Fact / ephemeral probes: the token must appear in the reply (ANSWER line preferred)."""
    ans = _answer_line(final)
    hit = _norm(token) in _norm(ans) if ans else _norm(token) in _norm(final)
    return float(hit), {"want": token, "answer_line": ans, "matched": hit}


def check_value(truth: str, answer_type: str, final: str) -> tuple[float, dict]:
    """Derived probes: compare the ANSWER line (else the last number in the reply)."""
    ans = _answer_line(final)
    nums = _numbers(ans) if ans else _numbers(final)
    want = _numbers(truth)
    if not nums or not want:
        return 0.0, {"want": truth, "answer_line": ans, "parsed": nums}
    got = nums[-1] if not ans else nums[0]
    ok = abs(got - want[0]) < (0.005 if answer_type == "money" else 0.5)
    return float(ok), {"want": truth, "answer_line": ans, "parsed": got, "matched": ok}


def check_rule(rule: dict, final: str) -> bool:
    final = final or ""
    if rule["rule"] == "ref_line":
        return re.search(rf"^\W*REF\W*:\W*{re.escape(rule['token'])}\W*$", final, re.M | re.I) is not None
    if rule["rule"] == "prefix_word":
        first = next((l for l in final.splitlines() if l.strip()), "")
        return _norm(first.strip().strip("*`_#> ")) == _norm(rule["token"])
    raise KeyError(rule["rule"])


@dataclass
class Turn:
    turn: int
    kind: str
    prompt: str
    meta: dict = field(default_factory=dict)
    ground_truth: str | None = None
    plant: dict | None = None  # {"path": Path, "content": str} for ephemeral turns
    check: Callable | None = None  # custom filler turns (e.g. poly exercises) score themselves


@dataclass
class Session:
    n: int
    seed: int
    workdir: Path
    turns: list[Turn]
    rules: list[dict]       # {"id", "rule", "token", "turn"}
    edges: list[dict]       # {"type", "distance", "establish", "probe"}
    dropped: list[dict]     # edges that found no free slot pair
    judge: Callable = cb.judge

    # ---------------------------------------------------------------- runner hooks
    def before_turn(self, k: int) -> None:
        """Plant the ephemeral file of turn k, remove the files of all earlier turns."""
        for t in self.turns:
            if t.plant is None:
                continue
            p = Path(t.plant["path"])
            if t.turn == k:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(t.plant["content"], encoding="utf-8")
            elif t.turn < k and p.exists():
                p.unlink()

    def check(self, k: int, final: str, history: list[dict] | None = None) -> tuple[float, dict]:
        t = self.turns[k - 1]
        if t.check is not None:
            return t.check(final)
        if t.kind == "question":
            return self.judge({"question": t.meta["question"], "ground_truth": t.ground_truth}, final)
        if t.kind in ("fact", "rule"):
            return 1.0, {"acknowledged": bool((final or "").strip())}  # not scored as a task
        if t.kind in ("ephemeral", "probe_fact", "probe_ephemeral"):
            return check_token(t.ground_truth, final)
        if t.kind in ("probe_derived", "control_derived"):
            return check_value(t.ground_truth, t.meta["answer_type"], final)
        raise KeyError(t.kind)

    def rule_adherence(self, k: int, final: str) -> dict:
        """{rule_id: {"ok", "distance"}} for every rule established before turn k."""
        return {r["id"]: {"ok": check_rule(r, final), "distance": k - r["turn"], "rule": r["rule"]}
                for r in self.rules if r["turn"] < k}

    # ---------------------------------------------------------------- reporting
    def describe(self) -> str:
        lines = [f"seqdep N={self.n} seed={self.seed} workdir={self.workdir}",
                 f"edges: {len(self.edges)} (dropped {len(self.dropped)}), distances used: "
                 f"{sorted({e['distance'] for e in self.edges})}, rules: "
                 f"{[(r['id'], r['rule'], r['turn']) for r in self.rules]}",
                 f"{'turn':>4} {'kind':18} {'d':>3} {'from':>4}  {'ground truth':28} prompt"]
        for t in self.turns:
            d = t.meta.get("distance", "")
            e = t.meta.get("establish_turn", "")
            gt = (t.ground_truth or "")[:28]
            head = t.prompt.split("Question:")[-1].strip() if t.kind == "question" else t.prompt
            head = re.sub(r"\s+", " ", head)[:90]
            lines.append(f"{t.turn:>4} {t.kind:18} {str(d):>3} {str(e):>4}  {gt:28} {head}")
        return "\n".join(lines)


# ------------------------------------------------------------------- generation
def usable_distances(n: int, distances=DISTANCES) -> list[int]:
    return [d for d in distances if d <= n - 2]


class FillerMismatch(ValueError):
    """Raised when a custom filler list does not match the number of filler slots; carries
    the number of slots so the caller can pick another n."""

    def __init__(self, needed: int, given: int):
        super().__init__(f"session needs {needed} filler turns, {given} given")
        self.needed, self.given = needed, given


def build_session(n: int, seed: int, workdir: Path | str, distances=DISTANCES,
                  edges_per_turn: float = 0.2, filler: list[Callable[[int], "Turn"]] | None = None,
                  edge_types: tuple[str, ...] = ("fact", "derived", "ephemeral")) -> Session:
    """Deterministic session script for (n, seed). `workdir` is where the Context-Bench
    files live for this trial (ephemeral files are planted under <workdir>/jobs/).

    `filler`: optional list of factories `f(k) -> Turn` for the filler slots (e.g. coding
    exercises from bench/tasks_polyglot.py) instead of Context-Bench questions; its length
    must equal the number of filler slots (FillerMismatch otherwise). Derived edges need
    Context-Bench filler, so pass edge_types=("fact", "ephemeral") with a custom filler."""
    if filler is not None and "derived" in edge_types:
        raise ValueError("derived edges need Context-Bench filler; use edge_types=('fact', 'ephemeral')")
    if n < 6:
        raise ValueError("seqdep sessions need at least 6 turns")
    workdir = Path(workdir)
    rng = random.Random(f"seqdep:{n}:{seed}")
    dists = usable_distances(n, distances)
    if not dists:
        raise ValueError(f"no distance in {distances} fits a session of {n} turns")

    # --- rules: early, so that adherence is observed over most of the session
    n_rules = 1 if n < 40 else 2
    rule_kinds = list(RULE_KINDS)
    rng.shuffle(rule_kinds)
    rule_slots = sorted(rng.sample(range(2, max(4, n // 4) + 1), n_rules))
    occupied: dict[int, str] = {}
    rules = []
    for i, (slot, rk) in enumerate(zip(rule_slots, rule_kinds)):
        token = f"{rng.choice(WORDS)}-{rng.randint(100, 999)}" if rk == "ref_line" else \
            f"{rng.choice(['NOTED', 'ACK', 'CHECK', 'LOGGED'])}-{rng.randint(10, 99)}"
        rules.append({"id": f"rule{i + 1}", "rule": rk, "token": token, "turn": slot})
        occupied[slot] = "rule"

    # --- edges: types cycle, every usable distance at least once, the rest random
    n_edges = max(len(dists), round(n * edges_per_turn))
    types = list(edge_types)
    rng.shuffle(types)
    etypes = [types[i % len(types)] for i in range(n_edges)]
    rng.shuffle(etypes)
    dlist = list(dists)
    rng.shuffle(dlist)
    dlist += [rng.choice(dists) for _ in range(n_edges - len(dlist))]
    wanted = sorted(zip(etypes, dlist), key=lambda x: -x[1])  # longest first: hardest to place
    edges, dropped = [], []
    for et, d in wanted:
        cands = [e for e in range(1, n - d + 1) if e not in occupied and e + d not in occupied]
        if not cands:
            dropped.append({"type": et, "distance": d})
            continue
        e = rng.choice(cands)
        occupied[e] = "question" if et == "derived" else et
        occupied[e + d] = "probe"
        edges.append({"type": et, "distance": d, "establish": e, "probe": e + d})
    edges.sort(key=lambda x: x["establish"])

    # --- controls for derived edges: same template, explicit name, any free slot
    templates = list(cb.DERIVED_TEMPLATES)
    rng.shuffle(templates)
    derived = [e for e in edges if e["type"] == "derived"]
    for i, e in enumerate(derived):
        e["template"] = templates[i % len(templates)]
        free = [s for s in range(1, n + 1) if s not in occupied]
        if free:
            s = rng.choice(free)
            occupied[s] = "control"
            e["control"] = s

    # --- Context-Bench questions: name-answer ids for derived bases, stratified rest
    n_filler = n - len(occupied)
    if filler is not None and len(filler) != n_filler:
        raise FillerMismatch(n_filler, len(filler))
    name_ids = cb.name_answer_ids()
    rng.shuffle(name_ids)
    base_ids = name_ids[:len(derived)]
    pool = [q for q in cb.stratified(min(n_filler + len(base_ids), 96), seed) if q not in base_ids]
    rng.shuffle(pool)
    while len(pool) < n_filler:  # only for n close to the pool size
        pool += [q for q in cb.stratified(96, seed + 1000) if q not in base_ids]
    qs = cb.questions()
    control_people = [qs[i]["ground_truth"] for i in name_ids[len(derived):]]

    # --- assemble
    turns: list[Turn] = []
    fact_slots = list(FACT_SLOTS)
    rng.shuffle(fact_slots)
    by_establish = {e["establish"]: e for e in edges}
    by_probe = {e["probe"]: e for e in edges}
    by_control = {e["control"]: e for e in edges if e.get("control")}
    by_rule = {r["turn"]: r for r in rules}
    fi = 0

    def question_turn(k: int, qid: int, role: str) -> Turn:
        q = qs[qid]
        text = q["input"].replace("{pwd}", str(workdir))
        return Turn(k, "question", f"Q{k}: {text}", ground_truth=str(q["ground_truth"]),
                    meta={"qid": qid, "qtype": q["qtype"], "difficulty": q["difficulty"], "role": role,
                          "question": q["input"], "required_files": q["agent_args"]["extra"].get("required_files")})

    for k in range(1, n + 1):
        if k in by_rule:
            r = by_rule[k]
            if r["rule"] == "ref_line":
                text = (f"Q{k}: Standing instruction for the rest of this session: include a line `REF: "
                        f"{r['token']}` in every reply from now on, whatever the question. Confirm briefly.")
            else:
                text = (f"Q{k}: Standing instruction for the rest of this session: start every reply from now "
                        f"on with the exact word `{r['token']}` on its own first line. Confirm briefly.")
            turns.append(Turn(k, "rule", text, ground_truth=r["token"],
                              meta={"rule_id": r["id"], "rule": r["rule"]}))
        elif k in by_establish:
            e = by_establish[k]
            if e["type"] == "fact":
                if not fact_slots:
                    raise ValueError(f"seqdep: more fact edges than FACT_SLOTS ({len(FACT_SLOTS)}); shorten the session")
                label, gen = fact_slots.pop()
                token = gen(rng)
                e["token"], e["label"] = token, label
                turns.append(Turn(k, "fact", f"Q{k}: For later reference: the {label} for this engagement is "
                                             f"{token}. No action needed now; just confirm you noted it.",
                                  ground_truth=token, meta={"label": label, "probe_turn": e["probe"],
                                                            "distance": e["distance"]}))
            elif e["type"] == "ephemeral":
                token = "".join(rng.choice("0123456789abcdef") for _ in range(8))
                path = workdir / "jobs" / f"job-{k}.status"
                e["token"] = token
                content = (f"job: nightly-export-{k}\nstatus: finished\nrun_id: {token}\n"
                           f"elapsed_s: {rng.randint(40, 900)}\n")
                turns.append(Turn(k, "ephemeral",
                                  f"Q{k}: A background job has just finished and wrote its status file to "
                                  f"{path}. Read that file now and tell me the run_id it reports (the file is "
                                  f"cleaned up automatically afterwards). {ANSWER_LINE}",
                                  ground_truth=token, plant={"path": str(path), "content": content},
                                  meta={"path": str(path), "probe_turn": e["probe"], "distance": e["distance"]}))
            else:  # derived base: a normal question whose answer is a person
                qid = base_ids.pop()
                e["base_qid"], e["base_truth"] = qid, str(qs[qid]["ground_truth"])
                t = question_turn(k, qid, "derived_base")
                t.meta.update(probe_turn=e["probe"], distance=e["distance"])
                turns.append(t)
        elif k in by_probe:
            e = by_probe[k]
            if e["type"] == "fact":
                turns.append(Turn(k, "probe_fact",
                                  f"Q{k}: Earlier in this session I gave you the {e['label']}. What was it? "
                                  f"{FROM_MEMORY} {ANSWER_LINE}",
                                  ground_truth=e["token"],
                                  meta={"establish_turn": e["establish"], "distance": e["distance"],
                                        "label": e["label"]}))
            elif e["type"] == "ephemeral":
                turns.append(Turn(k, "probe_ephemeral",
                                  f"Q{k}: What run_id did the job status file in Q{e['establish']} report? The "
                                  f"file no longer exists. {FROM_MEMORY} {ANSWER_LINE}",
                                  ground_truth=e["token"],
                                  meta={"establish_turn": e["establish"], "distance": e["distance"]}))
            else:
                dq = cb.derive(e["template"], e["base_truth"],
                               f"the person you identified as the answer to Q{e['establish']}")
                turns.append(Turn(k, "probe_derived",
                                  f"Q{k}: {dq['question']} Use the data files under {workdir} for the "
                                  f"lookup, but take the person from your own earlier answer to "
                                  f"Q{e['establish']} in this conversation. {ANSWER_LINE}",
                                  ground_truth=dq["ground_truth"],
                                  meta={"establish_turn": e["establish"], "distance": e["distance"],
                                        "template": e["template"], "answer_type": dq["answer_type"],
                                        "base_qid": e["base_qid"], "base_truth": e["base_truth"],
                                        "control_turn": e.get("control")}))
        elif k in by_control:
            e = by_control[k]
            name = control_people.pop() if control_people else e["base_truth"]
            dq = cb.derive(e["template"], name, name)
            turns.append(Turn(k, "control_derived",
                              f"Q{k}: {dq['question']} Use the data files under {workdir}. {ANSWER_LINE}",
                              ground_truth=dq["ground_truth"],
                              meta={"template": e["template"], "answer_type": dq["answer_type"],
                                    "name": name, "control_of": e["probe"]}))
        elif filler is not None:
            t = filler[fi](k)
            t.turn = k
            t.meta.setdefault("role", "filler")
            turns.append(t)
            fi += 1
        else:
            turns.append(question_turn(k, pool[fi], "filler"))
            fi += 1
    assert len(turns) == n and [t.turn for t in turns] == list(range(1, n + 1))
    return Session(n=n, seed=seed, workdir=workdir, turns=turns, rules=rules, edges=edges, dropped=dropped)


def parse_spec(spec: str) -> tuple[int, int]:
    """'seqdep:N@S' -> (N, S)."""
    body = spec.split(":", 1)[1]
    n, _, seed = body.partition("@")
    return int(n), int(seed or 0)


if __name__ == "__main__":
    import sys
    import tempfile
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(build_session(n, seed, Path(tempfile.gettempdir()) / "seqdep-dry" / "data").describe())
