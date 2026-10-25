"""SeqDep generator (bench/seqdep.py): determinism, distance coverage, solvability.

Solvability is shown by an oracle that only sees the conversation (the prompts and its
own earlier replies) plus the data files, and answers every probe correctly; an
amnesiac oracle without the conversation fails every probe. No model calls.

    python -m tests.test_seqdep
"""
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import contextbench as cb  # noqa: E402
from bench import seqdep  # noqa: E402

HAVE_DATA = cb.SRC.exists()


def fake_judge(ctx, final):
    """Deterministic stand-in for the LLM judge: ground truth appears in the reply."""
    return float(str(ctx["ground_truth"]).lower() in (final or "").lower()), {}


def oracle_reply(session: seqdep.Session, k: int, history: list[dict], amnesiac: bool = False) -> str:
    """What an agent with a perfect memory of the conversation would reply at turn k."""
    t = session.turns[k - 1]
    prompt = t.prompt
    if t.kind == "question":
        body = f"The answer is {t.ground_truth}.\nANSWER: {t.ground_truth}"
    elif t.kind in ("fact", "rule"):
        body = "Noted."
    elif t.kind == "ephemeral":
        path = re.search(r"status file to (.+?)\. Read", prompt).group(1)
        text = Path(path).read_text(encoding="utf-8")
        rid = re.search(r"run_id: (\S+)", text).group(1)
        body = f"The status file reports run_id {rid}.\nANSWER: {rid}"
    elif amnesiac:
        body = "I do not remember.\nANSWER: unknown"
    elif t.kind == "probe_fact":
        label = re.search(r"gave you the (.+?)\. What", prompt).group(1)
        src = next(h for h in history if h["kind"] == "fact" and f"the {label} for this engagement is " in h["prompt"])
        token = re.search(r"engagement is (.+?)\. No action", src["prompt"]).group(1)
        body = f"ANSWER: {token}"
    elif t.kind == "probe_ephemeral":
        e = int(re.search(r"file in Q(\d+) report", prompt).group(1))
        rid = re.search(r"ANSWER: (\S+)", history[e - 1]["final"]).group(1)
        body = f"ANSWER: {rid}"
    elif t.kind == "probe_derived":
        e = int(re.search(r"answer to Q(\d+)", prompt).group(1))
        name = re.search(r"ANSWER: (.+)", history[e - 1]["final"]).group(1).strip()
        truth = cb.derive(t.meta["template"], name, name)["ground_truth"]
        body = f"Looking that up for {name}: {truth}.\nANSWER: {truth}"
    elif t.kind == "control_derived":
        body = f"ANSWER: {t.ground_truth}"
    else:
        raise AssertionError(t.kind)
    # standing rules seen so far (from the prompts, as an agent would)
    for h in history:
        if h["kind"] != "rule":
            continue
        m = re.search(r"include a line `(REF: [^`]+)`", h["prompt"])
        if m:
            body += f"\n{m.group(1)}"
        m = re.search(r"exact word `([^`]+)` on its own first line", h["prompt"])
        if m:
            body = f"{m.group(1)}\n{body}"
    return body


def run_session(session: seqdep.Session, amnesiac: bool = False) -> tuple[list[tuple], list[dict]]:
    """Drive the session the way the runner does: pre-turn hook, reply, check, rules."""
    session.judge = fake_judge
    history, results = [], []
    for t in session.turns:
        session.before_turn(t.turn)
        final = oracle_reply(session, t.turn, history, amnesiac)
        score, details = session.check(t.turn, final, history)
        rules = session.rule_adherence(t.turn, final)
        results.append((t.turn, t.kind, score, rules, details))
        history.append({"turn": t.turn, "kind": t.kind, "prompt": t.prompt, "final": final, "meta": t.meta})
    return results, history


@unittest.skipUnless(HAVE_DATA, "Context-Bench files not present under bench/external")
class SeqDepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="seqdep-test-")
        self.work = Path(self.tmp.name) / "data"
        self.work.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_deterministic_per_seed(self):
        a = seqdep.build_session(60, 3, self.work)
        b = seqdep.build_session(60, 3, self.work)
        self.assertEqual([(t.kind, t.prompt, t.ground_truth, t.meta) for t in a.turns],
                         [(t.kind, t.prompt, t.ground_truth, t.meta) for t in b.turns])
        self.assertEqual(a.rules, b.rules)
        c = seqdep.build_session(60, 4, self.work)
        self.assertNotEqual([t.prompt for t in a.turns], [t.prompt for t in c.turns])

    def test_every_usable_distance_at_least_once(self):
        for n, seed in [(100, 0), (100, 1), (100, 2), (60, 0), (20, 0)]:
            s = seqdep.build_session(n, seed, self.work)
            want = set(seqdep.usable_distances(n))
            self.assertEqual(want, {e["distance"] for e in s.edges}, (n, seed))
            self.assertEqual(len(s.turns), n)
            self.assertEqual(s.dropped, [])
        self.assertEqual(set(seqdep.usable_distances(100)), set(seqdep.DISTANCES))

    def test_structure(self):
        s = seqdep.build_session(100, 0, self.work)
        kinds = {t.kind for t in s.turns}
        self.assertTrue({"question", "fact", "rule", "ephemeral", "probe_fact", "probe_derived",
                         "control_derived", "probe_ephemeral"} <= kinds)
        self.assertEqual([t.turn for t in s.turns], list(range(1, 101)))
        for t in s.turns:
            self.assertTrue(t.prompt.startswith(f"Q{t.turn}:") or t.kind == "question", t.turn)
            if t.kind.startswith("probe_"):
                self.assertEqual(t.turn - t.meta["establish_turn"], t.meta["distance"])
                self.assertEqual(s.turns[t.meta["establish_turn"] - 1].meta.get("probe_turn"), t.turn)
            if t.kind == "question":
                self.assertNotIn(t.meta["qid"], cb.EXCLUDED)
        qids = [t.meta["qid"] for t in s.turns if t.kind == "question"]
        self.assertEqual(len(qids), len(set(qids)), "a question must not repeat inside a session")
        self.assertEqual(len(s.rules), 2)
        self.assertEqual({r["rule"] for r in s.rules}, set(seqdep.RULE_KINDS))

    def test_oracle_solves_every_probe_and_obeys_rules(self):
        for n, seed in [(100, 0), (60, 1), (20, 2)]:
            s = seqdep.build_session(n, seed, self.work)
            results, _ = run_session(s)
            for turn, kind, score, rules, details in results:
                self.assertEqual(score, 1.0, (n, seed, turn, kind, details))
                for rid, v in rules.items():
                    self.assertTrue(v["ok"], (n, seed, turn, rid))
            self.assertTrue(any(r[3] for r in results), "rule adherence must be checked on later turns")
            # no planted file survives past the session's last turn
            self.assertFalse(list((self.work / "jobs").glob("*")) and
                             any(p.exists() for p in (self.work / "jobs").glob("*")
                                 if int(p.stem.split("-")[1]) < n))

    def test_amnesiac_fails_every_probe(self):
        s = seqdep.build_session(60, 0, self.work)
        results, _ = run_session(s, amnesiac=True)
        probes = [r for r in results if r[1] in ("probe_fact", "probe_derived", "probe_ephemeral")]
        self.assertTrue(probes)
        for turn, kind, score, _, details in probes:
            self.assertEqual(score, 0.0, (turn, kind, details))

    def test_ephemeral_file_is_gone_before_the_next_turn(self):
        s = seqdep.build_session(40, 0, self.work)
        eph = [t for t in s.turns if t.kind == "ephemeral"]
        self.assertTrue(eph)
        t = eph[0]
        s.before_turn(t.turn)
        self.assertTrue(Path(t.plant["path"]).exists())
        self.assertIn(t.ground_truth, Path(t.plant["path"]).read_text(encoding="utf-8"))
        s.before_turn(t.turn + 1)
        self.assertFalse(Path(t.plant["path"]).exists())

    def test_checkers(self):
        self.assertEqual(seqdep.check_token("TCK-123", "It was TCK-123.\n**ANSWER:** TCK-123")[0], 1.0)
        self.assertEqual(seqdep.check_token("TCK-123", "ANSWER: TCK-124")[0], 0.0)
        self.assertEqual(seqdep.check_value("$1,234.50", "money", "Total is $1,234.50\nANSWER: $1,234.50")[0], 1.0)
        self.assertEqual(seqdep.check_value("$1,234.50", "money", "ANSWER: 1234.5")[0], 1.0)
        self.assertEqual(seqdep.check_value("3", "int", "She owns three cars, so 3.")[0], 1.0)  # last number
        self.assertEqual(seqdep.check_value("3", "int", "ANSWER: 4 (she has 3 pets)")[0], 0.0)
        rule = {"rule": "ref_line", "token": "ORCHID-41"}
        self.assertTrue(seqdep.check_rule(rule, "answer\n\nREF: ORCHID-41"))
        self.assertTrue(seqdep.check_rule(rule, "**REF: ORCHID-41**\nanswer"))
        self.assertFalse(seqdep.check_rule(rule, "answer, ref ORCHID-41"))
        rule = {"rule": "prefix_word", "token": "NOTED-7"}
        self.assertTrue(seqdep.check_rule(rule, "NOTED-7\nanswer"))
        self.assertTrue(seqdep.check_rule(rule, "\n**NOTED-7**\nanswer"))
        self.assertFalse(seqdep.check_rule(rule, "answer\nNOTED-7"))

    def test_parse_spec(self):
        self.assertEqual(seqdep.parse_spec("seqdep:100@2"), (100, 2))
        self.assertEqual(seqdep.parse_spec("seqdep:20"), (20, 0))


if __name__ == "__main__":
    unittest.main()
