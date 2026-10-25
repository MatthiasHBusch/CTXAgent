"""Phase-0 benchmark plumbing (REDESIGN.md section 8): provider pinning, exclusions and
derived questions, web-ban audit, sweep scheduling, analysis statistics. No model calls.

    python -m tests.test_bench_phase0
"""
import argparse
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import analysis, audit, contextbench as cb, proxy, run  # noqa: E402

HAVE_DATA = cb.SRC.exists()


class ProxyPinTests(unittest.TestCase):
    def test_pins_anthropic_and_openai_families(self):
        for model, slug in (("anthropic/claude-opus-5.5", "anthropic"), ("anthropic/claude-opus-5.5[1m]", "anthropic"),
                            ("~openai/gpt-6.1-sol", "openai")):
            body, pin = proxy.pin_provider(json.dumps({"model": model, "stream": True, "messages": []}).encode())
            self.assertEqual(pin, {"order": [slug], "allow_fallbacks": False})
            j = json.loads(body)
            self.assertEqual(j["provider"], pin)
            self.assertEqual(j["model"], model)  # the model id itself is untouched
            self.assertTrue(j["stream"])

    def test_other_bodies_pass_through_byte_for_byte(self):
        raw = json.dumps({"model": "google/gemini-3.1-flash", "messages": []}).encode()
        self.assertEqual(proxy.pin_provider(raw), (raw, None))
        self.assertEqual(proxy.pin_provider(b"not json"), (b"not json", None))
        self.assertEqual(proxy.pin_provider(b""), (b"", None))
        self.assertEqual(proxy.pin_provider(b"[1,2]"), (b"[1,2]", None))

    def test_existing_provider_block_is_merged_not_dropped(self):
        body, pin = proxy.pin_provider(json.dumps({"model": "openai/gpt-6.1-sol",
                                                   "provider": {"data_collection": "deny"}}).encode())
        self.assertEqual(json.loads(body)["provider"], {"data_collection": "deny", "order": ["openai"],
                                                        "allow_fallbacks": False})


@unittest.skipUnless(HAVE_DATA, "Context-Bench files not present under bench/external")
class ContextBenchTests(unittest.TestCase):
    def test_excluded_ids_never_sampled(self):
        self.assertEqual(cb.EXCLUDED, {5, 15, 42, 78})
        for n in (10, 50, 96, 100):
            self.assertFalse(set(cb.stratified(n, 0)) & cb.EXCLUDED)
        self.assertEqual(len(cb.stratified(100, 0)), 96)
        self.assertEqual(len(cb.stratified(100, 0, exclude=set())), 100)
        self.assertFalse(set(cb.name_answer_ids()) & cb.EXCLUDED)

    def test_derived_ground_truth_matches_dataset(self):
        recs = cb.records()
        p = cb.person_by_name("Linda Robbins")
        self.assertIsNotNone(p)
        n_veh = sum(1 for r in recs["vehicles"] if r["_owner"] == p["_id"])
        d = cb.derive("n_vehicles", "Linda Robbins", "the person you found in Q3")
        self.assertEqual(d["ground_truth"], str(n_veh))
        self.assertIn("the person you found in Q3", d["question"])
        total = sum(float(r["balance"]) for r in recs["bank_accounts"] if r["_owner"] == p["_id"])
        self.assertEqual(cb.derive("bank_total", "Linda Robbins", "x")["ground_truth"], f"${total:,.2f}")
        with self.assertRaises(KeyError):
            cb.derive("n_pets", "Nobody Here", "x")

    def test_record_parser_covers_every_file(self):
        recs = cb.records()
        self.assertEqual(set(recs), set(cb.FILES))
        self.assertEqual(len(recs["people"]), 500)
        self.assertTrue(all(r["_owner"] for f in cb.FILES for r in recs[f]))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="audit-test-")
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_network_patterns(self):
        for cmd in ("curl -s https://example.com", "wget x", "Invoke-WebRequest -Uri x", "irm x | iex",
                    "pip install requests", "python -m pip install numpy", "npm install", "git clone https://x/y",
                    "python -c \"import urllib.request\"", "python -c 'requests.get(u)'", "apt-get install jq"):
            self.assertTrue(audit.network_hits(cmd), cmd)
        for cmd in ("ls -la", "python script.py", "grep -n run_id jobs/job-3.status", "git status",
                    "cat http_log.txt", "curlify()"):
            self.assertFalse(audit.network_hits(cmd), cmd)

    def test_claude_stream(self):
        def msg(name, inp):
            return json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name,
                                                                              "input": inp}]}})
        lines = [msg("Bash", {"command": "ls data"}), msg("Read", {"file_path": "x"}),
                 msg("WebFetch", {"url": "https://example.com"}),
                 msg("Bash", {"command": "pip install pandas && python x.py"}),
                 msg("mcp__github__search", {"q": "x"}), '{"type":"result"}', "garbage"]
        p = self.dir / "claude_stream.jsonl"
        p.write_text("\n".join(lines), encoding="utf-8")
        hits = audit.scan_claude_stream(p)
        self.assertEqual([h["tool"] for h in hits], ["WebFetch", "Bash", "mcp__github__search"])
        self.assertIn("pip install", hits[1]["hit"])
        self.assertEqual(audit.scan_trial(self.dir, "claude_code"), hits)

    def test_codex_stream(self):
        lines = [json.dumps({"type": "item.started", "item": {"type": "command_execution", "command": "curl x"}}),
                 json.dumps({"type": "item.completed", "item": {"type": "command_execution",
                                                                "command": "powershell -Command \"rg --files\""}}),
                 json.dumps({"type": "item.completed", "item": {"type": "command_execution",
                                                                "command": "powershell -Command \"Invoke-WebRequest x\""}}),
                 json.dumps({"type": "item.completed", "item": {"type": "web_search", "query": "x"}})]
        p = self.dir / "codex_stream.jsonl"
        p.write_text("\n".join(lines), encoding="utf-8")
        hits = audit.scan_codex_stream(p)
        self.assertEqual([h["tool"] for h in hits], ["command_execution", "web_search"])  # started items not double-counted
        self.assertIn("Invoke-WebRequest", hits[0]["hit"])

    def test_ctxagent_db_including_worker_chats(self):
        db = self.dir / "ctxdata" / "ctxagent.db"
        db.parent.mkdir()
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE chats (id TEXT PRIMARY KEY, title TEXT, conversation TEXT)")

        def seg(name, args):
            return {"id": "s1", "kind": "tool_call", "tool_calls": [{"function": {"name": name, "arguments": json.dumps(args)}}]}
        main = {"segments": [seg("read_file", {"path": "x"}), seg("run_command", {"command": "python join.py"})],
                "archive": [seg("run_command", {"command": "pip install rich"})]}
        worker = {"segments": [seg("web_search", {"query": "x"}), seg("fetch_url", {"url": "u"}),
                               seg("document_open", {"path": "p"})]}  # document tools are allowed
        con.execute("INSERT INTO chats VALUES (?,?,?)", ("c1", "main", json.dumps(main)))
        con.execute("INSERT INTO chats VALUES (?,?,?)", ("c2", "worker", json.dumps(worker)))
        con.commit()
        con.close()
        hits = audit.scan_trial(self.dir, "ctxagent_live")
        self.assertEqual(sorted(h["tool"] for h in hits), ["fetch_url", "run_command", "web_search"])
        self.assertEqual(audit.scan_ctxagent_db(self.dir / "missing.db"), [])


class SweepTests(unittest.TestCase):
    def args(self, **kw):
        base = dict(models="anthropic/claude-opus-5.5,openai/gpt-6.1-sol", windows="native,1000000",
                    tasks="seqdep:20@0", harnesses="claude_code,codex,ctxagent_live", reps=3, prefix="",
                    seed=7, no_shell=False)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_ctxagent_pinned_to_1m_and_collapsed_across_windows(self):
        jobs = run.sweep_jobs(self.args())
        ctx = [j for j in jobs if j["harness"] == "ctxagent_live"]
        self.assertEqual({j["window"] for j in ctx}, {run.CTX_WINDOW})
        self.assertEqual(len(ctx), 2 * 3)  # models x reps, not x windows
        cc = [j for j in jobs if j["harness"] == "claude_code"]
        self.assertEqual({j["window"] for j in cc}, {run.NATIVE, 1_000_000})
        self.assertEqual(len(jobs), len({j["tag"] for j in jobs}))
        self.assertEqual(len(jobs), 6 + 12 + 12)

    def test_order_is_shuffled_deterministically_by_seed(self):
        a = [j["tag"] for j in run.sweep_jobs(self.args())]
        b = [j["tag"] for j in run.sweep_jobs(self.args())]
        c = [j["tag"] for j in run.sweep_jobs(self.args(seed=8))]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, sorted(a))

    def test_codex_skipped_without_shell(self):
        jobs = run.sweep_jobs(self.args(no_shell=True))
        self.assertTrue(all(j["skip"] for j in jobs if j["harness"] == "codex"))
        self.assertTrue(all(not j["skip"] for j in jobs if j["harness"] != "codex"))

    def test_never_two_trials_of_the_same_model_and_harness(self):
        pending, running, done = run.sweep_jobs(self.args()), [], []
        step = 0
        while pending or running:
            while pending and len(running) < 3:
                j = run.pick_next(pending, running)
                if j is None:
                    break
                pending.remove(j)
                running.append(j)
                keys = [(x["model"], x["harness"]) for x in running]
                self.assertEqual(len(keys), len(set(keys)))
            done.append(running.pop(step % len(running)))  # finish one trial
            step += 1
        self.assertEqual(len(done), 30)

    def test_window_parsing_and_harness_flags(self):
        self.assertEqual(run.parse_window("native"), run.NATIVE)
        self.assertEqual(run.parse_window("1000000"), 1_000_000)
        self.assertEqual(run.window_slug(run.NATIVE), "wnative")
        self.assertEqual(run.window_slug(200_000), "w200k")
        a = argparse.Namespace(model="anthropic/claude-opus-5.5", window=run.NATIVE)
        self.assertEqual(run._cc_model(a), "anthropic/claude-opus-5.5")
        a.window = 1_000_000
        self.assertEqual(run._cc_model(a), "anthropic/claude-opus-5.5[1m]")
        self.assertTrue(run.ctx_tool_banned("web_search") and run.ctx_tool_banned("browser_open")
                        and run.ctx_tool_banned("fetch_url") and run.ctx_tool_banned("scholarly_search"))
        for keep in ("recall", "focus_file", "summarize_file", "curate_context", "compact_context", "pin",
                     "context_report", "agent_start", "run_command", "read_file",
                     "read_document", "document_open", "render_pdf", "view_image", "latex_build"):
            self.assertFalse(run.ctx_tool_banned(keep), keep)
        self.assertIn("seqdep:", run.TASK_BUILDERS)
        with self.assertRaises(SystemExit):
            run.setup_task(argparse.Namespace(task="nope:1"), Path("."))
        self.assertEqual(run.tool_matrix(argparse.Namespace(harness="codex", no_shell=False))["web_search"], "disabled")
        self.assertIn("Bash", run.tool_matrix(argparse.Namespace(harness="claude_code", no_shell=True))["disallowed"])
        self.assertNotIn("ctxagent_deeponly", run.ARMS)
        self.assertNotIn("ctxagent_truncate", run.ARMS)

    def test_live_arm_config_strips_web_tools_and_shell(self):
        a = argparse.Namespace(harness="ctxagent_live", model="anthropic/claude-opus-5.5", window=run.NATIVE,
                               no_shell=True, comp=[], summary_model="", max_out=20000, max_iters=100,
                               temperature=1.0, reasoning="high")
        cfg = run.build_ctx_config(a, Path("."))
        self.assertEqual(cfg.context_window, run.CTX_WINDOW)
        tools = set(cfg.enabled_tools)
        self.assertFalse({t for t in tools if run.ctx_tool_banned(t)})
        self.assertFalse(tools & set(run.SHELL_TOOLS))
        self.assertTrue({"recall", "curate_context", "compact_context", "pin", "context_report", "agent_start",
                         "read_document", "document_open", "render_pdf", "latex_build"} <= tools)
        a.no_shell = False
        self.assertIn("run_command", run.build_ctx_config(a, Path(".")).enabled_tools)


def _row(tag, harness, window, turns, wire, model="anthropic/claude-opus-5.5", **kw):
    r = {"tag": tag, "harness": harness, "model": model, "window": window, "task": "seqdep:20@0", "rep": 0,
         "turns": turns, "wire": wire, "billed_cost": sum(w["billed"] for w in wire),
         "billed_excl_turn1": sum(w["billed"] for w in wire if w["turn"] != 1),
         "wire_tokens": {"uncached": 10, "cache_write": 5, "cache_read": 85, "output": 3},
         "wire_by_model": {model: sum(w["billed"] for w in wire)}}
    r.update(kw)
    return r


def _turn(k, kind, score, **meta):
    return {"turn": k, "kind": kind, "score": score, "meta": meta, "harness_error": False}


class AnalysisTests(unittest.TestCase):
    def test_binomial_and_mcnemar(self):
        self.assertEqual(analysis.binom_two_sided(0, 0), 1.0)
        self.assertAlmostEqual(analysis.binom_two_sided(0, 5), 2 * 0.5 ** 5)
        self.assertAlmostEqual(analysis.binom_two_sided(2, 4), 1.0)
        m = analysis.mcnemar([(1, 0)] * 5 + [(1, 1)] * 10)
        self.assertEqual((m["a_only"], m["b_only"]), (5, 0))
        self.assertAlmostEqual(m["p"], 0.0625)
        s = analysis.sign_test([(0.5, 1.0)] * 6 + [(1.0, 1.0)])
        self.assertEqual((s["b_better"], s["a_better"]), (6, 0))

    def test_cluster_bootstrap(self):
        p, lo, hi = analysis.cluster_bootstrap([[1, 1, 1], [0, 0], [1, 0]], n=2000, seed=1)
        self.assertAlmostEqual(p, 4 / 7)
        self.assertTrue(lo <= p <= hi)
        self.assertEqual(analysis.cluster_bootstrap([[1, 1]], n=10)[1:], (None, None))
        self.assertEqual(analysis.cluster_bootstrap([], n=10), (None, None, None))

    def test_retention_split_by_wire_compaction(self):
        # context: 100k, 120k, then a drop to 40k at turn 8 (a compaction), then growth
        wire = [{"turn": t, "model": "anthropic/claude-opus-5.5", "uncached": c, "cache_write": 0, "cache_read": 0,
                 "output": 1, "billed": 0.01} for t, c in [(1, 100_000), (5, 120_000), (8, 40_000), (12, 60_000)]]
        turns = [_turn(2, "fact", 1.0), _turn(5, "probe_fact", 1.0, establish_turn=2, distance=3),
                 _turn(3, "ephemeral", 1.0), _turn(13, "probe_ephemeral", 0.0, establish_turn=3, distance=10),
                 _turn(9, "fact", 1.0), _turn(12, "probe_fact", 1.0, establish_turn=9, distance=3)]
        r = _row("t1", "claude_code", "native", turns, wire)
        recs = analysis.probe_records([r])
        by = {(x["turn"]): x["between"] for x in recs}
        self.assertEqual(by, {5: False, 13: True, 12: False})
        rc = analysis.retention_curve([r], boot=100, seed=0)
        a = analysis.arm(r)
        self.assertEqual(rc[(a, 3, False)][0], 2)
        self.assertEqual(rc[(a, 10, True)][:2], (1, 0.0))

    def test_paired_and_cost_anatomy_and_load(self):
        wire = [{"turn": 1, "model": "anthropic/claude-opus-5.5", "uncached": 10, "cache_write": 0,
                 "cache_read": 0, "output": 1, "billed": 0.5},
                {"turn": 2, "model": "anthropic/claude-opus-5.5", "uncached": 10, "cache_write": 0,
                 "cache_read": 0, "output": 1, "billed": 0.1}]
        ta = [_turn(1, "question", 1.0), _turn(2, "question", 0.0), _turn(3, "probe_fact", 1.0, establish_turn=1, distance=2)]
        tb = [_turn(1, "question", 1.0), _turn(2, "question", 1.0), _turn(3, "probe_fact", 0.0, establish_turn=1, distance=2)]
        rows = [_row("a", "claude_code", "native", ta, wire), _row("b", "ctxagent_live", "1000000", tb, wire),
                _row("c", "codex", "native", tb, wire, web_use=[{"hit": "curl"}]), {"tag": "d", "skipped": "x"}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            use, aside = analysis.load([p])
        self.assertEqual({r["tag"] for r in use}, {"a", "b"})
        self.assertEqual({r["_why"].split(" ")[0] for r in aside}, {"web_use", "skipped:"})
        pt = analysis.paired(use)
        key = next(k for k in pt if k[2] == "question")
        self.assertEqual(pt[key]["test"], "mcnemar")
        self.assertEqual(pt[key]["n"], 2)
        ca = analysis.cost_anatomy(use, boot=50, seed=0)
        c = ca[analysis.arm(use[0])]
        self.assertAlmostEqual(c["billed"][0], 0.6)
        self.assertAlmostEqual(c["billed_excl_turn1"][0], 0.1)
        self.assertAlmostEqual(c["billed_turn1"], 0.5)
        self.assertAlmostEqual(c["amortised_per_task"], 0.5 / analysis.AMORTISE_N + 0.1 / 2)
        self.assertAlmostEqual(c["cache_hit"], 0.85)
        sk = analysis.success_by_kind(use, boot=50, seed=0)
        self.assertEqual(sk[analysis.arm(use[0])]["question"][:2], (2, 0.5))



@unittest.skipUnless(HAVE_DATA, "Context-Bench files not present under bench/external")
class FinishTrialTests(unittest.TestCase):
    """End-to-end scoring of a seqdep session without any model: setup_task builds the
    turns, an oracle supplies the finals, finish_trial writes the row."""

    def test_seqdep_row(self):
        from tests.test_seqdep import oracle_reply, fake_judge
        with tempfile.TemporaryDirectory(prefix="ft-") as d:
            tdir = Path(d) / "trial"
            tdir.mkdir()
            out = Path(d) / "out.jsonl"
            a = argparse.Namespace(task="seqdep:12@0", seed=0, recap=False, harness="ctxagent_live", tag="t",
                                   model="anthropic/claude-opus-5.5", rep=0, window=run.CTX_WINDOW, reasoning="high",
                                   max_iters=100, out=str(out), no_shell=False, no_pin=False, comp=[])
            workdir, spec = run.setup_task(a, tdir)
            from bench import seqdep
            sess = seqdep.build_session(12, 0, workdir)
            sess.judge = fake_judge
            for t in spec:  # route the judge through the fake for question turns
                k = int(t["prompt"].split(":")[0][1:]) if t["prompt"].startswith("Q") else None
                t["check"] = (lambda k: lambda f, h=None: sess.check(k, f, h))(k)
                t["rules"] = (lambda k: lambda f: sess.rule_adherence(k, f))(k)
            history, turns = [], []
            for k, t in enumerate(spec, 1):
                t["pre"]()
                final = oracle_reply(sess, k, history)
                history.append({"turn": k, "kind": t["kind"], "prompt": t["prompt"], "final": final, "meta": t["meta"]})
                turns.append({**run._new_turn(), "final": final, "finish": "stop",
                              "trace": [{"source": "agent", "model": a.model, "provider": None, "uncached": 100,
                                         "cache_write": 0, "cache_read": 0, "output": 20, "reasoning": None,
                                         "reported_cost": None}]})
            # proxy log: turn 1 cold (no cache read), later turns warm
            recs = [{"ts": 0, "turn": k, "method": "POST", "path": "/api/v1/chat/completions", "status": 200,
                     "model": a.model, "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "cost": 0.01,
                                                 "prompt_tokens_details": {"cached_tokens": 0 if k == 1 else 900}}}
                    for k in range(1, 13)]
            (tdir / "proxy.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
            run.finish_trial(a, tdir, spec, turns, 1.0)
            row = json.loads(out.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(row["score"], 1.0)
        self.assertEqual(row["rule_adherence"], 1.0)
        self.assertGreater(row["rule_checks"], 0)
        self.assertFalse(row["warm_start"])
        self.assertAlmostEqual(row["billed_cost"], 0.12)
        self.assertAlmostEqual(row["billed_excl_turn1"], 0.11)
        self.assertIsNone(row["web_use"])
        self.assertEqual(row["tool_matrix"]["web"], "off")
        self.assertIn("probe_ephemeral", row["score_by_kind"])
        kinds = {t["kind"] for t in row["turns"]}
        self.assertIn("rule", kinds)
        probe = next(t for t in row["turns"] if t["kind"].startswith("probe_"))
        self.assertIn("establish_turn", probe["meta"])
        self.assertIsNotNone(probe["ground_truth"])
        self.assertTrue(all(t["rule_adherence"] is not None for t in row["turns"] if t["turn"] > 2))


if __name__ == "__main__":
    unittest.main()
