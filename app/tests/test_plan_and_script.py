"""plan and run_script.

* plan(plan) puts a plan into the chat (the UI draws the call's text as a message) and
  returns an empty result.
* run_script(script, language, args, cwd, timeout, interpreter) writes the source to a file
  and runs the file. The text never passes through a shell, so quotes, backslashes, `$`,
  backticks and newlines arrive unchanged (inline scripts in run_command lose them).
* Consecutive run_script calls of one response overlap (up to 4); a read next to a script
  does not join its group.
* smart_search prints file lines verbatim, so they work as edit_file's old_string.

Run:  python -m tests.test_plan_and_script
"""
import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctxagent import smart_search, tools
from ctxagent.tool_batch import execute_batch, parallel_kind

NASTY = r'''import sys
text = r"""He said "it's $HOME" and `backticks`; C:\temp\new, 100% {braces} <tags> & | ^ äöü €"""
print(text)
print(len(sys.argv) - 1, sys.argv[1:])
'''
NASTY_OUT = r'''He said "it's $HOME" and `backticks`; C:\temp\new, 100% {braces} <tags> & | ^ äöü €'''


def call(name, ident, **args):
    return {'id': ident, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


class Plan(unittest.TestCase):
    def test_returns_nothing_and_is_a_default_tool(self):
        sb = tools.Sandbox([tempfile.gettempdir()])
        result = asyncio.run(tools.t_plan(sb, plan="1. Find the code\n2. Change it"))
        self.assertEqual(result.text, "")
        self.assertFalse(result.error)
        self.assertFalse(result.mutates)
        with self.assertRaises(tools.ToolError):
            asyncio.run(tools.t_plan(sb, plan="  "))
        spec = tools.REGISTRY["plan"]
        self.assertEqual(spec.parameters["required"], ["plan"])
        self.assertIn("returns nothing", spec.description)
        self.assertIn("plan", tools.DEFAULT_TOOLS)

    def test_the_chat_draws_the_plan_as_a_message(self):
        js = (Path(__file__).resolve().parent.parent / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn("function addPlan(text)", js)
        self.assertIn("if (ev.name === 'plan' && ev.args && ev.args.plan) { addPlan(ev.args.plan); break; }", js)
        self.assertIn("if (tc.function.name === 'plan' && args.plan) { addPlan(args.plan); planCalls.add(tc.id); continue; }", js)


class Script(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sb = tools.Sandbox([str(self.root)])

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, **kw):
        return asyncio.run(tools.t_run_script(self.sb, cwd=str(self.root), **kw))

    def test_source_arrives_unchanged_and_arguments_are_passed(self):
        result = self.run_script(script=NASTY, args=["a b", "c\"d", "$x"])
        self.assertFalse(result.error, result.text)
        self.assertIn(NASTY_OUT, result.text)
        self.assertIn("3 ['a b', 'c\"d', '$x']", result.text)
        self.assertIn("exit=0", result.text)
        path = next(tools.script_dir().glob("script_*.py"))
        self.assertTrue(path.exists())
        self.assertIn(str(tools.script_dir()), result.text)        # the file's path is in the result

    def test_runs_in_cwd_and_reports_failure(self):
        ok = self.run_script(script="import os\nprint(os.getcwd())\nopen('made.txt', 'w').write('x')\n")
        self.assertIn(self.root.name, ok.text)
        self.assertTrue((self.root / "made.txt").exists())
        bad = self.run_script(script="import sys\nprint('to stderr', file=sys.stderr)\nsys.exit(3)\n")
        self.assertTrue(bad.error)
        self.assertIn("exit=3", bad.text)
        self.assertIn("to stderr", bad.text)
        slow = self.run_script(script="import time\ntime.sleep(30)\n", timeout=1)
        self.assertTrue(slow.error)
        self.assertIn("timed out after 1s", slow.text)

    @unittest.skipUnless(sys.platform == "win32", "PowerShell script")
    def test_powershell_script_keeps_dollar_quotes_and_umlauts(self):
        script = "$name = \"Wörld\"\n$items = @('a b', \"c`\"d\")\nWrite-Output \"Hello ${name}: $($items.Count) items, 'single' `$literal\"\nWrite-Output $args[0]\n"
        result = self.run_script(script=script, language="powershell", args=["x y"])
        self.assertFalse(result.error, result.text)
        self.assertIn("Hello Wörld: 2 items, 'single' $literal", result.text)
        self.assertIn("x y", result.text)

    def test_bad_arguments(self):
        for kw in ({"script": " "}, {"script": "x", "language": "ruby"}, {"script": "x", "args": "a b"},
                   {"script": "x", "interpreter": "no-such-interpreter-xyz"}):
            with self.assertRaises(tools.ToolError):
                self.run_script(**kw)
        spec = tools.REGISTRY["run_script"]
        self.assertTrue(spec.dangerous)
        self.assertIn("run at the same time", spec.description)
        self.assertIn("run_script", tools.DEFAULT_TOOLS)


class Parallel(unittest.IsolatedAsyncioTestCase):
    async def test_consecutive_scripts_overlap_and_results_keep_the_call_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = tools.Sandbox([tmp])

            class Session:
                cancel = asyncio.Event()

                async def _exec_tool(self, name, args):
                    return await tools.REGISTRY[name].fn(sb, **args)

            script = "import sys, time\ntime.sleep(float(sys.argv[1]))\nprint('done', sys.argv[2])\n"
            calls = [call('run_script', f's{i}', script=script, args=[str(delay), str(i)], cwd=tmp)
                     for i, delay in enumerate((2.0, 1.0, 1.5))]
            started = time.monotonic()
            results = [e async for e in execute_batch(Session(), calls) if e[0] == 'result']
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 4.0)                              # serial would be 4.5 s plus start-up
            self.assertEqual([e[1]['id'] for e in results], ['s0', 's1', 's2'])
            for i, event in enumerate(results):
                self.assertIn(f"done {i}", event[4].text)

    async def test_groups_are_reads_or_scripts_never_both(self):
        self.assertEqual(parallel_kind('run_script', {}), 'script')
        self.assertEqual(parallel_kind('read_file', {}), 'read')
        self.assertIsNone(parallel_kind('run_command', {}))
        self.assertIsNone(parallel_kind('edit_file', {}))
        active, peaks = 0, {}

        class Session:
            cancel = asyncio.Event()

            async def _exec_tool(self, name, args):
                nonlocal active
                active += 1
                peaks[args['path']] = active
                await asyncio.sleep(0.01)
                active -= 1
                return tools.ToolResult(args['path'])

        calls = [call('read_file', 'r1', path='r1'), call('run_script', 's1', path='s1'),
                 call('run_script', 's2', path='s2'), call('read_file', 'r2', path='r2')]
        [e async for e in execute_batch(Session(), calls)]
        self.assertEqual(peaks, {'r1': 1, 's1': 1, 's2': 2, 'r2': 1})


class SearchToEdit(unittest.TestCase):
    def test_lines_from_smart_search_work_as_old_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = "def area(r):\n\tif r < 0:\n\t\traise ValueError('negative')\n\treturn 3.14 * r * r\n"
            (root / "geo.py").write_text(source, encoding="utf-8")

            async def model(**kw):
                user = kw["messages"][1]["content"]
                if user.count("Folder: "):
                    return json.dumps({"files": [{"name": "geo.py", "relevance": "high"}], "folders": []})
                return json.dumps({"ranges": [{"start": 1, "end": 4, "kind": "core", "note": "area"}]})

            report = asyncio.run(smart_search.search(root, "the area function", "", model, "m"))
            text = smart_search.render(report, root, "the area function", "m")
            # What the agent sees: `<6-wide number><TAB><the line>`. The part after the first tab is the line.
            shown = [line.split("\t", 1)[1] for line in text.splitlines() if line[:6].strip().isdigit()]
            old = "\n".join(shown[1:3])
            self.assertEqual(old, "\tif r < 0:\n\t\traise ValueError('negative')")
            sb = tools.Sandbox([tmp])
            result = asyncio.run(tools.t_edit_file(sb, str(root / "geo.py"), old,
                                                   "\tif r < 0:\n\t\treturn 0.0"))
            self.assertFalse(result.error)
            self.assertIn("return 0.0", (root / "geo.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
