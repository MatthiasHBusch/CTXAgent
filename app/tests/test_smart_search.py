"""smart_search: a small model walks a folder tree and returns the relevant line ranges.

* The walk starts at the given folder and opens only the subfolders the model names.
* Named files are read; the model's line ranges are printed with file and line numbers.
* A file larger than one request is cut into portions at line boundaries.
* Limits (folders, files, output tokens) leave a note about what was left out.
* The model is Claude Haiku 5.5 by default, and the tool description says so.

The model is a scripted stand-in here; tests/smart_search_live.py runs the real one.

Run:  python -m tests.test_smart_search
"""
import asyncio
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctxagent import smart_search, tokens, tools
from ctxagent.tool_batch import is_parallel_read


def tree(root: Path) -> None:
    (root / "app").mkdir()
    (root / "app" / "core").mkdir()
    (root / "docs").mkdir()
    (root / "vendor").mkdir()
    (root / "app" / "server.py").write_text(
        "import os\n\n\ndef start(port):\n    return port\n\n\ndef stop():\n    pass\n", encoding="utf-8")
    (root / "app" / "core" / "cache.py").write_text(
        "\n".join(f"line {i}" for i in range(1, 41)) + "\n", encoding="utf-8")
    (root / "app" / "core" / "util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "docs" / "guide.md").write_text("# Guide\n\nText.\n", encoding="utf-8")
    (root / "vendor" / "lib.py").write_text("x = 1\n", encoding="utf-8")
    (root / "README.md").write_text("# Project\n", encoding="utf-8")


class Model:
    """Scripted answers: folder name -> (files, folders); file name -> ranges."""

    def __init__(self, folders: dict, files: dict):
        self.folders, self.files = folders, files
        self.calls: list[dict] = []
        self.broken: list[str] = []      # answers handed out before the real one

    async def __call__(self, **kw):
        user = kw["messages"][1]["content"]
        self.calls.append(kw)
        if self.broken:
            return self.broken.pop(0)
        folder = re.search(r"^Folder: (\S+)", user, re.M)
        if folder:
            files, dirs = self.folders.get(folder[1], ([], []))
            return json.dumps({"files": [{"name": n, "relevance": r} for n, r in files], "folders": dirs})
        name = re.search(r"^File: (\S+)", user, re.M)[1]
        answer = self.files.get(name, [])
        if callable(answer):
            answer = answer(user)
        return json.dumps({"ranges": [{"start": a, "end": b, "kind": r, "note": n}
                                      for a, b, r, n in answer]})

    def surveyed(self) -> list[str]:
        return [m[1] for m in (re.search(r"^Folder: (\S+)", c["messages"][1]["content"], re.M)
                               for c in self.calls) if m]


def run(root: Path, model: Model, **kw) -> smart_search.Report:
    return asyncio.run(smart_search.search(root, "where the server starts", "", model, "test-model", **kw))


class Walk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        tree(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_named_folders_are_opened_and_ranges_are_printed(self):
        model = Model({".": ([], ["app"]),
                       "app": ([("server.py", "high")], ["core"]),
                       "app/core": ([("cache.py", "medium")], [])},
                      {"app/server.py": [(4, 5, "core", "start function")],
                       "app/core/cache.py": [(10, 12, "support", "cache lines")]})
        report = run(self.root, model)
        self.assertEqual(sorted(model.surveyed()), [".", "app", "app/core"])       # not docs, not vendor
        self.assertEqual([h.path.name for h in report.files], ["server.py", "cache.py"])
        text = smart_search.render(report, self.root, "where the server starts", "test-model")
        self.assertIn("=== app/server.py  (9 lines)", text)
        self.assertIn("--- L4-5  [core] start function", text)
        self.assertIn("     4\tdef start(port):\n     5\t    return port", text)
        self.assertIn("    10\tline 10", text)
        self.assertNotIn("line 13", text)
        self.assertIn("3 folders surveyed, 2 files read, 2 ranges in 2 files", text)

    def test_folder_step_shows_the_level_2_overview_or_level_1_of_a_large_folder(self):
        model = Model({}, {})
        run(self.root, model)
        user = model.calls[0]["messages"][1]["content"]
        self.assertIn("Search goal:\nwhere the server starts", user)
        self.assertIn("── README.md", user)            # files with their outline
        self.assertIn("app/   [1 dirs, 1 files", user)
        self.assertIn("server.py  (", user)             # subfolders with the names of their entries
        self.assertNotIn("start(", user)                # but not their outlines
        with patch.object(smart_search, "FOLDER_VIEW_TOKENS", 20):
            text, dirs, files = smart_search.folder_view(self.root)
        self.assertIn("── README.md", text)
        self.assertNotIn("server.py", text)             # level 1: name and size only
        self.assertEqual(sorted(dirs), ["app", "docs", "vendor"])
        self.assertEqual(model.calls[0]["response_format"]["json_schema"]["schema"], smart_search.FOLDER_SCHEMA)
        self.assertEqual(model.calls[0]["reasoning_effort"], smart_search.EFFORT)

    def test_names_with_path_or_other_case_resolve_and_unknown_names_are_dropped(self):
        model = Model({".": ([("readme.md", "high"), ("missing.py", "high")], ["./app/", "nowhere"]),
                       "app": ([("app/server.py", "high")], [])},
                      {"README.md": [(1, 1, "core", "title")]})
        report = run(self.root, model)
        self.assertEqual(sorted(model.surveyed()), [".", "app"])
        self.assertEqual(sorted(h.path.name for h in report.files), ["README.md", "server.py"])

    def test_a_path_below_the_folder_is_taken_directly(self):
        model = Model({".": ([("app/core/cache.py", "high")], ["app/core"]),
                       "app/core": ([], [])},
                      {"app/core/cache.py": [(1, 2, "core", "start")]})
        report = run(self.root, model)
        self.assertEqual(sorted(model.surveyed()), [".", "app/core"])        # app itself is skipped
        self.assertEqual([h.path.name for h in report.files], ["cache.py"])
        outside = Model({".": ([("../x/secret.txt", "high")], [".."])}, {})
        self.assertEqual(run(self.root, outside).files, [])

    def test_limits_leave_a_note(self):
        model = Model({".": ([("README.md", "medium")], ["app", "docs", "vendor"]),
                       "app": ([("server.py", "high")], ["core"]),
                       "docs": ([("guide.md", "medium")], [])},
                      {"app/server.py": [(4, 5, "core", "start")]})
        report = run(self.root, model, max_folders=3, max_files=1)
        self.assertEqual(len(report.folders_surveyed), 3)
        self.assertEqual({p.name for p in report.folders_left}, {"vendor", "core"})
        self.assertEqual([h.path.name for h in report.files], ["server.py"])       # high before medium
        self.assertEqual({p.name for p in report.files_left}, {"README.md", "guide.md"})
        text = smart_search.render(report, self.root, "g", "m")
        self.assertIn("Named but not read (limit of 1 files): ", text)
        self.assertIn("Named but not looked into (limit of 3 folders): ", text)

    def test_nothing_found(self):
        report = run(self.root, Model({".": ([("README.md", "medium")], [])}, {}))
        text = smart_search.render(report, self.root, "g", "m")
        self.assertIn("No relevant lines found.", text)
        self.assertIn("Read, nothing relevant: README.md", text)

    def test_unparsable_answer_is_asked_again_and_a_failed_step_is_reported(self):
        model = Model({".": ([("README.md", "high")], [])}, {"README.md": [(1, 1, "core", "title")]})
        model.broken = ["not json"]
        report = run(self.root, model)
        self.assertEqual(len(report.files), 1)
        self.assertEqual(report.calls, 3)
        model = Model({".": ([("README.md", "high")], [])}, {})
        model.broken = ['{"files": [{"name": "README.md", "relevance": "high"}], "folders": []}', "x", "y"]
        report = run(self.root, model)
        self.assertEqual(len(report.errors), 1)
        self.assertIn("Failed steps:", smart_search.render(report, self.root, "g", "m"))


class Files(unittest.TestCase):
    def test_ranges_are_clipped_swapped_and_merged(self):
        raw = [{"start": 8, "end": 5, "kind": "support", "note": "a"},
               {"start": 9, "end": 12, "kind": "core", "note": "b"},
               {"start": 30, "end": 99, "kind": "support", "note": "c"},
               {"start": 200, "end": 210, "kind": "core", "note": "outside"},
               {"start": "x", "end": 3, "kind": "core", "note": "bad"}]
        merged = smart_search.merge_ranges(smart_search.clean_ranges(raw, 1, 40))
        self.assertEqual([(r.start, r.end, r.kind, r.note) for r in merged],
                         [(5, 12, "core", "a; b"), (30, 40, "support", "c")])

    def test_a_large_file_is_cut_into_portions_below_the_request_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lines = [f"value_{i} = compute_something({i}, 'text number {i}')" for i in range(1, 2001)]
            (root / "big.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
            seen = []

            def answer(user):
                first, last = map(int, re.search(r"lines (\d+)-(\d+)\. Answer", user).groups())
                seen.append((first, last, tokens.count(user)))
                return [(first, first, "core", "first line of the part"), (last + 5, last + 9, "core", "outside")]

            model = Model({".": ([("big.py", "high")], [])}, {"big.py": answer})
            with patch.object(smart_search, "REQUEST_TOKENS", 12_000):
                report = run(root, model)
            self.assertGreater(len(seen), 2)
            seen.sort()
            self.assertEqual(seen[0][0], 1)
            self.assertEqual(seen[-1][1], 2000)
            for (_, last, _), (first, _, _) in zip(seen, seen[1:]):
                self.assertEqual(first, last + 1)                      # no gap, no overlap
            self.assertTrue(all(size <= 12_000 for _, _, size in seen))
            self.assertEqual([r.start for r in report.files[0].ranges], [first for first, _, _ in seen])
            self.assertIn("Outline of the whole file:", model.calls[-1]["messages"][1]["content"])

    def test_output_limit_prints_core_ranges_first_and_lists_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("\n".join(f"alpha line {i}" for i in range(1, 801)) + "\n", encoding="utf-8")
            model = Model({".": ([("a.py", "high")], [])},
                          {"a.py": [(1, 300, "support", "background"), (400, 420, "core", "the core")]})
            report = run(root, model)
            text = smart_search.render(report, root, "g", "m", max_tokens=1_000)
            self.assertIn("--- L400-420  [core] the core", text)
            self.assertNotIn("alpha line 100\n", text)
            self.assertIn("Found but not printed (output limit of 1,000 tokens):\n  a.py:L1-300  [support] background", text)
            self.assertLess(tokens.count(text), 1_200)


class Tool(unittest.TestCase):
    def test_registered_with_haiku_named_in_the_description(self):
        spec = tools.REGISTRY["smart_search"]
        self.assertIn("Claude Haiku 5.5 by default", spec.description)
        self.assertTrue(spec.needs_ctx)
        self.assertIn("smart_search", tools.DEFAULT_TOOLS)
        self.assertEqual(spec.parameters["required"], ["goal"])
        self.assertFalse(is_parallel_read("smart_search", {"goal": "x"}))
        self.assertEqual(smart_search.DEFAULT_MODEL, "claude-haiku-5-5")
        self.assertLess(smart_search.REQUEST_TOKENS, smart_search.PRICE_STEP_TOKENS)

    def test_default_model_follows_the_route_with_a_key(self):
        with patch("ctxagent.config.key_problem", return_value=None):
            self.assertEqual(smart_search.default_model("claude-opus-5-5"), "claude-haiku-5-5")
            self.assertEqual(smart_search.default_model("anthropic/claude-opus-5.5"), "anthropic/claude-haiku-5.5")
            self.assertEqual(smart_search.default_model("openai/gpt-x"), "anthropic/claude-haiku-5.5")
        with patch("ctxagent.config.key_problem", side_effect=lambda m: None if m.startswith("claude-") else "no key"):
            self.assertEqual(smart_search.default_model("anthropic/claude-opus-5.5"), "claude-haiku-5-5")

    def test_tool_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tree(root)
            sb = tools.Sandbox([str(root)])
            model = Model({".": ([], ["app"]), "app": ([("server.py", "high")], [])},
                          {"app/server.py": [(4, 5, "core", "start function")]})
            ctx = {"complete_fn": model, "search_model": "claude-haiku-5-5"}
            with patch("ctxagent.config.key_problem", return_value=None):
                result = asyncio.run(tools.t_smart_search(sb, "where the server starts", path=str(root), ctx=ctx))
                self.assertIn("     4\tdef start(port):", result.text)
                self.assertEqual(result.meta["files"], 1)
                self.assertEqual(model.calls[0]["model"], "claude-haiku-5-5")
                with self.assertRaises(tools.ToolError):
                    asyncio.run(tools.t_smart_search(sb, " ", path=str(root), ctx=ctx))
                with self.assertRaises(tools.ToolError):
                    asyncio.run(tools.t_smart_search(sb, "x", path=str(root / "README.md"), ctx=ctx))


if __name__ == "__main__":
    unittest.main(verbosity=1)
