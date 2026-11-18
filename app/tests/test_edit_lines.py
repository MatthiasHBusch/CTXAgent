"""Editing by line range, checked by the text of the range's first and last line.

* edit_lines(path, start_line, end_line, start_text, end_text, new_text) replaces the lines
  only when start_text and end_text are exactly the text of those two lines. A line number
  alone would hit other lines once the file has changed.
* Neither text may be blank: a range starts and ends at lines that have text.
* prepare_edit_plan takes the same edits. There every line number refers to the file as it
  is, so several ranges of one file do not shift each other.
* Line endings, BOM and a missing final line end are kept.

Run:  python -m tests.test_edit_lines
"""
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctxagent import config, edit_plans, tools

SOURCE = ("import math\n"            # 1
          "\n"                       # 2
          "def area(r):\n"           # 3
          "    if r < 0:\n"          # 4
          "        raise ValueError('negative')\n"   # 5
          "    return math.pi * r * r\n"             # 6
          "\n"                       # 7
          "def perimeter(r):\n"      # 8
          "    return 2 * math.pi * r\n")            # 9


class EditLines(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "geo.py"
        self.path.write_bytes(SOURCE.encode("utf-8"))
        self.sb = tools.Sandbox([str(self.root)])

    def tearDown(self):
        self.tmp.cleanup()

    def edit(self, *args, **kw):
        return asyncio.run(tools.t_edit_lines(self.sb, str(self.path), *args, **kw))

    def test_replaces_the_range_when_both_texts_match(self):
        result = self.edit(4, 5, "    if r < 0:", "        raise ValueError('negative')",
                           "    if r < 0:\n        return 0.0")
        self.assertIn("lines 4-5 replaced by 2 line(s)", result.text)
        self.assertTrue(result.mutates)
        self.assertEqual(self.path.read_text(encoding="utf-8"),
                         SOURCE.replace("        raise ValueError('negative')\n", "        return 0.0\n"))

    def test_a_shifted_line_number_is_refused_and_nothing_changes(self):
        with self.assertRaises(tools.ToolError) as caught:
            self.edit(5, 6, "    if r < 0:", "    return math.pi * r * r", "x")
        message = str(caught.exception)
        self.assertIn("start_text does not match line 5", message)
        self.assertIn("Line 5 is: \"        raise ValueError('negative')\"", message)
        self.assertIn("That text is line 4.", message)
        self.assertEqual(self.path.read_bytes(), SOURCE.encode("utf-8"))
        with self.assertRaises(tools.ToolError) as caught:           # wrong indentation is a mismatch
            self.edit(4, 4, "if r < 0:", "if r < 0:", "x")
        self.assertIn("not a whole line of the file", str(caught.exception))
        with self.assertRaises(tools.ToolError) as caught:           # the end line is checked as well
            self.edit(3, 6, "def area(r):", "    return 2 * math.pi * r", "x")
        self.assertIn("end_text does not match line 6", str(caught.exception))
        self.assertEqual(self.path.read_bytes(), SOURCE.encode("utf-8"))

    def test_blank_texts_and_bad_ranges_are_refused(self):
        for args in ((2, 3, "", "def area(r):", "x"),            # empty
                     (2, 3, "   ", "def area(r):", "x"),         # blank
                     (3, 7, "def area(r):", "", "x"),
                     (6, 5, "a", "b", "x"),                      # end before start
                     (9, 12, "    return 2 * math.pi * r", "x", "x")):   # beyond the file
            with self.assertRaises(tools.ToolError):
                self.edit(*args)
        self.assertEqual(self.path.read_bytes(), SOURCE.encode("utf-8"))

    def test_delete_and_insert(self):
        self.edit(8, 9, "def perimeter(r):", "    return 2 * math.pi * r", "")
        self.assertEqual(self.path.read_text(encoding="utf-8"), SOURCE.split("def perimeter")[0])
        self.edit(1, 1, "import math", "import math", "import math\nimport sys")
        self.assertTrue(self.path.read_text(encoding="utf-8").startswith("import math\nimport sys\n\ndef area"))

    def test_crlf_bom_and_a_last_line_without_line_end_are_kept(self):
        raw = b"\xef\xbb\xbf" + "first\r\n  second\r\nthird".encode("utf-8")
        self.path.write_bytes(raw)
        self.edit(2, 3, "  second", "third", "  2nd\nmiddle\n3rd")
        self.assertEqual(self.path.read_bytes(), b"\xef\xbb\xbf" + "first\r\n  2nd\r\nmiddle\r\n3rd".encode("utf-8"))
        self.edit(1, 1, "first", "first", "1st\r\n")                 # a trailing line end adds no empty line
        self.assertEqual(self.path.read_bytes(), b"\xef\xbb\xbf" + "1st\r\n  2nd\r\nmiddle\r\n3rd".encode("utf-8"))

    def test_registered(self):
        spec = tools.REGISTRY["edit_lines"]
        self.assertEqual(spec.parameters["required"],
                         ["path", "start_line", "end_line", "start_text", "end_text", "new_text"])
        self.assertTrue(spec.dangerous)
        self.assertIn("edit_lines", tools.DEFAULT_TOOLS)


class PlanWithLineRanges(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "work"
        self.root.mkdir()
        self.path = self.root / "geo.py"
        self.path.write_bytes(SOURCE.encode("utf-8"))
        self.sb = tools.Sandbox([str(self.root)])
        self.data = patch.object(config, "DATA_DIR", Path(self.tmp.name) / "data")
        self.data.start()

    def tearDown(self):
        self.data.stop()
        self.tmp.cleanup()

    def test_all_ranges_use_the_numbers_of_the_file_as_it_is(self):
        changes = [{"path": str(self.path), "edits": [
            # Given top-down: the first edit adds two lines, the second still uses the old numbers.
            {"start_line": 4, "end_line": 5, "start_text": "    if r < 0:",
             "end_text": "        raise ValueError('negative')",
             "new_text": "    if r < 0:\n        # negative radius\n        # counts as empty\n        return 0.0"},
            {"start_line": 9, "end_line": 9, "start_text": "    return 2 * math.pi * r",
             "end_text": "    return 2 * math.pi * r", "new_text": "    return math.tau * r"},
            {"old_string": "import math\n", "new_string": "import math  # pi, tau\n"},
        ]}]
        plan = edit_plans.prepare(self.sb, "chat", changes)
        self.assertEqual(self.path.read_bytes(), SOURCE.encode("utf-8"))        # nothing written yet
        self.assertIn("+        return 0.0", plan["diff"])
        self.assertIn("-    return 2 * math.pi * r", plan["diff"])
        edit_plans.apply(self.sb, "chat", plan["plan_id"])
        text = self.path.read_text(encoding="utf-8")
        self.assertEqual(text, ("import math  # pi, tau\n\ndef area(r):\n    if r < 0:\n        # negative radius\n"
                                "        # counts as empty\n        return 0.0\n    return math.pi * r * r\n\n"
                                "def perimeter(r):\n    return math.tau * r\n"))

    def test_overlap_mismatch_and_incomplete_edits_are_refused(self):
        def prepare(*edits):
            return edit_plans.prepare(self.sb, "chat", [{"path": str(self.path), "edits": list(edits)}])

        a = {"start_line": 3, "end_line": 5, "start_text": "def area(r):",
             "end_text": "        raise ValueError('negative')", "new_text": "x"}
        b = {"start_line": 5, "end_line": 6, "start_text": "        raise ValueError('negative')",
             "end_text": "    return math.pi * r * r", "new_text": "y"}
        with self.assertRaisesRegex(ValueError, "overlap at line 5"):
            prepare(a, b)
        with self.assertRaisesRegex(ValueError, "start_text does not match line 3"):
            prepare({**a, "start_text": "def area(radius):"})
        with self.assertRaisesRegex(ValueError, "not blank"):
            prepare({**a, "end_text": ""})
        with self.assertRaisesRegex(ValueError, "old_string and new_string, or start_line"):
            prepare({"new_string": "x"})
        self.assertEqual(self.path.read_bytes(), SOURCE.encode("utf-8"))

    def test_schema_offers_both_kinds_of_edit(self):
        spec = tools.REGISTRY["prepare_edit_plan"]
        edit = spec.parameters["properties"]["changes"]["items"]["properties"]["edits"]["items"]
        for key in ("old_string", "new_string", "start_line", "end_line", "start_text", "end_text", "new_text"):
            self.assertIn(key, edit["properties"])
        self.assertNotIn("required", edit)
        self.assertIn("start_text and end_text", spec.description)


if __name__ == "__main__":
    unittest.main(verbosity=1)
