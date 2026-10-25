"""Tests for the document readers, level budgets, folder overviews, and
channel-aware dedup.

`level` deliberately means two different things, and both are tested:

  * **static outlines** take a STRUCTURAL level -- 0 is the barest skeleton and
    each step adds a layer, 5 is the full text. Free to compute, so nothing is
    rationed and the size follows from the file.
  * **LLM summaries** take a TOKEN BUDGET -- 100 / 300 / 1,000 / 3,000 / 10,000 /
    uncapped -- because there length is a real cost.

Run:  python -m tests.test_readers
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ctxagent import readers, tokens
from ctxagent.readers import folder
from ctxagent.compaction import CompactionEngine
from ctxagent.config import CompactionConfig
from ctxagent.segments import (
    Conversation, K_SYSTEM, K_USER, K_TOOL_CALL, K_TOOL_RESULT, S_SUPERSEDED,
)

_SAMPLE_TEX = r'''% A realistic engineering paper skeleton.
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage[ngerman,english]{babel}
\usepackage{siunitx}

\newcommand{\Mg}{\mathrm{Mg}}
\newcommand{\dd}[2]{\frac{\partial #1}{\partial #2}}

\title{Corrosion Kinetics of Magnesium Alloys under Cyclic Loading}
\author{M.~Busch \and J.~Doe}
\date{September 2026}

\begin{document}
\maketitle

\begin{abstract}
We investigate the corrosion behaviour of AZ31 magnesium alloys under cyclic
mechanical loading. A coupled electrochemical--mechanical model is proposed and
calibrated against \SI{500}{\hour} of immersion testing. The model reproduces
the observed pit-density evolution to within \SI{8}{\percent}.
\end{abstract}

\section{Introduction}
\label{sec:intro}

Magnesium alloys are attractive for lightweight structures because of their high
specific strength. Their poor corrosion resistance, however, remains the primary
barrier to adoption in load-bearing applications \cite{song2020}.

Previous work has treated corrosion and mechanical loading separately. This
separation is untenable once the passive film is repeatedly ruptured, as
Figure~\ref{fig:setup} shows.

\subsection{Scope of this work}
We restrict attention to AZ31 sheet in \SI{3.5}{\percent} NaCl solution. Thermal
effects are neglected throughout.

\subsubsection{Assumptions}
The electrolyte is assumed well mixed and the temperature constant at
\SI{298}{\kelvin}.

\section{Experimental setup}
\label{sec:setup}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/rig_overview.pdf}
  \caption{Overview of the cyclic-immersion rig, showing the \textbf{load frame}
  and the three-electrode cell used for potentiostatic control.}
  \label{fig:setup}
\end{figure}

Specimens were machined to \SI{2}{\milli\metre} thickness. Loading was applied
at \SI{5}{\hertz} with a stress ratio of $R = 0.1$.

\begin{table}[htbp]
  \centering
  \caption{Composition of the AZ31 sheet in weight percent.}
  \label{tab:composition}
  \begin{tabular}{lrrr}
    Element & Al & Zn & Mn \\
    wt\%    & 3.0 & 1.0 & 0.3 \\
  \end{tabular}
\end{table}

\section{Model}
\label{sec:model}

The corrosion current density is governed by

\begin{equation}
  \label{eq:butler}
  i = i_0 \left[ \exp\!\left(\frac{\alpha n F \eta}{RT}\right)
      - \exp\!\left(-\frac{(1-\alpha) n F \eta}{RT}\right) \right].
\end{equation}

Coupling to the mechanical field enters through the film-rupture rate:

\begin{align}
  \label{eq:rupture}
  \dot{\theta} &= k_r \dd{\varepsilon}{t} (1 - \theta) - k_h \theta.
\end{align}

\subsection{Numerical implementation}
Equations~\eqref{eq:butler} and~\eqref{eq:rupture} are integrated with a damped
Newton scheme to a tolerance of $10^{-11}$.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.6\textwidth]{figures/pit_density.pdf}
  \caption{Measured and predicted pit density over \SI{500}{\hour}.}
  \label{fig:pits}
\end{figure}

\section{Results and discussion}
\label{sec:results}

The model reproduces the measured pit density to within \SI{8}{\percent} over the
full test duration. Agreement degrades beyond \SI{400}{\hour}, which we
attribute to hydrogen accumulation not represented in the present formulation.

\section{Conclusions}
\label{sec:conclusions}

A coupled model has been presented and validated. Future work should address the
hydrogen-transport term.

\input{appendix_derivation}

\begin{thebibliography}{9}
\bibitem{song2020} G. Song, \emph{Corrosion of Magnesium Alloys}, 2020.
\end{thebibliography}

\end{document}
'''

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  -- {detail}" if detail and not cond else ""))


def tier(path: Path, t: int) -> str:
    """The structural body at level `t`, without the header line."""
    return readers.outline(path, t, include_header=False)


def structure_of(text: str) -> str:
    """Just the '--- structure ---' section, for 'does tier N mention X' checks."""
    marker = "--- structure ---"
    return text.split(marker)[1] if marker in text else text


# ---------------------------------------------------------------------------

def make_tree() -> Path:
    root = Path(tempfile.mkdtemp(prefix="ctxread_"))
    (root / "pkg").mkdir()
    (root / "pkg" / "deep").mkdir()

    (root / "pkg" / "solver.py").write_text(
        '"""Newton solver for nonlinear systems."""\n'
        "import numpy as np\n"
        "from scipy import linalg\n\n"
        "MAX_ITER = 50\n\n"
        "class Solver:\n"
        '    """Base solver."""\n'
        "    def __init__(self, tol: float = 1e-11) -> None:\n"
        '        """Set up."""\n'
        "        self.tol = tol\n"
        "    def solve(self, f, x0: np.ndarray) -> np.ndarray:\n"
        "        return x0\n\n"
        "class NewtonSolver(Solver):\n"
        "    def step(self, J, r):\n"
        "        return linalg.solve(J, -r)\n\n"
        "def make_solver(kind: str = 'newton') -> Solver:\n"
        "    return NewtonSolver()\n",
        encoding="utf-8")

    # A genuinely large module, so the budget search has something to choose from.
    classes = []
    for i in range(60):
        methods = [
            f"    def method_number_{j}(self, alpha, beta):\n"
            f'        """Compute variant {j} of transform {i}."""\n'
            f"        return {i} * {j} + alpha - beta"
            for j in range(10)
        ]
        classes.append(f"class Component{i}:\n"
                       f'    """Component {i} of the pipeline."""\n'
                       + "\n".join(methods))
    (root / "pkg" / "big_module.py").write_text(
        '"""A large module."""\n' + "\n\n".join(classes), encoding="utf-8")

    (root / "pkg" / "deep" / "util.js").write_text(
        "// helper utilities\n"
        "import fs from 'fs';\n"
        "export class Cache {\n"
        "  get(k) { return null; }\n"
        "}\n"
        "export function load(path) { return fs.readFileSync(path); }\n"
        "const build = (a, b) => a + b;\n",
        encoding="utf-8")

    (root / "README.md").write_text(
        "# Project Title\n\nIntro.\n\n## Installation\n### Requirements\n"
        "#### Deep\n## Usage\n### CLI\n## License\n", encoding="utf-8")

    (root / "data.csv").write_text(
        "name,age,city,score\nalice,30,Hamburg,9.5\nbob,25,Berlin,7.2\n"
        "carl,41,Hamburg,8.8\ndora,35,Bremen,6.1\n", encoding="utf-8")

    (root / "conf.json").write_text(json.dumps({
        "server": {"host": "localhost", "port": 8765,
                   "tls": {"enabled": True, "cert": "a.pem"}},
        "models": [{"id": "x", "ctx": 200000}],
    }), encoding="utf-8")

    (root / "paper.tex").write_text(_SAMPLE_TEX, encoding="utf-8")
    (root / "plain.txt").write_text(
        "\n".join(f"line {i}" for i in range(1, 40)), encoding="utf-8")

    (root / "nb.ipynb").write_text(json.dumps({
        "metadata": {"kernelspec": {"display_name": "Python 3"}},
        "cells": [
            {"cell_type": "markdown", "source": ["# Analysis\n", "## Setup\n"]},
            {"cell_type": "code", "source": ["import numpy as np\n", "x = 1\n"],
             "outputs": [{"output_type": "stream", "text": "ok"}]},
            {"cell_type": "code", "source": ["def helper(a):\n", "    return a\n"],
             "outputs": []},
        ],
    }), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# tiers
# ---------------------------------------------------------------------------

def test_code_tiers(root: Path):
    print("\n[1] code structural tiers")
    p = root / "pkg" / "solver.py"
    o = {t: readers.outline(p, t) for t in range(5)}

    check("tier 0 names the classes", "Solver" in o[0] and "NewtonSolver" in o[0])
    check("tier 0 omits method names", "solve(" not in structure_of(o[0]),
          structure_of(o[0])[:150])
    check("tier 1 adds method names", "solve" in o[1] and "step" in o[1])
    check("tier 1 hides signatures", "x0: np.ndarray" not in o[1])
    check("tier 2 adds full signatures",
          "tol: float=1e-11" in o[2].replace(" =", "=") or "tol: float = 1e-11" in o[2],
          o[2][:400])
    check("tier 2 adds base classes", "NewtonSolver(Solver)" in o[2])
    check("tier 3 adds docstrings", "Base solver" in o[3] and "Base solver" not in o[2])
    sizes = [tokens.count(o[i]) for i in range(5)]
    check("tiers are monotone in size", sizes == sorted(sizes), str(sizes))
    print(f"       tokens by tier: {sizes}")


def test_generic_code_tiers(root: Path):
    print("\n[2] non-Python code tiers")
    p = root / "pkg" / "deep" / "util.js"
    o0, o1, o2 = (tier(p, i) for i in (0, 1, 2))
    check("header comment captured", "helper utilities" in o0)
    check("tier 0 finds the class", "Cache" in o0)
    check("tier 0 omits functions", "load" not in structure_of(o0))
    check("tier 1 finds functions", "load" in o1)
    check("tier 1 finds arrow functions", "build" in o1, o1)
    check("tier 2 keeps the declaration text", "export function load(path)" in o2, o2)


def test_markdown_tiers(root: Path):
    print("\n[3] markdown tiers = heading depth")
    p = root / "README.md"
    o = {t: tier(p, t) for t in range(4)}
    check("tier 0 is the title only",
          "Project Title" in o[0] and "Installation" not in o[0])
    check("tier 1 adds H2", "Installation" in o[1] and "Requirements" not in o[1])
    check("tier 2 adds H3", "Requirements" in o[2] and "Deep" not in o[2])
    check("tier 3 adds H4", "Deep" in o[3])
    check("hidden-heading count reported", "deeper heading" in o[1])


def test_tabular_tiers(root: Path):
    print("\n[4] tabular tiers")
    p = root / "data.csv"
    o0, o1, o2, o3 = (tier(p, i) for i in (0, 1, 2, 3))
    check("tier 0 gives shape", "4 rows" in o0.replace(",", ""), o0[:200])
    check("tier 0 gives column names", "score" in o0 and "city" in o0)
    check("tier 1 gives row labels", "alice" in o1 and "dora" in o1)
    check("tier 2 gives stats", "unique=" in o2 or "min=" in o2, o2[:300])
    check("tier 3 renders rows", "Hamburg" in o3)


def test_structured_tiers(root: Path):
    print("\n[5] structured tiers = nesting depth")
    p = root / "conf.json"
    o0, o2 = tier(p, 0), tier(p, 2)
    check("tier 0 shows top keys", "server" in o0 and "models" in o0)
    check("tier 0 hides deep values", "a.pem" not in o0)
    check("tier 2 reaches nested scalars", "a.pem" in o2 and "8765" in o2)
    check("types reported, not dumped", "object(3 keys)" in o0, o0[:200])


def test_notebook_tiers(root: Path):
    print("\n[6] notebook tiers")
    p = root / "nb.ipynb"
    o0, o1, o2 = (tier(p, i) for i in (0, 1, 2))
    check("kernel reported", "Python 3" in o0)
    check("cell counts reported", "3 cells" in o0)
    check("tier 0 gives the H1", "Analysis" in o0)
    check("tier 1 lists code cells", "import numpy" in o1)
    check("tier 2 extracts code structure", "helper" in o2, o2[-400:])


# ---------------------------------------------------------------------------
# levels = token budgets
# ---------------------------------------------------------------------------

def test_static_levels(root: Path):
    print("\n[7] static levels are structural, not budgeted")
    small = root / "pkg" / "solver.py"
    big = root / "pkg" / "big_module.py"

    check("the LLM scale is the token one",
          [readers.LLM_LEVEL_TOKENS[i] for i in range(6)]
          == [100, 300, 1000, 3000, 10000, None],
          str(readers.LLM_LEVEL_TOKENS))
    check("static levels are described, not priced",
          "token" not in readers.describe_static_levels().lower(),
          readers.describe_static_levels())

    for path, label in ((small, "small"), (big, "big")):
        sizes = [tokens.count(readers.outline(path, i)) for i in range(5)]
        check(f"{label} file: detail grows with level", sizes == sorted(sizes),
              str(sizes))
        print(f"       {label} file tokens by level: {sizes}")

    # The same level means the same *kind* of detail on both files -- which is
    # the whole point of reverting: level 2 is "full signatures" everywhere.
    s2, b2 = readers.outline(small, 2), readers.outline(big, 2)
    check("level 2 gives signatures on both files",
          "tol: float" in s2.replace(" =", "=").replace("=", " = ")
          or "tol: float" in s2,
          s2[:300])
    check("level 2 reaches methods on the big file too",
          "method_number_0" in b2, b2[:300])
    check("a big file is not truncated to fit anything",
          tokens.count(b2) > 2000, str(tokens.count(b2)))

    check("level 3 adds docstrings", "Base solver" in readers.outline(small, 3))
    lvl5 = readers.outline(big, 5)
    check("level 5 returns the full document",
          "def method_number_9" in lvl5 and "return 0 * 0" in lvl5.replace("  ", " "),
          lvl5[:200])
    check("level 5 header says so", "full content" in lvl5.splitlines()[0])


def test_latex(root: Path):
    print("\n[8] LaTeX outline")
    p = root / "paper.tex"
    check("detected as latex", readers.detect_kind(p) == "latex")
    o = {i: readers.outline(p, i) for i in range(5)}

    check("L0 has class, title and top-level sections",
          "article" in o[0] and "Corrosion Kinetics" in o[0]
          and "Introduction" in o[0])
    check("L0 hides subsections", "Scope of this work" not in o[0])
    check("L0 counts figures and equations",
          "2 figures" in o[0] and "2 display equations" in o[0], o[0])
    check("L0 reports the abstract without quoting it",
          "abstract: present" in o[0])

    check("L1 adds every sectioning depth",
          "Scope of this work" in o[1] and "Assumptions" in o[1])
    check("L1 lists packages", "siunitx" in o[1])
    check("L1 opens the abstract", "AZ31 magnesium alloys" in o[1])
    check("sections are indented from the shallowest used",
          o[1].splitlines()[[i for i, l in enumerate(o[1].splitlines())
                             if "Introduction" in l][0]].startswith(r"\section"),
          "an article starting at \\section must not be indented")

    check("L2 gives figure captions and graphics paths",
          "three-electrode cell" in o[2] and "figures/rig_overview.pdf" in o[2])
    check("L2 gives figure and table labels",
          "fig:setup" in o[2] and "tab:composition" in o[2])
    check("L2 gives equation labels", "eq:butler" in o[2] and "eq:rupture" in o[2])

    check("L3 gives paragraph openings",
          "Magnesium alloys are attractive" in o[3])
    check("L3 attributes paragraphs to their section",
          "\\subsection{Scope of this work}" in o[3], o[3][-800:])
    check("the bibliography is not mistaken for prose",
          "thebibliography" not in o[3] and "G. Song" not in o[3])

    check("L4 lists \\input children", "appendix_derivation" in o[4])
    check("L4 lists custom macros", "\\dd" in o[4] and "\\Mg" in o[4])

    # siunitx: the unit is usually the part that matters
    check("siunitx quantities keep their units",
          "500 hour" in o[3] and "298 kelvin" in o[3], o[3][-900:])
    text = readers.extract_text(p)
    check("extract_text keeps units too", "8 percent" in text)
    check("extract_text drops markup", "\\section" not in text and "usepackage" not in text)
    sizes = [tokens.count(o[i]) for i in range(5)]
    check("detail grows with level", sizes == sorted(sizes), str(sizes))
    print(f"       tokens by level: {sizes}")


def test_folder_levels(root: Path):
    print("\n[9] folder overview levels")
    f = {i: folder.overview(root, i) for i in range(4)}

    check("L0 lists names only", "README.md" in f[0] and "Project Title" not in f[0])
    check("L0 shows subdirs", "pkg/" in f[0])
    check("L1 opens files at level 0", "Project Title" in f[1])
    check("L1 shows subfolder at level -1 (name only)",
          "pkg/" in f[1] and "solver.py" not in f[1], f[1][-300:])
    check("L2 lists subfolder contents (level 0)", "solver.py" in f[2])
    check("L3 descends two levels", "Solver" in f[3])
    sizes = [tokens.count(f[i]) for i in range(4)]
    check("size grows with level", sizes == sorted(sizes), str(sizes))

    capped = folder.overview(root, 4, token_budget=300)  # backstop, not per-file
    check("budget is enforced", "truncated" in capped.lower() or "budget" in capped.lower(),
          capped[-200:])
    check("budget actually caps the size", tokens.count(capped) < tokens.count(f[3]),
          f"{tokens.count(capped)} vs {tokens.count(f[3])}")
    check("default budget scales with level",
          folder.default_budget(5) > folder.default_budget(2),
          f"{folder.default_budget(2)} vs {folder.default_budget(5)}")
    print(f"       tokens by level: {sizes}")


def test_detect_and_extract(root: Path):
    print("\n[10] kind detection and text extraction")
    check("python detected", readers.detect_kind(root / "pkg" / "solver.py") == "python")
    check("js detected as c-like",
          readers.detect_kind(root / "pkg" / "deep" / "util.js") == "c-like")
    check("markdown detected", readers.detect_kind(root / "README.md") == "markdown")
    check("csv detected", readers.detect_kind(root / "data.csv") == "csv")
    check("notebook detected", readers.detect_kind(root / "nb.ipynb") == "notebook")
    check("unknown falls back", readers.detect_kind(root / "nope.qqq") == "unknown")
    nb_text = readers.extract_text(root / "nb.ipynb")
    check("notebook text includes code and output",
          "import numpy" in nb_text and "ok" in nb_text)
    check("csv text extracted", "Hamburg" in readers.extract_text(root / "data.csv"))


def test_llm_prompt_offline(root: Path):
    print("\n[11] LLM summary prompt construction (no network)")
    from ctxagent.readers import llm as rllm

    sys_p, user, meta = rllm.build_prompt(root / "pkg" / "solver.py", 1)
    check("LLM level 1 states its 300-token budget", "300 tokens" in sys_p, sys_p[:250])
    check("max_tokens tracks the budget", 300 <= meta["max_output_tokens"] <= 600,
          str(meta["max_output_tokens"]))
    check("user prompt carries the outline", "--- structure ---" in user)
    check("user prompt carries the content", "linalg.solve" in user)
    check("meta records the kind", meta["kind"] == "python")
    check("short file is not truncated", meta["truncated"] is False)

    _, _, m5 = rllm.build_prompt(root / "pkg" / "solver.py", 5)
    check("level 5 asks for no length limit",
          m5["max_output_tokens"] == rllm.UNCAPPED_OUTPUT_TOKENS,
          str(m5["max_output_tokens"]))

    big = root / "huge.txt"
    big.write_text("\n".join(f"line {i} of a very long document" for i in range(200_000)),
                   encoding="utf-8")
    _, _, meta_big = rllm.build_prompt(big, 2)
    check("huge file is truncated", meta_big["truncated"] is True)
    check("truncated prompt stays within budget",
          meta_big["sent_tokens"] < 70_000, str(meta_big["sent_tokens"]))
    print(f"       200k-line file -> {meta_big['sent_tokens']:,} prompt tokens")


def test_channel_dedup():
    print("\n[12] channel-aware supersession")

    async def fake(**kw):
        return "x"

    def add(conv, n, tool, body, detail, channel, mutates=False):
        cid = f"c{n}"
        conv.new("assistant", K_TOOL_CALL, "", tool_calls=[{
            "id": cid, "type": "function",
            "function": {"name": tool, "arguments": "{}"}}])
        return conv.new("tool", K_TOOL_RESULT, body, tool_call_id=cid, tool_name=tool,
                        resource_key="file:x.py", detail=detail, channel=channel,
                        mutates=mutates, observation={"version": "v2" if n >= 6 else "v1",
                            "representation": channel, "coverage": {"kind": "whole"}, "complete": True})

    conv = Conversation()
    conv.new("system", K_SYSTEM, "s", pinned=True)
    conv.new("user", K_USER, "go")
    full = add(conv, 1, "read_file", "FULL\n" + "\n".join(f"l{i}" for i in range(500)),
               100, "raw")
    sem = add(conv, 2, "summarize_file", "LLM says why\n" + "pad " * 400, 1, "semantic")
    out1 = add(conv, 3, "summarize_file",
               "OUT1\n" + "\n".join(f"def f{i}()" for i in range(300)), 2, "outline")

    cfg = CompactionConfig(rewrite_interval_batches=1, hot_zone_tokens=10 ** 7, masking_enabled=False,
                           rewrite_cost_gate=False,  # dedup mechanics, not economics
                           significance_enabled=False, deep_enabled=False,
                           protected_recent_observations=0)
    eng = CompactionEngine("t", cfg, fake)
    asyncio.run(eng.run(conv, 200_000))
    check("a shallow outline does not evict a full read", full.state != S_SUPERSEDED)
    check("an outline does not evict an LLM summary", sem.state != S_SUPERSEDED)

    add(conv, 4, "summarize_file",
        "OUT2\n" + "\n".join(f"def g{i}()" for i in range(300)), 3, "outline")
    asyncio.run(eng.run(conv, 200_000))
    check("a deeper outline evicts a shallower one", out1.state == S_SUPERSEDED)
    check("...without touching other channels",
          full.state != S_SUPERSEDED and sem.state != S_SUPERSEDED)

    add(conv, 5, "edit_file", "Edited x.py", 100, "raw", mutates=True)
    add(conv, 6, "read_file", "NEW\n" + "\n".join(f"l{i}" for i in range(500)), 100, "raw")
    asyncio.run(eng.run(conv, 200_000))
    check("a different version preserves prior evidence", full.state != S_SUPERSEDED)
    check("an edit does not erase a different representation", sem.state != S_SUPERSEDED,
          "Historical summaries remain evidence of the earlier version")


def test_fold(root: Path):
    print("\n[13] folded views (focus_file)")
    from ctxagent.readers import fold

    py = root / "pkg" / "big_module.py"
    full = tokens.count(py.read_text(encoding="utf-8"))
    out = fold.fold(py, ["Component7.method_number_3"])
    check("the selected method is unfolded in full",
          "def method_number_3(self, alpha, beta):" in out
          and "return 7 * 3 + alpha - beta" in out, out[:400])
    check("its class is opened to show the path", "class Component7" in out)
    check("sibling classes stay collapsed",
          "class Component8" in out and "method_number_3(self, alpha, beta):"
          not in out.split("class Component8")[1][:400])
    check("unrelated methods are not shown",
          out.count("return 7 * 3 + alpha - beta") == 1)
    check("far cheaper than reading the file",
          tokens.count(out) < full / 3, f"{tokens.count(out)} vs {full}")
    check("line numbers are shown for code", "| " in out and "L" in out)
    print(f"       big module {full} tok -> folded view {tokens.count(out)} tok")

    # structured data: a folded JSON with one key open
    j = fold.fold(root / "conf.json", ["server.tls"])
    check("nested key unfolds its whole subtree",
          '"cert": "a.pem"' in j, j)
    sibling_lines = [l for l in j.splitlines() if l.strip().startswith("· host")]
    check("siblings stay one line each, with a value preview",
          len(sibling_lines) == 1 and '"localhost"' in sibling_lines[0], j)
    check("unrelated branches stay folded",
          "models" in j and '"ctx"' not in j.split("models")[1][:80], j)

    # markdown / latex by heading
    md = fold.fold(root / "README.md", ["Usage"])
    check("markdown unfolds by heading", "## Usage" in md)
    tex = fold.fold(root / "paper.tex", ["Model"])
    check("latex unfolds by section",
          "eq:butler" in tex and "The corrosion current density" in tex)
    check("latex keeps other sections folded",
          "Magnesium alloys are attractive" not in tex)

    # tabular: a column
    csvout = fold.fold(root / "data.csv", ["city"])
    check("csv unfolds one column", "Hamburg" in csvout and "Bremen" in csvout)
    check("other columns stay collapsed",
          "· name" in csvout and "alice" not in csvout, csvout)

    # selector forms
    check("regex selector works",
          "def method_number_9" in fold.fold(py, ["/method_number_9$/"]))
    check("line range works on any file",
          "line 5" in fold.fold(root / "plain.txt", ["L5-8"]))
    check("several selectors open several parts",
          fold.fold(py, ["Component1.method_number_0",
                         "Component2.method_number_0"]).count("▼ ") == 2)

    # failure modes must teach, not just fail
    miss = fold.fold(py, ["no_such_thing"])
    check("a miss says so and lists what is addressable",
          "NO MATCH" in miss and "class Component0" in miss)
    capped = fold.fold(py, ["Component7"], max_tokens=50)
    check("an oversized selection is refused, not truncated mid-way",
          "NOT SHOWN" in capped and "exceeds the limit" in capped, capped[:600])


def main() -> int:
    root = make_tree()
    print(f"ctxagent readers -- tests   ({root})")
    for t in (test_code_tiers, test_generic_code_tiers, test_markdown_tiers,
              test_tabular_tiers, test_structured_tiers, test_notebook_tiers,
              test_static_levels, test_latex, test_folder_levels,
              test_detect_and_extract, test_llm_prompt_offline):
        t(root)
    test_fold(root)
    test_channel_dedup()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
