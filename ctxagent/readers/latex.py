"""Structural outlines for LaTeX sources.

A .tex file is unusually well suited to static outlining: the structure is
explicit markup rather than something to be inferred, so an outline can be exact
and still tiny. A 40-page paper outlines to a few hundred tokens with every
section, figure caption and equation label intact.

Tiers:

  0  document class, title, author, top-level sections, counts
  1  + every sectioning depth, packages, abstract opening
  2  + figures and tables with captions, labels and graphics paths;
     labelled equations
  3  + the first sentence of every paragraph, attributed to its section
  4  + a second sentence per paragraph, the full abstract, \\input/\\include
     children, and custom macro definitions

Brace matching is done properly rather than with a greedy regex, because
captions routinely contain nested groups (`\\caption{The $\\alpha$-phase
\\textbf{after} etching}`) and a naive `\\{(.*?)\\}` truncates them.
"""

from __future__ import annotations

import re

# Sectioning commands, outermost first. Index == depth - 1.
SECTION_ORDER = [
    "part", "chapter", "section", "subsection", "subsubsection",
    "paragraph", "subparagraph",
]
_SECTION_RE = re.compile(
    r"\\(" + "|".join(SECTION_ORDER) + r")\*?\s*(?:\[[^\]]*\])?\s*\{",
)
_ENV_BEGIN = re.compile(r"\\begin\s*\{([A-Za-z*]+)\}")
_INCLUDE_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]*)\}")
_NEWCMD_RE = re.compile(r"\\(?:newcommand|renewcommand|DeclareMathOperator)\*?\s*\{?\\([A-Za-z@]+)")
_CITE_RE = re.compile(r"\\(?:cite|citep|citet|autocite|parencite)\b")
_REF_RE = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref|pageref)\b")

FLOAT_ENVS = {"figure", "figure*", "wrapfigure", "subfigure", "SCfigure"}
TABLE_ENVS = {"table", "table*", "longtable", "sidewaystable"}
MATH_ENVS = {"equation", "equation*", "align", "align*", "gather", "gather*",
             "multline", "multline*", "eqnarray", "IEEEeqnarray"}
CODE_ENVS = {"lstlisting", "verbatim", "minted", "algorithm", "algorithmic"}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# lexing helpers
# ---------------------------------------------------------------------------

def strip_comments(src: str) -> str:
    """Remove % comments, respecting \\% escapes. Keeps line count intact."""
    out = []
    for line in src.split("\n"):
        res, i, n = [], 0, len(line)
        while i < n:
            c = line[i]
            if c == "\\" and i + 1 < n:
                res.append(line[i:i + 2])
                i += 2
                continue
            if c == "%":
                break
            res.append(c)
            i += 1
        out.append("".join(res))
    return "\n".join(out)


def braced(src: str, open_idx: int) -> tuple[str, int]:
    """Content of the group starting at `open_idx` (which must be '{').

    Returns (content, index just past the closing brace).
    """
    if open_idx >= len(src) or src[open_idx] != "{":
        return "", open_idx
    depth, i = 0, open_idx
    while i < len(src):
        c = src[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[open_idx + 1:i], i + 1
        i += 1
    return src[open_idx + 1:], len(src)


def _arg_after(src: str, cmd: str) -> str | None:
    """The first braced argument of `\\cmd`, anywhere in src."""
    m = re.search(r"\\" + cmd + r"\s*(?:\[[^\]]*\])?\s*\{", src)
    if not m:
        return None
    return braced(src, m.end() - 1)[0]


#: siunitx is ubiquitous in engineering sources, and a naive command-stripper
#: turns `\SI{500}{\hour}` into "500 ." -- dropping the unit, which is usually
#: the part that matters. Expand these before stripping anything else.
_SI_UNIT = re.compile(r"\\(?:si|unit)\s*\{([^}]*)\}")
_SI_QTY = re.compile(r"\\(?:SI|qty|SIrange|qtyrange)\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}\s*\{([^}]*)\}")
_SI_NUM = re.compile(r"\\(?:num|numrange)\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")


def _expand_units(t: str) -> str:
    def unit_words(u: str) -> str:
        parts = re.findall(r"\\([A-Za-z]+)", u)
        return " ".join(parts) if parts else u.strip()

    t = _SI_QTY.sub(lambda m: f"{m.group(1).strip()} {unit_words(m.group(2))}", t)
    t = _SI_UNIT.sub(lambda m: unit_words(m.group(1)), t)
    t = _SI_NUM.sub(lambda m: m.group(1).strip(), t)
    return t


def to_text(src: str) -> str:
    """LaTeX -> readable prose, for extract_text and the LLM summariser."""
    s = strip_comments(src)
    body = s.split(r"\begin{document}", 1)
    s = body[1] if len(body) > 1 else s
    s = s.split(r"\end{document}", 1)[0]
    # Drop wholly non-prose environments.
    for env in ("tikzpicture", "lstlisting", "verbatim", "minted", "tabular",
                "algorithmic", "thebibliography"):
        s = re.sub(r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}",
                   f"[{env} omitted]", s, flags=re.S)
    s = re.sub(r"\\(?:label|cite\w*|ref|eqref|autoref|cref|Cref|index)\s*\{[^}]*\}", "", s)
    # Units before generic command stripping, or "\SI{500}{\hour}" becomes "500".
    s = _expand_units(s)
    s = re.sub(r"\\(?:section|subsection|subsubsection|chapter|part|paragraph)\*?\s*\{",
               "\n\n## ", s)
    s = re.sub(r"\\(?:textbf|textit|emph|texttt|mathrm|text)\s*\{", "", s)
    s = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", "\n", s)
    s = re.sub(r"\\[A-Za-z@]+\*?\s*(?:\[[^\]]*\])?", " ", s)
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n\n", s).strip()


def _clean_inline(text: str) -> str:
    """Flatten a fragment (a caption, a heading, a sentence) to plain text."""
    t = re.sub(r"\\(?:label|cite\w*|ref|eqref|autoref|cref|Cref)\s*\{[^}]*\}", "", text)
    t = _expand_units(t)
    t = re.sub(r"\\and\b", " and ", t)
    t = re.sub(r"\\(?:textbf|textit|emph|texttt|mathrm|text|textsc)\s*\{", "", t)
    t = re.sub(r"\\[A-Za-z@]+\*?\s*(?:\[[^\]]*\])?", " ", t)
    t = t.replace("{", "").replace("}", "").replace("~", " ").replace("\\", "")
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _line_of(src: str, idx: int) -> int:
    return src.count("\n", 0, idx) + 1


def _sections(src: str) -> list[dict]:
    out = []
    for m in _SECTION_RE.finditer(src):
        title, end = braced(src, m.end() - 1)
        out.append({
            "cmd": m.group(1),
            "depth": SECTION_ORDER.index(m.group(1)) + 1,
            "title": _clean_inline(title),
            "line": _line_of(src, m.start()),
            "start": m.start(),
            "body_start": end,
        })
    for i, s in enumerate(out):
        s["body_end"] = out[i + 1]["start"] if i + 1 < len(out) else len(src)
    return out


def _environments(src: str) -> list[dict]:
    """Every environment block, with its caption/label/graphics if it has them."""
    out = []
    for m in _ENV_BEGIN.finditer(src):
        name = m.group(1)
        close = re.search(r"\\end\s*\{" + re.escape(name) + r"\}", src[m.end():])
        end = m.end() + close.end() if close else len(src)
        block = src[m.start():end]
        caption = _arg_after(block, "caption")
        entry = {
            "env": name,
            "line": _line_of(src, m.start()),
            "caption": _clean_inline(caption) if caption else None,
            "label": _arg_after(block, "label"),
            "graphics": re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}",
                                   block),
        }
        out.append(entry)
    return out


def _paragraph_openings(src: str, sec: dict, max_sentences: int) -> list[str]:
    """First sentence(s) of each paragraph inside a section's body."""
    body = src[sec["body_start"]:sec["body_end"]]
    # Floats, display maths, code and the bibliography are not prose; without
    # this the reference list reads as the closing paragraph of the last section.
    body = re.sub(r"\\begin\{(figure|table|longtable|tabular|equation|align|gather|"
                  r"multline|tikzpicture|lstlisting|verbatim|minted|algorithm|"
                  r"algorithmic|thebibliography)\*?\}.*?\\end\{\1\*?\}",
                  " ", body, flags=re.S)
    body = re.sub(r"\\bibliography\w*\s*\{[^}]*\}|\\printbibliography", " ", body)
    out = []
    for chunk in re.split(r"\n\s*\n", body):
        text = _clean_inline(chunk)
        if len(text) < 40:
            continue
        sentences = _SENTENCE_END.split(text)
        opening = " ".join(sentences[:max_sentences]).strip()
        if len(opening) > 400:
            opening = opening[:400].rsplit(" ", 1)[0] + " …"
        out.append(opening)
    return out


# ---------------------------------------------------------------------------
# outline
# ---------------------------------------------------------------------------

def outline(raw: str, level: int) -> str:
    src = strip_comments(raw)
    doc_split = src.split(r"\begin{document}", 1)
    preamble = doc_split[0]
    body = doc_split[1] if len(doc_split) > 1 else src

    out: list[str] = []

    # ---- preamble -------------------------------------------------------
    out.append("--- document ---")
    cls = re.search(r"\\documentclass\s*(?:\[([^\]]*)\])?\s*\{([^}]*)\}", preamble)
    if cls:
        opts = f"  [{cls.group(1)}]" if cls.group(1) else ""
        out.append(f"class: {cls.group(2)}{opts}")
    for field in ("title", "author", "date"):
        val = _arg_after(preamble, field) or _arg_after(body, field)
        if val:
            out.append(f"{field}: {_clean_inline(val)[:300]}")

    if level >= 1:
        pkgs = []
        for m in re.finditer(r"\\usepackage\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", preamble):
            pkgs += [p.strip() for p in m.group(1).split(",") if p.strip()]
        if pkgs:
            shown = pkgs if level >= 2 else pkgs[:20]
            out.append(f"packages ({len(pkgs)}): " + ", ".join(shown)
                       + (" …" if len(shown) < len(pkgs) else ""))

    abstract = None
    am = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    if am:
        abstract = _clean_inline(am.group(1))
    if abstract:
        if level >= 4:
            out.append(f"abstract: {abstract}")
        elif level >= 1:
            first = _SENTENCE_END.split(abstract)[0]
            out.append(f"abstract: {first[:400]}")
        else:
            out.append(f"abstract: present ({len(abstract.split())} words)")

    # ---- sections -------------------------------------------------------
    secs = _sections(body)
    max_depth = 3 if level == 0 else (5 if level == 1 else 7)
    shown = [s for s in secs if s["depth"] <= max_depth]
    out.append("")
    out.append(f"--- sections ({len(secs)} total, showing depth ≤ {max_depth}) ---")
    if shown:
        # Indent relative to the shallowest command the document actually uses:
        # an article that starts at \section should not be indented two levels
        # because \part and \chapter exist in the abstract ordering.
        base = min(s["depth"] for s in shown)
        for s in shown:
            indent = "  " * (s["depth"] - base)
            suffix = f"   (line {s['line']})" if level >= 1 else ""
            out.append(f"{indent}\\{s['cmd']}{{{s['title']}}}{suffix}")
    else:
        out.append("(none at this depth)")
    hidden = len(secs) - len(shown)
    if hidden:
        out.append(f"… {hidden} deeper sectioning command(s) hidden at this level")

    # ---- floats and math -------------------------------------------------
    envs = _environments(body)
    figs = [e for e in envs if e["env"] in FLOAT_ENVS]
    tabs = [e for e in envs if e["env"] in TABLE_ENVS]
    maths = [e for e in envs if e["env"] in MATH_ENVS]
    codes = [e for e in envs if e["env"] in CODE_ENVS]

    if level < 2:
        bits = [f"{len(figs)} figures", f"{len(tabs)} tables",
                f"{len(maths)} display equations", f"{len(codes)} code/algorithm blocks",
                f"{len(_CITE_RE.findall(body))} citations",
                f"{len(_REF_RE.findall(body))} cross-references"]
        out.append("")
        out.append("--- contents ---")
        out.append(", ".join(bits))
    else:
        for label, items in (("figures", figs), ("tables", tabs)):
            out.append("")
            out.append(f"--- {label} ({len(items)}) ---")
            for e in items:
                head = f"  line {e['line']}"
                if e["label"]:
                    head += f"  [{e['label']}]"
                out.append(head)
                if e["caption"]:
                    cap = e["caption"] if level >= 3 else e["caption"][:200]
                    out.append(f"      caption: {cap}")
                if e["graphics"]:
                    out.append(f"      graphics: {', '.join(e['graphics'])}")
            if not items:
                out.append("  (none)")
        labelled = [e for e in maths if e["label"]]
        out.append("")
        out.append(f"--- display equations ({len(maths)}, {len(labelled)} labelled) ---")
        for e in labelled[:60]:
            out.append(f"  line {e['line']}  [{e['label']}]  ({e['env']})")
        if codes:
            out.append("")
            out.append(f"--- code / algorithms ({len(codes)}) ---")
            for e in codes[:40]:
                out.append(f"  line {e['line']}  ({e['env']})"
                           + (f"  {e['caption']}" if e["caption"] else ""))

    # ---- paragraph openings ----------------------------------------------
    if level >= 3:
        n_sent = 1 if level == 3 else 2
        out.append("")
        out.append(f"--- paragraph openings ({n_sent} sentence"
                   f"{'s' if n_sent > 1 else ''} each) ---")
        targets = shown or [{"title": "(document)", "depth": 1, "cmd": "document",
                             "body_start": 0, "body_end": len(body)}]
        total = 0
        for s in targets:
            opens = _paragraph_openings(body, s, n_sent)
            if not opens:
                continue
            out.append(f"  \\{s['cmd']}{{{s['title']}}}")
            for o in opens[: 12 if level >= 4 else 6]:
                out.append(f"      • {o}")
            total += len(opens)
        if not total:
            out.append("  (no prose paragraphs found)")

    # ---- children and macros ---------------------------------------------
    if level >= 4:
        children = _INCLUDE_RE.findall(src)
        if children:
            out.append("")
            out.append(f"--- \\input / \\include ({len(children)}) ---")
            out.append(", ".join(children))
            out.append("  (outline each child separately; they are not inlined here)")
        macros = _NEWCMD_RE.findall(preamble)
        if macros:
            out.append("")
            out.append(f"--- custom macros ({len(macros)}) ---")
            out.append(", ".join("\\" + m for m in dict.fromkeys(macros)))

    words = len(to_text(raw).split())
    out.append("")
    out.append(f"[{len(secs)} sections, {len(figs)} figures, {len(tabs)} tables, "
               f"{len(maths)} display equations, ~{words} words of prose]")
    return "\n".join(out)
