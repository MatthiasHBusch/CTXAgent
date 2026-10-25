"""Structural outlines for source files.

Python goes through `ast`, which is exact. Everything else goes through a
line-oriented regex pass that recognises the declaration forms of C-like
languages, Ruby, MATLAB, Julia, R, shell and SQL -- imprecise, but it never
needs a parser per language and it degrades into "lines that look like
declarations", which is still far better than the whole file.

Level semantics (`level` clamps into range):

  0  module docstring / head comment, imports, top-level class & function NAMES
  1  + methods, + top-level constants
  2  + full signatures with defaults, annotations and decorators
  3  + the first line of every docstring
  4  + full docstrings, + nested functions
"""

from __future__ import annotations

import ast
import re
import textwrap


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

def _sig(node: ast.FunctionDef | ast.AsyncFunctionDef, full: bool) -> str:
    if not full:
        return f"{node.name}(…)"
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "…"
    ret = ""
    if node.returns is not None:
        try:
            ret = f" -> {ast.unparse(node.returns)}"
        except Exception:
            ret = ""
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{node.name}({args}){ret}"


def _doc(node, level: int) -> list[str]:
    if level < 3:
        return []
    try:
        d = ast.get_docstring(node)
    except Exception:
        d = None
    if not d:
        return []
    d = d.strip()
    if level == 3:
        first = d.split("\n\n")[0].strip().splitlines()
        return ['    """' + " ".join(x.strip() for x in first[:2])[:200] + '"""']
    return ['    """' + textwrap.shorten(" ".join(d.split()), 600, placeholder=" …") + '"""']


def _decorators(node, level: int) -> list[str]:
    if level < 2 or not getattr(node, "decorator_list", None):
        return []
    out = []
    for d in node.decorator_list:
        try:
            out.append(f"@{ast.unparse(d)}")
        except Exception:
            pass
    return out


def outline_python(src: str, level: int) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return (f"[could not parse as Python: {e}]\n"
                + outline_generic(src, level, "python-fallback"))

    out: list[str] = []

    # --- head: module docstring and imports ---------------------------------
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        text = " ".join(mod_doc.split())
        cap = 200 if level == 0 else 500 if level < 3 else 1500
        out.append("--- module docstring ---")
        out.append(textwrap.shorten(text, cap, placeholder=" …"))
        out.append("")

    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = ("." * (node.level or 0)) + (node.module or "")
            if level >= 2:
                imports += [f"{mod}.{a.name}" for a in node.names]
            else:
                imports.append(mod)
    if imports:
        uniq = sorted(set(imports))
        shown = uniq if level >= 2 else uniq[:30]
        out.append(f"--- imports ({len(uniq)}) ---")
        out.append(", ".join(shown) + (" …" if len(shown) < len(uniq) else ""))
        out.append("")

    # --- top-level constants -------------------------------------------------
    if level >= 1:
        consts = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        try:
                            v = ast.unparse(node.value)
                        except Exception:
                            v = "…"
                        consts.append(f"{t.id} = {textwrap.shorten(v, 70, placeholder='…')}")
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id.isupper():
                    consts.append(f"{node.target.id}: …")
        if consts:
            out.append(f"--- module constants ({len(consts)}) ---")
            out += consts[: 8 if level == 1 else 60]
            out.append("")

    # --- classes and functions ----------------------------------------------
    out.append("--- structure ---")
    n_cls = n_fn = 0
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            n_cls += 1
            bases = ""
            if level >= 2 and node.bases:
                try:
                    bases = "(" + ", ".join(ast.unparse(b) for b in node.bases) + ")"
                except Exception:
                    bases = ""
            out += _decorators(node, level)
            out.append(f"class {node.name}{bases}:  # line {node.lineno}")
            out += _doc(node, level)
            if level >= 1:
                members = [
                    m for m in node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                if not members:
                    out.append("    (no methods)")
                for m in members:
                    n_fn += 1
                    for d in _decorators(m, level):
                        out.append(f"    {d}")
                    out.append("    " + _sig(m, level >= 2))
                    out += ["    " + x for x in _doc(m, level)]
                if level >= 1:
                    attrs = [
                        t.id for m in node.body if isinstance(m, ast.Assign)
                        for t in m.targets if isinstance(t, ast.Name)
                    ]
                    if attrs and level >= 2:
                        out.append(f"    # class attrs: {', '.join(attrs[:15])}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n_fn += 1
            out += _decorators(node, level)
            out.append(_sig(node, level >= 2) + f"  # line {node.lineno}")
            out += _doc(node, level)
            if level >= 4:
                nested = [
                    c.name for c in ast.walk(node)
                    if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and c is not node
                ]
                if nested:
                    out.append(f"    # nested: {', '.join(nested[:12])}")

    if level == 0:
        # Level 0 wants names only, not signatures.
        out = [
            re.sub(r"^(async )?def (\w+)\(.*", r"def \2(…)", ln) for ln in out
        ]

    out.append("")
    out.append(f"[{n_cls} classes, {n_fn} functions/methods]")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Everything else
# ---------------------------------------------------------------------------

_DECL_PATTERNS = [
    # (label, regex)  -- matched against a stripped line
    ("class", re.compile(r"^(?:export\s+|public\s+|private\s+|abstract\s+|final\s+|static\s+)*"
                         r"(?:class|interface|struct|enum|trait|impl|type)\s+([A-Za-z_]\w*)")),
    ("func", re.compile(r"^(?:export\s+|public\s+|private\s+|protected\s+|static\s+|async\s+|"
                        r"inline\s+|virtual\s+|override\s+)*"
                        r"(?:function|fn|func|def|sub|proc)\s+([A-Za-z_]\w*)")),
    ("func", re.compile(r"^(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\("
                        r"[^)]*\)\s*(?::[^=]+)?=>")),
    ("method", re.compile(r"^(?:public|private|protected|static|async|override|virtual)"
                          r"[\w\s<>,\[\]]*\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$")),
    ("matlab", re.compile(r"^function\s+(?:\[[^\]]*\]|\w+)?\s*=?\s*([A-Za-z_]\w*)\s*\(")),
    ("sql", re.compile(r"^create\s+(?:or\s+replace\s+)?(?:table|view|function|procedure|index)"
                       r"\s+(?:if\s+not\s+exists\s+)?[`\"\[]?([\w.]+)", re.I)),
    ("shell", re.compile(r"^(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\)\s*\{")),
]

_COMMENT_STARTS = ("#", "//", "%", "--", "*", "/*", '"""', "'''", ";")


def _head_comment(lines: list[str], max_lines: int) -> list[str]:
    out = []
    for ln in lines[:40]:
        s = ln.strip()
        if not s:
            if out:
                break
            continue
        if s.startswith(_COMMENT_STARTS):
            out.append(s)
            if len(out) >= max_lines:
                break
        else:
            break
    return out


def outline_generic(src: str, level: int, kind: str) -> str:
    lines = src.splitlines()
    out: list[str] = []

    head = _head_comment(lines, 3 if level == 0 else 10 if level < 3 else 30)
    if head:
        out.append("--- header comment ---")
        out += head
        out.append("")

    if level >= 1:
        imports = [
            ln.strip() for ln in lines[:200]
            if re.match(r"^\s*(import|#include|using|require|from|library|source|use)\b", ln)
        ]
        if imports:
            out.append(f"--- imports ({len(imports)}) ---")
            out += imports[: 10 if level < 2 else 60]
            out.append("")

    out.append("--- structure ---")
    counts: dict[str, int] = {}
    found = 0
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if not s or s.startswith(_COMMENT_STARTS):
            continue
        indent = len(ln) - len(ln.lstrip())
        for label, rx in _DECL_PATTERNS:
            m = rx.match(s)
            if not m:
                continue
            is_class = label == "class"
            # level 0 -> classes only; level 1+ -> everything
            if level == 0 and not is_class:
                break
            # level 1 -> names only; level 2+ -> the declaration line verbatim
            if level >= 2:
                text = re.sub(r"\s*\{\s*$", "", s)
                text = textwrap.shorten(text, 160, placeholder=" …")
            else:
                text = f"{'class' if is_class else 'def'} {m.group(1)}(…)"
            out.append(" " * min(indent, 8) + f"{text}  # line {i}")
            counts[label] = counts.get(label, 0) + 1
            found += 1
            break
        if found > 800:
            out.append("… (truncated)")
            break

    if not found:
        out.append("(no declarations recognised)")
        if level >= 1:
            out.append("--- first lines ---")
            out += lines[: 10 * max(1, level)]

    out.append("")
    out.append("[" + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) + "]" if counts else "[]")
    return "\n".join(out)
