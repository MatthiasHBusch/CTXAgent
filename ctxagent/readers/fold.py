"""Folded views: the whole document collapsed, a chosen part open.

`summarize_file` answers "what is in here?". `read_document` answers "show me
everything". Neither answers the question you actually have most of the time:
*"show me this one part, and enough of the rest that I know where it sits."*

That is what this does. It builds a node tree for the document, prints every
node as one collapsed line, and expands the nodes you select to their full
source -- like a folded JSON viewer with one key open, or a file browser with
one directory expanded. Ancestors of a match are opened so the path is visible;
everything else stays a single line.

The saving is the point: pulling one 40-line method out of a 2,000-line module
costs ~600 tokens instead of ~20,000, and unlike a bare extract it arrives with
the surrounding structure, so the agent can see what else is there and ask for
it by name next time.

Selectors
---------
    run                      label or path, case-insensitive; exact wins
    CompactionEngine.run     dotted path
    server.tls               nested key in JSON/YAML/TOML
    /^_snap/                 regex, between slashes
    L120-160                 raw line range, works on any file

Node trees are built per format, reusing what each reader already knows:
Python via `ast`, other code via declaration regexes, markdown/LaTeX by heading
depth, JSON/YAML by nesting, notebooks by cell, PDFs by bookmark, tabular by
column.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import tokens
from . import MAX_OUTLINE_CHARS, detect_kind, read_text_file

MAX_UNFOLD_TOKENS = 24_000
MAX_NODES = 4_000


@dataclass
class Node:
    label: str                      # what the collapsed line shows
    path: str                       # addressable identity
    kind: str = "node"
    start: int = 0                  # 1-based, inclusive
    end: int = 0                    # 1-based, inclusive
    detail: str = ""                # collapsed extra (a value preview, a size)
    children: list["Node"] = field(default_factory=list)
    body: str | None = None         # explicit content (when not line-addressed)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


# ---------------------------------------------------------------------------
# tree builders
# ---------------------------------------------------------------------------

def _py_tree(src: str) -> list[Node]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    def build(node, prefix: str) -> list[Node]:
        out = []
        for n in node.body:
            if isinstance(n, ast.ClassDef):
                path = f"{prefix}{n.name}"
                out.append(Node(f"class {n.name}", path, "class",
                                n.lineno, _end(n), children=build(n, path + ".")))
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                path = f"{prefix}{n.name}"
                pre = "async def " if isinstance(n, ast.AsyncFunctionDef) else "def "
                try:
                    sig = f"({ast.unparse(n.args)})"
                except Exception:
                    sig = "(…)"
                out.append(Node(pre + n.name + sig, path, "function",
                                n.lineno, _end(n)))
            elif isinstance(n, ast.Assign) and prefix == "":
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        out.append(Node(t.id, t.id, "constant", n.lineno, _end(n)))
        return out

    def _end(n) -> int:
        return getattr(n, "end_lineno", n.lineno) or n.lineno

    return build(tree, "")


_DECL = re.compile(
    r"^\s*(?:export\s+|public\s+|private\s+|protected\s+|static\s+|async\s+|"
    r"abstract\s+|final\s+|inline\s+)*"
    r"(class|interface|struct|enum|trait|impl|type|function|fn|func|def|sub|proc)"
    r"\s+([A-Za-z_]\w*)")
_DECL_ARROW = re.compile(r"^\s*(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\(")
_DECL_MATLAB = re.compile(r"^\s*function\s+(?:\[[^\]]*\]|\w+)?\s*=?\s*([A-Za-z_]\w*)\s*\(")


def _generic_code_tree(src: str) -> list[Node]:
    lines = src.split("\n")
    found: list[Node] = []
    for i, ln in enumerate(lines, 1):
        name = kind = None
        m = _DECL.match(ln)
        if m:
            kind, name = ("class" if m.group(1) in
                          ("class", "interface", "struct", "enum", "trait", "impl", "type")
                          else "function"), m.group(2)
        elif _DECL_ARROW.match(ln):
            kind, name = "function", _DECL_ARROW.match(ln).group(1)
        elif _DECL_MATLAB.match(ln):
            kind, name = "function", _DECL_MATLAB.match(ln).group(1)
        if name:
            found.append(Node(ln.strip().rstrip("{").strip()[:110], name, kind, i, i))
    for a, b in zip(found, found[1:]):
        a.end = max(a.start, b.start - 1)
    if found:
        found[-1].end = len(lines)
    return found


def _heading_tree(pairs: list[tuple[int, str, int]], total_lines: int) -> list[Node]:
    """pairs: (depth, title, line). Nests by depth."""
    roots: list[Node] = []
    stack: list[Node] = []
    for depth, title, line in pairs:
        node = Node(title, title, "heading", line, total_lines)
        while stack and _depth_of(stack[-1]) >= depth:
            stack.pop()
        node.detail = f"h{depth}"
        if stack:
            stack[-1].children.append(node)
            node.path = stack[-1].path + " > " + title
        else:
            roots.append(node)
        stack.append(node)
    # close ranges: a heading ends where the next heading at any depth begins
    flat = sorted((n for r in roots for n in r.walk()), key=lambda n: n.start)
    for a, b in zip(flat, flat[1:]):
        a.end = max(a.start, b.start - 1)
    return roots


def _depth_of(n: Node) -> int:
    try:
        return int(n.detail[1:])
    except (ValueError, IndexError):
        return 1


def _markdown_tree(src: str) -> list[Node]:
    lines = src.split("\n")
    pairs, in_fence = [], False
    for i, ln in enumerate(lines, 1):
        if re.match(r"^\s*(```|~~~)", ln):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", ln)
        if m:
            pairs.append((len(m.group(1)), m.group(2).strip(), i))
    return _heading_tree(pairs, len(lines))


def _latex_tree(src: str) -> list[Node]:
    from . import latex as tex

    clean = tex.strip_comments(src)
    secs = tex._sections(clean)
    if not secs:
        return []
    base = min(s["depth"] for s in secs)
    pairs = [(s["depth"] - base + 1, s["title"] or s["cmd"], s["line"]) for s in secs]
    return _heading_tree(pairs, len(clean.split("\n")))


def _obj_tree(obj, prefix: str = "", depth: int = 0) -> list[Node]:
    out: list[Node] = []
    if depth > 8:
        return out
    # Containers carry a body too: unfolding `server.tls` must print the whole
    # subtree, not an empty frame. Nodes are line-less here, so the dump is the
    # only content they have.
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}{k}"
            out.append(Node(str(k), path, _type_name(v), detail=_preview(v),
                            children=_obj_tree(v, path + ".", depth + 1),
                            body=_dump(v)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:200]):
            path = f"{prefix.rstrip('.')}[{i}]"
            out.append(Node(f"[{i}]", path, _type_name(v), detail=_preview(v),
                            children=_obj_tree(v, path + ".", depth + 1),
                            body=_dump(v)))
    return out


def _type_name(v) -> str:
    if isinstance(v, dict):
        return "object"
    if isinstance(v, list):
        return "array"
    return type(v).__name__


def _preview(v) -> str:
    if isinstance(v, dict):
        return f"{{{len(v)} keys}}"
    if isinstance(v, list):
        return f"[{len(v)}]"
    s = json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= 60 else s[:57] + "…"


def _dump(v) -> str:
    return json.dumps(v, indent=2, ensure_ascii=False, default=str)


def _notebook_tree(src: str) -> list[Node]:
    try:
        nb = json.loads(src)
    except Exception:
        return []
    out = []
    for i, cell in enumerate(nb.get("cells", []), 1):
        s = cell.get("source", "")
        text = "".join(s) if isinstance(s, list) else str(s)
        first = next((l for l in text.splitlines() if l.strip()), "")
        out.append(Node(f"cell {i} [{cell.get('cell_type')}] {first[:80]}",
                        f"cell{i}", cell.get("cell_type", "cell"),
                        detail=f"{len(text.splitlines())} lines", body=text))
    return out


def _tabular_tree(path: Path, kind: str) -> list[Node]:
    try:
        import pandas as pd
    except Exception:
        return []
    try:
        if kind == "excel":
            sheets = pd.read_excel(path, sheet_name=None)
            out = []
            for name, df in sheets.items():
                cols = [Node(str(c), f"{name}.{c}", "column",
                             detail=str(df[c].dtype),
                             body=df[c].to_string(max_rows=500)) for c in df.columns]
                out.append(Node(f"sheet {name}", name, "sheet",
                                detail=f"{len(df)}x{len(df.columns)}", children=cols))
            return out
        df = (pd.read_parquet(path) if kind == "parquet"
              else pd.read_csv(path, sep="\t" if kind == "tsv" else ",",
                               nrows=200_000, low_memory=False))
        return [Node(str(c), str(c), "column", detail=str(df[c].dtype),
                     body=df[c].to_string(max_rows=500)) for c in df.columns]
    except Exception:
        return []


def _pdf_tree(path: Path) -> list[Node]:
    from . import docs

    kind, doc = docs._pdf_open(path)
    if doc is None:
        return []
    pages = docs._pdf_pages_text(kind, doc)
    toc = docs._pdf_outline_entries(kind, doc)
    if toc:
        pairs = [(d, t, p) for d, t, p in toc]
        roots, stack = [], []
        for d, title, page in pairs:
            node = Node(title, title, "bookmark", page, page, detail=f"p.{page}")
            while stack and _bm_depth(stack[-1]) >= d:
                stack.pop()
            node.detail = f"p.{page}|d{d}"
            if stack:
                stack[-1].children.append(node)
                node.path = stack[-1].path + " > " + title
            else:
                roots.append(node)
            stack.append(node)
        flat = sorted((n for r in roots for n in r.walk()), key=lambda n: n.start)
        for a, b in zip(flat, flat[1:]):
            a.end = max(a.start, b.start - 1)
        if flat:
            flat[-1].end = len(pages)
        for n in flat:
            n.body = "\n".join(pages[max(0, n.start - 1):n.end])
            n.detail = n.detail.split("|")[0]
        return roots
    return [Node(f"page {i}", f"p{i}", "page", i, i,
                 detail=f"{len(t.split())} words", body=t)
            for i, t in enumerate(pages, 1)]


def _bm_depth(n: Node) -> int:
    m = re.search(r"\|d(\d+)", n.detail)
    return int(m.group(1)) if m else 1


def _docx_tree(path: Path) -> list[Node]:
    from . import docs

    d = docs._docx(path)
    if d is None:
        return []
    paras = d.paragraphs
    pairs = []
    for i, p in enumerate(paras):
        style = (p.style.name or "") if p.style is not None else ""
        m = re.match(r"Heading (\d)", style)
        if m and p.text.strip():
            pairs.append((int(m.group(1)), p.text.strip(), i + 1))
    roots = _heading_tree(pairs, len(paras))
    for n in (x for r in roots for x in r.walk()):
        n.body = "\n".join(p.text for p in paras[n.start - 1:n.end] if p.text.strip())
    return roots


def build_tree(path: Path, kind: str, src: str | None = None) -> list[Node]:
    if kind == "python":
        return _py_tree(src or "")
    if kind in ("c-like", "ruby", "matlab", "julia", "r", "shell", "sql"):
        return _generic_code_tree(src or "")
    if kind == "markdown":
        return _markdown_tree(src or "")
    if kind == "latex":
        return _latex_tree(src or "")
    if kind in ("json", "jsonl"):
        try:
            data = json.loads((src or "").split("\n")[0] if kind == "jsonl" else (src or ""))
        except Exception:
            return []
        return _obj_tree(data)
    if kind == "yaml":
        try:
            import yaml

            return _obj_tree(yaml.safe_load(src or ""))
        except Exception:
            return []
    if kind == "toml":
        try:
            import tomllib

            return _obj_tree(tomllib.loads(src or ""))
        except Exception:
            return []
    if kind == "notebook":
        return _notebook_tree(src or "")
    if kind in ("csv", "tsv", "excel", "parquet"):
        return _tabular_tree(path, kind)
    if kind == "pdf":
        return _pdf_tree(path)
    if kind == "docx":
        return _docx_tree(path)
    return []


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

_LINE_RANGE = re.compile(r"^L(\d+)(?:\s*-\s*(\d+))?$", re.I)


def _matches(node: Node, sel: str) -> int:
    """0 = no match; higher = better. Lets an exact path beat a substring."""
    s = sel.strip()
    if s.startswith("/") and s.endswith("/") and len(s) > 2:
        try:
            return 3 if re.search(s[1:-1], node.path, re.I) or \
                        re.search(s[1:-1], node.label, re.I) else 0
        except re.error:
            return 0
    low = s.lower()
    if node.path.lower() == low or node.label.lower() == low:
        return 5
    if node.path.lower().endswith("." + low) or node.path.lower().endswith("> " + low):
        return 4
    if low in node.path.lower() or low in node.label.lower():
        return 2
    return 0


def select(roots: list[Node], selectors: list[str]) -> tuple[list[Node], list[str]]:
    """Best match per selector. Returns (nodes, selectors that matched nothing)."""
    all_nodes = [n for r in roots for n in r.walk()]
    chosen, missed = [], []
    for sel in selectors:
        scored = [(_matches(n, sel), -len(n.path), n) for n in all_nodes]
        scored = [t for t in scored if t[0] > 0]
        if not scored:
            missed.append(sel)
            continue
        best = max(scored, key=lambda t: (t[0], t[1]))[2]
        if best not in chosen:
            chosen.append(best)
    return chosen, missed


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _collapsed(node: Node, level: int) -> str:
    bits = [node.label]
    if level >= 1 and node.detail:
        bits.append(f"  {node.detail}")
    if level >= 1 and node.start:
        span = f"L{node.start}" + (f"-{node.end}" if node.end > node.start else "")
        bits.append(f"  [{span}]")
    if node.children:
        bits.append(f"  ({len(node.children)} inside)")
    return "".join(bits)


def _node_body(node: Node, lines: list[str]) -> str:
    if node.body is not None:
        return node.body
    if node.start:
        return "\n".join(lines[node.start - 1:node.end])
    return ""


def fold(path: Path, selectors: list[str], *, level: int = 1,
         kind: str | None = None, max_tokens: int = MAX_UNFOLD_TOKENS) -> str:
    kind = kind or detect_kind(path)
    src = ""
    if kind not in ("pdf", "docx", "pptx", "excel", "parquet"):
        try:
            src = read_text_file(path)
        except Exception as e:
            return f"[could not read {path}: {e}]"
    lines = src.split("\n")

    roots = build_tree(path, kind, src)

    # Raw line ranges work on anything, including formats with no tree at all.
    explicit: list[Node] = []
    rest: list[str] = []
    for sel in selectors:
        m = _LINE_RANGE.match(sel.strip())
        if m and lines:
            a = max(1, int(m.group(1)))
            b = min(len(lines), int(m.group(2) or m.group(1)))
            explicit.append(Node(f"lines {a}-{b}", f"L{a}-{b}", "range", a, b))
        else:
            rest.append(sel)

    chosen, missed = select(roots, rest)
    chosen = explicit + chosen

    if not roots and not explicit:
        return (f"[no foldable structure found in {path.name} ({kind}); "
                f"use summarize_file, or a line range like L1-80]")

    # Explicit ranges are not part of the document's tree, so they need to be
    # rendered as roots of their own or they are selected and never shown.
    roots = explicit + roots

    open_ids = {id(n) for n in chosen}
    ancestors: set[int] = set()

    def mark(node: Node) -> bool:
        hit = id(node) in open_ids
        for c in node.children:
            if mark(c):
                hit = True
                ancestors.add(id(node))
        return hit

    for r in roots:
        mark(r)

    out: list[str] = []
    header = [f"=== folded view: {path}  ·  {kind} ===",
              f"selectors: {', '.join(selectors) or '(none)'}"]
    if missed:
        header.append(f"NO MATCH for: {', '.join(missed)}  "
                      f"— names below are addressable; retry with one of them")
    out += header
    out.append("")

    budget = {"used": 0, "over": False}

    def emit(node: Node, depth: int) -> None:
        pad = "  " * depth
        opened = id(node) in open_ids
        if opened:
            body = _node_body(node, lines)
            cost = tokens.count(body)
            if budget["used"] + cost > max_tokens:
                budget["over"] = True
                out.append(f"{pad}▼ {_collapsed(node, level)}  "
                           f"[NOT SHOWN: {cost} tokens exceeds the limit; "
                           f"raise max_tokens or select something smaller]")
                return
            budget["used"] += cost
            out.append(f"{pad}▼ {_collapsed(node, level)}")
            width = len(str(node.end or 0))
            for k, ln in enumerate(body.split("\n")):
                if node.start and node.body is None:
                    out.append(f"{pad}  {str(node.start + k).rjust(width)} | {ln}")
                else:
                    out.append(f"{pad}  | {ln}")
            out.append(f"{pad}▲ end {node.path}")
            return
        out.append(f"{pad}· {_collapsed(node, level)}")
        if id(node) in ancestors:
            for c in node.children:
                emit(c, depth + 1)

    for r in roots:
        emit(r, 0)

    total = sum(1 for r in roots for _ in r.walk())
    out.append("")
    out.append(f"[{len(chosen)} of {total} nodes unfolded; "
               f"~{budget['used']} tokens of content shown"
               + ("; TRUNCATED" if budget["over"] else "") + "]")
    if not chosen:
        out.append("[nothing selected — the structure above is addressable by "
                   "label or path; pass one to `select`]")
    return "\n".join(out)[:MAX_OUTLINE_CHARS]
