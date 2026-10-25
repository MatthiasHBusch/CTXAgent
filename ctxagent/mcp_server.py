"""An MCP server exposing ctxagent's readers to Claude Code (or any MCP client).

The three tools here fill gaps in a standard coding-agent toolset, which
normally offers only `read whole file` and `grep for a pattern`:

  folder_overview   -- what is in this tree, with content, at a chosen depth
  document_outline  -- the structure of one file, without its content
  focus_file        -- one named part in full, embedded in the collapsed rest

All three are deterministic, need no model call, and cost nothing but the
tokens they return. That last property is the point: an agent cannot compact
what it never read, and unlike a summary these cannot hallucinate.

Run:  python -m ctxagent.mcp_server
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `python -m ctxagent.mcp_server` and `python path/to/mcp_server.py` alike.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from ctxagent.readers import (  # noqa: E402
    MAX_LEVEL,
    SUPPORTED_EXTENSIONS,
    describe_static_levels,
    outline,
)
from ctxagent.readers import fold as fold_mod  # noqa: E402
from ctxagent.readers import folder as folder_mod  # noqa: E402

mcp = FastMCP("ctxread")

# Optional confinement. Unset means "anything the process can read", which
# matches what the client's own file tools can already reach.
_ROOTS = [Path(p).resolve() for p in
          os.environ.get("CTXREAD_ROOTS", "").split(os.pathsep) if p.strip()]


def _resolve(path: str) -> Path:
    """Resolve `path` against cwd and check it against CTXREAD_ROOTS."""
    p = Path(os.path.expanduser(path))
    p = (Path.cwd() / p).resolve() if not p.is_absolute() else p.resolve()
    if _ROOTS and not any(p == r or r in p.parents for r in _ROOTS):
        raise ValueError(
            f"{p} is outside the allowed roots "
            f"({', '.join(str(r) for r in _ROOTS)})")
    return p


@mcp.tool(
    description=(
        "Map a directory tree, showing each file's CONTENTS at a decreasing "
        "level of detail as you descend. Use this to orient yourself in an "
        "unfamiliar codebase or folder BEFORE reading anything -- it answers "
        "'what is in here and which file do I want?' in one call, where a "
        "glob gives you only names and forces you to guess.\n\n"
        "level 0 = names only (a plain listing).\n"
        "level 1 = every file outlined at level 0 (classes, top headings).\n"
        "level 2 = every file outlined at level 1 (function signatures too).\n"
        "level 3+ = correspondingly deeper.\n"
        "Descending into a nested folder costs one further level, so detail "
        "fades with distance and the total stays bounded. Start at level 2 "
        "for a source tree you do not know; raise it for one subdirectory "
        "rather than for the whole repo."
    )
)
def folder_overview(path: str, level: int = 2, token_budget: int = 0) -> str:
    """Recursive folder map with per-file content at decreasing detail."""
    root = _resolve(path)
    if not root.is_dir():
        return f"[not a directory: {root}]"
    return folder_mod.overview(root, max(0, int(level)),
                               token_budget=max(0, int(token_budget)))


@mcp.tool(
    description=(
        "Show the STRUCTURE of one document without its content: class and "
        "function signatures for code, heading trees for markdown/LaTeX, "
        "column names and dtypes for CSV/Excel, key structure for JSON/YAML, "
        "the outline of a PDF or .docx. Use it instead of reading a file "
        "whole when you need to know what a file contains rather than what "
        "it says -- typically 10-50x cheaper.\n\n"
        "level 0 = barest skeleton (class/section names).\n"
        "level 1 = the usual choice: signatures and headings.\n"
        "level 2-4 = progressively more (docstrings, fields, figures).\n"
        "level 5 = the full text, i.e. an ordinary read.\n"
        "Levels here are structural depth, not a token budget: the output is "
        "as long as the file's structure requires."
    )
)
def document_outline(path: str, level: int = 1) -> str:
    """Deterministic structural outline of a single document."""
    p = _resolve(path)
    if p.is_dir():
        return (f"[{p} is a directory -- use folder_overview]")
    if not p.exists():
        return f"[not found: {p}]"
    return outline(p, max(0, min(MAX_LEVEL, int(level))))


@mcp.tool(
    description=(
        "Read ONE named part of a file in full, embedded in a collapsed "
        "outline of everything else -- a folded JSON viewer with a single key "
        "expanded. Prefer this over grep-then-read-at-an-offset: you get the "
        "whole part with nothing truncated, plus the surrounding structure, "
        "so you can see what else exists and name it next.\n\n"
        "`select` accepts several forms, and you may pass more than one:\n"
        "  'run'                  -- by name\n"
        "  'CompactionEngine.run' -- dotted path (code, and JSON/YAML keys)\n"
        "  '/^_snap/'             -- regex between slashes\n"
        "  'L120-160'             -- a raw line range, works on ANY file\n"
        "Exact paths outrank suffixes, which outrank regexes, which outrank "
        "substring hits. If nothing matches you get the list of addressable "
        "names back, so the retry can be precise. If the selection exceeds "
        "max_tokens it is refused with its size rather than silently cut."
    )
)
def focus_file(path: str, select: list[str], level: int = 1,
               max_tokens: int = 24_000) -> str:
    """Unfold the selected parts of a file inside its collapsed outline."""
    p = _resolve(path)
    if p.is_dir():
        return f"[{p} is a directory -- use folder_overview]"
    if not p.exists():
        return f"[not found: {p}]"
    if isinstance(select, str):          # tolerate a bare string
        select = [select]
    if not select:
        return "[focus_file needs at least one selector; see the tool description]"
    return fold_mod.fold(p, list(select), level=max(0, int(level)),
                         max_tokens=max(500, int(max_tokens)))


@mcp.tool(
    description=("Which file types these readers understand, and what each "
                 "outline level means. Call once if unsure.")
)
def reader_capabilities() -> str:
    """Supported extensions and the meaning of each level."""
    return (
        "Structural levels (document_outline, folder_overview, focus_file):\n"
        + describe_static_levels()
        + "\n\nRecognised extensions:\n  "
        + ", ".join(SUPPORTED_EXTENSIONS)
        + "\n\nAnything else is treated as plain text; line ranges (L10-40) "
          "work on it regardless."
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
