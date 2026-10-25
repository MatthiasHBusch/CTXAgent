"""Recursive folder overviews, with the same level-of-detail dial as files.

    level 0   just the directory listing — names, sizes, nothing opened
    level L   list the entries, and for each
                 file       -> static outline at level L-1
                 subfolder  -> folder overview at level L-2

Subfolders lose an extra level on every descent, so detail decays twice as fast
going down as it does going in. That is what keeps the output bounded: a level-3
overview of a deep tree shows rich detail at the top, names three levels down,
and stops. Without the extra decrement, `summarize_folder(repo, 3)` on a large
project would try to outline several hundred files.

A hard token budget backs that up, so a pathological tree truncates cleanly
instead of flooding the context it was meant to protect. That budget is a
backstop on the survey as a whole, not a per-file allowance -- individual static
outlines are not token-budgeted.
"""

from __future__ import annotations

from pathlib import Path

from .. import tokens
from . import MAX_OUTLINE_CHARS, detect_kind, outline

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".next",
    ".idea", ".vscode", ".ctxagent", "site-packages", ".ipynb_checkpoints",
    "target", ".gradle", ".tox", "env",
}
SKIP_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".bin", ".o", ".a",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".jpg", ".jpeg", ".png", ".gif",
    ".bmp", ".ico", ".svg", ".mp3", ".mp4", ".avi", ".mov", ".wav", ".ttf",
    ".woff", ".woff2", ".eot", ".db", ".sqlite", ".lock", ".pack", ".idx",
}

MAX_ENTRIES_PER_DIR = 60
DEFAULT_TOKEN_BUDGET = 12_000


def default_budget(level: int) -> int:
    """A backstop cap for a survey at `level`.

    Deeper surveys open more files and show more of each, so the ceiling rises
    with the level. This is not a per-file allowance: static outlines size
    themselves from the file.
    """
    from . import MAX_LEVEL

    # Static outlines are not token-budgeted, so this is a backstop against a
    # pathological tree rather than a per-file allowance: deeper surveys open
    # more files and each file shows more, so allow proportionally more.
    if level - 1 >= MAX_LEVEL:    # files rendered as full text
        return 200_000
    return DEFAULT_TOKEN_BUDGET * max(1, level)


class _Budget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self.exhausted = False

    def spend(self, text: str) -> bool:
        self.used += tokens.count(text)
        if self.used >= self.limit:
            self.exhausted = True
            return False
        return True


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def _entries(d: Path) -> tuple[list[Path], list[Path]]:
    dirs, files = [], []
    try:
        for e in sorted(d.iterdir(), key=lambda x: x.name.lower()):
            if e.is_symlink() or getattr(e, "is_junction", lambda: False)():
                continue
            if e.name in SKIP_DIRS or e.name.startswith("~$"):
                continue
            if e.is_dir():
                if not e.name.startswith("."):
                    dirs.append(e)
            elif e.is_file():
                if e.suffix.lower() in SKIP_SUFFIXES:
                    continue
                if e.name.startswith(".") and e.name not in (".env.example", ".gitignore"):
                    continue
                files.append(e)
    except PermissionError:
        pass
    return dirs, files


def overview(
    root: Path,
    level: int,
    *,
    budget: _Budget | None = None,
    depth: int = 0,
    token_budget: int = 0,
) -> str:
    """Render a folder overview at `level`. See the module docstring."""
    if budget is None:
        budget = _Budget(token_budget or default_budget(level))

    pad = "  " * depth
    dirs, files = _entries(root)
    out: list[str] = []

    if depth == 0:
        out.append(f"=== folder overview: {root}  ·  level={level} ===")

    total_size = 0
    for f in files:
        try:
            total_size += f.stat().st_size
        except OSError:
            pass
    out.append(f"{pad}{root.name or str(root)}/   "
               f"[{len(dirs)} dirs, {len(files)} files, {_human(total_size)}]")

    if level < 0:
        return "\n".join(out)

    # ---- level 0: a plain listing, nothing opened ------------------------
    if level == 0:
        for d in dirs[:MAX_ENTRIES_PER_DIR]:
            out.append(f"{pad}  {d.name}/")
        for f in files[:MAX_ENTRIES_PER_DIR]:
            try:
                sz = _human(f.stat().st_size)
            except OSError:
                sz = "?"
            out.append(f"{pad}  {f.name}  ({sz})")
        extra = (len(dirs) + len(files)) - min(len(dirs), MAX_ENTRIES_PER_DIR) \
                - min(len(files), MAX_ENTRIES_PER_DIR)
        if extra > 0:
            out.append(f"{pad}  … {extra} more entries")
        return "\n".join(out)

    # ---- level >= 1: open each file at level-1 ---------------------------
    file_level = level - 1
    for f in files[:MAX_ENTRIES_PER_DIR]:
        if budget.exhausted:
            break
        kind = detect_kind(f)
        try:
            sz = _human(f.stat().st_size)
        except OSError:
            sz = "?"
        header = f"{pad}  ── {f.name}  [{kind}, {sz}]"
        out.append(header)
        budget.spend(header)
        try:
            # No per-file banner: the `── name [kind, size]` line above already
            # says what this is, and at low levels the banner would otherwise
            # consume most of the file's budget.
            body = outline(f, file_level, kind, include_header=False)
        except Exception as e:
            body = f"[could not outline: {type(e).__name__}: {e}]"
        body = body.strip()
        if not body:
            continue
        indented = "\n".join(f"{pad}     {ln}" for ln in body.splitlines())
        if not budget.spend(indented):
            out.append(f"{pad}     [survey token budget reached — raise "
                       f"token_budget or narrow the path]")
            break
        out.append(indented)

    if len(files) > MAX_ENTRIES_PER_DIR:
        out.append(f"{pad}  … {len(files) - MAX_ENTRIES_PER_DIR} more files not shown")

    # ---- subfolders: level - 2 ------------------------------------------
    sub_level = level - 2
    for d in dirs[:MAX_ENTRIES_PER_DIR]:
        if budget.exhausted:
            out.append(f"{pad}  … remaining subfolders omitted (token budget)")
            break
        out.append("")
        out.append(overview(d, sub_level, budget=budget, depth=depth + 1))

    if depth == 0:
        out.append("")
        out.append(f"[overview used ~{budget.used:,} tokens"
                   + (f" of a {budget.limit:,} budget — TRUNCATED]" if budget.exhausted
                      else f" of a {budget.limit:,} budget]"))
    return "\n".join(out)[:MAX_OUTLINE_CHARS]
