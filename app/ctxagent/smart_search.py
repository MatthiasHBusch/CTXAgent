"""smart_search: a small model walks a folder tree and returns the lines that matter.

The agent states a goal. A small model (Claude Haiku 5.5 by default) then does what
the agent would otherwise do in many sequential tool calls:

  1. Folder step. It sees one folder as `summarize_folder` shows it and names the files
     to read and the subfolders to look into. The view is level 2 (each file with its
     docstring opening and function names, each subfolder with the names of its entries)
     while that fits `FOLDER_VIEW_TOKENS`, else level 1 (class names; subfolders by name
     and size). The walk starts at the given folder and follows only the folders named.
  2. File step. Each named file is sent with line numbers; the model returns the line
     ranges that matter for the goal.
  3. The ranges are printed with their location, most relevant first, up to an output
     limit; what does not fit is listed as locations only.

These are plain completion calls with a JSON schema, not an agent loop: no call sees
another call's answer, so the calls of one step run in parallel.

Request size. Claude Haiku 5.5 costs five times as much per token once a prompt has
more than 100,000 tokens. A file that does not fit `REQUEST_TOKENS` is cut into
portions of that size at line boundaries, each sent on its own.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config, readers, tokens
from .readers import folder as foldermod

#: Default model, as a direct Anthropic API id; `default_model` picks the route.
DEFAULT_MODEL = "claude-haiku-5-5"
#: Prompt tokens above which Claude Haiku 5.5 is billed at five times the rate.
PRICE_STEP_TOKENS = 100_000
#: Largest request in local (tiktoken) tokens. Haiku 5.5's own tokenizer counts more
#: tokens for the same text (1.4 to 1.9 times, SMART_SEARCH.md), so this stays well below
#: the price step.
REQUEST_TOKENS = 45_000
#: Room in each request for the system prompt, the goal and the outline of a split file.
OVERHEAD_TOKENS = 4_000

#: A folder is shown at level 2 while that view has at most this many local tokens.
FOLDER_VIEW_TOKENS = 30_000
#: Reasoning effort of the model calls.
EFFORT = "high"

MAX_FOLDERS = 24           # folder steps per search
MAX_FILES = 20             # files read per search
MAX_PORTIONS = 6           # portions read of one file
MAX_FILE_BYTES = 3_000_000
PARALLEL_CALLS = 6
DEFAULT_OUTPUT_TOKENS = 12_000
MERGE_GAP = 2              # ranges this many lines apart are printed as one

RELEVANCE = ("high", "medium")     # of a file, judged from the folder view
KINDS = ("core", "support")        # of a line range, judged from the file

_FOLDER_SYS = """\
You steer a file search for an AI agent that works on the user's files. The agent sent a
search goal. After the search it sees only the line ranges that were found, and then starts
its work (usually changing or extending code).

You see ONE folder: its files, each with a short outline, and its subfolders (with the names
of their entries, or in a large folder only their name and size).
Name the files to read and the subfolders to look into next.

Facts
- Each file you name is read in full by another request. Each subfolder you name costs
  another request like this one, and its own files and subfolders after that.
- The agent needs the code the goal is about, and what it must also read to work on it (a
  caller, a setting, a test that pins the behaviour).
- A file the search misses costs the agent a search of its own; a file read without a hit costs
  one cheap request. In doubt, a file is worth naming as "medium".
- Where many files follow one pattern (a module per tool, per format, per test), the file that
  defines the pattern and one example serve the agent better than all of them.
- Documents and notes that describe the code are worth reading when the goal asks how
  something is meant to work, rarely when it asks where the code is.
- A goal may ask for something this project does not have. Then nothing here fits, and empty
  lists tell the agent so.
- relevance: "high" = the goal is probably about this file. "medium" = the agent will probably
  need it as well, or might.
- Names are given as listed. A file or folder further down that the agent's instructions name
  can be given by its path, and is then read or opened directly."""

_FILE_SYS = """\
You pick the lines of a file that an AI agent needs for a search goal. The agent sees only
the line ranges you return, printed with their line numbers, and then starts its work
(usually changing or extending code).

Facts
- The file was picked from its name and outline alone, so many files turn out to hold nothing
  for the goal. An empty list is the right answer for those, and for a goal that asks for
  something this project does not have. The agent is told which files were read without a
  hit; a range that only shows that something is absent or unrelated adds nothing.
- A needed line that is missing costs the agent more than a few lines too many: it has to
  search for it again. Lines that are probably needed are worth including as "support".
- kind "core": lines the goal is about: the code to change, the definition asked for, the place
  where the behaviour is decided. kind "support": lines the agent must also read to work on the
  core correctly: a caller it has to adapt, a setting or constant the core reads, a test that
  pins the behaviour. Lines that merely mention the topic are neither.
- A range is a run of consecutive lines that belong together: a function, a class, a block, a
  paragraph, a table. The whole definition when the agent needs all of it; for a long body that
  is mostly unrelated, its signature and the part that matters as separate ranges.
- note: what the lines are, in at most 12 words.
- Line numbers are the numbers printed at the start of each line."""

FOLDER_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"},
                           "relevance": {"type": "string", "enum": list(RELEVANCE)}},
            "required": ["name", "relevance"], "additionalProperties": False}},
        "folders": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["files", "folders"], "additionalProperties": False,
}
FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "ranges": {"type": "array", "items": {
            "type": "object",
            "properties": {"start": {"type": "integer"}, "end": {"type": "integer"},
                           "kind": {"type": "string", "enum": list(KINDS)},
                           "note": {"type": "string"}},
            "required": ["start", "end", "kind", "note"], "additionalProperties": False}},
    },
    "required": ["ranges"], "additionalProperties": False,
}


def default_model(chat_model: str = "") -> str:
    """Claude Haiku 5.5 on the route the chat uses, or on the other one when only that
    route has a key."""
    direct = DEFAULT_MODEL
    routed = config.provider_counterpart(direct) or direct
    first, second = (direct, routed) if (chat_model or "").startswith("claude-") else (routed, direct)
    if config.key_problem(first) and not config.key_problem(second):
        return second
    return first


@dataclass
class Range:
    start: int
    end: int
    kind: str = "support"
    note: str = ""


@dataclass
class FileHit:
    path: Path
    relevance: str
    total_lines: int = 0
    extracted: bool = False      # line numbers refer to the extracted text (read_document)
    partial: str = ""            # why the file was not read completely
    ranges: list[Range] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


@dataclass
class Report:
    folders_surveyed: list[Path] = field(default_factory=list)
    folders_left: list[Path] = field(default_factory=list)     # named, not surveyed (limit)
    files_left: list[Path] = field(default_factory=list)       # named, not read (limit)
    files: list[FileHit] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    calls: int = 0


# ---------------------------------------------------------------------------
# model calls
# ---------------------------------------------------------------------------

def parse_json(text: str) -> dict:
    """The JSON object in a model answer, with or without a code fence around it."""
    text = (text or "").strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced[1]
    try:
        value = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("answer is not a JSON object")
    return value


async def _ask(complete_fn, model: str, system: str, user: str, name: str, schema: dict,
               max_tokens: int, report: Report) -> dict:
    """One schema-bound completion; a second try when the answer does not parse."""
    last: Exception | None = None
    for _ in range(2):
        report.calls += 1
        text = await complete_fn(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            reasoning_effort=EFFORT,
            pricing_mode="standard",
            response_format={"type": "json_schema",
                             "json_schema": {"name": name, "strict": True, "schema": schema}},
        )
        try:
            return parse_json(text)
        except ValueError as exc:
            last = exc
    raise ValueError(f"no JSON answer ({last})")


def _task(goal: str, instructions: str) -> str:
    text = f"Search goal:\n{goal.strip()}\n"
    if (instructions or "").strip():
        text += f"\nInstructions from the agent:\n{instructions.strip()}\n"
    return text


# ---------------------------------------------------------------------------
# folder step
# ---------------------------------------------------------------------------

def folder_view(path: Path) -> tuple[str, dict[str, Path], dict[str, Path]]:
    """(text, subfolders by name, files by name) of one folder: the level-2 overview
    while it is small enough, else level 1, plus the names the overview leaves out of
    a very large folder."""
    dirs, files = foldermod._entries(path)
    budget = REQUEST_TOKENS - OVERHEAD_TOKENS
    text = foldermod.overview(path, 2, token_budget=FOLDER_VIEW_TOKENS + 1_000)
    if tokens.count(text) > FOLDER_VIEW_TOKENS or "TRUNCATED]" in text:
        text = foldermod.overview(path, 1, token_budget=budget)
    shown = foldermod.MAX_ENTRIES_PER_DIR
    rest = [f"  {d.name}/" for d in dirs[shown:]] + [f"  {f.name}" for f in files
                                                      if f"── {f.name}  [" not in text]
    if rest:
        text += "\n\nFurther entries of this folder, names only:\n" + "\n".join(rest)
    if tokens.count(text) > budget:
        text = tokens.truncate_to_tokens(text, budget)
    return text, {d.name: d for d in dirs}, {f.name: f for f in files}


def _entry(name, table: dict[str, Path]) -> Path | None:
    """The listed entry a model-given name means (exact, without path, any case)."""
    name = str(name or "").strip().strip("/\\")
    if name in table:
        return table[name]
    base = re.split(r"[/\\]", name)[-1]
    if base in table:
        return table[base]
    lowered = {key.lower(): value for key, value in table.items()}
    return lowered.get(base.lower())


def _below(name, path: Path, root: Path) -> Path | None:
    """A model-given path of several parts, below this folder or below the search root."""
    name = str(name or "").strip().strip("/\\")
    if not re.search(r"[/\\]", name):
        return None
    for base in (path, root):
        try:
            target = (base / name).resolve()
            target.relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        if target.exists() and target != path:
            return target
    return None


async def _survey(path: Path, root: Path, task: str, complete_fn, model: str, report: Report):
    """One folder step: ([(file, relevance)], [subfolder])."""
    text, dirs, files = folder_view(path)
    rel = path.relative_to(root).as_posix() if path != root else "."
    user = f"{task}\nFolder: {rel}  (search started at: {root})\n\n{text}"
    answer = await _ask(complete_fn, model, _FOLDER_SYS, user, "folder_choice", FOLDER_SCHEMA, 8_000, report)
    chosen_files = []
    for item in answer.get("files") or []:
        name = item.get("name") if isinstance(item, dict) else item
        target = _below(name, path, root)
        if target is None or not target.is_file():
            target = _entry(name, files)
        if target is not None:
            relevance = item.get("relevance") if isinstance(item, dict) else None
            chosen_files.append((target, relevance if relevance in RELEVANCE else "medium"))
    chosen_dirs = []
    for name in answer.get("folders") or []:
        target = _below(name, path, root)
        if target is None or not target.is_dir():
            target = _entry(name, dirs)
        if target is not None:
            chosen_dirs.append(target)
    return chosen_files, chosen_dirs


# ---------------------------------------------------------------------------
# file step
# ---------------------------------------------------------------------------

def load_lines(path: Path) -> tuple[list[str], bool, str]:
    """(lines, extracted, partial). Text files give their own lines; containers (pdf,
    docx, xlsx, ...) and very large files the extracted text, as read_document shows it."""
    kind = readers.detect_kind(path)
    size = path.stat().st_size
    if readers._is_binaryish(kind) or kind in ("notebook", "parquet") or size > MAX_FILE_BYTES:
        text = readers.extract_text(path, kind)
        partial = (f"first {readers.MAX_TEXT_CHARS:,} characters of the extracted text"
                   if len(text) >= readers.MAX_TEXT_CHARS else "")
        return text.splitlines(), True, partial
    return readers.read_text_file(path).splitlines(), False, ""


def numbered(lines: list[str], first: int) -> str:
    return "\n".join(f"{first + i}: {line}" for i, line in enumerate(lines))


def portions(lines: list[str], budget: int) -> list[tuple[int, int]]:
    """Cut `lines` into runs of at most `budget` tokens (numbered), as (first, last)
    1-based line numbers. A single line longer than the budget is its own portion."""
    out, start, used = [], 0, 0
    for i, line in enumerate(lines):
        cost = tokens.count(f"{i + 1}: {line}") + 1
        if used and used + cost > budget:
            out.append((start + 1, i))
            start, used = i, 0
        used += cost
    if lines:
        out.append((start + 1, len(lines)))
    return out


def clean_ranges(raw, first: int, last: int) -> list[Range]:
    """Valid ranges inside lines first..last of a model answer."""
    out = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            start, end = int(item.get("start")), int(item.get("end"))
        except (TypeError, ValueError):
            continue
        if start > end:
            start, end = end, start
        start, end = max(start, first), min(end, last)
        if start > end:
            continue
        kind = item.get("kind") if item.get("kind") in KINDS else "support"
        out.append(Range(start, end, kind, " ".join(str(item.get("note") or "").split())[:120]))
    return out


def merge_ranges(ranges: list[Range]) -> list[Range]:
    """Sort, and join ranges that overlap or lie within MERGE_GAP lines."""
    out: list[Range] = []
    for r in sorted(ranges, key=lambda r: (r.start, r.end)):
        if out and r.start <= out[-1].end + MERGE_GAP + 1:
            prev = out[-1]
            prev.end = max(prev.end, r.end)
            if r.kind == "core":
                prev.kind = "core"
            if r.note and r.note not in prev.note:
                prev.note = f"{prev.note}; {r.note}" if prev.note else r.note
        else:
            out.append(Range(r.start, r.end, r.kind, r.note))
    return out


async def _read(hit: FileHit, root: Path, task: str, complete_fn, model: str, report: Report,
                gate: asyncio.Semaphore) -> None:
    """The file step for one file: its relevant ranges into `hit`."""
    hit.lines, hit.extracted, hit.partial = await asyncio.to_thread(load_lines, hit.path)
    hit.total_lines = len(hit.lines)
    if not hit.lines:
        return
    parts = portions(hit.lines, REQUEST_TOKENS - OVERHEAD_TOKENS)
    outline = ""
    if len(parts) > 1:
        try:
            outline = readers.outline(hit.path, 1, include_header=False)
        except Exception:  # noqa: BLE001
            outline = ""
        outline = tokens.truncate_to_tokens(outline, 2_000)
        if len(parts) > MAX_PORTIONS:
            hit.partial = (hit.partial + "; " if hit.partial else "") + \
                f"lines 1-{parts[MAX_PORTIONS - 1][1]} of {hit.total_lines}"
            parts = parts[:MAX_PORTIONS]
    rel = hit.path.relative_to(root).as_posix()

    async def one(index: int, first: int, last: int) -> list[Range]:
        head = f"{task}\nFile: {rel}  ({hit.total_lines} lines"
        if len(parts) > 1:
            head += (f"; this is part {index + 1} of {len(parts)}, lines {first}-{last}. Answer for "
                     f"these lines only)\n\nOutline of the whole file:\n{outline}")
        else:
            head += ")"
        user = f"{head}\n\nLines:\n{numbered(hit.lines[first - 1:last], first)}"
        async with gate:
            answer = await _ask(complete_fn, model, _FILE_SYS, user, "line_ranges", FILE_SCHEMA, 12_000, report)
        return clean_ranges(answer.get("ranges"), first, last)

    results = await asyncio.gather(*(one(i, a, b) for i, (a, b) in enumerate(parts)),
                                   return_exceptions=True)
    found: list[Range] = []
    for (first, last), result in zip(parts, results):
        if isinstance(result, Exception):
            report.errors.append(f"{rel} lines {first}-{last}: {type(result).__name__}: {result}")
        else:
            found.extend(result)
    hit.ranges = merge_ranges(found)


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------

async def search(root: Path, goal: str, instructions: str, complete_fn, model: str, *,
                 max_folders: int = MAX_FOLDERS, max_files: int = MAX_FILES) -> Report:
    """Walk the tree under `root` and collect the relevant ranges."""
    report = Report()
    task = _task(goal, instructions)
    gate = asyncio.Semaphore(PARALLEL_CALLS)
    pending: list[Path] = [root]
    seen: set[Path] = {root}
    candidates: dict[Path, str] = {}

    while pending and len(report.folders_surveyed) < max_folders:
        room = max_folders - len(report.folders_surveyed)
        batch, pending = pending[:room], pending[room:]

        async def step(path: Path):
            async with gate:
                return await _survey(path, root, task, complete_fn, model, report)

        results = await asyncio.gather(*(step(p) for p in batch), return_exceptions=True)
        for path, result in zip(batch, results):
            report.folders_surveyed.append(path)
            if isinstance(result, Exception):
                report.errors.append(f"{path}: {type(result).__name__}: {result}")
                continue
            files, dirs = result
            for target, relevance in files:
                if candidates.get(target) != "high":
                    candidates[target] = relevance
            for d in dirs:
                if d not in seen:
                    seen.add(d)
                    pending.append(d)
    report.folders_left = pending

    ordered = sorted(candidates, key=lambda p: (RELEVANCE.index(candidates[p]), str(p).lower()))
    report.files_left = ordered[max_files:]
    hits = [FileHit(path, candidates[path]) for path in ordered[:max_files]]

    async def read(hit: FileHit):
        try:
            await _read(hit, root, task, complete_fn, model, report, gate)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"{hit.path}: {type(exc).__name__}: {exc}")

    await asyncio.gather(*(read(hit) for hit in hits))
    report.files = hits
    return report


def render(report: Report, root: Path, goal: str, model: str, max_tokens: int = DEFAULT_OUTPUT_TOKENS) -> str:
    """The tool result: ranges with their lines, most relevant first, then what was left out."""
    def rel(path: Path) -> str:
        return path.relative_to(root).as_posix() if path != root else "."

    with_hits = [hit for hit in report.files if hit.ranges]
    head = [f"smart_search in {root}",
            f"goal: {' '.join(goal.split())}",
            f"[{model}: {len(report.folders_surveyed)} folders surveyed, {len(report.files)} files read, "
            f"{sum(len(h.ranges) for h in with_hits)} ranges in {len(with_hits)} files, {report.calls} model calls]"]

    # Which ranges get their lines: core before support, in file order, while the limit holds.
    order = sorted(((KINDS.index(r.kind), fi, ri) for fi, hit in enumerate(with_hits)
                    for ri, r in enumerate(hit.ranges)))
    used = tokens.count("\n".join(head))
    printed: set[tuple[int, int]] = set()
    bodies: dict[tuple[int, int], str] = {}
    for _, fi, ri in order:
        hit, r = with_hits[fi], with_hits[fi].ranges[ri]
        body = "\n".join(f"{n:6d}\t{hit.lines[n - 1]}" for n in range(r.start, r.end + 1))
        cost = tokens.count(body) + 25
        if used + cost > max_tokens:
            continue
        used += cost
        printed.add((fi, ri))
        bodies[(fi, ri)] = body

    out = list(head)
    listed: list[str] = []
    for fi, hit in enumerate(with_hits):
        shown = [ri for ri in range(len(hit.ranges)) if (fi, ri) in printed]
        for ri, r in enumerate(hit.ranges):
            if ri not in shown:
                listed.append(f"  {rel(hit.path)}:L{r.start}-{r.end}  [{r.kind}] {r.note}")
        if not shown:
            continue
        notes = []
        if hit.extracted:
            notes.append("line numbers of the extracted text, as read_document shows it")
        if hit.partial:
            notes.append(f"searched only: {hit.partial}")
        out.append("")
        out.append(f"=== {rel(hit.path)}  ({hit.total_lines} lines" + ("; " + "; ".join(notes) if notes else "") + ")")
        for ri in shown:
            r = hit.ranges[ri]
            out.append(f"--- L{r.start}-{r.end}  [{r.kind}] {r.note}")
            out.append(bodies[(fi, ri)])

    if not with_hits:
        out += ["", "No relevant lines found."]
    if listed:
        out += ["", f"Found but not printed (output limit of {max_tokens:,} tokens):"] + listed
    empty = [rel(hit.path) for hit in report.files if not hit.ranges]
    if empty:
        out += ["", "Read, nothing relevant: " + ", ".join(empty)]
    if report.files_left:
        out += ["", f"Named but not read (limit of {len(report.files)} files): "
                + ", ".join(rel(p) for p in report.files_left)]
    if report.folders_left:
        out += ["", f"Named but not looked into (limit of {len(report.folders_surveyed)} folders): "
                + ", ".join(rel(p) + "/" for p in report.folders_left)]
    if report.errors:
        out += ["", "Failed steps:"] + [f"  {e}" for e in report.errors[:10]]
    return "\n".join(out)
