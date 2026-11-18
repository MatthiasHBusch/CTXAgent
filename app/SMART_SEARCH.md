# smart_search

`smart_search(goal, instructions="", path=".", max_tokens=12000)` hands the search that opens a
task to a small model. It replaces the run of `list_dir`, `grep_search` and `read_file` calls an
agent makes before it can start: one tool call returns the relevant line ranges with their file
and line numbers.

Code: `ctxagent/smart_search.py`, tool function `tools.t_smart_search`.
Tests: `tests/test_smart_search.py` (scripted model), `tests/smart_search_live.py` (real model, paid).

## How it works

Plain completion calls with a JSON schema. No call sees another call's answer, so the calls of
one step run in parallel (6 at a time).

1. **Folder step.** The model sees one folder as `summarize_folder` shows it and names the files
   to read (`high` or `medium`) and the subfolders to open. The view is level 2 (each file with
   its docstring opening and function names, each subfolder with the names of its entries) while
   that is at most 30,000 local tokens, else level 1. A path further down, for example one the
   agent's instructions name, is taken directly.
2. **File step.** Each named file is sent with line numbers. The model returns line ranges, each
   `core` (what the goal is about) or `support` (what the agent must also read), with a note.
3. **Result.** Ranges are printed with path and line numbers in `read_file`'s format, `core`
   first, up to `max_tokens`. What does not fit is listed as locations. The result also names
   the files read without a hit, and what a limit left out.

The walk starts at `path` and follows only the folders the model names.

## Model and cost

Claude Haiku 5.5 at reasoning effort `high`, on the route of the chat (`claude-haiku-5-5` on the
Anthropic API, `anthropic/claude-haiku-5.5` on OpenRouter), or on the other route when only
that one has a key. Usage is booked like summary calls.

Haiku 5.5 costs $0.10 / $0.50 per million tokens (input / output) for prompts of up to 100,000
tokens and $0.50 / $2.50 above. Requests are therefore capped at `REQUEST_TOKENS` = 45,000 local
(tiktoken) tokens. A larger file is cut into portions of that size at line boundaries; each
portion is sent with the file's outline. Measured on this repository, Haiku 5.5 counts 1.43 to
1.74 times the local tokens per search, and 1.94 times for `web/app.js` alone, so 45,000 local
tokens stay below 90,000 of its own.

Limits per search: 24 folders, 20 files, 6 portions per file.

## Live results (2026-11-17, OpenRouter, this repository at a4c5fb9 plus smart_search)

Seven goals from `tests/smart_search_live.py`, each run twice at effort `high`:

| | |
|---|---|
| Expected files found | all, in every run with something to find |
| Goals with nothing to find | no lines returned, in 4 of 4 runs |
| Cost per search | $0.0014 to $0.037, mean $0.011 |
| Time per search | 4 to 31 s |
| Folders opened | 1 to 4 |
| Files read | 0 to 16; about three in ten without a hit |
| Lines returned | 7 to 22 % of the lines read; 35 to 39 % where the goal names one file |

At effort `low` the same goals cost 17 % less and took 40 % less time, but one
nothing-to-find run returned 843 lines of background.

Two goals on the larger development checkout (benchmark and paper folders) cost $0.021 and
$0.027, opened 4 and 8 folders and read 11 and 20 files. One of them went into third-party
repositories below `bench/external`; `instructions` can name folders to skip.

The prompts state costs as facts and lean towards reading a file too many rather than missing
one (the user's choice): a missed file costs the agent a search of its own, a file read without
a hit one cheap request.

## Not covered yet

- No setting for the model or the effort; both are constants in `smart_search.py`.
- Not tried on repositories with deep trees or very large files, or with other models.
- Ranges in PDFs and Office files use the line numbers of the extracted text, as `read_document`
  shows it. Checked offline only.

## From the search to the change

smart_search prints the file's own lines with their numbers, so edits can be written from its
result without another read. Tools that go with it:

- `plan(plan)` shows the user a plan in the chat as a message of its own and returns nothing.
- `edit_lines(path, start_line, end_line, start_text, end_text, new_text)` replaces a line range.
  `start_text` and `end_text` are the exact text of the range's first and last line and must not
  be blank; the edit is refused when they do not match, so a shifted line number cannot hit
  other lines.
- `prepare_edit_plan` takes the same line-range edits next to exact-string edits. All line
  numbers of a plan refer to the file as it is, so several ranges of one file do not shift
  each other. `apply_edit_plan` writes the plan.
- `run_script(script, language, args, cwd, timeout, interpreter)` writes a script to a file and
  runs it; the source never passes through a shell. Consecutive calls of one response run at
  the same time (up to 4).

A small change fits three responses:

1. `smart_search`
2. `plan` and `prepare_edit_plan` (the diff comes back)
3. `apply_edit_plan` and the test, for example `run_script`

Whether a model follows this path on its own has not been measured in a chat.
