"""LLM-generated document summaries.

The static outliner reports structure that is literally in the file. This one
reports *meaning*, which structure cannot give you: what the document argues,
what the module is for, what the data appears to measure.

The two compose. For a long file we send the model the static outline plus a
bounded slice of the body, rather than the whole thing — the outline is a far
better index than the first 8k characters would be, and it keeps the summariser
call cheap even for a 400-page PDF.

**Here `level` is a token budget**: 100 / 300 / 1,000 / 3,000 / 10,000 /
uncapped. Static outlines take a *structural* level instead, because a
deterministic outline costs nothing and has no length to ration -- whereas a
model will write as much as you let it, and every token is billed. The budget
sets both the instruction ("at most N tokens") and the API `max_tokens`, so an
over-eager model is cut off by the provider rather than by your context window.
"""

from __future__ import annotations

from pathlib import Path

from .. import tokens
from . import detect_kind, extract_text, outline

# level -> shape of the answer. The length comes from readers.LLM_LEVEL_TOKENS.
_INSTRUCTIONS = {
    0: "ONE sentence stating what this file is and what it is for. Nothing else.",
    1: "A short paragraph: the purpose of this file, how it is organised, and "
       "what someone would use it for.",
    2: "Structured bullets under short bold headings: what it contains, its main "
       "components or sections, the key facts a reader must know, and anything "
       "surprising or non-obvious.",
    3: "A section-by-section walkthrough. For each major part say what it does "
       "and why it exists. Include concrete specifics: names, numbers, "
       "parameters, dependencies.",
    4: "An exhaustive summary. Cover every section. Preserve all specifics: "
       "identifiers, signatures, numbers, units, file paths, versions. Someone "
       "must be able to work from this without opening the file.",
    5: "A complete account, at whatever length the document requires. Omit "
       "nothing of substance. Preserve every identifier, number, path and "
       "quoted string exactly.",
}

#: Ceiling for level 5 ("full detail") -- unbounded in intent, but a request
#: still needs a finite max_tokens.
UNCAPPED_OUTPUT_TOKENS = 16_000

_SYS = """\
You summarise a document for an AI agent that will act on your summary without
reading the original. Accuracy beats elegance.

RULES
- Quote identifiers, paths, numbers, units, versions and error strings EXACTLY.
- Never invent. If the excerpt is truncated, say what you could not see rather
  than guessing at it.
- No preamble, no "This document...", no meta-commentary about summarising.
- Length: {length}

TASK
{instruction}"""

MAX_INPUT_TOKENS = 60_000


def _length_spec(level: int) -> tuple[str, int]:
    """(instruction text, max_tokens) for a level."""
    from . import LLM_LEVEL_TOKENS

    budget = LLM_LEVEL_TOKENS[level]
    if budget is None:
        return ("as long as the material genuinely needs. Do not pad.",
                UNCAPPED_OUTPUT_TOKENS)
    # ~0.75 words per token is a good rule of thumb for English prose.
    words = int(budget * 0.75)
    return (f"at most {budget} tokens (roughly {words} words). This is a hard "
            f"limit -- being cut off mid-sentence is worse than being brief.",
            budget + 200)


def build_prompt(path: Path, level: int, kind: str | None = None) -> tuple[str, str, dict]:
    """Return (system, user, meta). Separated out so it is testable offline."""
    from . import MAX_LEVEL

    level = max(0, min(MAX_LEVEL, int(level)))
    length, max_out = _length_spec(level)
    instruction = _INSTRUCTIONS[level]
    kind = kind or detect_kind(path)

    # The outline is the model's index into the document. Its level is a
    # *structural* level, unrelated to this call's token budget: ask for one
    # step deeper than the answer, but never level 5, which would be the whole
    # file twice over.
    try:
        struct = outline(path, min(level + 1, MAX_LEVEL - 1), kind)
    except Exception as e:
        struct = f"[outline failed: {e}]"

    try:
        body = extract_text(path, kind)
    except Exception as e:
        body = f"[text extraction failed: {e}]"

    struct_tokens = tokens.count(struct)
    body_budget = max(2_000, MAX_INPUT_TOKENS - struct_tokens - 500)
    body_full = tokens.count(body)
    if body_full > body_budget:
        body = tokens.truncate_to_tokens(body, body_budget)
        truncated = True
    else:
        truncated = False

    user = (
        f"# File: {path.name}  (kind: {kind})\n\n"
        f"## Structural outline\n{struct}\n\n"
        f"## Content{' (truncated — the middle was elided)' if truncated else ''}\n{body}"
    )
    return (
        _SYS.format(length=length, instruction=instruction),
        user,
        {"kind": kind, "level": level, "truncated": truncated,
         "body_tokens": body_full, "sent_tokens": tokens.count(user),
         "max_output_tokens": max_out},
    )


async def summarize(path: Path, level: int, model: str, complete_fn,
                    kind: str | None = None) -> tuple[str, dict]:
    sys_prompt, user, meta = build_prompt(path, level, kind)
    text = await complete_fn(
        model=model,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user}],
        max_tokens=meta["max_output_tokens"],
        temperature=0.0,
    )
    return (text or "").strip(), meta
