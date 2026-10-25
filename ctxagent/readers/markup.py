"""Outlines for prose formats: markdown, reStructuredText, HTML, LaTeX, plain text.

For heading-structured formats the level maps directly onto heading depth:
level 0 shows the title only, level 1 adds H2, level 2 adds H3, and so on. That
is exactly the "headlines only, level says how far" behaviour asked for, and it
means an outline of a long document is almost free.
"""

from __future__ import annotations

import re
import textwrap

_FENCE = re.compile(r"^\s*(```|~~~)")
_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT = re.compile(r"^(=+|-+)\s*$")


def outline_markdown(src: str, level: int) -> str:
    lines = src.splitlines()
    max_depth = level + 1          # level 0 -> H1 only
    out: list[str] = []
    headings: list[tuple[int, str, int]] = []

    in_fence = False
    n_code = n_table = n_img = n_link = 0
    for i, ln in enumerate(lines):
        if _FENCE.match(ln):
            in_fence = not in_fence
            if in_fence:
                n_code += 1
            continue
        if in_fence:
            continue
        m = _ATX.match(ln)
        if m:
            headings.append((len(m.group(1)), m.group(2).strip(), i + 1))
            continue
        if _SETEXT.match(ln) and i > 0 and lines[i - 1].strip():
            headings.append((1 if ln.strip()[0] == "=" else 2, lines[i - 1].strip(), i))
            continue
        if ln.lstrip().startswith("|") and ln.count("|") >= 2:
            n_table += 1
        n_img += ln.count("![")
        n_link += len(re.findall(r"(?<!!)\[[^\]]+\]\(", ln))

    if not headings:
        out.append("(no headings found)")
        out.append("--- first lines ---")
        out += lines[: 8 * max(1, level + 1)]
        return "\n".join(out)

    title = next((h for h in headings if h[0] == 1), headings[0])
    out.append(f"# {title[1]}")
    out.append("")

    shown = [h for h in headings if h[0] <= max_depth and h is not title]
    if shown:
        out.append(f"--- headings (depth ≤ {max_depth}) ---")
        for depth, text, ln_no in shown:
            indent = "  " * (depth - 1)
            suffix = f"   (line {ln_no})" if level >= 2 else ""
            out.append(f"{indent}{'#' * depth} {text}{suffix}")
    elif max_depth == 1:
        out.append(f"(document has {len(headings)} headings; "
                   f"raise the level to see below H1)")

    deeper = len([h for h in headings if h[0] > max_depth])
    if deeper:
        out.append(f"… {deeper} deeper heading(s) hidden at this level")

    out.append("")
    out.append(f"[{len(headings)} headings, {n_code} code blocks, "
               f"{n_table} table rows, {n_img} images, {n_link} links]")
    return "\n".join(out)


_RST_UNDERLINE = re.compile(r"^([=\-`:'\"~^_*+#<>])\1{2,}\s*$")


def outline_rst(src: str, level: int) -> str:
    lines = src.splitlines()
    seen: list[str] = []
    headings: list[tuple[int, str, int]] = []
    for i, ln in enumerate(lines):
        if _RST_UNDERLINE.match(ln) and i > 0 and lines[i - 1].strip():
            ch = ln.strip()[0]
            if ch not in seen:
                seen.append(ch)
            headings.append((seen.index(ch) + 1, lines[i - 1].strip(), i))
    max_depth = level + 1
    out = [f"--- headings (depth ≤ {max_depth}) ---"]
    for d, t, n in headings:
        if d <= max_depth:
            out.append("  " * (d - 1) + f"{t}   (line {n})")
    out.append("")
    out.append(f"[{len(headings)} headings]")
    return "\n".join(out) if headings else outline_text(src, level)


def outline_html(src: str, level: int) -> str:
    max_depth = level + 1
    title = re.search(r"<title[^>]*>(.*?)</title>", src, re.I | re.S)
    out = []
    if title:
        out.append(f"title: {' '.join(title.group(1).split())}")
    heads = re.findall(r"<h([1-6])[^>]*>(.*?)</h\1>", src, re.I | re.S)
    out.append(f"--- headings (depth ≤ {max_depth}) ---")
    shown = 0
    for lvl, text in heads:
        d = int(lvl)
        if d <= max_depth:
            out.append("  " * (d - 1) + f"h{d}: {' '.join(re.sub(r'<[^>]+>', ' ', text).split())[:120]}")
            shown += 1
    if not shown:
        out.append("(none at this depth)")
    n_links = len(re.findall(r"<a\s", src, re.I))
    n_imgs = len(re.findall(r"<img\s", src, re.I))
    out.append("")
    out.append(f"[{len(heads)} headings, {n_links} links, {n_imgs} images]")
    return "\n".join(out)


def html_to_text(src: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(src, "html.parser")
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        return re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n")).strip()
    except Exception:
        s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", src)
        s = re.sub(r"(?s)<[^>]+>", " ", s)
        return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", s)).strip()


_TEX_SECT = re.compile(r"^\s*\\(part|chapter|section|subsection|subsubsection|paragraph)\*?\{(.+?)\}")
_TEX_ORDER = ["part", "chapter", "section", "subsection", "subsubsection", "paragraph"]


def outline_latex(src: str, level: int) -> str:
    max_depth = level + 2  # \section is depth 3 in the canonical order
    out = []
    title = re.search(r"\\title\{(.+?)\}", src, re.S)
    if title:
        out.append(f"title: {' '.join(title.group(1).split())}")
    out.append(f"--- sections ---")
    n = 0
    for ln in src.splitlines():
        m = _TEX_SECT.match(ln)
        if not m:
            continue
        d = _TEX_ORDER.index(m.group(1)) + 1
        n += 1
        if d <= max_depth:
            out.append("  " * (d - 1) + f"\\{m.group(1)}: {m.group(2)}")
    out.append("")
    out.append(f"[{n} sectioning commands]")
    return "\n".join(out)


def outline_text(src: str, level: int) -> str:
    lines = src.splitlines()
    n_show = min(len(lines), 6 * (4 ** min(level, 4)))
    words = len(src.split())
    out = [f"--- first {n_show} of {len(lines)} lines ---"]
    out += lines[:n_show]
    if n_show < len(lines):
        out.append(f"… {len(lines) - n_show} more lines")
    out.append("")
    out.append(f"[{len(lines)} lines, {words} words, {len(src)} chars]")
    return "\n".join(out)
