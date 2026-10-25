"""Outlines for structured data: JSON, JSONL, YAML, TOML, INI, XML.

The level maps onto nesting depth: level 0 shows the top-level keys, each step
descends one more layer. Values are reported as types and shapes rather than
contents, so a 40 MB JSON file still outlines in a few hundred tokens.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MAX_KEYS_PER_LEVEL = 60


def _type_of(v) -> str:
    if isinstance(v, dict):
        return f"object({len(v)} keys)"
    if isinstance(v, list):
        if not v:
            return "array(0)"
        inner = {type(x).__name__ for x in v[:50]}
        return f"array({len(v)} x {'|'.join(sorted(inner))})"
    if isinstance(v, str):
        return f"string(len {len(v)})"
    if isinstance(v, bool):
        return "bool"
    if v is None:
        return "null"
    return type(v).__name__


def _sample(v, level: int) -> str:
    if level < 2:
        return ""
    if isinstance(v, (dict, list)):
        return ""
    s = repr(v)
    return f" = {s[:80]}" + ("…" if len(s) > 80 else "")


def _walk(node, depth: int, max_depth: int, out: list[str], prefix: str = "", level: int = 0) -> None:
    pad = "  " * depth
    if isinstance(node, dict):
        keys = list(node.keys())
        for k in keys[:MAX_KEYS_PER_LEVEL]:
            v = node[k]
            out.append(f"{pad}{k}: {_type_of(v)}{_sample(v, level)}")
            if depth + 1 <= max_depth and isinstance(v, (dict, list)):
                _walk(v, depth + 1, max_depth, out, prefix, level)
        if len(keys) > MAX_KEYS_PER_LEVEL:
            out.append(f"{pad}… {len(keys) - MAX_KEYS_PER_LEVEL} more keys")
    elif isinstance(node, list):
        if not node:
            return
        # Describe the shape of the first element; lists are usually homogeneous.
        out.append(f"{pad}[0]: {_type_of(node[0])}")
        if depth + 1 <= max_depth and isinstance(node[0], (dict, list)):
            _walk(node[0], depth + 1, max_depth, out, prefix, level)
        if len(node) > 1:
            out.append(f"{pad}… {len(node) - 1} more element(s), same shape assumed")


def _outline_obj(data, level: int) -> str:
    max_depth = level + 1
    out = [f"root: {_type_of(data)}", f"--- structure (depth ≤ {max_depth}) ---"]
    _walk(data, 0, max_depth, out, level=level)
    return "\n".join(out)


def outline(path: Path, level: int, kind: str) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")

    if kind == "json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return f"[invalid JSON: {e}]"
        return _outline_obj(data, level)

    if kind == "jsonl":
        lines = [l for l in text.splitlines() if l.strip()]
        out = [f"{len(lines):,} JSON lines"]
        if lines:
            try:
                first = json.loads(lines[0])
                out.append("--- shape of the first record ---")
                out.append(_outline_obj(first, level))
            except json.JSONDecodeError as e:
                out.append(f"[first line is not valid JSON: {e}]")
        return "\n".join(out)

    if kind == "yaml":
        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError:
            return "[`PyYAML` is not installed — run `pip install pyyaml`]"
        except Exception as e:
            return f"[invalid YAML: {e}]"
        return _outline_obj(data, level)

    if kind == "toml":
        try:
            import tomllib

            data = tomllib.loads(text)
        except Exception as e:
            return f"[could not parse TOML: {e}]"
        return _outline_obj(data, level)

    if kind == "ini":
        import configparser

        cp = configparser.ConfigParser()
        try:
            cp.read_string(text)
        except Exception as e:
            return f"[could not parse INI: {e}]"
        out = [f"{len(cp.sections())} sections"]
        for s in cp.sections():
            out.append(f"[{s}]")
            if level >= 1:
                for k in cp[s]:
                    v = cp[s][k]
                    out.append(f"  {k}" + (f" = {v[:60]}" if level >= 2 else ""))
        return "\n".join(out)

    if kind == "xml":
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(text)
        except Exception as e:
            return f"[could not parse XML: {e}]"
        out = [f"root: <{root.tag}>  ({len(list(root))} children)"]

        def walk(el, depth: int):
            if depth > level + 1:
                return
            seen: dict[str, int] = {}
            for child in el:
                seen[child.tag] = seen.get(child.tag, 0) + 1
            for tag, n in list(seen.items())[:MAX_KEYS_PER_LEVEL]:
                out.append("  " * depth + f"<{tag}> x{n}")
                first = el.find(tag)
                if first is not None:
                    if level >= 2 and first.attrib:
                        out.append("  " * (depth + 1) + "attrs: " + ", ".join(first.attrib))
                    walk(first, depth + 1)

        walk(root, 1)
        return "\n".join(out)

    return "[unsupported structured format]"
