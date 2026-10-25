"""Outlines for tabular data: CSV, TSV, Excel, Parquet.

  level 0  shape + column names (+ sheet names)
  level 1  + inferred dtypes + the index / first-column values ("row names")
  level 2  + per-column statistics: nulls, uniques, range or top categories
  level 3  + the first rows rendered
  level 4+ + more rows

Uses pandas when available and falls back to the stdlib `csv` module, so a CSV
outline works in any environment.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

MAX_ROWS_SCAN = 200_000


def _sep(kind: str) -> str:
    return "\t" if kind == "tsv" else ","


def _pandas():
    try:
        import pandas as pd

        return pd
    except Exception:
        return None


# ---------------------------------------------------------------------------

def extract(path: Path, kind: str) -> str:
    """Full content as delimited text (bounded)."""
    pd = _pandas()
    if kind in ("csv", "tsv"):
        if pd is None:
            return path.read_text(encoding="utf-8", errors="replace")
        df = pd.read_csv(path, sep=_sep(kind), nrows=MAX_ROWS_SCAN)
        return df.to_csv(index=False)
    if kind == "excel":
        if pd is None:
            return "[pandas/openpyxl required to read Excel files]"
        sheets = pd.read_excel(path, sheet_name=None)
        parts = []
        for name, df in sheets.items():
            parts.append(f"### sheet: {name}  ({len(df)} rows x {len(df.columns)} cols)")
            parts.append(df.to_csv(index=False))
        return "\n".join(parts)
    if kind == "parquet":
        if pd is None:
            return "[pandas/pyarrow required to read Parquet files]"
        return pd.read_parquet(path).to_csv(index=False)
    return path.read_text(encoding="utf-8", errors="replace")


def _describe_frame(df, level: int, name: str = "") -> list[str]:
    pd = _pandas()
    out: list[str] = []
    label = f"sheet `{name}`: " if name else ""
    out.append(f"{label}{len(df):,} rows x {len(df.columns)} columns")

    cols = list(df.columns)
    if level == 0:
        out.append("columns: " + ", ".join(str(c) for c in cols[:200]))
        if len(cols) > 200:
            out.append(f"… {len(cols) - 200} more columns")
        return out

    # level >= 1: dtypes + index
    out.append("--- columns ---")
    for c in cols[:300]:
        dt = str(df[c].dtype)
        line = f"  {c}  [{dt}]"
        if level >= 2:
            s = df[c]
            nulls = int(s.isna().sum())
            uniq = int(s.nunique(dropna=True))
            line += f"  nulls={nulls}  unique={uniq}"
            try:
                if pd is not None and pd.api.types.is_numeric_dtype(s) and uniq:
                    line += f"  min={s.min():.6g}  max={s.max():.6g}  mean={s.mean():.6g}"
                elif uniq:
                    top = s.dropna().astype(str).value_counts().head(3)
                    line += "  top=" + ", ".join(f"{k!r}x{v}" for k, v in top.items())
            except Exception:
                pass
        out.append(line)
    if len(cols) > 300:
        out.append(f"  … {len(cols) - 300} more columns")

    if level >= 1:
        idx = df.index
        try:
            named = idx.name is not None or str(idx.dtype) == "object"
        except Exception:
            named = False
        if named:
            vals = [str(v) for v in list(idx[:12])]
            out.append(f"--- row index ({idx.name or 'unnamed'}) ---")
            out.append(", ".join(vals) + (" …" if len(idx) > 12 else ""))
        elif level >= 1 and len(cols):
            first = df[cols[0]]
            vals = [str(v) for v in list(first.head(12))]
            out.append(f"--- first column `{cols[0]}` (row labels) ---")
            out.append(", ".join(vals) + (" …" if len(df) > 12 else ""))

    if level >= 3:
        n = 10 if level == 3 else 40
        out.append(f"--- first {min(n, len(df))} rows ---")
        try:
            out.append(df.head(n).to_string(max_colwidth=40))
        except Exception:
            out.append(str(df.head(n)))
    return out


def outline(path: Path, level: int, kind: str) -> str:
    pd = _pandas()

    if kind == "excel":
        if pd is None:
            return "[pandas + openpyxl are required to outline Excel files]"
        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except Exception as e:
            return f"[could not read workbook: {e}]"
        out = [f"--- workbook: {len(sheets)} sheet(s) ---",
               "sheets: " + ", ".join(sheets)]
        if level == 0:
            for name, df in sheets.items():
                out.append(f"  {name}: {len(df):,} rows x {len(df.columns)} cols  "
                           f"[{', '.join(str(c) for c in list(df.columns)[:12])}"
                           f"{' …' if len(df.columns) > 12 else ''}]")
            return "\n".join(out)
        for name, df in sheets.items():
            out.append("")
            out += _describe_frame(df, level, name)
        return "\n".join(out)

    if kind == "parquet":
        if pd is None:
            return "[pandas + pyarrow are required to outline Parquet files]"
        df = pd.read_parquet(path)
        return "\n".join(_describe_frame(df, level))

    # --- CSV / TSV ---------------------------------------------------------
    if pd is not None:
        try:
            df = pd.read_csv(path, sep=_sep(kind), nrows=MAX_ROWS_SCAN,
                             low_memory=False)
            lines = _describe_frame(df, level)
            if len(df) >= MAX_ROWS_SCAN:
                lines.append(f"(scan capped at {MAX_ROWS_SCAN:,} rows)")
            return "\n".join(lines)
        except Exception:
            pass  # fall through to the stdlib reader

    text = path.read_text(encoding="utf-8", errors="replace")
    rdr = csv.reader(io.StringIO(text), delimiter=_sep(kind))
    rows = []
    for i, r in enumerate(rdr):
        rows.append(r)
        if i > 5000:
            break
    if not rows:
        return "(empty file)"
    header = rows[0]
    body = rows[1:]
    out = [f"{len(body):,} data rows x {len(header)} columns",
           "columns: " + ", ".join(header)]
    if level >= 1 and body:
        out.append(f"--- first column `{header[0]}` (row labels) ---")
        out.append(", ".join(r[0] for r in body[:12] if r))
    if level >= 3:
        n = 10 if level == 3 else 40
        out.append(f"--- first {min(n, len(body))} rows ---")
        for r in body[:n]:
            out.append("  " + " | ".join(x[:30] for x in r))
    return "\n".join(out)
