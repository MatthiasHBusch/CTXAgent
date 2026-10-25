"""Bounded LaTeX source inspection and managed, noninteractive compilation.

Static analysis is explicitly best effort: macro-generated inputs/labels/cites
cannot be resolved without TeX. Builds use latexmk without rc loading or shell
escape. This reduces surprises; it does NOT sandbox TeX or included packages.
"""
from __future__ import annotations

from collections import defaultdict
import re
import shutil
import sys
from pathlib import Path

from . import jobs
from .observations import version
from .readers.latex import strip_comments

MAX_FILES = 200
MAX_BYTES = 8 * 1024 * 1024
INPUT = re.compile(r'\\(?:input|include|subfile)\s*\{([^{}]+)\}')
LABEL = re.compile(r'\\label\s*\{([^{}]+)\}')
REF = re.compile(r'\\(?:ref|eqref|autoref|cref|Cref|pageref)\*?\s*\{([^{}]+)\}')
CITE = re.compile(r'\\(?:[A-Za-z]*cite[A-Za-z]*)\*?(?:\s*\[[^\]]*\])*\s*\{([^{}]+)\}')
BIB = re.compile(r'\\(?:bibliography|addbibresource)(?:\s*\[[^\]]*\])?\s*\{([^{}]+)\}')


def inspect(sb, main: str) -> dict:
    main_path = sb.resolve(main, must_exist=True)
    if main_path.suffix.lower() != '.tex':
        raise ValueError('main must be a .tex source')
    files, missing, dynamic, blocked = [], [], [], []
    labels, refs, citations, bibkeys = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    queue, seen, seen_bibs = [main_path], set(), set()
    total = 0
    truncated = False
    while queue:
        if len(files) >= MAX_FILES:
            truncated = True
            break
        p = queue.pop(0)
        if p in seen:
            continue
        seen.add(p)
        p = sb.resolve(str(p), must_exist=True)
        total += p.stat().st_size
        if total > MAX_BYTES:
            truncated = True
            break
        text = strip_comments(p.read_text(encoding='utf-8', errors='replace'))
        files.append({'path': str(p), 'sha256': version(p)})
        for regex, target in [(LABEL, labels), (REF, refs), (CITE, citations)]:
            for match in regex.finditer(text):
                location = {'path': str(p), 'line': text.count('\n', 0, match.start()) + 1}
                for key in match[1].split(','):
                    target[key.strip()].append(location)
        for regex, bib in [(INPUT, False), (BIB, True)]:
            for match in regex.finditer(text):
                for name in match[1].split(','):
                    if '\\' in name or '#' in name:
                        dynamic.append({'source': str(p), 'expression': name})
                        continue
                    name = name.strip()
                    child = Path(name)
                    if not child.suffix:
                        child = child.with_suffix('.bib' if bib else '.tex')
                    if not child.is_absolute():
                        # TeX's default relative lookup is cwd/main directory, not
                        # necessarily the included file's own directory.
                        child = main_path.parent / child
                    try:
                        child = sb.resolve(str(child))
                    except Exception as exc:
                        blocked.append({'source': str(p), 'target': name, 'reason': str(exc)})
                        continue
                    if not child.is_file():
                        missing.append({'source': str(p), 'target': str(child)})
                    elif bib:
                        if child in seen_bibs:
                            continue
                        seen_bibs.add(child)
                        if child.stat().st_size + total > MAX_BYTES:
                            truncated = True
                            continue
                        total += child.stat().st_size
                        btext = child.read_text(encoding='utf-8', errors='replace')
                        for key in re.findall(r'@(?!(?:comment|string|preamble)\b)\w+\s*\{\s*([^,\s]+)\s*,', btext, re.I):
                            bibkeys[key].append(str(child))
                    else:
                        queue.append(child)
    return {'main': str(main_path), 'files': files, 'missing_inputs': missing, 'blocked_inputs': blocked,
            'dynamic_inputs': dynamic, 'truncated': truncated,
            'duplicate_labels': {k: v for k, v in labels.items() if len(v) > 1},
            'unresolved_references': {k: v for k, v in refs.items() if k not in labels},
            'unresolved_citations': {k: v for k, v in citations.items() if k not in bibkeys and k != '*'},
            'duplicate_bibkeys': {k: v for k, v in bibkeys.items() if len(v) > 1},
            'limitations': 'Static candidates only. Macro-generated inputs, custom citation commands, '
                          'external labels and TeX search paths are not fully resolved. Build diagnostics are authoritative.'}


def shell_command(argv: list[str]) -> str:
    if sys.platform == 'win32':
        return '& ' + ' '.join("'" + arg.replace("'", "''") + "'" for arg in argv) + '; exit $LASTEXITCODE'
    import shlex
    return shlex.join(argv)


def build(sb, main: str, owner: str, engine: str = 'pdflatex', timeout: int = 180) -> dict:
    if engine not in ('pdflatex', 'lualatex', 'xelatex'):
        raise ValueError('engine must be pdflatex, lualatex or xelatex')
    p = sb.resolve(main, must_exist=True)
    if p.suffix.lower() != '.tex':
        raise ValueError('main must be a .tex source')
    executable = shutil.which('latexmk')
    if not executable:
        raise RuntimeError('latexmk is not on PATH; install TeX Live or MiKTeX with latexmk/Perl')
    if not shutil.which(engine):
        raise RuntimeError(f'{engine} is not on PATH')
    out = sb.resolve(str(p.parent / '.ctxagent-build'))
    out.mkdir(exist_ok=True)
    argv = [executable, '-norc', {'pdflatex': '-pdf', 'lualatex': '-lualatex', 'xelatex': '-xelatex'}[engine],
            '-interaction=nonstopmode', '-halt-on-error', '-file-line-error', '-recorder',
            '-latexoption=-no-shell-escape', '-outdir=' + str(out), str(p)]
    data = jobs.start(shell_command(argv), p.parent, owner, timeout)
    return {'job': data, 'main': str(p), 'main_sha256': version(p), 'engine': engine,
            'argv': argv, 'output_dir': str(out), 'pdf': str(out / (p.stem + '.pdf')),
            'log': str(out / (p.stem + '.log')),
            'next': 'Wait for job completion, inspect latex_diagnostics(log), then render_pdf(pdf). '
                    'Existing PDF may be stale if this build fails.'}


def diagnostics(path: Path, max_entries: int = 200) -> dict:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('Log exceeds 8 MiB; use read_job_log for bounded inspection')
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    entries = []
    for i, line in enumerate(lines):
        item = None
        match = re.match(r'^(.*\.(?:tex|sty|cls)):(\d+):\s*(.*)', line)
        if match:
            item = {'severity': 'error', 'file': match[1], 'line': int(match[2]), 'message': match[3]}
        elif line.startswith('!'):
            item = {'severity': 'error', 'message': line[1:].strip()}
        elif re.search(r'(?:LaTeX|Package \S+|Class \S+) Warning:', line):
            item = {'severity': 'warning', 'message': line.strip()}
        elif re.search(r'(?:Over|Under)full \\[hv]box', line):
            item = {'severity': 'layout', 'message': line.strip()}
        if item:
            item['log_line'] = i + 1
            item['context'] = '\n'.join(lines[i:min(i + 3, len(lines))])
            entries.append(item)
    return {'log': str(path), 'sha256': version(path), 'entries': entries[:max_entries],
            'total_entries': len(entries), 'truncated': len(entries) > max_entries,
            'output_written': any('Output written on ' in line for line in lines),
            'fatal': any('Fatal error' in line or 'Emergency stop' in line for line in lines),
            'note': 'Compiler output is not visual verification. Page layout requires render_pdf.'}
