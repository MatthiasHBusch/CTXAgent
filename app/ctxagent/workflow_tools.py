"""LaTeX and safe-edit adapters; enabled explicitly for existing chats."""
import asyncio
import json
from . import edit_plans, latex_workflow


def register(registry, ToolSpec, ToolResult, params):
    def result(value):
        return ToolResult(text=json.dumps(value, ensure_ascii=False, indent=2))

    async def latex_inspect(sb, main):
        return result(await asyncio.to_thread(latex_workflow.inspect, sb, main))

    async def latex_build(sb, main, engine='pdflatex', timeout=180, ctx=None):
        return result(await asyncio.to_thread(latex_workflow.build, sb, main, ctx['session'].chat_id, engine, timeout))

    async def latex_diagnostics(sb, path, max_entries=200):
        p = sb.resolve(path, must_exist=True)
        return result(await asyncio.to_thread(latex_workflow.diagnostics, p, max(1, min(1000, max_entries))))

    async def prepare_edit_plan(sb, changes, ctx=None):
        return result(await asyncio.to_thread(edit_plans.prepare, sb, ctx['session'].chat_id, changes))

    async def apply_edit_plan(sb, plan_id, rollback=False, ctx=None):
        return result(await asyncio.to_thread(edit_plans.apply, sb, ctx['session'].chat_id, plan_id, rollback))

    specs = [
        ToolSpec('latex_inspect', 'Static check of a LaTeX main file and its input/include graph: source hashes, missing inputs, duplicate labels, unresolved reference and citation candidates. TeX is not run.',
                 params({'main': {'type': 'string'}}, ['main']), latex_inspect, 'latex'),
        ToolSpec('latex_build', 'Run latexmk as a job (no rc files, no shell escape); returns the job id and the PDF and log paths. After a failed build the PDF on disk is the old one.',
                 params({'main': {'type': 'string'}, 'engine': {'type': 'string', 'enum': ['pdflatex','lualatex','xelatex']},
                         'timeout': {'type': 'integer', 'minimum': 1, 'maximum': 604800}}, ['main']),
                 latex_build, 'latex', dangerous=True, needs_ctx=True),
        ToolSpec('latex_diagnostics', 'Errors, warnings and layout messages parsed from a TeX log.',
                 params({'path': {'type': 'string'}, 'max_entries': {'type': 'integer'}}, ['path']), latex_diagnostics, 'latex'),
        ToolSpec('prepare_edit_plan', 'Prepare exact-string edits across up to 30 files as one plan and return its unified diff; nothing is written until apply_edit_plan. BOM and CRLF are preserved.',
                 params({'changes': {'type': 'array', 'minItems': 1, 'maxItems': 30, 'items': {
                     'type': 'object', 'properties': {'path': {'type': 'string'}, 'expected_sha256': {'type': 'string'},
                     'edits': {'type': 'array', 'items': {'type': 'object', 'properties': {
                         'old_string': {'type': 'string'}, 'new_string': {'type': 'string'}, 'replace_all': {'type': 'boolean'}},
                         'required': ['old_string','new_string']}}}, 'required': ['path','edits']}}}, ['changes']),
                 prepare_edit_plan, 'editing', needs_ctx=True),
        ToolSpec('apply_edit_plan', 'Write a prepared plan if the files are unchanged since it was prepared. Files are replaced one by one with backups, not as one transaction; rollback=true restores the backups.',
                 params({'plan_id': {'type': 'string'}, 'rollback': {'type': 'boolean'}}, ['plan_id']),
                 apply_edit_plan, 'editing', dangerous=True, needs_ctx=True),
    ]
    for spec in specs:
        registry[spec.name] = spec
