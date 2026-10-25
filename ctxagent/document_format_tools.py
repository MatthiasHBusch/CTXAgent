"""Registration only. Native operations are dispatched through document_call."""
import asyncio
import json

from . import document_formats as native


def register(registry, ToolSpec, ToolResult, obj):
    string = lambda description='': {'type': 'string', 'description': description}
    integer = lambda lo, hi: {'type': 'integer', 'minimum': lo, 'maximum': hi}
    array = lambda items, hi: {'type': 'array', 'items': items, 'minItems': 1, 'maxItems': hi}
    fields = {'document_id': string(), 'expected_version': string('Current workspace SHA-256 version.')}
    required = list(fields)

    async def document_call(sb, operation, arguments_json, ctx=None):
        from .document_catalog import decode
        name, args = decode(ctx['session'], operation, arguments_json)
        return await ctx['session']._exec_tool(name, args)

    registry['document_call'] = ToolSpec('document_call',
        'Execute an enabled operation from the DOCUMENT OPERATION CATALOG immediately before live files. '
        'operation is its exact name; arguments_json is a JSON object encoded as a string. '
        'The catalog contains schemas and compatible document ids. If absent, open a compatible document first. '
        'Prepared edit plans require diff/summary review before application. No arbitrary tool dispatch.',
        obj({'operation': string(), 'arguments_json': {'type': 'string', 'maxLength': 200000}},
            ['operation', 'arguments_json']), document_call, group='documents', needs_ctx=True)

    def wrapper(name):
        async def call(sb, ctx=None, **args):
            session = ctx['session']
            if name.startswith('document_pdf_'):
                result = await asyncio.to_thread(native.pdf, sb, session.conv.workspace, operation=name, **args)
            elif name.startswith('document_docx_'):
                result = await asyncio.to_thread(native.docx, sb, session.chat_id, session.conv.workspace, operation=name, **args)
            elif name.startswith('document_xlsx_'):
                result = await asyncio.to_thread(native.xlsx, sb, session.chat_id, session.conv.workspace, operation=name, **args)
            elif name == 'document_mesh_inspect':
                result = await asyncio.to_thread(native.mesh, sb, session.conv.workspace, **args)
            else:
                result = await asyncio.to_thread(native.obj_transform, sb, session.chat_id, session.conv.workspace, **args)
            images = result.pop('images', [])
            return ToolResult(json.dumps(result, ensure_ascii=False, indent=2), images=images)
        return call

    pages = array(integer(1, 1000000), 20)
    definitions = [
        ('document_pdf_inspect', 'PDF page count, metadata and bookmark tree; read-only, no OCR/editing.', {}, []),
        ('document_pdf_text', 'Extract complete text from 1..20 chosen 1-based PDF pages. Refuses oversize output rather than silently truncating. Not OCR/layout evidence.',
         {'pages': pages, 'max_chars': integer(1, 40000)}, ['pages']),
        ('document_pdf_render', 'Render PDF pages as native visual evidence. Select 1..6 1-based pages; 72..200 DPI with raster safety limits.',
         {'pages': array(integer(1, 1000000), 6), 'dpi': integer(72, 200)}, ['pages']),
        ('document_docx_inspect', 'Inspect zero-based top-level DOCX body paragraphs and direct text-run counts, not tables/headers/drawings. Offset/limit make omitted paragraphs explicit.',
         {'offset': integer(0, 1000000), 'limit': integer(1, 100)}, []),
        ('document_docx_replace', 'Prepare a reviewed DOCX run-local text replacement plan. old_text must occur once within one direct plain text run of paragraph. No run-crossing/control characters/fields. Styles and unrelated ZIP members preserved; verify layout externally.',
         {'updates': array(obj({'paragraph': integer(0, 1000000), 'old_text': string(), 'new_text': string()},
                              ['paragraph', 'old_text', 'new_text']), 100)}, ['updates']),
        ('document_xlsx_inspect', 'Inspect XLSX sheet names, or stored cells of one exact sheet. Values/formula caches only; dates/styles/formulas are not evaluated. Omitted cells explicit.',
         {'sheet': string(), 'max_cells': integer(1, 500)}, []),
        ('document_xlsx_update', 'Prepare a reviewed XLSX existing-cell edit plan. Exact sheet and cell address; value_json encodes string/number/bool/null. No formulas/rich text or absent cells. Cell styles/unrelated package members preserved; formula caches remain stale until Excel recalculates. No macro/signed-package editing.',
         {'sheet': string(), 'updates': array(obj({'cell': string(), 'value_json': string()},
                                                ['cell', 'value_json']), 100)}, ['sheet', 'updates']),
        ('document_mesh_inspect', 'Inspect OBJ or binary STL vertex/face counts and bounds. No external material loading, rendering, units inference or full topology validation. ASCII STL/other CAD formats unsupported.', {}, []),
        ('document_obj_transform', 'Prepare a reviewed OBJ positive-uniform-scale then translation plan for vertex coordinates only. Faces/normals/material lines unchanged; returns before/after bounds and exact diff. Optional homogeneous w must be 1. Units unspecified; not a CAD scene transform.',
         {'translate': {**array({'type': 'number'}, 3), 'minItems': 3}, 'scale': {'type': 'number', 'exclusiveMinimum': 0}}, ['translate']),
    ]
    for name, description, props, extra in definitions:
        registry[name] = ToolSpec(name, description, obj({**fields, **props}, required + extra),
                                  wrapper(name), group='documents', needs_ctx=True)
