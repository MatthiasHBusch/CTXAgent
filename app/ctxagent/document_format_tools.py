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
        'Run an operation from the document operation catalog. The catalog precedes the live files and lists the operations available for the open documents with their parameters. arguments_json is a JSON object as text.',
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
        ('document_pdf_inspect', 'Page count, metadata and bookmark tree of a PDF.', {}, []),
        ('document_pdf_text', 'Text of up to 20 PDF pages (1-based), no OCR. Output over max_chars is refused, not cut.',
         {'pages': pages, 'max_chars': integer(1, 40000)}, ['pages']),
        ('document_pdf_render', 'Render up to 6 PDF pages (1-based) as images, 72-200 DPI.',
         {'pages': array(integer(1, 1000000), 6), 'dpi': integer(72, 200)}, ['pages']),
        ('document_docx_inspect', 'Top-level body paragraphs of a DOCX (zero-based) with their text-run counts; tables, headers and drawings are not included.',
         {'offset': integer(0, 1000000), 'limit': integer(1, 100)}, []),
        ('document_docx_replace', 'Prepare a plan replacing text inside single runs of a DOCX: old_text occurs once within one plain text run of the paragraph. Styles and the rest of the package are kept. Returns the diff; document_apply_edits writes it.',
         {'updates': array(obj({'paragraph': integer(0, 1000000), 'old_text': string(), 'new_text': string()},
                              ['paragraph', 'old_text', 'new_text']), 100)}, ['updates']),
        ('document_xlsx_inspect', 'Sheet names of an XLSX, or the stored cells of one sheet: values and cached formula results; nothing is evaluated.',
         {'sheet': string(), 'max_cells': integer(1, 500)}, []),
        ('document_xlsx_update', 'Prepare a plan setting existing cells of a sheet; value_json is a JSON string, number, bool or null. Formulas and absent cells are not supported. Styles are kept; cached formula results stay stale until Excel recalculates. document_apply_edits writes it.',
         {'sheet': string(), 'updates': array(obj({'cell': string(), 'value_json': string()},
                                                ['cell', 'value_json']), 100)}, ['sheet', 'updates']),
        ('document_mesh_inspect', 'Vertex and face counts and bounds of an OBJ or binary STL.', {}, []),
        ('document_obj_transform', 'Prepare a plan that scales (uniform, positive) and then translates the vertex coordinates of an OBJ; faces, normals and materials stay. Returns the bounds before and after and the diff; document_apply_edits writes it.',
         {'translate': {**array({'type': 'number'}, 3), 'minItems': 3}, 'scale': {'type': 'number', 'exclusiveMinimum': 0}}, ['translate']),
    ]
    for name, description, props, extra in definitions:
        registry[name] = ToolSpec(name, description, obj({**fields, **props}, required + extra),
                                  wrapper(name), group='documents', needs_ctx=True)
