"""Canvas tools: canvas_open / canvas_close are native; the operations (canvas.OPERATIONS)
exist only while a canvas is open and are offered like document operations (the catalog
behind document_call, or deferred tools on the direct Anthropic API)."""
from __future__ import annotations

import asyncio
import json

from .canvas import LIFECYCLE, OPERATIONS, CanvasError

CANVAS_TOOLS = list(LIFECYCLE) + list(OPERATIONS)

_POINT = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}
_BOX = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4,
        'description': '[x0, y0, x1, y1] in canvas pixels'}
_COLOUR = {'type': 'string', 'description': 'CSS colour name or #rrggbb / #rrggbbaa (aa = opacity)'}
_NUMBER = {'type': 'number'}
_OP = {'type': 'object', 'required': ['type'], 'properties': {
    'type': {'type': 'string', 'enum': ['line', 'polyline', 'arrow', 'rect', 'ellipse', 'circle', 'polygon',
                                        'marker', 'text', 'erase', 'paste']},
    'points': {'type': 'array', 'items': _POINT, 'minItems': 2, 'maxItems': 4000},
    'from': _POINT, 'to': _POINT, 'at': _POINT, 'center': _POINT, 'box': _BOX,
    'radius': _NUMBER, 'width': _NUMBER, 'head': _NUMBER, 'size': _NUMBER, 'padding': _NUMBER,
    'max_width': _NUMBER, 'color': _COLOUR, 'fill': _COLOUR, 'outline': _COLOUR,
    'background': _COLOUR, 'border': _COLOUR, 'text': {'type': 'string'},
    'align': {'type': 'string', 'enum': ['left', 'center', 'right']},
    'shape': {'type': 'string', 'enum': ['dot', 'cross', 'x']}, 'path': {'type': 'string'}}}


def register(registry, ToolSpec, ToolResult, ToolError, params):
    def canvas_of(ctx):
        if not ctx or not ctx.get('session'):
            raise ToolError('This tool requires a chat session')
        return ctx['session'].conv.canvas

    def receipt(desc, note='', **extra):
        data = {'version': desc['version'], 'label': desc['label'], 'size': [desc['width'], desc['height']],
                **extra, **({'note': note} if note else {})}
        return ToolResult(text=json.dumps(data, ensure_ascii=False))

    # Said once, when the canvas opens; the receipts of the operations stay short.
    OPENED = ('Shown to the user in the Canvas panel. The current version is the image at the end of each '
              'of your requests from now on; a single image, not one per version.')

    async def run(fn, *args, **kw):
        try:
            return await asyncio.to_thread(fn, *args, **kw)
        except CanvasError as exc:
            raise ToolError(f'{exc}. The canvas is unchanged.')

    async def canvas_open(sb, path=None, width=1600, height=1000, background='white', label='', ctx=None):
        canvas = canvas_of(ctx)
        replaced = canvas.is_open
        if path:
            p = sb.resolve(path, must_exist=True)
            desc = await run(canvas.open_file, p, label)
        else:
            desc = await run(canvas.open_blank, width, height, background, label)
        return receipt(desc, OPENED + (' The previous canvas and its versions were closed.' if replaced else ''),
                       operations=list(OPERATIONS))

    async def canvas_close(sb, ctx=None):
        canvas = canvas_of(ctx)
        if not canvas.is_open:
            return ToolResult(text='No canvas is open.')
        canvas.close()
        return ToolResult(text='Canvas closed; it is no longer in your requests or shown to the user.')

    async def canvas_draw(sb, ops, label='', ctx=None):
        canvas = canvas_of(ctx)
        boxes: list = []
        desc = await run(canvas.draw, ops, label, sb if sb.roots else None, boxes)
        return receipt(desc, drew=desc['summary'], **({'text_boxes': boxes} if boxes else {}))

    async def canvas_script(sb, code, label='', timeout=60, ctx=None):
        canvas = canvas_of(ctx)
        desc, output = await run(canvas.run_script, code, label, timeout)
        return receipt(desc, **({'stdout': output} if output.strip() else {}))

    async def canvas_focus(sb, box=None, rulers=None, ctx=None):
        canvas = canvas_of(ctx)
        await run(canvas.set_focus, box, rulers)
        return ToolResult(text=json.dumps({'focus': canvas.focus, 'rulers': canvas.rulers,
                                           'note': 'Your next request shows ' + (
                                               'this region enlarged.' if canvas.focus else 'the whole canvas.')}))

    async def canvas_undo(sb, steps=1, to_version=None, ctx=None):
        canvas = canvas_of(ctx)
        desc = await run(canvas.undo, steps, to_version)
        return receipt(desc, 'Back at this version; later versions stay available through to_version.')

    async def canvas_save(sb, path, format=None, ctx=None):
        canvas = canvas_of(ctx)
        target = sb.resolve(path)
        saved = await run(canvas.save, target, format)
        return ToolResult(text=f'Saved canvas version {canvas.current()["version"]} to {saved}',
                          resource_key=sb.key(saved), mutates=True)

    specs = [
        ToolSpec('canvas_open',
            'Open the live canvas: an image shown to the user in the Canvas panel and kept current in your '
            'context (one image at the end of every request), with drawing operations while it is open. '
            'path opens an image file (also view_image(canvas=true)); without path a blank canvas of '
            'width x height px (default 1600x1000) and background. One canvas per chat; opening another '
            'replaces it.',
            params({'path': {'type': 'string'}, 'width': {'type': 'integer', 'minimum': 16, 'maximum': 4096},
                    'height': {'type': 'integer', 'minimum': 16, 'maximum': 4096}, 'background': _COLOUR,
                    'label': {'type': 'string'}}, []),
            canvas_open, 'canvas', needs_ctx=True),
        ToolSpec('canvas_close', 'Close the canvas: it leaves your context and the user\'s Canvas panel.',
                 params({}, []), canvas_close, 'canvas', needs_ctx=True),
        ToolSpec('canvas_draw',
            'Draw on the open canvas: any number of shapes in one call, one undo step. Coordinates in canvas '
            'pixels (the rulers of your canvas image). Per type: line/polyline points[[x,y],...] color width; '
            'arrow from to color width head; rect box fill outline width radius; ellipse box fill outline; '
            'circle center radius fill outline; polygon points fill outline; marker at size shape(dot|cross|x) '
            'color; text at (top-left, or the anchor x with align) text size color align max_width(wraps) '
            'background border padding; erase box (back to the background); paste path(image file) box '
            '(fitted inside). Colours with alpha (#rrggbbaa) blend. The result gives the box each text '
            'took (text_boxes, by op index).',
            params({'ops': {'type': 'array', 'items': _OP, 'minItems': 1, 'maxItems': 300},
                    'label': {'type': 'string', 'description': 'Name of this step in the version history.'}},
                   ['ops']),
            canvas_draw, 'canvas', needs_ctx=True),
        ToolSpec('canvas_script',
            'Draw on the open canvas with a short Python script (its own process, your user rights, '
            'timeout default 60 s). Given: img (current canvas, Pillow RGBA), draw (ImageDraw on img), W, H, '
            'font(size), Image/ImageDraw/ImageFont/ImageFilter/ImageOps, np (numpy). Whatever img holds at '
            'the end becomes the next version (assign a new image to resize or replace). print() output '
            'is returned. For plots, grids, many shapes, pixel operations.',
            params({'code': {'type': 'string'}, 'label': {'type': 'string'},
                    'timeout': {'type': 'integer', 'minimum': 1, 'maximum': 300}}, ['code']),
            canvas_script, 'canvas', dangerous=True, needs_ctx=True),
        ToolSpec('canvas_focus',
            'Zoom your view of the canvas: from the next request your canvas image shows only box, '
            'enlarged (up to 4x), for detail work; box null shows the whole canvas again. rulers toggles '
            'the coordinate rulers of your copy. The user\'s panel outlines the region; the canvas itself '
            'does not change.',
            params({'box': {'anyOf': [_BOX, {'type': 'null'}]}, 'rulers': {'type': 'boolean'}}, []),
            canvas_focus, 'canvas', needs_ctx=True),
        ToolSpec('canvas_undo',
            'Go back on the canvas: steps versions back (default 1), or to_version (any listed version, '
            'also a later one, which redoes). The version history is in the header of your canvas image.',
            params({'steps': {'type': 'integer', 'minimum': 1}, 'to_version': {'type': 'integer'}}, []),
            canvas_undo, 'canvas', needs_ctx=True),
        ToolSpec('canvas_save',
            'Save the current canvas version as an image file in a working directory '
            '(format png, jpeg or webp; default from the file extension, else png).',
            params({'path': {'type': 'string'}, 'format': {'type': 'string', 'enum': ['png', 'jpeg', 'webp']}},
                   ['path']),
            canvas_save, 'canvas', dangerous=True, needs_ctx=True),
    ]
    for spec in specs:
        registry[spec.name] = spec
