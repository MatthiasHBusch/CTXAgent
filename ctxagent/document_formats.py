"""Conservative native adapters. No external links/materials/scripts are loaded.

OOXML edits rewrite selected XML members only; every other member is copied
unchanged. PDF is read/render-only. Mesh transforms support OBJ vertex geometry,
not a general CAD scene or simulation. All edit output is staged for review.
"""
import io
import json
import math
import posixpath
import re
import struct
import zipfile

from . import artifacts, edit_plans

BY_SUFFIX = {
    '.pdf': ('document_pdf_inspect', 'document_pdf_text', 'document_pdf_render'),
    '.docx': ('document_docx_inspect', 'document_docx_replace', 'document_apply_edits'),
    '.xlsx': ('document_xlsx_inspect', 'document_xlsx_update', 'document_apply_edits'),
    '.obj': ('document_mesh_inspect', 'document_obj_transform', 'document_apply_edits'),
    '.stl': ('document_mesh_inspect',),
}
NAMES = tuple(dict.fromkeys(n for names in BY_SUFFIX.values() for n in names
                           if n != 'document_apply_edits'))
READ_ONLY = frozenset(n for n in NAMES if n.endswith(('inspect', 'text', 'render')))
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
S = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'


def _source(sb, workspace, document_id, expected_version, operation):
    from .document_edits import _checked
    return _checked(sb, workspace, document_id, expected_version, operation, binary=True)


def _unchanged(path, raw):
    if path.read_bytes() != raw:
        raise ValueError('Source changed during native operation; refresh before retrying.')


def _receipt(path, raw, **data):
    _unchanged(path, raw)
    return {'source_sha256': edit_plans.sha(raw), **data}


def pdf(sb, workspace, document_id, expected_version, operation, pages=None, dpi=120, max_chars=20000):
    import fitz
    view, path, raw, _ = _source(sb, workspace, document_id, expected_version, operation)
    with fitz.open(stream=raw, filetype='pdf') as doc:
        if doc.needs_pass:
            raise ValueError('Unlock the encrypted PDF before using these tools.')
        if operation == 'document_pdf_inspect':
            toc = doc.get_toc()
            if len(toc) > 500:
                raise ValueError('Bookmark tree exceeds 500 entries; use existing focused PDF readers.')
            return _receipt(path, raw, page_count=len(doc), metadata=doc.metadata, bookmarks=toc,
                            editing='read/render-only; no OCR, redaction or text reflow')
        pages = [1] if pages is None else pages
        if not pages or len(pages) > 20 or any(type(p) is not int or not 1 <= p <= len(doc) for p in pages):
            raise ValueError('Select 1..20 valid 1-based page numbers.')
        if operation == 'document_pdf_text':
            if type(max_chars) is not int or not 1 <= max_chars <= 40000:
                raise ValueError('max_chars must be 1..40000.')
            texts = [{'page': p, 'text': doc[p - 1].get_text()} for p in dict.fromkeys(pages)]
            if sum(len(p['text']) for p in texts) > max_chars:
                raise ValueError('Selected text exceeds max_chars; choose fewer pages. No text silently truncated.')
            return _receipt(path, raw, pages=texts, note='Extracted text, not OCR or visual/layout evidence.')
    images = artifacts.render(path, pages, dpi)
    if any(image.get('source_sha256') != edit_plans.sha(raw) for image in images):
        raise ValueError('PDF changed before rendering; refresh before retrying.')
    return _receipt(path, raw, images=images)


def _package(raw):
    z = zipfile.ZipFile(io.BytesIO(raw))
    infos = z.infolist()
    if len(infos) > 10000 or len({i.filename for i in infos}) != len(infos):
        z.close()
        raise ValueError('Ambiguous or oversized OOXML package.')
    if sum(i.file_size for i in infos) > 32 * 1024 * 1024 or any(i.file_size > 16 * 1024 * 1024 for i in infos):
        z.close()
        raise ValueError('OOXML expanded size exceeds safety limits.')
    if any(i.flag_bits & 1 or i.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) for i in infos):
        z.close()
        raise ValueError('Encrypted/unsupported OOXML compression.')
    return z


def _xml(data):
    from lxml import etree
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise ValueError('DTD/entities are not supported.')
    root = etree.fromstring(data, parser=etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False))
    # Also catches UTF-16/32 declarations that evade the byte-level precheck.
    if root.getroottree().docinfo.doctype:
        raise ValueError('DTD/entities are not supported.')
    return root


def _bytes(node):
    from lxml import etree
    return etree.tostring(node, encoding='UTF-8', xml_declaration=True, standalone=True)


def _rewrite(z, changes):
    if any(n.startswith('_xmlsignatures/') for n in z.namelist()):
        raise ValueError('Signed Office packages cannot be edited without invalidating signatures.')
    for name, data in changes.items():
        _xml(data)
        if name not in z.namelist():
            raise ValueError('Cannot add an unknown OOXML member.')
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as target:
        target.comment = z.comment
        for info in z.infolist():
            target.writestr(info, changes.get(info.filename, z.read(info.filename)))
    return out.getvalue()


def _binary_plan(sb, owner, view, path, raw, output, changes, note):
    _unchanged(path, raw)
    result = edit_plans.prepare_binary(sb, owner, str(path), edit_plans.sha(raw), output,
                                      json.dumps(changes, ensure_ascii=False, indent=2),
                                      {'id': view['id'], 'path': str(path)})
    result['note'] = note
    result['next'] = 'Review change_summary, then document_call(document_apply_edits, {plan_id}). Source unchanged.'
    return result


def _paragraphs(root):
    return root.findall(f'{{{W}}}body/{{{W}}}p')


def _text_nodes(p):
    # Deliberately direct runs only, not nested drawings/text boxes/fields.
    return p.findall(f'{{{W}}}r/{{{W}}}t')


def _plain(text):
    return isinstance(text, str) and not re.search(r'[\x00-\x1f\ud800-\udfff\ufffe\uffff]', text)


def docx(sb, owner, workspace, document_id, expected_version, operation, updates=None, offset=0, limit=50):
    view, path, raw, _ = _source(sb, workspace, document_id, expected_version, operation)
    with _package(raw) as z:
        root = _xml(z.read('word/document.xml'))
        paragraphs = _paragraphs(root)
        if operation == 'document_docx_inspect':
            if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
                raise ValueError('Use offset >= 0 and limit 1..100.')
            data = [{'paragraph': i, 'text': ''.join(n.text or '' for n in _text_nodes(p)),
                     'text_runs': len(_text_nodes(p))} for i, p in enumerate(paragraphs)
                    if offset <= i < offset + limit]
            if sum(len(p['text']) for p in data) > 40000:
                raise ValueError('Paragraph selection exceeds 40000 characters; narrow the range.')
            return _receipt(path, raw, paragraph_count=len(paragraphs), paragraphs=data,
                            table_count=len(root.findall(f'{{{W}}}body/{{{W}}}tbl')),
                            note='Zero-based top-level body paragraphs; only direct plain-text runs shown/editable. Tables/headers/drawings are not edited.')
        if not updates or len(updates) > 100:
            raise ValueError('Provide 1..100 run-local paragraph replacements.')
        preview = []
        for update in updates:
            index, old, new = update['paragraph'], update['old_text'], update['new_text']
            if type(index) is not int or not 0 <= index < len(paragraphs) or not old or not _plain(old) or not _plain(new):
                raise ValueError('Invalid paragraph/text. No control characters or run-crossing replacements.')
            p = paragraphs[index]
            if p.find(f'.//{{{W}}}fldChar') is not None or p.find(f'.//{{{W}}}instrText') is not None:
                raise ValueError('Paragraph contains a field; refusing a run edit.')
            nodes = _text_nodes(p)
            if sum((n.text or '').count(old) for n in nodes) != 1:
                raise ValueError('Text must occur exactly once within one direct text run; inspect formatting boundaries.')
            node = next(n for n in nodes if old in (n.text or ''))
            before = node.text
            node.text = node.text.replace(old, new, 1)
            if node.text.startswith(' ') or node.text.endswith(' '):
                node.set(XML_SPACE, 'preserve')
            preview.append({'paragraph': index, 'before_run': before, 'after_run': node.text})
        output = _rewrite(z, {'word/document.xml': _bytes(root)})
    return _binary_plan(sb, owner, view, path, raw, output, preview,
                        'Single-run text edits only. Run/paragraph styles and all other package members preserved. Review summary; verify in Word/rendered output for layout.')


def _sheets(z):
    root = _xml(z.read('xl/workbook.xml'))
    rels = _xml(z.read('xl/_rels/workbook.xml.rels'))
    targets = {r.get('Id'): r for r in rels}
    out = {}
    for sheet in root.findall(f'{{{S}}}sheets/{{{S}}}sheet'):
        rel = targets[sheet.get(f'{{{R}}}id')]
        if rel.get('TargetMode') == 'External':
            raise ValueError('External worksheet relationship is not supported.')
        target = rel.get('Target', '')
        name = posixpath.normpath(target.lstrip('/') if target.startswith('/') else 'xl/' + target)
        if not name.startswith('xl/') or name not in z.namelist():
            raise ValueError('Invalid worksheet relationship.')
        out[sheet.get('name')] = name
    return root, out


def _cell_info(cell, shared):
    kind = cell.get('t', 'n')
    value = cell.find(f'{{{S}}}v')
    stored = value.text if value is not None else None
    if kind == 's' and stored is not None:
        item = shared[int(stored)]
        text = ''.join(n.text or '' for n in item.iter(f'{{{S}}}t'))
        rich = item.find(f'{{{S}}}r') is not None
    elif kind == 'inlineStr':
        item = cell.find(f'{{{S}}}is')
        text = ''.join(n.text or '' for n in item.iter(f'{{{S}}}t')) if item is not None else ''
        rich = item is not None and item.find(f'{{{S}}}r') is not None
    else:
        text, rich = stored, False
    f = cell.find(f'{{{S}}}f')
    return {'cell': cell.get('r'), 'kind': kind, 'value': text, 'style': cell.get('s'),
            'formula': f.text if f is not None else None, 'is_formula': f is not None, 'rich_text': rich}


def xlsx(sb, owner, workspace, document_id, expected_version, operation, sheet=None, updates=None, max_cells=100):
    from lxml import etree
    from .document_edits import _json
    view, path, raw, _ = _source(sb, workspace, document_id, expected_version, operation)
    with _package(raw) as z:
        workbook, sheets = _sheets(z)
        if operation == 'document_xlsx_inspect' and sheet is None:
            return _receipt(path, raw, sheets=list(sheets), note='Select a sheet to inspect stored cells. Formula values are caches, not recalculated results.')
        if sheet not in sheets:
            raise ValueError('Use an exact worksheet name from inspection.')
        root = _xml(z.read(sheets[sheet]))
        cells = {c.get('r'): c for c in root.findall(f'{{{S}}}sheetData/{{{S}}}row/{{{S}}}c')}
        shared = list(_xml(z.read('xl/sharedStrings.xml'))) if 'xl/sharedStrings.xml' in z.namelist() else []
        if operation == 'document_xlsx_inspect':
            if type(max_cells) is not int or not 1 <= max_cells <= 500:
                raise ValueError('max_cells must be 1..500.')
            data = [_cell_info(c, shared) for c in list(cells.values())[:max_cells]]
            if len(json.dumps(data)) > 40000:
                raise ValueError('Cell selection exceeds 40000 characters; reduce max_cells.')
            return _receipt(path, raw, sheet=sheet, stored_cell_count=len(cells), cells=data,
                            omitted_cells=max(0, len(cells) - len(data)), note='Stored values/formula caches only. Styles/dates are not evaluated.')
        if not updates or len(updates) > 100:
            raise ValueError('Provide 1..100 existing-cell updates.')
        preview, seen = [], set()
        for update in updates:
            address = update['cell']
            if address in seen or address not in cells:
                raise ValueError('Duplicate or absent stored cell; inserting cells is not supported.')
            seen.add(address)
            cell = cells[address]
            before = _cell_info(cell, shared)
            if before['is_formula'] or before['rich_text']:
                raise ValueError('Formula/rich-text cells are not editable with this adapter.')
            value = _json(update['value_json'])
            if isinstance(value, (dict, list)) or (isinstance(value, str) and not _plain(value)):
                raise ValueError('Use a scalar JSON value without control characters.')
            if type(value) in (int, float):
                from decimal import Decimal
                if not math.isfinite(value) or len(Decimal(str(value)).as_tuple().digits) > 15:
                    raise ValueError('Excel number exceeds supported finite 15-digit precision; use a string.')
            for child in list(cell):
                if child.tag in (f'{{{S}}}v', f'{{{S}}}is'):
                    cell.remove(child)
            cell.attrib.pop('t', None)
            if isinstance(value, str):
                cell.set('t', 'inlineStr')
                t = etree.SubElement(etree.SubElement(cell, f'{{{S}}}is'), f'{{{S}}}t')
                t.text = value
                t.set(XML_SPACE, 'preserve')
            elif value is not None:
                if isinstance(value, bool):
                    cell.set('t', 'b')
                etree.SubElement(cell, f'{{{S}}}v').text = ('1' if value else '0') if isinstance(value, bool) else str(value)
            preview.append({'sheet': sheet, 'cell': address, 'before': before, 'after_value': value})
        calc = workbook.find(f'{{{S}}}calcPr')
        if calc is None:
            calc = etree.Element(f'{{{S}}}calcPr')
            later = {'oleSize', 'customWorkbookViews', 'pivotCaches', 'smartTagPr', 'smartTagTypes',
                     'webPublishing', 'fileRecoveryPr', 'webPublishObjects', 'extLst'}
            index = next((i for i, c in enumerate(workbook) if etree.QName(c).localname in later), len(workbook))
            workbook.insert(index, calc)
        calc.set('calcMode', 'auto')
        calc.set('fullCalcOnLoad', '1')
        calc.set('forceFullCalc', '1')
        output = _rewrite(z, {sheets[sheet]: _bytes(root), 'xl/workbook.xml': _bytes(workbook)})
    return _binary_plan(sb, owner, view, path, raw, output, preview,
                        'Existing non-formula, plain cells only. Cell styles and unrelated package parts preserved. Formula caches elsewhere remain stale until Excel recalculates; no formulas are inserted.')


def mesh_summary(raw, suffix):
    vertices, faces = [], []
    if suffix == '.obj':
        text = raw.decode('utf-8-sig', errors='strict')
        for line in text.splitlines():
            fields = line.split('#', 1)[0].split()
            if not fields:
                continue
            if fields[0] == 'v':
                if len(fields) not in (4, 5) or (len(fields) == 5 and float(fields[4]) != 1):
                    raise ValueError('OBJ supports 3D vertices with optional homogeneous w=1 only.')
                vertices.append(tuple(float(v) for v in fields[1:4]))
            elif fields[0] == 'f':
                if len(fields) < 4:
                    raise ValueError('OBJ face needs at least three vertices.')
                refs = [int(f.split('/')[0]) for f in fields[1:]]
                if any(r == 0 or (r < 0 and -r > len(vertices)) for r in refs):
                    raise ValueError('Invalid OBJ vertex reference.')
                faces.append(refs)
            if len(vertices) > 200000 or len(faces) > 200000:
                raise ValueError('Mesh exceeds 200000 vertices/faces.')
        if any(r > len(vertices) for face in faces for r in face):
            raise ValueError('OBJ face references a missing vertex.')
        kind, face_count = 'OBJ', len(faces)
    elif suffix == '.stl':
        if len(raw) < 84:
            raise ValueError('This initial STL adapter supports binary STL only.')
        count = struct.unpack_from('<I', raw, 80)[0]
        if count > 100000 or len(raw) != 84 + 50 * count:
            raise ValueError('Expected binary STL with exact triangle byte count (up to 100000). ASCII STL is not yet supported.')
        for i in range(count):
            values = struct.unpack_from('<12fH', raw, 84 + 50 * i)
            if not all(math.isfinite(v) for v in values[:12]):
                raise ValueError('STL contains nonfinite coordinates/normals.')
            vertices.extend(tuple(values[j:j + 3]) for j in (3, 6, 9))
        kind, face_count = 'binary STL', count
    else:
        raise ValueError('Supported mesh formats: OBJ and binary STL.')
    if not vertices or any(not all(math.isfinite(v) for v in p) for p in vertices):
        raise ValueError('Mesh has no finite vertices.')
    low = [min(p[i] for p in vertices) for i in range(3)]
    high = [max(p[i] for p in vertices) for i in range(3)]
    return {'format': kind, 'vertex_records': len(vertices), 'faces': face_count,
            'bounds': {'min': low, 'max': high},
            'note': 'Geometry summary, not visual evidence or full topology validation. Units unspecified; external materials not loaded.'}


def mesh(sb, workspace, document_id, expected_version):
    view, path, raw, _ = _source(sb, workspace, document_id, expected_version, 'document_mesh_inspect')
    return _receipt(path, raw, **mesh_summary(raw, path.suffix.lower()))


def obj_transform(sb, owner, workspace, document_id, expected_version, translate, scale=1.0):
    from .document_edits import _structured_plan
    view, path, raw, _ = _source(sb, workspace, document_id, expected_version, 'document_obj_transform')
    if len(translate) != 3 or not all(type(v) in (int, float) and math.isfinite(v) for v in translate):
        raise ValueError('Provide three finite translation coordinates.')
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Uniform scale must be finite and positive; no mirroring/nonuniform normals.')
    before = mesh_summary(raw, '.obj')
    text = raw.decode('utf-8-sig')
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        fields = line.split('#', 1)[0].split()
        if not fields or fields[0] != 'v':
            continue
        values = [float(fields[j + 1]) * scale + translate[j] for j in range(3)]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Transform produces nonfinite coordinates.')
        ending = '\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else ''
        comment = ' #' + line.split('#', 1)[1].rstrip('\r\n') if '#' in line else ''
        extra = ' 1' if len(fields) == 5 else ''
        lines[i] = 'v ' + ' '.join(format(v, '.17g') for v in values) + extra + comment + ending
    output = ''.join(lines)
    after = mesh_summary(output.encode('utf-8'), '.obj')
    _unchanged(path, raw)
    result = _structured_plan(sb, owner, view, path, raw, text, output,
        'Positive uniform scale then translation of OBJ v coordinates only. Faces, normals, materials and other lines untouched. Units not inferred; review bounds and exact diff.')
    result['geometry_before'], result['geometry_after'] = before, after
    return result
