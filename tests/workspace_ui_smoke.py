"""Mocked Edge/Playwright document controls; no model calls, no real chat writes."""
import copy
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from ctxagent import config, tools

ROOT = Path(__file__).resolve().parent.parent
cfg = config.SessionConfig(model='test/model', workdirs=['C:\\work'],
                           enabled_tools=['read_file']).to_dict()
workspace = {'views': [], 'tokens': 0, 'budget': 24000}
posted = []


def stats():
    return {'effective_window': 50000, 'frozen_tokens': 10000, 'hot_tokens': 1000,
            'total': 11000 + workspace['tokens'] + (200 if workspace['views'] else 0), 'original': 12000,
            'saved': 1000, 'pressure': .22, 'hot_start': 3, 'tool_overhead': 100,
            'next_tail_ratio': .08, 'deep_ready': True, 'rewrite_interval_batches': 10,
            'rewrite_pending_batches': 0,
            'document_catalog_tokens': 200 if workspace['views'] else 0,
            'workspace_tokens': workspace['tokens'], 'workspace': copy.deepcopy(workspace)}


with sync_playwright() as pw:
    browser = pw.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width': 1500, 'height': 1100})
    page.set_default_timeout(8000)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    def serve(route):
        path = route.request.url.split('http://ctx.test', 1)[-1]
        method = route.request.method
        body = route.request.post_data_json if route.request.post_data else None
        if path == '/' or path.startswith('/static/'):
            file = ROOT / 'web' / ('index.html' if path == '/' else path.removeprefix('/static/'))
            if not file.resolve().is_relative_to((ROOT / 'web').resolve()) or not file.is_file():
                route.fulfill(status=404); return
            route.fulfill(path=str(file), content_type=('text/html' if path == '/' else 'application/javascript' if path.endswith('.js') else 'text/css')); return
        if path == '/api/health':
            data = {'ok': True, 'backend_build': config.BACKEND_BUILD, 'restart_required': False,
                    'accounting_version': 2, 'has_key': True, 'tools': tools.catalogue(),
                    'default_tools': tools.DEFAULT_TOOLS, 'defaults': cfg}
        elif path == '/api/chats':
            data = [{'id': 'A', 'title': 'Document workspace demo'}]
        elif path == '/api/chats/A':
            data = {'id': 'A', 'title': 'Document workspace demo', 'config': cfg, 'usage': {},
                    'segments': [], 'compaction_log': [], 'stats': stats()}
        elif path == '/api/chats/A/context':
            data = {'stats': stats(), 'usage': {}, 'segments': [], 'compaction_log': []}
        elif path == '/api/chats/A/documents' and method == 'POST':
            posted.append(body)
            assert body['path'] == 'C:\\work\\draft.py'
            assert body['select'] == ['first', 'L20-30']
            assert body['max_tokens'] == 8000
            workspace.update(tokens=420, views=[{'id': 'doc1', 'path': body['path'],
                'select': body['select'], 'tokens': 350, 'version': 'abc123abcdef',
                'status': 'ready', 'error': ''}])
            data = {'workspace': workspace, 'stats': stats()}
        elif path == '/api/chats/A/documents/doc1/close':
            posted.append(body)
            assert body['summary'] == 'Finished updating the methods.'
            workspace.update(views=[], tokens=0)
            data = {'workspace': workspace, 'stats': stats()}
        elif path == '/api/chats/A/documents':
            data = {'workspace': workspace, 'stats': stats()}
        elif path == '/api/chats/A/config':
            posted.append(body)
            cfg.update(body['config'])
            data = {'config': cfg, 'stats': stats()}
        elif path == '/api/models':
            data = {'models': []}
        elif path in ('/api/providers/check', '/api/flex/check'):
            data = {'ok': True, 'providers': []}
        else:
            route.fulfill(status=404, json={'detail': path}); return
        route.fulfill(json=data)
    page.route('**/*', serve)
    page.goto('http://ctx.test')
    try:
        page.locator('#chattitle').filter(has_text='Document workspace demo').wait_for()
    except Exception:
        print('Browser errors:', errors)
        print('Visible page:', page.locator('body').inner_text()[:2000])
        raise
    if 'panel-open' not in (page.locator('#app').get_attribute('class') or ''):
        page.locator('#btn-panel').click()
    page.locator('#btn-doc-enable').click()
    page.wait_for_function("document.querySelector('#btn-doc-enable').hidden")
    assert all(n in cfg['enabled_tools'] for n in ('document_open','document_close','document_list',
                   'document_prepare_edits','document_json_update','document_table_update','document_apply_edits',
                   'document_pdf_inspect','document_pdf_text','document_pdf_render',
                   'document_docx_inspect','document_docx_replace',
                   'document_xlsx_inspect','document_xlsx_update',
                   'document_mesh_inspect','document_obj_transform'))
    page.locator('#doc-path').fill('C:\\work\\draft.py')
    page.locator('#doc-select').fill('first\nL20-30')
    page.locator('#btn-doc-open').click()
    page.locator('#doc-list').get_by_text('C:\\work\\draft.py', exact=True).wait_for()
    page.locator('#btn-doc-refresh').click()
    page.wait_for_function("document.getElementById('s-doc-tools').textContent === '200'")
    page.screenshot(path=str(ROOT / '.ctxagent' / 'workspace-ui-smoke.png'))
    page.locator('#doc-list').get_by_role('button', name='Close', exact=True).click()
    page.locator('#doc-close-note').fill('Finished updating the methods.')
    page.locator('.modal').get_by_role('button', name='Close view', exact=True).click()
    page.locator('#doc-list').get_by_text('No open documents.', exact=True).wait_for()
    assert len(posted) == 3, posted
    assert not errors, errors
    browser.close()
print('Workspace UI smoke: enabled tools, opened/refreshed/closed selected view; no browser errors.')
