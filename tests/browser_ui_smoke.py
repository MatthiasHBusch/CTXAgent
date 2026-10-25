"""Offline Playwright smoke for live browser research; no server or model calls.

Run: python tests/browser_ui_smoke.py
Requires Playwright and Edge, or a Playwright-installed Chromium fallback.
All HTTP requests are fulfilled locally; delayed routes exercise viewEpoch races.
"""
import base64
import copy
import json
import sys
from pathlib import Path

from playwright.sync_api import Error, expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ctxagent.config import BACKEND_BUILD, SessionConfig  # noqa: E402

TOOLS = ['browser_open', 'browser_view', 'browser_list', 'browser_render',
         'browser_find', 'browser_note', 'browser_summarize', 'browser_notes',
         'browser_close', 'web_search', 'web_open', 'web_find']
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aYeUAAAAASUVORK5CYII=')
RAW = 'RAW_PAGE_BODY_MUST_NEVER_ENTER_CHAT'


def main():
    cfg = SessionConfig(model='test/model', workdirs=['C:\\work'],
                        enabled_tools=['read_file']).to_dict()
    workspaces = {cid: {'views': [], 'tokens': 0, 'max_tokens': 24000,
                        'shared_tokens': 0, 'note_count': 0} for cid in ('A', 'B', 'CHILD')}
    notes, actions, patches, artifact_requests, unexpected, pending = [], [], [], [], [], []
    controls = {'hold_get': False, 'hold_action': False, 'hold_send': False, 'fail_next': False}
    errors = []

    def stats(cid='A'):
        return {'total': 3000 + workspaces[cid]['tokens'], 'original': 4000,
                'saved': 1000, 'pressure': .1, 'frozen_tokens': 2000,
                'hot_tokens': 1000, 'hot_start': 1, 'tool_overhead': 100,
                'effective_window': 30000, 'next_tail_ratio': .1, 'deep_ready': True,
                'workspace_tokens': 0, 'browser_tokens': workspaces[cid]['tokens']}

    def view(title='Research source', cid='A'):
        return {'id': 'tab-' + cid, 'page_id': 'page-' + cid,
                'url': 'https://example.test/research', 'title': title,
                'fetched_at': '2026-01-01T00:00:00Z', 'sha256': 'a' * 64,
                'status': 'ready', 'error': '', 'tokens': 400, 'max_tokens': 4000,
                'offset': 0, 'end': 12000, 'total_chars': 22000, 'max_links': 50,
                'images': []}

    def note(text='Verified research summary', title='Summary of source'):
        return {'id': 'note-' + str(len(notes) + 1), 'title': title, 'text': text,
                'url': 'https://example.test/research', 'sha256': 'a' * 64,
                'fetched_at': '2026-01-01T00:00:00Z', 'coverage': 'page',
                'model': 'summary/test', 'tokens': 40, 'kind': 'summary'}

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel='msedge', headless=True)
        except Error:
            browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1500, 'height': 1200})
        page.set_default_timeout(8000)
        page.on('pageerror', lambda error: errors.append(str(error)))

        def serve(route):
            url, method = route.request.url, route.request.method
            if not url.startswith('http://ctx.test/'):
                unexpected.append(url); route.fulfill(status=403); return
            path = url.removeprefix('http://ctx.test')
            body = route.request.post_data_json if route.request.post_data else {}
            if path == '/' or path.startswith('/static/'):
                file = ROOT / 'web' / ('index.html' if path == '/' else path.removeprefix('/static/'))
                if not file.resolve().is_relative_to((ROOT / 'web').resolve()) or not file.is_file():
                    route.fulfill(status=404); return
                route.fulfill(path=str(file), content_type=('text/html' if path == '/' else 'application/javascript' if path.endswith('.js') else 'text/css'))
                return
            if path == '/api/health':
                data = {'has_key': True, 'backend_build': BACKEND_BUILD,
                        'accounting_version': 2, 'server_started_at': 0, 'server_pid': 1,
                        'tools': [{'name': n, 'description': n} for n in ['read_file'] + TOOLS]}
            elif path == '/api/chats':
                data = [{'id': cid, 'title': 'Browser chat ' + cid} for cid in workspaces]
            elif path == '/api/models':
                data = {'models': []}
            elif path.startswith(('/api/providers', '/api/flex')):
                data = {'ok': True, 'allowed': [], 'providers': [], 'total': 0}
            elif path.startswith('/api/chats/'):
                parts = path.split('/'); cid = parts[3]; suffix = '/'.join(parts[4:])
                ws = workspaces[cid]
                if not suffix:
                    data = {'id': cid, 'title': 'Browser chat ' + cid, 'config': cfg,
                            'usage': {}, 'stats': stats(cid), 'delegated': cid == 'CHILD',
                            'segments': [{'id': 'user-1', 'kind': 'user', 'body': 'Existing question', 'tokens': 5},
                                         {'id': 'call-1', 'kind': 'tool_call', 'body': '', 'tool_calls': [
                                             {'id': 'tc-1', 'function': {'name': 'read_file', 'arguments': '{"path":"sample.txt"}'}}]},
                                         {'id': 'result-1', 'kind': 'tool_result', 'tool_call_id': 'tc-1', 'body': 'Existing tool result', 'original_tokens': 5},
                                         {'id': 'asst-1', 'kind': 'assistant_text', 'body': 'Existing answer', 'tokens': 5}]}
                elif suffix == 'context':
                    data = {'stats': stats(cid), 'usage': {}, 'segments': [], 'compaction_log': []}
                elif suffix == 'documents':
                    data = {'workspace': {'views': [], 'tokens': 0, 'budget': 24000}, 'stats': stats(cid)}
                elif suffix == 'config':
                    assert method == 'PATCH'; patches.append(copy.deepcopy(body))
                    cfg.update(body['config']); data = {'config': cfg, 'stats': stats(cid)}
                elif suffix == 'browser':
                    assert method == 'GET'; data = copy.deepcopy(ws)
                    if controls['hold_get']:
                        controls['hold_get'] = False; pending.append((route, data)); return
                elif suffix == 'browser/action':
                    assert method == 'POST'; actions.append((cid, copy.deepcopy(body)))
                    tool, args = body['tool'], body['arguments']
                    assert tool in cfg['enabled_tools']
                    if controls['fail_next']:
                        controls['fail_next'] = False
                        route.fulfill(status=400, json={'detail': 'Mock browser failure'}); return
                    result = {'id': 'receipt', 'text': RAW}
                    if tool == 'browser_open':
                        assert args == {'url': 'https://example.test/research', 'render_js': True,
                                        'max_tokens': 4000, 'max_chars': 12000}
                        ws.update(views=[view(cid=cid)], tokens=400, shared_tokens=400)
                    elif tool == 'browser_render':
                        assert args == {'tab_id': 'tab-' + cid, 'scroll_y': 0, 'max_tokens': 8000}
                        ws['views'][0]['images'] = [{'id': 'shot-' + cid, 'label': 'Page screenshot', 'width': 1, 'height': 1}]
                    elif tool == 'browser_summarize':
                        assert args == {'tab_id': 'tab-' + cid, 'focus': '', 'max_tokens': 800, 'coverage': 'page'}
                        result = note(); notes.append(result); ws['note_count'] = len(notes)
                    elif tool == 'browser_note':
                        assert args == {'tab_id': 'tab-' + cid, 'note': 'Manual source observation', 'title': '', 'quotation': ''}
                        result = note(args['note'], 'Manual note'); notes.append(result); ws['note_count'] = len(notes)
                    elif tool == 'browser_close':
                        assert args == {'tab_id': 'tab-' + cid, 'summary': '', 'summarize': True, 'max_tokens': 800}
                        result = note('Archived close summary', 'Close summary'); notes.append(result)
                        ws.update(views=[], tokens=0, shared_tokens=0, note_count=len(notes))
                    elif tool == 'browser_notes':
                        assert args['limit'] == 20
                        if args.get('note_id'):
                            result = next(n for n in notes if n['id'] == args['note_id'])
                        else:
                            offset = args['offset']; batch = notes[offset:offset + 20]
                            result = {'notes': [{k: v for k, v in n.items() if k != 'text'} for n in batch],
                                      'total': len(notes), 'next_offset': offset + len(batch),
                                      'at_end': offset + len(batch) >= len(notes)}
                    data = {'result': result, 'workspace': copy.deepcopy(ws), 'stats': stats(cid), 'usage': {}}
                    if controls['hold_action']:
                        controls['hold_action'] = False; pending.append((route, data)); return
                elif suffix.startswith('artifacts/'):
                    artifact_requests.append(path); route.fulfill(content_type='image/png', body=PNG); return
                elif suffix == 'send':
                    assert controls['hold_send']; controls['hold_send'] = False
                    pending.append((route, None)); return
                else:
                    unexpected.append(path); route.fulfill(status=404); return
            else:
                unexpected.append(path); route.fulfill(status=404); return
            route.fulfill(json=data)

        page.route('**/*', serve)
        page.goto('http://ctx.test/')
        page.wait_for_function("S.cfg && !browserUI.busy && document.querySelector('#browser-status').textContent === 'Refreshing tabs complete.'")
        if 'panel-open' not in (page.locator('#app').get_attribute('class') or ''):
            page.locator('#btn-panel').click()
        page.locator('[data-tab="context"]').click()
        page.locator('#browser-details > summary').click()
        page.wait_for_function('!browserUI.busy')
        assert page.locator('#btn-browser-open').is_disabled()
        expect(page.locator('#btn-browser-refresh')).to_be_enabled()  # GET needs no tools
        page.locator('#btn-browser-enable').click()
        page.wait_for_function("!browserUI.busy && document.querySelector('#btn-browser-enable').hidden")
        assert set(TOOLS) <= set(cfg['enabled_tools'])
        assert 'read_file' in cfg['enabled_tools'] and len(patches) == 1
        page.locator('#browser-url').fill('javascript:alert(1)')
        page.locator('#btn-browser-open').click()
        assert 'HTTP(S)' in page.locator('#browser-status').inner_text()
        assert not actions
        page.locator('#browser-url').fill('https://example.test/research')
        page.locator('#browser-js').check()
        page.locator('#btn-browser-open').click()
        page.wait_for_function("!browserUI.busy && document.querySelector('.browser-tab')")
        assert '1 tabs' in page.locator('#browser-budget').inner_text()
        assert '400 / 24.0k' in page.locator('#browser-budget').inner_text()
        assert 'a' * 64 in page.locator('#browser-tabs').inner_text()

        assert not page.locator('#msgwrap .turn-activity').evaluate('(e) => e.open')
        page.locator('#msgwrap .turn-activity > summary').click()
        assert 'Existing tool result' in page.locator('#msgwrap').inner_text()

        # Render/screenshot and note drafts survive metadata refresh; screenshots
        # stay in the sidebar and cannot become chat media or tool results.
        page.locator('#browser-tabs textarea').fill('Manual source observation')
        page.locator('#browser-tabs').get_by_role('button', name='Render', exact=True).click()
        page.wait_for_function('!browserUI.busy')
        assert page.locator('#browser-tabs textarea').input_value() == 'Manual source observation'
        page.locator('.browser-images > summary').click()
        page.wait_for_function("document.querySelector('.browser-images img').complete && document.querySelector('.browser-images img').naturalWidth > 0")
        assert page.locator('.browser-images img').get_attribute('src').endswith('/api/chats/A/artifacts/shot-A')
        assert artifact_requests == ['/api/chats/A/artifacts/shot-A']
        page.locator('#btn-browser-refresh').click(); page.wait_for_function('!browserUI.busy')
        assert page.locator('.browser-images').evaluate('(e) => e.open')
        page.locator('#browser-tabs').get_by_role('button', name='Save note', exact=True).click()
        page.wait_for_function('!browserUI.busy')
        assert page.locator('#browser-tabs textarea').input_value() == ''
        assert 'Manual source observation' in page.locator('#browser-note-view').inner_text()
        page.locator('#browser-tabs').get_by_role('button', name='Summarize', exact=True).click()
        page.wait_for_function('!browserUI.busy')
        assert 'Verified research summary' in page.locator('#browser-note-view').inner_text()
        page.locator('#btn-browser-notes').click(); page.wait_for_function('!browserUI.busy')
        page.locator('#browser-notes-list').get_by_role('button', name='Show note').first.click()
        page.wait_for_function('!browserUI.busy')
        assert 'Manual source observation' in page.locator('#browser-note-view').inner_text()
        assert RAW not in page.locator('body').inner_text()
        assert 'Verified research summary' not in page.locator('#msgwrap').inner_text()
        assert page.locator('#msgwrap img').count() == 0
        assert 'Existing answer' in page.locator('#msgwrap').inner_text()
        assert page.locator('#msgwrap .turn-activity').evaluate('(e) => e.open')
        shot = ROOT / '.ctxagent' / 'browser-ui-smoke.png'; shot.parent.mkdir(exist_ok=True)
        page.set_viewport_size({'width': 1500, 'height': 2000})
        page.locator('#browser-workspace').screenshot(path=str(shot))
        page.set_viewport_size({'width': 1500, 'height': 1200})

        # Busy state blocks repeated actions; streaming blocks mutations but not
        # tab inspection. Programmatic entry points enforce the same lock.
        controls['hold_send'] = True
        page.locator('#input').fill('Mock streaming request'); page.locator('#btn-send').click()
        page.wait_for_function('S.streaming')
        assert page.locator('#btn-browser-open').is_disabled()
        assert page.locator('#browser-tabs').get_by_role('button', name='Close', exact=True).is_disabled()
        before = len(actions)
        page.evaluate("browserAction('browser_close', {tab_id:'tab-A'}, 'Blocked')")
        assert len(actions) == before
        assert page.locator('#btn-browser-refresh').is_enabled()
        page.wait_for_timeout(50)
        route, _ = pending.pop(); route.fulfill(content_type='text/event-stream', body='data: {"type":"final"}\n\n')
        page.wait_for_function('!S.streaming')
        page.evaluate('refreshContext()')  # wait through send-final context/document/browser refresh
        page.wait_for_function('!browserUI.busy')

        # Hold A GET across A -> B -> A: same chat ID, different viewEpoch.
        controls['hold_get'] = True; page.locator('#btn-browser-refresh').click()
        page.wait_for_function('!!browserUI.busy'); page.wait_for_timeout(50)
        old_route, old_data = pending.pop()
        assert page.locator('#btn-browser-refresh').is_disabled()
        page.evaluate("openChat('B')")
        assert page.locator('#browser-tabs textarea').count() == 0
        assert page.locator('#browser-note-view').inner_text() == ''
        page.evaluate("openChat('A')")
        old_data['views'][0]['title'] = 'STALE RESPONSE MUST NOT APPEAR'
        old_route.fulfill(json=old_data); page.wait_for_timeout(50)
        assert 'STALE RESPONSE' not in page.locator('#browser-tabs').inner_text()

        # An old action failure/finalizer cannot clear a new view's busy flag.
        controls['hold_action'] = True
        page.locator('#browser-tabs').get_by_role('button', name='Summarize', exact=True).click()
        page.wait_for_function('!!browserUI.busy'); page.wait_for_timeout(50)
        old_route, _ = pending.pop()
        page.evaluate("openChat('B')")
        controls['hold_get'] = True
        page.evaluate("void openChat('A')")
        page.wait_for_function("S.chatId === 'A' && !!browserUI.busy"); page.wait_for_timeout(50)
        new_route, new_data = pending.pop()
        old_route.fulfill(status=500, body='STALE FAILURE'); page.wait_for_timeout(50)
        assert page.locator('#browser-workspace').get_attribute('aria-busy') == 'true'
        assert 'STALE FAILURE' not in page.locator('#browser-status').inner_text()
        new_route.fulfill(json=new_data); page.wait_for_function('!browserUI.busy')

        # Bounded errors are local, retry is explicit, and default close bills
        # the summary model while removing the live page.
        controls['fail_next'] = True
        page.locator('#browser-tabs').get_by_role('button', name='Render', exact=True).click()
        page.wait_for_function('!browserUI.busy')
        assert 'Mock browser failure' in page.locator('#browser-status').inner_text()
        assert page.locator('#btn-browser-open').is_enabled()
        page.locator('#browser-tabs').get_by_role('button', name='Close', exact=True).click()
        page.wait_for_function('!browserUI.busy')
        assert page.locator('.browser-tab').count() == 0
        assert '0 tabs' in page.locator('#browser-budget').inner_text()
        page.locator('#btn-browser-notes').click(); page.wait_for_function('!browserUI.busy')
        assert 'Close summary' in page.locator('#browser-notes-list').inner_text()
        assert page.locator('#btn-browser-notes-prev').is_disabled()
        assert page.locator('#btn-browser-notes-next').is_disabled()

        # Paginated archive listing has bounded 20-note requests, never loops.
        notes.extend(note('Archived item', 'Archive ' + str(i)) for i in range(25))
        page.locator('#btn-browser-notes').click(); page.wait_for_function('!browserUI.busy')
        assert page.locator('#browser-notes-list .browser-note-row').count() == 20
        page.locator('#btn-browser-notes-next').click(); page.wait_for_function('!browserUI.busy')
        assert page.locator('#btn-browser-notes-prev').is_enabled()
        assert page.locator('#btn-browser-notes-next').is_disabled()
        page.locator('#btn-browser-notes-prev').click(); page.wait_for_function('!browserUI.busy')
        assert page.locator('#btn-browser-notes-prev').is_disabled()

        workspaces['CHILD'].update(views=[view('Worker cached source', 'CHILD')], tokens=400)
        page.evaluate("openChat('CHILD')")
        assert page.locator('#btn-browser-open').is_disabled()
        assert page.locator('#btn-browser-enable').is_disabled()
        assert page.locator('#btn-browser-refresh').is_enabled()
        assert page.locator('#browser-tabs').get_by_role('button', name='Render', exact=True).is_disabled()
        before = len(actions)
        page.evaluate("browserAction('browser_note', {tab_id:'tab-CHILD',note:'blocked'}, 'Blocked')")
        assert len(actions) == before
        assert 'read-only' in page.locator('#browser-lock').inner_text()
        assert not errors, errors
        assert not unexpected, unexpected
        assert not pending, pending
        browser.close()
    print('Browser UI smoke passed: offline open/render/summary/notes/archive/close, screenshot, locks, A-B-A races; no model calls.')


if __name__ == '__main__':
    main()
