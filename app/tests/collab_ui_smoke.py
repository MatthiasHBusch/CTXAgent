"""Opt-in Edge/Playwright regression: messages queued while a turn runs, the shared
scratchpad (autosave, Ctrl+Enter idle and during a turn) and the Canvas tab.
All HTTP endpoints are mocked. No model generation or real chat modifications.
Screenshots go to $CTXAGENT_SMOKE_SHOTS when set.
"""
import copy
import io
import json
import os
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

from ctxagent import config, tools

root = Path(__file__).resolve().parent.parent
shots = os.environ.get('CTXAGENT_SMOKE_SHOTS')
cfg = config.SessionConfig(model='test/model', enabled_tools=['read_file', 'canvas_open'],
                           workdirs=['C:\\work']).to_dict()
stats = dict(total=20000, original=20000, saved=0, effective_window=379000, frozen_tokens=0, hot_tokens=20000,
             hot_start=0, hot_zone_budget=10000, tool_overhead=4824, pressure=0.05, deep_target_ratio=0.1,
             deep_ready=False, rewrite_interval_batches=2, review_candidate_tokens=0)
chats = {key: dict(id=key, title='Chat ' + key, config=copy.deepcopy(cfg), usage={}, stats=copy.deepcopy(stats),
                   segments=[], compaction_log=[], running=False, queued=[]) for key in ('A', 'B')}
health = dict(ok=True, backend_build=config.BACKEND_BUILD, accounting_version=2, restart_required=False,
              has_key=True, defaults=cfg, default_tools=tools.DEFAULT_TOOLS, tools=tools.catalogue())
pads = {'A': {'text': 'notes for A', 'version': 3}, 'B': {'text': '', 'version': 0}}


def png(colour, size=(320, 200)):
    out = io.BytesIO()
    Image.new('RGB', size, colour).save(out, 'PNG')
    return out.getvalue()


ARTIFACTS = {'v1': png((240, 240, 240)), 'v2': png((200, 60, 60)), 'v3': png((60, 120, 200))}


def version(n, label):
    return {'version': n, 'id': f'v{n}', 'label': label, 'summary': 'rect', 'at': 0, 'width': 320, 'height': 200}


def listing(n, focus=None):
    versions = [version(i, f'step {i}') for i in range(1, n + 1)]
    return {'open': True, 'head': n - 1, 'focus': focus, 'rulers': True, 'source': None, 'revision': n,
            'current': versions[-1], 'versions': versions}


CLOSED = {'open': False, 'head': -1, 'focus': None, 'rulers': True, 'source': None, 'revision': 0,
          'current': None, 'versions': []}
canvas = {'A': copy.deepcopy(CLOSED), 'B': copy.deepcopy(CLOSED)}

with sync_playwright() as pw:
    browser = pw.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width': 1500, 'height': 1000})
    errors, requests = [], []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.add_init_script('''
      const originalFetch = window.fetch.bind(window);
      window.streams = {}; window.sendBodies = [];
      window.fetch = async (url, opts) => {
        if (String(url).endsWith('/send')) {
          const id = String(url).split('/')[3];
          window.sendBodies.push(JSON.parse(opts.body));
          return new Response(new ReadableStream({start(c) { window.streams[id] = c; }}),
            {headers:{'content-type':'text/event-stream'}});
        }
        return originalFetch(url, opts);
      };
      window.emit = (id, event) => window.streams[id].enqueue(
        new TextEncoder().encode('data: ' + JSON.stringify(event) + '\\n\\n'));
      window.endStream = id => {window.streams[id].close(); delete window.streams[id];};
    ''')
    queue = {'A': [], 'B': []}
    running = {'A': False, 'B': False}

    def serve(route):
        req = route.request
        path = req.url.split('http://ctx.test', 1)[-1]
        body = req.post_data_json if req.post_data else None
        requests.append((req.method, path, body))
        if path == '/':
            return route.fulfill(body=(root / 'web/index.html').read_text(encoding='utf-8'), content_type='text/html')
        if path.startswith('/static/'):
            f = root / 'web' / path.removeprefix('/static/')
            if not f.resolve().is_relative_to((root / 'web').resolve()) or not f.is_file():
                return route.fulfill(status=404)
            return route.fulfill(body=f.read_bytes(), content_type='text/javascript' if f.suffix == '.js' else 'text/css')
        if path == '/api/health': data = health
        elif path == '/api/chats': data = [dict(id=c['id'], title=c['title']) for c in chats.values()]
        elif path == '/api/providers/check': data = dict(ok=True, message='mocked provider')
        elif path == '/api/providers/flex': data = dict(known=True, available=False, candidates=[])
        elif path == '/api/models': data = {'models': []}
        elif path.startswith('/api/chats/'):
            parts = path.split('/'); cid = parts[3]; c = chats[cid]
            if '/artifacts/' in path:
                return route.fulfill(body=ARTIFACTS[parts[5]], content_type='image/png')
            if path.endswith('/scratchpad'):
                if req.method == 'PUT':
                    pads[cid] = {'text': body['text'], 'version': pads[cid]['version'] + 1, 'dropped_agent_edits': 0}
                data = pads[cid]
            elif path.endswith('/queue') and req.method == 'POST':
                if not running[cid]:
                    return route.fulfill(status=409, json={'detail': 'No turn is running; send the message instead'})
                item = {'id': f'q{len(queue[cid]) + 1}', 'text': body['message'],
                        'kind': 'scratchpad' if body['scratchpad'] else 'message'}
                queue[cid].append(item)
                data = {'queued': item, 'queue': queue[cid]}
            elif '/queue/' in path and req.method == 'DELETE':
                queue[cid] = [q for q in queue[cid] if q['id'] != parts[5]]
                data = {'queue': queue[cid]}
            elif path.endswith('/canvas/close'):
                canvas[cid] = copy.deepcopy(CLOSED); data = canvas[cid]
            elif path.endswith('/canvas'): data = canvas[cid]
            elif path.endswith('/context'): data = {**c, 'config': c['config']['compaction']}
            elif path.endswith('/stop'): data = {'ok': True}
            else: data = {**c, 'running': running[cid], 'queued': queue[cid]}
        else: data = {}
        route.fulfill(json=data)

    page.route('**/*', serve)
    page.goto('http://ctx.test/')
    page.wait_for_function("S.chatId === 'A' && S.cfg !== null")
    page.wait_for_function("document.querySelector('#sp-text').value === 'notes for A'")

    # -- scratchpad autosave carries the version it started from
    page.locator('#sp-text').fill('notes for A\n- check the plot')
    page.wait_for_function("document.querySelector('#sp-status').textContent === 'saved'")
    put = [r for r in requests if r[0] == 'PUT' and r[1].endswith('/scratchpad')][-1]
    assert put[2] == {'text': 'notes for A\n- check the plot', 'base_version': 3}, put

    # -- Ctrl+Enter while idle starts a turn that reads the scratchpad
    page.locator('#sp-text').press('Control+Enter')
    page.wait_for_function("!!window.streams.A")
    assert page.evaluate("window.sendBodies.at(-1)") == {'scratchpad': True}
    running['A'] = True
    page.evaluate("""emit('A', {type:'user_message', segment:{kind:'user',
      body:'[scratchpad] The user asks you to look at the shared scratchpad (version 4).\\n<scratchpad>\\nnotes\\n</scratchpad>'}})""")
    page.wait_for_function("!!document.querySelector('#msgwrap details.sp-msg')")

    # -- Enter during the turn queues; the chip leaves when the message is delivered
    assert 'queue' in page.locator('.composer-hint').inner_text()
    page.locator('#input').fill('also label the axes')
    page.locator('#input').press('Enter')
    page.wait_for_function("document.querySelectorAll('#queue-chips .queue-chip').length === 1")
    assert page.locator('#input').input_value() == ''
    assert 'also label the axes' in page.locator('#queue-chips').inner_text()
    page.locator('#input').fill('one more thing')
    page.locator('#input').press('Enter')
    page.wait_for_function("document.querySelectorAll('#queue-chips .queue-chip').length === 2")
    page.locator('#queue-chips .queue-chip').nth(1).locator('.q-x').click()  # take it back
    page.wait_for_function("document.querySelectorAll('#queue-chips .queue-chip').length === 1")
    assert page.locator('#input').input_value() == 'one more thing'
    page.locator('#input').fill('')
    # Ctrl+Enter in the scratchpad during the turn queues a scratchpad message
    page.locator('#sp-text').press('Control+Enter')
    page.wait_for_function("document.querySelectorAll('#queue-chips .queue-chip').length === 2")
    assert any(r[0] == 'POST' and r[1].endswith('/queue') and r[2] == {'message': '', 'scratchpad': True}
               for r in requests)

    # -- the agent opens a canvas: the panel switches to the Canvas tab
    page.evaluate("showPanelTab('usage', {quiet: true})")
    canvas['A'] = listing(1)
    page.evaluate(f"emit('A', {{type:'canvas', canvas:{json.dumps(listing(1))}}})")
    page.wait_for_function("document.querySelector('#tab-canvas').classList.contains('active')")
    page.wait_for_function("document.querySelector('#canvas-img').naturalWidth === 320")
    assert not page.locator('#canvas-live').is_hidden()
    assert 'Version 1 (current)' in page.locator('#canvas-meta').inner_text()

    page.evaluate("""emit('A', {type:'user_message', queued_id:'q1', segment:{kind:'user', body:'also label the axes'}})""")
    page.wait_for_function("document.querySelectorAll('#queue-chips .queue-chip').length === 1")

    # -- later changes while another tab is shown put a dot on the Canvas tab
    page.evaluate("showPanelTab('context', {quiet: true})")
    focus = [40, 20, 200, 120]
    canvas['A'] = listing(3, focus)
    page.evaluate(f"emit('A', {{type:'canvas', canvas:{json.dumps(listing(3, focus))}}})")
    page.wait_for_function("document.querySelector('#tab-canvas').classList.contains('has-news')")
    assert page.locator('#tab-context').get_attribute('aria-selected') == 'true'
    page.locator('#tab-canvas').click()
    page.wait_for_function("!document.querySelector('#tab-canvas').classList.contains('has-news')")
    assert page.locator('#canvas-history button').count() == 3
    assert not page.locator('#canvas-focus').is_hidden()
    box = page.locator('#canvas-focus').bounding_box(); frame = page.locator('#canvas-img').bounding_box()
    assert abs(box['width'] / frame['width'] - 0.5) < 0.02, (box, frame)        # 160 of 320 px
    assert page.locator('#canvas-download').get_attribute('download') == 'canvas-v3.png'
    if shots:
        page.screenshot(path=str(Path(shots) / 'collab_canvas.png'))

    # -- an older version can be viewed; the current one is a click away
    page.locator('#canvas-history button').nth(2).click()  # oldest (newest first)
    page.wait_for_function("document.querySelector('#canvas-img').src.endsWith('/v1')")
    assert page.locator('#canvas-focus').is_hidden()
    assert not page.locator('#btn-canvas-current').is_hidden()
    page.locator('#btn-canvas-current').click()
    page.wait_for_function("document.querySelector('#canvas-img').src.endsWith('/v3')")

    # -- click enlarges
    page.locator('#canvas-frame').click()
    page.wait_for_selector('.modal.canvas-zoom img')
    page.keyboard.press('Escape')
    page.wait_for_selector('.modal.canvas-zoom', state='detached')

    # -- the turn ends; the queued scratchpad message is still pending
    running['A'] = False
    queue['A'] = []
    page.evaluate("emit('A', {type:'final'}); endStream('A')")
    page.wait_for_function("!S.runs.has('A')")
    assert page.locator('#btn-canvas-close').is_enabled()
    page.locator('#btn-canvas-close').click()
    page.locator('.modal .btn.danger').click()
    page.wait_for_function("document.querySelector('#canvas-live').hidden")
    assert not page.locator('#canvas-empty').is_hidden()
    assert any(r[1].endswith('/canvas/close') for r in requests)

    # -- another chat: its own scratchpad and no canvas
    page.evaluate("openChat('B')")
    page.wait_for_function("S.chatId === 'B' && document.querySelector('#sp-text').value === ''")
    assert not page.locator('#canvas-empty').is_hidden()
    if shots:
        page.screenshot(path=str(Path(shots) / 'collab_chat_b.png'))

    assert not errors, errors
    browser.close()
print('collab ui smoke ok')
