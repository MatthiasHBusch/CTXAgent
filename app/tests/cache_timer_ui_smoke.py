"""Opt-in Edge/Playwright check of the cache timer beside the send button.
All HTTP endpoints are mocked. No model request. SHOT_DIR=<folder> keeps screenshots of the
composer in each state.
"""
import copy
import os
import re
import time
from pathlib import Path
from playwright.sync_api import sync_playwright
from ctxagent import config, tools

root = Path(__file__).resolve().parent.parent
shots = Path(os.environ['SHOT_DIR']) if os.environ.get('SHOT_DIR') else None
cfg = config.SessionConfig(model='anthropic/claude-test', enabled_tools=['read_file'], compaction=config.CompactionConfig(),
                           reasoning_effort='high', workdirs=['C:\\work']).to_dict()
stats = dict(total=264546, original=333667, saved=69121, effective_window=379000, frozen_tokens=54044, hot_tokens=205678,
             hot_start=133, hot_zone_budget=10000, tool_overhead=4824, pressure=264546/379000, deep_target_ratio=0.1,
             deep_ready=True, rewrite_interval_batches=2, review_candidate_tokens=190000)


def clock(age, queued=0, **more):
    return {'warm_at': time.time() - age, 'ttl': 300, 'step': 270, 'max': 13, 'queued': queued, 'refreshable': True,
            'why_not': None, 'note': None, 'now': time.time(), **more}


chats = {key: dict(id=key, title='Chat ' + key, config=copy.deepcopy(cfg), usage={}, stats=copy.deepcopy(stats),
                   segments=[], compaction_log=[]) for key in ('A', 'B')}
chats['A']['cache'] = clock(48)                     # 4:12 left
chats['B']['cache'] = None                          # no request yet: no chip
health = dict(ok=True, backend_build=config.BACKEND_BUILD, accounting_version=2, restart_required=False, has_key=True,
              defaults=cfg, default_tools=tools.DEFAULT_TOOLS, tools=tools.catalogue())

with sync_playwright() as pw:
    browser = pw.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width': 1500, 'height': 1000})
    errors, posted = [], []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.add_init_script('''
      const originalFetch = window.fetch.bind(window);
      window.streams = {};
      window.fetch = async (url, opts) => {
        if (String(url).endsWith('/send')) {
          const id = String(url).split('/')[3];
          return new Response(new ReadableStream({start(c) { window.streams[id] = c; }}),
            {headers:{'content-type':'text/event-stream'}});
        }
        return originalFetch(url, opts);
      };
      window.emit = (id, event) => window.streams[id].enqueue(
        new TextEncoder().encode('data: ' + JSON.stringify(event) + '\\n\\n'));
      window.endStream = id => {window.streams[id].close(); delete window.streams[id];};
    ''')

    def serve(route):
        path = route.request.url.split('http://ctx.test', 1)[-1]
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
            c = chats[path.split('/')[3]]
            if path.endswith('/cache/refresh'):
                delta = route.request.post_data_json['delta']
                posted.append(delta)
                c['cache'] = {**c['cache'], 'queued': max(0, min(13, c['cache']['queued'] + delta)), 'now': time.time()}
                data = c['cache']
            elif path.endswith('/cache'): data = {**c['cache'], 'now': time.time(), 'usage': {}}
            elif path.endswith('/context'): data = {**c, 'config': c['config']['compaction']}
            elif path.endswith('/stop'): data = {'ok': True}
            else: data = {**c, 'cache': c['cache'] and {**c['cache'], 'now': time.time()}}
        else: data = {}
        route.fulfill(json=data)

    page.route('**/*', serve)
    chats['A']['cache'] = clock(48)                 # counted from now, not from the browser's start
    page.goto('http://ctx.test/')
    page.wait_for_function("S.chatId === 'A' && S.cfg !== null")
    chip, label = page.locator('#btn-cache'), page.locator('#cache-left')
    step = [0]

    def shot(name):
        step[0] += 1
        if shots:
            shots.mkdir(parents=True, exist_ok=True)
            page.locator('#composer').screenshot(path=str(shots / f'{step[0]}_{name}.png'))

    def state(data):
        page.evaluate("d => setCacheClock('A', d)", data)

    # The countdown, from the chat as it was opened.
    page.wait_for_function("!document.querySelector('#btn-cache').hidden")
    assert re.fullmatch(r'4:[01]\d', label.inner_text()), label.inner_text()
    assert chip.is_enabled() and re.search(r'expires in 4:[01]\d\. Click: keep it 4:30 longer', chip.get_attribute('title'))
    box, send = chip.bounding_box(), page.locator('#btn-send').bounding_box()
    assert 0 < send['x'] - (box['x'] + box['width']) < 20, (box, send)      # directly left of Send
    shot('counting')

    # Each click queues one more refresh; right-click and Shift+click take one back.
    chip.click()
    page.wait_for_function("document.querySelector('#cache-left').textContent.includes('\\u21bb1')")
    assert re.fullmatch(r'8:[34]\d \u21bb1', label.inner_text()), label.inner_text()
    chip.click()
    page.wait_for_function("document.querySelector('#cache-left').textContent.includes('\\u21bb2')")
    assert re.fullmatch(r'13:[01]\d \u21bb2', label.inner_text()), label.inner_text()
    assert re.search(r'2 refreshes queued, the next in 3:[34]\d', chip.get_attribute('title')), chip.get_attribute('title')
    shot('two_queued')
    chip.click(button='right')
    page.wait_for_function("document.querySelector('#cache-left').textContent.includes('\\u21bb1')")
    chip.click(modifiers=['Shift'])
    page.wait_for_function("!document.querySelector('#cache-left').textContent.includes('\\u21bb')")
    chip.click(button='right')                                # nothing queued: no request
    page.wait_for_timeout(200)
    assert posted == [1, 1, -1, -1], posted

    # The last minute, and after it.
    state(clock(255))
    assert 'warn' in chip.get_attribute('class') and re.fullmatch(r'0:4\d', label.inner_text())
    shot('last_minute')
    state(clock(400))
    assert label.inner_text() == 'expired' and chip.is_disabled()
    assert 'writes the whole prompt again' in chip.get_attribute('title')
    shot('expired')
    state(clock(60, refreshable=False, why_not='A refresh repeats the last request, and there has been none since the app started.'))
    assert chip.is_disabled() and 'since the app started' in chip.get_attribute('title')
    shot('not_refreshable')
    state(clock(60, note='cache not hit: wrote 9 of 10 prompt tokens'))
    assert 'Last refresh: cache not hit' in chip.get_attribute('title')

    # A request of a running turn restarts the countdown; its usage event brings the queue.
    state(clock(100))
    assert re.fullmatch(r'3:[12]\d', label.inner_text()), label.inner_text()
    page.locator('#input').fill('go')
    page.locator('#btn-send').click()
    page.wait_for_function("!!window.streams.A")
    page.evaluate("emit('A', {type: 'assistant_start'})")
    page.wait_for_function("/^(5:00|4:5\\d)$/.test(document.querySelector('#cache-left').textContent)")
    page.evaluate("d => emit('A', {type: 'usage', usage: {}, cache: d})", clock(2, queued=1))
    page.wait_for_function("document.querySelector('#cache-left').textContent.includes('\\u21bb1')")
    shot('during_a_turn')
    page.evaluate("endStream('A')")
    page.wait_for_function("!S.streaming")

    # A chat without a request has no chip; coming back shows it again.
    page.evaluate("openChat('B')")
    page.wait_for_function("S.chatId === 'B' && S.cfg !== null")
    assert chip.is_hidden()
    page.evaluate("openChat('A')")
    page.wait_for_function("!document.querySelector('#btn-cache').hidden")
    assert not errors, errors
    browser.close()
print('cache timer UI smoke: ok')
