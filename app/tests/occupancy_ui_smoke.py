"""Opt-in Edge/Playwright check: the context bar in provider tokens against the deep-compaction
limit, and settings that save on change (also while a turn runs). All endpoints are mocked."""
import copy
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from ctxagent import config, readers, tools

root = Path(__file__).resolve().parent.parent
cfg = config.SessionConfig(model='claude-opus-5-5', context_window=700_000, max_output_tokens=32_000).to_dict()
cfg['compaction']['deep_trigger_ratio'] = 0.9
health = {'ok': True, 'backend_build': config.BACKEND_BUILD, 'accounting_version': 2, 'restart_required': False,
          'has_key': True, 'tools': tools.catalogue(), 'default_tools': tools.DEFAULT_TOOLS, 'defaults': cfg,
          'static_levels': readers.STATIC_LEVELS}
# The chat the note was about: a 700k window, 90 % trigger, Claude's tokenizer counting 1.7x the
# local estimate. The bar used to read "~386k" (655k usable / 1.7) with a frozen/mutable split.
cal = 1.7
total = 120_000
stats = {'total': total, 'original': 150_000, 'saved': 30_000, 'effective_window': int(700_000 / cal),
         'pressure': total * cal / 700_000, 'frozen_tokens': 50_000, 'hot_tokens': 50_000, 'hot_start': 40,
         'tool_overhead': 15_000, 'deep_target_ratio': .1, 'deep_ready': True, 'token_calibration': cal,
         'window_tokens': 700_000, 'usable_tokens': 655_000, 'deep_active': True, 'deep_limit_tokens': 630_000,
         'used_tokens': round(total * cal), 'history_tokens': round(100_003 * cal), 'live_tokens': round(4_997 * cal),
         'tool_tokens': round(15_000 * cal), 'original_tokens': round(150_000 * cal), 'saved_tokens': round(30_000 * cal),
         'rewrite_interval_batches': 10, 'document_catalog_tokens': 0}
SYSTEM = 'You are ctxagent.'
chat = {'id': 'occ', 'title': 'Game chat', 'config': cfg, 'stats': stats,
        'usage': {'by_source': {'keepalive': {'requests': 3, 'cost': 0.0612}}},
        'segments': [{'id': 's0', 'seq': 0, 'kind': 'system', 'role': 'system', 'body': SYSTEM, 'state': 'original'}],
        'compaction_log': [], 'running': False, 'queued': []}
patches, state = [], {'running': False}

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width': 1500, 'height': 1100})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error) + ' @ ' + (error.stack or '')[:400]))

    def serve(route):
        request = route.request
        path = request.url.split('http://ctx.test', 1)[-1].split('?')[0]
        if path == '/':
            return route.fulfill(body=(root / 'web/index.html').read_text(encoding='utf-8'), content_type='text/html')
        if path.startswith('/static/'):
            file = root / 'web' / path.removeprefix('/static/')
            if not file.is_file():
                return route.fulfill(status=404)
            return route.fulfill(body=file.read_bytes(), content_type='text/javascript' if file.suffix == '.js' else 'text/css')
        if path == '/api/chats/occ/config' and request.method == 'PATCH':
            patch = json.loads(request.post_data)['config']
            patches.append(patch)
            new = copy.deepcopy(chat['config'])
            for k, v in patch.items():
                new.setdefault(k, {}).update(v) if k == 'compaction' else new.__setitem__(k, v)
            chat['config'] = new
            steered = new['compaction'].get('rewrite_policy') == 'agent'
            system = SYSTEM + ('\n\nManaging your own context' if steered else '')
            return route.fulfill(json={'config': new, 'pending': state['running'], 'stats': stats, 'system': system})
        data = {'/api/health': health, '/api/chats': [{'id': 'occ', 'title': chat['title']}],
                '/api/chats/occ': chat, '/api/models': {'models': []},
                '/api/chats/occ/context': {'usage': chat['usage'], 'stats': stats, 'segments': [], 'compaction_log': [],
                                           'config': chat['config']['compaction']},
                '/api/providers': {'allowed': [], 'total': 0},
                '/api/providers/check': {'ok': True, 'message': 'Synthetic test'}}.get(path, {})
        route.fulfill(json=data)

    page.route('**/*', serve)
    page.goto('http://ctx.test/')
    page.wait_for_function("document.getElementById('chattitle').textContent === 'Game chat'")
    page.locator('[data-tab="context"]').click()
    page.wait_for_timeout(200)
    # Provider tokens against the limit: 204k of 630k (90 % of 700k), not "/ 386k".
    assert page.locator('#tokpill').inner_text() == '204.0k / 630.0k', page.locator('#tokpill').inner_text()
    assert page.locator('#s-limit').inner_text() == '630.0k (90% of the window)'
    assert page.locator('#s-window').inner_text() == '700.0k'
    assert page.locator('#s-total').inner_text().startswith('204.0k')
    assert '426.0k free' in page.locator('#bigbar').inner_text(), page.locator('#bigbar').inner_text()
    assert '× 1.70' in page.locator('#s-ctx-note').inner_text()
    assert '45.0k of the window stay free' in page.locator('#s-ctx-note').inner_text()
    pane = page.locator('#pane-context').inner_text().lower()
    assert 'frozen' not in pane.split('segments')[0] and 'mutable' not in pane, pane[:400]
    page.locator('#pane-context').screenshot(path=str(root / '.ctxagent/occupancy-panel.png'))

    # Settings save when a value is committed; no Save buttons.
    page.locator('[data-tab="settings"]').click()
    assert page.locator('#btn-savecfg').count() == 0 and page.locator('#btn-savecmp').count() == 0
    page.fill('#f-maxout', '64000')
    assert not patches                                  # typing alone does not save
    page.press('#f-maxout', 'Tab')
    page.wait_for_function("document.getElementById('cfg-state').textContent === 'Saved'")
    assert patches[-1]['max_output_tokens'] == 64000
    # While the agent works: saved, applied at its next model request.
    state['running'] = True
    page.select_option('#f-reasoning', 'low')
    page.wait_for_function("document.getElementById('cfg-state').textContent.includes('next model request')")
    assert patches[-1]['reasoning_effort'] == 'low'
    page.locator('#pane-settings .pane-actions').screenshot(path=str(root / '.ctxagent/settings-pending.png'))
    page.locator('[data-tab="compaction"]').click()
    # Self-steering on: the chat's System instructions box shows the new section at once.
    box = page.locator('#msgwrap .notice.system-prompt')
    assert 'Managing your own context' not in box.inner_text()
    state['running'] = False
    page.locator('#c-selfsteer').check()
    page.wait_for_function("document.querySelector('#msgwrap .notice.system-prompt').textContent.includes('Managing your own context')")
    state['running'] = True
    page.fill('#c-cool', '8')
    page.press('#c-cool', 'Tab')
    page.wait_for_function("document.getElementById('cmp-state').textContent.includes('next model request')")
    assert patches[-1]['compaction']['deep_cooldown_turns'] == 8
    # The token-threshold hint is gone; the cache keep-alive is a switch that saves on change.
    assert page.locator('#c-newtokens').count() == 0
    assert page.locator('#c-keepalive').is_checked()
    n = len(patches)
    page.locator('#c-keepalive').uncheck()
    for _ in range(50):
        if len(patches) > n:
            break
        page.wait_for_timeout(100)
    assert patches[-1]['compaction']['cache_keepalive'] is False, patches[-1]
    assert '(3)' in page.locator('#u-keepalive-cost').text_content(), page.locator('#u-keepalive-cost').text_content()
    assert not errors, errors
    print('occupancy UI smoke passed: provider-token bar against the deep limit, settings saved on change')
    browser.close()
