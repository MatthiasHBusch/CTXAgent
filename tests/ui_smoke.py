"""Headless UI check with synthetic data, no live chat edits or API calls."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from ctxagent import accounting, config, tools, readers
from tests.test_cache_accounting import RAW, RATES

root = Path(__file__).resolve().parent.parent
usage = {'cost': 1, 'prompt_tokens': 1000, 'cache_read': 0}
accounting.record(usage, RAW, model='test-model', provider='test-provider', price={'rates': RATES},
                  diagnostic={'reasons': ['synthetic demonstration'], 'shared_prefix_tokens_estimate': 9000})
cfg = config.SessionConfig().to_dict()
health = {'ok': True, 'backend_build': config.BACKEND_BUILD, 'accounting_version': 2, 'restart_required': False, 'has_key': True, 'tools': tools.catalogue(), 'default_tools': tools.DEFAULT_TOOLS,
          'defaults': cfg, 'static_levels': readers.STATIC_LEVELS}
stats = {'total': 11000, 'original': 15000, 'saved': 4000, 'effective_window': 100000, 'pressure': .11, 'frozen_tokens': 10000, 'hot_tokens': 1000, 'hot_zone_budget': 10000, 'hot_start': 5, 'tool_overhead': 200, 'next_tail_ratio': .25, 'deep_ready': True}
context = {'usage': usage, 'stats': stats, 'segments': [], 'compaction_log': [], 'config': cfg['compaction']}
chat = {'id': 'ui-test', 'title': 'Cost accounting visual test', 'config': cfg, 'segments': [],
        'usage': usage, 'stats': stats, 'compaction_log': []}

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width': 1500, 'height': 1100})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    def serve(route):
        path = route.request.url.split('http://ctx.test', 1)[-1]
        if path == '/':
            return route.fulfill(body=(root / 'web/index.html').read_text(encoding='utf-8'), content_type='text/html')
        if path.startswith('/static/'):
            file = root / 'web' / path.removeprefix('/static/')
            if not file.resolve().is_relative_to((root/'web').resolve()) or not file.is_file():
                return route.fulfill(status=404)
            return route.fulfill(body=file.read_bytes(), content_type='text/javascript' if file.suffix == '.js' else 'text/css')
        data = {'/api/health': health, '/api/chats': [{'id': 'ui-test', 'title': chat['title']}],
                '/api/chats/ui-test': chat, '/api/chats/ui-test/context': context,
                '/api/models': {'models': []}, '/api/providers': {'allowed': [], 'total': 0},
                '/api/providers/check': {'ok': True, 'message': 'Synthetic test'}}.get(path, {})
        route.fulfill(json=data)
    page.route('**/*', serve)
    page.goto('http://ctx.test/')
    page.locator('#u-input-cost').wait_for()
    page.wait_for_timeout(500)
    page.locator('[data-tab="context"]').click()
    page.wait_for_timeout(300)
    assert '$0.005000' in page.locator('#u-input-cost').inner_text()
    assert '$0.015000' in page.locator('#u-reason-cost').inner_text()
    assert '33%' in page.locator('#u-reason-cost').inner_text()
    assert '70%' in page.locator('#u-recent').inner_text()
    dest = root / '.ctxagent/ui-cost-smoke.png'
    page.screenshot(path=str(dest), full_page=True)
    assert not errors, errors
    health['backend_build'] = 'old-backend'
    page.reload()
    page.wait_for_timeout(300)
    assert 'BACKEND RESTART REQUIRED' in page.locator('#backend-status').inner_text()
    page.screenshot(path=str(root / '.ctxagent/ui-restart-warning.png'), full_page=True)
    print('UI smoke passed; screenshot:', dest)
    browser.close()
