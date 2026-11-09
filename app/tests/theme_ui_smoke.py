"""Offline real-Edge check of the UI integration contract; APIs mocked, no model calls.

Covers what other components rely on: theme tokens and window.PiTheme, the
top-bar mount points (#open-settings, #profile-slot), the optional settings
bundle, and the pi:chat-loaded / pi:chat-config-changed / pi:keys-changed events.
Run: python -m tests.theme_ui_smoke
"""
import copy
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from ctxagent import tools
from ctxagent.config import BACKEND_BUILD, SessionConfig

ROOT = Path(__file__).resolve().parent.parent
THEMES = ['light', 'dark', 'contrast', 'paper', 'nord', 'ember']
TOKENS = ['--bg', '--bg-elev', '--panel', '--fg', '--fg-muted', '--accent', '--accent-fg', '--border',
          '--danger', '--warn', '--ok', '--code-bg', '--radius', '--font', '--font-mono']


def main():
    cfg = SessionConfig(model='test/model', enabled_tools=['read_file']).to_dict()
    health = {'ok': True, 'has_key': False, 'tools': tools.catalogue(), 'backend_build': BACKEND_BUILD,
              'accounting_version': 2, 'server_started_at': 0, 'server_pid': 1}
    requests, errors = [], []

    def handler(route):
        path = route.request.url.split('http://ctx.test', 1)[-1].split('?')[0]
        requests.append((route.request.method, path))
        data = {}
        if path == '/api/health':
            data = health
        elif path == '/api/chats':
            data = [{'id': 'A', 'title': 'Contract chat'}, {'id': 'B', 'title': 'Second chat'}]
        elif path in ('/api/chats/A', '/api/chats/B'):
            cid = path.rsplit('/', 1)[1]
            data = {'id': cid, 'title': 'Contract chat' if cid == 'A' else 'Second chat', 'config': copy.deepcopy(cfg),
                    'segments': [], 'usage': {}, 'stats': None, 'compaction_log': []}
        elif path.endswith('/context'):
            data = {'segments': [], 'stats': None, 'usage': {}, 'compaction_log': []}
        elif path.endswith('/documents'):
            data = {'workspace': {'views': [], 'tokens': 0, 'budget': 24000}}
        elif path.endswith('/browser'):
            data = {'views': [], 'tokens': 0, 'max_tokens': 24000, 'note_count': 0}
        elif path == '/api/models':
            data = {'models': []}
        elif path == '/api/providers':
            data = {'allowed': [], 'total': 0}
        route.fulfill(status=200, content_type='application/json', body=json.dumps(data))

    def static(route):
        path = route.request.url.split('http://ctx.test', 1)[-1].split('?')[0]
        # app.js's contract is checked in isolation: the settings bundle is withheld
        # (its 404 must be harmless); tests/settings_ui_smoke.py covers the modal.
        if path in ('/static/settings.js', '/static/settings.css'):
            route.fulfill(status=404); return
        file = ROOT / 'web' / ('index.html' if path == '/' else path.removeprefix('/static/').lstrip('/'))
        route.fulfill(path=str(file)) if file.is_file() else route.fulfill(status=404)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel='msedge', headless=True)
        context = browser.new_context(viewport={'width': 1400, 'height': 900}, color_scheme='light')
        # The saved theme must already be active when the first stylesheet applies.
        context.add_init_script("""
          try { if (!sessionStorage.getItem('seeded')) { localStorage.setItem('pi.theme', 'paper'); sessionStorage.setItem('seeded', '1'); } } catch (e) {}
          window.__loaded = [];
          window.addEventListener('pi:chat-loaded', e => window.__loaded.push(e.detail && e.detail.id));
          document.addEventListener('readystatechange', () => {
            if (document.readyState === 'interactive' && !window.__firstTheme)
              window.__firstTheme = document.documentElement.getAttribute('data-theme');
          });
        """)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.route('http://ctx.test/**', static)
        page.route('http://ctx.test/api/**', handler)
        page.goto('http://ctx.test/')
        page.wait_for_function("S.chatId === 'A' && S.cfg !== null")

        # Theme registry and persistence.
        assert page.evaluate('window.__firstTheme') == 'paper'
        listed = page.evaluate('PiTheme.list()')
        assert [t['id'] for t in listed] == ['system'] + THEMES, listed
        assert all(isinstance(t['dark'], bool) and t['label'] for t in listed), listed
        assert page.evaluate('PiTheme.current()') == 'paper'
        assert page.locator('#theme-select').input_value() == 'paper'
        for theme in THEMES:
            assert page.evaluate(f"PiTheme.apply('{theme}')") == theme
            assert page.evaluate("document.documentElement.dataset.theme") == theme
            values = page.evaluate("names => names.map(n => getComputedStyle(document.documentElement).getPropertyValue(n).trim())", TOKENS)
            assert all(values), (theme, dict(zip(TOKENS, values)))
            dark = next(t['dark'] for t in listed if t['id'] == theme)
            assert page.evaluate("getComputedStyle(document.documentElement).colorScheme") == ('dark' if dark else 'light'), theme
        assert page.evaluate("PiTheme.apply('no-such-theme')") is None
        assert page.evaluate("localStorage.getItem('pi.theme')") == 'ember'
        page.select_option('#theme-select', 'system')
        assert page.evaluate("PiTheme.current()") == 'system'
        assert page.evaluate("document.documentElement.dataset.theme") == 'light'
        page.emulate_media(color_scheme='dark')
        page.wait_for_function("document.documentElement.dataset.theme === 'dark'")
        page.emulate_media(color_scheme='light')
        page.wait_for_function("document.documentElement.dataset.theme === 'light'")
        page.reload()
        page.wait_for_function("S.chatId === 'A' && S.cfg !== null")
        assert page.evaluate("PiTheme.current()") == 'system'

        # Mount points and the optional settings bundle.
        gear = page.locator('#open-settings')
        assert gear.get_attribute('title') == 'Settings' and gear.locator('svg').count() == 1
        assert page.locator('#profile-slot').count() == 1
        assert page.locator('link[href="/static/settings.css"]').count() == 1
        settings_js = page.locator('script[src="/static/settings.js"]')
        assert settings_js.count() == 1 and settings_js.get_attribute('defer') is not None
        scripts = page.evaluate("[...document.scripts].map(s => s.getAttribute('src')).filter(Boolean)")
        assert scripts.index('/static/app.js') < scripts.index('/static/settings.js'), scripts

        # Without settings.js the gear opens the per-chat Model tab...
        page.evaluate("setPanelOpen(false)")
        gear.click()
        page.wait_for_function("document.querySelector('#pane-settings').classList.contains('active')")
        assert 'panel-open' in page.locator('#app').get_attribute('class')
        # ...and a later listener that calls preventDefault() takes over.
        page.locator('[data-tab="context"]').click()
        page.evaluate("document.getElementById('open-settings').addEventListener('click', e => { e.preventDefault(); window.__custom = true; })")
        gear.click()
        page.wait_for_timeout(100)
        assert page.evaluate('window.__custom') is True
        assert page.locator('#pane-context').evaluate("e => e.classList.contains('active')")

        # No key: friendly empty state that points to Settings.
        assert page.locator('#banner').get_by_role('button', name='Add a key in Settings').count() == 1
        assert page.locator('#msgwrap .key-callout').count() == 1

        # Events: chat-loaded on load and switch; config-changed re-fetches only config.
        assert page.evaluate('window.__loaded')[-1] == 'A'
        page.evaluate("openChat('B')")
        page.wait_for_function("window.__loaded[window.__loaded.length - 1] === 'B'")
        cfg['model'] = 'changed/elsewhere'
        before = len(requests)
        page.evaluate("window.dispatchEvent(new CustomEvent('pi:chat-config-changed', {detail: {chatId: 'B'}}))")
        page.wait_for_function("document.getElementById('f-model').value === 'changed/elsewhere'")
        assert ('GET', '/api/chats/B') in requests[before:]
        assert page.locator('#modelpill').inner_text().startswith('elsewhere') and page.locator('#modelpill').get_attribute('title').startswith('changed/elsewhere')
        # Keys added elsewhere: banner and empty-state prompt disappear.
        health['has_key'] = True
        page.evaluate("window.dispatchEvent(new CustomEvent('pi:keys-changed'))")
        page.wait_for_function("!document.querySelector('#banner .banner') && !document.querySelector('#msgwrap .key-callout')")
        assert page.evaluate("typeof PiApp.openPanelTab === 'function' && PiApp.chatId === 'B'")

        (ROOT / '.ctxagent').mkdir(exist_ok=True)
        page.screenshot(path=str(ROOT / '.ctxagent' / 'theme-ui-smoke.png'))
        assert not errors, errors
        browser.close()
    print('Theme/contract UI smoke passed: tokens in all themes, PiTheme API + persistence + system, '
          'mount points, settings fallback, chat-loaded/config-changed/keys-changed events; offline only.')


if __name__ == '__main__':
    main()
