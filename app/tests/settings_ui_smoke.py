"""Settings menu + profile slot against a real temporary server (Edge/Playwright).

Starts `ctxagent.server` on a spare port with a temporary data folder and the
file key store, so neither the user's chats nor the OS credential store are
touched. Provider variables are blanked (a .env next to the app cannot fill
them) and the model/provider catalogues are stubbed in the browser: no network,
no model calls. The key entered is a dummy.

    python -B -m tests.settings_ui_smoke [screenshot-dir]
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DUMMY_KEY = 'sk-or-v1-uitest-0000000000000000abcd'


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def main(out_dir=None):
    out = Path(out_dir or tempfile.mkdtemp(prefix='pi-settings-shots-'))
    out.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get('PI_SMOKE_PORT') or free_port())
    if port == 8765:
        raise SystemExit('refusing to use the live server port 8765')
    data = tempfile.TemporaryDirectory(prefix='pi-settings-data-')
    env = {**os.environ, 'CTXAGENT_DATA': data.name, 'CTXAGENT_PORT': str(port),
           'CTXAGENT_KEYSTORE': 'file', 'PYTHONDONTWRITEBYTECODE': '1',
           # Present-but-empty: load_env() uses setdefault, so a .env cannot fill these.
           'OPENROUTER_API_KEY': '', 'ANTHROPIC_API_KEY': '', 'ANTHROPIC_WORKSPACE_ID': '',
           'PYTHONPATH': str(ROOT)}
    # cwd = the temp folder, so a .env in the current directory is not read either.
    server = subprocess.Popen([sys.executable, '-B', '-m', 'ctxagent.server'], cwd=data.name,
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f'http://127.0.0.1:{port}'
    try:
        for _ in range(100):
            try:
                if httpx.get(base + '/api/health', timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            raise RuntimeError('server did not start: ' + server.stderr.read().decode(errors='replace')[-2000:])
        run_browser(base, out)
        # The key went to the temp file store, masked everywhere else.
        listing = httpx.get(base + '/api/settings/keys').text
        assert DUMMY_KEY not in listing and 'sk-or-…abcd' in listing, listing
        assert DUMMY_KEY in (Path(data.name) / 'credentials.json').read_text(encoding='utf-8')
        for name in ('settings.json', 'profiles.json', 'keys.json'):
            path = Path(data.name) / name
            assert not path.exists() or DUMMY_KEY not in path.read_text(encoding='utf-8'), name
        print('settings UI smoke passed; screenshots in', out)
    finally:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()
        data.cleanup()


def run_browser(base, out):
    models = {'models': [{'id': m, 'name': m, 'provider': 'openrouter', 'context_length': 1_000_000,
                          'prompt_price': None, 'completion_price': None, 'input_modalities': ['text']}
                         for m in ('anthropic/claude-opus-5.5', 'anthropic/claude-sonnet-5.5',
                                   'anthropic/claude-fable-5.1', 'openai/gpt-6.1-sol', 'openai/gpt-6-astra',
                                   'openai/gpt-6.1-terra', 'google/gemini-4-argon')]}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        page.set_default_timeout(10000)
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
        # Offline catalogues (the real ones are public network calls).
        page.route('**/api/models', lambda r: r.fulfill(json=models))
        page.route('**/api/providers', lambda r: r.fulfill(json={'total': 0, 'allowed': [], 'regions': []}))
        page.route('**/api/providers/check', lambda r: r.fulfill(json={'ok': True, 'message': 'stubbed'}))
        page.route('**/api/providers/flex', lambda r: r.fulfill(json={'available': False, 'known': False, 'candidates': []}))
        page.goto(base + '/')
        page.wait_for_selector('#profile-slot .pi-profile-btn:not([hidden])')

        # First run: no key, banner offers Settings.
        page.wait_for_selector('#banner .pi-banner-action')
        page.click('#banner .pi-banner-action')
        page.wait_for_selector('#pi-settings:not([hidden]) [data-key="OPENROUTER_API_KEY"]')
        assert page.get_attribute('#pi-tab-keys', 'aria-selected') == 'true'
        page.screenshot(path=str(out / '1-keys-empty.png'))

        card = page.locator('[data-key="OPENROUTER_API_KEY"]')
        card.get_by_role('button', name='Add key').click()
        field = card.locator('input[type="password"]')
        assert field.get_attribute('autocomplete') == 'off'
        field.fill(DUMMY_KEY)
        card.get_by_role('button', name='Save', exact=True).click()
        page.wait_for_selector('[data-key="OPENROUTER_API_KEY"] .pi-masked')
        assert page.inner_text('[data-key="OPENROUTER_API_KEY"] .pi-masked') == 'sk-or-…abcd'
        assert DUMMY_KEY not in page.content()
        assert page.inner_text('#banner').strip() == ''
        page.screenshot(path=str(out / '2-keys-saved.png'))

        # Remove asks in-page first; cancel keeps it.
        page.locator('[data-key="OPENROUTER_API_KEY"]').get_by_role('button', name='Remove').click()
        page.wait_for_selector('.pi-confirm')
        page.keyboard.press('Escape')  # cancels the confirmation, not the dialog
        assert page.is_visible('#pi-settings') and not page.is_visible('.pi-confirm')

        # Profiles: list, then the editor of a built-in (read-only) and a duplicate.
        page.click('#pi-tab-profiles')
        page.wait_for_selector('text=Opus 5.5 (self-steered, 700k)')
        page.screenshot(path=str(out / '3-profiles.png'))
        page.locator('.pi-card', has_text='Sonnet 5.5 (self-steered, 1M)').get_by_role('button', name='Duplicate').click()
        page.wait_for_selector('#pi-f-name')
        assert page.input_value('#pi-f-name').startswith('Sonnet 5.5 (self-steered, 1M) copy')
        assert page.is_checked('input[name="pi-summary-choice"][value="same"]')
        assert page.is_disabled('#pi-f-compaction-summary_pricing_mode')
        page.fill('#pi-f-max_output_tokens', '12000')
        # xhigh is offered for the chat model, not for the summariser.
        efforts = lambda sel: page.eval_on_selector(sel, 'el => [...el.options].map(o => o.value)')
        assert efforts('#pi-f-reasoning_effort')[0] == 'xhigh'
        assert 'xhigh' not in efforts('#pi-f-compaction-summary_reasoning')
        page.get_by_role('button', name='Save profile').click()
        page.wait_for_selector('text=Saved. Existing chats keep their own settings.')
        page.screenshot(path=str(out / '4-profile-editor.png'), full_page=False)
        page.get_by_role('button', name='← All profiles').click()

        # Keyboard: arrow keys move between sections; Escape closes and restores focus.
        page.focus('#pi-tab-profiles')
        page.keyboard.press('ArrowDown')
        assert page.get_attribute('#pi-tab-newchat', 'aria-selected') == 'true'
        page.wait_for_selector('#pi-default-profile')
        page.click('#pi-tab-appearance')
        page.wait_for_selector('#pi-zoom')
        page.screenshot(path=str(out / '5-appearance.png'))
        # Insights: off by default; the switch is an app preference.
        page.click('#pi-tab-insights')
        page.wait_for_selector('#pi-insights')
        assert not page.is_checked('#pi-insights')
        page.check('#pi-insights')
        page.wait_for_selector('text=Recording from the next request on.')
        assert httpx.get(base + '/api/settings/insights').json()['enabled'] is True
        page.screenshot(path=str(out / '5b-insights.png'))
        page.uncheck('#pi-insights')
        page.wait_for_selector('text=Recording stopped.')
        page.click('#pi-tab-about')
        page.wait_for_selector('text=Data folder')
        page.screenshot(path=str(out / '6-about.png'))
        page.keyboard.press('Escape')
        assert not page.is_visible('#pi-settings')

        # Profile slot: the first chat has no profile; apply one, then modify it in the chat.
        slot = page.locator('#profile-slot .pi-profile-btn')
        assert 'No profile' in slot.inner_text()
        slot.click()
        page.get_by_role('menuitem', name='Apply another profile…').click()
        page.select_option('.pi-profile-menu select', 'builtin-terra-6-1')
        page.locator('.pi-profile-menu').get_by_role('button', name='Apply').click()
        page.wait_for_function("document.querySelector('#profile-slot').innerText.includes('Terra 6.1')")
        assert '(modified)' not in slot.inner_text()
        page.wait_for_function("document.querySelector('#f-model').value === 'openai/gpt-6.1-terra'")
        page.click('[data-tab="settings"]')
        page.fill('#f-maxout', '9000')
        page.press('#f-maxout', 'Tab')
        page.wait_for_function("document.querySelector('#profile-slot').innerText.includes('(modified)')")
        slot.click()
        page.wait_for_selector('.pi-profile-menu:not([hidden])')
        page.screenshot(path=str(out / '7-profile-slot-menu.png'))
        page.get_by_role('menuitem', name='Reset chat to profile').click()
        page.locator('.pi-profile-menu .pi-confirm').get_by_role('button', name='Reset').click()
        page.wait_for_function("!document.querySelector('#profile-slot').innerText.includes('(modified)')")
        page.wait_for_function("document.querySelector('#f-maxout').value === '32000'")
        # Known checkout artifact, not this feature: with core.autocrlf=true the vendored
        # markdown-it file gets CRLF line endings and no longer matches its SRI hash.
        sri = [e for e in errors if 'integrity' in e and 'markdown-it' in e]
        if sri:
            print('WARNING (pre-existing): vendored markdown-it blocked by SRI (CRLF checkout)')
        errors = [e for e in errors if e not in sri]
        assert not errors, errors
        browser.close()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
