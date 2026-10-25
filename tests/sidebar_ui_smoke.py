"""Offline real-Edge sidebar hierarchy regression; no workers/provider/network calls."""
import copy
import json
from pathlib import Path

from playwright.sync_api import sync_playwright
from ctxagent import tools
from ctxagent.config import BACKEND_BUILD, SessionConfig

ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg = SessionConfig(enabled_tools=[]).to_dict()
    chats = [
        {'id': 'G', 'title': 'Nested worker', 'agent_parent': 'C1', 'agent_root': 'A'},
        {'id': 'C2', 'title': 'Second worker', 'agent_parent': 'A', 'agent_root': 'A'},
        {'id': 'C1', 'title': 'First worker', 'agent_parent': 'A', 'agent_root': 'A'},
        {'id': 'C3', 'title': 'Recovered under root', 'agent_parent': 'deleted', 'agent_root': 'A'},
        {'id': 'A', 'title': 'Main <A> & research'},
        {'id': 'B', 'title': 'Main B'},
        {'id': 'O', 'title': 'Orphan worker', 'agent_parent': 'missing', 'agent_root': 'missing'},
        {'id': 'L1', 'title': 'Cycle anchor', 'agent_parent': 'L2'},
        {'id': 'L2', 'title': 'Cycle child', 'agent_parent': 'L1'},
        {'id': 'PLAIN', 'title': 'Agent: ordinary manual chat'},
    ]
    requests, deleted, errors = [], [], []
    stats = {'total': 500, 'original': 500, 'saved': 0, 'saved_ratio': 0,
             'pressure': .01, 'frozen_tokens': 0, 'hot_tokens': 500,
             'workspace_tokens': 0, 'workspace': {'views': [], 'tokens': 0, 'budget': 24000}}

    def handler(route):
        path = route.request.url.split('http://ctx.test', 1)[-1]
        method = route.request.method
        requests.append((method, path))
        data = {}
        if path == '/api/health':
            data = {'has_key': True, 'tools': tools.catalogue(), 'backend_build': BACKEND_BUILD,
                    'accounting_version': 2, 'server_started_at': 0, 'server_pid': 1}
        elif path == '/api/chats':
            data = copy.deepcopy(chats)
        elif path.startswith('/api/chats/'):
            parts = path.split('/')
            cid = parts[3]
            row = next((c for c in chats if c['id'] == cid), {'id': cid, 'title': cid})
            if len(parts) == 4:
                if method == 'DELETE':
                    deleted.append(cid)
                    chats[:] = [c for c in chats if c['id'] != cid]
                else:
                    data = {**row, 'config': cfg, 'segments': [], 'usage': {}, 'stats': stats,
                            'delegated': bool(row.get('agent_parent')), 'running': False}
            elif parts[4] == 'context':
                data = {'segments': [], 'stats': stats, 'usage': {}, 'compaction_log': []}
            elif parts[4] == 'documents':
                data = {'workspace': stats['workspace'], 'stats': stats}
            elif parts[4] == 'browser':
                data = {'workspace': {'views': [], 'tokens': 0, 'budget': 24000}}
            elif parts[4] == 'agents':
                data = {'agents': [], 'cost': {'cost': 0}}
        elif path == '/api/models':
            data = {'models': []}
        elif path == '/api/providers':
            data = {'allowed': [], 'total': 0}
        route.fulfill(status=200, content_type='application/json', body=json.dumps(data))

    def static(route):
        path = route.request.url.split('http://ctx.test', 1)[-1].split('?')[0]
        file = ROOT / 'web' / ('index.html' if path == '/' else path.removeprefix('/static/').lstrip('/'))
        route.fulfill(path=str(file)) if file.exists() else route.fulfill(status=404)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1500, 'height': 1100})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.route('http://ctx.test/**', static)
        page.route('http://ctx.test/api/**', handler)
        page.goto('http://ctx.test/')
        page.wait_for_function("S.chatId === 'A' && S.cfg !== null")
        page.wait_for_function("document.getElementById('chattitle').textContent === 'Main <A> & research'")
        page.wait_for_load_state('networkidle')

        def row(cid):
            return page.locator(f'#chatlist [data-chat-id="{cid}"]')

        def button(cid):
            return row(cid).locator('.chat-open')

        # Latest worker must not steal startup selection. All groups default closed.
        assert row('A').is_visible() and row('B').is_visible()
        assert not row('C1').is_visible() and not row('G').is_visible()
        assert button('A').get_attribute('aria-expanded') == 'false'
        assert button('A').locator('.chat-caret').inner_text() == '\u25b8'
        assert row('A').locator('.chat-agent-count').inner_text() == '4'
        assert row('A').locator('.t').inner_text() == 'Main <A> & research'
        assert row('PLAIN').is_visible()  # Titles never determine ownership.
        assert row('O').is_visible()     # Deleted roots never hide orphan chats.
        assert row('L1').is_visible() and not row('L2').is_visible()
        (ROOT / '.ctxagent').mkdir(exist_ok=True)
        page.locator('#sidebar').screenshot(path=str(ROOT / '.ctxagent' / 'sidebar-folded.png'))

        # Clicking selected root folds locally: no network/reload/draft loss/epoch change.
        page.locator('#input').fill('Keep this unsent draft')
        page.evaluate("window.beforeFold = {epoch:S.viewEpoch, view:document.getElementById('msgwrap')}; S.runs.set('A', {})")
        before = list(requests)
        button('A').click()
        assert row('C1').is_visible() and row('C2').is_visible() and row('C3').is_visible()
        assert not row('G').is_visible()
        assert button('A').get_attribute('aria-expanded') == 'true'
        assert page.evaluate("S.viewEpoch === beforeFold.epoch && document.getElementById('msgwrap') === beforeFold.view && S.runs.has('A')")
        assert requests == before
        assert page.locator('#input').input_value() == 'Keep this unsent draft'
        button('A').click()
        assert not row('C1').is_visible()
        assert requests == before
        assert page.locator('#input').input_value() == 'Keep this unsent draft'
        page.evaluate("S.runs.delete('A')")

        # Native keyboard activation and focus survive local rerenders.
        button('A').focus()
        button('A').press('Enter')
        assert button('A').get_attribute('aria-expanded') == 'true'
        assert button('A').evaluate('(e) => e === document.activeElement')
        button('A').press('Space')
        assert button('A').get_attribute('aria-expanded') == 'false'
        button('A').press('Enter')

        # Nested agent clicks open the child and expand just its descendants.
        button('C1').click()
        page.wait_for_function("S.chatId === 'C1' && S.cfg !== null")
        assert row('G').is_visible()
        assert row('A').evaluate("e => e.classList.contains('contains-active')")
        assert button('C1').get_attribute('aria-current') == 'page'
        before = list(requests)
        button('C1').click()
        assert not row('G').is_visible() and requests == before
        button('C1').press('Enter')
        assert row('G').is_visible()
        page.evaluate('loadChats()')
        assert row('G').is_visible() and button('A').get_attribute('aria-expanded') == 'true'

        # Direct inspect/programmatic navigation reveals ancestors, not unrelated groups.
        page.evaluate("S.expandedChats.clear(); renderChats(); openChat('G')")
        page.wait_for_function("S.chatId === 'G' && S.cfg !== null")
        assert row('G').is_visible() and button('A').get_attribute('aria-expanded') == 'true'
        assert button('C1').get_attribute('aria-expanded') == 'true'
        assert button('L1').get_attribute('aria-expanded') == 'false'

        # Return to main; second click collapses, without deleting the retained draft.
        button('A').click()
        page.wait_for_function("S.chatId === 'A' && S.cfg !== null")
        assert page.locator('#input').input_value() == 'Keep this unsent draft'
        button('A').click()
        assert not row('G').is_visible() and not row('C2').is_visible()
        button('A').click()
        page.on('dialog', lambda dialog: dialog.accept())
        row('C2').locator('.x').click()
        page.wait_for_function("!S.chats.some(c => c.id === 'C2')")
        assert deleted == ['C2'] and page.evaluate('S.chatId') == 'A'
        assert button('A').get_attribute('aria-expanded') == 'true'
        assert row('A').locator('.chat-agent-count').inner_text() == '3'

        # Out-of-order list refreshes cannot replace new metadata with stale rows.
        page.evaluate("""async () => {
            const original = api, current = S.chats; let calls = 0, release;
            api = async (path, ...args) => {
                if (path !== '/api/chats') return original(path, ...args);
                if (++calls === 1) return new Promise(resolve => { release = () => resolve([{id:'stale',title:'Old response'}]); });
                return current;
            };
            try { const first = loadChats(), second = loadChats(); await second; release(); await first; }
            finally { api = original; }
        }""")
        assert page.evaluate("S.chats.some(c => c.id === 'A') && !S.chats.some(c => c.id === 'stale')")
        assert button('A').get_attribute('aria-expanded') == 'true'
        (ROOT / '.ctxagent').mkdir(exist_ok=True)
        page.locator('#sidebar').screenshot(path=str(ROOT / '.ctxagent' / 'sidebar-ui-smoke.png'))
        assert errors == [], errors
        browser.close()
    print('Sidebar UI smoke passed: default folded roots, nested/direct inspection, keyboard/focus, draft/run preservation, deletion and stale refresh guard; offline only.')


if __name__ == '__main__':
    main()
