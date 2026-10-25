"""Offline, opt-in real-Edge Markdown regression (no server/provider required).

Run from the repository root: python -m tests.markdown_ui_smoke
All HTTP requests are fulfilled/aborted locally, including nested vendor assets.
The send fixture feeds byte-fragmented SSE into the app's real fetch/SSE reader.
Screenshot (also on assertion failure): .ctxagent/markdown-ui-smoke.png
"""
import copy
from pathlib import Path
from urllib.parse import unquote, urlsplit

from playwright.sync_api import sync_playwright

from ctxagent import tools
from ctxagent.config import BACKEND_BUILD, SessionConfig

ROOT = Path(__file__).resolve().parent.parent
WEB = (ROOT / 'web').resolve()
ORIGIN = 'http://127.0.0.1:8765'

# Intentionally blank lines between *every* pipe-delimited record, including
# header/delimiter: the user's spaced-row table must still be one real table.
SPACED_TABLE = '''| Feature | Status | Count |

| :--- | :---: | ---: |

| Tables | supported | 2 |

| Streaming | literal-safe | 3 |'''
NORMAL_TABLE = r'''| Text | Code |
| --- | --- |
| left\|right | `a\|b` |
| ordinary | `x_y **z**` |'''
SHOWCASE = '''# Markdown renderer regression

''' + SPACED_TABLE + '''

## Structure and inline formatting

3. third item
   - nested child
     - deep child
4. fourth item

- outer bullet
  1. nested ordered item
  2. second nested item

**bold words**, *italic words*, ~~removed words~~ and `x_y **literal**`.

> Quoted text
>
> - quoted nested item

---

[Safe documentation](https://example.invalid/docs?q=one&n=two "Documentation")

''' + NORMAL_TABLE
SECURITY = r'''Raw HTML must be literal, not executable:
<script>window.__markdownInjected += 1</script>
<img src="https://images.invalid/raw.png" onerror="window.__markdownInjected += 1">
<svg onload="window.__markdownInjected += 1"><a href="javascript:alert(1)">svg</a></svg>
<iframe srcdoc="<script>parent.__markdownInjected += 1</script>"></iframe>
<a href="javascript:window.__markdownInjected++" onclick="window.__markdownInjected++">raw link</a>

[bad](javascript:alert%281%29)
[bad mixed](JaVaScRiPt:alert%281%29)
[bad entity](javascript&#58;alert%281%29)
[bad data](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)
[bad control](java&#x09;script:alert%281%29)
[bad vbscript](vbscript:msgbox%281%29)

![remote image](https://images.invalid/tracker.png "Remote image")
![quote \\" onerror=\\"window.__markdownInjected++](https://images.invalid/quoted.png)
![data image](data:image/svg+xml;base64,PHN2ZyBvbmxvYWQ9YWxlcnQoMSk+)

[attribute boundary](https://example.invalid/ "\" onmouseover=\"window.__markdownInjected++")
[entity title](https://example.invalid/ "&quot; onclick=&quot;window.__markdownInjected++")

- [ ] literal unchecked task marker
- [x] literal checked task marker
'''
USER_TEXT = '**Do not format me** <img src=x onerror="window.__markdownInjected++">\n' + SPACED_TABLE
FENCE_BODY = '''**literal bold** _literal italic_ ~~literal strike~~
| A | B |

| --- | --- |

| x | y |
<script>window.__markdownInjected += 1</script>
`one backtick` and <tag attr="quoted"> & raw
'''
INLINE = r'''`x_y *not italic* **not bold** <tag> & literal` and ``a ` b``.

Escaped \*not emphasis\* and \`not code\` and \<safe\>.

`left|right` and **real bold** and *real italic*.
'''
BOUNDARIES = '''| Header | Value |
| --- | --- |
| first | one |

ordinary prose | not a table row

| independent | prose |

This paragraph must survive.

- list paragraph one

  list paragraph two
'''


def static_file(url):
    """Resolve exact /static/vendor/** paths; reject traversal and symlink escape."""
    parsed = urlsplit(url)
    if f'{parsed.scheme}://{parsed.netloc}' != ORIGIN:
        return None
    path = unquote(parsed.path)
    if path == '/':
        relative = 'index.html'
    elif path.startswith('/static/'):
        relative = path[len('/static/'):]
    else:
        return None
    # Backslashes are separators on Windows, but not valid URL path separators.
    if not relative or '\\' in relative or '\x00' in relative:
        return None
    candidate = (WEB / relative).resolve()
    try:
        candidate.relative_to(WEB)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def assert_safe(host):
    """Inspect parsed DOM, not just text: source must never create active HTML."""
    assert host.locator('script, iframe, object, embed, svg, math, style, link, base, meta, form, button').count() == 0
    # Read-only rendered task boxes are allowed; source-provided active controls
    # are not. Literal task markers are an equally safe representation.
    assert host.locator('input:not([type="checkbox"]), input:not([disabled])').count() == 0
    violations = host.evaluate('''root => {
      const bad = [];
      for (const node of root.querySelectorAll('*')) {
        for (const attr of node.attributes) {
          if (/^on/i.test(attr.name) || ['srcdoc', 'autofocus'].includes(attr.name))
            bad.push(node.outerHTML);
          if (['href', 'src', 'xlink:href', 'action', 'formaction'].includes(attr.name)) {
            const value = attr.value.replace(/[\\u0000-\\u0020\\u007f]+/g, '').toLowerCase();
            if (/^(javascript|vbscript|data):/.test(value)) bad.push(node.outerHTML);
          }
        }
        if (node.matches('a[target="_blank"]')) {
          const rel = new Set((node.getAttribute('rel') || '').toLowerCase().split(/\\s+/));
          if (!rel.has('noopener') || !rel.has('noreferrer')) bad.push(node.outerHTML);
        }
      }
      return bad;
    }''')
    assert not violations, violations
    # Accept image source links or literal alt/source text, never an autoload.
    assert host.locator('img, video, audio, source, picture').count() == 0


def main():
    cfg = SessionConfig(model='offline/markdown-fixture', enabled_tools=[]).to_dict()
    cfg['compaction']['document_compaction_enabled'] = False
    workspace = {'views': [], 'tokens': 0, 'budget': 24000}
    stats = {'total': 500, 'original': 500, 'saved': 0, 'saved_ratio': 0,
             'pressure': .01, 'frozen_tokens': 0, 'hot_tokens': 500,
             'workspace_tokens': 0, 'workspace': workspace,
             'effective_window': 50000, 'hot_start': 0, 'tool_overhead': 0,
             'rewrite_pending_batches': 0, 'rewrite_interval_batches': 3,
             'next_tail_ratio': 0, 'deep_ready': False}
    fixtures = {'showcase': SHOWCASE, 'security': SECURITY, 'inline': INLINE,
                'boundaries': BOUNDARIES}
    for fence in ('```', '~~~'):
        for complete in (True, False):
            key = ('backtick' if fence == '```' else 'tilde') + ('_complete' if complete else '_unfinished')
            body = FENCE_BODY + ('```nested backticks are literal\n' if fence == '~~~' else '')
            fixtures[key] = fence + 'text\n' + body + (fence + '\n' if complete else '')
    segments = [{'id': 'user-history', 'kind': 'user', 'body': USER_TEXT}]
    segments += [{'id': name, 'kind': 'assistant_text', 'body': body}
                 for name, body in fixtures.items()]
    original_segments = copy.deepcopy(segments)
    requests, external, unknown, errors, failures = [], [], [], [], []
    screenshot = ROOT / '.ctxagent' / 'markdown-ui-smoke.png'

    def serve(route):
        request = route.request
        parsed = urlsplit(request.url)
        if f'{parsed.scheme}://{parsed.netloc}' != ORIGIN:
            external.append((request.resource_type, request.url))
            return route.abort()
        path = parsed.path
        requests.append((request.method, path))
        if not path.startswith('/api/'):
            file = static_file(request.url)
            if file is None:
                if path != '/favicon.ico':
                    unknown.append((request.method, path))
                return route.fulfill(status=404, body='not found')
            return route.fulfill(path=str(file))
        if path == '/api/health':
            data = {'ok': True, 'has_key': True, 'tools': tools.catalogue(),
                    'backend_build': BACKEND_BUILD, 'accounting_version': 2,
                    'server_started_at': 0, 'server_pid': 1, 'defaults': cfg}
        elif path == '/api/chats':
            data = [{'id': 'MD', 'title': 'Offline Markdown regression'}]
        elif path == '/api/chats/MD':
            data = {'id': 'MD', 'title': 'Offline Markdown regression',
                    'config': cfg, 'segments': segments, 'usage': {}, 'stats': stats,
                    'running': False, 'compaction_log': []}
        elif path == '/api/chats/MD/context':
            data = {'segments': [], 'usage': {}, 'stats': stats, 'compaction_log': []}
        elif path == '/api/chats/MD/documents':
            data = {'workspace': workspace, 'stats': stats}
        elif path == '/api/chats/MD/browser':
            data = {'workspace': workspace}
        elif path == '/api/chats/MD/agents':
            data = {'agents': [], 'cost': {'cost': 0}}
        elif path == '/api/models':
            data = {'models': []}
        elif path in ('/api/providers', '/api/providers/check'):
            data = {'ok': True, 'allowed': [], 'total': 0, 'message': 'offline fixture'}
        elif path == '/api/providers/flex':
            data = {'known': True, 'available': False, 'candidates': []}
        else:
            unknown.append((request.method, path))
            return route.fulfill(status=404, json={'error': 'unmocked endpoint'})
        route.fulfill(json=data)

    # This independent guard check must not require the vendor to exist yet.
    assert static_file(ORIGIN + '/static/../tests/markdown_ui_smoke.py') is None
    assert static_file(ORIGIN + '/static/%2e%2e/tests/markdown_ui_smoke.py') is None
    assert static_file(ORIGIN + '/static/%5c..%5ctests/markdown_ui_smoke.py') is None
    assert static_file('https://example.invalid/static/app.js') is None
    assert static_file(ORIGIN + '/static/app.js') == WEB / 'app.js'

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel='msedge', headless=True)
        context = browser.new_context(viewport={'width': 1500, 'height': 1100}, service_workers='block')
        context.route('**/*', serve)  # No route.continue(): nothing reaches the network.
        page = context.new_page()
        page.set_default_timeout(10000)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.add_init_script('''
          window.__markdownInjected = 0;
          window.__markdownSends = [];
          const originalFetch = window.fetch.bind(window);
          window.fetch = async (url, options) => {
            const target = new URL(String(url), location.href);
            if (target.origin === location.origin && target.pathname === '/api/chats/MD/send') {
              window.__markdownSends.push(JSON.parse(options.body));
              return new Response(new ReadableStream({start(controller) {
                window.__markdownStream = controller;
              }}), {headers: {'content-type': 'text/event-stream'}});
            }
            return originalFetch(url, options);
          };
          window.__markdownEmit = event => {
            const bytes = new TextEncoder().encode('data: ' + JSON.stringify(event) + '\\n\\n');
            // Fragment UTF-8, JSON and the SSE delimiter across reader chunks.
            for (let i = 0; i < bytes.length; i += 7)
              window.__markdownStream.enqueue(bytes.slice(i, i + 7));
          };
        ''')

        def check(name, action):
            try:
                action()
                print('PASS:', name, flush=True)
            except AssertionError as exc:
                failures.append(f'{name}: {exc}')
                print('FAIL:', name, str(exc), flush=True)

        try:
            page.goto(ORIGIN + '/')
            page.wait_for_function("S.chatId === 'MD' && S.cfg !== null")
            page.wait_for_function("document.querySelectorAll('#msgwrap .assistant .bubble').length === " + str(len(fixtures)))
            page.wait_for_load_state('networkidle')
            bubbles = page.locator('#msgwrap .assistant .bubble')
            historical = {name: bubbles.nth(index) for index, name in enumerate(fixtures)}
            showcase = historical['showcase']

            def layout():
                assert page.locator('#msgwrap').evaluate("e => e.classList.contains('wrap')"), 'chat switches must preserve the message-column wrapper'
                box = page.locator('#msgwrap').bounding_box()
                assert box and box['width'] <= 781, 'message column should retain its readable maximum width'
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), 'Markdown must not widen the whole page'
            check('readable message-column layout after opening a chat', layout)

            def tables():
                assert showcase.locator('table').count() == 2, 'spaced and normal tables need real table DOM'
                spaced = showcase.locator('table').nth(0)
                assert spaced.locator('thead th').all_text_contents() == ['Feature', 'Status', 'Count']
                assert spaced.locator('tbody tr').count() == 2
                assert spaced.locator('tbody td').all_text_contents() == ['Tables', 'supported', '2', 'Streaming', 'literal-safe', '3']
                for index, alignment in enumerate(('left', 'center', 'right')):
                    for selector in ('thead th', 'tbody tr:first-child td'):
                        actual = spaced.locator(selector).nth(index).evaluate('e => getComputedStyle(e).textAlign')
                        assert actual == alignment, (index, actual, alignment)
                normal = showcase.locator('table').nth(1)
                assert normal.locator('tbody tr').count() == 2
                assert normal.locator('tbody tr').nth(0).locator('td').all_text_contents() == ['left|right', 'a|b']
                assert normal.locator('td code').all_text_contents() == ['a|b', 'x_y **z**']
            check('historical spaced/normal tables, alignment and escaped/code pipes', tables)

            def structure():
                assert showcase.locator('h1').text_content() == 'Markdown renderer regression'
                assert showcase.locator('h2').text_content() == 'Structure and inline formatting'
                assert showcase.locator('ol[start="3"]').count() == 1
                assert showcase.locator('ol[start="3"] > li').count() == 2
                assert showcase.locator('ol[start="3"] ul ul > li').text_content().strip() == 'deep child'
                assert showcase.locator('ul ol > li').count() == 2
                assert showcase.locator('strong, b').filter(has_text='bold words').count() == 1
                assert showcase.locator('em, i').filter(has_text='italic words').count() == 1
                assert showcase.locator('s, del').text_content() == 'removed words'
                assert showcase.locator('blockquote').count() == 1
                assert showcase.locator('blockquote ul li').text_content().strip() == 'quoted nested item'
                assert showcase.locator('hr').count() == 1
                link = showcase.get_by_role('link', name='Safe documentation')
                assert link.get_attribute('href') == 'https://example.invalid/docs?q=one&n=two'
                assert_safe(showcase)
            check('headings, nested/numbered lists, emphasis, quote, rule and safe links', structure)

            def inline_code():
                host = historical['inline']
                assert host.locator('code').all_text_contents() == ['x_y *not italic* **not bold** <tag> & literal', 'a ` b', 'left|right']
                assert host.locator('code *').count() == 0, 'inline code must not contain emphasis or HTML'
                assert '*not emphasis*' in host.text_content()
                assert '`not code`' in host.text_content()
                assert '<safe>' in host.text_content()
                assert host.locator('strong, b').all_text_contents() == ['real bold']
                assert host.locator('em, i').all_text_contents() == ['real italic']
            check('escaped Markdown and multi-backtick inline code remain literal', inline_code)

            def fences():
                for name in fixtures:
                    if not name.startswith(('backtick_', 'tilde_')):
                        continue
                    host = historical[name]
                    expected = FENCE_BODY + ('```nested backticks are literal\n' if name.startswith('tilde') else '')
                    assert host.locator('pre > code').count() == 1, name
                    assert host.locator('pre > code').text_content() == expected, name
                    assert host.locator('pre code *').count() == 0, name
                    assert host.locator('table, strong, b, em, i, s, del, script').count() == 0, name
                    assert_safe(host)
            check('complete AND unfinished backtick/tilde fences preserve literal source', fences)

            def boundaries():
                host = historical['boundaries']
                assert host.locator('table').count() == 1
                assert host.locator('tbody tr').count() == 1, 'blank-line normalization swallowed prose'
                text = host.text_content()
                for fragment in ('ordinary prose | not a table row', '| independent | prose |',
                                 'This paragraph must survive.', 'list paragraph one', 'list paragraph two'):
                    assert fragment in text, fragment
                assert host.locator('li p').count() == 2, 'normal blank lines in loose lists must survive'
            check('blank-line table extension does not swallow prose/list boundaries', boundaries)

            def security():
                host = historical['security']
                assert_safe(host)
                assert '<script>window.__markdownInjected += 1</script>' in host.text_content()
                assert 'remote image' in host.text_content(), 'safe image representation lost alt text'
                assert 'literal unchecked task marker' in host.text_content()
                assert 'literal checked task marker' in host.text_content()
                assert page.evaluate('window.__markdownInjected') == 0
                assert external == [], external
            check('source injection, unsafe protocols, task markers and external images inert', security)

            def plain_history():
                host = page.locator('#msgwrap .user .bubble').first
                assert host.text_content() == USER_TEXT
                assert host.locator('*').count() == 0
                assert segments == original_segments, 'fixture source mutated'
                assert page.evaluate('src => typeof md(src) === "string" && md(src) === md(src)', SPACED_TABLE)
            check('historical user stays plain text; md(src) API/source unchanged', plain_history)

            # Real click -> send() -> mocked fetch Response -> real SSE decoder ->
            # incremental assistant rendering. No direct call to its event handler.
            page.locator('#input').fill(USER_TEXT)
            page.locator('#btn-send').click()
            page.wait_for_function('!!window.__markdownStream')
            assert page.evaluate('window.__markdownSends') == [{'message': USER_TEXT}]
            assert page.locator('#msgwrap .user .bubble').last.text_content() == USER_TEXT
            assert page.locator('#msgwrap .user .bubble').last.locator('*').count() == 0
            page.evaluate('__markdownEmit({type:"assistant_start"})')
            prefix = ''
            chunks = [
                'STREAM_START café 中文 🚢\n\n',
                SPACED_TABLE + '\n\n',
                '``', '`text\n',
                'STREAM_LITERAL_1 **never bold**\n',
                '| Fence | Must |\n\n| --- | --- |\n\n| stay | literal |\n',
                '<img src="https://images.invalid/stream.png" onerror="window.__markdownInjected++">\n',
                'STREAM_LITERAL_2 <script>window.__markdownInjected++</script>\n',
                '```\n\nSTREAM_END **finished**\n\n',
                '~~~text\nSTREAM_UNFINISHED *never italic*\n' + SPACED_TABLE,
            ]
            for index, chunk in enumerate(chunks):
                prefix += chunk
                page.evaluate('text => __markdownEmit({type:"delta", text})', chunk)
                # Compare parsed/canonical HTML: browsers serialize harmless
                # entities (notably &quot; inside code text) differently from md().
                page.wait_for_function('''src => {
                  const expected = document.createElement('template');
                  expected.innerHTML = md(src);
                  return document.querySelector('#msgwrap .assistant:last-child .bubble').innerHTML === expected.innerHTML;
                }''', arg=prefix)
                stream_host = page.locator('#msgwrap .assistant .bubble').last

                def partial(index=index, stream_host=stream_host):
                    text = stream_host.text_content()
                    assert 'STREAM_START café 中文 🚢' in text, 'partial UTF-8/source lost'
                    if index >= 1:
                        assert stream_host.locator('table').count() == 1
                    if index >= 4:
                        assert 'STREAM_LITERAL_1 **never bold**' in text
                        expected_code = prefix.split('```text\n', 1)[1].split('```\n', 1)[0]
                        assert stream_host.locator('pre code').first.text_content() == expected_code, 'partial fenced source was altered/lost'
                    if index >= 7:
                        assert 'STREAM_LITERAL_2 <script>window.__markdownInjected++</script>' in text
                    assert_safe(stream_host)
                check(f'SSE partial {index + 1}/{len(chunks)}: accumulated text/safety', partial)

            def final_stream():
                host = page.locator('#msgwrap .assistant .bubble').last
                assert host.locator('pre code').count() == 2
                assert host.locator('pre code').nth(1).text_content().rstrip('\n') == ('STREAM_UNFINISHED *never italic*\n' + SPACED_TABLE)
                assert host.locator('pre code *').count() == 0
                assert host.locator('strong, b').all_text_contents() == ['finished']
                assert 'STREAM_END finished' in host.text_content()
                assert_safe(host)
            check('finished stream retains complete and unfinished code without text loss', final_stream)
            page.evaluate('''() => {
              __markdownEmit({type:'final'});
              window.__markdownStream.close();
            }''')
            page.wait_for_function("!S.runs.has('MD')")
            page.wait_for_load_state('networkidle')
            assert page.evaluate('window.__markdownInjected') == 0
            assert external == [], f'external autoload attempted: {external}'
            assert unknown == [], f'unmocked endpoints/assets: {unknown}'
            assert errors == [], errors
            assert segments == original_segments
        finally:
            screenshot.parent.mkdir(exist_ok=True)
            if not page.is_closed():
                page.evaluate('''() => {
                  const messages = document.getElementById('messages');
                  if (messages) messages.style.scrollBehavior = 'auto';
                  const showcase = document.querySelector('#msgwrap .assistant .bubble');
                  if (showcase) showcase.scrollIntoView({block: 'start', behavior: 'instant'});
                  else if (messages) messages.scrollTop = 0;
                }''')
                page.screenshot(path=str(screenshot), full_page=True)
            browser.close()
    if failures:
        raise AssertionError('Markdown UI regression failures:\n' + '\n'.join(failures))
    print('Markdown UI smoke passed: historical + byte-fragmented SSE rendering; offline only.')
    print('Screenshot:', screenshot)


if __name__ == '__main__':
    main()
