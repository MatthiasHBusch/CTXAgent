"""Manual offline real-Edge/Chromium renderer smoke (not part of run_all).

Run from the repository root: python -B -m tests.browser_render_smoke
Requires Playwright, a usable Edge/Chromium installation, and Pillow. No model
or public page is contacted: the ORIGINAL backend route guard runs, but its
allowed Route.fetch/continue_ calls receive fixed in-memory responses instead
of sending requests. Synthetic public DNS is supplied only for fixture.test.
Browser launch remains real, with background networking disabled and external
host resolution disabled as additional defense (not an OS network sandbox).
Only the inspection PNG survives; chat, snapshots and artifacts are temporary.
"""
from __future__ import annotations

import asyncio
import base64
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import socket
import tempfile
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import urlsplit

import httpx
from PIL import Image
from playwright.async_api import BrowserType, Route

from ctxagent import agent, artifacts, browser_tools, browsing, config, llm, store
from ctxagent.browser_workspace import BrowserWorkspace
from ctxagent.config import SessionConfig
from ctxagent.segments import Conversation, K_MEDIA, K_USER

URL = 'https://fixture.test/render'
MARKER = 'INLINE_JS_EXECUTED_OFFLINE_7D39'
OUTPUT = Path(__file__).resolve().parents[1] / '.ctxagent' / 'browser-render-smoke.png'
WRITE_METHODS = {'POST', 'PUT', 'PATCH', 'DELETE'}
HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Offline real-browser renderer smoke</title>
<style>
html,body { margin:0; background:#edf2f7; font-family:Arial,sans-serif; }
body { height:2400px; }
h1 { position:absolute; left:40px; top:20px; margin:0; color:#16324f; }
#status { position:absolute; left:40px; top:85px; }
#network { position:absolute; left:40px; top:120px; }
#canvas-top { position:absolute; left:40px; top:180px; }
#lower-label { position:absolute; left:40px; top:975px; font-size:24px; }
#canvas-lower { position:absolute; left:40px; top:1040px; }
</style></head><body>
<h1>Real Edge / Chromium: offline renderer smoke</h1>
<div id="status">JavaScript has not executed</div><div id="network"></div>
<canvas id="canvas-top" width="240" height="120"></canvas>
<div id="lower-label">Scrolled live pixels: blue + yellow JS canvas</div>
<canvas id="canvas-lower" width="240" height="120"></canvas>
<script>
const marker = 'INLINE_JS_EXECUTED_OFFLINE_7D39';
document.getElementById('status').textContent = marker;
document.body.dataset.executed = marker;
const topCanvas = document.getElementById('canvas-top').getContext('2d');
topCanvas.fillStyle = '#ff0000'; topCanvas.fillRect(0,0,120,120);
topCanvas.fillStyle = '#00ff00'; topCanvas.fillRect(120,0,120,120);
const lowerCanvas = document.getElementById('canvas-lower').getContext('2d');
lowerCanvas.fillStyle = '#0000ff'; lowerCanvas.fillRect(0,0,120,120);
lowerCanvas.fillStyle = '#ffff00'; lowerCanvas.fillRect(120,0,120,120);
fetch('/read-proof').then(r => r.text()).then(text => {
  document.getElementById('network').textContent = text;
  document.body.dataset.readStatus = 'fulfilled-offline';
});
Promise.all(['POST','PUT','PATCH','DELETE'].map(method =>
  fetch('/write-proof?method=' + method, {method, body:'MUST_NOT_LEAVE_BROWSER'})
    .then(() => method + ':UNEXPECTED_SUCCESS', () => method + ':blocked')
)).then(results => { document.body.dataset.writeStatus = results.sort().join(','); });
</script></body></html>'''


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def inspect_png(raw, *, scrolled=False):
    require(raw.startswith(b'\x89PNG\r\n\x1a\n'), 'Screenshot lacks real PNG signature')
    with Image.open(io.BytesIO(raw)) as image:
        require(image.format == 'PNG', 'Screenshot is not PNG')
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        require(image.size == (1280, 900), f'Unexpected viewport: {image.size}')
        rgb = image.convert('RGB')
        y = 100 if scrolled else 200
        expected = ((0, 0, 255), (255, 255, 0)) if scrolled else ((255, 0, 0), (0, 255, 0))
        observed = (rgb.getpixel((60, y)), rgb.getpixel((200, y)))
        require(observed == expected, f'JS canvas pixels mismatch: {observed} != {expected}')
        require(len(rgb.getcolors(maxcolors=1_000_000) or []) > 10,
                'Screenshot is blank or lacks actual page rendering')
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
            'width': 1280, 'height': 900, 'canvas_samples': observed}


async def smoke():
    continued, aborted, dns_calls, launches, responses = [], [], [], [], []
    original_abort, original_launch, original_fulfill = Route.abort, BrowserType.launch, Route.fulfill

    class OfflineResponse:
        status = 200

        def __init__(self, body, content_type):
            self.body, self.content_type, self.disposed = body, content_type, False

        async def dispose(self):
            self.disposed = True

    def public_fixture_dns(host, port, *args, **kwargs):
        dns_calls.append(str(host))
        if host != 'fixture.test':
            raise socket.gaierror('Offline smoke denies DNS for every non-fixture host')
        # A synthetic globally routable address satisfies the REAL validate_url.
        # No request uses this address; fetch/continue_ are intercepted below.
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '',
                 ('93.184.216.34', port))]

    def fixture_response(route, api):
        request = route.request
        continued.append({'url': request.url, 'method': request.method,
                          'resource_type': request.resource_type, 'intercepted_api': api})
        parsed = urlsplit(request.url)
        require(parsed.scheme == 'https' and parsed.hostname == 'fixture.test',
                'Unexpected allowed request; refusing to touch the network')
        require(request.method in {'GET', 'HEAD'}, 'Backend guard allowed a write request')
        require(parsed.path in {'/render', '/read-proof', '/favicon.ico'},
                f'Unexpected fixture path: {parsed.path}')
        return OfflineResponse(HTML if parsed.path == '/render' else 'READ_REQUEST_FULFILLED_OFFLINE',
                               'text/html' if parsed.path == '/render' else 'text/plain')

    async def offline_continue(route, *args, **kwargs):
        response = fixture_response(route, 'continue_')
        await original_fulfill(route, status=response.status, content_type=response.content_type,
                               body=response.body)

    async def offline_fetch(route, *args, **kwargs):
        require(kwargs.get('max_redirects') == 0, 'Renderer fetch could follow unvalidated redirects')
        response = fixture_response(route, 'fetch(max_redirects=0)')
        responses.append(response)
        return response

    async def offline_fulfill(route, *args, **kwargs):
        response = kwargs.pop('response', None)
        if response is not None:
            require(isinstance(response, OfflineResponse), 'Unexpected non-fixture API response')
            kwargs.update(status=response.status, content_type=response.content_type, body=response.body)
        return await original_fulfill(route, *args, **kwargs)

    async def record_abort(route, *args, **kwargs):
        aborted.append({'url': route.request.url, 'method': route.request.method})
        return await original_abort(route, *args, **kwargs)

    async def real_launch(browser_type, *args, **kwargs):
        kwargs['args'] = list(kwargs.get('args') or []) + [
            '--disable-background-networking', '--disable-component-update',
            '--disable-sync', '--no-first-run', '--no-default-browser-check',
            '--metrics-recording-only', '--host-resolver-rules=MAP * ~NOTFOUND',
        ]
        record = {'channel': kwargs.get('channel', 'playwright-chromium'),
                  'headless': kwargs.get('headless')}
        launches.append(record)
        try:
            browser = await original_launch(browser_type, *args, **kwargs)
        except Exception as exc:
            record['error'] = str(exc)[:300]
            raise
        record['version'] = browser.version
        return browser

    def check_guard(start_continued, start_aborted, html):
        writes = {item['method'] for item in aborted[start_aborted:]
                  if '/write-proof?' in item['url']}
        require(writes == WRITE_METHODS, f'Write requests not aborted by guard: {writes}')
        require(all(item['method'] in {'GET', 'HEAD'} for item in continued[start_continued:]),
                'A write reached continue_/fulfill')
        require(any('/read-proof' in item['url'] for item in continued[start_continued:]),
                'Inline JS fetch never reached the original guard')
        require(f'data-executed="{MARKER}"' in html, 'Inline JS did not mutate the DOM')
        require('data-read-status="fulfilled-offline"' in html, 'GET fixture did not complete')
        require('data-write-status="DELETE:blocked,PATCH:blocked,POST:blocked,PUT:blocked"' in html,
                'JS did not observe every blocked write request')

    with tempfile.TemporaryDirectory(prefix='ctxagent-render-smoke-') as temp, ExitStack() as stack:
        root = Path(temp)
        for target, name, value in ((config, 'DATA_DIR', root),
                                    (config, 'BLOB_DIR', root / 'blobs'),
                                    (config, 'DB_PATH', root / 'chats.db'),
                                    (store, 'DB_PATH', root / 'chats.db')):
            stack.enter_context(patch.object(target, name, value))
        stack.enter_context(patch.object(socket, 'getaddrinfo', public_fixture_dns))
        stack.enter_context(patch.object(Route, 'continue_', offline_continue))
        stack.enter_context(patch.object(Route, 'fetch', offline_fetch))
        stack.enter_context(patch.object(Route, 'fulfill', offline_fulfill))
        stack.enter_context(patch.object(Route, 'abort', record_abort))
        stack.enter_context(patch.object(BrowserType, 'launch', real_launch))
        forbidden_model = stack.enter_context(patch.object(llm, 'complete',
            new=AsyncMock(side_effect=AssertionError('Model calls forbidden in renderer smoke'))))
        forbidden_session_model = stack.enter_context(patch.object(agent.AgentSession, '_complete',
            new=AsyncMock(side_effect=AssertionError('Session model calls forbidden'))))
        forbidden_http = stack.enter_context(patch.object(httpx.AsyncClient, 'request',
            new=AsyncMock(side_effect=AssertionError('Python HTTP forbidden'))))
        forbidden_sync_http = stack.enter_context(patch.object(httpx.Client, 'request',
            new=Mock(side_effect=AssertionError('Python HTTP forbidden'))))
        forbidden_summary = stack.enter_context(patch.object(browser_tools, 'summarize',
            new=AsyncMock(side_effect=AssertionError('Close must not summarize'))))

        # Direct backend render: Playwright launches a real browser, evaluates
        # inline JS, and screenshots a canvas that was blank in the input HTML.
        start_continued, start_aborted = len(continued), len(aborted)
        final, raw, png = await browsing._render(URL, screenshot=True)
        require(final == URL, f'Unexpected final URL: {final}')
        check_guard(start_continued, start_aborted, raw.decode())
        initial_pixels = inspect_png(png)

        store.init()
        cfg = SessionConfig(workdirs=[str(root)], enabled_tools=list(browser_tools.BROWSER_TOOLS))
        cid = store.create_chat('Offline real renderer smoke', cfg.to_dict())
        conv = Conversation()
        conv.new('user', K_USER, 'Frozen renderer smoke request; never include source pixels in history.')
        session = agent.AgentSession(cid, cfg, conv)
        conv.freeze_prefix(len(conv.segments))
        frozen_history = json.dumps(conv.to_messages(), sort_keys=True)
        require(isinstance(conv.browser, BrowserWorkspace), 'Conversation lacks BrowserWorkspace')
        data = {**browsing._extract(raw, final, 'text/html', 'utf-8'),
                'id': 'a' * 32, 'owner': cid, 'url': final, 'requested_url': URL,
                'sha256': hashlib.sha256(raw).hexdigest(), 'fetched_at': 1234567890.0}
        (browsing._directory() / (data['id'] + '.json')).write_text(json.dumps(data), encoding='utf-8')
        tab = conv.browser.attach(cid, data, max_tokens=8000)

        # Exercise the actual browser_render tool, including fresh cache/artifact,
        # scroll evaluation, live native pixels, and real session persistence.
        start_continued, start_aborted = len(continued), len(aborted)
        result = await browser_tools.browser_render(session.sandbox(), tab_id=tab,
            scroll_y=960, max_tokens=8000, ctx={'session': session})
        receipt = json.loads(result.text)
        view = conv.browser.get(tab)
        require(view['status'] == 'ready' and len(view['images']) == 1, 'Live image was not attached')
        fresh = browsing._page(view['page_id'], cid)
        require(MARKER in fresh['text'], 'JS-mutated DOM absent from refreshed extracted snapshot')
        writes = {item['method'] for item in aborted[start_aborted:] if '/write-proof?' in item['url']}
        require(writes == WRITE_METHODS, 'browser_render did not block all write requests')
        require(any('/read-proof' in item['url'] for item in continued[start_continued:]),
                'browser_render did not run inline JS fetch')
        image = view['images'][0]
        live_png = artifacts.artifact_path(image['id']).read_bytes()
        scrolled_pixels = inspect_png(live_png, scrolled=True)
        require(scrolled_pixels['sha256'] == image['id'], 'Artifact hash mismatch')
        require(initial_pixels['sha256'] != image['id'], 'Scroll did not change screenshot pixels')
        require(image['source_sha256'] == fresh['sha256'], 'Artifact is not linked to rendered source')
        live = conv.browser.messages()
        require(MARKER in json.dumps(live) and any(block.get('type') == 'image_ref'
            for message in live for block in message['content']), 'Native live image reference missing')
        native = artifacts.expand_messages(live)
        native_blocks = [block for message in native for block in message['content']
                         if block.get('type') == 'image_url']
        require(len(native_blocks) == 1, 'Native API image inclusion missing')
        encoded = native_blocks[0]['image_url']['url']
        require(encoded.startswith('data:image/png;base64,') and
                base64.b64decode(encoded.split(',', 1)[1]) == live_png, 'Native image bytes differ')
        require(0 < conv.browser.token_count() <= 8000, 'Live context is not bounded')

        serialized = conv.dumps()
        stored = json.dumps(store.get_chat(cid)['conversation'])
        for value in (serialized, stored, result.text):
            require(MARKER not in value and 'BEGIN UNTRUSTED PAGE TEXT' not in value,
                    'Page body leaked into serialization or tool receipt')
            require('data:image/' not in value and '"image_ref"' not in value,
                    'Native image content leaked into history/receipt')
        require(not any(segment.kind == K_MEDIA for segment in conv.segments), 'Media segment persisted')
        require(json.dumps(conv.to_messages(), sort_keys=True) == frozen_history, 'Frozen history changed')
        restored = Conversation.from_dict(conv.to_dict())
        restored.browser.refresh(cid)
        require(any(block.get('type') == 'image_ref' for message in restored.browser.messages()
                    for block in message['content']), 'Image spec cannot rehydrate from cached artifact')

        tokens_before_close = conv.browser.token_count()
        close_result = await browser_tools.browser_close(session.sandbox(), tab_id=tab,
            summarize=False, ctx={'session': session})
        closed = json.loads(close_result.text)
        require(closed['closed'] == [tab] and closed['handoffs'] == [], 'Unexpected close result')
        require(conv.browser.messages() == [] and conv.browser.token_count() == 0,
                'Close did not remove live text and pixels')
        require(Conversation.from_dict(store.get_chat(cid)['conversation']).browser.views == [],
                'Closed pixel specs persisted')
        require(json.dumps(conv.to_messages(), sort_keys=True) == frozen_history, 'Close changed history')
        for forbidden in (forbidden_model, forbidden_session_model, forbidden_http,
                          forbidden_sync_http, forbidden_summary):
            forbidden.assert_not_called()
        require(dns_calls and set(dns_calls) == {'fixture.test'}, 'Unexpected DNS lookup')
        require(all(response.disposed for response in responses), 'Backend did not dispose fixture responses')
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(png)
        require(hashlib.sha256(OUTPUT.read_bytes()).hexdigest() == initial_pixels['sha256'],
                'Inspection screenshot differs from verified screenshot')
        print(json.dumps({'status': 'PASS', 'browser_launches': launches,
            'direct_render_png': initial_pixels, 'browser_render_scrolled_png': scrolled_pixels,
            'allowed_requests_fulfilled_offline': continued, 'guard_aborted_requests': aborted,
            'synthetic_dns_calls': len(dns_calls), 'live_tokens_before_close': tokens_before_close,
            'live_tokens_after_close': conv.browser.token_count(), 'inspection_png': str(OUTPUT),
            'checks': ['inline JS DOM mutation', 'JS canvas exact RGB pixels', 'JS GET fulfilled offline',
                'POST/PUT/PATCH/DELETE blocked by original route guard', 'real viewport scroll',
                'BrowserWorkspace native image expansion', 'bounded live context',
                'body/native image absent from serialized conversation', 'immutable frozen history',
                'close summarize=False removes live pixels', 'no Python HTTP or model calls'],
            'limitations': ['No public network, DNS, redirects, or TLS are exercised.',
                'Routing interception and browser flags are not an OS packet-capture/network sandbox.',
                'This is backend + native-message-boundary verification, not frontend/provider vision testing.',
                'Snapshot/artifact/chat fixtures are temporary; only the initial viewport PNG is retained.']},
            indent=2))


if __name__ == '__main__':
    asyncio.run(asyncio.wait_for(smoke(), timeout=380))
