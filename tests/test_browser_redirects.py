"""Offline JS-browser single-hop interception regressions (no browser/network)."""
import os
import socket
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

_DATA = tempfile.TemporaryDirectory(prefix='ctxagent-browser-redirect-tests-')
os.environ.setdefault('CTXAGENT_DATA', _DATA.name)
from ctxagent import browsing as b


class FakeRoute:
    """Model fetch's default redirect-following behavior to detect unsafe calls."""
    def __init__(self, url='https://public.example/page', method='GET',
                 resource_type='document', status=200, location=None):
        self.request = SimpleNamespace(url=url, method=method, resource_type=resource_type)
        self.response = SimpleNamespace(status=status, headers={'location': location} if location else {},
                                        dispose=AsyncMock())
        self.sent = []
        self.fetch_options = []
        self.fetch = AsyncMock(side_effect=self._fetch)
        self.abort = AsyncMock()
        self.fulfill = AsyncMock()
        self.continue_ = AsyncMock()

    async def _fetch(self, **kwargs):
        self.fetch_options.append(kwargs)
        self.sent.append(self.request.url)
        if kwargs.get('max_redirects', 20) and self.response.headers.get('location'):
            self.sent.append(self.response.headers['location'])
        return self.response


class BrowserRedirectTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(socket, 'getaddrinfo', return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))]))

    async def test_public_to_private_never_fetches_target_for_any_resource(self):
        targets = ['http://127.0.0.1/secret', 'http://169.254.169.254/latest/meta-data',
                   'http://10.0.0.1/', 'http://[::1]/', 'https://private.example/']
        for resource in ('document', 'script', 'stylesheet', 'image', 'xhr', 'fetch'):
            for target in targets:
                with self.subTest(resource=resource, target=target):
                    route = FakeRoute(resource_type=resource, status=302, location=target)
                    errors = []
                    await b._browser_route(route, errors)
                    self.assertEqual(route.sent, ['https://public.example/page'])
                    self.assertEqual(route.fetch_options, [{'max_redirects': 0, 'timeout': 30000}])
                    route.abort.assert_awaited_once()
                    route.fulfill.assert_not_awaited()
                    route.continue_.assert_not_awaited()
                    route.response.dispose.assert_awaited_once()
                    self.assertEqual(len(errors), 1)
                    self.assertIn('browser_open (render_js=false)', errors[0])
                    self.assertIn('final URL', errors[0])

    async def test_all_3xx_refused_even_public_relative_or_missing_location(self):
        for status in (300, 301, 302, 303, 304, 305, 306, 307, 308, 399):
            for location in ('https://other-public.example/', '/relative', None):
                with self.subTest(status=status, location=location):
                    route = FakeRoute(status=status, location=location)
                    errors = []
                    await b._browser_route(route, errors)
                    self.assertEqual(route.sent, [route.request.url])
                    route.abort.assert_awaited_once()
                    route.fulfill.assert_not_awaited()
                    self.assertEqual(len(errors), 1)

    async def test_public_get_and_head_fulfilled_without_changing_browser_url(self):
        for method, status in (('GET', 200), ('HEAD', 204), ('GET', 404)):
            with self.subTest(method=method, status=status):
                route = FakeRoute(method=method, status=status)
                errors = []
                await b._browser_route(route, errors)
                self.assertEqual(route.sent, [route.request.url])
                self.assertEqual(route.fetch_options, [{'max_redirects': 0, 'timeout': 30000}])
                route.fulfill.assert_awaited_once_with(response=route.response)
                route.response.dispose.assert_awaited_once()
                route.abort.assert_not_awaited()
                route.continue_.assert_not_awaited()
                self.assertEqual(errors, [])

    async def test_post_and_other_disallowed_methods_never_fetched(self):
        for method in ('POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS', 'CONNECT', 'TRACE'):
            with self.subTest(method=method):
                route = FakeRoute(method=method)
                await b._browser_route(route, [])
                route.fetch.assert_not_awaited()
                route.fulfill.assert_not_awaited()
                route.continue_.assert_not_awaited()
                route.abort.assert_awaited_once()

    async def test_private_first_hop_and_media_font_never_fetched(self):
        for route in (FakeRoute(url='http://127.0.0.1/'),
                      FakeRoute(url='file:///secret'),
                      FakeRoute(resource_type='media'), FakeRoute(resource_type='font')):
            with self.subTest(url=route.request.url, resource=route.request.resource_type):
                await b._browser_route(route, [])
                route.fetch.assert_not_awaited()
                route.fulfill.assert_not_awaited()
                route.continue_.assert_not_awaited()
                route.abort.assert_awaited_once()

    async def test_fetch_failure_aborts_without_falling_back_to_continue(self):
        route = FakeRoute()
        route.fetch.side_effect = TimeoutError('offline timeout')
        await b._browser_route(route, [])
        route.abort.assert_awaited_once()
        route.fulfill.assert_not_awaited()
        route.continue_.assert_not_awaited()

    def fake_renderer(self, routes, fail_navigation=False):
        """Supply only mocked Playwright APIs, even if Playwright is not installed."""
        context = SimpleNamespace(set_default_timeout=Mock(), route=AsyncMock(),
                                  route_web_socket=AsyncMock(), new_page=AsyncMock())
        page = SimpleNamespace(url='https://public.example/page', goto=AsyncMock(),
                               wait_for_timeout=AsyncMock(), evaluate=AsyncMock(),
                               content=AsyncMock(return_value='<html>offline public</html>'),
                               screenshot=AsyncMock(return_value=b'png'))
        context.new_page.return_value = page
        browser = SimpleNamespace(new_context=AsyncMock(return_value=context), close=AsyncMock())
        playwright = SimpleNamespace(chromium=SimpleNamespace(launch=AsyncMock(return_value=browser)))
        # Special methods must be defined on the type, not an instance namespace.
        class Manager:
            async def __aenter__(self): return playwright
            async def __aexit__(self, *args): return False
        api = SimpleNamespace(async_playwright=Manager)
        self.enterContext(patch.dict(sys.modules, {'playwright.async_api': api}))

        async def goto(url, **kwargs):
            pattern, guard = context.route.call_args.args
            self.assertEqual(pattern, '**/*')
            for route in routes:
                await guard(route)
            if fail_navigation:
                raise RuntimeError('net::ERR_FAILED')
        page.goto.side_effect = goto
        return browser, context, page

    async def test_navigation_redirect_has_clear_error_and_browser_closes(self):
        route = FakeRoute(status=302, location='http://127.0.0.1/')
        browser, _, _ = self.fake_renderer([route], fail_navigation=True)
        with self.assertRaisesRegex(ValueError, 'browser_open .*render_js=false'):
            await b._render(route.request.url)
        self.assertEqual(route.sent, [route.request.url])
        route.fulfill.assert_not_awaited()
        browser.close.assert_awaited_once()

    async def test_subresource_redirect_fails_render_not_partial_success(self):
        document = FakeRoute()
        script = FakeRoute(url='https://public.example/script.js', resource_type='script',
                           status=307, location='http://10.0.0.1/secret.js')
        browser, _, _ = self.fake_renderer([document, script])
        with self.assertRaisesRegex(ValueError, 'including subresources'):
            await b._render(document.request.url)
        document.fulfill.assert_awaited_once_with(response=document.response)
        script.fulfill.assert_not_awaited()
        self.assertEqual(script.sent, [script.request.url])
        browser.close.assert_awaited_once()

    async def test_request_budget_timeouts_and_context_isolation_preserved(self):
        routes = [FakeRoute(url=f'https://public.example/resource{i}') for i in range(101)]
        browser, context, page = self.fake_renderer(routes)
        result = await b._render('https://public.example/page', screenshot=True, scroll_y=10)
        self.assertEqual(result, (page.url, b'<html>offline public</html>', b'png'))
        for route in routes[:100]:
            route.fetch.assert_awaited_once_with(max_redirects=0, timeout=30000)
            route.continue_.assert_not_awaited()
        routes[100].fetch.assert_not_awaited()
        routes[100].abort.assert_awaited_once()
        context.set_default_timeout.assert_called_once_with(30000)
        browser.new_context.assert_awaited_once_with(viewport={'width': 1280, 'height': 900},
                                                     accept_downloads=False, service_workers='block')
        page.goto.assert_awaited_once_with('https://public.example/page',
                                          wait_until='domcontentloaded', timeout=30000)
        self.assertEqual([call.args[0] for call in page.wait_for_timeout.await_args_list], [500, 200])
        page.evaluate.assert_awaited_once_with('y => window.scrollTo(0, y)', 10)
        page.screenshot.assert_awaited_once_with(full_page=False)
        context.route_web_socket.assert_awaited_once()
        browser.close.assert_awaited_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
