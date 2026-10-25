"""Offline browser/search security, pagination and source-receipt regressions."""
import json
import os
import socket
import tempfile
import unittest
from unittest.mock import patch
import httpx
_DATA=tempfile.TemporaryDirectory(prefix='ctxagent-web-tests-')
os.environ.setdefault('CTXAGENT_DATA',_DATA.name)
from ctxagent import browsing as b


class WebTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(socket,'getaddrinfo',return_value=[
            (socket.AF_INET,socket.SOCK_STREAM,6,'',('93.184.216.34',443))]))
    def http(self,handler):
        cls=httpx.AsyncClient
        self.enterContext(patch.object(b.httpx,'AsyncClient',lambda **kw:cls(transport=httpx.MockTransport(handler),**kw)))

    async def test_public_url_policy_and_dns_redirect_revalidation(self):
        for url in ['file:///secret','http://127.0.0.1','http://[::1]','http://169.254.169.254',
                    'http://10.0.0.1','https://user:password@example.com','https://example.com:8888','https://example.com\nheader']:
            with self.assertRaises(ValueError,msg=url): b.validate_url(url)
        self.assertEqual(b.validate_url('example.com'),'https://example.com/')
        with patch.object(socket,'getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('192.168.0.1',443))]):
            with self.assertRaises(ValueError): b.validate_url('https://dns-private.example')
        sent=[]
        def handler(r): sent.append(str(r.url)); return httpx.Response(302,headers={'location':'http://127.0.0.1/secret'})
        self.http(handler)
        with self.assertRaises(ValueError): await b.open_page('a',url='https://public.example')
        self.assertEqual(len(sent),1)

    async def test_html_links_title_paging_find_and_owner_guard(self):
        html=b'<html><title>Example</title><body><script>secret script</script><p>Evidence alpha beta</p><a href="/source">Primary</a></body></html>'
        self.http(lambda r:httpx.Response(200,content=html,headers={'content-type':'text/html'}))
        first,images=await b.open_page('a',url='https://example.com',max_chars=8)
        self.assertEqual(first['title'],'Example'); self.assertFalse(first['at_end'])
        self.assertEqual(images,[]); self.assertEqual(len(first['sha256']),64)
        self.assertEqual(first['links'][0]['url'],'https://example.com/source')
        second,_=await b.open_page('a',page_id=first['id'],offset=first['next_offset'],max_chars=40000)
        self.assertTrue(second['at_end']); self.assertNotIn('secret script',second['text'])
        matches=b.find('a',first['id'],'alpha')
        self.assertEqual(matches['total_matches'],1)
        self.assertIn('alpha',matches['matches'][0]['context'])
        with self.assertRaises(PermissionError): await b.open_page('other',page_id=first['id'])
        with self.assertRaises(PermissionError): b.find('other',first['id'],'alpha')
        with self.assertRaises(ValueError): await b.open_page('a',url='x',page_id=first['id'])
        with self.assertRaises(ValueError): await b.open_page('a',page_id=first['id'],screenshot=True)

    async def test_size_bounds_and_unsupported_binary(self):
        self.http(lambda r:httpx.Response(200,content=b'x'*(b.MAX_BYTES+1)))
        with self.assertRaises(ValueError): await b._get('https://example.com')
        with self.assertRaises(ValueError): b._extract(b'\0\0binary','https://example.com','application/octet-stream')
        with self.assertRaises(ValueError): await b.open_page('a',url='https://example.com',max_chars=50000)

    async def test_cross_host_redirect_drops_credentials(self):
        sent=[]
        def handler(r):
            sent.append(dict(r.headers))
            return httpx.Response(302,headers={'location':'https://other.example/'}) if len(sent)==1 else httpx.Response(200,text='ok')
        self.http(handler)
        await b._get('https://example.com',{'X-Subscription-Token':'private-key'})
        self.assertIn('x-subscription-token',sent[0]); self.assertNotIn('x-subscription-token',sent[1])

    async def test_search_rss_query_only_and_results_are_candidates(self):
        requests=[]
        def handler(r):
            requests.append(r)
            return httpx.Response(200,text='<rss><channel><item><title>Primary</title><link>https://source.example</link><description>Candidate snippet</description></item></channel></rss>')
        self.http(handler)
        data=await b.search('public query',engine='bing')
        self.assertEqual(data['results'][0]['title'],'Primary')
        self.assertEqual(requests[0].url.params['q'],'public query')
        self.assertNotIn('source evidence',requests[0].content.decode())
        self.assertIn('not claim verification',data['note'])

    async def test_search_fallback_and_ddg_redirect_unwrap(self):
        def handler(r):
            if r.url.host=='www.bing.com':return httpx.Response(503)
            return httpx.Response(200,text='<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fsource.example%2Fpaper">Paper</a><div class="result__snippet">Snippet</div></div>')
        self.http(handler)
        data=await b.search('query',engine='duckduckgo')
        self.assertEqual(data['engine'],'duckduckgo')
        self.assertEqual(data['results'][0]['url'],'https://source.example/paper')
        self.assertEqual(data['fallback_errors'],[])

    async def test_js_render_receipt_and_image_attachment_contract(self):
        from PIL import Image
        import io
        buf=io.BytesIO(); Image.new('RGB',(32,32),'white').save(buf,format='PNG')
        async def render(*args):return 'https://final.example',b'<html><title>JS</title><body>Rendered text</body></html>',buf.getvalue()
        with patch.object(b,'_render',side_effect=render):
            data,images=await b.open_page('owner',url='https://example.com',render_js=True,screenshot=True)
        self.assertEqual(data['title'],'JS'); self.assertEqual(len(images),1)
        self.assertEqual(images[0]['source_sha256'],data['sha256'])
        self.assertEqual(data['url'],'https://final.example')

    async def test_pdf_text_and_plaintext_not_inferred_visual_layout(self):
        import fitz
        doc=fitz.open(); doc.new_page().insert_text((72,72),'PDF evidence'); raw=doc.tobytes(); doc.close()
        data=b._extract(raw,'https://example.com/paper.pdf','application/pdf')
        self.assertEqual(data['kind'],'pdf_text'); self.assertIn('PDF evidence',data['text'])
        text=b._extract(b'plain text','https://example.com/text','text/plain')
        self.assertEqual(text['text'],'plain text')

    async def test_find_limits_literal_no_regex_and_engine_errors_explicit(self):
        self.http(lambda r:httpx.Response(200,text='<html><body>a.b a.b</body></html>',headers={'content-type':'text/html'}))
        data,_=await b.open_page('owner',url='https://example.com')
        hits=b.find('owner',data['id'],'a.b',max_matches=1)
        self.assertEqual(hits['total_matches'],2); self.assertTrue(hits['more_matches'])
        with self.assertRaises(ValueError): b.find('owner',data['id'],'',20)
        with self.assertRaises(ValueError): await b.search('query',engine='unknown')


if __name__=='__main__': unittest.main(verbosity=2)
