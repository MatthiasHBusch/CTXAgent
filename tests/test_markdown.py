"""Offline renderer and vendored-asset checks; Node checks skip if Node is absent."""
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / 'web/vendor/markdown-it-15.0.2.min.js'
NODE = r"""
const fs = require('fs'), vm = require('vm');
const sandbox = {window: {markdownit: require('./web/vendor/markdown-it-15.0.2.min.js')}, URL};
vm.runInNewContext(fs.readFileSync('./web/markdown.js','utf8'), sandbox);
const fixtures = JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify(fixtures.map(src => sandbox.window.ChatMarkdown.render(src))));
"""


class MarkdownTests(unittest.TestCase):
    def render(self, *sources):
        node = shutil.which('node')
        if not node:
            self.skipTest('Node not installed; run tests.markdown_ui_smoke with Edge instead')
        result = subprocess.run([node, '-e', NODE], input=json.dumps(sources), cwd=ROOT,
                                text=True, encoding='utf-8', capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_bundle_integrity_license_and_script_order(self):
        digest = hashlib.sha256(BUNDLE.read_bytes()).digest()
        self.assertEqual(digest.hex(), '635972b985228e8af9f0143647c68616b7a3bb09f6946e7e4a52e43dcf5e7be5')
        index = (ROOT / 'web/index.html').read_text(encoding='utf-8')
        self.assertIn('integrity="sha256-' + base64.b64encode(digest).decode() + '"', index)
        self.assertLess(index.index('/static/vendor/markdown-it-15.0.2.min.js'), index.index('/static/markdown.js'))
        self.assertLess(index.index('/static/markdown.js'), index.index('/static/app.js'))
        self.assertIn('MIT', (ROOT / 'web/vendor/markdown-it-LICENSE.txt').read_text(encoding='utf-8'))

    def test_tables_normal_spaced_alignment_and_distinct_tables(self):
        normal = '| Tools | Purpose |\n|---|---|\n| browser_open | Open a page |\n| browser_close | Close it |'
        spaced = normal.replace('\n', '\n\n')
        a, b, aligned, separate = self.render(normal, spaced, '| L | R |\n|:---|---:|\n|x|y|', spaced+'\n\n'+spaced)
        self.assertEqual(a, b)
        self.assertEqual(a.count('<table>'), 1)
        self.assertEqual(a.count('<td>'), 4)
        self.assertIn('browser_open', a)
        self.assertIn('text-align:left', aligned)
        self.assertIn('text-align:right', aligned)
        self.assertEqual(separate.count('<table>'), 2)

    def test_fenced_unfinished_nested_and_indented_code_is_literal(self):
        body = '| A | B |\n\n|---|---|\n\n| **bold** | <img src=x> |'
        variants = ['```markdown\n'+body+'\n```', '~~~markdown\n'+body+'\n~~~',
                    '```markdown\n'+body, '~~~markdown\n'+body,
                    '> ```markdown\n'+''.join('> '+line+'\n' for line in body.splitlines())+'> ```',
                    '\n'.join('    '+line for line in body.splitlines())]
        for html in self.render(*variants):
            self.assertIn('<pre><code', html)
            self.assertNotIn('<table>', html)
            self.assertNotIn('<strong>', html)
            self.assertNotIn('<img', html)
            self.assertIn('**bold**', html)
        # An indented literal fence must not swallow a later real spaced table.
        html = self.render('    ```\n\n'+body)[0]
        self.assertIn('<table>', html)

    def test_nested_lists_start_emphasis_tasks_and_code(self):
        src = '3. Third\n   - nested\n     - deeper\n4. Fourth\n\n**bold** _italic_ ~~gone~~ `**literal**`\n\n- [x] done\n- [ ] todo'
        html = self.render(src)[0]
        self.assertIn('<ol start="3">', html)
        self.assertEqual(html.count('<ul>'), 3)
        self.assertIn('<strong>bold</strong>', html)
        self.assertIn('<em>italic</em>', html)
        self.assertIn('<s>gone</s>', html)
        self.assertIn('<code>**literal**</code>', html)
        self.assertEqual(html.count('type="checkbox"'), 2)
        self.assertEqual(html.count(' disabled '), 2)
        self.assertEqual(html.count(' checked'), 1)

    def test_links_images_and_raw_html_fail_closed(self):
        src = ('[safe](https://example.org/?a=1&b=2) https://example.org/autolink\n\n'
               '[bad](javascript:alert(1)) [bad](data:text/html,evil) [local](file:///secret) '
               '[relative](/api/chats) [creds](https://user:pass@example.org)\n\n'
               '![remote](https://tracking.example/pixel)\n\n'
               '<img src=x onerror="window.pwned=1"><script>window.pwned=1</script>')
        html = self.render(src)[0]
        self.assertEqual(html.count('<a '), 3)
        self.assertNotRegex(html, r'<(?:img|script|iframe)\b')
        self.assertIn('[Image: remote]', html)
        self.assertIn('&lt;script&gt;', html)
        for match in re.findall(r'<a\b[^>]+>', html):
            self.assertIn('rel="noopener noreferrer"', match)
            self.assertIn('referrerpolicy="no-referrer"', match)
            self.assertIn('target="_blank"', match)
            self.assertNotRegex(match, r'href="(?:javascript|data|file|/api)', match)

    def test_partial_fences_and_private_placeholder_never_lose_code(self):
        sources = ['```js\nconst x = "**raw**";', '```js\nconst x = "**raw**";\n```',
                   '~~~\n<script>bad</script>', '``a ` b`` and **bold**', 'before\x00 19\x00after']
        html = self.render(*sources)
        self.assertIn('const x = &quot;**raw**&quot;;', html[0])
        self.assertIn('const x = &quot;**raw**&quot;;', html[1])
        self.assertIn('&lt;script&gt;bad&lt;/script&gt;', html[2])
        self.assertIn('<code>a ` b</code>', html[3])
        self.assertIn('before', html[4])
        self.assertIn('after', html[4])
        self.assertNotIn('undefined', html[4])


if __name__ == '__main__':
    unittest.main()
