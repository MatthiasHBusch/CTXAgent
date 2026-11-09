"""The live canvas: drawing operations, versions and undo, the model's copy (focus, rulers),
the script worker, saving, persistence, the agent integration and the cache trailer.
Offline: artifacts and files live in a temporary folder; the model is a scripted stream."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

from ctxagent import cache_policy, config, paths, tools
from ctxagent.canvas import Canvas, CanvasError, OPERATIONS, apply_ops
from ctxagent.segments import Conversation
from ctxagent.tools import ToolResult
from tests.test_productivity import make_session


class Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        p = patch.object(config, 'DATA_DIR', self.root / 'data')
        p.start()
        self.addCleanup(p.stop)
        (self.root / 'work').mkdir()

    def blank(self, w=400, h=300):
        c = Canvas()
        c.open_blank(w, h, 'white')
        return c


class DrawTests(Base):
    def test_shapes_land_where_asked(self):
        c = self.blank()
        c.draw([{'type': 'rect', 'box': [10, 10, 60, 60], 'fill': '#ff0000'},
                {'type': 'line', 'points': [[100, 150], [300, 150]], 'color': 'blue', 'width': 5},
                {'type': 'circle', 'center': [200, 60], 'radius': 20, 'fill': 'green'},
                {'type': 'arrow', 'from': [20, 250], 'to': [180, 250], 'color': 'black', 'width': 4},
                {'type': 'polygon', 'points': [[300, 200], [380, 200], [340, 280]], 'fill': 'yellow'},
                {'type': 'marker', 'at': [380, 20], 'size': 12, 'shape': 'cross', 'color': 'black'},
                {'type': 'text', 'at': [220, 100], 'text': 'Hello', 'size': 20, 'color': 'black',
                 'background': 'white', 'border': 'black'}], label='first drawing')
        img = c.image()
        self.assertEqual(img.getpixel((30, 30))[:3], (255, 0, 0))
        self.assertEqual(img.getpixel((200, 150))[:3], (0, 0, 255))
        self.assertEqual(img.getpixel((200, 60))[:3], (0, 128, 0))
        self.assertEqual(img.getpixel((340, 220))[:3], (255, 255, 0))
        self.assertEqual(img.getpixel((5, 5))[:3], (255, 255, 255))
        self.assertEqual(c.current()['label'], 'first drawing')
        self.assertEqual(c.current()['version'], 2)
        self.assertIn('rect', c.current()['summary'])

    def test_translucent_shapes_blend(self):
        c = self.blank()
        c.draw([{'type': 'rect', 'box': [0, 0, 100, 100], 'fill': '#0000ff'},
                {'type': 'rect', 'box': [50, 0, 150, 100], 'fill': '#ff000080'}])
        mixed = c.image().getpixel((75, 50))
        self.assertTrue(100 < mixed[0] < 160 and 90 < mixed[2] < 160, mixed)  # red over blue, half opaque
        self.assertEqual(c.image().getpixel((125, 50))[:3][1], 127)          # red over white

    def test_erase_and_paste(self):
        c = self.blank()
        c.draw([{'type': 'rect', 'box': [0, 0, 200, 200], 'fill': 'black'}])
        Image.new('RGB', (40, 20), (0, 200, 0)).save(self.root / 'work' / 'pic.png')
        sb = tools.Sandbox([str(self.root / 'work')])
        c.draw([{'type': 'erase', 'box': [0, 0, 50, 50]},
                {'type': 'paste', 'path': 'pic.png', 'box': [100, 100, 180, 140]},
                {'type': 'paste', 'path': 'pic.png', 'box': [-30, 250, 10, 270]}], sandbox=sb)
        img = c.image()
        self.assertEqual(img.getpixel((10, 10))[:3], (255, 255, 255))
        self.assertEqual(img.getpixel((140, 120))[:3], (0, 200, 0))
        self.assertEqual(img.getpixel((5, 260))[:3], (0, 200, 0))        # cut at the left edge

    def test_bad_ops_change_nothing(self):
        c = self.blank()
        before = c.current()['id']
        for ops in ([], [{'type': 'spiral'}], [{'type': 'line', 'points': [[0, 0]]}],
                    [{'type': 'rect', 'box': [0, 0, 1, 1], 'fill': 'notacolour'}],
                    [{'type': 'text', 'at': [0, 0]}], [{'type': 'paste', 'path': 'x.png', 'box': [0, 0, 9, 9]}]):
            with self.assertRaises(CanvasError):
                c.draw(ops)
        self.assertEqual(c.current()['id'], before)
        with self.assertRaisesRegex(CanvasError, 'op 1'):
            apply_ops(c.image(), [{'type': 'rect', 'box': [0, 0, 5, 5]}, {'type': 'nope'}], (255, 255, 255, 255))


class HistoryTests(Base):
    def test_undo_redo_and_new_versions(self):
        c = self.blank()
        for i in range(3):
            c.draw([{'type': 'marker', 'at': [10 + i * 20, 10]}], label=f'step {i}')
        self.assertEqual(c.current()['version'], 4)
        c.undo(2)
        self.assertEqual(c.current()['version'], 2)
        c.undo(to_version=4)                       # redo
        self.assertEqual(c.current()['version'], 4)
        c.undo(1)
        c.draw([{'type': 'marker', 'at': [300, 10]}])
        self.assertEqual([v['version'] for v in c.versions], [1, 2, 3, 4, 5])
        self.assertEqual(c.head, 4)
        with self.assertRaises(CanvasError):
            c.undo(to_version=99)
        with self.assertRaises(CanvasError):
            c.undo(10)

    def test_history_is_capped_but_keeps_the_head(self):
        c = self.blank(64, 64)
        with patch('ctxagent.canvas.MAX_VERSIONS', 4):
            for i in range(6):
                c.draw([{'type': 'marker', 'at': [i * 5, 5]}])
        self.assertEqual(len(c.versions), 4)
        self.assertEqual(c.current()['version'], 7)

    def test_round_trip_with_the_conversation(self):
        conv = Conversation()
        conv.canvas.open_blank(100, 80)
        conv.canvas.draw([{'type': 'rect', 'box': [0, 0, 10, 10], 'fill': 'red'}])
        conv.canvas.set_focus([0, 0, 50, 40])
        clone = Conversation.from_dict(json.loads(conv.dumps()))
        self.assertEqual(clone.canvas.current()['id'], conv.canvas.current()['id'])
        self.assertEqual(clone.canvas.focus, [0, 0, 50, 40])
        self.assertEqual(clone.canvas.revision, conv.canvas.revision)
        self.assertFalse(Conversation.from_dict({}).canvas.is_open)


class ModelCopyTests(Base):
    def test_whole_canvas_scaled_with_rulers(self):
        c = self.blank(2048, 1024)
        view = c.rendition()
        self.assertEqual(view['scale'], 0.5)
        self.assertEqual((view['width'], view['height']), (1024 + 30, 512 + 30))
        self.assertIn('whole canvas at 0.5x', c.header())
        message = c.messages()[0]
        self.assertEqual(message['content'][1], {'type': 'image_ref', 'image': view})
        self.assertGreater(c.token_count(), 2000)

    def test_focus_enlarges_a_region(self):
        c = self.blank(800, 600)
        c.set_focus([100, 100, 300, 200], rulers=False)
        view = c.rendition()
        self.assertEqual(view['scale'], 4.0)                     # capped zoom
        self.assertEqual((view['width'], view['height']), (800, 400))
        self.assertIn('x 100-300, y 100-200', c.header())
        self.assertIs(c.rendition(), view)                       # cached per state
        c.set_focus(None)
        self.assertIsNot(c.rendition(), view)
        with self.assertRaises(CanvasError):
            c.set_focus([796, 596, 900, 900])                    # under 8x8 px on the canvas
        self.assertTrue(c.has_artifact(c.rendition()['id']))

    def test_closed_canvas_is_absent(self):
        c = Canvas()
        self.assertEqual(c.messages(), [])
        self.assertEqual(c.token_count(), 0)
        with self.assertRaises(CanvasError):
            c.draw([{'type': 'marker', 'at': [1, 1]}])


class ScriptAndSaveTests(Base):
    def test_script_draws_and_prints(self):
        c = self.blank(100, 100)
        desc, out = c.run_script("draw.rectangle([0, 0, 49, 99], fill='red')\nprint('W', W, 'H', H)")
        self.assertEqual(c.image().getpixel((10, 10))[:3], (255, 0, 0))
        self.assertIn('W 100 H 100', out)
        self.assertEqual(desc['version'], 2)
        c.run_script("img = img.resize((50, 20))")
        self.assertEqual(c.image().size, (50, 20))

    def test_script_failure_and_timeout_change_nothing(self):
        c = self.blank(64, 64)
        head = c.current()['id']
        with self.assertRaisesRegex(CanvasError, 'ZeroDivisionError'):
            c.run_script("1/0")
        with self.assertRaisesRegex(CanvasError, 'within 2 s'):
            c.run_script("import time\ntime.sleep(10)", timeout=2)
        with self.assertRaises(CanvasError):
            c.run_script("img = 'not an image'")
        self.assertEqual(c.current()['id'], head)

    def test_frozen_app_runs_the_worker_through_run_module(self):
        argv = paths.module_command('ctxagent.canvas_worker', 'a', 'b', 'c', frozen=True)
        self.assertEqual(argv[1:3], [paths.RUN_MODULE_FLAG, 'ctxagent.canvas_worker'])

    def test_save_formats(self):
        c = self.blank(40, 30)
        c.save(self.root / 'work' / 'out.png')
        c.save(self.root / 'work' / 'out.jpg')
        with Image.open(self.root / 'work' / 'out.jpg') as im:
            self.assertEqual((im.format, im.size), ('JPEG', (40, 30)))
        with self.assertRaises(CanvasError):
            c.save(self.root / 'work' / 'out.bmp')


class AgentTests(Base):
    def session(self):
        s = make_session(root=self.root / 'work')
        return s

    async def test_operations_appear_only_while_open(self):
        s = self.session()
        names = s.tool_names()
        self.assertIn('canvas_open', names)
        self.assertFalse(set(OPERATIONS) & set(names))
        r = await s._exec_tool('canvas_open', {'width': 300, 'height': 200})
        self.assertFalse(r.error, r.text)
        self.assertTrue(set(OPERATIONS) <= set(s.tool_names()))
        schema = {t['function']['name'] for t in s.tool_schema()}
        self.assertFalse(set(OPERATIONS) & schema)               # late-bound: not native tools
        from ctxagent import document_catalog
        catalog = json.loads(document_catalog.messages(s)[0]['content'].split('\n', 4)[-1])
        self.assertIn({'id': 'canvas', 'operations': list(OPERATIONS)}, catalog['documents'])
        r = await s._exec_tool('document_call', {'operation': 'canvas_draw', 'arguments_json': json.dumps(
            {'ops': [{'type': 'rect', 'box': [0, 0, 20, 20], 'fill': 'red'}], 'label': 'square'})})
        self.assertFalse(r.error, r.text)
        self.assertEqual(s.conv.canvas.current()['label'], 'square')
        r = await s._exec_tool('canvas_close', {})
        self.assertFalse(set(OPERATIONS) & set(s.tool_names()))

    async def test_view_image_opens_on_the_canvas(self):
        s = self.session()
        Image.new('RGB', (120, 90), (10, 20, 30)).save(self.root / 'work' / 'photo.png')
        r = await s._exec_tool('view_image', {'path': 'photo.png', 'canvas': True})
        self.assertFalse(r.error, r.text)
        self.assertFalse(r.images)
        self.assertEqual(s.conv.canvas.current()['width'], 120)

    async def test_canvas_goes_last_behind_the_tail_mark_and_the_ui_hears_of_changes(self):
        s = self.session()
        s.cfg.model = 'anthropic/claude-opus-5.5'
        await s._exec_tool('canvas_open', {'width': 200, 'height': 100})
        seen = []

        async def stream(**kw):
            seen.append(kw)
            if len(seen) == 1:
                yield {'type': 'tool_calls', 'tool_calls': [{'id': 'd1', 'type': 'function', 'function': {
                    'name': 'document_call', 'arguments': json.dumps({'operation': 'canvas_draw', 'arguments_json':
                        json.dumps({'ops': [{'type': 'marker', 'at': [5, 5]}]})})}}]}
            else:
                yield {'type': 'text', 'text': 'drawn'}
            yield {'type': 'done', 'finish_reason': 'stop'}
        with patch('ctxagent.agent.llm.stream_chat', stream):
            events = [ev async for ev in s.run_turn('draw a dot')]
        first = seen[0]
        self.assertEqual(first['uncached_tail'], 1)
        last = first['messages'][-1]
        self.assertEqual(last['content'][1]['type'], 'image_ref')
        self.assertTrue(last['content'][0]['text'].startswith('LIVE CANVAS'))
        self.assertNotEqual(seen[1]['messages'][-1]['content'][1]['image']['id'],
                            last['content'][1]['image']['id'])          # the new version
        self.assertTrue(any(e['type'] == 'canvas' for e in events))
        self.assertNotIn(last, s._last_request['messages'])               # the shared prefix has no canvas
        marked = cache_policy.apply(first['messages'], 'anthropic/claude-opus-5.5', 0, tail=True, trailer=1)[0]
        self.assertFalse(any('cache_control' in b for b in marked[-1]['content']))
        self.assertTrue(any('cache_control' in b for m in marked[:-1] if isinstance(m['content'], list)
                            for b in m['content']))

    def test_direct_anthropic_request_marks_before_the_canvas(self):
        from ctxagent import anthropic_api
        c = self.blank(64, 64)
        messages = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'hello there'}] + c.messages()
        with patch('ctxagent.anthropic_api.expand_messages', side_effect=lambda m: m):
            kwargs = anthropic_api.build_request(model='claude-opus-5-5', messages=messages, tools=None,
                                                 max_tokens=100, reasoning_effort='off', frozen_msg_count=0,
                                                 previous_cache_anchor=None, cache_ttl='5m', uncached_tail=1)
        self.assertNotIn('cache_control', kwargs)
        blocks = kwargs['messages'][-1]['content']
        hello = next(b for b in blocks if b.get('text') == 'hello there')
        self.assertIn('cache_control', hello)
        self.assertFalse(any('cache_control' in b for b in blocks if b is not hello))

    async def test_artifact_route_serves_canvas_versions(self):
        import httpx
        from ctxagent import agent as agentmod, server
        s = self.session()
        await s._exec_tool('canvas_open', {'width': 50, 'height': 50})
        aid = s.conv.canvas.current()['id']
        with patch.object(agentmod, 'get_session', return_value=s), patch.object(server, '_ordinary_chat', lambda c: None):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://t') as client:
                self.assertEqual((await client.get(f'/api/chats/x/artifacts/{aid}')).status_code, 200)
                listing = (await client.get('/api/chats/x/canvas')).json()
                self.assertEqual(listing['current']['id'], aid)
                closed = (await client.post('/api/chats/x/canvas/close')).json()
                self.assertFalse(closed['open'])
                self.assertEqual((await client.get(f'/api/chats/x/artifacts/{aid}')).status_code, 404)


if __name__ == '__main__':
    unittest.main(verbosity=2)
