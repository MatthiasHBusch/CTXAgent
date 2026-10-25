"""Offline sidebar hierarchy tests: isolated SQLite, no workers or providers."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ctxagent import config, store


class ChatHierarchyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ctxagent-hierarchy-')
        self.addCleanup(self.tmp.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        data = Path(self.tmp.name)
        for module, name, value in (
            (config, 'DATA_DIR', data),
            (config, 'BLOB_DIR', data / 'blobs'),
            (store, 'DB_PATH', data / 'chats.db'),
        ):
            self.stack.enter_context(patch.object(module, name, value))
        store.init()

        from ctxagent import delegation, jobs, llm
        self.guards = []
        for module, names in (
            (delegation, ('init', 'for_chat', 'status', 'reap', '_launch')),
            (jobs, ('start', 'status')),
            (llm, ('complete', 'stream_chat')),
        ):
            for name in names:
                self.guards.append(self.stack.enter_context(patch.object(
                    module, name, side_effect=AssertionError(f'{module.__name__}.{name} called'))))

    def tearDown(self):
        for guard in self.guards:
            guard.assert_not_called()

    def chat(self, title, updated, cfg=None, usage=None):
        cid = store.create_chat(title, cfg or {})
        with store._conn() as db:
            db.execute('UPDATE chats SET created_at=?,updated_at=?,usage=? WHERE id=?',
                       (updated - 1, updated, json.dumps(usage or {}), cid))
        return cid

    def task_table(self):
        # The list needs only these columns, not a task's state/cost/requests.
        with store._conn() as db:
            db.execute('CREATE TABLE agent_tasks (child TEXT UNIQUE, parent TEXT, root TEXT)')

    def task(self, child, parent, root):
        with store._conn() as db:
            db.execute('INSERT INTO agent_tasks VALUES (?,?,?)', (child, parent, root))

    def assert_hierarchy(self, rows, expected):
        self.assertEqual({row['id']: (row['agent_parent'], row['agent_root'])
                          for row in rows}, expected)

    def test_no_task_table_preserves_existing_fields_and_order(self):
        old = self.chat('Main', 10)
        usage = {'cost_usd': 1.25, 'requests': 3, 'prompt_tokens': 42}
        new = self.chat('Agent: not actually delegated', 20,
                        {'model': 'offline-test', 'temperature': 0.25}, usage)
        before = {cid: store.get_chat(cid) for cid in (old, new)}
        rows = store.list_chats()
        self.assertEqual(rows, [
            {'id': new, 'title': 'Agent: not actually delegated', 'created_at': 19,
             'updated_at': 20, 'model': 'offline-test', 'usage': usage,
             'agent_parent': None, 'agent_root': None},
            {'id': old, 'title': 'Main', 'created_at': 9, 'updated_at': 10,
             'model': None, 'usage': {}, 'agent_parent': None, 'agent_root': None},
        ])
        self.assertEqual({cid: store.get_chat(cid) for cid in before}, before)
        with store._conn() as db:
            self.assertIsNone(db.execute(
                "SELECT name FROM sqlite_master WHERE name='agent_tasks'").fetchone())

    def test_roots_children_and_grandchildren_have_exact_metadata(self):
        self.task_table()
        root = self.chat('Main', 10)
        child = self.chat('Research', 50)
        sibling = self.chat('Review', 20)
        grandchild = self.chat('Nested', 30)
        other_root = self.chat('Other main', 60)
        other_child = self.chat('Other task', 40)
        self.task(child, root, root)
        self.task(sibling, root, root)
        self.task(grandchild, child, root)
        self.task(other_child, other_root, other_root)
        rows = store.list_chats()
        self.assertEqual([row['id'] for row in rows],
                         [other_root, child, other_child, grandchild, sibling, root])
        self.assert_hierarchy(rows, {
            root: (None, None), child: (root, root), sibling: (root, root),
            grandchild: (child, root), other_root: (None, None),
            other_child: (other_root, other_root),
        })
        self.assertTrue(all(set(row) == {
            'id', 'title', 'created_at', 'updated_at', 'model', 'usage',
            'agent_parent', 'agent_root'} for row in rows))

    def test_missing_and_deleted_parent_or_root_metadata_is_retained(self):
        self.task_table()
        root = self.chat('Deleted root', 10)
        parent = self.chat('Deleted parent', 20)
        child = self.chat('Survivor', 30)
        orphan = self.chat('Never had a local parent', 40)
        self.task(parent, root, root)
        self.task(child, parent, root)
        self.task(orphan, 'missing-parent', 'missing-root')
        store.delete_chat(parent)
        store.delete_chat(root)
        self.assert_hierarchy(store.list_chats(), {
            child: (parent, root), orphan: ('missing-parent', 'missing-root'),
        })

    def test_titles_do_not_infer_hierarchy(self):
        self.task_table()
        root = self.chat('Agent: misleading root', 10)
        lookalike = self.chat('Agent: delegated-looking title', 20)
        child = self.chat('Agent: delegated-looking title', 30)
        ordinary_title = self.chat('Ordinary title', 40)
        self.task(child, root, root)
        self.task(ordinary_title, child, root)
        self.assert_hierarchy(store.list_chats(), {
            root: (None, None), lookalike: (None, None),
            child: (root, root), ordinary_title: (child, root),
        })

    def test_queries_are_constant_bulk_and_read_only(self):
        self.task_table()
        root = self.chat('Root', 1)
        connect = store.sqlite3.connect

        def listing_statements():
            statements = []

            def traced_connect(*args, **kwargs):
                db = connect(*args, **kwargs)
                db.set_trace_callback(statements.append)
                return db

            with patch.object(store.sqlite3, 'connect', side_effect=traced_connect):
                store.list_chats()
            # Connection setup may issue PRAGMA; listing must otherwise only read.
            queries = [sql for sql in statements if not sql.startswith('PRAGMA')]
            self.assertTrue(all(sql.startswith('SELECT') for sql in queries), queries)
            return queries

        small = listing_statements()
        for i in range(30):
            self.task(self.chat(f'Child {i}', i + 2), root, root)
        large = listing_statements()
        self.assertEqual(len(small), len(large))
        self.assertLessEqual(len(large), 2)

    async def api_chats(self):
        import httpx
        from ctxagent.server import app
        # ASGITransport does not run lifespan/startup (which initializes workers).
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://test') as client:
            response = await client.get('/api/chats')
        self.assertEqual(response.status_code, 200)
        return response.json()

    async def test_api_without_task_table(self):
        cid = self.chat('Agent: still an ordinary chat', 10,
                        {'model': 'offline-api'}, {'requests': 7})
        rows = await self.api_chats()
        self.assert_hierarchy(rows, {cid: (None, None)})
        self.assertEqual(rows[0]['model'], 'offline-api')
        self.assertEqual(rows[0]['usage'], {'requests': 7})
        with store._conn() as db:
            self.assertIsNone(db.execute(
                "SELECT name FROM sqlite_master WHERE name='agent_tasks'").fetchone())

    async def test_api_hierarchy_and_deleted_parent(self):
        self.task_table()
        root = self.chat('Main', 10)
        child = self.chat('Same title', 20)
        nested = self.chat('Nested', 30)
        ordinary = self.chat('Same title', 40)
        survivor = self.chat('Orphan', 50)
        self.task(child, root, root)
        self.task(nested, child, root)
        self.task(survivor, 'deleted-parent', 'deleted-root')
        rows = await self.api_chats()
        self.assertEqual([row['id'] for row in rows],
                         [survivor, ordinary, nested, child, root])
        self.assert_hierarchy(rows, {
            root: (None, None), child: (root, root), nested: (child, root),
            ordinary: (None, None), survivor: ('deleted-parent', 'deleted-root'),
        })
        store.delete_chat(child)
        self.assert_hierarchy(await self.api_chats(), {
            root: (None, None), nested: (child, root), ordinary: (None, None),
            survivor: ('deleted-parent', 'deleted-root'),
        })


if __name__ == '__main__':
    unittest.main()
