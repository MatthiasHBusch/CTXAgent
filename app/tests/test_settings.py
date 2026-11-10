"""Offline tests: API-key store, settings profiles, app prefs, summary-model
resolution and the /api/settings endpoints. Never touches the real credential
store (fake keyring / temp files) and never calls a provider (mock transport)."""
import asyncio
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from ctxagent import blobs, config, defaults, keys, profiles, store
from ctxagent.config import CompactionConfig, SessionConfig

OR_KEY = 'sk-or-v1-0123456789abcdef0123456789abcdefa1b2'
ANT_KEY = 'sk-ant-api03-ZYXWVUTSRQPONMLKJIHGFEDCBAwxyz'
KEY_NAMES = ('OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY', 'ANTHROPIC_WORKSPACE_ID')


class PasswordDeleteError(Exception):
    pass


class FakeKeyring:
    """keyring-compatible backend held in memory."""
    priority = 1

    def __init__(self):
        self.data = {}

    def get_password(self, service, name):
        return self.data.get((service, name))

    def set_password(self, service, name, value):
        self.data[(service, name)] = value

    def delete_password(self, service, name):
        if (service, name) not in self.data:
            raise PasswordDeleteError(name)
        del self.data[(service, name)]


class Isolated(unittest.TestCase):
    """Temp data folder, fake keyring, no provider keys in the environment."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for module, name, value in ((store, 'DB_PATH', self.root / 'test.db'),
                                    (config, 'DATA_DIR', self.root),
                                    (config, 'BLOB_DIR', self.root / 'blobs'),
                                    (blobs, 'BLOB_DIR', self.root / 'blobs')):
            p = patch.object(module, name, value); p.start(); self.addCleanup(p.stop)
        env = patch.dict(os.environ); env.start(); self.addCleanup(env.stop)
        for name in KEY_NAMES:
            os.environ.pop(name, None)
        self.keyring = FakeKeyring()
        keys.use_backend(keys.KeyringBackend(self.keyring))
        self.addCleanup(keys.use_backend, None)
        store.init()

    def client(self):
        from ctxagent.server import app
        return TestClient(app, base_url='http://127.0.0.1:8770')


# ---------------------------------------------------------------------------
# Key store
# ---------------------------------------------------------------------------

class KeyStoreTests(Isolated):
    def test_store_then_env_lookup_order(self):
        self.assertEqual(keys.lookup('OPENROUTER_API_KEY'), ('', None))
        os.environ['OPENROUTER_API_KEY'] = 'sk-or-v1-from-env-0000000000000000'
        self.assertEqual(keys.lookup('OPENROUTER_API_KEY')[1], 'env')
        keys.set_key('OPENROUTER_API_KEY', OR_KEY)
        self.assertEqual(keys.lookup('OPENROUTER_API_KEY'), (OR_KEY, 'store'))
        self.assertEqual(config.api_key(), OR_KEY)
        self.assertEqual(self.keyring.data[('CTXAgent', 'OPENROUTER_API_KEY')], OR_KEY)
        keys.delete_key('OPENROUTER_API_KEY')
        self.assertEqual(keys.lookup('OPENROUTER_API_KEY')[1], 'env')
        keys.delete_key('OPENROUTER_API_KEY')  # deleting an absent key is not an error

    def test_key_stored_under_the_former_name_is_used_and_copied(self):
        # Saved while the app was called PiAgent (Credential Manager service "PiAgent").
        self.keyring.data[('PiAgent', 'OPENROUTER_API_KEY')] = OR_KEY
        self.assertEqual(keys.lookup('OPENROUTER_API_KEY'), (OR_KEY, 'store'))
        self.assertEqual(self.keyring.data[('CTXAgent', 'OPENROUTER_API_KEY')], OR_KEY)
        keys.delete_key('OPENROUTER_API_KEY')        # gone under both names, not revived
        self.assertEqual(self.keyring.data, {})
        self.assertEqual(keys.lookup('OPENROUTER_API_KEY'), ('', None))

    def test_forget_all_clears_the_store_but_not_the_environment(self):
        # The uninstaller runs this when the user ticks "remove my data".
        keys.set_key('OPENROUTER_API_KEY', OR_KEY)
        keys.set_key('MISTRAL_API_KEY', 'mk-0123456789abcdef0123')
        os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-api03-from-env-000000000000'
        self.assertEqual(sorted(keys.forget_all()), ['MISTRAL_API_KEY', 'OPENROUTER_API_KEY'])
        self.assertEqual(self.keyring.data, {})
        self.assertEqual(keys.lookup('OPENROUTER_API_KEY'), ('', None))
        self.assertEqual(keys.lookup('ANTHROPIC_API_KEY')[1], 'env')
        self.assertEqual(keys.forget_all(), [])  # idempotent

    def test_config_key_functions_and_key_problem(self):
        self.assertIn('OPENROUTER_API_KEY', config.key_problem('anthropic/claude-sonnet-5.5'))
        self.assertIn('ANTHROPIC_API_KEY', config.key_problem('claude-sonnet-5-5'))
        keys.set_key('ANTHROPIC_API_KEY', ANT_KEY)
        keys.set_key('ANTHROPIC_WORKSPACE_ID', 'wrkspc_0123456789')
        self.assertEqual(config.anthropic_api_key(), ANT_KEY)
        self.assertEqual(config.anthropic_workspace_id(), 'wrkspc_0123456789')
        self.assertIsNone(config.key_problem('claude-sonnet-5-5'))
        # Store keys never leak into the process environment (shell tools inherit it).
        self.assertNotIn('ANTHROPIC_API_KEY', os.environ)

    def test_index_holds_names_not_values(self):
        keys.set_key('OPENROUTER_API_KEY', OR_KEY)
        keys.set_key('MISTRAL_API_KEY', 'mistral-secret-value-1234')
        index = (self.root / keys.INDEX_FILE).read_text(encoding='utf-8')
        self.assertIn('MISTRAL_API_KEY', index)
        self.assertNotIn(OR_KEY, index)
        self.assertIn('MISTRAL_API_KEY', [k['name'] for k in keys.list_keys()])
        for bad in ('lower', 'A', 'BAD-NAME', 'X' * 80):
            with self.assertRaises(ValueError):
                keys.set_key(bad, 'value')
        with self.assertRaises(ValueError):
            keys.set_key('OPENROUTER_API_KEY', 'two\nlines')
        with self.assertRaises(ValueError):
            keys.set_key('OPENROUTER_API_KEY', '   ')

    def test_masking(self):
        self.assertEqual(keys.mask(OR_KEY), 'sk-or-…a1b2')
        self.assertEqual(keys.mask(ANT_KEY), 'sk-ant-…wxyz')
        self.assertEqual(keys.mask('wrkspc_0123456789abcd'), 'wrkspc_…abcd')
        self.assertEqual(keys.mask('short'), '••••')
        self.assertEqual(keys.mask(''), '')
        listed = json.dumps(keys.list_keys())
        keys.set_key('OPENROUTER_API_KEY', OR_KEY)
        listed = json.dumps(keys.list_keys(), ensure_ascii=False)
        self.assertNotIn(OR_KEY, listed)
        self.assertIn('sk-or-…a1b2', listed)
        self.assertIn('error ' + keys.mask(OR_KEY), keys.redact('error ' + OR_KEY, OR_KEY))

    def test_file_fallback(self):
        keys.use_backend(keys.FileBackend())
        keys.set_key('OPENROUTER_API_KEY', OR_KEY)
        path = self.root / keys.FALLBACK_FILE
        self.assertTrue(path.exists())
        if os.name != 'nt':
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(keys.lookup('OPENROUTER_API_KEY'), (OR_KEY, 'store'))
        keys.delete_key('OPENROUTER_API_KEY')
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {})

    def test_env_only_backend_is_read_only(self):
        keys.use_backend(keys.EnvOnlyBackend())
        os.environ['ANTHROPIC_API_KEY'] = ANT_KEY
        self.assertEqual(keys.lookup('ANTHROPIC_API_KEY'), (ANT_KEY, 'env'))
        with self.assertRaises(keys.KeyStoreError):
            keys.set_key('ANTHROPIC_API_KEY', 'x' * 20)

    def test_backend_selection(self):
        from keyring.backends import fail
        with patch.dict(os.environ, {'CTXAGENT_KEYSTORE': 'env'}):
            self.assertIsInstance(keys._select_backend(), keys.EnvOnlyBackend)
        with patch.dict(os.environ, {'CTXAGENT_KEYSTORE': 'file'}):
            self.assertIsInstance(keys._select_backend(), keys.FileBackend)
        with patch.dict(os.environ, {'CTXAGENT_KEYSTORE': 'auto'}), \
                patch('keyring.get_keyring', return_value=fail.Keyring()):
            self.assertIsInstance(keys._select_backend(), keys.FileBackend)
        with patch.dict(os.environ, {'CTXAGENT_KEYSTORE': 'auto'}), \
                patch('keyring.get_keyring', return_value=FakeKeyring()):
            self.assertIsInstance(keys._select_backend(), keys.KeyringBackend)
        with patch.dict(os.environ, {'CTXAGENT_KEYSTORE': 'keyring'}), \
                patch('keyring.get_keyring', return_value=fail.Keyring()):
            with self.assertRaises(keys.KeyStoreError):
                keys._select_backend()

    def test_suite_default_never_uses_real_store(self):
        # tests/__init__.py pins the subprocess default for every offline suite.
        import tests  # noqa: F401
        self.assertEqual(os.environ.get('CTXAGENT_KEYSTORE'), 'env')

    def test_broken_store_falls_back_to_env(self):
        class Broken(FakeKeyring):
            def get_password(self, service, name):
                raise RuntimeError('locked')
        keys.use_backend(keys.KeyringBackend(Broken()))
        os.environ['OPENROUTER_API_KEY'] = OR_KEY
        self.assertEqual(keys.lookup('OPENROUTER_API_KEY'), (OR_KEY, 'env'))


class KeyCheckTests(Isolated):
    def run_check(self, name, handler, value=None):
        seen = []

        def record(request):
            seen.append(request)
            return handler(request)
        result = asyncio.run(keys.test_key(name, value, transport=httpx.MockTransport(record)))
        return result, seen

    def test_openrouter_ok_and_rejected(self):
        keys.set_key('OPENROUTER_API_KEY', OR_KEY)
        result, seen = self.run_check('OPENROUTER_API_KEY', lambda r: httpx.Response(
            200, json={'data': {'label': 'sk-or-v1-012...a1b2', 'usage': 1.5, 'limit': None}}))
        self.assertTrue(result['ok'])
        self.assertEqual(str(seen[0].url), keys.OPENROUTER_KEY_URL)
        self.assertEqual(seen[0].headers['authorization'], 'Bearer ' + OR_KEY)
        self.assertNotIn(OR_KEY, str(seen[0].url))
        self.assertNotIn(OR_KEY, json.dumps(result))
        result, _ = self.run_check('OPENROUTER_API_KEY', lambda r: httpx.Response(401, json={'error': {}}))
        self.assertFalse(result['ok'])
        self.assertIn('rejected', result['message'])

    def test_unsaved_value_is_tested(self):
        result, seen = self.run_check('OPENROUTER_API_KEY', lambda r: httpx.Response(200, json={'data': {}}), OR_KEY)
        self.assertTrue(result['ok'])
        self.assertEqual(seen[0].headers['authorization'], 'Bearer ' + OR_KEY)
        self.assertEqual(keys.get('OPENROUTER_API_KEY'), '')  # testing does not store

    def test_anthropic_workspace_header_and_hint(self):
        keys.set_key('ANTHROPIC_API_KEY', ANT_KEY)
        keys.set_key('ANTHROPIC_WORKSPACE_ID', 'wrkspc_01abcdef')
        result, seen = self.run_check('ANTHROPIC_API_KEY', lambda r: httpx.Response(200, json={'data': []}))
        self.assertTrue(result['ok'])
        self.assertTrue(str(seen[0].url).startswith(keys.ANTHROPIC_MODELS_URL))
        self.assertEqual(seen[0].headers['x-api-key'], ANT_KEY)
        self.assertEqual(seen[0].headers['anthropic-workspace-id'], 'wrkspc_01abcdef')
        keys.delete_key('ANTHROPIC_WORKSPACE_ID')
        result, _ = self.run_check('ANTHROPIC_API_KEY', lambda r: httpx.Response(
            400, json={'error': {'message': 'requests with this key must include the anthropic-workspace-id header ' + ANT_KEY}}))
        self.assertFalse(result['ok'])
        self.assertIn('ANTHROPIC_WORKSPACE_ID', result['message'])
        self.assertNotIn(ANT_KEY, json.dumps(result))

    def test_missing_and_untestable(self):
        result, seen = self.run_check('OPENROUTER_API_KEY', lambda r: httpx.Response(200))
        self.assertFalse(result['ok'])
        self.assertEqual(seen, [])
        self.assertFalse(asyncio.run(keys.test_key('ANTHROPIC_WORKSPACE_ID'))['ok'])

    def test_network_error_is_redacted(self):
        def boom(request):
            raise httpx.ConnectError('failed for ' + OR_KEY)
        result, _ = self.run_check('OPENROUTER_API_KEY', boom, OR_KEY)
        self.assertFalse(result['ok'])
        self.assertNotIn(OR_KEY, json.dumps(result))


# ---------------------------------------------------------------------------
# Summary model resolution
# ---------------------------------------------------------------------------

class SummaryModelTests(Isolated):
    def test_default_is_same_as_chat_model(self):
        self.assertEqual(CompactionConfig().summary_model, '')
        cfg = SessionConfig(model='anthropic/claude-sonnet-5.5', pricing_mode='prefer_flex')
        self.assertEqual(cfg.effective_summary_model(), 'anthropic/claude-sonnet-5.5')
        self.assertEqual(cfg.effective_summary_pricing_mode(), 'prefer_flex')

    def test_old_chat_keeps_explicit_summary_model(self):
        old = {'model': 'anthropic/claude-sonnet-5', 'pricing_mode': 'prefer_flex',
               'compaction': {'summary_model': 'openai/gpt-5.6-luna', 'summary_pricing_mode': 'standard'}}
        cfg = SessionConfig.from_dict(old)
        self.assertEqual(cfg.effective_summary_model(), 'openai/gpt-5.6-luna')
        self.assertEqual(cfg.effective_summary_pricing_mode(), 'standard')
        self.assertEqual(SessionConfig.from_dict(cfg.to_dict()).compaction.summary_model, 'openai/gpt-5.6-luna')
        self.assertEqual(SessionConfig.from_dict({'compaction': {'summary_model': '  '}}).compaction.summary_model, '')
        with self.assertRaises(ValueError):
            CompactionConfig.from_dict({'summary_model': 7})

    def test_session_routes_summaries_through_resolver(self):
        from ctxagent.agent import AgentSession
        from ctxagent.segments import Conversation
        cfg = SessionConfig(model='chat/model', pricing_mode='flex_only')
        s = AgentSession('resolve', cfg, Conversation())
        self.assertEqual(s.engine.summary_model(), 'chat/model')
        with patch('ctxagent.llm.complete', new=AsyncMock(return_value='ok')) as complete:
            asyncio.run(s._complete(model='', messages=[], max_tokens=10))
        kwargs = complete.call_args.kwargs
        self.assertEqual(kwargs['model'], 'chat/model')
        self.assertEqual(kwargs['pricing_mode'], 'flex_only')
        # A caller naming a different model gets the summary tier, not the chat's.
        with patch('ctxagent.llm.complete', new=AsyncMock(return_value='ok')) as complete:
            asyncio.run(s._complete(model='other/model', messages=[], max_tokens=10))
        self.assertEqual(complete.call_args.kwargs['pricing_mode'], 'standard')
        # Changing the chat config (as PATCH /config does) is followed immediately.
        s.cfg = SessionConfig.from_dict({**cfg.to_dict(), 'compaction': {'summary_model': 'cheap/summariser',
                                                                         'summary_pricing_mode': 'standard'}})
        s.engine.cfg = s.cfg.compaction
        self.assertEqual(s.engine.summary_model(), 'cheap/summariser')
        with patch('ctxagent.llm.complete', new=AsyncMock(return_value='ok')) as complete:
            asyncio.run(s._complete(model=s.engine.summary_model(), messages=[], max_tokens=10))
        self.assertEqual(complete.call_args.kwargs['pricing_mode'], 'standard')

    def test_no_call_site_reads_the_raw_field(self):
        root = Path(__file__).resolve().parent.parent / 'ctxagent'
        offenders = []
        for file in root.rglob('*.py'):
            if file.name == 'config.py':
                continue
            for i, line in enumerate(file.read_text(encoding='utf-8').splitlines(), 1):
                if re.search(r'compaction\.summary_model|cfg\.summary_model', line) and 'return self.cfg.summary_model' not in line:
                    offenders.append(f'{file.name}:{i}')
        self.assertEqual(offenders, [])

    def test_send_checks_the_key_of_the_effective_summariser(self):
        os.environ['ANTHROPIC_API_KEY'] = ANT_KEY
        cfg = SessionConfig(model='claude-haiku-5-5')
        cfg.compaction.summary_model = 'openai/gpt-5.6-luna'
        cid = store.create_chat('explicit', cfg.to_dict())
        r = self.client().post(f'/api/chats/{cid}/send', json={'message': 'hi'})
        self.assertEqual(r.status_code, 400)
        self.assertIn('compaction model openai/gpt-5.6-luna', r.json()['detail'])
        same = SessionConfig(model='claude-haiku-5-5')
        self.assertIsNone(config.key_problem(same.effective_summary_model()))


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class ProfileTests(Isolated):
    def test_builtins_are_valid_and_read_only(self):
        listed = profiles.list_profiles()
        builtins = [p for p in listed if p['builtin']]
        self.assertEqual(len(builtins), 7)
        models = {p['effective']['model'] for p in builtins}
        self.assertEqual(models, {'anthropic/claude-opus-5.5', 'anthropic/claude-sonnet-5.5',
                                  'anthropic/claude-fable-5.1', 'openai/gpt-6.1-sol', 'openai/gpt-6-astra',
                                  'openai/gpt-6.1-terra', 'google/gemini-4-argon'})
        for p in builtins:
            self.assertIsNone(p['error'])
            self.assertEqual(p['effective']['compaction']['summary_model'], '')
            self.assertNotIn('workdirs', p['effective'])
            with self.assertRaises(profiles.ReadOnly):
                profiles.update(p['id'], name='x')
            with self.assertRaises(profiles.ReadOnly):
                profiles.delete(p['id'])
        # Models that manage their context reliably are self-steered at 1M; the others run
        # on the engine at 400k. Every starter enables every switchable tool.
        from ctxagent import tools
        steered = {'anthropic/claude-fable-5.1', 'anthropic/claude-opus-5.5', 'openai/gpt-6.1-sol',
                   'openai/gpt-6-astra', 'anthropic/claude-sonnet-5.5', 'openai/gpt-6.1-terra'}
        every = {t['name'] for t in tools.catalogue()}
        for p in builtins:
            cfg = SessionConfig.from_dict(p['effective'])
            on = cfg.model in steered
            self.assertEqual(cfg.compaction.self_steered, on, cfg.model)
            window = 700_000 if cfg.model == 'anthropic/claude-opus-5.5' else 1_000_000 if on else 400_000
            self.assertEqual(cfg.context_window, window, cfg.model)   # Opus 5.5 = the app default
            self.assertEqual(set(cfg.enabled_tools), every, cfg.model)
            self.assertEqual('compact_context' in cfg.active_tools(), on, cfg.model)

    def test_crud_partial_storage_and_validation(self):
        p = profiles.create('Mine', {'model': 'x/y', 'compaction': {'deep_min_gain_tokens': 12345, 'bogus': 1},
                                     'bogus_top': 1, 'workdirs': ['C:/should/not/stick']})
        self.assertEqual(p['config'], {'model': 'x/y', 'compaction': {'deep_min_gain_tokens': 12345}})
        self.assertEqual(p['effective']['context_window'], SessionConfig().context_window)
        self.assertEqual(profiles.create('Mine')['name'], 'Mine (2)')
        with self.assertRaises(profiles.ProfileError):
            profiles.create('Bad', {'pricing_mode': 'cheap'})
        with self.assertRaises(profiles.ProfileError):
            profiles.create('Bad', {'compaction': {'deep_target_ratio': 3}})
        with self.assertRaises(profiles.ProfileError):
            profiles.create('   ')
        updated = profiles.update(p['id'], name='Renamed', config_dict={'model': 'a/b'})
        self.assertEqual((updated['name'], updated['config']), ('Renamed', {'model': 'a/b'}))
        dup = profiles.duplicate('builtin-sonnet-5-5')
        self.assertFalse(dup['builtin'])
        self.assertEqual(dup['effective'], profiles.get('builtin-sonnet-5-5')['effective'])
        profiles.delete(p['id'])
        with self.assertRaises(profiles.NotFound):
            profiles.get(p['id'])
        self.assertFalse(any('should/not' in json.dumps(x) for x in profiles.list_profiles()))

    def test_future_fields_are_carried_generically(self):
        p = profiles.create('Partial', {'model': 'x/y'})
        eff = profiles.get(p['id'])['effective']
        # Everything SessionConfig knows appears, although the profile stored one key.
        self.assertEqual(set(eff), set(SessionConfig().to_dict()) - profiles.EXCLUDED)
        self.assertEqual(set(eff['compaction']), set(CompactionConfig().to_dict()))

    def test_export_import_round_trip(self):
        keys.set_key('OPENROUTER_API_KEY', OR_KEY)
        a = profiles.create('Alpha', {'model': 'a/a', 'system_prompt': 'Be terse.'})
        data = profiles.export()
        self.assertEqual(data['format'], 'ctxagent-profiles')
        self.assertEqual([x['name'] for x in data['profiles']], ['Alpha'])
        self.assertNotIn('id', data['profiles'][0])
        self.assertNotIn(OR_KEY, json.dumps(data))
        created = profiles.import_profiles(json.loads(json.dumps(data)))
        self.assertEqual(created[0]['name'], 'Alpha (2)')
        self.assertNotEqual(created[0]['id'], a['id'])
        self.assertEqual(created[0]['effective'], a['effective'])
        self.assertEqual(len(profiles.export(['builtin-opus-5-5'])['profiles']), 1)
        with self.assertRaises(profiles.ProfileError):
            profiles.import_profiles({'profiles': [{'name': 'ok', 'config': {}}, {'name': 'bad', 'config': {'pricing_mode': 'x'}}]})
        self.assertEqual(len([p for p in profiles.list_profiles() if not p['builtin']]), 2)  # nothing half-imported
        with self.assertRaises(profiles.ProfileError):
            profiles.import_profiles({'format': 'other', 'profiles': []})

    def test_modified_detection_ignores_workdirs(self):
        cfg = profiles.config_for_chat('builtin-opus-5-5').to_dict()
        self.assertEqual(cfg['profile_id'], 'builtin-opus-5-5')
        status = profiles.chat_status(cfg)
        self.assertFalse(status['modified'])
        cfg['workdirs'] = [str(self.root)]
        self.assertFalse(profiles.chat_status(cfg)['modified'])
        cfg['temperature'] = 0.2
        cfg['compaction']['deep_min_gain_tokens'] = 1234
        status = profiles.chat_status(cfg)
        self.assertTrue(status['modified'])
        self.assertEqual(status['differences'], ['compaction.deep_min_gain_tokens', 'temperature'])
        legacy = SessionConfig().to_dict()
        self.assertEqual(profiles.chat_status(legacy)['profile'], None)
        legacy['profile_id'] = 'p-gone'
        self.assertTrue(profiles.chat_status(legacy)['missing'])

    def test_prefs_validation_and_defaults(self):
        self.assertEqual(profiles.load_prefs(), profiles.DEFAULT_PREFS)
        for bad in ({'zoom': 300}, {'zoom': True}, {'new_chat_mode': 'x'}, {'theme': '<script>'},
                    {'default_profile_id': 'p-missing'}, {'default_workdir': str(self.root / 'nope')},
                    {'default_workdir': 'relative/path'}, {'unknown': 1}):
            with self.assertRaises(profiles.ProfileError, msg=str(bad)):
                profiles.update_prefs(bad)
        prefs = profiles.update_prefs({'zoom': 110, 'theme': 'light', 'default_workdir': str(self.root),
                                       'default_profile_id': 'builtin-terra-6-1'})
        self.assertEqual((prefs['zoom'], prefs['theme']), (110, 'light'))
        self.assertEqual(defaults.get()['model'], 'openai/gpt-6.1-terra')
        self.assertEqual(defaults.create_config().workdirs, [str(self.root)])
        self.assertEqual(defaults.create_config().profile_id, 'builtin-terra-6-1')
        # The explicit "use as new-chat defaults" action wins over the default profile.
        defaults.save(SessionConfig(model='saved/model').to_dict())
        self.assertEqual(profiles.load_prefs()['default_profile_id'], '')
        self.assertEqual(defaults.get()['model'], 'saved/model')
        p = profiles.create('Default me')
        profiles.update_prefs({'default_profile_id': p['id']})
        profiles.delete(p['id'])
        self.assertEqual(profiles.load_prefs()['default_profile_id'], '')

    def test_no_secrets_in_settings_files(self):
        keys.set_key('OPENROUTER_API_KEY', OR_KEY)
        profiles.create('Any', {'model': 'x/y'})
        profiles.update_prefs({'zoom': 90})
        for name in (profiles.PROFILES_FILE, profiles.PREFS_FILE, keys.INDEX_FILE):
            self.assertNotIn(OR_KEY, (self.root / name).read_text(encoding='utf-8'))
        self.assertNotIn(OR_KEY, json.dumps(SessionConfig().to_dict()))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class EndpointTests(Isolated):
    def setUp(self):
        super().setUp()
        self.c = self.client()

    def test_key_endpoints_mask_and_never_echo(self):
        r = self.c.put('/api/settings/keys/OPENROUTER_API_KEY', json={'value': OR_KEY})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn(OR_KEY, r.text)
        entry = r.json()['key']
        self.assertEqual((entry['masked'], entry['source'], entry['stored']), ('sk-or-…a1b2', 'store', True))
        listing = self.c.get('/api/settings/keys')
        self.assertNotIn(OR_KEY, listing.text)
        self.assertTrue(listing.json()['status']['openrouter'])
        health = self.c.get('/api/health')
        self.assertTrue(health.json()['has_openrouter_key'])
        self.assertNotIn(OR_KEY, health.text)
        self.assertEqual(self.c.put('/api/settings/keys/bad-name', json={'value': 'x'}).status_code, 422)
        self.assertEqual(self.c.put('/api/settings/keys/OPENROUTER_API_KEY', json={'value': ''}).status_code, 422)
        too_long = 'sk-or-v1-' + 'q' * 5000
        r = self.c.put('/api/settings/keys/OPENROUTER_API_KEY', json={'value': too_long})
        self.assertEqual(r.status_code, 422)
        self.assertNotIn('qqqqqqqq', r.text)  # rejected secrets are not echoed back
        r = self.c.post('/api/settings/keys/OPENROUTER_API_KEY/test', json={'value': too_long})
        self.assertEqual(r.status_code, 422)
        self.assertNotIn('qqqqqqqq', r.text)
        r = self.c.delete('/api/settings/keys/OPENROUTER_API_KEY')
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()['key']['configured'])

    def test_key_test_endpoint_uses_mocked_transport(self):
        keys.set_key('OPENROUTER_API_KEY', OR_KEY)
        original = keys.test_key

        async def mocked(name, value=None, transport=None):
            return await original(name, value, transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={'data': {'usage': 0}})))
        with patch.object(keys, 'test_key', mocked):
            r = self.c.post('/api/settings/keys/OPENROUTER_API_KEY/test', json={})
        self.assertTrue(r.json()['ok'])
        self.assertNotIn(OR_KEY, r.text)

    def test_store_failure_is_503_without_value(self):
        class Refusing(FakeKeyring):
            def set_password(self, service, name, value):
                raise RuntimeError('denied ' + value)
        keys.use_backend(keys.KeyringBackend(Refusing()))
        r = self.c.put('/api/settings/keys/OPENROUTER_API_KEY', json={'value': OR_KEY})
        self.assertEqual(r.status_code, 503)
        self.assertNotIn(OR_KEY, r.text)

    def test_local_only_guard(self):
        self.assertEqual(self.c.get('/api/settings').status_code, 200)
        self.assertEqual(self.c.get('/api/settings', headers={'host': 'evil.example'}).status_code, 403)
        self.assertEqual(self.c.get('/api/settings/keys', headers={'origin': 'https://evil.example'}).status_code, 403)
        self.assertEqual(self.c.get('/api/settings/keys', headers={'origin': 'null'}).status_code, 403)
        self.assertEqual(self.c.get('/api/settings', headers={'origin': 'http://127.0.0.1:8770'}).status_code, 200)
        self.assertEqual(self.c.get('/api/settings', headers={'host': 'piagent.localhost:9000'}).status_code, 200)

    def test_profile_lifecycle_through_chats(self):
        c = self.c
        r = c.post('/api/chats', json={'title': 'From profile', 'profile_id': 'builtin-sonnet-5-5'})
        self.assertEqual(r.status_code, 200)
        cid = r.json()['id']
        chat = store.get_chat(cid)['config']
        self.assertEqual(chat['model'], 'anthropic/claude-sonnet-5.5')
        self.assertEqual(chat['profile_id'], 'builtin-sonnet-5-5')
        self.assertFalse(c.get(f'/api/settings/chats/{cid}/profile').json()['modified'])
        self.assertEqual(c.post('/api/chats', json={'profile_id': 'p-missing'}).status_code, 404)

        # Edit in the chat -> modified; reset -> back to the profile, workdirs kept.
        self.assertEqual(c.patch(f'/api/chats/{cid}/config', json={'config': {
            'reasoning_effort': 'low', 'workdirs': [str(self.root)]}}).status_code, 200)
        status = c.get(f'/api/settings/chats/{cid}/profile').json()
        self.assertEqual((status['modified'], status['differences']), (True, ['reasoning_effort']))
        r = c.post(f'/api/settings/chats/{cid}/profile/reset', json={})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['config']['reasoning_effort'], 'medium')
        self.assertEqual(r.json()['config']['workdirs'], [str(self.root)])
        self.assertFalse(r.json()['status']['modified'])

        # Built-ins are read-only; save as new profile links the chat.
        c.patch(f'/api/chats/{cid}/config', json={'config': {'temperature': 0.4}})
        self.assertEqual(c.post(f'/api/settings/chats/{cid}/profile/save').status_code, 409)
        r = c.post(f'/api/settings/chats/{cid}/profile/save-as', json={'name': 'My Sonnet'})
        self.assertEqual(r.status_code, 200)
        mine = r.json()['profile']
        self.assertTrue(r.json()['linked'])
        self.assertEqual(store.get_chat(cid)['config']['profile_id'], mine['id'])
        self.assertFalse(r.json()['status']['modified'])
        self.assertEqual(mine['effective']['temperature'], 0.4)

        # A second chat on the same profile; changing the profile does not rewrite it.
        other = c.post('/api/chats', json={'profile_id': mine['id']}).json()['id']
        c.patch(f'/api/chats/{cid}/config', json={'config': {'max_output_tokens': 9000}})
        r = c.post(f'/api/settings/chats/{cid}/profile/save')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(profiles.get(mine['id'])['effective']['max_output_tokens'], 9000)
        self.assertEqual(store.get_chat(other)['config']['max_output_tokens'], 32000)
        other_status = c.get(f'/api/settings/chats/{other}/profile').json()
        self.assertEqual(other_status['differences'], ['max_output_tokens'])

        # Apply a different profile explicitly.
        r = c.post(f'/api/settings/chats/{other}/profile/reset', json={'profile_id': 'builtin-terra-6-1'})
        self.assertEqual(r.json()['config']['model'], 'openai/gpt-6.1-terra')
        self.assertEqual(r.json()['config']['profile_id'], 'builtin-terra-6-1')

        # Deleting the profile leaves the chat alone and marks it missing.
        self.assertEqual(c.delete(f'/api/settings/profiles/{mine["id"]}').status_code, 200)
        self.assertEqual(store.get_chat(cid)['config']['max_output_tokens'], 9000)
        self.assertTrue(c.get(f'/api/settings/chats/{cid}/profile').json()['missing'])

    def test_busy_chat_defers_reset_to_the_next_request(self):
        from ctxagent import agent
        cid = self.c.post('/api/chats', json={'profile_id': 'builtin-opus-5-5'}).json()['id']
        s = agent.get_session(cid)
        self.c.patch(f'/api/chats/{cid}/config', json={'config': {'max_output_tokens': 9000}})
        s.turn_running = True
        try:
            r = self.c.post(f'/api/settings/chats/{cid}/profile/reset', json={})
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.json()['pending'])
            self.assertNotEqual(r.json()['config']['max_output_tokens'], 9000)
            self.assertEqual(s.cfg.max_output_tokens, 9000)              # the running turn keeps it
            self.assertTrue(self.c.get(f'/api/chats/{cid}').json()['config_pending'])
        finally:
            s.turn_running = False
        s._apply_pending_config()
        self.assertEqual(store.get_chat(cid)['config']['max_output_tokens'], r.json()['config']['max_output_tokens'])

    def test_profile_crud_endpoints(self):
        c = self.c
        r = c.post('/api/settings/profiles', json={'name': 'Fresh', 'config': {'model': 'x/y'}})
        self.assertEqual(r.status_code, 200)
        pid = r.json()['profile']['id']
        self.assertEqual(c.patch(f'/api/settings/profiles/{pid}', json={'config': {'pricing_mode': 'nope'}}).status_code, 422)
        self.assertEqual(c.patch('/api/settings/profiles/builtin-opus-5-5', json={'name': 'x'}).status_code, 409)
        self.assertEqual(c.get('/api/settings/profiles/p-nope').status_code, 404)
        r = c.patch(f'/api/settings/profiles/{pid}', json={'name': 'Fresh 2', 'description': 'd', 'config': {'model': 'z/z'}})
        self.assertEqual(r.json()['profile']['effective']['model'], 'z/z')
        dup = c.post('/api/settings/profiles/builtin-fable-5-1/duplicate', json={}).json()['profile']
        self.assertEqual(dup['effective']['model'], 'anthropic/claude-fable-5.1')
        exported = c.post('/api/settings/profiles/export', json={}).json()
        self.assertEqual(len(exported['profiles']), 2)
        r = c.post('/api/settings/profiles/import', json={'data': exported})
        self.assertEqual(len(r.json()['created']), 2)
        self.assertEqual(c.post('/api/settings/profiles/import', json={'data': {'nope': 1}}).status_code, 422)
        cid = store.create_chat('src', SessionConfig(model='from/chat').to_dict())
        r = c.post('/api/settings/profiles', json={'name': 'From chat', 'from_chat': cid})
        self.assertEqual(r.json()['profile']['effective']['model'], 'from/chat')

    def test_prefs_endpoint_and_new_chat_modes(self):
        c = self.c
        self.assertEqual(c.patch('/api/settings/prefs', json={'prefs': {'zoom': 10}}).status_code, 422)
        r = c.patch('/api/settings/prefs', json={'prefs': {'default_profile_id': 'builtin-terra-6-1',
                                                            'default_workdir': str(self.root)}})
        self.assertEqual(r.status_code, 200)
        cid = c.post('/api/chats', json={'title': 'x'}).json()['id']
        cfg = store.get_chat(cid)['config']
        self.assertEqual((cfg['model'], cfg['workdirs']), ('openai/gpt-6.1-terra', [str(self.root)]))
        source = store.create_chat('src', SessionConfig(model='source/model').to_dict())
        # Default mode: the New chat button (source_chat) still copies the open chat.
        inherit = c.post('/api/chats', json={'source_chat': source}).json()['id']
        self.assertEqual(store.get_chat(inherit)['config']['model'], 'source/model')
        c.patch('/api/settings/prefs', json={'prefs': {'new_chat_mode': 'profile'}})
        fresh = c.post('/api/chats', json={'source_chat': source}).json()['id']
        self.assertEqual(store.get_chat(fresh)['config']['model'], 'openai/gpt-6.1-terra')
        overview = c.get('/api/settings').json()
        self.assertEqual(overview['product'], 'CTXAgent')
        self.assertEqual(Path(overview['data_dir']), self.root.resolve())

    def test_open_folder_is_restricted(self):
        from ctxagent import settings_api
        opened = []
        with patch.object(settings_api, '_open_in_file_manager', opened.append):
            self.assertEqual(self.c.post('/api/settings/open-folder', json={'which': 'logs'}).status_code, 200)
            self.assertEqual(self.c.post('/api/settings/open-folder', json={'which': 'C:/Windows'}).status_code, 422)
        self.assertEqual(opened, [self.root / 'logs'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
