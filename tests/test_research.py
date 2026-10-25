"""Deterministic metadata/evidence/manifest tests; network responses are mocked."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from ctxagent import research, experiments, tools

ITEM = {'DOI': '10.1234/example', 'title': ['A & B: 50% efficiency'],
        'author': [{'family': 'Example', 'given': 'Alice'}],
        'published': {'date-parts': [[2020,1,1]]}, 'type': 'journal-article',
        'container-title': ['Test Journal']}


class ResearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sb = tools.Sandbox([str(self.root)])

    def test_doi_validation_and_bibtex_escape(self):
        self.assertEqual(research.doi_value('https://doi.org/10.1234/example'), '10.1234/example')
        with self.assertRaises(ValueError): research.doi_value('not a DOI')
        bib = research.bibtex(research.metadata(ITEM), 'Example2020')
        self.assertIn(r'\&', bib)
        self.assertIn(r'50\%', bib)
        self.assertIn('Alice', bib)
        with self.assertRaises(ValueError): research.bibtex(research.metadata(ITEM), r'bad\input{x}')

    async def test_lookup_and_search_fixed_endpoint(self):
        requests = []
        def handle(request):
            requests.append(request)
            message = {'total-results': 1, 'items': [ITEM]} if request.url.params else ITEM
            return httpx.Response(200, json={'status': 'ok', 'message': message})
        real_client = httpx.AsyncClient
        def client(**kw): return real_client(transport=httpx.MockTransport(handle), **kw)
        with patch.object(research.httpx, 'AsyncClient', client):
            item = await research.lookup('10.1234/example')
            hits = await research.search('test title', 1)
        self.assertEqual(item['doi'], ITEM['DOI'])
        self.assertEqual(len(hits['results']), 1)
        self.assertTrue(all(r.url.host == 'api.crossref.org' for r in requests))

    async def test_wrong_doi_and_oversized_response_fail(self):
        async def wrong(*args, **kw): return {**ITEM, 'DOI': '10.9999/other'}
        with patch.object(research, 'request_crossref', wrong):
            with self.assertRaisesRegex(ValueError, 'does not match'):
                await research.lookup('10.1234/example')
        real_client = httpx.AsyncClient
        def client(**kw): return real_client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b'x' * 100)), **kw)
        with patch.object(research.httpx, 'AsyncClient', client), patch.object(research, 'MAX_RESPONSE', 10):
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                await research.request_crossref('')

    def test_text_evidence_whitespace_hash_and_no_false_claim_validation(self):
        p = self.root / 'paper.txt'
        p.write_text('The measurement\n was 3.2 volts. More context.', encoding='utf-8')
        record = research.record_evidence(self.sb, str(p), 'measurement was 3.2 volts.',
                                         'A proposed claim', str(self.root/'evidence'))
        self.assertTrue(Path(record['record_path']).is_file())
        self.assertIn('claim_support_requires_review', record['status'])
        self.assertEqual(len(record['source_sha256']), 64)
        with self.assertRaisesRegex(ValueError, 'not found'):
            research.record_evidence(self.sb, str(p), 'invented quotation', 'claim', str(self.root/'evidence'))
        self.assertEqual(len(list((self.root/'evidence').glob('*.json'))), 1)

    def test_pdf_requires_page_and_validates_quote(self):
        import fitz
        p = self.root/'paper.pdf'
        with fitz.open() as doc:
            doc.new_page().insert_text((50,50), 'First page')
            doc.new_page().insert_text((50,50), 'Measured value is 42.')
            doc.save(p)
        with self.assertRaises(ValueError):
            research.record_evidence(self.sb,str(p),'Measured value','claim',str(self.root/'e'))
        record = research.record_evidence(self.sb,str(p),'Measured value is 42.','claim',str(self.root/'e'),page=2)
        self.assertEqual(record['locator']['page'],2)
        with self.assertRaises(ValueError):
            research.record_evidence(self.sb,str(p),'Measured value','claim',str(self.root/'e'),page=1)

    def test_manifest_no_overwrite_and_drift(self):
        code=self.root/'run.py'; code.write_text('print(1)')
        data=self.root/'data.csv'; data.write_text('x\n1\n')
        plan=self.root/'planned.json'; final=self.root/'result.json'
        record=experiments.create(self.sb,str(plan),str(self.root),'python run.py',[str(code)],[str(data)],{'x':1},{'numpy':42})
        self.assertEqual(record['phase'],'planned')
        with self.assertRaises(FileExistsError):
            experiments.create(self.sb,str(plan),str(self.root),'python run.py',[str(code)],[],{}, {})
        data.write_text('x\n2\n')
        result=experiments.finalize(self.sb,str(plan),str(final),[str(self.root/'missing.csv')],'owner')
        self.assertEqual(len(result['input_or_code_drift']),1)
        self.assertEqual(len(result['missing_outputs']),1)
        self.assertFalse(result['job_success_and_declared_inputs_unchanged'])
        self.assertEqual(json.loads(plan.read_text())['phase'],'planned')

    def test_manifest_job_comparison_and_terminal_requirement(self):
        code=self.root/'run.py'; code.write_text('print(1)')
        output=self.root/'out.txt'; output.write_text('result')
        plan=self.root/'plan.json'
        experiments.create(self.sb,str(plan),str(self.root),'python run.py',[str(code)],[],{}, {})
        status={'state':'running','command':'python run.py','cwd':str(self.root)}
        with patch.object(experiments.jobs,'status',return_value=status):
            with self.assertRaises(ValueError):
                experiments.finalize(self.sb,str(plan),str(self.root/'done.json'),[str(output)],'owner','job')
        status['state']='succeeded'; status['command']='different command'
        with patch.object(experiments.jobs,'status',return_value=status):
            result=experiments.finalize(self.sb,str(plan),str(self.root/'done.json'),[str(output)],'owner','job')
        self.assertFalse(result['command_matches_job'])
        self.assertFalse(result['job_success_and_declared_inputs_unchanged'])

    def test_environment_does_not_capture_environment_variables(self):
        with patch.dict('os.environ',{'TEST_SECRET_TOKEN':'do-not-record'}):
            report=experiments.environment()
        self.assertNotIn('do-not-record',json.dumps(report))
        self.assertIn('python',report)


if __name__ == '__main__':
    unittest.main(verbosity=2)
