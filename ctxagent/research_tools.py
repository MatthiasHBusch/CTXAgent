"""Opt-in scholarly discovery, textual evidence and reproducibility interfaces."""
import asyncio
import json
from . import experiments, research


def register(registry, ToolSpec, ToolResult, params):
    def result(data):
        return ToolResult(text=json.dumps(data, ensure_ascii=False, indent=2))

    async def scholarly_search(sb, query, limit=5):
        return result(await research.search(query, limit))

    async def doi_lookup(sb, doi, citation_key=None):
        item = await research.lookup(doi)
        if citation_key:
            item['bibtex'] = research.bibtex(item, citation_key)
        return result(item)

    async def record_evidence(sb, source, quotation, claim, output_dir, page=None, doi=None):
        return result(await asyncio.to_thread(research.record_evidence, sb, source, quotation,
                                             claim, output_dir, page, doi))

    async def environment_report(sb):
        return result(await asyncio.to_thread(experiments.environment))

    async def experiment_manifest(sb, output, cwd, command, code_files, inputs, parameters, seeds):
        return result(await asyncio.to_thread(experiments.create, sb, output, cwd, command,
                                             code_files, inputs, parameters, seeds))

    async def finalize_experiment(sb, manifest, output, outputs, job_id=None, ctx=None):
        return result(await asyncio.to_thread(experiments.finalize, sb, manifest, output,
                                             outputs, ctx['session'].chat_id, job_id))

    string = {'type': 'string'}
    paths = {'type': 'array', 'items': string, 'maxItems': 200}
    specs = [
        ToolSpec('scholarly_search', 'Search Crossref scholarly metadata. Sends ONLY the query to api.crossref.org. '
                 'Results are candidates, not evidence of scientific claims; verify exact DOI/title/authors.',
                 params({'query': string, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 20}}, ['query']),
                 scholarly_search, 'research-web'),
        ToolSpec('doi_lookup', 'Resolve an exact DOI via Crossref. Optionally return escaped BibTeX using citation_key. '
                 'Metadata may contain omissions/errors; does not download the paper or verify claims.',
                 params({'doi': string, 'citation_key': string}, ['doi']), doi_lookup, 'research-web'),
        ToolSpec('record_evidence', 'Verify a quotation is present in a local source after whitespace normalization, then '
                 'write an evidence JSON record with source SHA-256, claim, quotation, locator and context. PDF requires page. '
                 'Text presence does NOT establish that the source supports the claim; DOI-to-file identity is not verified.',
                 params({'source': string, 'quotation': string, 'claim': string, 'output_dir': string,
                         'page': {'type': 'integer', 'minimum': 1}, 'doi': string},
                        ['source','quotation','claim','output_dir']), record_evidence, 'research', dangerous=True),
        ToolSpec('environment_report', 'Report Python/platform and selected scientific package versions. '
                 'No environment variables or secrets are read into the report.',
                 params({}, []), environment_report, 'science'),
        ToolSpec('experiment_manifest', 'Create a planned reproducibility manifest WITHOUT running the command. '
                 'Hashes explicit code/input files, records parameters/seeds (does NOT set them), environment and Git state. '
                 'Refuses to overwrite an existing manifest. Use start_job for clean execution.',
                 params({'output': string, 'cwd': string, 'command': string, 'code_files': paths,
                         'inputs': paths, 'parameters': {'type': 'object'}, 'seeds': {'type': 'object'}},
                        ['output','cwd','command','code_files','inputs','parameters','seeds']),
                 experiment_manifest, 'science', dangerous=True),
        ToolSpec('finalize_experiment', 'Write a separate result manifest after execution. Hash explicit outputs, detect '
                 'code/input drift, compare declared command/cwd with owned managed job, report missing outputs. '
                 'An existing output is not proof it was produced by this job; inspect results and clean-run reproducibility.',
                 params({'manifest': string, 'output': string, 'outputs': paths, 'job_id': string},
                        ['manifest','output','outputs']), finalize_experiment, 'science', dangerous=True, needs_ctx=True),
    ]
    for spec in specs:
        registry[spec.name] = spec
