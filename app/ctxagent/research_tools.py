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
        ToolSpec('scholarly_search', 'Search Crossref metadata; only the query is sent. Returns candidate records (DOI, title, authors).',
                 params({'query': string, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 20}}, ['query']),
                 scholarly_search, 'research-web'),
        ToolSpec('doi_lookup', 'Crossref metadata for a DOI; with citation_key also as BibTeX. The paper itself is not downloaded.',
                 params({'doi': string, 'citation_key': string}, ['doi']), doi_lookup, 'research-web'),
        ToolSpec('record_evidence', 'Check that a quotation occurs in a local source (whitespace-normalised) and write an evidence JSON record: source SHA-256, claim, quotation, locator, context. A PDF needs page. Presence of the text does not show that the source supports the claim.',
                 params({'source': string, 'quotation': string, 'claim': string, 'output_dir': string,
                         'page': {'type': 'integer', 'minimum': 1}, 'doi': string},
                        ['source','quotation','claim','output_dir']), record_evidence, 'research', dangerous=True),
        ToolSpec('environment_report', 'Python, platform and versions of scientific packages. Reads no environment variables.',
                 params({}, []), environment_report, 'science'),
        ToolSpec('experiment_manifest', 'Write a reproducibility manifest for a planned command without running it: hashes of code and input files, parameters and seeds as declared, environment, Git state. An existing manifest is not overwritten.',
                 params({'output': string, 'cwd': string, 'command': string, 'code_files': paths,
                         'inputs': paths, 'parameters': {'type': 'object'}, 'seeds': {'type': 'object'}},
                        ['output','cwd','command','code_files','inputs','parameters','seeds']),
                 experiment_manifest, 'science', dangerous=True),
        ToolSpec('finalize_experiment', "Write a result manifest after the run: hashes of the outputs, drift of code and inputs since the plan, comparison with the job's command and cwd, missing outputs.",
                 params({'manifest': string, 'output': string, 'outputs': paths, 'job_id': string},
                        ['manifest','output','outputs']), finalize_experiment, 'science', dangerous=True, needs_ctx=True),
    ]
    for spec in specs:
        registry[spec.name] = spec
