"""Opt-in Crossref + clean-process experiment smoke. Synthetic local inputs only."""
import asyncio
import json
from pathlib import Path
import sys
import time
import uuid

from ctxagent import experiments, jobs, research, tools
from ctxagent.latex_workflow import shell_command


async def main():
    root = Path(__file__).resolve().parent.parent
    sb = tools.Sandbox([str(root)])
    dest = root / '.ctxagent' / 'research-smoke' / uuid.uuid4().hex
    dest.mkdir(parents=True)
    paper = await research.lookup('10.1038/nature14539')
    (dest/'metadata.json').write_text(json.dumps(paper,indent=2),encoding='utf-8')
    (dest/'reference.bib').write_text(research.bibtex(paper,'LeCun2015DeepLearning'),encoding='utf-8')
    source = dest/'source.txt'
    source.write_text('Synthetic evidence: the calibration value is 3.2 volts.\n',encoding='utf-8')
    evidence = research.record_evidence(sb,str(source),'the calibration value is 3.2 volts.',
                                        'The synthetic fixture reports 3.2 volts.',str(dest/'evidence'))
    script = dest/'experiment.py'
    script.write_text("import json, random\nfrom pathlib import Path\nrandom.seed(42)\n"
                      "Path('result.json').write_text(json.dumps({'values':[random.random() for _ in range(3)]}))\n",encoding='utf-8')
    command = shell_command([sys.executable,str(script)])
    plan = experiments.create(sb,str(dest/'planned.json'),str(dest),command,[str(script)],[],{}, {'random':42})
    job = jobs.start(command,dest,'research-smoke',30)
    while jobs.status(job['id'],'research-smoke')['state'] not in jobs.TERMINAL:
        await asyncio.sleep(.1)
    result = experiments.finalize(sb,plan['manifest_path'],str(dest/'recorded.json'),
                                  [str(dest/'result.json')],'research-smoke',job['id'])
    assert result['job']['state']=='succeeded'
    assert not result['input_or_code_drift'] and not result['missing_outputs']
    print(json.dumps({'doi':paper['doi'],'title':paper['title'],'year':paper['year'],
                     'evidence_status':evidence['status'],'job_state':result['job']['state'],
                     'result_sha256':result['outputs'][0]['sha256'],'directory':str(dest)},indent=2))


if __name__ == '__main__':
    asyncio.run(main())
