"""Offline process fixture: real worker/session/tools with a deterministic model.

Only tests invoke this script; production has no mock-model switch.
"""
import asyncio
import json
from pathlib import Path
import sys
import shlex
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ctxagent import llm, agent_worker, delegation, jobs, store

async def model(**kwargs):
    messages=kwargs['messages']
    task=next((m.get('content','') for m in reversed(messages) if m['role']=='user'), '')
    if isinstance(task,str) and 'BLOCK' in task:
        await asyncio.sleep(100)
    if not any(m['role']=='tool' for m in messages):
        name,args=('agent_start',{'task':'BLOCK descendant','timeout':20}) if isinstance(task,str) and 'NEST' in task else ('read_file',{'path':'input.txt'})
        yield {'type':'tool_calls','tool_calls':[{'id':'fixture-read','type':'function','function':{
            'name':name,'arguments':json.dumps(args)}}]}
    else:
        await asyncio.sleep(0.15)
        yield {'type':'text','text':'Offline delegated result: '+messages[-1].get('content','')[-100:]}
    yield {'type':'done','finish_reason':'stop'}

def fixture_launch(identifier,timeout,root):
    script=str(Path(__file__).resolve())
    quote=lambda s:"'"+s.replace("'","''")+"'"
    command=('& '+quote(sys.executable)+' -u '+quote(script)+' '+quote(identifier)) if sys.platform=='win32' else ' '.join(map(shlex.quote,[sys.executable,'-u',script,identifier]))
    def publish(jid):
        with store._conn() as c: c.execute('UPDATE agent_tasks SET job=? WHERE id=?',(jid,identifier))
    jobs.start(command,Path(script).parent.parent,root,timeout,publish=publish,preserve_managed_children=True)

llm.stream_chat=model
delegation._launch=fixture_launch
if __name__=='__main__':
    raise SystemExit(agent_worker.main(sys.argv[1]))
