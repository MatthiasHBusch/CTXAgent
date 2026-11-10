"""Small stable delegation and browsing tool interfaces; services own the state."""
import asyncio
import json
from . import delegation, browsing

AGENT_TOOLS = ['agent_start','agent_status','agent_list','agent_result','agent_wait','agent_cancel','agent_send']
WEB_TOOLS = ['web_search','web_open','web_find']


def register(registry, Spec, Result, params):
    def output(data, **kw):
        return Result(json.dumps(data,ensure_ascii=False,indent=2),**kw)
    def session(ctx):
        if not ctx or not ctx.get('session'):
            raise ValueError('Requires a chat session')
        return ctx['session']
    async def start(sb, task, context='', overrides_json='{}', inherit_documents=True,
                    read_only=False, timeout=None, segment_ids=None, ctx=None):
        from .document_edits import _json
        overrides = _json(overrides_json)
        return output(await asyncio.to_thread(delegation.start,session(ctx),task,context,
                         overrides,inherit_documents,read_only,timeout,segment_ids))
    async def status(sb, agent_id, ctx=None):
        return output(await asyncio.to_thread(delegation.status,agent_id,session(ctx).chat_id))
    async def listing(sb, limit=30, ctx=None):
        return output(await asyncio.to_thread(delegation.list_agents,session(ctx).chat_id,limit))
    async def result(sb, agent_id, offset=0, max_chars=12000, ctx=None):
        return output(await asyncio.to_thread(delegation.result,agent_id,session(ctx).chat_id,offset,max_chars))
    async def wait(sb, agent_id, seconds=10, ctx=None):
        if not isinstance(seconds,int) or not 0<=seconds<=60:
            raise ValueError('seconds must be 0..60')
        deadline=asyncio.get_running_loop().time()+seconds
        while True:
            data=await asyncio.to_thread(delegation.status,agent_id,session(ctx).chat_id)
            if data['state'] in delegation.TERMINAL or asyncio.get_running_loop().time()>=deadline:
                return output(data)
            await asyncio.sleep(0.3)
    async def cancel(sb, agent_id, descendants=True, ctx=None):
        return output(await asyncio.to_thread(delegation.cancel,agent_id,session(ctx).chat_id,descendants))
    async def send(sb, agent_id, message, overrides_json='{}', timeout=None, ctx=None):
        from .document_edits import _json
        return output(await asyncio.to_thread(delegation.send,agent_id,session(ctx).chat_id,
                                              message,_json(overrides_json),timeout))
    async def search(sb, query, max_results=5, engine='auto', ctx=None):
        return output(await browsing.search(query,max_results,engine))
    async def open_page(sb, url=None, page_id=None, offset=0, max_chars=12000, max_links=20,
                        render_js=False, screenshot=False, ctx=None):
        data,images=await browsing.open_page(session(ctx).chat_id,url,page_id,offset,max_chars,
                                            max_links,render_js,screenshot)
        return output(data,images=images,resource_key='url:'+data['url'])
    async def find(sb, page_id, pattern, max_matches=20, ctx=None):
        return output(await asyncio.to_thread(browsing.find,session(ctx).chat_id,page_id,pattern,max_matches))
    string={'type':'string'}
    integer=lambda lo,hi:{'type':'integer','minimum':lo,'maximum':hi}
    definitions=[
      ('agent_start','Start a child agent on a task and return its agent_id at once. It inherits a copy of the settings, not the conversation: it sees task, context and the items named in segment_ids (max 20), plus the live view specs unless inherit_documents=false. Children run in parallel and survive Stop and restart. read_only=true blocks writing tools. A child offloads work but adds cost: it is a second model loop that re-reads its own context on every request, and the handoff and its report cost tokens on both sides. It only helps for a large, independent task; a small one is cheaper to do yourself. Costs are billed per child.',
       {'task':string,'context':string,'overrides_json':{'type':'string','description':'JSON object of SessionConfig overrides; compaction merges partially; default {}.'},
        'inherit_documents':{'type':'boolean'},'read_only':{'type':'boolean'},'timeout':integer(1,604800),
        'segment_ids':{'type':'array','items':string,'maxItems':20}},['task'],start),
      ('agent_status','State, child chat id, usage, limits and errors of a child. A lost or failed child may already have changed files.',{'agent_id':string},['agent_id'],status),
      ('agent_list',"Agents started by this chat and their descendants. Their usage is not part of this chat's usage.",{'limit':integer(1,100)},[],listing),
      ('agent_result','Final report of a child, by character offset. The full trajectory is in the child chat.',{'agent_id':string,'offset':integer(0,100000000),'max_chars':integer(1,40000)},['agent_id'],result),
      ('agent_wait','Wait up to 60 seconds for a child to change state or finish. Returns the status, not the report.',{'agent_id':string,'seconds':integer(0,60)},['agent_id'],wait),
      ('agent_cancel','Stop a child and, by default, its descendants. Edits already made stay, as do jobs it detached.',{'agent_id':string,'descendants':{'type':'boolean'}},['agent_id'],cancel),
      ('agent_send','Follow-up message to a finished child; it keeps its conversation, live views and settings.',{'agent_id':string,'message':string,'overrides_json':string,'timeout':integer(1,604800)},['agent_id','message'],send),
      ('web_search','Search the public web (Bing RSS, DuckDuckGo, Brave with a key). Returns titles, URLs and unverified snippets.',{'query':string,'max_results':integer(1,10),'engine':{'type':'string','enum':['auto','bing','duckduckgo','brave']}},['query'],search),
      ('web_open',"Fetch a public HTTP(S) page, PDF or text: page_id, text from offset, links. A page_id reads this chat's cached snapshot, a url fetches fresh. render_js loads the page in an isolated browser without login cookies; screenshot adds a viewport image. Private addresses are blocked.",
       {'url':string,'page_id':string,'offset':integer(0,100000000),'max_chars':integer(1,40000),
        'max_links':integer(0,200),'render_js':{'type':'boolean'},'screenshot':{'type':'boolean'}},[],open_page),
      ('web_find','Find a literal phrase (case-insensitive) in a cached page; returns offsets with context.',{'page_id':string,'pattern':string,'max_matches':integer(1,100)},['page_id','pattern'],find),
    ]
    for name,desc,props,required,fn in definitions:
        registry[name]=Spec(name,desc,params(props,required),fn,'agents' if name.startswith('agent_') else 'web',
                            dangerous=name in {'agent_start','agent_send','agent_cancel'},needs_ctx=True)
