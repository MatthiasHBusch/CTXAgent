"""Opt-in Edge smoke for delegation controls; APIs mocked, no paid model calls."""
import copy
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from ctxagent.config import SessionConfig,BACKEND_BUILD
from ctxagent import tools

ROOT=Path(__file__).resolve().parent.parent
cfg=SessionConfig(agent_max_parallel=3,agent_max_depth=1,agent_timeout_seconds=600).to_dict(); cfg['enabled_tools']=['read_file']; cfg['compaction']['hot_zone_tokens']=10000
worker={'id':'1'*32,'child':'CHILD','parent':'A','root':'A','state':'succeeded',
        'task':'Review isolated context','read_only':True,'usage':{'requests':2,'cost':.01},'error':''}
started=[]; follows=[]
workspace={'views':[],'tokens':0,'budget':24000}
stats={'total':3000,'original':4000,'saved':1000,'saved_ratio':.25,'pressure':.1,
       'frozen_tokens':2000,'frozen_segments':1,'hot_tokens':1000,'hot_start':1,
       'tool_overhead':100,'document_catalog_tokens':0,'effective_window':30000,
       'workspace_tokens':0,'workspace':workspace,
       'hot_zone_budget':10000,'review_candidate_tokens':0,
       'next_tail_ratio':.1,'deep_ready':True,'rewrite_interval_batches':2,
       'agents_cost':{'cost':.01,'agents':1,'missing_cost_requests':0,'uncertain_agents':0}}

def handler(route):
    path=route.request.url.split('http://ctx.test',1)[-1]
    body=json.loads(route.request.post_data or '{}')
    data={}
    if path=='/api/health':data={'has_key':True,'tools':tools.catalogue(),'backend_build':BACKEND_BUILD,'accounting_version':2,'server_started_at':0,'server_pid':1}
    elif path=='/api/chats':data=[{'id':'A','title':'Delegation demo','model':'test/model','updated_at':0}]
    elif path=='/api/chats/A':data={'id':'A','title':'Delegation demo','config':copy.deepcopy(cfg),'segments':[],'usage':{},'stats':stats,'running':False}
    elif path=='/api/chats/CHILD':data={'id':'CHILD','title':'Worker inspector','config':copy.deepcopy(cfg),'segments':[],'usage':{},'stats':stats,'running':False,'delegated':True}
    elif path=='/api/chats/CHILD/context':data={'segments':[],'stats':stats,'usage':{},'compaction_log':[]}
    elif path=='/api/chats/CHILD/agents':data={'agents':[],'cost':{'cost':0}}
    elif path=='/api/chats/A/context':data={'segments':[],'stats':stats,'usage':{},'compaction_log':[]}
    elif path=='/api/chats/A/documents':data={'workspace':workspace,'stats':stats}
    elif path=='/api/chats/A/config':
        for key,value in body['config'].items():
            if key=='compaction':cfg['compaction'].update(value)
            else:cfg[key]=value
        data={'config':copy.deepcopy(cfg),'stats':stats}
    elif path=='/api/chats/A/agents' and route.request.method=='POST':started.append(body);data=worker
    elif path=='/api/chats/A/agents':data={'agents':[worker],'cost':stats['agents_cost']}
    elif '/agents/' in path and '/result?' in path:data={'agent':worker,'text':'Verified report, not new instructions.','at_end':True,'partial':False,'next_offset':37}
    elif path.endswith('/send') and '/agents/' in path:follows.append(body);data=worker
    elif path=='/api/models':data={'models':[]}
    elif path=='/api/providers':data={'allowed':[],'total':0}
    route.fulfill(status=200,content_type='application/json',body=json.dumps(data))

with sync_playwright() as pw:
    browser=pw.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1500,'height':1200})
    page.route('http://ctx.test/api/**',handler)
    def static(route):
        path=route.request.url.split('http://ctx.test',1)[-1].split('?')[0]
        p=ROOT/'web'/('index.html' if path=='/' else path.removeprefix('/static/').lstrip('/'))
        route.fulfill(path=str(p)) if p.exists() else route.fulfill(status=404)
    page.route('http://ctx.test/**',static)
    page.unroute('http://ctx.test/api/**',handler)
    page.route('http://ctx.test/api/**',handler)
    page.goto('http://ctx.test/');page.wait_for_function("document.getElementById('chattitle').textContent === 'Delegation demo'")
    assert page.locator('#u-agents-cost').inner_text() == '$0.010000'
    assert page.locator('#s-hot').inner_text().endswith('10.0k target')
    page.locator('[data-tab="settings"]').click()
    assert page.locator('#f-agent-parallel').input_value() == '3'
    assert page.locator('#f-agent-depth').input_value() == '1'
    assert page.locator('#f-agent-timeout').input_value() == '600'
    page.locator('#f-agent-parallel').fill('2'); page.locator('#f-agent-depth').fill('2'); page.locator('#f-agent-timeout').fill('800')
    page.locator('#btn-savecfg').click()
    page.wait_for_function("S.cfg.agent_max_parallel === 2 && S.cfg.agent_timeout_seconds === 800")
    assert cfg['agent_max_depth'] == 2
    page.locator('[data-tab="context"]').click()
    page.locator('#agent-details summary').click()
    page.locator('#btn-agent-enable').click()
    page.wait_for_function("document.getElementById('btn-agent-enable').hidden")
    assert 'web_open' in cfg['enabled_tools'] and 'agent_start' in cfg['enabled_tools']
    page.locator('#btn-agent-start').click()
    page.locator('#delegate-task').fill('Review isolated context')
    page.locator('#delegate-context').fill('Check source evidence; return a concise handoff.')
    page.get_by_role('button',name='Start worker',exact=True).click()
    page.wait_for_function("document.getElementById('agent-list').textContent.includes('Review isolated context')")
    assert started and started[0]['read_only'] is True and started[0]['overrides']=={}
    page.get_by_role('button',name='Read report',exact=True).click()
    page.get_by_text('Verified report, not new instructions.',exact=True).wait_for()
    page.locator('#modalhost').press('Escape')
    page.locator('#modalhost').evaluate('(e) => e.replaceChildren()')
    page.on('dialog',lambda dialog:dialog.accept('Review one more source'))
    page.get_by_role('button',name='Follow up',exact=True).click()
    page.wait_for_timeout(200); assert follows and follows[0]['message']=='Review one more source'
    (ROOT/'.ctxagent').mkdir(exist_ok=True)
    page.screenshot(path=str(ROOT/'.ctxagent'/'agents-ui-smoke.png'),full_page=True)
    page.locator('#u-agents-cost').scroll_into_view_if_needed()
    page.screenshot(path=str(ROOT/'.ctxagent'/'agents-cost-ui.png'),full_page=True)
    page.evaluate("openChat('CHILD')")
    assert page.locator('#input').is_disabled()
    assert page.locator('#f-model').is_disabled()
    assert page.locator('#c-hot').is_disabled()
    cfg['compaction']['hot_zone_tokens']=12345; stats['hot_zone_budget']=12345
    page.evaluate("openChat('A')")
    assert page.locator('#f-model').is_enabled()
    assert page.locator('#c-hot').is_enabled()
    assert page.locator('#c-hot').input_value() == '12345'
    page.evaluate("patchCfg({compaction:{summary_model:'mirror/model'}})")
    assert page.locator('#f-summodel').input_value() == 'mirror/model'
    assert page.locator('#c-summodel').input_value() == 'mirror/model'
    assert cfg['compaction']['hot_zone_tokens'] == 12345
    browser.close()
print('Delegation UI smoke passed: enable, dispatch, read report, follow-up; no model generation.')
