"""Opt-in Edge/Playwright regression: interleaved streams, drafts and fetch races.
All HTTP endpoints are mocked. No model generation or real chat modifications.
"""
import copy
from pathlib import Path
from playwright.sync_api import sync_playwright
from ctxagent import config, tools

root = Path(__file__).resolve().parent.parent
cfg = config.SessionConfig(model='test/model', enabled_tools=['read_file'],
                           workdirs=['C:\\work']).to_dict()
cfg['compaction']['document_compaction_enabled'] = False
cfg['compaction']['hot_zone_tokens'] = 10000
stats = dict(total=264546, original=333667, saved=69121, effective_window=379000, frozen_tokens=54044, hot_tokens=205678, hot_start=133, hot_zone_budget=10000, tool_overhead=4824, pressure=264546/379000, next_tail_ratio=0.1, deep_ready=True, rewrite_interval_batches=2, review_candidate_tokens=190000)
chats = {key: dict(id=key, title='Chat '+key, config=copy.deepcopy(cfg), usage={}, stats=copy.deepcopy(stats),
                  segments=[], compaction_log=[]) for key in ('A', 'B')}
chats['B']['config']['model'] = 'other/model'
health = dict(ok=True, backend_build=config.BACKEND_BUILD, accounting_version=2,
              restart_required=False, has_key=True, defaults=cfg,
              default_tools=tools.DEFAULT_TOOLS, tools=tools.catalogue())

with sync_playwright() as pw:
    browser = pw.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width':1500,'height':1000})
    errors=[]; requests=[]
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.add_init_script('''
      const originalFetch = window.fetch.bind(window);
      window.streams = {}; window.held = {}; window.holdPaths = new Set();
      window.fetch = async (url, opts) => {
        if (String(url).endsWith('/send') || String(url).endsWith('/compact')) {
          const id = String(url).split('/')[3];
          return new Response(new ReadableStream({start(c) { window.streams[id] = c; }}),
            {headers:{'content-type':'text/event-stream'}});
        }
        if (window.holdPaths.has(String(url))) {
          window.holdPaths.delete(String(url));
          const response = await originalFetch(url, opts);
          return new Promise(resolve => {window.held[String(url)] = () => resolve(response);});
        }
        return originalFetch(url, opts);
      };
      window.emit = (id, event) => window.streams[id].enqueue(
        new TextEncoder().encode('data: ' + JSON.stringify(event) + '\\n\\n'));
      window.endStream = id => {window.streams[id].close(); delete window.streams[id];};
    ''')
    def serve(route):
        path=route.request.url.split('http://ctx.test',1)[-1]
        requests.append((path, route.request.post_data_json if route.request.post_data else None))
        if path=='/':
            return route.fulfill(body=(root/'web/index.html').read_text(encoding='utf-8'),content_type='text/html')
        if path.startswith('/static/'):
            f=root/'web'/path.removeprefix('/static/')
            if not f.resolve().is_relative_to((root/'web').resolve()) or not f.is_file():
                return route.fulfill(status=404)
            return route.fulfill(body=f.read_bytes(),content_type='text/javascript' if f.suffix=='.js' else 'text/css')
        if path=='/api/chats' and route.request.method=='POST':
            body=route.request.post_data_json
            inherited=chats[body['source_chat']]['config'] if 'source_chat' in body else cfg
            chats['C']=dict(id='C',title='Chat C',config=copy.deepcopy(inherited),usage={},stats=copy.deepcopy(stats),segments=[])
            return route.fulfill(json={'id':'C'})
        if path=='/api/health': data=health
        elif path=='/api/chats': data=[dict(id=c['id'],title=c['title']) for c in chats.values()]
        elif path=='/api/providers/check': data=dict(ok=True,message='mocked provider')
        elif path=='/api/providers/flex': data=dict(known=True,available=False,candidates=[])
        elif path=='/api/models': data={'models':[]}
        elif path.startswith('/api/chats/'):
            parts=path.split('/'); c=chats[parts[3]]
            if path.endswith('/context'): data={**c,'config':c['config']['compaction']}
            elif path.endswith('/config'):
                patch=route.request.post_data_json['config']
                for key,val in patch.items():
                    if key=='compaction': c['config'][key].update(val)
                    else: c['config'][key]=val
                data={'config':c['config'],'stats':{}}
            elif path.endswith('/defaults') or path.endswith('/stop'): data={'ok':True}
            else: data=c
        else: data={}
        route.fulfill(json=data)
    page.route('**/*',serve)
    page.goto('http://ctx.test/')
    page.wait_for_function("S.chatId === 'A' && S.cfg !== null")
    # Independent drafts, restored only on return.
    page.locator('#input').fill('A draft')
    page.evaluate("openChat('B')")
    assert page.locator('#input').input_value()==''
    page.locator('#input').fill('B draft')
    page.evaluate("openChat('A')")
    assert page.locator('#input').input_value()=='A draft'
    page.locator('#btn-send').click()
    page.wait_for_function("!!window.streams.A")
    page.evaluate("emit('A', {type:'delta', text:'A first response'})")
    page.locator('#input').fill('A next draft')
    page.evaluate("openChat('B')")
    assert page.locator('#input').input_value()=='B draft'
    assert 'A first response' not in page.locator('#msgwrap').inner_text()
    page.locator('#btn-send').click()
    page.wait_for_function("!!window.streams.B")
    page.locator('#input').fill('B next draft')
    # Hidden chat starts a new bubble, a tool, visual evidence, stats and notices.
    page.evaluate("""() => {
      emit('A', {type:'assistant_start'});
      emit('A', {type:'routing',routing_notice:'Trying Flex endpoints: openai/flex. Lower availability/higher latency are possible.'});
      emit('A', {type:'delta', text:'A hidden response'});
      emit('A', {type:'tool_start', id:'a-tool', name:'read_file', args:{path:'A-only'}});
      emit('A', {type:'tool_result', id:'a-tool', tokens:42, ms:1, ok:true, segment:{body:'A result'}});
      emit('A', {type:'tool_start', id:'a-fail', name:'run_command', args:{command:'A command'}});
      emit('A', {type:'tool_result', id:'a-fail', tokens:12, ms:2, ok:false, segment:{body:'A command failed'}});
      emit('A', {type:'media', segment:{kind:'media',body:'A visual',images:[{id:'image-a',label:'A figure'}]}});
      emit('A', {type:'stats', stats:{total:12345}});
      emit('A', {type:'error', error:'A notice only'});
      emit('B', {type:'delta',text:'B visible response'});
    }""")
    page.wait_for_function("document.querySelector('#msgwrap').textContent.includes('B visible response')")
    assert 'A hidden' not in page.locator('#msgwrap').inner_text()
    assert 'A-only' not in page.locator('#msgwrap').inner_text()
    assert 'A notice' not in page.locator('#msgwrap').inner_text()
    assert page.evaluate('S.stats.total') != 12345
    assert page.locator('#input').input_value()=='B next draft'
    page.evaluate("openChat('A')")
    assert page.locator('#msgwrap .turn-activity').count() == 2
    assert all(page.locator('#msgwrap .turn-activity').nth(i).get_attribute('open') is None for i in range(2))
    assert 'A hidden response' in page.locator('#msgwrap').inner_text()
    assert 'Trying Flex endpoints' not in page.locator('#msgwrap').inner_text()
    assert 'A-only' not in page.locator('#msgwrap').inner_text()
    assert 'A result' not in page.locator('#msgwrap').inner_text()
    assert 'A notice' not in page.locator('#msgwrap').inner_text()
    assert '2 errors' in page.locator('#msgwrap .turn-activity').nth(1).inner_text()
    assert 'has-errors' in page.locator('#msgwrap .turn-activity').nth(1).get_attribute('class')
    page.locator('#msgwrap .turn-activity > summary').first.click()
    assert 'Trying Flex endpoints' in page.locator('#msgwrap').inner_text()
    assert 'A-only' not in page.locator('#msgwrap').inner_text()
    page.locator('#msgwrap .turn-activity > summary').nth(1).click()
    assert 'A-only' in page.locator('#msgwrap').inner_text()
    assert 'A result' in page.locator('#msgwrap').inner_text()
    assert 'A command failed' in page.locator('#msgwrap').inner_text()
    assert 'A notice' in page.locator('#msgwrap').inner_text()
    assert page.evaluate("[...document.querySelector('#msgwrap').children].every(e => e.matches('.msg.user, .msg.assistant, .turn-activity'))")
    assert page.locator('#msgwrap .media-evidence a').get_attribute('href').startswith('/api/chats/A/')
    assert 'B visible' not in page.locator('#msgwrap').inner_text()
    assert page.locator('#input').input_value()=='A next draft'
    page.evaluate("openChat('B')")
    page.evaluate("endStream('A')")
    page.wait_for_function("!S.runs.has('A')")
    assert page.locator('#btn-send').get_attribute('title')=='Stop'
    page.locator('#btn-send').click()
    page.wait_for_timeout(100)
    assert ('/api/chats/B/stop', None) in requests
    assert not any(path=='/api/chats/A/stop' for path,_ in requests)
    page.evaluate("endStream('B')")
    page.wait_for_function("!S.runs.has('B')")
    # A late initial-load response must not overwrite the selected B view/settings.
    page.evaluate("holdPaths.add('/api/chats/A'); void openChat('A')")
    page.wait_for_function("!!held['/api/chats/A']")
    page.evaluate("openChat('B')")
    page.evaluate("held['/api/chats/A']()")
    page.wait_for_timeout(100)
    assert page.locator('#chattitle').inner_text()=='Chat B'
    assert page.locator('#f-model').input_value()=='other/model'
    # Late context responses cannot replace the new chat's context table/stats.
    page.evaluate("holdPaths.add('/api/chats/B/context'); void refreshContext('B')")
    page.wait_for_function("!!held['/api/chats/B/context']")
    page.evaluate("openChat('A')")
    page.evaluate("held['/api/chats/B/context']()")
    page.wait_for_timeout(100)
    assert page.locator('#f-model').input_value()=='test/model'
    # New chat uses the last selected chat's complete saved snapshot, not the
    # global template, stale UI config or unsaved form controls.
    chats['A']['config']['model']='source/model'
    chats['A']['config']['pricing_mode']='prefer_flex'
    chats['A']['config']['context_window']=543210
    chats['A']['config']['agent_timeout_seconds']=321
    page.locator('[data-tab="settings"]').click()
    page.locator('#f-model').fill('unsaved/model')
    page.locator('#btn-new').click()
    page.wait_for_function("S.chatId === 'C' && S.cfg !== null")
    assert [body for path,body in requests if path=='/api/chats' and body][-1]=={'title':'New chat','source_chat':'A'}
    assert chats['C']['config']==chats['A']['config']
    assert page.locator('#f-model').input_value()=='source/model'
    assert page.locator('#f-ctx').input_value()=='543210'
    assert page.locator('#input').input_value()==''
    assert not chats['C']['segments']
    page.locator('[data-tab="compaction"]').click()
    assert not page.locator('#c-docs').is_checked()
    assert page.locator('#c-target').input_value()=='0.1'
    assert page.locator('#c-mask').is_disabled()
    page.locator('#c-docs').check()
    assert page.locator('#c-mask').is_enabled()
    page.locator('#c-docs').uncheck()
    assert page.locator('#c-mask').is_checked()  # choices preserved, master is off
    assert page.locator('#c-interval').is_enabled()  # linked freezing is independent of document switch
    assert not page.locator('#c-review').is_checked()
    assert page.locator('#c-delivery').input_value() == '0'
    page.locator('#c-review').check()
    page.locator('#c-delivery').fill('8000')
    page.locator('#c-interval').fill('10')
    page.locator('#btn-savecmp').click()
    page.wait_for_function("S.cfg.compaction.review_before_freeze === true")
    assert chats['C']['config']['compaction']['initial_result_max_tokens'] == 8000
    assert chats['C']['config']['compaction']['rewrite_interval_batches'] == 10
    page.locator('#c-review').scroll_into_view_if_needed()
    page.screenshot(path=str(root/'.ctxagent/freeze-review-settings.png'),full_page=True)
    page.screenshot(path=str(root/'.ctxagent/chat-engine-settings.png'),full_page=True)
    page.locator('#c-target').scroll_into_view_if_needed()
    page.screenshot(path=str(root/'.ctxagent/chat-deep-target.png'),full_page=True)
    # Explicit template capture names its source chat, and uses saved configuration.
    page.locator('[data-tab="settings"]').click()
    page.on('dialog',lambda d:d.accept())
    page.locator('#btn-defaults').click()
    page.wait_for_timeout(100)
    assert ('/api/chats/C/defaults', None) in requests
    # Persisted history groups every consecutive non-message block, keeping
    # all assistant outputs/commentary visible as boundaries within each turn.
    chats['C']['segments'] = [{'kind':'system','body':'System instructions fixture'}]
    for i in range(2):
        chats['C']['segments'].extend([
            {'kind':'user','body':f'User request {i}'},
            {'kind':'summary','body':f'Compacted history {i}'},
            {'kind':'diagnostic','body':f'Diagnostic event {i}'},
            {'kind':'tool_call','body':f'Work commentary {i}','tool_calls':[{'id':f'call-{i}','function':{'name':'read_file','arguments':'{"path":"source.py"}'}}]},
            {'kind':'tool_result','tool_call_id':f'call-{i}','body':f'Source result {i}','original_tokens':12},
            {'kind':'media','body':f'Visual evidence {i}','images':[]},
            {'kind':'assistant_text','body':f'Intermediate answer {i}'},
            {'kind':'tool_call','body':'','tool_calls':[{'id':f'late-{i}','function':{'name':'run_command','arguments':'{"command":"echo fixture"}'}}]},
            {'kind':'tool_result','tool_call_id':f'late-{i}','body':f'Command result {i}','original_tokens':10},
            {'kind':'tool_result','tool_call_id':f'orphan-{i}','body':f'Orphan result {i}','original_tokens':5},
            {'kind':'summary','body':f'Later compacted history {i}'},
            {'kind':'assistant_text','body':f'Final answer {i}'},
        ])
    from ctxagent.segments import Segment
    chats['C']['segments'] = [Segment(seq=i+1, role='user' if row['kind']=='user' else 'tool' if row['kind']=='tool_result' else 'assistant', **row).to_dict() for i,row in enumerate(chats['C']['segments'])]
    page.evaluate("openChat('C')")
    page.wait_for_function("document.querySelectorAll('#msgwrap .turn-activity').length === 7")
    assert all(page.locator('#msgwrap .turn-activity').nth(i).get_attribute('open') is None for i in range(7))
    text=page.locator('#msgwrap').inner_text()
    for i in range(2):
        for visible in ('User request','Work commentary','Intermediate answer','Final answer'):
            assert f'{visible} {i}' in text
        for folded in ('Source result','Compacted history','Diagnostic event','Visual evidence','Command result','Orphan result'):
            assert f'{folded} {i}' not in text
    assert 'System instructions fixture' not in text
    assert 'read_file' not in text
    order=page.evaluate("[...document.querySelector('#msgwrap').children].map(e => e.matches('.msg.user') ? 'user' : e.matches('.msg.assistant') ? 'assistant' : e.matches('.turn-activity') ? 'activity' : 'unexpected')")
    assert order==['activity','user','activity','assistant','activity','assistant','activity','assistant','user','activity','assistant','activity','assistant','activity','assistant']
    page.locator('[data-tab="context"]').click()
    page.evaluate("document.querySelector('#messages').scrollTop=0")
    page.screenshot(path=str(root/'.ctxagent/chat-collapsed-activity.png'),full_page=True)
    # One click reveals full results; native summary keyboard access also works.
    page.locator('#msgwrap .turn-activity > summary').nth(2).click()
    assert 'Source result 0' in page.locator('#msgwrap').inner_text()
    assert 'Visual evidence 0' in page.locator('#msgwrap').inner_text()
    page.locator('#msgwrap .turn-activity .tool-head').first.click()
    assert 'Source result 0' not in page.locator('#msgwrap').inner_text()
    page.locator('#msgwrap .turn-activity .tool-head').first.click()
    page.locator('#msgwrap .turn-activity > summary').nth(3).focus()
    page.keyboard.press('Enter')
    assert 'Command result 0' in page.locator('#msgwrap').inner_text()
    assert 'Orphan result 0' in page.locator('#msgwrap').inner_text()
    assert 'Later compacted history 0' in page.locator('#msgwrap').inner_text()
    page.screenshot(path=str(root/'.ctxagent/chat-expanded-activity.png'),full_page=True)
    page.keyboard.press('Enter')
    assert 'Orphan result 0' not in page.locator('#msgwrap').inner_text()
    # Clicking New during a different chat's pending load still captures that
    # selection, and its late response cannot overwrite the new chat.
    page.evaluate("holdPaths.add('/api/chats/B'); void openChat('B')")
    page.wait_for_function("!!held['/api/chats/B'] && S.cfg === null")
    page.evaluate("newChat()")
    assert [body for path,body in requests if path=='/api/chats' and body][-1]=={'title':'New chat','source_chat':'B'}
    assert chats['C']['config']==chats['B']['config']
    page.evaluate("held['/api/chats/B']()")
    page.wait_for_timeout(100)
    assert page.locator('#f-model').input_value()=='other/model'
    # No selected chat falls back to the saved template.
    page.evaluate("S.chatId = null; S.cfg = null; newChat()")
    assert [body for path,body in requests if path=='/api/chats' and body][-1]=={'title':'New chat'}
    assert chats['C']['config']==cfg
    assert not errors, errors

    assert not errors, errors
    print('PASS: independent drafts; interleaved streams/tools/media/errors; inactive stats; Stop ownership; hidden completion; load/context races; server defaults; engine controls; source-specific default save.')
    browser.close()
