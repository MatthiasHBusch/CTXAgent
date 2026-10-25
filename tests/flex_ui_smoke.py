"""Opt-in headless Flex UI checks with mocked APIs; no prompts/network generation."""
import copy
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from ctxagent import config, tools, flex
from tests.test_flex import EPS, DIR

root=Path(__file__).resolve().parent.parent
cfg=config.SessionConfig(model='openai/gpt-6-astra').to_dict()
health={'ok':True,'backend_build':config.BACKEND_BUILD,'accounting_version':2,'restart_required':False,
        'has_key':True,'tools':tools.catalogue(),'default_tools':tools.DEFAULT_TOOLS,'defaults':cfg}
chat={'id':'ui-test','title':'Flex pricing test','config':cfg,'usage':{},'stats':{},'segments':[],'compaction_log':[]}
patches=[]; model_requests=[]

with sync_playwright() as playwright:
    browser=playwright.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1500,'height':1200})
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    def serve(route):
        path=route.request.url.split('http://ctx.test',1)[-1]
        if path=='/':
            return route.fulfill(body=(root/'web/index.html').read_text(encoding='utf-8'),content_type='text/html')
        if path.startswith('/static/'):
            file=root/'web'/path.removeprefix('/static/')
            if not file.resolve().is_relative_to((root/'web').resolve()) or not file.is_file():
                return route.fulfill(status=404)
            return route.fulfill(body=file.read_bytes(),content_type='text/javascript' if file.suffix=='.js' else 'text/css')
        if path=='/api/providers/flex':
            body=route.request.post_data_json
            eps=EPS if body['model']=='openai/gpt-6-astra' else []
            return route.fulfill(json=flex.catalogue(body['model'],eps,DIR,body['policy'],body.get('prompt_tokens',0)))
        if path=='/api/chats/ui-test/config':
            body=route.request.post_data_json['config']; patches.append(copy.deepcopy(body))
            for key,value in body.items():
                if key=='compaction': cfg['compaction'].update(value)
                else: cfg[key]=value
            return route.fulfill(json={'config':cfg,'stats':{}})
        if path=='/api/providers/check':
            body=route.request.post_data_json
            return route.fulfill(json={'ok':True,'message':'Region check '+body['model']})
        if path.endswith('/send'):
            model_requests.append(path)
        data={'/api/health':health,'/api/chats':[{'id':'ui-test','title':chat['title']}],
              '/api/chats/ui-test':chat,'/api/chats/ui-test/context':{**chat,'config':cfg['compaction']},
              '/api/models':{'models':[]},'/api/providers':{'allowed':[],'total':0}}.get(path,{})
        route.fulfill(json=data)
    page.route('**/*',serve)
    page.goto('http://ctx.test/')
    page.locator('[data-tab="settings"]').click()
    page.wait_for_timeout(500)
    assert page.locator('#f-model-pricing').input_value()=='standard'
    page.select_option('#f-model-pricing','prefer_flex')
    page.wait_for_timeout(300)
    assert '50% lower' in page.locator('#model-flex').inner_text()
    assert '$5.000' in page.locator('#model-flex').inner_text()
    assert '$25.000' in page.locator('#model-flex').inner_text()
    assert 'standard pricing' in page.locator('#model-flex').inner_text()
    assert 'Region check openai/gpt-6-astra' in page.locator('#model-check').inner_text()
    assert 'Region check' in page.locator('#sum-check').inner_text()
    page.select_option('#f-sum-pricing','flex_only')
    page.wait_for_timeout(300)
    assert 'discovery unavailable' in page.locator('#sum-flex').inner_text().lower()
    page.locator('#btn-savecfg').click()
    page.wait_for_timeout(300)
    assert patches[-1]['pricing_mode']=='prefer_flex'
    assert patches[-1]['compaction']['summary_pricing_mode']=='flex_only'
    page.locator('#pane-settings').evaluate('(element) => { element.scrollTop = 0; }')
    page.screenshot(path=str(root/'.ctxagent/flex-selector.png'),full_page=True)
    page.locator('#sum-flex').scroll_into_view_if_needed()
    page.screenshot(path=str(root/'.ctxagent/flex-summary-selector.png'),full_page=True)
    page.reload()
    page.wait_for_timeout(400)
    page.locator('[data-tab="settings"]').click()
    assert page.locator('#f-model-pricing').input_value()=='prefer_flex'
    assert page.locator('#f-sum-pricing').input_value()=='flex_only'
    page.locator('#f-model').fill('unknown/model')
    page.select_option('#f-model-pricing','flex_only')
    page.wait_for_timeout(800)
    assert 'no standard-price fallback' in page.locator('#model-flex').inner_text()
    assert not errors, errors
    assert not model_requests, model_requests
    print('Flex UI checks passed: discovered tariff, independent modes, persistence, no generation, unavailable state, both region checks.')
    browser.close()
