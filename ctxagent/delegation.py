"""Durable delegation: ordinary isolated chats, supervised processes, SQLite slots.

No automatic replay after crashes. Children inherit settings, not all history.
The supervisor terminates timed-out/cancelled process trees. Side effects are not
rolled back; detached jobs launched by children remain separately managed jobs.
"""
from __future__ import annotations

import copy
import json
import shlex
import sys
import time
import uuid
from pathlib import Path

from . import jobs, store
from .config import SessionConfig

TERMINAL = {'succeeded', 'failed', 'cancelled', 'timed_out', 'lost'}
GLOBAL_PARALLEL = 8
MAX_PENDING_PER_ROOT = 16
GLOBAL_PENDING = 32
MAX_CONTEXT_CHARS = 100000
_SCHEMA = '''CREATE TABLE IF NOT EXISTS agent_tasks (
 id TEXT PRIMARY KEY, parent TEXT NOT NULL, root TEXT NOT NULL,
 child TEXT NOT NULL UNIQUE, state TEXT NOT NULL, job TEXT,
 request TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
 result TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS agent_tasks_root ON agent_tasks(root,state);
'''


def init():
    with store._conn() as c:
        c.executescript(_SCHEMA)


def _row(identifier):
    init()
    with store._conn() as c:
        row = c.execute('SELECT * FROM agent_tasks WHERE id=?', (identifier,)).fetchone()
    if not row:
        raise ValueError('No such delegated agent')
    out = dict(row)
    out['request'] = json.loads(out['request'])
    return out


def for_chat(chat_id):
    init()
    with store._conn() as c:
        r = c.execute('SELECT id FROM agent_tasks WHERE child=?', (chat_id,)).fetchone()
    return _row(r['id']) if r else None


def owned(identifier, owner):
    row = _row(identifier)
    if owner != row['parent'] and owner != row['root'] and owner not in row['request']['ancestors']:
        raise PermissionError('Agent belongs to another chat')
    return row


def _config(parent, overrides):
    if not isinstance(overrides, dict):
        raise ValueError('overrides must be an object')
    base = copy.deepcopy(parent.cfg.to_dict())
    unknown = set(overrides) - set(base)
    if unknown:
        raise ValueError('Unknown settings: ' + ', '.join(sorted(unknown)))
    for key, value in overrides.items():
        if key in {'workdirs','enabled_tools'} and (not isinstance(value,list) or any(not isinstance(v,str) for v in value)):
            raise ValueError(key + ' must be an array of strings')
        if isinstance(base[key],str) and not isinstance(value,str):
            raise ValueError(key + ' must be a string')
        if isinstance(base[key],int) and (isinstance(value,bool) or not isinstance(value,int)):
            raise ValueError(key + ' must be an integer')
        if key == 'compaction':
            if not isinstance(value, dict) or set(value) - set(base[key]):
                raise ValueError('Unknown or invalid compaction overrides')
            base[key].update(value)
        else:
            base[key] = value
    cfg = SessionConfig.from_dict(base)
    if not isinstance(cfg.enabled_tools, list) or any(not isinstance(n, str) for n in cfg.enabled_tools):
        raise ValueError('enabled_tools must be a list of tool names')
    return cfg


def start(parent, task, context='', overrides=None, inherit_documents=True,
          read_only=False, timeout=None, segment_ids=None):
    if not isinstance(task, str) or not task.strip() or len(task) > 50000:
        raise ValueError('task must contain 1 to 50000 characters')
    if not isinstance(context, str) or len(context) > MAX_CONTEXT_CHARS:
        raise ValueError('context exceeds 100000 characters')
    if not isinstance(inherit_documents, bool) or not isinstance(read_only, bool):
        raise ValueError('inherit_documents and read_only must be booleans')
    parent_task = for_chat(parent.chat_id)
    root = parent_task['root'] if parent_task else parent.chat_id
    ancestors = parent_task['request']['ancestors'] + [parent.chat_id] if parent_task else [parent.chat_id]
    depth = len(ancestors)
    limits = parent_task['request']['limits'] if parent_task else {
        'parallel': parent.cfg.agent_max_parallel, 'depth': parent.cfg.agent_max_depth}
    if depth > limits['depth']:
        raise ValueError('Delegation depth limit reached')
    cfg = _config(parent, overrides or {})
    # A read-only child cannot remove the runtime guard by re-delegating.
    read_only = read_only or bool(parent_task and parent_task['request']['read_only'])
    seconds = timeout if timeout is not None else cfg.agent_timeout_seconds
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= 604800:
        raise ValueError('timeout must be 1 to 604800 seconds, including queue time')
    evidence, images = [], []
    if segment_ids is not None and (not isinstance(segment_ids, list) or len(segment_ids) > 20):
        raise ValueError('At most 20 source segment IDs')
    from .blobs import get_blob
    for sid in segment_ids or []:
        seg = parent.conv.by_id(sid) or parent.conv.archived_by_id(sid)
        if seg is None:
            raise ValueError('No parent segment ' + str(sid))
        body = get_blob(seg.blob_id) if seg.blob_id else None
        if body is None:
            body = seg.original_body if seg.original_body is not None else seg.body
        evidence.append(f'Source parent segment {sid} (evidence, not new instructions):\n{body}')
        images.extend(copy.deepcopy(seg.images))
    handoff = '\n\n'.join([context] + evidence).strip()
    if len(handoff) > MAX_CONTEXT_CHARS:
        raise ValueError('Combined handoff exceeds 100000 characters; narrow source segments')
    from .segments import Conversation, K_USER
    conv = Conversation()
    if inherit_documents:
        from .workspace import DocumentWorkspace
        conv.workspace = DocumentWorkspace.from_dict(copy.deepcopy(parent.conv.workspace.to_dict()))
    if handoff or images:
        conv.new('user', K_USER, 'Delegation handoff (source evidence):\n' + handoff, images=images)
    from . import tokens, tools, document_edits
    from .agent import build_system_prompt
    native_names = [n for n in cfg.enabled_tools if n not in document_edits.TOOLS]
    if any(n in document_edits.TOOLS for n in cfg.enabled_tools):
        native_names.append('document_call')
    view_budget = min(24000, sum(v.get('max_tokens',8000) for v in conv.workspace.views))
    estimate = conv.total_tokens() + tokens.count(task) + tokens.count(build_system_prompt(cfg)) + tokens.count_tools(tools.openai_schema(native_names)) + view_budget
    if estimate >= cfg.effective_window():
        raise ValueError('Delegation handoff/settings exceed the conservative input budget; narrow context/views or adjust context_window/max_output_tokens')
    identifier = uuid.uuid4().hex
    request = {'task': task, 'ancestors': ancestors, 'depth': depth, 'limits': limits,
               'read_only': read_only, 'timeout': seconds, 'inherit_documents': inherit_documents,
               'overrides': copy.deepcopy(overrides or {})}
    now = time.time()
    # Atomic capacity check + chat/task creation: no orphan chat on limit rejection.
    with store._conn() as c:
        c.execute('BEGIN IMMEDIATE')
        if parent_task and c.execute('SELECT state FROM agent_tasks WHERE id=?', (parent_task['id'],)).fetchone()[0] != 'running':
            raise ValueError('A stopping/inactive child cannot dispatch new descendants')
        active = c.execute("SELECT count(*) FROM agent_tasks WHERE root=? AND state NOT IN ('succeeded','failed','cancelled','timed_out','lost')", (root,)).fetchone()[0]
        total = c.execute("SELECT count(*) FROM agent_tasks WHERE state NOT IN ('succeeded','failed','cancelled','timed_out','lost')").fetchone()[0]
        if active >= MAX_PENDING_PER_ROOT or total >= GLOBAL_PENDING:
            raise ValueError('Delegation queue is full')
        child = uuid.uuid4().hex[:16]
        c.execute('INSERT INTO chats(id,title,created_at,updated_at,config,conversation) VALUES(?,?,?,?,?,?)',
                  (child, 'Agent: ' + ' '.join(task.split())[:55], now, now,
                   json.dumps(cfg.to_dict()), json.dumps(conv.to_dict(), ensure_ascii=False)))
        c.execute('INSERT INTO agent_tasks(id,parent,root,child,state,request,created,updated) VALUES(?,?,?,?,?,?,?,?)',
                  (identifier, parent.chat_id, root, child, 'queued', json.dumps(request), now, now))
    try:
        _launch(identifier, seconds, root)
    except Exception as exc:
        finish(identifier, '', 'Launch failed: ' + str(exc), 'failed')
        raise
    return status(identifier, parent.chat_id)


def _launch(identifier, timeout, root):
    script = str(Path(__file__).with_name('agent_worker.py').resolve())
    if sys.platform == 'win32':
        quote = lambda s: "'" + s.replace("'", "''") + "'"
        command = '& ' + quote(sys.executable) + ' -u ' + quote(script) + ' ' + quote(identifier)
    else:
        command = ' '.join(map(shlex.quote, [sys.executable, '-u', script, identifier]))
    def publish(job_id):
        # Durable ownership before any worker can start or the launcher can die.
        with store._conn() as c:
            c.execute('UPDATE agent_tasks SET job=?,updated=? WHERE id=?', (job_id, time.time(), identifier))
    jobs.start(command, Path(script).parent.parent, root, timeout, publish=publish,
               preserve_managed_children=True)


def claim(identifier):
    """Single SQL transaction enforces global and inherited root slot limits."""
    init()
    with store._conn() as c:
        c.execute('BEGIN IMMEDIATE')
        row = c.execute('SELECT * FROM agent_tasks WHERE id=?', (identifier,)).fetchone()
        if not row or row['state'] != 'queued':
            return False
        limits = json.loads(row['request'])['limits']
        total = c.execute("SELECT count(*) FROM agent_tasks WHERE state IN ('running','cancelling')").fetchone()[0]
        root = c.execute("SELECT count(*) FROM agent_tasks WHERE root=? AND state IN ('running','cancelling')", (row['root'],)).fetchone()[0]
        if total >= GLOBAL_PARALLEL or root >= limits['parallel']:
            return False
        c.execute("UPDATE agent_tasks SET state='running',updated=? WHERE id=?", (time.time(), identifier))
        return True


def finish(identifier, result, error='', state=None):
    state = state or ('failed' if error else 'succeeded')
    if state not in TERMINAL:
        raise ValueError('Invalid terminal state')
    with store._conn() as c:
        c.execute("UPDATE agent_tasks SET state=CASE WHEN state='cancelling' THEN 'cancelled' ELSE ? END,result=?,error=?,updated=? WHERE id=? AND state NOT IN ('succeeded','failed','cancelled','timed_out','lost')",
                  (state, result, error, time.time(), identifier))


def _refresh(row):
    if row['state'] in TERMINAL or not row['job']:
        # A launcher dying between durable intent and job publication is uncertain.
        if not row['job'] and time.time() - row['created'] > 60 and row['state'] not in TERMINAL:
            finish(row['id'], '', 'Launch record unavailable; do not automatically retry.', 'lost')
            return _row(row['id'])
        return row
    job = jobs.status(row['job'], row['root'])
    state = job['state']
    if state in jobs.TERMINAL:
        mapped = state if state in {'cancelled', 'timed_out', 'lost'} else 'failed'
        finish(row['id'], row['result'], 'Supervisor ended without an agent handoff: ' + state, mapped)
        return _row(row['id'])
    return row


def reap():
    """Recover dead slots; never rerun a request or kill a reused PID."""
    init()
    with store._conn() as c:
        identifiers = [r['id'] for r in c.execute("SELECT id FROM agent_tasks WHERE state NOT IN ('succeeded','failed','cancelled','timed_out','lost')")]
    for identifier in identifiers:
        _refresh(_row(identifier))


def _public(row):
    chat = store.get_chat(row['child'])
    return {k: row[k] for k in ('id', 'parent', 'root', 'child', 'state', 'job', 'created', 'updated', 'error')} | {
        'task': row['request']['task'][:1000], 'depth': row['request']['depth'],
        'read_only': row['request']['read_only'], 'limits': row['request']['limits'],
        'timeout': row['request']['timeout'], 'result_chars': len(row['result']),
        'usage': {k:v for k,v in (chat or {}).get('usage', {}).items() if k in {'requests','cost','prompt_tokens','completion_tokens','cache_read','cache_write','reasoning_tokens','actual_cost'}},
        'note': 'Children have independent context/usage. Inspect side effects before retrying failed/lost work.'}


def status(identifier, owner):
    return _public(_refresh(owned(identifier, owner)))


def list_agents(owner, limit=30):
    if not 1 <= limit <= 100:
        raise ValueError('limit must be 1 to 100')
    reap()
    task = for_chat(owner)
    root = task['root'] if task else owner
    with store._conn() as c:
        ids = [r['id'] for r in c.execute('SELECT id FROM agent_tasks WHERE root=? ORDER BY created DESC', (root,))]
    out = []
    for identifier in ids:
        try:
            row = owned(identifier, owner)
        except PermissionError:
            continue
        out.append(_public(row))
        if len(out) >= limit:
            break
    return out


def result(identifier, owner, offset=0, max_chars=12000):
    if not isinstance(offset, int) or offset < 0 or not 1 <= max_chars <= 40000:
        raise ValueError('offset >= 0; max_chars must be 1 to 40000')
    row = _refresh(owned(identifier, owner))
    text = row['result']
    return {'agent': _public(row), 'text': text[offset:offset+max_chars], 'offset': offset,
            'next_offset': min(offset+max_chars, len(text)), 'at_end': offset+max_chars >= len(text),
            'partial': row['state'] != 'succeeded'}


def cancel(identifier, owner, descendants=True):
    row = owned(identifier, owner)
    with store._conn() as c:
        c.execute('BEGIN IMMEDIATE')
        family = [dict(r) for r in c.execute("SELECT * FROM agent_tasks WHERE root=? AND state NOT IN ('succeeded','failed','cancelled','timed_out','lost')", (row['root'],))]
        ids = [r['id'] for r in family if r['id'] == identifier or
               (descendants and row['child'] in json.loads(r['request'])['ancestors'])]
        # Mark the entire family atomically BEFORE the snapshot can race a new
        # descendant. start() checks its parent in the same serialized SQL path.
        for aid in ids:
            c.execute("UPDATE agent_tasks SET state='cancelling',updated=? WHERE id=?", (time.time(),aid))
    for aid in ids:
        child = owned(aid, owner)
        if child['job']:
            jobs.cancel(child['job'], child['root'])
    return status(identifier, owner)


def send(identifier, owner, message, overrides=None, timeout=None):
    row = _refresh(owned(identifier, owner))
    if row['state'] not in TERMINAL or row['state'] == 'lost':
        raise ValueError('Agent must be completed and not lost; inspect uncertain side effects first')
    if row['job'] and jobs.status(row['job'], row['root'])['state'] not in jobs.TERMINAL:
        raise ValueError('Supervisor is still completing; wait briefly before the follow-up')
    if not isinstance(message, str) or not message.strip() or len(message) > 50000:
        raise ValueError('message must contain 1..50000 characters')
    from .agent import get_session
    session = get_session(row['child'])
    if session is None:
        raise ValueError('Child chat no longer exists')
    cfg = _config(session, overrides or {})
    seconds = timeout if timeout is not None else cfg.agent_timeout_seconds
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= 604800:
        raise ValueError('Invalid timeout')
    request = copy.deepcopy(row['request'])
    request['previous_runs'] = (request.get('previous_runs', []) + [{'job':row['job'], 'task':request['task'], 'state':row['state']}])[-20:]
    request.update(task=message, overrides=overrides or {}, timeout=seconds)
    with store._conn() as c:
        c.execute('BEGIN IMMEDIATE')
        root_count = c.execute("SELECT count(*) FROM agent_tasks WHERE root=? AND state NOT IN ('succeeded','failed','cancelled','timed_out','lost')", (row['root'],)).fetchone()[0]
        total_count = c.execute("SELECT count(*) FROM agent_tasks WHERE state NOT IN ('succeeded','failed','cancelled','timed_out','lost')").fetchone()[0]
        if root_count >= MAX_PENDING_PER_ROOT or total_count >= GLOBAL_PENDING:
            raise ValueError('Delegation queue is full')
        changed = c.execute("UPDATE agent_tasks SET state='queued',job=NULL,request=?,created=?,updated=?,result='',error='' WHERE id=? AND state IN ('succeeded','failed','cancelled','timed_out')", (json.dumps(request),time.time(),time.time(),identifier)).rowcount
        if not changed:
            raise ValueError('Agent state changed; follow-up was not launched')
        c.execute('UPDATE chats SET config=? WHERE id=?', (json.dumps(cfg.to_dict()),row['child']))
    try:
        _launch(identifier, seconds, row['root'])
    except Exception as exc:
        finish(identifier, '', 'Follow-up launch failed: '+str(exc), 'failed')
        raise
    return status(identifier, owner)


def active_chat(chat_id):
    task = for_chat(chat_id)
    return bool(task and _refresh(task)['state'] not in TERMINAL)


READ_ONLY_TOOLS = frozenset({'read_file','list_dir','glob_search','grep_search','read_document',
 'summarize_file','summarize_folder','focus_file','view_image','render_pdf','recall',
 'context_report','document_open','document_close','document_list','job_status','list_jobs',
 'read_job_log','environment_report','latex_inspect','scholarly_search','doi_lookup',
 'agent_start','agent_status','agent_result','agent_wait','agent_list','agent_cancel','agent_send',
 'web_search','web_open','web_find','fetch_url'})


def cost_summary(owner):
    """Unique child-chat cumulative charges (including their summaries/followups).

    Never merge this into the parent's usage ledger: that would double count.
    Unknown costs and interrupted workers stay explicitly labelled.
    """
    init()
    with store._conn() as c:
        if not c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chats'").fetchone():
            return {'cost':0.0,'agents':0,'requests':0,'missing_cost_requests':0,'uncertain_agents':0}
        rows = c.execute("SELECT a.child,a.state,COALESCE(ch.usage,json_extract(a.request,'$.archived_usage'),'{}') AS usage FROM agent_tasks a LEFT JOIN chats ch ON ch.id=a.child WHERE a.root=? OR a.parent=? OR EXISTS (SELECT 1 FROM json_each(a.request,'$.ancestors') WHERE value=?)", (owner,owner,owner)).fetchall()
    cost,requests,missing,uncertain = 0.0,0,0,0
    for row in rows:
        usage=json.loads(row['usage'] or '{}')
        cost += float(usage.get('cost') or 0)
        requests += int(usage.get('requests') or 0)
        missing += int(usage.get('missing_cost_requests') or 0)
        uncertain += int(row['state'] in {'lost','failed','timed_out','cancelled','cancelling'})
    return {'cost':cost,'agents':len(rows),'requests':requests,
            'missing_cost_requests':missing,'uncertain_agents':uncertain}


def archive_chat_usage(chat_id):
    task=for_chat(chat_id)
    chat=store.get_chat(chat_id)
    if task and chat:
        request=task['request']
        request['archived_usage']=chat['usage']
        with store._conn() as c:
            c.execute('UPDATE agent_tasks SET request=? WHERE id=?', (json.dumps(request),task['id']))


def read_only_name(name):
    from .document_formats import READ_ONLY
    from .browser_tools import BROWSER_TOOLS
    return name == 'document_call' or name in READ_ONLY_TOOLS or name in READ_ONLY or name in BROWSER_TOOLS


def tool_allowed(chat_id, name, args):
    task = for_chat(chat_id)
    if not task or not task['request']['read_only']:
        return True
    if name == 'document_call':
        from .document_formats import READ_ONLY
        operation = args.get('operation')
        return isinstance(operation, str) and operation in READ_ONLY
    return read_only_name(name)
