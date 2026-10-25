/* ctxagent UI */
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const fmt = n => n == null ? '—' : n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);
const pct = x => (x * 100).toFixed(0) + '%';

const DOCUMENT_TOOL_NAMES = [
  'document_open', 'document_close', 'document_list', 'document_prepare_edits',
  'document_json_update', 'document_table_update', 'document_apply_edits',
  'document_pdf_inspect', 'document_pdf_text', 'document_pdf_render',
  'document_docx_inspect', 'document_docx_replace',
  'document_xlsx_inspect', 'document_xlsx_update',
  'document_mesh_inspect', 'document_obj_transform',
];

const AGENT_WEB_TOOLS = ['agent_start','agent_status','agent_list','agent_result','agent_wait','agent_cancel','agent_send','web_search','web_open','web_find'];
let agentRefreshTimer = null;
const BROWSER_TOOL_NAMES = ['browser_open', 'browser_view', 'browser_list', 'browser_render', 'browser_find', 'browser_note', 'browser_summarize', 'browser_notes', 'browser_close', 'web_search', 'web_open', 'web_find'];
const browserUI = {seq: 0, busy: null, notesOffset: 0, notesTotal: 0, notesNext: 0, notesAtEnd: true};

const S = {
  chats: [], chatId: null, cfg: null, stats: null, usage: null,
  expandedChats: new Set(), chatListSeq: 0,
  tools: [], segMap: new Map(), runs: new Map(), views: new Map(), drafts: new Map(),
  renderId: null, viewEpoch: 0, contextSeq: 0, delegated: false, agentsCost: null,
  get streaming() { return this.runs.has(this.chatId); },
};

/* ------------------------------------------------------------------ api */
async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!r.ok) throw new Error((await r.text()).slice(0, 400));
  return r.status === 204 ? null : r.json();
}

async function sse(path, body, onEvent, signal) {
  const r = await fetch(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body), signal,
  });
  if (!r.ok) { onEvent({ type: 'error', error: (await r.text()).slice(0, 500) }); return; }
  const rd = r.body.getReader(); const dec = new TextDecoder(); let buf = '';
  while (true) {
    const { done, value } = await rd.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const raw = buf.slice(0, i); buf = buf.slice(i + 2);
      if (!raw.startsWith('data:')) continue;
      try { onEvent(JSON.parse(raw.slice(5).trim())); } catch (_) {}
    }
  }
}

/* -------------------------------------------------------------- markdown */
function esc(s) { return s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }

// Sentinel for parked fenced-code blocks. It must survive String.prototype.trim()
// and never occur in user text -- a space would be eaten by the trim() below
// and silently break code-block rendering.
function md(src) {
  // One renderer for stored messages, SSE deltas and folded summaries.
  return window.ChatMarkdown ? window.ChatMarkdown.render(src) : `<pre>${esc(String(src ?? ''))}</pre>`;
}

/* ------------------------------------------------------------- rendering */
const wrapEl = () => S.renderId ? S.views.get(S.renderId) : $('#msgwrap');

function inChat(id, fn) {
  const previous = S.renderId; S.renderId = id;
  try { fn(); } finally { S.renderId = previous; }
}
function saveDraft() {
  if (S.chatId) S.drafts.set(S.chatId, $('#input').value);
}

function clearMessages() { activityViews.delete(wrapEl()); wrapEl().innerHTML = ''; S.segMap.clear(); }

function scrollDown(force) {
  if (S.renderId && S.renderId !== S.chatId) return;
  const m = $('#messages');
  if (force || m.scrollHeight - m.scrollTop - m.clientHeight < 250) m.scrollTop = m.scrollHeight;
}

const activityViews = new WeakMap();

function activity() {
  const host = wrapEl();
  let a = activityViews.get(host);
  if (!a || host.lastElementChild !== a.node) {
    // One closed row for each contiguous run of non-message activity.
    const node = el('details', 'turn-activity');
    const summary = el('summary', '', 'Agent activity');
    const body = el('div', 'activity-body'); node.append(summary, body); host.appendChild(node);
    a = {node, summary, body, tools:0, notices:0, errors:0}; activityViews.set(host, a);
  }
  return a;
}

function activityLabel(a) {
  const parts = [];
  if (a.tools) parts.push(`${a.tools} tool call${a.tools === 1 ? '' : 's'}`);
  if (a.notices) parts.push(`${a.notices} event${a.notices === 1 ? '' : 's'}`);
  if (a.errors) parts.push(`${a.errors} error${a.errors === 1 ? '' : 's'}`);
  a.summary.textContent = `Activity${parts.length ? ' - ' + parts.join(' - ') : ''} (click to open/close)`;
  a.summary.title = a.summary.textContent;
  a.node.classList.toggle('has-errors', a.errors > 0);
}

function addUser(text) {
  activityViews.delete(wrapEl());
  const d = el('div', 'msg user');
  d.appendChild(el('div', 'role', 'You'));
  const b = el('div', 'bubble'); b.textContent = text; d.appendChild(b);
  wrapEl().appendChild(d); scrollDown(true); return d;
}

function addAssistant() {
  const d = el('div', 'msg assistant');
  d.appendChild(el('div', 'role', 'Assistant'));
  const b = el('div', 'bubble'); d.appendChild(b);
  // All assistant text remains a visible boundary, including tool commentary.
  activityViews.delete(wrapEl());
  wrapEl().appendChild(d); scrollDown(true);
  return { node: d, bubble: b, raw: '' };
}

function addSummary(text) {
  const d = el('div', 'msg summary');
  d.appendChild(el('div', 'role', '⎘ compacted history'));
  const b = el('div', 'bubble'); b.innerHTML = md(text); d.appendChild(b);
  const a = activity(); a.body.appendChild(d); a.notices++; activityLabel(a);
}

function argPreview(args) {
  if (!args || typeof args !== 'object') return '';
  const k = ['path', 'command', 'pattern', 'url', 'segment_id'].find(x => args[x]);
  const v = k ? String(args[k]) : JSON.stringify(args);
  return v.length > 110 ? v.slice(0, 110) + '…' : v;
}

function addToolCard(name, args) {
  const card = el('div', 'tool');
  const head = el('div', 'tool-head');
  const sp = el('div', 'spin');
  head.appendChild(sp);
  head.appendChild(el('span', 'nm', name));
  head.appendChild(el('span', 'arg', argPreview(args)));
  const st = el('span', 'st', '…'); head.appendChild(st);
  const body = el('div', 'tool-body'); // opening the activity row also reveals outputs
  head.onclick = () => body.classList.toggle('hidden');
  card.appendChild(head); card.appendChild(body);
  const a = activity(); a.body.appendChild(card); a.tools++; activityLabel(a); scrollDown();
  return { card, head, spin: sp, st, body, group:a };
}

function markToolError(card) {
  if (card.failed) return;
  card.failed = true; card.st.classList.add('err');
  card.group.errors++; activityLabel(card.group);
}

function addNotice(html, cls) {
  const d = el('div', 'notice' + (cls ? ' ' + cls : ''));
  d.innerHTML = html;
  const a = activity(); a.body.appendChild(d);
  if (cls === 'err') a.errors++; else a.notices++;
  activityLabel(a);
  scrollDown();
}

function compactionNotice(rep) {
  const tiers = (rep.tiers || []).map(t => {
    if (t.tier === '1a-dedup') return `supersede ${t.superseded} · dedup ${t.duplicates} → −${fmt(t.freed)}`;
    if (t.tier === '1b-mask') return `mask ${t.masked} · offload ${t.offloaded} → −${fmt(t.freed)}`;
    if (t.tier === '2-condense') return `condense ${t.count} → −${fmt(t.freed)}`;
    if (t.tier === '3-deep') return t.error ? `deep FAILED: ${t.error}`
      : `deep: ${t.compacted_segments} msgs (${fmt(t.compacted_tokens)}) → ${fmt(t.summary_tokens)} summary, kept ${t.kept_verbatim} verbatim (tail ${pct(t.tail_ratio)})${t.nested_prior ? ', nested prior' : ''}${t.target_tokens != null ? `; target ${fmt(t.target_tokens)}, actual ${pct(t.actual_ratio)}` : ''}${t.warning ? '; ' + t.warning : ''}${t.summary_clipped ? '; summary exceeded budget and was shortened (sources archived)' : ''}`;
    return t.tier;
  });
  addNotice(
    `<b>${rep.deep ? 'Deep compaction' : 'Context compacted'}</b> — ${fmt(rep.before)} → ${fmt(rep.after)} tokens (−${fmt(rep.freed)})` +
    `<div class="cmp-tiers">${tiers.map(esc).join('<br>')}</div>`
  );
}

/* ----------------------------------------------------------- chat loading */
function chatHierarchy() {
  const byId = new Map(S.chats.map(c => [c.id, c])), parent = new Map(), children = new Map();
  for (const c of S.chats) {
    // A missing intermediate worker can still be grouped under its main chat.
    const owner = [c.agent_parent, c.agent_root].find(id => id && id !== c.id && byId.has(id));
    if (owner) parent.set(c.id, owner);
  }
  // Corrupt/imported relationships must never hide chats or recurse forever.
  for (const c of S.chats) {
    const seen = new Set(); let id = c.id;
    while (parent.has(id)) {
      if (seen.has(id)) { parent.delete(id); break; }
      seen.add(id); id = parent.get(id);
    }
  }
  for (const c of S.chats) {
    const id = parent.get(c.id);
    if (id) { if (!children.has(id)) children.set(id, []); children.get(id).push(c); }
  }
  return {byId, parent, children, roots: S.chats.filter(c => !parent.has(c.id))};
}

function revealChatAncestors(id) {
  const {parent} = chatHierarchy();
  while (parent.has(id)) { id = parent.get(id); S.expandedChats.add(id); }
}

function renderChats() {
  const list = $('#chatlist'), {byId, parent, children, roots} = chatHierarchy();
  const focused = list.contains(document.activeElement) ? document.activeElement.closest('[data-chat-id]')?.dataset.chatId : null;
  list.replaceChildren();
  for (const id of S.expandedChats) if (!byId.has(id)) S.expandedChats.delete(id);
  const ancestors = new Set(); let active = S.chatId;
  while (parent.has(active)) { active = parent.get(active); ancestors.add(active); }
  const count = c => (children.get(c.id) || []).reduce((n, child) => n + 1 + count(child), 0);
  function append(c, host, depth) {
    const kids = children.get(c.id) || [], expanded = S.expandedChats.has(c.id);
    const group = el('div', 'chat-group');
    const d = el('div', 'chat-item' + (c.id === S.chatId ? ' active' : '') + (ancestors.has(c.id) ? ' contains-active' : ''));
    d.dataset.chatId = c.id;
    const title = c.title || 'Untitled chat';
    const button = el('button', 'chat-open'); button.type = 'button';
    button.title = title; button.setAttribute('aria-label', title);
    if (c.id === S.chatId) button.setAttribute('aria-current', 'page');
    const caret = el('span', 'chat-caret', kids.length ? (expanded ? '\u25be' : '\u25b8') : '');
    caret.setAttribute('aria-hidden', 'true'); button.appendChild(caret);
    button.appendChild(el('span', 't', title + (S.runs.has(c.id) ? ' \u00b7 running' : '')));
    if (kids.length) {
      const total = count(c), badge = el('span', 'chat-agent-count', String(total));
      badge.title = `${total} agent chat${total === 1 ? '' : 's'}`;
      button.appendChild(badge);
      button.setAttribute('aria-expanded', String(expanded));
      button.setAttribute('aria-controls', `chat-children-${c.id}`);
    }
    button.onclick = () => {
      if (c.id === S.chatId) {
        if (kids.length) {
          if (S.expandedChats.has(c.id)) S.expandedChats.delete(c.id); else S.expandedChats.add(c.id);
          renderChats();
        }
        return; // Folding must not reload history, interrupt a run or lose a draft.
      }
      if (kids.length) S.expandedChats.add(c.id);
      openChat(c.id);
    };
    d.appendChild(button);
    const x = el('button', 'x', '\u00d7'); x.type = 'button';
    x.setAttribute('aria-label', `Delete ${title}`);
    x.onclick = async ev => {
      ev.stopPropagation();
      if (!confirm(`Delete "${title}"?`)) return;
      await api(`/api/chats/${c.id}`, { method: 'DELETE' });
      S.views.delete(c.id); S.drafts.delete(c.id); S.expandedChats.delete(c.id);
      if (S.chatId === c.id) {
        S.chatId = null; S.cfg = null; S.viewEpoch++; clearMessages(); resetBrowser();
        $('#input').value = ''; autoGrow(); setSendState();
        $('#chattitle').textContent = 'No chat selected';
      }
      await loadChats();
    };
    d.appendChild(x); group.appendChild(d);
    if (kids.length) {
      const nested = el('div', 'chat-children'); nested.id = `chat-children-${c.id}`;
      nested.hidden = !expanded;
      for (const child of kids) append(child, nested, depth + 1);
      group.appendChild(nested);
    }
    host.appendChild(group);
  }
  for (const c of roots) append(c, list, 0);
  if (focused) [...list.querySelectorAll('.chat-open')].find(b => b.closest('[data-chat-id]').dataset.chatId === focused)?.focus({preventScroll: true});
}

async function loadChats() {
  const seq = ++S.chatListSeq, chats = await api('/api/chats');
  if (seq !== S.chatListSeq) return;
  S.chats = chats; renderChats();
}

async function newChat() {
  // Capture the selected chat even while it loads. The server copies its full
  // saved settings into a fresh chat, not unsaved controls or conversation.
  const body = { title: 'New chat' };
  if (S.chatId) body.source_chat = S.chatId;
  const { id } = await api('/api/chats', { method: 'POST', body });
  await openChat(id);
}

async function openChat(id) {
  revealChatAncestors(id);
  saveDraft();
  const epoch = ++S.viewEpoch;
  const hadRun = S.runs.has(id);
  S.chatId = id; S.cfg = null; S.stats = null; S.usage = null; S.agentsCost = null;
  renderChats();
  resetBrowser();
  $('#doc-list').replaceChildren(); $('#doc-status').textContent = '';
  clearTimeout(agentRefreshTimer); $('#agent-list').replaceChildren(); $('#agent-status').textContent = ''; S.delegated = false;
  $('#doc-path').value = ''; $('#doc-select').value = '';
  $('#modalhost').replaceChildren();
  $('#chattitle').textContent = 'Loading…';
  const old = $('#msgwrap');
  const view = S.views.get(id) || el('div');
  S.views.set(id, view);
  if (old !== view) { old.removeAttribute('id'); old.replaceWith(view); }
  view.id = 'msgwrap';
  view.classList.add('wrap');
  $('#input').value = S.drafts.get(id) || ''; autoGrow(); setSendState();
  // Invalidate checks issued by the previous settings view immediately.
  for (const [key, seq] of _checkSeq) _checkSeq.set(key, seq + 1);
  for (const [key, seq] of _flexSeq) _flexSeq.set(key, seq + 1);
  try {
    const data = await api(`/api/chats/${id}`);
    if (S.chatId !== id || S.viewEpoch !== epoch) return;
    S.cfg = data.config; S.delegated = !!data.delegated;
    $('#chattitle').textContent = data.title;
    if (!hadRun && !S.runs.has(id)) {
      clearMessages(); renderSegments(data.segments || []);
    }
    $('#empty')?.remove();
    fillSettings(); fillCompaction(); renderTools(); renderWorkdirs();
    applyStats(data.stats); S.usage = data.usage; applyUsage();
    setSendState(); scrollDown(true); $('#input').focus();
    await refreshContext(id);
    if (S.chatId === id && S.viewEpoch === epoch) loadChats();
  } catch (error) {
    if (S.chatId === id && S.viewEpoch === epoch) addNotice(esc(String(error)), 'err');
  }
}

function addMedia(segment) {
  const container = el('div', 'media-evidence');
  container.append(el('div', 'media-label', segment.body));
  for (const image of segment.images || []) {
    const link = document.createElement('a');
    link.href = `/api/chats/${encodeURIComponent(S.renderId || S.chatId)}/artifacts/${encodeURIComponent(image.id)}`;
    link.target = '_blank'; link.rel = 'noopener';
    const img = document.createElement('img');
    img.src = link.href; img.alt = image.label || 'Visual evidence'; img.loading = 'lazy';
    link.append(img); container.append(link);
  }
  const a = activity(); a.body.append(container); a.notices++; activityLabel(a);
  scrollDown();
}

function renderSegments(segs) {
  let pendingCards = new Map();
  for (const s of segs) {
    if (s.kind === 'system') {
      addNotice(`<b>System instructions</b><div>${md(s.body || '')}</div>`); continue;
    }
    if (s.kind === 'media') { addMedia(s); continue; }
    if (s.kind === 'user') { pendingCards.clear(); addUser(s.body); continue; }
    if (s.kind === 'summary') { addSummary(s.body); continue; }
    if (s.kind === 'assistant_text') {
      pendingCards.clear();
      const a = addAssistant(); a.bubble.innerHTML = md(s.body); continue;
    }
    if (s.kind === 'tool_call') {
      if (s.body && s.body.trim()) {
        pendingCards.clear();
        const a = addAssistant(); a.bubble.innerHTML = md(s.body);
      }
      for (const tc of s.tool_calls || []) {
        let args = {}; try { args = JSON.parse(tc.function.arguments || '{}'); } catch (_) {}
        const card = addToolCard(tc.function.name, args);
        pendingCards.set(tc.id, card);
      }
      continue;
    }
    if (s.kind === 'tool_result') {
      const card = pendingCards.get(s.tool_call_id);
      if (card) {
        card.spin.remove();
        card.st.textContent = `${fmt(s.original_tokens)} tok`;
        card.body.textContent = s.body;
        pendingCards.delete(s.tool_call_id);
      } else {
        // Compaction/imports can leave results without their original call.
        const orphan = addToolCard('Tool result', null); orphan.spin.remove();
        orphan.group.tools--; orphan.group.notices++; activityLabel(orphan.group);
        orphan.st.textContent = `${fmt(s.original_tokens)} tok`;
        orphan.body.textContent = s.body;
      }
      continue;
    }
    addNotice(`<b>${esc(s.kind || 'Event')}</b><pre>${esc(s.body || '')}</pre>`);
  }
}

/* -------------------------------------------------------------- sending */
async function send() {
  const id = S.chatId;
  if (S.streaming) { await api(`/api/chats/${id}/stop`, { method: 'POST' }); return; }
  if (S.delegated) return;
  const ta = $('#input'); const text = ta.value.trim();
  if (!text || !id || !S.cfg) return;
  ta.value = ''; S.drafts.delete(id); autoGrow();
  addUser(text);
  $('#empty')?.remove();

  const ctrl = new AbortController();
  S.runs.set(id, { ctrl, kind: 'send' }); setSendState(); loadChats();
  let cur = null;
  const cards = new Map();

  try {
    await sse(`/api/chats/${id}/send`, { message: text }, ev => inChat(id, () => {
      switch (ev.type) {
        case 'assistant_start': cur = null; break;
        case 'delta':
          if (!ev.text) break;
          if (!cur) cur = addAssistant();
          cur.raw += ev.text; cur.bubble.innerHTML = md(cur.raw); scrollDown();
          break;
        case 'media': addMedia(ev.segment); break;
        case 'tool_start': {
          cur = null; // assistant text stays visible; commands start a new activity block
          cards.set(ev.id, addToolCard(ev.name, ev.args));
          break;
        }
        case 'tool_result': {
          if (ev.name?.startsWith('agent_') && S.chatId === id) refreshAgents(id);
          const c = cards.get(ev.id);
          if (c) {
            c.spin.remove();
            c.st.textContent = `${ev.ms} ms · ${fmt(ev.tokens)} tok`;
            if (!ev.ok) markToolError(c);
            c.body.textContent = ev.segment.body;
          }
          break;
        }
        case 'compaction': compactionNotice(ev.report); break;
        case 'stats': if (S.chatId === id) applyStats(ev.stats); break;
        case 'routing': addNotice(esc(ev.routing_notice || 'Service tier routing updated'), 'routing'); break;
        case 'usage': if (S.chatId === id) { S.usage = ev.usage; applyUsage(); } break;
        case 'error': addNotice(`<b>Error</b> ${esc(ev.error)}`, 'err'); break;
        case 'cancelled': addNotice('<b>Stopped</b>'); break;
        case 'notice': addNotice(esc(ev.text || ''), 'routing'); break;
        case 'final': scrollDown(); break;
      }
    }), ctrl.signal);
  } catch (e) {
    if (e.name !== 'AbortError') inChat(id, () => addNotice(`<b>Error</b> ${esc(String(e))}`, 'err'));
  } finally {
    S.runs.delete(id);
    if (S.chatId === id) { setSendState(); refreshContext(id); }
    loadChats();
  }
}

function setSendState() {
  const b = $('#btn-send');
  b.textContent = S.streaming ? '■' : '↑';
  b.classList.toggle('stop', S.streaming);
  b.title = S.streaming ? 'Stop' : 'Send';
  b.disabled = !S.chatId || !S.cfg || S.delegated;
  $('#btn-compact').disabled = !S.cfg || S.streaming || S.delegated;
  for (const id of ['btn-doc-open', 'btn-doc-refresh', 'btn-doc-enable']) $('#' + id).disabled = !S.cfg || S.streaming || S.delegated;
  $$('#doc-list button').forEach(b => b.disabled = !S.cfg || S.streaming);
  $('#input').disabled = !S.cfg || S.delegated;
  $$('#pane-settings input, #pane-settings select, #pane-settings textarea, #pane-compaction input, #pane-compaction select').forEach(e => e.disabled = !S.cfg || S.delegated);
  updateDocumentControls(); updateBrowserControls();
  $('#input').placeholder = S.delegated ? 'Worker-owned chat: inspect here; send follow-ups from the parent Agents panel.' : 'Ask anything... (Enter to send, Shift+Enter for newline)';
  $('#btn-agent-start').disabled = !S.cfg || S.streaming || S.delegated;
  $('#btn-agent-enable').disabled = !S.cfg || S.streaming || S.delegated;
  $('#btn-agent-enable').hidden = AGENT_WEB_TOOLS.every(n => S.cfg?.enabled_tools?.includes(n));
  for (const id of ['btn-savecfg', 'btn-savecmp', 'btn-defaults']) $( '#' + id).disabled = !S.cfg || S.delegated;

}

/* ---------------------------------------------------------- context pane */
function applyStats(st) {
  if (!st) return;
  S.stats = st;
  if (st.agents_cost) { S.agentsCost = st.agents_cost; renderAgentsCost(); }
  const w = st.effective_window || 1;
  const fz = Math.min(100, 100 * st.frozen_tokens / w);
  const ht = Math.min(100 - fz, 100 * (st.hot_tokens + (st.workspace_tokens || 0) + (st.browser_tokens || 0) + (st.document_catalog_tokens || 0)) / w);
  $('#meter').innerHTML =
    `<div class="seg-frozen" style="width:${fz}%"></div><div class="seg-hot" style="width:${ht}%"></div>`;
  $('#tokpill').textContent = `${fmt(st.total)} / ${fmt(w)}`;
  $('#savedpill').textContent = `saved ${fmt(st.saved)}`;

  const bb = $('#bigbar');
  if (bb) {
    const free = Math.max(0, 100 - fz - ht);
    bb.innerHTML =
      `<div class="f" style="width:${fz}%">${fz > 12 ? fmt(st.frozen_tokens) : ''}</div>` +
      `<div class="h" style="width:${ht}%">${ht > 12 ? fmt(st.hot_tokens) : ''}</div>` +
      `<div class="free" style="width:${free}%">${free > 12 ? fmt(w - st.total) + ' free' : ''}</div>`;
    $('#s-total').textContent = fmt(st.total);
    $('#s-orig').textContent = fmt(st.original);
    $('#s-saved').textContent = `${fmt(st.saved)} (${st.original ? pct(st.saved / st.original) : '0%'})`;
    const p = $('#s-press');
    p.textContent = pct(st.pressure);
    p.className = 'v ' + (st.pressure > 0.9 ? 'bad' : st.pressure > 0.7 ? 'warn' : 'good');
    $('#s-frozen').textContent = `${fmt(st.frozen_tokens)} · ${st.hot_start} msgs`;
    $('#s-hot').textContent = `${fmt(st.hot_tokens)} actual / ${fmt(st.hot_zone_budget ?? S.cfg?.compaction?.hot_zone_tokens ?? 0)} target`;
    $('#s-hot-note').textContent = `Actual unfrozen history, not a hard ${fmt(st.hot_zone_budget ?? S.cfg?.compaction?.hot_zone_tokens ?? 0)} cap. ` +
      (st.review_candidate_tokens != null ? `${fmt(st.review_candidate_tokens)} tokens can retire at the next review; fresh/protected batches and cadence can exceed the target.` : 'Protected evidence and pending review can exceed the target.');
    $('#s-tools').textContent = fmt(st.tool_overhead);
    $('#s-doc-tools').textContent = fmt(st.document_catalog_tokens || 0);
    $('#s-cadence').textContent = `${st.rewrite_pending_batches ?? 0} / ${st.rewrite_interval_batches ?? '?'} batches`;
    $('#s-tail').textContent = `${pct(st.next_tail_ratio)}${st.deep_ready ? '' : ' (cooling)'}`;
  }
  if (st.workspace) renderDocuments(st.workspace);
}

function renderAgentsCost() {
  const cost = S.agentsCost;
  $('#u-agents-cost').textContent = cost ? '$' + Number(cost.cost || 0).toFixed(6) +
    (cost.missing_cost_requests ? ` + ? (${cost.missing_cost_requests} requests)` : '') +
    (cost.uncertain_agents ? ' + ? (interrupted workers)' : '') : 'not available';
}

function applyUsage() {
  renderAgentsCost();
  const u = S.usage || {};
  $('#u-req').textContent = fmt(u.requests || 0);
  $('#u-pt').textContent = fmt(u.prompt_tokens || 0);
  $('#u-ct').textContent = fmt(u.completion_tokens || 0);
  $('#u-cr').textContent = fmt(u.cache_read || 0);
  $('#u-chr').textContent = u.cache_hit_rate != null ? pct(u.cache_hit_rate) : '—';
  const money = v => v == null ? '—' : '$' + Number(v).toFixed(6);
  $('#u-cost').textContent = money(u.cost);
  $('#u-recent').textContent = `${u.last_cache_hit_rate == null ? '—' : pct(u.last_cache_hit_rate)} / ${u.recent_cache_hit_rate == null ? '—' : pct(u.recent_cache_hit_rate)}`;
  $('#u-cw').textContent = fmt(u.cache_write || 0);
  const accountingReady = u.accounting_version === 2;
  const fields = {input: 'input', read: 'cache_read', write: 'cache_write', output: 'output', reason: 'reasoning', unsplit: 'output_unsplit'};
  const estimatedSum = Object.values(u.component_estimates || {}).reduce((sum, value) => sum + Number(value || 0), 0);
  const knownCostShare = value => estimatedSum > 0 && value != null ? ` (${pct(Number(value) / estimatedSum)})` : '';
  for (const [id, category] of Object.entries(fields)) {
    const missing = (u.component_unknown_requests || {})[category] || 0;
    const value = (u.component_estimates || {})[category];
    $(`#u-${id}-cost`).textContent = accountingReady ? money(value) + knownCostShare(value) + (missing ? ` + ? (${missing})` : '') : 'restart required';
  }
  $('#u-legacy-cost').textContent = money(u.legacy_unallocated_cost ?? u.cost);
  $('#u-summary-cost').textContent = money(u.by_source?.summary?.cost ?? 0);
  $('#u-allocation-note').textContent = 'Percentages are shares of known category estimates only, excluding historical/unknown costs. Categories use request-time tariffs, not provider itemization. Reasoning is part of completion, not an extra charge. '
    + `Unknown category requests are marked ?. Reconciliation residual: ${money(u.reconciliation_residual ?? 0)}; reported cost without complete tariff: ${money(u.unreconciled_reported_cost ?? 0)}; requests missing cost: ${u.missing_cost_requests || 0}. Historical summary calls were not recorded.`;
  const last = (u.recent_requests || []).filter(r => r.source === 'agent').slice(-1)[0];
  $('#u-tier').textContent = last ? `${last.routing?.reported_service_tier || 'not reported'} (requested ${last.routing?.requested_tier || 'unknown'})` : '?';
  $('#u-cache-diagnostic').textContent = last ? JSON.stringify({model: last.model, provider: last.provider,
    generation_id: last.generation_id, routing: last.routing, ...last.cache_diagnostic, pricing_basis: last.pricing?.basis}, null, 2) : 'Available after the next request on the updated server.';
}

async function refreshContext(id = S.chatId) {
  if (!id || S.chatId !== id) return;
  const epoch = S.viewEpoch, seq = ++S.contextSeq;
  const d = await api(`/api/chats/${id}/context`);
  if (S.chatId !== id || S.viewEpoch !== epoch || S.contextSeq !== seq) return;
  applyStats(d.stats);
  S.usage = d.usage; applyUsage();
  renderSegList(d.segments);
  renderCmpLog(d.compaction_log);
  await refreshDocuments(id);
  if (S.chatId === id && S.viewEpoch === epoch && S.contextSeq === seq) await refreshBrowser(id);
}

function renderDocuments(workspace) {
  const views = workspace?.views || [];
  $('#doc-tokens').textContent = `${fmt(workspace?.tokens || 0)} / ${fmt(workspace?.budget || 24000)} tok`;
  const host = $('#doc-list'); host.replaceChildren();
  if (!views.length) host.appendChild(el('p', 'usage-note', 'No open documents.'));
  for (const view of views) {
    const row = el('div', 'sect');
    row.appendChild(el('div', '', view.path));
    row.appendChild(el('p', 'usage-note', `${view.select ? view.select.join(', ') : 'Full text'} · ${fmt(view.tokens)} tok · ${view.status} · ${(view.version || '').slice(0, 12)}`));
    if (view.error) row.appendChild(el('p', 'usage-note err', view.error));
    const b = el('button', 'btn', 'Close'); b.disabled = S.streaming || S.delegated;
    b.onclick = () => closeDocument(view.id);
    row.appendChild(b); host.appendChild(row);
  }
  $('#btn-doc-enable').hidden = DOCUMENT_TOOL_NAMES.every(n => S.cfg?.enabled_tools?.includes(n));
}

async function refreshDocuments(id = S.chatId) {
  if (!id || S.chatId !== id) return;
  const epoch = S.viewEpoch;
  try {
    const data = await api(`/api/chats/${id}/documents`);
    if (S.chatId !== id || S.viewEpoch !== epoch) return;
    renderDocuments(data.workspace); applyStats(data.stats);
  } catch (error) {
    if (S.chatId === id && S.viewEpoch === epoch) $('#doc-status').textContent = String(error);
  }
}

async function openDocument() {
  const id = S.chatId, epoch = S.viewEpoch;
  const path = $('#doc-path').value.trim();
  if (!path || !id) return;
  const selectors = $('#doc-select').value.split('\n').map(s => s.trim()).filter(Boolean);
  try {
    const data = await api(`/api/chats/${id}/documents`, {method: 'POST', body: {
      path, select: selectors.length ? selectors : null, max_tokens: Number($('#doc-budget').value)
    }});
    if (S.chatId !== id || S.viewEpoch !== epoch) return;
    $('#doc-status').textContent = 'Opened. Latest content will be supplied on the next model request.';
    renderDocuments(data.workspace); applyStats(data.stats);
  } catch (error) {
    if (S.chatId === id && S.viewEpoch === epoch) $('#doc-status').textContent = String(error);
  }
}

function closeDocument(documentId) {
  const id = S.chatId, epoch = S.viewEpoch;
  modal('Close document view', host => {
    host.appendChild(el('p', 'usage-note', 'The live content leaves context; historical reads remain. Optional handoff note:'));
    const note = el('textarea'); note.id = 'doc-close-note'; note.rows = 4; host.appendChild(note);
  }, [{label: 'Close view', primary: true, fn: async () => {
    if (S.chatId !== id || S.viewEpoch !== epoch) return;
    try {
      await api(`/api/chats/${id}/documents/${documentId}/close`, {method: 'POST', body: {summary: $('#doc-close-note').value}});
      await refreshContext(id);
    } catch (error) {
      if (S.chatId === id && S.viewEpoch === epoch) $('#doc-status').textContent = String(error);
    }
  }}]);
}

/* -------------------------------------------------- browser research UI */
function browserStatus(text, error = false) {
  $('#browser-status').textContent = String(text || '').slice(0, 600);
  $('#browser-status').classList.toggle('err', error);
}

function resetBrowser() {
  browserUI.seq++; browserUI.busy = null;
  browserUI.notesOffset = 0; browserUI.notesTotal = 0; browserUI.notesNext = 0; browserUI.notesAtEnd = true;
  $('#browser-url').value = ''; $('#browser-js').checked = false;
  $('#browser-view-budget').value = 4000;
  $('#browser-tabs').replaceChildren(); $('#browser-notes-list').replaceChildren(); $('#browser-note-view').replaceChildren();
  $('#browser-notes-range').textContent = ''; browserStatus('');
  renderBrowser({views: [], tokens: 0, max_tokens: 24000, note_count: 0});
}

function updateBrowserControls() {
  const unavailable = !S.chatId || !S.cfg || !!browserUI.busy;
  const locked = S.streaming || S.delegated;
  $('#browser-workspace').setAttribute('aria-busy', String(!!browserUI.busy));
  $$('#browser-workspace [data-browser-tool]').forEach(control => {
    control.disabled = unavailable || !S.cfg?.enabled_tools?.includes(control.dataset.browserTool) || (locked && !control.dataset.browserRead);
  });
  $('#btn-browser-refresh').disabled = unavailable;
  $('#btn-browser-enable').disabled = unavailable || locked;
  $('#btn-browser-enable').hidden = BROWSER_TOOL_NAMES.every(n => S.cfg?.enabled_tools?.includes(n));
  $('#btn-browser-notes-prev').disabled ||= browserUI.notesOffset === 0;
  $('#btn-browser-notes-next').disabled ||= browserUI.notesAtEnd;
  $('#browser-lock').textContent = S.delegated ? 'Worker-owned chat: browser mutations are read-only here.' : S.streaming ? 'Browser mutations are disabled while this chat is running.' : '';
}

// One request at a time per visible epoch. Old successes/errors/finalizers cannot
// touch a new view, including A -> B -> A. Reads need no enabled browser tools.
async function browserRequest(request, label, success, mutating = true) {
  if (!S.chatId || !S.cfg || browserUI.busy || (mutating && (S.streaming || S.delegated))) return;
  const scope = {id: S.chatId, epoch: S.viewEpoch, seq: ++browserUI.seq};
  const current = () => S.chatId === scope.id && S.viewEpoch === scope.epoch && browserUI.seq === scope.seq;
  browserUI.busy = scope; browserStatus(label + '…'); updateBrowserControls();
  try {
    const data = await request(scope.id);
    if (!current()) return;
    if (data.workspace) renderBrowser(data.workspace);
    if (data.stats) applyStats(data.stats);
    if (data.usage) { S.usage = data.usage; applyUsage(); }
    success?.(data);
    browserStatus(label + ' complete.');
  } catch (error) {
    if (current()) browserStatus(error.message || error, true);
  } finally {
    if (current() && browserUI.busy === scope) { browserUI.busy = null; updateBrowserControls(); }
  }
}

async function refreshBrowser(id = S.chatId) {
  if (id !== S.chatId) return;
  return browserRequest(cid => api(`/api/chats/${encodeURIComponent(cid)}/browser`), 'Refreshing tabs', data => renderBrowser(data), false);
}

function browserAction(tool, args, label, success) {
  if (!S.cfg?.enabled_tools?.includes(tool)) return;
  return browserRequest(cid => api(`/api/chats/${encodeURIComponent(cid)}/browser/action`, {
    method: 'POST', body: {tool, arguments: args}
  }), label, data => success?.(data.result), tool !== 'browser_notes');
}

function openBrowser() {
  const url = $('#browser-url').value.trim();
  try { if (!['http:', 'https:'].includes(new URL(url).protocol)) throw new Error(); }
  catch (_) { browserStatus('Enter a public HTTP(S) URL.', true); return; }
  const maxTokens = Number($('#browser-view-budget').value);
  if (!Number.isInteger(maxTokens) || maxTokens < 100 || maxTokens > 12000) { browserStatus('View token limit must be 100–12000.', true); return; }
  return browserAction('browser_open', {url, render_js: $('#browser-js').checked, max_tokens: maxTokens, max_chars: 12000}, 'Opening page');
}

function enableBrowserTools() {
  return browserRequest(cid => api(`/api/chats/${encodeURIComponent(cid)}/config`, {
    method: 'PATCH', body: {config: {enabled_tools: [...new Set([...(S.cfg.enabled_tools || []), ...BROWSER_TOOL_NAMES])]}}
  }), 'Enabling browser tools', data => { S.cfg = data.config; renderTools(); setSendState(); });
}

function browserButton(tool, label, action, read = false) {
  const button = el('button', 'btn', label), id = S.chatId, epoch = S.viewEpoch;
  button.dataset.browserTool = tool;
  if (read) button.dataset.browserRead = 'true';
  button.onclick = () => { if (S.chatId === id && S.viewEpoch === epoch) action(); };
  return button;
}

function browserSource(host, source) {
  const url = String(source.url || '').slice(0, 8000);
  const line = el('div', 'browser-source');
  try {
    if (!['http:', 'https:'].includes(new URL(url).protocol)) throw new Error();
    const link = el('a', '', url); link.href = url; link.target = '_blank'; link.rel = 'noopener noreferrer'; line.append(link);
  } catch (_) { line.textContent = url; }
  host.append(line, el('p', 'browser-source', `Cached ${source.fetched_at || '—'} · SHA-256 ${String(source.sha256 || '—').slice(0, 64)}`));
}

function renderBrowser(workspace) {
  const views = (workspace?.views || []).slice(0, 50), host = $('#browser-tabs');
  // Preserve local drafts and folding when metadata changes; never retain raw
  // source bodies. These maps exist only during this synchronous sidebar render.
  const drafts = new Map($$('textarea[data-tab-id]', host).map(e => [e.dataset.tabId, e.value]));
  const folds = new Map($$('details[data-fold]', host).map(e => [e.dataset.fold, e.open]));
  $('#browser-budget').textContent = `· ${views.length} tabs · ${fmt(workspace?.tokens || 0)} / ${fmt(workspace?.max_tokens || 24000)} tok`;
  $('#browser-note-count').textContent = `(${workspace?.note_count || 0})`;
  host.replaceChildren();
  if (!views.length) host.append(el('p', 'usage-note', 'No open browser tabs.'));
  if (workspace?.shared_tokens != null) host.append(el('p', 'usage-note', `Shared live context: ${fmt(workspace.shared_tokens)} tokens (browser + documents).`));
  for (const view of views) {
    const tab = el('details', 'browser-tab'); tab.dataset.fold = view.id; tab.dataset.tabId = view.id;
    tab.open = folds.get(view.id) ?? true;
    tab.append(el('summary', '', String(view.title || view.url || view.id).slice(0, 500)));
    browserSource(tab, view);
    tab.append(el('p', 'usage-note', `${view.status || 'ready'} · ${fmt(view.tokens)} / ${fmt(view.max_tokens)} tok · chars ${view.offset ?? 0}–${view.end ?? 0} / ${fmt(view.total_chars)}`));
    if (view.error) tab.append(el('p', 'usage-note err', String(view.error).slice(0, 600)));
    const actions = el('div', 'browser-actions');
    actions.append(
      browserButton('browser_render', 'Render', () => browserAction('browser_render', {tab_id: view.id, scroll_y: 0, max_tokens: 8000}, 'Rendering page')),
      browserButton('browser_summarize', 'Summarize', () => browserAction('browser_summarize', {tab_id: view.id, focus: '', max_tokens: 800, coverage: 'page'}, 'Summarizing page', showBrowserNote)),
      browserButton('browser_close', 'Close', () => browserAction('browser_close', {tab_id: view.id, summary: '', summarize: true, max_tokens: 800}, 'Closing and archiving summary'))
    );
    actions.lastChild.title = 'Archive a summary with the configured summary model, then remove this live view (billed).';
    tab.append(actions);
    const field = el('div', 'field'), note = el('textarea');
    note.dataset.tabId = view.id; note.dataset.browserTool = 'browser_note'; note.rows = 2; note.maxLength = 12000;
    note.setAttribute('aria-label', 'Manual research note'); note.placeholder = 'Manual research note (max 2000 tokens)…'; note.value = drafts.get(view.id) || '';
    field.append(note);
    field.append(browserButton('browser_note', 'Save note', () => {
      if (!note.value.trim()) { browserStatus('Enter a note before saving.', true); return; }
      browserAction('browser_note', {tab_id: view.id, note: note.value.trim(), title: '', quotation: ''}, 'Saving note', result => {
        // The response has re-rendered the tab, so clear only its current draft.
        for (const e of $$('textarea[data-tab-id]', host)) if (e.dataset.tabId === view.id) e.value = '';
        showBrowserNote(result);
      });
    }));
    tab.append(field);
    if (view.images?.length) {
      const images = el('details', 'browser-images'); images.dataset.fold = view.id + ':images'; images.open = folds.get(images.dataset.fold) ?? false;
      images.append(el('summary', '', `Screenshots (${view.images.length})`));
      for (const image of view.images.slice(0, 20)) {
        const link = el('a'); link.href = `/api/chats/${encodeURIComponent(S.chatId)}/artifacts/${encodeURIComponent(image.id)}`;
        link.target = '_blank'; link.rel = 'noopener noreferrer';
        const img = el('img'); img.src = link.href; img.alt = String(image.label || 'Page screenshot').slice(0, 500); img.loading = 'lazy';
        link.append(img); images.append(link);
      }
      tab.append(images);
    }
    host.append(tab);
  }
  updateBrowserControls();
}

function showBrowserNote(note) {
  const host = $('#browser-note-view'); host.replaceChildren();
  if (!note) return;
  $('#browser-archive').open = true;
  host.append(el('h4', '', String(note.title || 'Research note').slice(0, 500)));
  browserSource(host, note);
  host.append(el('p', 'usage-note', `${note.coverage || 'page'} coverage · ${note.model || note.kind || 'manual'} · ${fmt(note.tokens)} tok · cached-source note, not independently verified`));
  if (note.quotation) host.append(el('pre', '', String(note.quotation).slice(0, 4000)));
  host.append(el('pre', '', String(note.text || '').slice(0, 16000)));
}

function listBrowserNotes(offset = 0) {
  return browserAction('browser_notes', {offset, limit: 20}, 'Listing archived notes', result => {
    browserUI.notesOffset = offset; browserUI.notesTotal = result.total || 0;
    browserUI.notesNext = result.next_offset ?? offset + (result.notes || []).length;
    browserUI.notesAtEnd = result.at_end !== false;
    const host = $('#browser-notes-list'); host.replaceChildren();
    $('#browser-notes-range').textContent = result.total ? `${offset + 1}–${browserUI.notesNext} of ${result.total} notes` : 'No archived notes.';
    for (const note of (result.notes || []).slice(0, 20)) {
      const row = el('div', 'browser-note-row'); row.append(el('div', '', String(note.title || note.id).slice(0, 500)));
      browserSource(row, note);
      row.append(browserButton('browser_notes', 'Show note', () => browserAction('browser_notes', {note_id: note.id, offset: 0, limit: 20}, 'Loading archived note', showBrowserNote), true));
      host.append(row);
    }
  });
}

async function refreshAgents(id = S.chatId) {
  if (!id || S.chatId !== id) return;
  const epoch = S.viewEpoch; clearTimeout(agentRefreshTimer);
  try {
    const d = await api(`/api/chats/${id}/agents`);
    if (S.chatId !== id || S.viewEpoch !== epoch) return;
    if (d.cost) { S.agentsCost = d.cost; renderAgentsCost(); }
    const agents = d.agents || [], host = $('#agent-list'); host.replaceChildren();
    $('#agent-status').textContent = `${agents.length} workers shown; child usage is separate from parent usage.`;
    for (const a of agents) {
      const row = el('div', 'sect'); row.appendChild(el('p', '', `${a.state}: ${a.task}`));
      row.appendChild(el('p', 'usage-note', `${a.read_only ? 'read-only' : 'write-capable'} - ${a.usage?.requests || 0} requests - child ${a.child}`));
      if (a.error) row.appendChild(el('p', 'usage-note err', a.error));
      const inspect = el('button', 'btn', 'Inspect chat'); inspect.onclick = () => openChat(a.child); row.appendChild(inspect);
      const result = el('button', 'btn', 'Read report');
      result.onclick = () => showAgentReport(id, a.id); row.appendChild(result);
      if (!['succeeded','failed','cancelled','timed_out','lost'].includes(a.state)) {
        const cancel = el('button', 'btn', 'Cancel'); cancel.onclick = async () => {
          if (!confirm('Cancel worker and descendants? Edits are not undone.')) return;
          try { await api(`/api/chats/${id}/agents/${a.id}/cancel`, {method:'POST'}); await refreshAgents(id); }
          catch (error) { if (S.chatId === id) $('#agent-status').textContent = String(error); }
        }; row.appendChild(cancel);
      } else if (a.state !== 'lost') {
        const follow = el('button', 'btn', 'Follow up'); follow.onclick = () => {
          const message = prompt('Follow-up task (retains child context/settings):'); if (!message) return;
          api(`/api/chats/${id}/agents/${a.id}/send`, {method:'POST',body:{message}}).then(() => refreshAgents(id)).catch(e => { if (S.chatId === id) $('#agent-status').textContent = String(e); });
        }; row.appendChild(follow);
      }
      host.appendChild(row);
    }
    if (agents.some(a => !['succeeded','failed','cancelled','timed_out','lost'].includes(a.state)))
      agentRefreshTimer = setTimeout(() => { if (S.chatId === id && S.viewEpoch === epoch) refreshAgents(id); }, 4000);
  } catch (error) { if (S.chatId === id && S.viewEpoch === epoch) $('#agent-status').textContent = String(error); }
}

async function showAgentReport(id, aid, offset = 0) {
  const epoch = S.viewEpoch;
  try {
    const report = await api(`/api/chats/${id}/agents/${aid}/result?offset=${offset}&max_chars=12000`);
    if (S.chatId !== id || S.viewEpoch !== epoch) return;
    modal(`Agent handoff (${report.agent.state})`, host => {
      host.appendChild(el('p', 'usage-note', `Offset ${offset}; ${report.partial ? 'incomplete/uncertain' : 'completed'}. Verify child claims against sources.`));
      const pre = el('pre'); pre.textContent = report.text; host.appendChild(pre);
    }, report.at_end ? [] : [{label:'Next page',fn:() => showAgentReport(id, aid, report.next_offset)}]);
  } catch (error) { if (S.chatId === id && S.viewEpoch === epoch) $('#agent-status').textContent = String(error); }
}

function startAgent() {
  const id = S.chatId, epoch = S.viewEpoch;
  modal('Delegate task', host => {
    host.appendChild(el('p','usage-note','Child inherits saved settings and live view specs. Only this explicit handoff is copied, not the full history. Model calls incur costs.'));
    for (const [key,label] of [['task','Task'],['context','Context / success criteria'],['overrides','Settings overrides (JSON; optional)']]) {
      host.appendChild(el('label','',label)); const t = el('textarea'); t.id = 'delegate-' + key; t.rows = key === 'task' ? 3 : 2;
      if (key === 'overrides') t.value = '{}'; host.appendChild(t);
    }
    const label = el('label'); const check = el('input'); check.type = 'checkbox'; check.id = 'delegate-readonly'; check.checked = true;
    label.appendChild(check); label.appendChild(document.createTextNode(' Enforce read-only tools (recommended for reviews/research)')); host.appendChild(label);
  }, [{label:'Start worker',primary:true,fn:async () => {
    if (S.chatId !== id || S.viewEpoch !== epoch) return;
    try {
      await api(`/api/chats/${id}/agents`,{method:'POST',body:{task:$('#delegate-task').value,
        context:$('#delegate-context').value, overrides:JSON.parse($('#delegate-overrides').value || '{}'), read_only:$('#delegate-readonly').checked}});
      $('#modalhost').replaceChildren(); $('#agent-details').open = true; await refreshAgents(id); await loadChats();
    } catch (error) { if (S.chatId === id && S.viewEpoch === epoch) $('#agent-status').textContent = String(error); }
  }}]);
}

function renderSegList(segs) {
  const host = $('#seglist'); host.innerHTML = '';
  $('#segcount').textContent = `(${segs.length})`;
  for (const s of segs.slice().reverse()) {
    const d = el('div', 'seg ' + (s.hot ? 'hot' : 'frozen'));
    d.appendChild(el('div', 'bar'));
    d.appendChild(el('span', 'kind', s.tool_name || s.kind.replace('assistant_text', 'assistant').replace('tool_result', 'result').replace('tool_call', 'call')));
    d.appendChild(el('span', 'pv', s.preview || '—'));
    if (s.state !== 'original') d.appendChild(el('span', 'tag ' + s.state, s.state.slice(0, 4)));
    else if (s.pinned) d.appendChild(el('span', 'tag pinned', 'pin'));
    const tk = el('span', 'tk', s.original_tokens > s.tokens ? `${fmt(s.tokens)}←${fmt(s.original_tokens)}` : fmt(s.tokens));
    d.appendChild(tk);
    d.title = `${s.kind} · significance ${s.significance} · ${s.frozen ? 'frozen' : 'hot'}${s.note ? ' · ' + s.note : ''}${s.resource_key ? '\n' + s.resource_key : ''}`;
    d.onclick = () => showSegment(s);
    host.appendChild(d);
  }
}

async function showSegment(s) {
  const id = S.chatId, epoch = S.viewEpoch;
  const full = await api(`/api/chats/${id}/segments/${s.id}/full`);
  if (S.chatId !== id || S.viewEpoch !== epoch) return;
  const changed = full.original && full.original !== full.current;
  modal(`Segment ${s.id}`, host => {
    const meta = el('div');
    meta.innerHTML =
      `<div class="kv"><span class="k">kind</span><span class="v">${esc(s.kind)}</span></div>` +
      `<div class="kv"><span class="k">state</span><span class="v">${esc(s.state)}${s.note ? ' — ' + esc(s.note) : ''}</span></div>` +
      `<div class="kv"><span class="k">zone</span><span class="v">${s.frozen ? 'frozen (cache-eligible)' : 'hot (rewritable)'}</span></div>` +
      `<div class="kv"><span class="k">significance</span><span class="v">${s.significance}</span></div>` +
      `<div class="kv"><span class="k">tokens</span><span class="v">${s.tokens} (was ${s.original_tokens})</span></div>` +
      (s.resource_key ? `<div class="kv"><span class="k">resource</span><span class="v">${esc(s.resource_key)}</span></div>` : '');
    host.appendChild(meta);
    host.appendChild(el('h4', null, 'In context now'));
    const a = el('pre'); a.textContent = full.current; host.appendChild(a);
    if (changed) {
      host.appendChild(el('h4', null, 'Original (stored, recoverable)'));
      const b = el('pre'); b.textContent = full.original.slice(0, 20000); host.appendChild(b);
    }
  }, [
    changed ? { label: 'Restore original', primary: true, fn: async () => {
      await api(`/api/chats/${id}/segments/${s.id}/restore`, { method: 'POST' });
      refreshContext();
    } } : null,
    { label: s.pinned ? 'Unpin' : 'Pin (never compact)', fn: async () => {
      await api(`/api/chats/${id}/segments/${s.id}/pin`, { method: 'POST' });
      refreshContext();
    } },
  ].filter(Boolean));
}

function renderCmpLog(log) {
  const host = $('#cmplog'); if (!host) return;
  host.innerHTML = '';
  if (!log || !log.length) { host.appendChild(el('p', 'hint', 'Nothing compacted yet.')); return; }
  for (const r of log.slice().reverse()) {
    const d = el('div', 'logitem' + (r.deep ? ' deep' : ''));
    const hd = el('div', 'hd');
    hd.appendChild(el('span', null, r.deep ? 'deep compaction' : 'incremental'));
    hd.appendChild(el('span', null, `−${fmt(r.freed)}`));
    d.appendChild(hd);
    d.appendChild(el('div', 'sub', `${fmt(r.before)} → ${fmt(r.after)} · ${(r.tiers || []).map(t => t.tier).join(', ')} · ${r.duration_ms || 0} ms`));
    host.appendChild(d);
  }
}

/* --------------------------------------------------------------- tools UI */
function renderTools() {
  const host = $('#toollist'); host.innerHTML = '';
  const on = new Set(S.cfg?.enabled_tools || []);
  for (const t of S.tools) {
    const row = el('label', 'toolrow');
    const cb = el('input'); cb.type = 'checkbox'; cb.checked = on.has(t.name); cb.disabled = S.delegated || S.streaming;
    cb.onchange = () => {
      const set = new Set(S.cfg.enabled_tools);
      cb.checked ? set.add(t.name) : set.delete(t.name);
      S.cfg.enabled_tools = [...set];
      updateBrowserControls();
      patchCfg({ enabled_tools: S.cfg.enabled_tools });
    };
    row.appendChild(cb);
    const nm = el('span', 'nm', t.name); row.appendChild(nm);
    if (t.dangerous) row.appendChild(el('span', 'warn', 'writes'));
    row.title = t.description;
    host.appendChild(row);
  }
}

function renderWorkdirs() {
  const host = $('#wdlist'); host.innerHTML = '';
  const wds = S.cfg?.workdirs || [];
  if (!wds.length) host.appendChild(el('p', 'hint', 'No directory yet — file tools are disabled until you add one.'));
  for (const w of wds) {
    const d = el('div', 'wd');
    d.appendChild(el('span', 'p', w));
    const x = el('button', 'x', '×');
    x.onclick = () => {
      S.cfg.workdirs = S.cfg.workdirs.filter(p => p !== w);
      patchCfg({ workdirs: S.cfg.workdirs }); renderWorkdirs();
    };
    d.appendChild(x); host.appendChild(d);
  }
  const btn = $('#btn-wd');
  if (wds.length) {
    const name = wds[0].replace(/[\\/]+$/, '').split(/[\\/]/).pop() || wds[0];
    $('#wdhint').textContent = name + (wds.length > 1 ? ` +${wds.length - 1}` : '');
    btn.title = wds.join('\n');
    btn.classList.remove('unset');
  } else {
    $('#wdhint').textContent = 'set working directory';
    btn.title = 'File and shell tools are disabled until a directory is chosen';
    btn.classList.add('unset');
  }
}

/* ------------------------------------------------------------ settings UI */
function fillSettings() {
  const c = S.cfg; if (!c) return;
  $('#f-model').value = c.model;
  $('#f-model-pricing').value = c.pricing_mode || 'standard';
  $('#f-sum-pricing').value = c.compaction?.summary_pricing_mode || 'standard';
  $('#f-ctx').value = c.context_window;
  $('#f-maxout').value = c.max_output_tokens;
  $('#f-maxiter').value = c.max_tool_iterations ?? 0;
  $('#f-agent-parallel').value = c.agent_max_parallel ?? 4;
  $('#f-agent-depth').value = c.agent_max_depth ?? 2;
  $('#f-agent-timeout').value = c.agent_timeout_seconds ?? 3600;
  $('#f-temp').value = c.temperature;
  $('#f-temp-v').textContent = c.temperature;
  $('#f-sys').value = c.system_prompt || '';
  $('#f-summodel').value = c.compaction?.summary_model || '';
  $('#f-reasoning').value = c.reasoning_effort || 'high';
  $('#f-sumreasoning').value = c.compaction?.summary_reasoning || 'high';
  $('#f-provpolicy').value = c.provider_policy || 'eu_us';
  applyProvider(providerOf(c.model));
  $('#modelpill').textContent = c.model + (c.pricing_mode && c.pricing_mode !== 'standard' ? ' · ' + c.pricing_mode.replace('_', ' ') : '');
  checkProviders();
}

function fillCompaction() {
  const k = S.cfg?.compaction; if (!k) return;
  $('#c-hot-state').textContent = `Saved budget: ${fmt(k.hot_zone_tokens)} tokens. Changes require Save engine config.`;
  $('#c-interval').value = k.rewrite_interval_batches ?? 5;
  $('#c-newtokens').value = k.rewrite_max_new_tokens ?? 20000;
  $('#c-pressure').value = k.rewrite_pressure_ratio ?? 0.70;
  $('#c-enabled').checked = k.enabled;
  $('#c-review').checked = k.review_before_freeze === true;
  $('#c-delivery').value = k.initial_result_max_tokens ?? 0;
  $('#c-docs').checked = k.document_compaction_enabled !== false;
  $('#c-target').value = k.deep_target_ratio ?? 0.1;
  $('#c-hot').value = k.hot_zone_tokens; $('#c-hot-v').textContent = fmt(k.hot_zone_tokens);
  $('#c-protobs').value = k.protected_recent_observations;
  $('#c-protusr').value = k.protected_recent_user_turns;
  $('#c-dedup').checked = k.dedup_enabled;
  $('#c-mask').checked = k.masking_enabled;
  $('#c-offload').checked = k.offload_enabled;
  $('#c-sig').checked = k.significance_enabled;
  $('#c-sigt').value = k.significance_threshold; $('#c-sigt-v').textContent = k.significance_threshold;
  $('#c-deep').checked = k.deep_enabled;
  $('#c-trig').value = k.deep_trigger_ratio; $('#c-trig-v').textContent = pct(k.deep_trigger_ratio);
  $('#c-cool').value = k.deep_cooldown_turns;
  $('#c-mingain').value = k.deep_min_gain_tokens;
  $('#c-summodel').value = k.summary_model;
  updateDocumentControls(); explainTail();
}

function updateDocumentControls() {
  const disabled = S.delegated || !$('#c-docs').checked;
  for (const id of ['c-dedup', 'c-mask', 'c-offload', 'c-sig', 'c-sigt']) {
    const input = $('#' + id); input.disabled = disabled;
    input.closest('.field, .switchrow')?.classList.toggle('document-disabled', disabled);
  }
}

function explainTail() {
  $('#c-tailexpl').textContent = `Target: reduce TOTAL context to ${pct(+$('#c-target').value)} of its current size, not by that amount. Includes summary and tool/system overhead. Explicit pins and the newest tool group can impose a larger minimum; this is reported. Recalled documents can be archived again, losslessly.`;
}

function readCompaction() {
  return {
    enabled: $('#c-enabled').checked,
    review_before_freeze: $('#c-review').checked,
    initial_result_max_tokens: +$('#c-delivery').value,
    document_compaction_enabled: $('#c-docs').checked,
    deep_target_ratio: +$('#c-target').value,
    hot_zone_tokens: +$('#c-hot').value,
    rewrite_interval_batches: +$('#c-interval').value,
    rewrite_max_new_tokens: +$('#c-newtokens').value,
    rewrite_pressure_ratio: +$('#c-pressure').value,
    protected_recent_observations: +$('#c-protobs').value,
    protected_recent_user_turns: +$('#c-protusr').value,
    dedup_enabled: $('#c-dedup').checked,
    masking_enabled: $('#c-mask').checked,
    offload_enabled: $('#c-offload').checked,
    significance_enabled: $('#c-sig').checked,
    significance_threshold: +$('#c-sigt').value,
    deep_enabled: $('#c-deep').checked,
    deep_trigger_ratio: +$('#c-trig').value,
    deep_cooldown_turns: +$('#c-cool').value,
    deep_min_gain_tokens: +$('#c-mingain').value,
    summary_model: $('#c-summodel').value.trim(),
  };
}

// ---- model pricing -------------------------------------------------------
const PRICES = new Map();
async function loadPrices() {
  try {
    const d = await api('/api/models');
    for (const m of d.models || []) PRICES.set(m.id, m);
  } catch (_) {}
}

function showPrice(model, hostId, ctxTokens) {
  const host = $(hostId);
  if (!host) return;
  const m = PRICES.get(model);
  if (!m || m.prompt_price == null) {
    host.className = 'pricebox';
    host.innerHTML = '';
    return;
  }
  const inM = Number(m.prompt_price) * 1e6;
  const outM = Number(m.completion_price) * 1e6;
  // A turn re-sends the whole context, so input price x context is the number
  // that actually predicts the bill -- not the headline per-token rate.
  const est = ctxTokens ? (inM * ctxTokens / 1e6) : null;
  host.className = 'pricebox show';
  host.innerHTML =
    '<div class="row"><span>Catalogue minimum (may require a tier)</span></div>' +
    `<div class="row"><span>input</span><span>$${inM.toFixed(3)} / Mtok</span></div>` +
    `<div class="row"><span>output</span><span>$${outM.toFixed(3)} / Mtok</span></div>` +
    `<div class="row"><span>context</span><span>${fmt(m.context_length || 0)}</span></div>` +
    (est != null
      ? `<div class="row est"><span>a full ${fmt(ctxTokens)} context costs</span>` +
        `<span>$${est.toFixed(4)} / request</span></div>`
      : '');
}

// ---- provider region checks ---------------------------------------------
const _checkSeq = new Map();
async function checkOne(model, hostId) {
  const host = $(hostId);
  if (!host) return;
  _checkSeq.set(hostId, (_checkSeq.get(hostId) || 0) + 1);
  if (!model) { host.className = 'provcheck'; host.textContent = ''; return; }
  if (S.keys && !S.keys[providerOf(model)]) {
    host.className = 'provcheck show bad';
    host.innerHTML = `<b>✗ no API key</b> — ${esc(model)} needs <code>${KEY_NAMES[providerOf(model)]}</code> in .env.`;
    return;
  }
  const policy = $('#f-provpolicy').value || 'eu_us';
  if (policy !== 'eu_us') {
    host.className = 'provcheck show busy';
    host.textContent = 'Region policy is off — any provider may serve this model.';
    return;
  }
  const seq = (_checkSeq.get(hostId) || 0) + 1;
  _checkSeq.set(hostId, seq);
  host.className = 'provcheck show busy';
  host.textContent = `checking where ${model} can be served…`;
  try {
    const d = await api('/api/providers/check', { method: 'POST', body: { model, policy } });
    if (seq !== _checkSeq.get(hostId)) return;
    host.className = 'provcheck show ' + (d.ok ? 'ok' : 'bad');
    host.innerHTML = `<b>${d.ok ? '✓ EU/US available' : '✗ not available in-region'}</b> — ${esc(d.message)}`;
  } catch (e) {
    if (seq !== _checkSeq.get(hostId)) return;
    host.className = 'provcheck show busy';
    host.textContent = 'Could not check provider regions.';
  }
}

const _flexSeq = new Map();
async function checkFlex(which, refresh = false) {
  const host = $(`#${which}-flex`);
  const model = $(which === 'model' ? '#f-model' : '#f-summodel').value.trim();
  const mode = $(`#f-${which}-pricing`).value;
  const seq = (_flexSeq.get(which) || 0) + 1; _flexSeq.set(which, seq);
  if (!model) { host.className = 'provcheck'; host.textContent = ''; return; }
  if (providerOf(model) === 'anthropic') {
    host.className = 'provcheck show busy';
    host.textContent = 'Anthropic has no Flex tier: direct API requests are billed at standard rates. (Batch, -50%, is asynchronous and unsuitable for chat.)';
    return;
  }
  host.className = 'provcheck show busy';
  host.textContent = 'Discovering Flex endpoints and advertised prices (no generation)...';
  try {
    const data = await api('/api/providers/flex', {method: 'POST', body: {
      model, policy: $('#f-provpolicy').value || 'eu_us',
      prompt_tokens: Math.max(0, Math.ceil(S.stats?.total || 0)), needs_tools: which === 'model', refresh
    }});
    if (_flexSeq.get(which) !== seq) return;
    host.className = 'provcheck show ' + (data.available ? 'ok' : mode === 'flex_only' ? 'bad' : 'busy');
    host.replaceChildren();
    const title = data.available ? 'Flex endpoints found' : (data.known ? 'No eligible Flex endpoints advertised' : 'Flex discovery unavailable');
    host.append(el('b', '', title));
    for (const row of (data.candidates || []).slice(0, 6)) {
      const amount = key => row.rates?.[key] == null ? '?' : '$' + (row.rates[key] * 1e6).toFixed(3);
      const reduction = key => row.discount_fractions?.[key] == null ? 'unknown comparison' : `${Math.round(row.discount_fractions[key] * 100)}% lower`;
      host.append(el('div', 'flex-endpoint', `${row.tag}: input ${amount('prompt')} / output ${amount('completion')} per Mtok. ${reduction('prompt')} input, ${reduction('completion')} output vs matching standard endpoint.`));
      if (row.status != null && row.status !== 0) host.append(el('div', '', 'Endpoint currently reports degraded/unavailable status; capacity is not guaranteed.'));
    }
    const behavior = mode === 'standard' ? 'Standard selected; Flex will not be requested by this setting.' :
      mode === 'flex_only' ? 'Flex only: no standard-price fallback if unavailable.' :
      'Try Flex: eligible endpoints first, then standard pricing only if unavailable/rejected before streaming. No retry after partial output or network timeout.';
    host.append(el('p', 'usage-note', `${behavior} Prices use an estimated ${data.prompt_tokens_estimate || 0} prompt tokens for long-context thresholds. Returned tier and provider cost determine the actual bill. Metadata checks are free; real generation attempts are billed normally.`));
  } catch (error) {
    if (_flexSeq.get(which) !== seq) return;
    host.className = 'provcheck show busy';
    host.textContent = 'Flex discovery failed. Check backend restart/network. No model test was sent.';
  }
}

/* ----------------------------------------------- provider / API key switch */
// Bare Claude ids use the direct Anthropic API, everything else OpenRouter
// (same rule as config.key_problem on the backend).
const PROVIDER_DEFAULTS = {
  anthropic: {model: 'claude-opus-5-5', summary: 'claude-sonnet-5-5'},
  openrouter: {model: 'anthropic/claude-sonnet-5', summary: 'openai/gpt-5.6-luna'},
};
const KEY_NAMES = {anthropic: 'ANTHROPIC_API_KEY', openrouter: 'OPENROUTER_API_KEY'};
const providerOf = id => /^claude-/.test(id || '') ? 'anthropic' : 'openrouter';

// anthropic/claude-opus-5.5[:batch] <-> claude-opus-5-5[-20251101]; null if unknown.
function counterpart(id, target) {
  if (!id || providerOf(id) === target) return id || null;
  const known = S.models?.[target] || new Set();
  if (target === 'anthropic') {
    const m = id.match(/^anthropic\/(claude-[^:]+)/);
    if (!m) return null;
    const base = m[1].replace(/\./g, '-');
    if (!known.size) return base;
    return [...known].find(k => k === base || (k.startsWith(base + '-') && /^\d{8}$/.test(k.slice(base.length + 1)))) || null;
  }
  const candidate = 'anthropic/' + id.replace(/-\d{8}$/, '').replace(/-(\d+)-(\d+)$/, '-$1.$2');
  return !known.size || known.has(candidate) ? candidate : null;
}

function applyProvider(p) {
  $('#f-provider').value = p;
  const list = p === 'anthropic' ? 'modellist-anthropic' : 'modellist';
  for (const id of ['#f-model', '#f-summodel', '#c-summodel']) $(id).setAttribute('list', list);
  $('#f-model-label').textContent = p === 'anthropic' ? 'Model id (Anthropic API)' : 'Model id (OpenRouter)';
  $('#f-provider-note').textContent = S.keys && !S.keys[p]
    ? `No ${KEY_NAMES[p]} in .env: requests through this provider will be refused.`
    : 'Switching maps the chat and compaction models to their counterparts. Save settings to apply.';
}

function switchProvider() {
  const p = $('#f-provider').value;
  for (const [sel, kind] of [['#f-model', 'model'], ['#f-summodel', 'summary']]) {
    const current = $(sel).value.trim();
    if (providerOf(current) !== p) $(sel).value = counterpart(current, p) || PROVIDER_DEFAULTS[p][kind];
  }
  applyProvider(p);
  checkProviders();
}

// Flex is an OpenRouter tier. The direct Anthropic API has none, so only
// Standard is offered for a model on that route.
function syncPricing(which, model) {
  const select = $(`#f-${which}-pricing`);
  const direct = providerOf(model) === 'anthropic';
  for (const o of select.options) o.disabled = direct && o.value !== 'standard';
  if (direct) select.value = 'standard';
  $(`#${which}-flex-refresh`).disabled = direct;
}

function checkProviders() {
  const agent = $('#f-model').value.trim();
  const summ = $('#f-summodel').value.trim();
  if (agent && $('#f-provider').value !== providerOf(agent)) applyProvider(providerOf(agent));
  syncPricing('model', agent);
  syncPricing('sum', summ);
  checkOne(agent, '#model-check');
  checkOne(summ, '#sum-check');
  checkFlex('model');
  checkFlex('sum');
  showPrice(agent, '#model-price', +$('#f-ctx').value || 0);
  // The summariser sees the compacted trajectory, not the whole window.
  showPrice(summ, '#sum-price', S.stats ? S.stats.total : 0);
}

async function patchCfg(patch) {
  const id = S.chatId, epoch = S.viewEpoch;
  if (!id) return false;
  try {
    const oldSummary = S.cfg?.compaction?.summary_model;
    const r = await api(`/api/chats/${id}/config`, { method: 'PATCH', body: { config: patch } });
    if (S.chatId !== id || S.viewEpoch !== epoch) return false;
    S.cfg = r.config; applyStats(r.stats);
    // Synchronize untouched summary-model mirrors without replacing unsaved edits.
    for (const name of ['#f-summodel', '#c-summodel']) {
      if ($(name).value.trim() === oldSummary) $(name).value = S.cfg.compaction.summary_model;
    }
    return true;
  } catch (err) {
    alert(`Settings were NOT saved: ${err.message || err}`);
    throw err;
  }
}

/* ----------------------------------------------------------------- modal */
function modal(title, build, buttons) {
  const host = $('#modalhost');
  const m = el('div', 'modal');
  const box = el('div', 'box');
  const hd = el('div', 'hd'); hd.appendChild(el('span', null, title));
  hd.appendChild(el('span', 'spacer'));
  const close = el('button', 'iconbtn', '×'); close.onclick = () => m.remove(); hd.appendChild(close);
  const bd = el('div', 'bd'); build(bd);
  const ft = el('div', 'ft');
  for (const b of buttons || []) {
    const btn = el('button', 'btn' + (b.primary ? ' primary' : ''), b.label);
    btn.onclick = async () => { await b.fn(); if (!b.keepOpen) m.remove(); };
    ft.appendChild(btn);
  }
  const done = el('button', 'btn', 'Close'); done.onclick = () => m.remove(); ft.appendChild(done);
  box.appendChild(hd); box.appendChild(bd); box.appendChild(ft); m.appendChild(box);
  m.onclick = e => { if (e.target === m) m.remove(); };
  host.appendChild(m);
  return m;
}

async function pickDirectory() {
  let cur = '';
  const owner = S.chatId;
  const m = modal('Choose a working directory', bd => { bd.id = 'dirbd'; }, [
    { label: 'Use this directory', primary: true, keepOpen: true, fn: async () => {
      if (!cur || S.chatId !== owner) return;
      const set = new Set(S.cfg.workdirs); set.add(cur);
      S.cfg.workdirs = [...set];
      await patchCfg({ workdirs: S.cfg.workdirs });
      renderWorkdirs(); m.remove();
    } },
  ]);
  const bd = $('#dirbd', m);
  async function draw(path) {
    const d = await api('/api/browse', { method: 'POST', body: { path } });
    cur = d.path; bd.innerHTML = '';
    bd.appendChild(el('div', 'crumb', d.path || 'Pick a drive or your home folder'));
    if (d.parent) {
      const up = el('div', 'dirrow', '⬑ ..'); up.onclick = () => draw(d.parent); bd.appendChild(up);
    }
    for (const x of d.dirs) {
      const r = el('div', 'dirrow', '📁 ' + x.name);
      r.onclick = () => draw(x.path);
      bd.appendChild(r);
    }
    if (!d.dirs.length) bd.appendChild(el('p', 'hint', '(no subdirectories)'));
  }
  draw(S.cfg?.workdirs?.[0] || '');
}

async function forceCompact() {
  if (!S.chatId || !S.cfg || S.streaming) return;
  const id = S.chatId;
  modal('Deep compaction', bd => {
    bd.innerHTML = `<p>Reduce total context to <b>${pct(S.cfg.compaction.deep_target_ratio ?? .1)}</b> of its current size. Fixed overhead, explicit pins and the newest tool group may limit reduction; the result reports this. Originals stay recallable.</p>`;
    const t = el('textarea'); t.id = 'cmpinstr'; t.rows = 3;
    t.placeholder = 'Optional focus instructions'; bd.appendChild(t);
  }, [
    { label: 'Compact now', primary: true, fn: async () => {
      if (S.chatId !== id || S.runs.has(id)) return;
      const instructions = $('#cmpinstr')?.value || '';
      const ctrl = new AbortController(); S.runs.set(id, {ctrl, kind: 'compact'});
      setSendState(); addNotice('<b>Compacting…</b>');
      try {
        await sse(`/api/chats/${id}/compact`, { instructions }, ev => inChat(id, () => {
          if (ev.type === 'compaction') compactionNotice(ev.report);
          if (ev.type === 'stats' && S.chatId === id) applyStats(ev.stats);
          if (ev.type === 'error') addNotice(`<b>Error</b> ${esc(ev.error)}`, 'err');
        }), ctrl.signal);
      } catch (error) {
        inChat(id, () => addNotice(esc(String(error)), 'err'));
      } finally {
        S.runs.delete(id);
        if (S.chatId === id) { setSendState(); refreshContext(id); }
        loadChats();
      }
    } },
  ]);
}

/* ------------------------------------------------------------------ init */
function autoGrow() {
  const ta = $('#input');
  ta.style.height = 'auto';
  ta.style.height = Math.min(220, ta.scrollHeight) + 'px';
}

async function init() {
  const h = await api('/api/health');
  S.tools = h.tools;
  if (h.backend_build !== 'browser-workspace-v1' || h.restart_required || h.accounting_version !== 2) {
    $('#backend-status').className = 'banner';
    $('#backend-status').textContent = 'BACKEND RESTART REQUIRED. Browser refresh alone does not activate new code. Cost categories/cadence may be unavailable. Stop the Python server, restart it, then reload.';
  } else {
    $('#backend-status').className = '';
    $('#backend-status').textContent = '';
    $('#keystate').title = `Backend ${h.backend_build}, PID ${h.server_pid}, started ${new Date(h.server_started_at * 1000).toLocaleString()}`;
  }
  $('#keystate').textContent = h.has_key ? '' : 'no API key';
  S.keys = {openrouter: h.has_openrouter_key ?? h.has_key, anthropic: !!h.has_anthropic_key};
  for (const o of $('#f-provider').options) {
    if (!S.keys[o.value]) o.textContent += ' — no key in .env';
  }
  if (!h.has_key) {
    $('#banner').innerHTML =
      '<div class="banner">No <code>OPENROUTER_API_KEY</code> or <code>ANTHROPIC_API_KEY</code> found. Put it in a <code>.env</code> file next to the app and restart.</div>';
  }
  $('#foot').textContent = `${h.tools.length} tools · ${h.levels_text || ''}`;
  S.llmLevels = h.llm_levels || {};
  S.models = {openrouter: new Set(), anthropic: new Set()};

  await loadChats();
  const first = chatHierarchy().roots[0];
  if (first) openChat(first.id); else newChat();

  // models for autocomplete (best effort)
  api('/api/models').then(d => {
    for (const m of d.models || []) {
      PRICES.set(m.id, m);
      const provider = m.provider === 'anthropic' ? 'anthropic' : 'openrouter';
      S.models[provider].add(m.id);
      const o = el('option'); o.value = m.id;
      o.label = `${m.name || m.id} · ${fmt(m.context_length)} ctx`;
      $(provider === 'anthropic' ? '#modellist-anthropic' : '#modellist').appendChild(o);
    }
    checkProviders();
  }).catch(() => {});

  // wiring
  $('#btn-new').onclick = newChat;
  $('#btn-send').onclick = send;
  $('#btn-compact').onclick = forceCompact;
  $('#btn-doc-open').onclick = openDocument;
  $('#btn-doc-refresh').onclick = () => refreshDocuments();
  $('#btn-browser-open').onclick = openBrowser;
  $('#btn-browser-refresh').onclick = () => refreshBrowser();
  $('#btn-browser-enable').onclick = enableBrowserTools;
  $('#btn-browser-notes').onclick = () => listBrowserNotes();
  $('#btn-browser-notes-prev').onclick = () => listBrowserNotes(Math.max(0, browserUI.notesOffset - 20));
  $('#btn-browser-notes-next').onclick = () => listBrowserNotes(browserUI.notesNext);
  $('#browser-details').ontoggle = () => { if ($('#browser-details').open) refreshBrowser(); };
  $('#browser-url').onkeydown = event => { if (event.key === 'Enter') { event.preventDefault(); openBrowser(); } };
  $('#btn-agents-refresh').onclick = () => refreshAgents();
  $('#btn-agent-start').onclick = startAgent;
  $('#btn-agent-enable').onclick = async () => {
    if (await patchCfg({enabled_tools: [...new Set([...(S.cfg.enabled_tools || []), ...AGENT_WEB_TOOLS])]})) { renderTools(); setSendState(); }
  };
  $('#agent-details').ontoggle = () => { if ($('#agent-details').open) refreshAgents(); };
  $('#btn-doc-enable').onclick = async () => {
    try {
      if (await patchCfg({enabled_tools: [...new Set([...(S.cfg.enabled_tools || []), ...DOCUMENT_TOOL_NAMES])]})) {
        renderTools(); await refreshDocuments();
      }
    } catch (error) { $('#doc-status').textContent = String(error); }
  };
  $('#btn-panel').onclick = () => {
    $('#app').classList.toggle('panel-open');
    $('#btn-panel').classList.toggle('on');
  };
  $('#btn-addwd').onclick = pickDirectory;
  $('#btn-wd').onclick = pickDirectory;

  $$('.tab').forEach(t => t.onclick = () => {
    $$('.tab').forEach(x => x.classList.remove('active'));
    $$('.tabpane').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    $('#pane-' + t.dataset.tab).classList.add('active');
    if (t.dataset.tab === 'context' || t.dataset.tab === 'compaction') refreshContext();
  });

  const ta = $('#input');
  ta.addEventListener('input', () => { saveDraft(); autoGrow(); });
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });

  $('#f-temp').oninput = e => $('#f-temp-v').textContent = e.target.value;
  let _cpTimer = null;
  const deferCheck = () => { clearTimeout(_cpTimer); _cpTimer = setTimeout(checkProviders, 450); };
  $('#f-model').oninput = deferCheck;
  $('#f-summodel').oninput = deferCheck;
  $('#f-ctx').oninput = deferCheck;
  $('#f-provpolicy').onchange = checkProviders;
  $('#f-provider').onchange = switchProvider;
  for (const which of ['model', 'sum']) {
    $(`#f-${which}-pricing`).onchange = () => checkFlex(which);
    $(`#${which}-flex-refresh`).onclick = () => checkFlex(which, true);
  }
  api('/api/providers').then(d => {
    $('#prov-count').textContent =
      `${d.allowed.length} of ${d.total} OpenRouter providers qualify as EU/US.`;
  }).catch(() => {});
  $('#c-hot').oninput = e => {
    $('#c-hot-v').textContent = fmt(+e.target.value);
    $('#c-hot-state').textContent = `UNSAVED: ${fmt(+e.target.value)} tokens. Press Save engine config.`;
  };
  $('#c-sigt').oninput = e => $('#c-sigt-v').textContent = e.target.value;
  $('#c-trig').oninput = e => $('#c-trig-v').textContent = pct(+e.target.value);
  $('#c-target').oninput = explainTail;
  $('#c-docs').onchange = updateDocumentControls;

  $('#btn-savecfg').onclick = async () => {
    const saved = await patchCfg({
      model: $('#f-model').value.trim(),
      context_window: +$('#f-ctx').value,
      max_output_tokens: +$('#f-maxout').value,
      max_tool_iterations: +$('#f-maxiter').value,
      agent_max_parallel: +$('#f-agent-parallel').value,
      agent_max_depth: +$('#f-agent-depth').value,
      agent_timeout_seconds: +$('#f-agent-timeout').value,
      temperature: +$('#f-temp').value,
      system_prompt: $('#f-sys').value,
      reasoning_effort: $('#f-reasoning').value,
      provider_policy: $('#f-provpolicy').value,
      pricing_mode: $('#f-model-pricing').value,
      compaction: {
        summary_model: $('#f-summodel').value.trim(),
        summary_reasoning: $('#f-sumreasoning').value,
        summary_pricing_mode: $('#f-sum-pricing').value,
      },
    });
    if (!saved) return;
    fillSettings(); // Do not silently reset unsaved engine controls.
    $('#btn-savecfg').textContent = 'Saved ✓';
    setTimeout(() => $('#btn-savecfg').textContent = 'Save settings', 1200);
  };

  $('#btn-savecmp').onclick = async () => {
    if (!await patchCfg({ compaction: readCompaction() })) return;
    fillCompaction(); refreshContext();
    $('#btn-savecmp').textContent = 'Saved ✓';
    setTimeout(() => $('#btn-savecmp').textContent = 'Save engine config', 1200);
  };

  $('#btn-defaults').onclick = async () => {
    const id = S.chatId;
    if (!id || !S.cfg) return;
    await api(`/api/chats/${id}/defaults`, {method: 'POST'});
    if (S.chatId === id) alert('Saved settings, Engine and Tools are now the defaults for new chats. Unsaved controls were not copied.');
  };
  setSendState();
}

init().catch(e => {
  document.body.innerHTML = `<div class="empty"><h2>Failed to start</h2><p>${esc(String(e))}</p></div>`;
});
