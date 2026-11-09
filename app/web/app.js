/* CTXAgent UI (ctxagent backend).

   Integration points for other scripts (web/settings.js and friends):
   - window 'pi:chat-loaded'          fired after the current chat (and config) loads or switches
   - window 'pi:chat-config-changed'  {chatId}: re-fetch and re-render that chat's settings
   - window 'pi:keys-changed'         re-check /api/health (API key state, banners)
   - #open-settings click             falls back to the per-chat Model tab unless a later
                                      listener calls event.preventDefault() or window.PiSettings exists
   - window.PiApp                     small helper API (see the end of this file)
   Themes live in web/themes/ (window.PiTheme). */
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const fmt = n => n == null ? '—' : n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);
const pct = x => (x * 100).toFixed(0) + '%';
const icon = (name, cls = 'ic') => `<svg class="${cls}" aria-hidden="true" focusable="false"><use href="#i-${name}"></use></svg>`;
const iconButton = (cls, name, label) => {
  const b = el('button', cls); b.type = 'button'; b.innerHTML = icon(name);
  b.title = label; b.setAttribute('aria-label', label); return b;
};
const appName = () => ($('.brand-name')?.textContent || document.title || 'CTXAgent').trim();
// Browser storage is a convenience only: private windows and WebView profiles may refuse it.
const store = {
  get(key) { try { return localStorage.getItem(key); } catch (_) { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch (_) {} },
};
function errorText(error) {
  const text = String(error?.message || error || 'Unknown error');
  try {
    const data = JSON.parse(text);
    if (data && data.detail != null) return typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
  } catch (_) {}
  return text;
}

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
  chats: [], chatId: null, cfg: null, stats: null, usage: null, hasKey: true, chatFilter: '',
  expandedChats: new Set(), chatListSeq: 0,
  tools: [], segMap: new Map(), runs: new Map(), views: new Map(), drafts: new Map(),
  renderId: null, viewEpoch: 0, contextSeq: 0, delegated: false, agentsCost: null,
  // Messages waiting for the running turn's next step (per chat), and chats whose turn
  // runs on the server without this tab streaming it (opened while running).
  queued: new Map(), remote: new Set(),
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
// The friendly per-chat placeholder disappears as soon as anything is rendered.
function clearEmpty() { wrapEl()?.querySelector(':scope > .chat-empty')?.remove(); }

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
    clearEmpty();
    // One closed row for each contiguous run of non-message activity.
    const node = el('details', 'turn-activity');
    const summary = el('summary', '', 'Activity');
    const body = el('div', 'activity-body'); node.append(summary, body); host.appendChild(node);
    a = {node, summary, body, tools:0, notices:0, errors:0}; activityViews.set(host, a);
  }
  return a;
}

function activityLabel(a) {
  const parts = [];
  if (a.tools) parts.push(`${a.tools} tool call${a.tools === 1 ? '' : 's'}`);
  if (a.notices) parts.push(`${a.notices} event${a.notices === 1 ? '' : 's'}`);
  const running = !!a.body.querySelector('.tool .spin');
  const title = running ? 'Working' : 'Activity';
  const errors = a.errors ? `${a.errors} error${a.errors === 1 ? '' : 's'}` : '';
  a.summary.replaceChildren(el('span', 'act-title', title));
  if (parts.length) a.summary.append(el('span', 'act-meta', parts.join(' · ')));
  if (errors) a.summary.append(el('span', 'act-err', errors));
  a.summary.title = [title, ...parts, errors].filter(Boolean).join(' · ') + '. Click to show or hide details.';
  a.node.classList.toggle('has-errors', a.errors > 0);
  a.node.classList.toggle('running', running);
}

function addUser(text) {
  activityViews.delete(wrapEl());
  clearEmpty();
  const d = el('div', 'msg user');
  d.appendChild(el('div', 'role', 'You'));
  const b = el('div', 'bubble');
  if (text.startsWith('[scratchpad]')) renderScratchpadMessage(b, text); else b.textContent = text;
  d.appendChild(b);
  wrapEl().appendChild(d); scrollDown(true); return d;
}

function renderScratchpadMessage(bubble, text) {
  const details = el('details', 'sp-msg');
  const version = (text.match(/\(version (\d+)\)/) || [])[1];
  details.append(el('summary', null, 'Scratchpad shared with the agent' + (version ? ` (version ${version})` : '')));
  const inner = (text.match(/<scratchpad>\n([\s\S]*)\n<\/scratchpad>/) || [])[1] ?? text;
  details.append(el('pre', null, inner));
  bubble.append(details);
}

function addAssistant() {
  clearEmpty();
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
  const head = el('button', 'tool-head'); head.type = 'button';
  head.setAttribute('aria-expanded', 'true');
  const sp = el('span', 'spin');
  head.appendChild(sp);
  head.appendChild(el('span', 'nm', name));
  head.appendChild(el('span', 'arg', argPreview(args)));
  const st = el('span', 'st', '…'); head.appendChild(st);
  const body = el('div', 'tool-body'); // opening the activity row also reveals outputs
  head.onclick = () => head.setAttribute('aria-expanded', String(!body.classList.toggle('hidden')));
  card.appendChild(head); card.appendChild(body);
  const a = activity(); a.body.appendChild(card); a.tools++; activityLabel(a); scrollDown();
  return { card, head, spin: sp, st, body, group:a };
}

function finishToolCard(card, status) {
  card.spin.remove();
  card.st.textContent = status;
  activityLabel(card.group);
}

function markToolError(card) {
  if (card.failed) return;
  card.failed = true; card.st.classList.add('err');
  card.st.title = 'The tool reported an error';
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
  // Search keeps the hierarchy: a group stays visible (and open) while any
  // descendant matches; folding state is untouched and returns when cleared.
  const query = S.chatFilter.trim().toLowerCase(), matched = new Map();
  const matches = c => {
    if (!matched.has(c.id)) matched.set(c.id, (c.title || 'Untitled chat').toLowerCase().includes(query)
      || (children.get(c.id) || []).some(matches));
    return matched.get(c.id);
  };
  function append(c, host, depth) {
    if (query && !matches(c)) return;
    const kids = (children.get(c.id) || []).filter(k => !query || matches(k));
    const expanded = query ? kids.length > 0 : S.expandedChats.has(c.id);
    const group = el('div', 'chat-group');
    const d = el('div', 'chat-item' + (c.id === S.chatId ? ' active' : '') + (ancestors.has(c.id) ? ' contains-active' : ''));
    d.dataset.chatId = c.id;
    const title = c.title || 'Untitled chat';
    const button = el('button', 'chat-open'); button.type = 'button';
    button.title = title; button.setAttribute('aria-label', title);
    if (c.id === S.chatId) button.setAttribute('aria-current', 'page');
    const caret = el('span', 'chat-caret', kids.length ? (expanded ? '\u25be' : '\u25b8') : '');
    caret.setAttribute('aria-hidden', 'true'); button.appendChild(caret);
    if (S.runs.has(c.id)) { const dot = el('span', 'run-dot'); dot.setAttribute('aria-hidden', 'true'); button.appendChild(dot); }
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
        if (kids.length && !query) {
          if (S.expandedChats.has(c.id)) S.expandedChats.delete(c.id); else S.expandedChats.add(c.id);
          renderChats();
        }
        closeSidebarPeek();
        return; // Folding must not reload history, interrupt a run or lose a draft.
      }
      if (kids.length) S.expandedChats.add(c.id);
      openChat(c.id);
    };
    button.onkeydown = ev => {
      if (ev.key === 'F2') { ev.preventDefault(); renameChat(c.id); }
      else if (ev.key === 'Delete') { ev.preventDefault(); deleteChat(c); }
    };
    d.appendChild(button);
    const rename = iconButton('row-act ren', 'pencil', `Rename ${title}`);
    rename.title = 'Rename (F2)';
    rename.onclick = ev => { ev.stopPropagation(); renameChat(c.id); };
    const x = iconButton('row-act x', 'trash', `Delete ${title}`);
    x.title = 'Delete (Del)';
    x.onclick = ev => { ev.stopPropagation(); deleteChat(c); };
    const actions = el('span', 'row-actions'); actions.append(rename, x);
    d.append(actions); group.appendChild(d);
    if (kids.length) {
      const nested = el('div', 'chat-children'); nested.id = `chat-children-${c.id}`;
      nested.hidden = !expanded;
      for (const child of kids) append(child, nested, depth + 1);
      group.appendChild(nested);
    }
    host.appendChild(group);
  }
  for (const c of roots) append(c, list, 0);
  if (!list.children.length) {
    const empty = el('div', 'chatlist-empty');
    if (query) empty.textContent = `No chats match \u201c${S.chatFilter.trim()}\u201d.`;
    else {
      empty.append(el('div', '', 'No chats yet.'));
      const start = el('button', 'btn', 'Start a chat'); start.type = 'button'; start.onclick = () => newChat();
      empty.append(start);
    }
    list.appendChild(empty);
  }
  if (focused) [...list.querySelectorAll('.chat-open')].find(b => b.closest('[data-chat-id]').dataset.chatId === focused)?.focus({preventScroll: true});
}

async function deleteChat(c) {
  const title = c.title || 'Untitled chat';
  const ok = await confirmModal({
    title: 'Delete chat', confirmLabel: 'Delete', danger: true,
    message: `Delete \u201c${title}\u201d? Its history and settings are removed from this computer. This cannot be undone.`,
  });
  if (!ok) return;
  try {
    await api(`/api/chats/${c.id}`, { method: 'DELETE' });
  } catch (error) {
    toast(`Could not delete \u201c${title}\u201d: ${errorText(error)}`, 'err');
    return;
  }
  S.views.delete(c.id); S.drafts.delete(c.id); S.expandedChats.delete(c.id);
  if (S.chatId === c.id) {
    S.chatId = null; S.cfg = null; S.viewEpoch++; clearMessages(); resetBrowser();
    $('#input').value = ''; autoGrow(); setSendState();
    $('#chattitle').textContent = 'No chat selected';
    $('#modelpill').textContent = '\u2014';
    renderEmptyChat();
  }
  await loadChats();
}

function renameChat(id = S.chatId) {
  if (!id) return;
  const row = S.chats.find(c => c.id === id);
  const current = row?.title || (id === S.chatId ? $('#chattitle').textContent : '');
  modal('Rename chat', bd => {
    const field = el('div', 'field'), label = el('label', '', 'Title'), input = el('input');
    label.htmlFor = 'rename-input'; input.type = 'text'; input.id = 'rename-input';
    input.value = current; input.maxLength = 200; input.spellcheck = false;
    input.addEventListener('keydown', ev => {
      if (ev.key === 'Enter') { ev.preventDefault(); input.closest('.modal')?.querySelector('.ft .btn.primary')?.click(); }
    });
    field.append(label, input); bd.appendChild(field);
    requestAnimationFrame(() => input.select());
  }, [{label: 'Rename', primary: true, fn: async () => {
    const title = $('#rename-input').value.trim();
    if (!title) { $('#rename-input').focus(); return false; }
    try {
      await api(`/api/chats/${id}/title`, {method: 'PATCH', body: {title}});
    } catch (error) {
      toast(`Rename failed: ${errorText(error)}`, 'err'); return false;
    }
    const chat = S.chats.find(c => c.id === id); if (chat) chat.title = title;
    if (S.chatId === id) $('#chattitle').textContent = title;
    renderChats(); loadChats();
  }}], {size: 'small', cancelLabel: 'Cancel'});
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
  closeSidebarPeek();
  saveDraft();
  const epoch = ++S.viewEpoch;
  const hadRun = S.runs.has(id);
  S.chatId = id; S.cfg = null; S.stats = null; S.usage = null; S.agentsCost = null;
  renderChats();
  resetBrowser();
  $('#doc-list').replaceChildren(); $('#doc-status').textContent = '';
  renderCanvas(null);
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
    if (data.running && !S.runs.has(id)) S.remote.add(id); else S.remote.delete(id);
    loadScratchpad(id);
    loadCanvas(id);
    if (!S.runs.has(id)) {
      const left = data.queued || [];
      if (data.running) S.queued.set(id, left.map(q => ({ ...q })));
      else {
        // Queued for a turn that ended while no tab streamed it: back into the box.
        S.queued.set(id, []);
        restoreToComposer(left.filter(q => q.kind !== 'scratchpad').map(q => q.text));
        for (const q of left) api(`/api/chats/${id}/queue/${q.id}`, { method: 'DELETE' }).catch(() => {});
      }
    }
    renderQueue();
    if (!hadRun && !S.runs.has(id)) {
      clearMessages(); renderSegments(data.segments || []);
    }
    $('#empty')?.remove();
    fillSettings(); fillCompaction(); renderTools(); renderWorkdirs(); showConfigPending(!!data.config_pending);
    applyStats(data.stats); S.usage = data.usage; applyUsage();
    if (!view.children.length) renderEmptyChat();
    setSendState(); scrollDown(true); $('#input').focus();
    // Contract for settings.js and other listeners: the current chat and its config.
    window.PiCurrentChat = data;
    window.dispatchEvent(new CustomEvent('pi:chat-loaded', {detail: data}));
    if (activePanelTab() === 'agents') refreshAgents(id);
    await refreshContext(id);
    if (S.chatId === id && S.viewEpoch === epoch) loadChats();
  } catch (error) {
    if (S.chatId === id && S.viewEpoch === epoch) addNotice(esc(String(error)), 'err');
  }
}

// Re-read one chat's saved settings without touching its transcript, e.g. after
// another component (the app Settings modal) changed them.
async function reloadChatConfig(id = S.chatId) {
  if (!id || S.chatId !== id) { loadChats(); return; }
  const epoch = S.viewEpoch;
  try {
    const data = await api(`/api/chats/${id}`);
    if (S.chatId !== id || S.viewEpoch !== epoch) return;
    S.cfg = data.config; S.delegated = !!data.delegated;
    $('#chattitle').textContent = data.title;
    fillSettings(); fillCompaction(); renderTools(); renderWorkdirs(); showConfigPending(!!data.config_pending);
    showSystemPrompt((data.segments || []).find(s => s.kind === 'system')?.body);
    if (data.stats) applyStats(data.stats);
    setSendState(); refreshEmptyChat(); loadChats();
  } catch (error) {
    if (S.chatId === id && S.viewEpoch === epoch) toast(`Could not reload chat settings: ${errorText(error)}`, 'err');
  }
}

function renderEmptyChat() {
  const host = wrapEl();
  if (!host || host.querySelector(':scope > .chat-empty')) return;
  const box = el('div', 'empty chat-empty');
  const mark = el('div', 'empty-mark'); mark.setAttribute('aria-hidden', 'true');
  mark.innerHTML = '<svg viewBox="0 0 24 24"><use href="#i-ctx"/></svg>';
  box.appendChild(mark);
  if (!S.chatId) {
    box.append(el('h2', '', 'No chat selected'), el('p', '', 'Pick a chat on the left, or start a new one.'));
    const row = el('div', 'btnrow'), start = el('button', 'btn primary', 'New chat');
    start.type = 'button'; start.onclick = () => newChat(); row.append(start); box.append(row);
    host.appendChild(box); return;
  }
  if (S.delegated) {
    box.append(el('h2', '', 'Worker-owned chat'),
      el('p', '', 'A delegated agent owns this chat. Inspect it here; send follow-ups from the parent chat’s Agents tab.'));
    host.appendChild(box); return;
  }
  box.append(el('h2', '', 'What should we work on?'),
    el('p', '', 'Ask a question, or give the agent a project folder and the tools it may use. The details panel shows what the compaction engine does to the context, turn by turn.'));
  if (!S.hasKey) {
    const callout = el('div', 'key-callout');
    callout.innerHTML = icon('key');
    callout.append(el('b', '', 'Connect a model provider first'),
      el('p', '', `${appName()} needs an OpenRouter or Anthropic API key before it can answer. Add one in Settings > API keys.`));
    const row = el('div', 'btnrow'), open = el('button', 'btn primary', 'Open Settings');
    open.type = 'button'; open.onclick = openAppSettings; row.append(open); callout.append(row);
    box.append(callout);
  }
  const starters = el('div', 'starter-grid');
  const starter = (name, title, detail, action) => {
    const b = el('button', 'starter'); b.type = 'button'; b.innerHTML = icon(name);
    b.append(el('b', '', title), el('span', '', detail)); b.onclick = action; starters.append(b);
  };
  const wds = S.cfg?.workdirs || [];
  starter('folder', wds.length ? 'Working directory' : 'Set a working directory',
    wds.length ? wds[0] : 'File and shell tools stay inside it.', pickDirectory);
  starter('toolbox', 'Choose tools', `${S.tools.filter(t => (S.cfg?.enabled_tools || []).includes(t.name)).length} of ${S.tools.length} tools enabled`, () => showPanelTab('tools'));
  starter('sliders', 'Model & pricing', S.cfg?.model || 'Pick a model for this chat', () => showPanelTab('settings'));
  box.append(starters);
  host.appendChild(box);
}

function refreshEmptyChat() {
  const current = $('#msgwrap > .chat-empty');
  if (!current) return;
  current.remove(); renderEmptyChat();
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
  const allCards = [];
  for (const s of segs) {
    if (s.kind === 'system') {
      addNotice(`<b>System instructions</b><div>${md(s.body || '')}</div>`, 'system-prompt'); continue;
    }
    if (s.kind === 'media') { addMedia(s); continue; }
    if (s.kind === 'tool_change') continue;  // deferred tools surfaced/removed (direct Anthropic API)
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
        pendingCards.set(tc.id, card); allCards.push(card);
      }
      continue;
    }
    if (s.kind === 'tool_result') {
      const card = pendingCards.get(s.tool_call_id);
      if (card) {
        card.body.textContent = s.body;
        finishToolCard(card, `${fmt(s.original_tokens)} tok`);
        pendingCards.delete(s.tool_call_id);
      } else {
        // Compaction/imports can leave results without their original call.
        const orphan = addToolCard('Tool result', null);
        orphan.group.tools--; orphan.group.notices++;
        orphan.body.textContent = s.body;
        finishToolCard(orphan, `${fmt(s.original_tokens)} tok`);
      }
      continue;
    }
    addNotice(`<b>${esc(s.kind || 'Event')}</b><pre>${esc(s.body || '')}</pre>`);
  }
  // Stored history is never still running: a call without a stored result was
  // interrupted (or its result was compacted away).
  for (const card of allCards) if (card.spin.isConnected) finishToolCard(card, 'no result');
}

/* -------------------------------------------------------------- sending */
/* ------------------------------------------------------ queued messages */
// A message sent while a turn runs waits on the server (AgentSession.queue) and goes
// in before the turn's next request; a chip above the composer shows it until then.
function queueFor(id) { if (!S.queued.has(id)) S.queued.set(id, []); return S.queued.get(id); }

function renderQueue() {
  const host = $('#queue-chips'); if (!host) return;
  const items = S.chatId ? (S.queued.get(S.chatId) || []) : [];
  host.replaceChildren(...items.map(item => {
    const chip = el('span', 'queue-chip');
    const label = item.kind === 'scratchpad' ? 'Scratchpad'
      : (item.text.length > 70 ? item.text.slice(0, 70) + '…' : item.text);
    chip.title = (item.later ? 'Sent when this turn ends: ' : 'Sent with the next step: ') + (item.text || 'the scratchpad');
    chip.append(el('span', 'q-label', label));
    if (item.id) {
      const x = el('button', 'q-x', '×'); x.type = 'button';
      x.title = 'Take back into the message box'; x.setAttribute('aria-label', 'Remove queued message');
      x.onclick = () => unqueueMessage(item.id);
      chip.append(x);
    }
    return chip;
  }));
  host.hidden = !items.length;
}

function restoreToComposer(texts) {
  texts = texts.filter(Boolean);
  if (!texts.length) return;
  const ta = $('#input');
  ta.value = [...texts, ta.value].filter(Boolean).join('\n\n'); autoGrow(); saveDraft();
}

async function queueMessage(id, text, scratchpad = false) {
  try {
    const r = await api(`/api/chats/${id}/queue`, { method: 'POST', body: { message: text, scratchpad } });
    queueFor(id).push(r.queued);
  } catch (error) {
    if (!String(error).includes('No turn is running')) throw error;
    S.remote.delete(id);
    if (S.runs.has(id)) {
      // This tab's turn is just ending: send it as the next message when it has.
      queueFor(id).push({ id: null, text, kind: scratchpad ? 'scratchpad' : 'message', later: true });
    } else if (S.chatId === id) {
      if (scratchpad) sendScratchpad(); else send(text);
      return;
    }
  }
  if (S.chatId === id) renderQueue();
}

async function unqueueMessage(qid) {
  const id = S.chatId;
  const item = queueFor(id).find(q => q.id === qid);
  try { await api(`/api/chats/${id}/queue/${qid}`, { method: 'DELETE' }); }
  catch (_) { return; }  // already delivered
  S.queued.set(id, queueFor(id).filter(q => q.id !== qid));
  if (item && item.kind !== 'scratchpad') restoreToComposer([item.text]);
  renderQueue();
}

// Enter: send, or queue for the next step while a turn runs. The button still stops.
function submitComposer() {
  const id = S.chatId;
  if (!id || S.delegated || !S.cfg) return;
  if (!S.streaming && !S.remote.has(id)) return send();
  const ta = $('#input'); const text = ta.value.trim();
  if (!text) return;
  ta.value = ''; S.drafts.delete(id); autoGrow();
  queueMessage(id, text).catch(error => {
    restoreToComposer([text]);
    addNotice(`<b>Not queued</b> ${esc(String(error))}`, 'err');
  });
}

/* ------------------------------------------------------------ scratchpad */
// One short text per chat that the user and the agent both write (ctxagent/scratchpad.py).
// It is not in the context: Ctrl+Enter hands it to the agent, at once or with the
// running turn's next step. Saves carry the version they started from; the server
// re-applies the agent's edits made since then.
const pad = { chat: null, version: 0, saved: '', timer: null };

function padStatus(text) { const s = $('#sp-status'); if (s) { s.textContent = text; s.title = text; } }

async function saveScratchpad() {
  clearTimeout(pad.timer);
  const id = pad.chat, ta = $('#sp-text');
  if (!id || !ta || ta.value === pad.saved) return;
  const text = ta.value;
  try {
    const d = await api(`/api/chats/${id}/scratchpad`, { method: 'PUT', body: { text, base_version: pad.version } });
    if (pad.chat !== id) return;
    pad.version = d.version; pad.saved = d.text;
    if (d.text !== text && ta.value === text) ta.value = d.text;  // the agent's edits merged in
    padStatus(d.dropped_agent_edits ? `saved; ${d.dropped_agent_edits} agent edit(s) overridden by yours` : 'saved');
  } catch (error) { padStatus('not saved: ' + errorText(error)); }
}

async function loadScratchpad(id) {
  const ta = $('#sp-text'); if (!ta) return;
  if (pad.chat && pad.chat !== id) await saveScratchpad();  // the previous chat's last keystrokes
  pad.chat = id || null; pad.version = 0; pad.saved = ''; ta.value = '';
  ta.disabled = !id; $('#sp-send').disabled = !id;
  padStatus('');
  if (!id) return;
  try {
    const d = await api(`/api/chats/${id}/scratchpad`);
    if (pad.chat !== id) return;
    pad.version = d.version; pad.saved = d.text; ta.value = d.text;
  } catch (error) { padStatus('unavailable: ' + errorText(error)); }
}

// After scratchpad_write: show the agent's version; typing in progress is saved first and
// rebased onto it.
async function refreshScratchpad(id) {
  if (pad.chat !== id) return;
  const ta = $('#sp-text');
  if (ta.value !== pad.saved) await saveScratchpad();
  else {
    try {
      const d = await api(`/api/chats/${id}/scratchpad`);
      if (pad.chat !== id || ta.value !== pad.saved) return;
      pad.version = d.version; pad.saved = d.text; ta.value = d.text;
    } catch (_) { return; }
  }
  padStatus('the agent wrote here');
  const box = $('#scratchpad'); box.classList.add('agent-wrote');
  setTimeout(() => box.classList.remove('agent-wrote'), 1500);
}

async function sendScratchpad() {
  const id = S.chatId;
  if (!id || S.delegated || !S.cfg) return;
  await saveScratchpad();
  if (S.streaming || S.remote.has(id)) {
    queueMessage(id, '', true).then(() => padStatus('goes to the agent with its next step'))
      .catch(error => padStatus('not sent: ' + errorText(error)));
    return;
  }
  padStatus('sent to the agent');
  send(undefined, { scratchpad: true });
}

// After a turn: what it handed back. A normal end sends it as the next message; after
// Stop or an error it goes back into the message box.
function afterRun(id, returned) {
  const later = queueFor(id).filter(q => q.later);
  S.queued.set(id, []);
  if (S.chatId === id) renderQueue();
  const back = [...(returned?.messages || []), ...later];
  if (!back.length || S.chatId !== id) return;
  const texts = back.filter(q => q.kind !== 'scratchpad').map(q => q.text);
  const pad = back.some(q => q.kind === 'scratchpad');
  if (returned && returned.reason !== 'ended') { restoreToComposer(texts); return; }
  if (texts.length) {
    send(texts.join('\n\n'));
    if (pad) queueMessage(id, '', true).catch(() => {});
  } else if (pad) sendScratchpad();
}

async function send(override, { scratchpad = false } = {}) {
  const id = S.chatId;
  if (S.streaming && override === undefined && !scratchpad) { await api(`/api/chats/${id}/stop`, { method: 'POST' }); return; }
  if (S.delegated || S.streaming) return;
  const ta = $('#input'); const text = scratchpad ? '' : (override ?? ta.value).trim();
  if ((!text && !scratchpad) || !id || !S.cfg) return;
  if (!scratchpad && override === undefined) { ta.value = ''; S.drafts.delete(id); autoGrow(); }
  if (!scratchpad) addUser(text);  // a scratchpad message comes back as a user_message event
  $('#empty')?.remove();

  const ctrl = new AbortController();
  S.runs.set(id, { ctrl, kind: 'send' }); setSendState(); loadChats();
  let cur = null, returned = null;
  const cards = new Map();

  try {
    await sse(`/api/chats/${id}/send`, scratchpad ? { scratchpad: true } : { message: text }, ev => inChat(id, () => {
      switch (ev.type) {
        case 'assistant_start': cur = null; break;
        case 'user_message':
          if (ev.queued_id) S.queued.set(id, queueFor(id).filter(q => q.id !== ev.queued_id));
          if (S.chatId === id) renderQueue();
          cur = null; addUser(ev.segment.body); break;
        case 'queue_returned': returned = ev; break;
        case 'delta':
          if (!ev.text) break;
          if (!cur) cur = addAssistant();
          cur.raw += ev.text; cur.bubble.innerHTML = md(cur.raw); scrollDown();
          break;
        case 'media': addMedia(ev.segment); break;
        case 'canvas': if (S.chatId === id) canvasChanged(ev.canvas); break;
        case 'config_applied':
          if (S.chatId === id) {
            if (!SAVE.settings.dirty && !SAVE.settings.busy && !SAVE.engine.dirty && !SAVE.engine.busy) S.cfg = ev.config;
            showSystemPrompt(ev.system);
            for (const which of Object.keys(SAVE)) {
              if ($(SAVE[which].state).classList.contains('pending')) saveState(which, 'Saved · in effect', 'good');
            }
          }
          break;
        case 'tool_start': {
          cur = null; // assistant text stays visible; commands start a new activity block
          cards.set(ev.id, addToolCard(ev.name, ev.args));
          break;
        }
        case 'tool_result': {
          if (ev.name?.startsWith('agent_') && S.chatId === id) refreshAgents(id);
          if (ev.name === 'scratchpad_write' && ev.ok) refreshScratchpad(id);
          const c = cards.get(ev.id);
          if (c) {
            c.body.textContent = ev.segment.body;
            finishToolCard(c, `${ev.ms} ms · ${fmt(ev.tokens)} tok`);
            if (!ev.ok) markToolError(c);
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
    afterRun(id, returned);
  }
}

function setSendState() {
  const b = $('#btn-send');
  b.innerHTML = icon(S.streaming ? 'stop' : 'send');
  b.classList.toggle('stop', S.streaming);
  b.title = S.streaming ? 'Stop' : 'Send';
  b.setAttribute('aria-label', b.title);
  const hint = $('.composer-hint');
  if (hint) hint.innerHTML = (S.streaming || S.remote.has(S.chatId))
    ? '<kbd>Enter</kbd> queue for the next step · <kbd>Shift</kbd>+<kbd>Enter</kbd> new line'
    : '<kbd>Enter</kbd> send · <kbd>Shift</kbd>+<kbd>Enter</kbd> new line';
  b.disabled = !S.chatId || !S.cfg || S.delegated;
  $('#btn-rename').disabled = !S.chatId;
  $$('#toollist input[type=checkbox]').forEach(cb => cb.disabled = !S.cfg || S.delegated || S.streaming);
  $('#btn-compact').disabled = !S.cfg || S.streaming || S.delegated;
  for (const id of ['btn-doc-open', 'btn-doc-refresh', 'btn-doc-enable']) $('#' + id).disabled = !S.cfg || S.streaming || S.delegated;
  $$('#doc-list button').forEach(b => b.disabled = !S.cfg || S.streaming);
  $('#btn-canvas-close').disabled = !S.cfg || S.streaming || S.delegated;
  $('#input').disabled = !S.cfg || S.delegated;
  $$('#pane-settings input, #pane-settings select, #pane-settings textarea, #pane-compaction input, #pane-compaction select').forEach(e => e.disabled = !S.cfg || S.delegated);
  updateSelfSteerControls(); updateBrowserControls();
  $('#input').placeholder = S.delegated ? 'Worker-owned chat: inspect here; send follow-ups from the parent Agents panel.' : (S.chatId ? 'Ask anything…' : 'Select or start a chat first');
  $('#btn-agent-start').disabled = !S.cfg || S.streaming || S.delegated;
  $('#btn-agent-enable').disabled = !S.cfg || S.streaming || S.delegated;
  $('#btn-agent-enable').hidden = AGENT_WEB_TOOLS.every(n => S.cfg?.enabled_tools?.includes(n));
  $('#btn-defaults').disabled = !S.cfg || S.delegated;

}

/* ---------------------------------------------------------- context pane */
function applyStats(st) {
  if (!st) return;
  S.stats = st;
  if (st.agents_cost) { S.agentsCost = st.agents_cost; renderAgentsCost(); }
  // Provider tokens throughout (the local estimate x the measured calibration), against the
  // model's real window. The limit is where deep compaction runs (its trigger share of the
  // window, at most the window minus the reserved output and headroom).
  const cal = st.token_calibration || 1;
  const win = st.window_tokens || Math.round((st.effective_window || 1) * cal);
  const used = st.used_tokens ?? Math.round((st.total || 0) * cal);
  const limit = Math.min(win, st.deep_limit_tokens || win);
  const fill = used / Math.max(1, limit), chip = $('#ctx-meter');
  const limitName = st.deep_active === false ? 'prompt limit' : 'deep compaction';
  $('#meter').innerHTML = `<span class="seg-used" style="width:${Math.min(100, 100 * fill)}%"></span>`;
  $('#tokpill').textContent = `${fmt(used)} / ${fmt(limit)}`;
  chip.classList.toggle('warn', fill > 0.8 && fill <= 0.95);
  chip.classList.toggle('bad', fill > 0.95);
  chip.title = `Context: ${fmt(used)} tokens; ${limitName} at ${fmt(limit)} (${pct(limit / win)} of the ${fmt(win)} window). Opens the Context tab.`;
  $('#savedpill').textContent = `saved ${fmt(st.saved_tokens ?? st.saved)}`;

  const bb = $('#bigbar');
  if (bb) {
    const u = Math.min(100, 100 * used / win), l = Math.max(u, Math.min(100, 100 * limit / win));
    const free = l - u, over = 100 - l;
    bb.innerHTML =
      `<div class="used" style="width:${u}%">${u > 12 ? fmt(used) : ''}</div>` +
      `<div class="free" style="width:${free}%">${free > 14 ? fmt(Math.max(0, limit - used)) + ' free' : ''}</div>` +
      `<div class="over" style="width:${over}%" title="Above the ${limitName} limit"></div>`;
    $('#s-total').textContent = `${fmt(used)} (${pct(fill)} of the limit)`;
    const lim = $('#s-limit');
    lim.textContent = `${fmt(limit)} (${pct(limit / win)} of the window)`;
    lim.className = 'v ' + (fill > 0.95 ? 'bad' : fill > 0.8 ? 'warn' : 'good');
    $('#s-limit-k').textContent = st.deep_active === false ? 'Prompt limit' : 'Deep compaction at';
    $('#s-window').textContent = fmt(win);
    $('#s-orig').textContent = fmt(st.original_tokens ?? st.original);
    $('#s-saved').textContent = `${fmt(st.saved_tokens ?? st.saved)} (${st.original ? pct(st.saved / st.original) : '0%'})`;
    $('#s-history').textContent = fmt(st.history_tokens ?? Math.round((st.frozen_tokens + st.hot_tokens) * cal));
    $('#s-live').textContent = fmt(st.live_tokens ?? 0);
    const reserve = st.usable_tokens != null ? win - st.usable_tokens : null;
    $('#s-ctx-note').textContent = (cal === 1
      ? 'Counted locally until the first response reports the real prompt size. '
      : `Provider tokens: the local count × ${cal.toFixed(2)}, measured on the last requests. `) +
      (reserve != null ? `${fmt(reserve)} of the window stay free for the answer and headroom, so a prompt never exceeds ${fmt(st.usable_tokens)}.` : '');
    $('#s-tools').textContent = fmt(st.tool_tokens ?? st.tool_overhead);
    $('#s-doc-tools').textContent = fmt(st.document_catalog_tokens || 0);
    $('#s-cadence').textContent = `${st.rewrite_pending_batches ?? 0} / ${st.rewrite_interval_batches ?? '?'} batches`;
    $('#s-tail').textContent = `${pct(st.deep_target_ratio)}${st.deep_ready ? '' : ' (cooling)'}`;
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
  const costpill = $('#costpill');
  costpill.hidden = u.cost == null;
  costpill.textContent = u.cost == null ? '' : '$' + Number(u.cost).toFixed(Number(u.cost) >= 10 ? 2 : 4);
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
    const row = el('div', 'item-card');
    row.appendChild(el('div', 'item-title', view.path));
    row.appendChild(el('p', 'usage-note', `${view.select ? view.select.join(', ') : 'Full text'} · ${fmt(view.tokens)} tok · ${view.status} · ${(view.version || '').slice(0, 12)}`));
    if (view.error) row.appendChild(el('p', 'usage-note err', view.error));
    const actions = el('div', 'btnrow');
    const b = el('button', 'btn sm', 'Close'); b.type = 'button'; b.disabled = S.streaming || S.delegated;
    b.onclick = () => closeDocument(view.id);
    actions.appendChild(b); row.appendChild(actions); host.appendChild(row);
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
    host.appendChild(el('p', '', 'The live content leaves the context; historical reads remain.'));
    const field = el('div', 'field'), label = el('label', '', 'Handoff note (optional)');
    const note = el('textarea'); note.id = 'doc-close-note'; note.rows = 4; label.htmlFor = note.id;
    field.append(label, note); host.appendChild(field);
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
  const button = el('button', 'btn sm', label), id = S.chatId, epoch = S.viewEpoch;
  button.type = 'button';
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
  $('#browser-budget').textContent = `${views.length} tabs · ${fmt(workspace?.tokens || 0)} / ${fmt(workspace?.max_tokens || 24000)} tok`;
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
    const actions = el('div', 'browser-actions btnrow');
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

/* ---------------------------------------------------------- canvas pane */
// The agent's live canvas: the current version, a focus outline (the region the agent
// zoomed its own copy to) and the version history. Older versions can be viewed here;
// the agent's undo decides which one is current.
const canvasState = { listing: null, chat: null, viewing: null };

function canvasUrl(aid, chat = canvasState.chat || S.chatId) {
  return `/api/chats/${encodeURIComponent(chat)}/artifacts/${encodeURIComponent(aid)}`;
}

async function loadCanvas(id = S.chatId) {
  if (!id) return;
  try {
    const listing = await api(`/api/chats/${encodeURIComponent(id)}/canvas`);
    if (S.chatId === id) { canvasState.chat = id; renderCanvas(listing); }
  } catch (e) { if (S.chatId === id) renderCanvas(null); }
}

function canvasChanged(listing) {
  const wasOpen = !!canvasState.listing?.open;
  canvasState.chat = S.chatId; canvasState.viewing = null;
  renderCanvas(listing);
  const active = $('#app').classList.contains('panel-open') && $('#tab-canvas').classList.contains('active');
  if (listing?.open && !wasOpen) showPanelTab('canvas', { quiet: true });
  else if (!active) $('#tab-canvas').classList.add('has-news');
}

function renderCanvas(listing) {
  canvasState.listing = listing;
  const open = !!listing?.open;
  $('#canvas-empty').hidden = open; $('#canvas-live').hidden = !open;
  if (!open) {
    canvasState.viewing = null;
    $('#canvas-history').replaceChildren(); $('#canvas-img').removeAttribute('src');
    $('#tab-canvas').classList.remove('has-news');
    return;
  }
  const versions = listing.versions || [];
  const current = listing.current;
  const shown = versions.find(v => v.version === canvasState.viewing) || current;
  if (shown === current) canvasState.viewing = null;
  const img = $('#canvas-img');
  img.src = canvasUrl(shown.id); img.alt = shown.label || `Canvas version ${shown.version}`;
  const dl = $('#canvas-download');
  dl.href = img.src; dl.setAttribute('download', `canvas-v${shown.version}.png`);
  // The focus outline belongs to the current version, in canvas pixels.
  const focus = $('#canvas-focus');
  const f = shown === current ? listing.focus : null;
  focus.hidden = !f;
  if (f) {
    const pct = (a, total) => `${(100 * a / total).toFixed(3)}%`;
    Object.assign(focus.style, { left: pct(f[0], shown.width), top: pct(f[1], shown.height),
      width: pct(f[2] - f[0], shown.width), height: pct(f[3] - f[1], shown.height) });
  }
  const parts = [`Version ${shown.version}${shown === current ? ' (current)' : ''}`, `${shown.width}×${shown.height} px`];
  if (shown.label) parts.splice(1, 0, shown.label);
  if (f) parts.push(`agent zoomed to x ${f[0]}–${f[2]}, y ${f[1]}–${f[3]}`);
  if (listing.source) parts.push(`from ${listing.source}`);
  $('#canvas-meta').textContent = parts.join(' · ');
  $('#btn-canvas-current').hidden = shown === current;
  $('#canvas-count').textContent = String(versions.length);
  const host = $('#canvas-history'); host.replaceChildren();
  for (const v of versions.slice().reverse()) {
    const b = el('button'); b.type = 'button';
    b.classList.toggle('head', v.version === current.version);
    b.classList.toggle('viewing', v === shown);
    b.title = [`Version ${v.version}`, v.label, v.summary].filter(Boolean).join(' · ');
    const thumb = el('img'); thumb.src = canvasUrl(v.id); thumb.alt = ''; thumb.loading = 'lazy';
    b.append(thumb, el('span', 'lbl', `v${v.version}${v.label ? ' · ' + v.label : ''}`));
    b.onclick = () => { canvasState.viewing = v.version; renderCanvas(canvasState.listing); };
    host.append(b);
  }
}

function initCanvas() {
  $('#canvas-frame').onclick = () => {
    const src = $('#canvas-img').src;
    if (!src) return;
    modal($('#canvas-meta').textContent || 'Canvas', bd => {
      const img = el('img'); img.src = src; img.alt = $('#canvas-img').alt; bd.append(img);
    }, [], { size: 'canvas-zoom' });
  };
  $('#btn-canvas-current').onclick = () => { canvasState.viewing = null; renderCanvas(canvasState.listing); };
  $('#btn-canvas-close').onclick = () => {
    const id = S.chatId;
    modal('Close the canvas?', bd => bd.append(el('p', '',
      'It leaves the agent\'s context and this panel, and its versions are discarded. Saved files stay.')),
    [{ label: 'Close canvas', danger: true, fn: async () => {
      const listing = await api(`/api/chats/${encodeURIComponent(id)}/canvas/close`, { method: 'POST' });
      if (S.chatId === id) { renderCanvas(listing); refreshContext(id); }
    } }], { size: 'small', cancelLabel: 'Keep it' });
  };
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
    if (!agents.length) host.appendChild(el('p', 'usage-note', 'No delegated workers yet.'));
    for (const a of agents) {
      const row = el('div', 'item-card agent-card'), head = el('div', 'item-head');
      head.append(el('span', 'state-badge ' + a.state, a.state), el('span', 'agent-task', a.task));
      row.appendChild(head);
      row.appendChild(el('p', 'usage-note', `${a.read_only ? 'read-only' : 'write-capable'} · ${a.usage?.requests || 0} requests · child ${a.child}`));
      if (a.error) row.appendChild(el('p', 'usage-note err', a.error));
      const actions = el('div', 'btnrow'), button = label => { const b = el('button', 'btn sm', label); b.type = 'button'; actions.appendChild(b); return b; };
      button('Inspect chat').onclick = () => openChat(a.child);
      button('Read report').onclick = () => showAgentReport(id, a.id);
      if (!['succeeded','failed','cancelled','timed_out','lost'].includes(a.state)) {
        button('Cancel').onclick = async () => {
          const ok = await confirmModal({title: 'Cancel worker', confirmLabel: 'Cancel worker', danger: true,
            message: 'Cancel this worker and its descendants? Edits it already made are not undone.'});
          if (!ok) return;
          try { await api(`/api/chats/${id}/agents/${a.id}/cancel`, {method:'POST'}); await refreshAgents(id); }
          catch (error) { if (S.chatId === id) $('#agent-status').textContent = errorText(error); }
        };
      } else if (a.state !== 'lost') {
        button('Follow up').onclick = () => followUpAgent(id, a);
      }
      row.appendChild(actions);
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

function followUpAgent(id, agentRow) {
  const epoch = S.viewEpoch;
  modal('Follow-up task', host => {
    host.appendChild(el('p', 'usage-note', 'The worker keeps its own context, live views and settings. Model calls incur costs.'));
    const field = el('div', 'field'), label = el('label', '', 'Message'), text = el('textarea');
    text.id = 'follow-up-message'; text.rows = 4; label.htmlFor = text.id;
    field.append(label, text); host.appendChild(field);
  }, [{label: 'Send follow-up', primary: true, fn: async () => {
    const message = $('#follow-up-message').value.trim();
    if (!message) { $('#follow-up-message').focus(); return false; }
    if (S.chatId !== id || S.viewEpoch !== epoch) return;
    api(`/api/chats/${id}/agents/${agentRow.id}/send`, {method:'POST', body:{message}})
      .then(() => refreshAgents(id))
      .catch(e => { if (S.chatId === id) $('#agent-status').textContent = errorText(e); });
  }}], {cancelLabel: 'Cancel'});
}

function startAgent() {
  const id = S.chatId, epoch = S.viewEpoch;
  modal('Delegate task', host => {
    host.appendChild(el('p','usage-note','Child inherits saved settings and live view specs. Only this explicit handoff is copied, not the full history. Model calls incur costs.'));
    for (const [key, label, rows] of [['task', 'Task', 3], ['context', 'Context / success criteria', 3], ['overrides', 'Settings overrides (JSON, optional)', 2]]) {
      const field = el('div', 'field'), l = el('label', '', label), t = el('textarea');
      t.id = 'delegate-' + key; t.rows = rows; l.htmlFor = t.id;
      if (key === 'overrides') { t.value = '{}'; t.spellcheck = false; }
      field.append(l, t); host.appendChild(field);
    }
    const label = el('label', 'switchrow'), check = el('input'); check.type = 'checkbox'; check.id = 'delegate-readonly'; check.checked = true;
    const text = el('span', 'lbl', 'Enforce read-only tools'); text.append(el('small', '', 'Recommended for reviews and research.'));
    label.append(check, text); host.appendChild(label);
  }, [{label:'Start worker',primary:true,fn:async () => {
    if (S.chatId !== id || S.viewEpoch !== epoch) return;
    try {
      await api(`/api/chats/${id}/agents`,{method:'POST',body:{task:$('#delegate-task').value,
        context:$('#delegate-context').value, overrides:JSON.parse($('#delegate-overrides').value || '{}'), read_only:$('#delegate-readonly').checked}});
      $('#modalhost').replaceChildren(); showPanelTab('agents', {quiet: true}); await refreshAgents(id); await loadChats();
    } catch (error) { if (S.chatId === id && S.viewEpoch === epoch) $('#agent-status').textContent = errorText(error); }
  }}], {cancelLabel: 'Cancel'});
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
    d.title = `${s.kind} · ${s.frozen ? 'frozen' : 'hot'}${s.note ? ' · ' + s.note : ''}${s.resource_key ? '\n' + s.resource_key : ''}`;
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
const TOOL_GROUPS = {
  files: 'Files', search: 'Search', shell: 'Shell', jobs: 'Background jobs', editing: 'Edit plans',
  documents: 'Documents', visual: 'Images & PDF pages', context: 'Context control', web: 'Web',
  browser: 'Browser research', agents: 'Delegation', 'research-web': 'Scholarly search',
  research: 'Evidence', science: 'Experiments', latex: 'LaTeX', general: 'General', other: 'Other',
};

function renderTools() {
  const host = $('#toollist'); host.replaceChildren();
  const on = new Set(S.cfg?.enabled_tools || []);
  const locked = !S.cfg || S.delegated || S.streaming;
  const query = ($('#tool-filter')?.value || '').trim().toLowerCase();
  const groups = new Map();
  for (const t of S.tools) {
    const g = t.group || 'other';
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(t);
  }
  $('#toolcount').textContent = S.tools.length ? `${S.tools.filter(t => on.has(t.name)).length} of ${S.tools.length} on` : '';
  for (const [group, tools] of groups) {
    const shown = query ? tools.filter(t => `${t.name} ${t.description || ''} ${TOOL_GROUPS[group] || group}`.toLowerCase().includes(query)) : tools;
    if (!shown.length) continue;
    const label = TOOL_GROUPS[group] || group.replace(/[-_]/g, ' ');
    const enabled = tools.filter(t => on.has(t.name)).length;
    const box = el('fieldset', 'toolgroup'), legend = el('legend'), title = el('label', 'toolgroup-title');
    const all = el('input'); all.type = 'checkbox'; all.disabled = locked;
    all.checked = enabled === tools.length; all.indeterminate = enabled > 0 && enabled < tools.length;
    all.setAttribute('aria-label', `Enable all ${label} tools`);
    all.onchange = () => setTools(tools.map(t => t.name), all.checked);
    title.append(all, el('span', '', label), el('span', 'count', `${enabled}/${tools.length}`));
    legend.appendChild(title); box.appendChild(legend);
    for (const t of shown) {
      const row = el('label', 'toolrow');
      const cb = el('input'); cb.type = 'checkbox'; cb.checked = on.has(t.name); cb.disabled = locked;
      cb.onchange = () => setTools([t.name], cb.checked);
      row.appendChild(cb);
      row.appendChild(el('span', 'nm', t.name));
      if (t.dangerous) { const w = el('span', 'warn', 'writes'); w.title = 'Can change files, run commands or start work'; row.appendChild(w); }
      else row.appendChild(el('span'));
      if (t.description) row.appendChild(el('span', 'desc', t.description));
      row.title = t.description || t.name;
      box.appendChild(row);
    }
    host.appendChild(box);
  }
  if (!host.children.length) host.appendChild(el('p', 'usage-note', query ? 'No tools match the filter.' : 'No tools available.'));
}

function setTools(names, enabled) {
  if (!S.cfg) return;
  const set = new Set(S.cfg.enabled_tools);
  for (const name of names) enabled ? set.add(name) : set.delete(name);
  S.cfg.enabled_tools = [...set];
  updateBrowserControls(); renderTools();
  patchCfg({ enabled_tools: S.cfg.enabled_tools })
    .then(saved => { if (saved) { renderTools(); setSendState(); } })
    .catch(() => reloadChatConfig(S.chatId));
}

function renderWorkdirs() {
  const host = $('#wdlist'); host.replaceChildren();
  const wds = S.cfg?.workdirs || [];
  if (!wds.length) host.appendChild(el('p', 'hint', 'No directory yet — file tools are disabled until you add one.'));
  for (const w of wds) {
    const d = el('div', 'wd');
    d.insertAdjacentHTML('beforeend', icon('folder'));
    const path = el('span', 'p', w); path.title = w; d.appendChild(path);
    const x = iconButton('iconbtn sm x', 'close', `Remove ${w}`);
    x.onclick = () => {
      S.cfg.workdirs = S.cfg.workdirs.filter(p => p !== w);
      patchCfg({ workdirs: S.cfg.workdirs }).catch(() => reloadChatConfig(S.chatId)); renderWorkdirs();
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
  refreshEmptyChat();
}

/* ------------------------------------------------------------ settings UI */
// A stored value the select does not list (xhigh in the summariser, max set through the
// API) gets its own option, so opening and saving the panel does not change it.
function setSelect(sel, value) {
  const el = $(sel);
  if (![...el.options].some(o => o.value === value)) el.append(new Option(value, value));
  el.value = value;
}

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
  setSelect('#f-reasoning', c.reasoning_effort || 'high');
  setSelect('#f-sumreasoning', c.compaction?.summary_reasoning || 'high');
  $('#f-provpolicy').value = c.provider_policy || 'eu_us';
  applyProvider(providerOf(c.model));
  // The provider prefix (anthropic/, openai/...) eats the chip's width; the full id is in the tooltip.
  $('#modelpill').textContent = String(c.model || '').split('/').pop() + (c.pricing_mode && c.pricing_mode !== 'standard' ? ' · ' + c.pricing_mode.replace('_', ' ') : '');
  $('#modelpill').title = `${c.model} (opens Model settings)`;
  checkProviders();
}

function fillCompaction() {
  const k = S.cfg?.compaction; if (!k) return;
  $('#c-hot-state').textContent = `Budget: ${fmt(k.hot_zone_tokens)} tokens.`;
  $('#c-interval').value = k.rewrite_interval_batches ?? 10;
  $('#c-newtokens').value = k.rewrite_max_new_tokens ?? 20000;
  $('#c-pressure').value = k.rewrite_pressure_ratio ?? 0.70;
  $('#c-enabled').checked = k.enabled;
  $('#c-review').checked = k.review_before_freeze === true;
  $('#c-delivery').value = k.initial_result_max_tokens ?? 0;
  $('#c-target').value = k.deep_target_ratio ?? 0.1;
  $('#c-hot').value = k.hot_zone_tokens; $('#c-hot-v').textContent = fmt(k.hot_zone_tokens);
  $('#c-protobs').value = k.protected_recent_observations;
  $('#c-protusr').value = k.protected_recent_user_turns;
  $('#c-deep').checked = k.deep_enabled;
  $('#c-trig').value = k.deep_trigger_ratio; $('#c-trig-v').textContent = pct(k.deep_trigger_ratio);
  $('#c-cool').value = k.deep_cooldown_turns;
  $('#c-mingain').value = k.deep_min_gain_tokens;
  $('#c-summodel').value = k.summary_model;
  $('#c-share').checked = k.summary_share_prefix !== false;
  $('#c-selfsteer').checked = k.rewrite_policy === 'agent';
  $('#c-guidance').value = k.guidance === 'live' ? 'live' : 'default';
  updateSelfSteerControls(); explainTail();
}

// Self-steered context management: one switch for compact_context + curate_context + the
// prompt section (config.SELF_STEERED_TOOLS); its details only matter while it is on.
function updateSelfSteerControls() {
  const engine = $('#c-enabled').checked, locked = !S.cfg || S.delegated;
  $('#c-selfsteer').disabled = locked || !engine;
  $('#c-selfgroup').disabled = locked || !engine || !$('#c-selfsteer').checked;
}

function explainTail() {
  $('#c-tailexpl').textContent = `Target: reduce TOTAL context to ${pct(+$('#c-target').value)} of its current size, not by that amount. Includes summary and tool/system overhead. Explicit pins and the newest tool group can impose a larger minimum; this is reported. Recalled documents can be archived again, losslessly.`;
}

function readCompaction() {
  return {
    enabled: $('#c-enabled').checked,
    review_before_freeze: $('#c-review').checked,
    initial_result_max_tokens: +$('#c-delivery').value,
    deep_target_ratio: +$('#c-target').value,
    hot_zone_tokens: +$('#c-hot').value,
    rewrite_interval_batches: +$('#c-interval').value,
    rewrite_max_new_tokens: +$('#c-newtokens').value,
    rewrite_pressure_ratio: +$('#c-pressure').value,
    protected_recent_observations: +$('#c-protobs').value,
    protected_recent_user_turns: +$('#c-protusr').value,
    deep_enabled: $('#c-deep').checked,
    deep_trigger_ratio: +$('#c-trig').value,
    deep_cooldown_turns: +$('#c-cool').value,
    deep_min_gain_tokens: +$('#c-mingain').value,
    summary_model: $('#c-summodel').value.trim(),
    summary_share_prefix: $('#c-share').checked,
    rewrite_policy: $('#c-selfsteer').checked ? 'agent' : 'auto',
    guidance: $('#c-guidance').value,
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
    host.innerHTML = `<b>✗ no API key</b> — ${esc(model)} needs <code>${KEY_NAMES[providerOf(model)]}</code>: add it in Settings &gt; API keys.`;
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
      prompt_tokens: Math.max(0, Math.ceil(S.stats?.used_tokens ?? S.stats?.total ?? 0)), needs_tools: which === 'model', refresh
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
  openrouter: {model: 'anthropic/claude-opus-5.5', summary: 'openai/gpt-5.6-luna'},
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
    ? `No ${KEY_NAMES[p]} configured (Settings > API keys): requests through this provider will be refused.`
    : 'Switching maps the chat and compaction models to their counterparts.';
}

function switchProvider() {
  const p = $('#f-provider').value;
  for (const [sel, kind] of [['#f-model', 'model'], ['#f-summodel', 'summary']]) {
    const current = $(sel).value.trim();
    if (!current && kind === 'summary') continue; // empty = same as the chat model
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
  const summ = $('#f-summodel').value.trim() || agent; // empty = same as the chat model
  if (agent && $('#f-provider').value !== providerOf(agent)) applyProvider(providerOf(agent));
  syncPricing('model', agent);
  syncPricing('sum', summ);
  $('#f-sum-pricing').disabled = !$('#f-summodel').value.trim(); // empty: follows the chat's tier
  checkOne(agent, '#model-check');
  checkOne(summ, '#sum-check');
  checkFlex('model');
  checkFlex('sum');
  showPrice(agent, '#model-price', +$('#f-ctx').value || 0);
  // The summariser sees the compacted trajectory, not the whole window.
  showPrice(summ, '#sum-price', S.stats ? (S.stats.used_tokens ?? S.stats.total) : 0);
}

/* --------------------------------------------- settings, saved on change */
// Every Settings and Engine control saves when its value is committed: a select, checkbox or
// slider at once, a text or number field on Enter or when it loses focus. While a turn runs
// the backend keeps the change for that turn's next model request (AgentSession.set_config).
function readSettings() {
  return {
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
  };
}

const SAVE = {
  settings: {
    fields: ['#f-model', '#f-ctx', '#f-maxout', '#f-maxiter', '#f-agent-parallel', '#f-agent-depth',
             '#f-agent-timeout', '#f-temp', '#f-sys', '#f-reasoning', '#f-provpolicy', '#f-model-pricing',
             '#f-summodel', '#f-sumreasoning', '#f-sum-pricing'],
    read: readSettings, refill: () => { fillSettings(); refreshEmptyChat(); }, state: '#cfg-state',
  },
  engine: {
    fields: ['#c-enabled', '#c-review', '#c-delivery', '#c-target', '#c-hot', '#c-interval', '#c-newtokens',
             '#c-pressure', '#c-protobs', '#c-protusr', '#c-deep', '#c-trig', '#c-cool', '#c-mingain',
             '#c-summodel', '#c-share', '#c-selfsteer', '#c-guidance'],
    read: () => ({ compaction: readCompaction() }), refill: () => { fillCompaction(); refreshContext(); },
    state: '#cmp-state',
  },
};

function saveState(which, text, kind = '') {
  const node = $(SAVE[which].state);
  if (!node) return;
  node.textContent = text;
  node.className = 'save-state' + (kind ? ' ' + kind : '');
}

function showConfigPending(pending) {
  for (const which of Object.keys(SAVE)) {
    if (!SAVE[which].dirty && !SAVE[which].busy) {
      saveState(which, pending ? 'Saved · applies at the next model request' : '', pending ? 'pending' : '');
    }
  }
}

function scheduleSave(which) {
  const job = SAVE[which];
  if (!S.chatId || !S.cfg || S.delegated) return;
  if (job.chat !== S.chatId) { job.chat = S.chatId; job.busy = false; }
  job.dirty = true;
  saveState(which, 'Saving…');
  clearTimeout(job.timer);
  job.timer = setTimeout(() => runSave(which), 250);
}

async function runSave(which) {
  const job = SAVE[which];
  if (job.busy || !job.dirty || job.chat !== S.chatId) return; // a running save starts the next
  job.dirty = false; job.busy = true;
  const id = S.chatId;
  let r = null;
  try { r = await patchCfg(job.read()); }
  catch (_) { if (S.chatId === id) saveState(which, 'Not saved', 'bad'); } // patchCfg showed why
  finally { job.busy = false; }
  if (S.chatId !== id) return;
  if (job.dirty) { runSave(which); return; } // changed again meanwhile: save that first
  if (!r) return;
  // The saved values back into the fields, except the one being edited.
  const active = document.activeElement;
  const keep = active?.matches?.('input:not([type=checkbox]), textarea') ? active.value : null;
  job.refill();
  if (keep !== null && active.value !== keep) active.value = keep;
  saveState(which, r.pending ? 'Saved · applies at the next model request' : 'Saved', r.pending ? 'pending' : 'good');
}

// The System instructions box at the top of the chat, after a settings change rewrote the
// system prompt (self-steering adds its section, a system prompt addition, a profile).
function showSystemPrompt(body) {
  const box = $('#msgwrap .notice.system-prompt');
  if (box && body != null) box.innerHTML = `<b>System instructions</b><div>${md(body)}</div>`;
}

async function patchCfg(patch) {
  const id = S.chatId, epoch = S.viewEpoch;
  if (!id) return false;
  try {
    const oldSummary = S.cfg?.compaction?.summary_model;
    const r = await api(`/api/chats/${id}/config`, { method: 'PATCH', body: { config: patch } });
    if (S.chatId !== id || S.viewEpoch !== epoch) return false;
    S.cfg = r.config; applyStats(r.stats);
    if (!r.pending) showSystemPrompt(r.system);
    // Synchronize untouched summary-model mirrors without replacing unsaved edits.
    for (const name of ['#f-summodel', '#c-summodel']) {
      if ($(name).value.trim() === oldSummary) $(name).value = S.cfg.compaction.summary_model;
    }
    return r; // {config, pending, stats}: pending = waits for the running turn's next request
  } catch (err) {
    toast(`Settings were NOT saved: ${errorText(err)}`, 'err');
    throw err;
  }
}

/* ----------------------------------------------------------------- modal */
let modalSeq = 0;
function modal(title, build, buttons, opts = {}) {
  // In-page dialog (the desktop WebView has no browser alert/confirm/prompt).
  // A button's fn may return false to keep the dialog open; a thrown error
  // keeps it open too and is shown as a toast. opts: {size: 'small',
  // cancelLabel, focus: 'cancel', onClose}.
  const host = $('#modalhost');
  const opener = document.activeElement;
  const m = el('div', 'modal' + (opts.size ? ' ' + opts.size : ''));
  const box = el('div', 'box');
  const titleId = `modal-title-${++modalSeq}`;
  box.setAttribute('role', 'dialog'); box.setAttribute('aria-modal', 'true'); box.setAttribute('aria-labelledby', titleId);
  let closed = false;
  const dismiss = () => {
    if (closed) return; closed = true;
    m.remove(); opts.onClose?.();
    if (opener && opener !== document.body && document.contains(opener)) opener.focus?.({preventScroll: true});
  };
  const hd = el('div', 'hd'), heading = el('h2', 'modal-title', title); heading.id = titleId;
  hd.appendChild(heading);
  hd.appendChild(el('span', 'spacer'));
  const close = iconButton('iconbtn sm', 'close', 'Close'); close.onclick = dismiss; hd.appendChild(close);
  const bd = el('div', 'bd'); build(bd);
  const ft = el('div', 'ft');
  for (const b of buttons || []) {
    const btn = el('button', 'btn' + (b.danger ? ' danger' : b.primary ? ' primary' : ''), b.label); btn.type = 'button';
    btn.onclick = async () => {
      btn.disabled = true;
      let result;
      try { result = await b.fn(); }
      catch (error) { toast(errorText(error), 'err'); return; }
      finally { btn.disabled = false; }
      if (result !== false && !b.keepOpen) dismiss();
    };
    ft.appendChild(btn);
  }
  const done = el('button', 'btn', opts.cancelLabel || 'Close'); done.type = 'button'; done.onclick = dismiss; ft.appendChild(done);
  box.appendChild(hd); box.appendChild(bd); box.appendChild(ft); m.appendChild(box);
  m.onclick = e => { if (e.target === m) dismiss(); };
  m.addEventListener('keydown', e => {
    if (e.key === 'Escape') { e.stopPropagation(); dismiss(); return; }
    if (e.key !== 'Tab') return;
    // Keep keyboard focus inside the dialog.
    const focusable = $$('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex]:not([tabindex="-1"])', box)
      .filter(x => x.offsetParent !== null);
    if (!focusable.length) return;
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  });
  m.dismiss = dismiss;
  host.appendChild(m);
  const target = opts.focus === 'cancel' ? done
    : bd.querySelector('input:not([type=checkbox]), textarea, select') || ft.querySelector('.btn.primary') || done;
  target.focus({preventScroll: true});
  return m;
}

function confirmModal({title, message, confirmLabel = 'Confirm', danger = false}) {
  return new Promise(resolve => {
    let settled = false;
    const settle = value => { if (!settled) { settled = true; resolve(value); } };
    modal(title, bd => bd.appendChild(el('p', 'confirm-text', message)),
      [{label: confirmLabel, primary: !danger, danger, fn: () => settle(true)}],
      {size: 'small', cancelLabel: 'Cancel', focus: danger ? 'cancel' : null, onClose: () => settle(false)});
  });
}

function toast(message, kind = 'ok', ms) {
  const host = $('#toasts'); if (!host) return;
  const t = el('div', 'toast ' + kind);
  t.setAttribute('role', kind === 'err' ? 'alert' : 'status');
  t.appendChild(el('span', 'toast-msg', String(message).slice(0, 600)));
  const x = iconButton('iconbtn sm', 'close', 'Dismiss'); x.onclick = () => t.remove(); t.appendChild(x);
  host.appendChild(t);
  while (host.children.length > 4) host.firstElementChild.remove();
  setTimeout(() => t.remove(), ms ?? (kind === 'err' ? 9000 : 3500));
}

async function pickDirectory() {
  let cur = '';
  const owner = S.chatId;
  if (!owner || !S.cfg) return;
  const m = modal('Choose a working directory', bd => { bd.id = 'dirbd'; }, [
    { label: 'Use this directory', primary: true, keepOpen: true, fn: async () => {
      if (!cur || S.chatId !== owner) return;
      const set = new Set(S.cfg.workdirs); set.add(cur);
      S.cfg.workdirs = [...set];
      await patchCfg({ workdirs: S.cfg.workdirs });
      renderWorkdirs(); m.dismiss();
    } },
  ], {cancelLabel: 'Cancel'});
  const bd = $('#dirbd', m);
  const row = (name, label, path) => {
    const r = el('button', 'dirrow'); r.type = 'button'; r.innerHTML = icon(name);
    r.appendChild(el('span', '', label)); r.onclick = () => draw(path); return r;
  };
  async function draw(path) {
    let d;
    try { d = await api('/api/browse', { method: 'POST', body: { path } }); }
    catch (error) { bd.replaceChildren(el('p', 'usage-note err', errorText(error))); return; }
    cur = d.path; bd.replaceChildren();
    bd.appendChild(el('div', 'crumb', d.path || 'Pick a drive or your home folder'));
    if (d.parent) bd.appendChild(row('up', '..', d.parent));
    for (const x of d.dirs) bd.appendChild(row('folder', x.name, x.path));
    if (!d.dirs.length) bd.appendChild(el('p', 'hint', '(no subdirectories)'));
    bd.querySelector('.dirrow')?.focus({preventScroll: true});
  }
  draw(S.cfg?.workdirs?.[0] || '');
}

function openAppSettings() { $('#open-settings').click(); }

async function forceCompact() {
  if (!S.chatId || !S.cfg || S.streaming) return;
  const id = S.chatId;
  modal('Deep compaction', bd => {
    bd.innerHTML = `<p>Reduce total context to <b>${pct(S.cfg.compaction.deep_target_ratio ?? .1)}</b> of its current size. Fixed overhead, explicit pins and the newest tool group may limit reduction; the result reports this. Originals stay recallable. The summary model is billed.</p>`;
    const field = el('div', 'field'), label = el('label', '', 'Focus instructions (optional)');
    const t = el('textarea'); t.id = 'cmpinstr'; t.rows = 3; label.htmlFor = t.id;
    t.placeholder = 'What must survive, what can go'; field.append(label, t); bd.appendChild(field);
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

/* ------------------------------------------------------ panel & layout */
const TAB_TITLES = {
  context: 'Context window', usage: 'Usage & cost', documents: 'Live documents', browser: 'Browser research',
  canvas: 'Live canvas', agents: 'Delegated agents', tools: 'Tools & directories', settings: 'Model & chat settings', compaction: 'Compaction engine',
};
const narrowSidebar = () => window.matchMedia('(max-width: 980px)').matches;

function activePanelTab() { return $('.tab.active')?.dataset.tab || 'context'; }

function setPanelOpen(open, persist = true) {
  $('#app').classList.toggle('panel-open', open);
  $('#btn-panel').classList.toggle('on', open);
  $('#btn-panel').setAttribute('aria-expanded', String(open));
  if (persist) store.set('pi.panel.open', open ? '1' : '0');
}

// Select a details tab (and open the panel). quiet skips the data refresh.
function showPanelTab(name, {quiet = false} = {}) {
  const tab = $(`.tab[data-tab="${name}"]`);
  if (!tab) return;
  setPanelOpen(true, !quiet);
  for (const t of $$('.tab')) {
    const on = t === tab;
    t.classList.toggle('active', on); t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1;
  }
  $$('.tabpane').forEach(p => p.classList.toggle('active', p.id === 'pane-' + name));
  $('#panel-title').textContent = TAB_TITLES[name] || tab.title;
  store.set('pi.panel.tab', name);
  tab.classList.remove('has-news');
  if (quiet) return;
  if (name === 'context' || name === 'compaction' || name === 'usage') refreshContext();
  else if (name === 'browser') refreshBrowser();
  else if (name === 'agents') refreshAgents();
  else if (name === 'documents') refreshDocuments();
  else if (name === 'canvas') loadCanvas();
}

function initPanel() {
  const tabs = $$('.tab');
  for (const t of tabs) t.onclick = () => showPanelTab(t.dataset.tab);
  // Roving focus: arrows move between tabs, Enter/Space activates.
  $('.tabs').addEventListener('keydown', ev => {
    const i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    const next = {ArrowDown: i + 1, ArrowRight: i + 1, ArrowUp: i - 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1}[ev.key];
    if (next == null) return;
    ev.preventDefault(); tabs[(next + tabs.length) % tabs.length].focus();
  });
  // Restore the last tab without overriding the open/closed state restored in index.html.
  const saved = store.get('pi.panel.tab'), wasOpen = $('#app').classList.contains('panel-open');
  showPanelTab(TAB_TITLES[saved] ? saved : 'context', {quiet: true});
  setPanelOpen(wasOpen, false);
  $('#btn-panel').onclick = () => setPanelOpen(!$('#app').classList.contains('panel-open'));
  $('#btn-panel-close').onclick = () => { setPanelOpen(false); $('#btn-panel').focus(); };
  $('#ctx-meter').onclick = () => showPanelTab('context');
  $('#savedpill').onclick = () => showPanelTab('compaction');
  $('#costpill').onclick = () => showPanelTab('usage');
  $('#modelpill').onclick = () => showPanelTab('settings');

  const resizer = $('#panel-resizer'), body = $('.panel-body'), app = $('#app');
  const width = () => body.getBoundingClientRect().width;
  const setWidth = (w, persist) => {
    w = Math.round(Math.max(300, Math.min(760, w)));
    app.style.setProperty('--panel-w', w + 'px');
    resizer.setAttribute('aria-valuenow', String(w));
    if (persist) store.set('pi.panel.width', String(w));
  };
  resizer.setAttribute('aria-valuenow', String(parseInt(getComputedStyle(app).getPropertyValue('--panel-w'), 10) || 400));
  resizer.addEventListener('pointerdown', ev => {
    if (ev.button !== 0) return;
    ev.preventDefault();
    const startX = ev.clientX, startW = width();
    resizer.setPointerCapture(ev.pointerId);
    resizer.classList.add('dragging'); document.body.classList.add('resizing');
    const move = e => setWidth(startW + (startX - e.clientX), false);
    const end = () => {
      resizer.removeEventListener('pointermove', move);
      resizer.classList.remove('dragging'); document.body.classList.remove('resizing');
      setWidth(width(), true);
    };
    resizer.addEventListener('pointermove', move);
    resizer.addEventListener('pointerup', end, {once: true});
    resizer.addEventListener('pointercancel', end, {once: true});
  });
  resizer.addEventListener('keydown', ev => {
    const step = ev.shiftKey ? 64 : 24;
    const w = {ArrowLeft: width() + step, ArrowRight: width() - step, Home: 760, End: 300}[ev.key];
    if (w == null) return;
    ev.preventDefault(); setWidth(w, true);
  });
  resizer.addEventListener('dblclick', () => setWidth(400, true));
}

function closeSidebarPeek() {
  $('#app').classList.remove('sidebar-peek');
  syncSidebarButton();
}

function syncSidebarButton() {
  const app = $('#app');
  const shown = narrowSidebar() ? app.classList.contains('sidebar-peek') : !app.classList.contains('sidebar-closed');
  $('#btn-sidebar').setAttribute('aria-expanded', String(shown));
}

function initSidebar() {
  const app = $('#app');
  $('#btn-sidebar').onclick = () => {
    if (narrowSidebar()) app.classList.toggle('sidebar-peek');
    else store.set('pi.sidebar.open', app.classList.toggle('sidebar-closed') ? '0' : '1');
    syncSidebarButton();
    if (!app.classList.contains('sidebar-closed') && (!narrowSidebar() || app.classList.contains('sidebar-peek'))) $('#chat-search').focus();
  };
  $('#scrim').onclick = closeSidebarPeek;
  window.matchMedia('(max-width: 980px)').addEventListener?.('change', () => { app.classList.remove('sidebar-peek'); syncSidebarButton(); });
  syncSidebarButton();
  const search = $('#chat-search');
  search.addEventListener('input', () => { S.chatFilter = search.value; renderChats(); });
  search.addEventListener('keydown', ev => {
    if (ev.key === 'Escape' && search.value) { ev.stopPropagation(); search.value = ''; S.chatFilter = ''; renderChats(); }
    else if (ev.key === 'ArrowDown') { ev.preventDefault(); $('#chatlist .chat-open')?.focus(); }
  });
  // Arrow keys walk the visible chats; Home/End jump; ArrowUp from the top returns to search.
  $('#chatlist').addEventListener('keydown', ev => {
    if (!ev.target.matches('.chat-open') || !['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(ev.key)) return;
    const rows = $$('#chatlist .chat-open').filter(b => b.offsetParent !== null), i = rows.indexOf(ev.target);
    ev.preventDefault();
    if (ev.key === 'ArrowUp' && i === 0) { search.focus(); return; }
    const next = {ArrowDown: Math.min(rows.length - 1, i + 1), ArrowUp: i - 1, Home: 0, End: rows.length - 1}[ev.key];
    rows[next]?.focus();
  });
  $('#btn-rename').onclick = () => renameChat();
  $('#chattitle').ondblclick = () => renameChat();
  $('#keystate').onclick = openAppSettings;
  document.addEventListener('keydown', ev => {
    if (ev.key !== 'Escape' || $('#modalhost').children.length) return;
    if (app.classList.contains('sidebar-peek')) { closeSidebarPeek(); $('#btn-sidebar').focus(); }
  });
}

function initThemePicker() {
  const select = $('#theme-select');
  if (!window.PiTheme) { select.closest('.theme-picker')?.remove(); return; }
  const fill = () => {
    select.replaceChildren(...PiTheme.list().map(t => {
      const o = el('option', '', t.id === 'system' ? `System (${t.dark ? 'dark' : 'light'})` : t.label);
      o.value = t.id; return o;
    }));
    select.value = PiTheme.current();
  };
  fill();
  select.onchange = () => PiTheme.apply(select.value);
  window.addEventListener('pi:theme-changed', fill);
}

// Fallback until the app-level Settings modal (settings.js) exists: open the
// per-chat Model tab. settings.js takes over by calling event.preventDefault()
// in its own click listener or by defining window.PiSettings.
function initSettingsButton() {
  $('#open-settings').addEventListener('click', ev => {
    setTimeout(() => { if (!ev.defaultPrevented && !window.PiSettings) showPanelTab('settings'); }, 0);
  });
}

/* ------------------------------------------------------------------ init */
function applyHealth(h) {
  S.tools = h.tools || S.tools || [];
  if (h.backend_build !== 'ctxagent-defaults-v1' || h.restart_required || h.accounting_version !== 2) {
    $('#backend-status').className = 'banner';
    $('#backend-status').textContent = 'BACKEND RESTART REQUIRED. Browser refresh alone does not activate new code. Cost categories/cadence may be unavailable. Stop the Python server, restart it, then reload.';
  } else {
    $('#backend-status').className = '';
    $('#backend-status').textContent = '';
    $('#foot').title = `Backend ${h.backend_build}, PID ${h.server_pid}, started ${new Date(h.server_started_at * 1000).toLocaleString()}`;
  }
  S.hasKey = !!h.has_key;
  $('#keystate').textContent = h.has_key ? '' : 'no API key';
  S.keys = {openrouter: h.has_openrouter_key ?? h.has_key, anthropic: !!h.has_anthropic_key};
  for (const o of $('#f-provider').options) {
    if (o.dataset.label == null) o.dataset.label = o.textContent;
    o.textContent = o.dataset.label + (S.keys[o.value] ? '' : ' — no key configured');
  }
  const banner = $('#banner'); banner.replaceChildren();
  if (!h.has_key) {
    const b = el('div', 'banner');
    b.innerHTML = icon('key');
    b.append(el('span', '', `No API key configured yet: ${appName()} cannot answer until you add one.`));
    const open = el('button', 'btn sm pi-banner-action', 'Add a key in Settings'); open.type = 'button';
    open.onclick = () => (window.PiSettings ? window.PiSettings.open('keys') : openAppSettings());
    b.append(open, el('small', '', 'Keys are kept in the Windows Credential Manager; an environment variable also works'));
    banner.appendChild(b);
  }
  $('#foot').textContent = `${S.tools.length} tools · ${h.levels_text || ''}`;
  S.llmLevels = h.llm_levels || {};
  refreshEmptyChat();
}

async function refreshHealth() {
  try {
    applyHealth(await api('/api/health'));
    if (S.cfg) { applyProvider(providerOf($('#f-model').value.trim() || S.cfg.model)); checkProviders(); }
  } catch (error) {
    toast(`Could not reach the backend: ${errorText(error)}`, 'err');
  }
}

async function init() {
  initThemePicker(); initPanel(); initCanvas(); initSidebar(); initSettingsButton();
  const h = await api('/api/health');
  applyHealth(h);
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
  $('#btn-send').onclick = () => send();
  $('#btn-compact').onclick = forceCompact;
  $('#btn-doc-open').onclick = openDocument;
  $('#btn-doc-refresh').onclick = () => refreshDocuments();
  $('#doc-path').onkeydown = event => { if (event.key === 'Enter') { event.preventDefault(); openDocument(); } };
  $('#btn-browser-open').onclick = openBrowser;
  $('#btn-browser-refresh').onclick = () => refreshBrowser();
  $('#btn-browser-enable').onclick = enableBrowserTools;
  $('#btn-browser-notes').onclick = () => listBrowserNotes();
  $('#btn-browser-notes-prev').onclick = () => listBrowserNotes(Math.max(0, browserUI.notesOffset - 20));
  $('#btn-browser-notes-next').onclick = () => listBrowserNotes(browserUI.notesNext);
  $('#browser-url').onkeydown = event => { if (event.key === 'Enter') { event.preventDefault(); openBrowser(); } };
  $('#btn-agents-refresh').onclick = () => refreshAgents();
  $('#btn-agent-start').onclick = startAgent;
  $('#btn-agent-enable').onclick = async () => {
    try {
      if (await patchCfg({enabled_tools: [...new Set([...(S.cfg.enabled_tools || []), ...AGENT_WEB_TOOLS])]})) { renderTools(); setSendState(); }
    } catch (error) { $('#agent-status').textContent = errorText(error); }
  };
  $('#btn-doc-enable').onclick = async () => {
    try {
      if (await patchCfg({enabled_tools: [...new Set([...(S.cfg.enabled_tools || []), ...DOCUMENT_TOOL_NAMES])]})) {
        renderTools(); await refreshDocuments();
      }
    } catch (error) { $('#doc-status').textContent = errorText(error); }
  };
  $('#btn-addwd').onclick = pickDirectory;
  $('#btn-wd').onclick = pickDirectory;
  $('#tool-filter').addEventListener('input', renderTools);

  const sp = $('#sp-text');
  sp.addEventListener('input', () => { clearTimeout(pad.timer); padStatus(''); pad.timer = setTimeout(saveScratchpad, 500); });
  sp.addEventListener('blur', saveScratchpad);
  sp.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && !e.isComposing) { e.preventDefault(); sendScratchpad(); }
  });
  $('#sp-send').onclick = sendScratchpad;
  const spBox = $('#scratchpad');
  const setPadOpen = open => {
    spBox.classList.toggle('collapsed', !open);
    $('#sp-toggle').setAttribute('aria-expanded', String(open));
    store.set('pi.scratchpad.open', open ? '1' : '0');
  };
  setPadOpen(store.get('pi.scratchpad.open') !== '0');
  $('#sp-toggle').onclick = () => setPadOpen(spBox.classList.contains('collapsed'));
  loadScratchpad(null);

  const ta = $('#input');
  ta.addEventListener('input', () => { saveDraft(); autoGrow(); });
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); submitComposer(); }
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
    $('#c-hot-state').textContent = `Budget: ${fmt(+e.target.value)} tokens (saved when you let go).`;
  };
  $('#c-trig').oninput = e => $('#c-trig-v').textContent = pct(+e.target.value);
  $('#c-target').oninput = explainTail;
  $('#c-selfsteer').onchange = updateSelfSteerControls;
  $('#c-enabled').addEventListener('change', updateSelfSteerControls);

  // Saved when a value is committed (see SAVE). A provider switch rewrites the model fields.
  for (const which of Object.keys(SAVE)) {
    for (const sel of SAVE[which].fields) $(sel).addEventListener('change', () => scheduleSave(which));
  }
  $('#f-provider').onchange = () => { switchProvider(); scheduleSave('settings'); };

  $('#btn-defaults').onclick = async () => {
    const id = S.chatId;
    if (!id || !S.cfg) return;
    try {
      await api(`/api/chats/${id}/defaults`, {method: 'POST'});
    } catch (error) { toast(`Defaults were NOT saved: ${errorText(error)}`, 'err'); return; }
    if (S.chatId === id) toast('Saved settings, Engine and Tools are now the defaults for new chats. Unsaved controls were not copied.');
  };

  // Other components (settings.js) announce changes they made elsewhere.
  window.addEventListener('pi:chat-config-changed', ev => reloadChatConfig(ev.detail?.chatId || S.chatId));
  window.addEventListener('pi:keys-changed', refreshHealth);
  setSendState();
}

// Small, stable surface for other scripts; everything else is internal.
window.PiApp = Object.freeze({
  get chatId() { return S.chatId; },
  get config() { return S.cfg; },
  openPanelTab: name => showPanelTab(name),
  reloadChat: id => reloadChatConfig(id || S.chatId),
  refreshHealth,
  toast: (message, kind) => toast(message, kind),
  confirm: options => confirmModal(options),
  modal: (title, build, buttons, opts) => modal(title, build, buttons, opts),
});

init().catch(e => {
  document.body.innerHTML = `<div class="empty fatal"><div class="empty-mark" aria-hidden="true"><svg viewBox="0 0 24 24"><use href="#i-ctx"/></svg></div><h2>Failed to start</h2><p>${esc(String(e))}</p><p>Check that the backend is running, then reload.</p></div>`;
});
