/* CTXAgent settings menu (API keys, profiles, new-chat defaults, appearance,
   about) and the top-bar profile slot.

   Self-contained: talks only to /api/settings/... and /api/chats, and to the
   GUI through these hooks:
     #open-settings              button that opens the menu
     #profile-slot               span that hosts the profile slot
     window 'pi:chat-loaded'     detail: chat object (GET /api/chats/{id}); fired by app.js
     window 'pi:chat-config-changed'  detail: {chatId}; fired HERE, app.js re-fetches the chat
     window 'pi:keys-changed'    detail: {openrouter, anthropic}; fired HERE after key edits
     window.PiTheme              optional {list(), current(), apply(name)}
     window.openChat / loadChats optional app.js globals used after "new chat from profile"
   Public: window.PiSettings = {open(section), close(), refreshSlot()}.
   No alert/confirm/prompt: every confirmation is in-page (pywebview-safe). */
(() => {
  'use strict';

  const API = '/api/settings';
  const SECTIONS = [
    ['keys', 'API keys'], ['profiles', 'Profiles'], ['newchat', 'New chats'],
    ['appearance', 'Appearance'], ['insights', 'Insights'], ['about', 'About'],
  ];
  const st = {
    overview: null, keys: null, profiles: [], defaultProfileId: '', prefs: {},
    chat: null, chatStatus: null, section: 'keys', editing: null, models: null, tools: null,
    opener: null, root: null, panel: null, slotSeq: 0,
  };

  /* ------------------------------------------------------------ helpers */
  function h(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value == null || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
      else if (key === 'value') node.value = value;
      else if (key === 'checked' || key === 'disabled' || key === 'selected') node[key] = !!value;
      else node.setAttribute(key, value === true ? '' : String(value));
    }
    for (const child of children.flat()) {
      if (child == null || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function errorText(detail, status) {
    if (Array.isArray(detail)) return detail.map(d => (d.loc ? d.loc.slice(1).join('.') + ': ' : '') + d.msg).join('; ');
    if (detail && typeof detail === 'object') return JSON.stringify(detail).slice(0, 300);
    return detail ? String(detail) : `HTTP ${status}`;
  }

  async function req(path, {method = 'GET', body} = {}) {
    const response = await fetch(path, {
      method, headers: body === undefined ? {} : {'Content-Type': 'application/json'},
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    let data = null;
    try { data = await response.json(); } catch (_) { /* empty body */ }
    if (!response.ok) throw new Error(errorText(data && data.detail, response.status));
    return data;
  }

  function status(node, text, kind) {
    node.textContent = text || '';
    node.className = 'pi-status' + (kind ? ' ' + kind : '');
  }

  function statusLine() { return h('div', {class: 'pi-status', role: 'status', 'aria-live': 'polite'}); }

  /** In-page confirmation inside `host`. Resolves true/false. Focuses Cancel. */
  function confirmInline(host, message, label, {danger = false} = {}) {
    host.querySelectorAll(':scope > .pi-confirm').forEach(n => n.remove());
    return new Promise(resolve => {
      const done = value => { box.remove(); resolve(value); };
      const cancel = h('button', {class: 'pi-btn', type: 'button', text: 'Cancel', onclick: () => done(false)});
      const ok = h('button', {class: 'pi-btn ' + (danger ? 'danger solid' : 'primary'), type: 'button', text: label, onclick: () => done(true)});
      const box = h('div', {class: 'pi-confirm', role: 'alertdialog', 'aria-label': label},
        h('p', {text: message}), h('div', {class: 'pi-actions'}, ok, cancel));
      box.addEventListener('keydown', e => { if (e.key === 'Escape') { e.stopPropagation(); done(false); } });
      host.append(box);
      cancel.focus();
    });
  }

  function download(name, data) {
    const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = h('a', {href: url, download: name});
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  function getPath(obj, path) { return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj); }
  function setPath(obj, path, value) {
    const parts = path.split('.'); let o = obj;
    for (const k of parts.slice(0, -1)) o = o[k] = o[k] || {};
    o[parts[parts.length - 1]] = value;
  }

  function fmtTokens(n) { return n >= 1e6 ? (n / 1e6) + 'M' : n >= 1000 ? Math.round(n / 1000) + 'k' : String(n); }

  /* ------------------------------------------------------------ data */
  async function loadOverview() {
    st.overview = await req(API);
    st.prefs = st.overview.prefs || {};
    return st.overview;
  }
  async function loadKeys() { st.keys = await req(API + '/keys'); return st.keys; }
  async function loadProfiles() {
    const data = await req(API + '/profiles');
    st.profiles = data.profiles || [];
    st.defaultProfileId = data.default_profile_id || '';
    return data;
  }
  function acceptProfiles(data) {
    if (data && data.profiles) { st.profiles = data.profiles; st.defaultProfileId = data.default_profile_id || ''; }
  }
  async function loadModels() {
    if (st.models) return st.models;
    try { st.models = (await req('/api/models')).models || []; } catch (_) { st.models = []; }
    const list = document.getElementById('pi-model-list') || document.body.appendChild(h('datalist', {id: 'pi-model-list'}));
    list.replaceChildren(...st.models.map(m => h('option', {value: m.id, label: m.name || m.id})));
    return st.models;
  }
  async function loadTools() {
    if (st.tools) return st.tools;
    try {
      const health = await req('/api/health');
      st.tools = {catalogue: health.tools || [], defaults: null};
    } catch (_) { st.tools = {catalogue: [], defaults: null}; }
    return st.tools;
  }
  const profileById = id => st.profiles.find(p => p.id === id) || null;

  /* ------------------------------------------------------------ modal */
  function build() {
    if (st.root) return st.root;
    const nav = h('div', {class: 'pi-settings-nav', role: 'tablist', 'aria-orientation': 'vertical', 'aria-label': 'Settings sections'});
    for (const [id, label] of SECTIONS) {
      nav.append(h('button', {type: 'button', role: 'tab', id: 'pi-tab-' + id, 'aria-controls': 'pi-settings-panel',
        'aria-selected': 'false', tabindex: '-1', 'data-section': id, text: label, onclick: () => show(id)}));
    }
    nav.addEventListener('keydown', e => {
      const tabs = [...nav.querySelectorAll('[role="tab"]')];
      const i = tabs.indexOf(document.activeElement);
      if (i < 0) return;
      const step = {ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1}[e.key];
      if (step) { e.preventDefault(); const t = tabs[(i + step + tabs.length) % tabs.length]; t.focus(); show(t.dataset.section, false); }
      if (e.key === 'Home') { e.preventDefault(); tabs[0].focus(); show(tabs[0].dataset.section, false); }
      if (e.key === 'End') { e.preventDefault(); tabs[tabs.length - 1].focus(); show(tabs[tabs.length - 1].dataset.section, false); }
    });
    st.panel = h('section', {class: 'pi-settings-panel', id: 'pi-settings-panel', role: 'tabpanel', tabindex: '0'});
    const closeBtn = h('button', {type: 'button', class: 'pi-icon-btn', 'aria-label': 'Close settings', title: 'Close (Esc)', text: '×', onclick: close});
    const dialog = h('div', {class: 'pi-settings-dialog', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'pi-settings-title'},
      h('div', {class: 'pi-settings-head'}, h('h2', {id: 'pi-settings-title', text: 'Settings'}), closeBtn),
      h('div', {class: 'pi-settings-body'}, nav, st.panel));
    st.root = h('div', {class: 'pi-settings', id: 'pi-settings', hidden: true},
      h('div', {class: 'pi-settings-backdrop', onclick: close}), dialog);
    st.root.addEventListener('keydown', e => {
      if (e.key === 'Escape') { e.preventDefault(); close(); return; }
      if (e.key !== 'Tab') return;
      const focusable = [...dialog.querySelectorAll('button, [href], input, select, textarea, summary, [tabindex]:not([tabindex="-1"])')]
        .filter(n => !n.disabled && n.offsetParent !== null);
      if (!focusable.length) return;
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    });
    document.body.append(st.root);
    return st.root;
  }

  async function open(section) {
    build();
    st.opener = document.activeElement;
    st.root.hidden = false;
    closeMenu();
    await show(section || st.section);
  }

  function close() {
    if (!st.root || st.root.hidden) return;
    st.root.hidden = true;
    st.editing = null;
    if (st.opener && document.contains(st.opener)) st.opener.focus();
  }

  async function show(section, focusTab = true) {
    st.section = section;
    for (const tab of st.root.querySelectorAll('[role="tab"]')) {
      const on = tab.dataset.section === section;
      tab.setAttribute('aria-selected', String(on));
      tab.tabIndex = on ? 0 : -1;
      if (on && focusTab) tab.focus();
    }
    st.panel.setAttribute('aria-labelledby', 'pi-tab-' + section);
    st.panel.replaceChildren(h('p', {class: 'pi-muted', text: 'Loading…'}));
    const render = {keys: renderKeys, profiles: renderProfiles, newchat: renderNewChat,
                    appearance: renderAppearance, insights: renderInsights, about: renderAbout}[section];
    try {
      await render(st.panel);
    } catch (error) {
      st.panel.replaceChildren(h('div', {class: 'pi-alert', text: 'Could not load this section: ' + error.message}));
    }
  }

  /* ------------------------------------------------------------ API keys */
  async function renderKeys(panel) {
    const data = await loadKeys();
    const backend = data.keystore || {};
    panel.replaceChildren(
      h('h3', {text: 'API keys'}),
      h('p', {class: 'pi-muted'}, 'Keys are stored in: ', h('b', {text: backend.label || 'unknown'}),
        '. A key in the store takes precedence over the same name in the environment or a .env file. ',
        'Keys are never shown in full, logged, or copied into chats and profiles.'),
    );
    if (!backend.writable) {
      panel.append(h('div', {class: 'pi-note', text: 'This installation has no writable key store (CTXAGENT_KEYSTORE=env). Set keys in the environment or .env instead.'}));
    }
    for (const key of data.keys) panel.append(keyCard(key, backend.writable));
    panel.append(otherKeyForm(backend.writable));
  }

  function sourceBadge(key) {
    if (!key.configured) return h('span', {class: 'pi-badge warn', text: 'not set'});
    if (key.source === 'store') return h('span', {class: 'pi-badge ok', text: 'credential store'});
    return h('span', {class: 'pi-badge', text: 'environment / .env'});
  }

  function keyCard(key, writable) {
    const line = statusLine();
    const body = h('div');
    const card = h('div', {class: 'pi-card', 'data-key': key.name},
      h('div', {class: 'pi-card-head'},
        h('span', {class: 'pi-card-title', text: key.label}), h('code', {text: key.name}), sourceBadge(key),
        h('span', {class: 'pi-spacer'}),
        key.configured ? h('span', {class: 'pi-masked', 'aria-label': 'masked value', text: key.masked}) : null),
      key.help ? h('div', {class: 'pi-muted pi-small', text: key.help}) : null,
      key.source === 'env' ? h('div', {class: 'pi-muted pi-small', text: 'Read from the environment. Saving a key here overrides it; edit .env to change it there.'}) : null,
      key.stored && key.env_present ? h('div', {class: 'pi-muted pi-small', text: 'An environment value also exists; it is used again if the stored key is removed.'}) : null,
      body, line);
    const actions = h('div', {class: 'pi-actions'});
    actions.append(h('button', {type: 'button', class: 'pi-btn' + (key.configured ? '' : ' primary'), disabled: !writable,
      text: key.configured ? (key.stored ? 'Replace' : 'Save a key here') : 'Add key', onclick: () => editKey(card, key, body, line)}));
    if (key.testable) {
      actions.append(h('button', {type: 'button', class: 'pi-btn', disabled: !key.configured, text: 'Test',
        title: 'Free check, nothing is generated or billed', onclick: () => testKey(key.name, null, line)}));
    }
    if (key.stored) {
      actions.append(h('button', {type: 'button', class: 'pi-btn danger', text: 'Remove', onclick: async () => {
        const fallback = key.env_present ? ' The environment / .env value will be used again.' : '';
        if (!await confirmInline(body, `Remove the stored ${key.label} from the credential store?${fallback}`, 'Remove', {danger: true})) return;
        try {
          const data = await req(`${API}/keys/${encodeURIComponent(key.name)}`, {method: 'DELETE'});
          keysChanged(data.status);
          await renderKeys(st.panel);
          flash(key.name, 'Removed.', 'ok');
        } catch (error) { status(line, error.message, 'err'); }
      }}));
    }
    body.append(actions);
    return card;
  }

  function flash(name, text, kind) {
    const line = st.panel.querySelector(`[data-key="${name}"] .pi-status`);
    if (line) status(line, text, kind);
  }

  function editKey(card, key, body, line) {
    body.querySelectorAll('.pi-keyform, .pi-confirm').forEach(n => n.remove());
    const input = h('input', {type: 'password', autocomplete: 'off', spellcheck: 'false', 'aria-label': `New value for ${key.name}`,
      placeholder: key.placeholder || 'paste the key', class: 'pi-grow', 'data-lpignore': 'true'});
    const reveal = h('button', {type: 'button', class: 'pi-btn', text: 'Show', 'aria-pressed': 'false', onclick: () => {
      const shown = input.type === 'text'; input.type = shown ? 'password' : 'text';
      reveal.textContent = shown ? 'Show' : 'Hide'; reveal.setAttribute('aria-pressed', String(!shown));
    }});
    const save = async () => {
      if (!input.value.trim()) { status(line, 'Paste a key first.', 'err'); input.focus(); return; }
      status(line, 'Saving…', 'busy');
      try {
        const data = await req(`${API}/keys/${encodeURIComponent(key.name)}`, {method: 'PUT', body: {value: input.value}});
        input.value = '';
        keysChanged(data.status);
        await renderKeys(st.panel);
        flash(key.name, 'Saved to the key store.' + (key.testable ? ' Use Test to verify it.' : ''), 'ok');
      } catch (error) { status(line, error.message, 'err'); }
    };
    input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); save(); } });
    const form = h('div', {class: 'pi-keyform'},
      h('div', {class: 'pi-row'}, input, reveal),
      h('div', {class: 'pi-actions'},
        h('button', {type: 'button', class: 'pi-btn primary', text: 'Save', onclick: save}),
        key.testable ? h('button', {type: 'button', class: 'pi-btn', text: 'Test before saving',
          onclick: () => input.value.trim() ? testKey(key.name, input.value, line) : status(line, 'Paste a key first.', 'err')}) : null,
        h('button', {type: 'button', class: 'pi-btn', text: 'Cancel', onclick: () => { input.value = ''; form.remove(); status(line, ''); }})));
    body.prepend(form);
    input.focus();
  }

  async function testKey(name, value, line) {
    status(line, 'Checking with the provider (free, no generation)…', 'busy');
    try {
      const r = await req(`${API}/keys/${encodeURIComponent(name)}/test`, {method: 'POST', body: value ? {value} : {}});
      const extra = [];
      if (r.detail) {
        if (r.detail.label) extra.push('label ' + r.detail.label);
        if (r.detail.limit != null) extra.push('limit $' + r.detail.limit);
        if (r.detail.usage != null) extra.push('used $' + Number(r.detail.usage).toFixed(2));
        if (r.detail.is_free_tier) extra.push('free tier');
      }
      status(line, (r.ok ? '✓ ' : '✗ ') + r.message + (extra.length ? ' (' + extra.join(', ') + ')' : ''), r.ok ? 'ok' : 'err');
    } catch (error) { status(line, error.message, 'err'); }
  }

  function otherKeyForm(writable) {
    const name = h('input', {type: 'text', placeholder: 'MISTRAL_API_KEY', spellcheck: 'false', 'aria-label': 'Key name', autocomplete: 'off'});
    const value = h('input', {type: 'password', placeholder: 'value', autocomplete: 'off', spellcheck: 'false', 'aria-label': 'Key value'});
    const line = statusLine();
    return h('details', {class: 'pi-more'}, h('summary', {text: 'Store a key for another provider'}),
      h('p', {class: 'pi-muted pi-small', text: 'For integrations that read other variables. Names are upper-case letters, digits and underscores.'}),
      h('div', {class: 'pi-grid2'}, h('div', {class: 'pi-field'}, name), h('div', {class: 'pi-field'}, value)),
      h('button', {type: 'button', class: 'pi-btn', disabled: !writable, text: 'Save key', onclick: async () => {
        try {
          const data = await req(`${API}/keys/${encodeURIComponent(name.value.trim())}`, {method: 'PUT', body: {value: value.value}});
          value.value = ''; keysChanged(data.status);
          await renderKeys(st.panel);
        } catch (error) { status(line, error.message, 'err'); }
      }}), line);
  }

  function keysChanged(state) {
    if (!state) return;
    window.dispatchEvent(new CustomEvent('pi:keys-changed', {detail: state}));
    // Today's app.js keeps key flags in its global S; keep it current until it listens itself.
    try {
      if (typeof S === 'object' && S) { S.keys = {openrouter: !!state.openrouter, anthropic: !!state.anthropic}; } // eslint-disable-line no-undef
      if (typeof checkProviders === 'function') checkProviders(); // eslint-disable-line no-undef
    } catch (_) { /* GUI without these globals */ }
    const any = state.openrouter || state.anthropic;
    const keystate = document.getElementById('keystate');
    if (keystate) keystate.textContent = any ? '' : 'no API key';
    const banner = document.getElementById('banner');
    if (banner && any && /API_KEY/.test(banner.textContent || '')) banner.replaceChildren();
  }

  /* ------------------------------------------------------------ profiles */
  async function renderProfiles(panel) {
    await loadProfiles();
    if (st.editing) return renderEditor(panel, st.editing);
    const fileInput = h('input', {type: 'file', accept: '.json,application/json', hidden: true, 'aria-label': 'Import profiles file'});
    const line = statusLine();
    fileInput.addEventListener('change', async () => {
      const file = fileInput.files && fileInput.files[0];
      fileInput.value = '';
      if (!file) return;
      try { await importText(await file.text(), line); } catch (error) { status(line, error.message, 'err'); }
    });
    const exportHost = h('div', {class: 'pi-export'});
    panel.replaceChildren(
      h('h3', {text: 'Profiles'}),
      h('p', {class: 'pi-muted', text: 'A profile is a named set of chat settings: model, reasoning, limits, pricing, tools, instructions, compaction and delegation. '
        + 'New chats copy a profile; the chat keeps its own copy and stays editable. Editing a profile never changes existing chats.'}),
      h('div', {class: 'pi-actions'},
        h('button', {type: 'button', class: 'pi-btn primary', text: 'New profile', onclick: () => createProfile({name: 'New profile', config: {}}, line)}),
        st.chat ? h('button', {type: 'button', class: 'pi-btn', text: 'New profile from this chat',
          onclick: () => createProfile({name: (st.chat.title || 'Chat') + ' settings', from_chat: st.chat.id}, line)}) : null,
        h('button', {type: 'button', class: 'pi-btn', text: 'Import…', onclick: () => fileInput.click()}),
        h('button', {type: 'button', class: 'pi-btn', text: 'Paste JSON…', onclick: () => pasteImport(exportHost, line)}),
        h('button', {type: 'button', class: 'pi-btn', text: 'Export my profiles', onclick: () => exportProfiles(null, exportHost, line)}),
        fileInput),
      line, exportHost);
    for (const p of st.profiles) panel.append(profileCard(p, line, exportHost));
  }

  function profileCard(p, line, exportHost) {
    const cfg = p.effective || {};
    const isDefault = p.id === st.defaultProfileId;
    const body = h('div');
    const summary = p.effective
      ? `${cfg.model} · reasoning ${cfg.reasoning_effort} · ${fmtTokens(cfg.context_window)} window · summariser ${cfg.compaction && cfg.compaction.summary_model ? cfg.compaction.summary_model : 'same as chat'}`
      : 'Invalid profile: ' + p.error;
    const card = h('div', {class: 'pi-card'},
      h('div', {class: 'pi-card-head'}, h('span', {class: 'pi-card-title', text: p.name}),
        p.builtin ? h('span', {class: 'pi-badge', text: 'built-in'}) : null,
        isDefault ? h('span', {class: 'pi-badge accent', text: 'default for new chats'}) : null,
        st.chat && st.chatStatus && st.chatStatus.profile_id === p.id ? h('span', {class: 'pi-badge ok', text: 'this chat'}) : null),
      h('div', {class: 'pi-mono pi-small', text: summary}),
      p.description ? h('div', {class: 'pi-muted pi-small', text: p.description}) : null,
      body);
    const actions = h('div', {class: 'pi-actions'},
      h('button', {type: 'button', class: 'pi-btn', text: p.builtin ? 'View' : 'Edit', onclick: () => { st.editing = p.id; renderProfiles(st.panel); }}),
      h('button', {type: 'button', class: 'pi-btn', text: 'Duplicate', onclick: async () => {
        try {
          const data = await req(`${API}/profiles/${encodeURIComponent(p.id)}/duplicate`, {method: 'POST', body: {}});
          acceptProfiles(data); st.editing = data.profile.id; await renderProfiles(st.panel);
        } catch (error) { status(line, error.message, 'err'); }
      }}),
      h('button', {type: 'button', class: 'pi-btn', disabled: isDefault || !p.effective, text: 'Use for new chats', onclick: async () => {
        try { await setPrefs({default_profile_id: p.id}); await renderProfiles(st.panel); } catch (error) { status(line, error.message, 'err'); }
      }}),
      h('button', {type: 'button', class: 'pi-btn', disabled: !p.effective, text: 'New chat', onclick: () => newChatFromProfile(p.id, line)}),
      h('button', {type: 'button', class: 'pi-btn', text: 'Export', onclick: () => exportProfiles([p.id], exportHost, line)}),
      p.builtin ? null : h('button', {type: 'button', class: 'pi-btn danger', text: 'Delete', onclick: async () => {
        if (!await confirmInline(body, `Delete the profile "${p.name}"? Chats created from it keep their settings.`, 'Delete', {danger: true})) return;
        try { acceptProfiles(await req(`${API}/profiles/${encodeURIComponent(p.id)}`, {method: 'DELETE'})); await renderProfiles(st.panel); refreshSlot(); }
        catch (error) { status(line, error.message, 'err'); }
      }}));
    body.append(actions);
    return card;
  }

  async function createProfile(body, line) {
    try {
      const data = await req(API + '/profiles', {method: 'POST', body});
      acceptProfiles(data); st.editing = data.profile.id;
      await renderProfiles(st.panel);
    } catch (error) { status(line, error.message, 'err'); }
  }

  async function exportProfiles(ids, host, line) {
    try {
      const data = await req(API + '/profiles/export', {method: 'POST', body: {ids}});
      if (!data.profiles.length) { status(line, 'No user profiles to export yet (built-ins can be exported one by one).', 'busy'); return; }
      const name = ids && ids.length === 1 ? `ctxagent-profile-${ids[0]}.json` : 'ctxagent-profiles.json';
      download(name, data);
      const text = h('textarea', {readonly: true, rows: '6', 'aria-label': 'Exported profile JSON'});
      text.value = JSON.stringify(data, null, 2);
      host.replaceChildren(h('p', {class: 'pi-muted pi-small', text: `Downloaded ${name} (${data.profiles.length} profile${data.profiles.length === 1 ? '' : 's'}; settings only, no keys). If your window blocks downloads, copy the JSON:`}), text,
        h('div', {class: 'pi-actions'},
          h('button', {type: 'button', class: 'pi-btn', text: 'Copy', onclick: async () => {
            try { await navigator.clipboard.writeText(text.value); status(line, 'Copied.', 'ok'); } catch (_) { text.select(); }
          }}),
          h('button', {type: 'button', class: 'pi-btn', text: 'Hide', onclick: () => host.replaceChildren()})));
    } catch (error) { status(line, error.message, 'err'); }
  }

  function pasteImport(host, line) {
    const text = h('textarea', {rows: '6', placeholder: 'Paste exported profile JSON', 'aria-label': 'Profile JSON to import'});
    host.replaceChildren(text, h('div', {class: 'pi-actions'},
      h('button', {type: 'button', class: 'pi-btn primary', text: 'Import', onclick: () => importText(text.value, line).catch(e => status(line, e.message, 'err'))}),
      h('button', {type: 'button', class: 'pi-btn', text: 'Cancel', onclick: () => host.replaceChildren()})));
    text.focus();
  }

  async function importText(raw, line) {
    let parsed;
    try { parsed = JSON.parse(raw); } catch (_) { throw new Error('That is not valid JSON.'); }
    const data = await req(API + '/profiles/import', {method: 'POST', body: {data: parsed}});
    acceptProfiles(data);
    await renderProfiles(st.panel);
    status(st.panel.querySelector('.pi-status'), `Imported ${data.created.length} profile${data.created.length === 1 ? '' : 's'}.`, 'ok');
  }

  async function newChatFromProfile(pid, line) {
    try {
      const {id} = await req('/api/chats', {method: 'POST', body: {title: 'New chat', profile_id: pid}});
      close();
      if (typeof window.openChat === 'function') {
        await window.openChat(id);
        if (typeof window.loadChats === 'function') window.loadChats();
      } else { location.reload(); }
    } catch (error) { if (line) status(line, error.message, 'err'); }
  }

  /* --- profile editor ------------------------------------------------- */
  const EFFORTS = [['high', 'high'], ['medium', 'medium'], ['low', 'low'], ['off', 'off'], ['default', 'provider default']];
  const CHAT_EFFORTS = [['xhigh', 'xhigh'], ...EFFORTS];
  const PRICING = [['standard', 'Standard'], ['prefer_flex', 'Try Flex, allow standard fallback'], ['flex_only', 'Flex only']];
  const HANDLED = new Set(['model', 'reasoning_effort', 'context_window', 'max_output_tokens', 'temperature', 'max_tool_iterations',
    'provider_policy', 'pricing_mode', 'system_prompt', 'enabled_tools', 'agent_max_parallel', 'agent_max_depth',
    'agent_timeout_seconds', 'compaction']);
  const SUMMARY_KEYS = new Set(['summary_model', 'summary_reasoning', 'summary_pricing_mode']);
  const COMPACTION_ESSENTIALS = [
    ['enabled', 'Compaction enabled'],
    ['rewrite_policy', 'Self-steered context management'], ['rewrite_interval_batches', 'Review hint every N tool batches'],
    ['initial_result_max_tokens', 'Initial tool-result cap (0 = unlimited)'],
    ['deep_enabled', 'Deep compaction'],
    ['deep_trigger_ratio', 'Deep compaction trigger (fraction of window)'], ['deep_target_ratio', 'Target remaining after deep pass'],
    ['summary_max_tokens', 'Summary max tokens'],
  ];
  const ESSENTIAL_KEYS = new Set(COMPACTION_ESSENTIALS.map(([k]) => k));
  const SELECTS = {
    reasoning_effort: CHAT_EFFORTS, 'compaction.summary_reasoning': EFFORTS,
    pricing_mode: PRICING, 'compaction.summary_pricing_mode': PRICING,
    provider_policy: [['eu_us', 'EU / US providers only'], ['any', 'Any provider']],
    'compaction.rewrite_policy': [['agent', 'On: the model manages its context (compact_context, curate_context)'],
      ['auto', 'Off: the engine manages the context with deep compaction']],
    'compaction.guidance': [['default', 'Standard'], ['live', 'Benchmarked text (frozen benchmark arm)']],
  };
  const humanize = key => key.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());

  function fieldFor(path, value, label, {disabled, help, attrs = {}} = {}) {
    const id = 'pi-f-' + path.replace(/\./g, '-');
    let input;
    if (SELECTS[path]) {
      input = h('select', {id, disabled}, SELECTS[path].map(([v, t]) => h('option', {value: v, text: t, selected: v === value})));
      if (!SELECTS[path].some(([v]) => v === value)) input.append(h('option', {value, text: String(value), selected: true}));
      input.dataset.kind = 'string';
    } else if (typeof value === 'boolean') {
      input = h('input', {type: 'checkbox', id, checked: value, disabled});
      input.dataset.kind = 'bool';
      input.dataset.path = path;
      return h('label', {class: 'pi-check'}, input, h('span', {}, label, help ? h('small', {class: 'pi-muted', text: ' ' + help}) : null));
    } else if (typeof value === 'number') {
      input = h('input', {type: 'number', id, value: String(value), step: 'any', disabled, ...attrs});
      input.dataset.kind = 'number';
    } else if (Array.isArray(value)) {
      input = h('input', {type: 'text', id, value: value.join(', '), disabled, spellcheck: 'false'});
      input.dataset.kind = 'list';
    } else {
      input = h('input', {type: 'text', id, value: value == null ? '' : String(value), disabled, spellcheck: 'false', ...attrs});
      input.dataset.kind = 'string';
    }
    input.dataset.path = path;
    return h('div', {class: 'pi-field'}, h('label', {for: id, text: label}), input, help ? h('small', {text: help}) : null);
  }

  function readField(input) {
    const kind = input.dataset.kind;
    if (kind === 'bool') return input.checked;
    if (kind === 'number') {
      if (input.value.trim() === '' || !Number.isFinite(Number(input.value))) throw new Error(`${input.dataset.path}: enter a number`);
      return Number(input.value);
    }
    if (kind === 'list') return input.value.split(',').map(s => s.trim()).filter(Boolean).map(s => (Number.isFinite(Number(s)) ? Number(s) : s));
    return input.value;
  }

  async function renderEditor(panel, pid) {
    const p = profileById(pid);
    if (!p) { st.editing = null; return renderProfiles(panel); }
    const [, tools] = await Promise.all([loadModels(), loadTools()]);
    const ro = p.builtin;
    const cfg = JSON.parse(JSON.stringify(p.effective || {}));
    const alertBox = h('div', {role: 'alert'});
    const line = statusLine();
    const form = h('form', {class: 'pi-profile-form', 'aria-label': `Profile ${p.name}`, novalidate: true});
    form.addEventListener('submit', e => e.preventDefault());

    const name = h('input', {type: 'text', id: 'pi-f-name', value: p.name, disabled: ro, maxlength: '80'});
    const description = h('textarea', {id: 'pi-f-description', rows: '2', disabled: ro, maxlength: '500'});
    description.value = p.description || '';
    form.append(h('fieldset', {class: 'pi-group'}, h('legend', {text: 'Profile'}),
      h('div', {class: 'pi-field'}, h('label', {for: 'pi-f-name', text: 'Name'}), name),
      h('div', {class: 'pi-field'}, h('label', {for: 'pi-f-description', text: 'Description'}), description)));

    form.append(h('fieldset', {class: 'pi-group'}, h('legend', {text: 'Model'}),
      fieldFor('model', cfg.model, 'Model id', {disabled: ro, attrs: {list: 'pi-model-list'},
        help: 'Bare Claude ids (claude-opus-5-5) use the Anthropic API key; ids with a provider prefix (anthropic/claude-opus-5.5) use OpenRouter.'}),
      h('div', {class: 'pi-grid2'},
        fieldFor('reasoning_effort', cfg.reasoning_effort, 'Reasoning effort', {disabled: ro}),
        fieldFor('context_window', cfg.context_window, 'Context window (tokens)', {disabled: ro, attrs: {min: '1000', step: '1000'}}),
        fieldFor('max_output_tokens', cfg.max_output_tokens, 'Max output tokens', {disabled: ro, attrs: {min: '1', step: '500'}}),
        fieldFor('temperature', cfg.temperature, 'Temperature', {disabled: ro, attrs: {min: '0', max: '2', step: '0.05'}, help: 'Not sent to current Claude models.'}),
        fieldFor('max_tool_iterations', cfg.max_tool_iterations, 'Tool iterations per turn (0 = unlimited)', {disabled: ro, attrs: {min: '0', step: '1'}}))));

    form.append(h('fieldset', {class: 'pi-group'}, h('legend', {text: 'Routing and pricing'}),
      h('div', {class: 'pi-grid2'},
        fieldFor('provider_policy', cfg.provider_policy, 'Provider region', {disabled: ro}),
        fieldFor('pricing_mode', cfg.pricing_mode, 'Pricing / service tier', {disabled: ro, help: 'Flex is an OpenRouter tier; the Anthropic API bills standard.'}))));

    // Summariser: "same as chat model" (default) or an explicit model.
    const comp = cfg.compaction || {};
    const same = !(comp.summary_model || '').trim();
    const sameRadio = h('input', {type: 'radio', name: 'pi-summary-choice', value: 'same', checked: same, disabled: ro});
    const otherRadio = h('input', {type: 'radio', name: 'pi-summary-choice', value: 'other', checked: !same, disabled: ro});
    const summaryModel = h('input', {type: 'text', id: 'pi-f-summary-model', list: 'pi-model-list', spellcheck: 'false',
      value: comp.summary_model || '', placeholder: 'e.g. openai/gpt-5.6-luna', disabled: ro || same, 'aria-label': 'Summariser model id'});
    const summaryPricing = fieldFor('compaction.summary_pricing_mode', comp.summary_pricing_mode, 'Summariser pricing tier', {disabled: ro || same,
      help: same ? 'Follows the chat\'s tier while the summariser is the chat model.' : ''});
    const syncSummary = () => {
      const isSame = sameRadio.checked;
      summaryModel.disabled = ro || isSame;
      summaryPricing.querySelector('select').disabled = ro || isSame;
      const small = summaryPricing.querySelector('small');
      if (small) small.textContent = isSame ? 'Follows the chat\'s tier while the summariser is the chat model.' : '';
      if (!isSame) summaryModel.focus();
    };
    sameRadio.addEventListener('change', syncSummary); otherRadio.addEventListener('change', syncSummary);
    form.append(h('fieldset', {class: 'pi-group'}, h('legend', {text: 'Summariser (compaction and LLM summaries)'}),
      h('div', {role: 'radiogroup', 'aria-label': 'Summariser model'},
        h('label', {class: 'pi-check'}, sameRadio, h('span', {}, 'Same as chat model ', h('small', {class: 'pi-muted', text: '(default: the conversation is already cached for it)'}))),
        h('label', {class: 'pi-check'}, otherRadio, h('span', {text: 'Another model'}))),
      h('div', {class: 'pi-field'}, summaryModel),
      h('div', {class: 'pi-grid2'},
        fieldFor('compaction.summary_reasoning', comp.summary_reasoning, 'Summariser reasoning effort', {disabled: ro}),
        summaryPricing)));

    const sys = h('textarea', {id: 'pi-f-system_prompt', rows: '4', disabled: ro, placeholder: 'Project conventions, house style, constraints…'});
    sys.value = cfg.system_prompt || '';
    sys.dataset.path = 'system_prompt'; sys.dataset.kind = 'string';
    form.append(h('fieldset', {class: 'pi-group'}, h('legend', {text: 'Instructions'}),
      h('div', {class: 'pi-field'}, h('label', {for: 'pi-f-system_prompt', text: 'System prompt addition'}), sys)));

    // Tools
    const enabled = new Set(cfg.enabled_tools || []);
    const toolBoxes = [];
    const toolsGrid = h('div', {class: 'pi-tools'});
    const catalogue = tools.catalogue.length ? tools.catalogue : [...enabled].map(n => ({name: n, description: ''}));
    for (const t of catalogue) {
      const box = h('input', {type: 'checkbox', checked: enabled.has(t.name), disabled: ro, value: t.name});
      toolBoxes.push(box);
      toolsGrid.append(h('label', {class: 'pi-check', title: t.description || ''}, box,
        h('span', {class: 'pi-mono', text: t.name}), t.dangerous ? h('span', {class: 'pi-badge warn', text: 'writes'}) : null));
    }
    form.append(h('fieldset', {class: 'pi-group'}, h('legend', {text: `Tools (${enabled.size} enabled)`}),
      ro ? null : h('div', {class: 'pi-actions'},
        h('button', {type: 'button', class: 'pi-btn', text: 'All', onclick: () => toolBoxes.forEach(b => { b.checked = true; })}),
        h('button', {type: 'button', class: 'pi-btn', text: 'None', onclick: () => toolBoxes.forEach(b => { b.checked = false; })})),
      toolsGrid));

    // Compaction: essentials first, the rest generically (fields added later appear here too).
    const essentials = h('div', {class: 'pi-grid2'});
    for (const [key, label] of COMPACTION_ESSENTIALS) {
      if (key in comp) essentials.append(fieldFor('compaction.' + key, comp[key], label, {disabled: ro}));
    }
    const rest = h('div', {class: 'pi-grid2'});
    for (const key of Object.keys(comp).sort()) {
      if (ESSENTIAL_KEYS.has(key) || SUMMARY_KEYS.has(key)) continue;
      rest.append(fieldFor('compaction.' + key, comp[key], humanize(key), {disabled: ro}));
    }
    form.append(h('fieldset', {class: 'pi-group'}, h('legend', {text: 'Compaction'}), essentials,
      h('details', {class: 'pi-more'}, h('summary', {text: 'All compaction settings'}), rest)));

    const delegation = h('div', {class: 'pi-grid2'},
      fieldFor('agent_max_parallel', cfg.agent_max_parallel, 'Parallel child agents (1-8)', {disabled: ro, attrs: {min: '1', max: '8', step: '1'}}),
      fieldFor('agent_max_depth', cfg.agent_max_depth, 'Delegation depth (1-4)', {disabled: ro, attrs: {min: '1', max: '4', step: '1'}}),
      fieldFor('agent_timeout_seconds', cfg.agent_timeout_seconds, 'Child timeout incl. queue (s)', {disabled: ro, attrs: {min: '1', step: '60'}}));
    form.append(h('fieldset', {class: 'pi-group'}, h('legend', {text: 'Delegation'}), delegation));

    const other = Object.keys(cfg).filter(k => !HANDLED.has(k)).sort();
    if (other.length) {
      form.append(h('fieldset', {class: 'pi-group'}, h('legend', {text: 'Other settings'}),
        h('div', {class: 'pi-grid2'}, other.map(k => fieldFor(k, cfg[k], humanize(k), {disabled: ro})))));
    }

    const back = () => { st.editing = null; renderProfiles(st.panel); };
    const save = async () => {
      alertBox.replaceChildren();
      let next;
      try {
        next = JSON.parse(JSON.stringify(cfg));
        for (const input of form.querySelectorAll('[data-path]')) setPath(next, input.dataset.path, readField(input));
        next.compaction.summary_model = sameRadio.checked ? '' : summaryModel.value.trim();
        if (otherRadio.checked && !next.compaction.summary_model) throw new Error('Enter the summariser model id, or choose "Same as chat model".');
        next.enabled_tools = toolBoxes.filter(b => b.checked).map(b => b.value);
      } catch (error) { alertBox.replaceChildren(h('div', {class: 'pi-alert', text: error.message})); return; }
      status(line, 'Saving…', 'busy');
      try {
        const data = await req(`${API}/profiles/${encodeURIComponent(p.id)}`, {method: 'PATCH',
          body: {name: name.value, description: description.value, config: next}});
        acceptProfiles(data);
        status(line, 'Saved. Existing chats keep their own settings.', 'ok');
        refreshSlot();
      } catch (error) {
        status(line, '');
        alertBox.replaceChildren(h('div', {class: 'pi-alert', text: 'Not saved: ' + error.message}));
      }
    };
    const head = h('div', {class: 'pi-row'},
      h('button', {type: 'button', class: 'pi-btn', text: '← All profiles', onclick: back}),
      h('h3', {class: 'pi-grow', text: (ro ? 'Built-in profile: ' : 'Edit profile: ') + p.name}));
    const footer = h('div', {class: 'pi-actions'},
      ro ? h('button', {type: 'button', class: 'pi-btn primary', text: 'Duplicate to edit', onclick: async () => {
        try {
          const data = await req(`${API}/profiles/${encodeURIComponent(p.id)}/duplicate`, {method: 'POST', body: {}});
          acceptProfiles(data); st.editing = data.profile.id; await renderProfiles(st.panel);
        } catch (error) { status(line, error.message, 'err'); }
      }}) : h('button', {type: 'button', class: 'pi-btn primary', text: 'Save profile', onclick: save}),
      h('button', {type: 'button', class: 'pi-btn', text: ro ? 'Back' : 'Cancel', onclick: back}));
    panel.replaceChildren(...[head,
      ro ? h('div', {class: 'pi-note', text: 'Built-in profiles are read-only starters. Duplicate one to change it.'}) : null,
      alertBox, form, footer, line].filter(Boolean));
    (ro ? footer.querySelector('button') : name).focus();
  }

  async function setPrefs(patch) {
    const data = await req(API + '/prefs', {method: 'PATCH', body: {prefs: patch}});
    st.prefs = data.prefs;
    if ('default_profile_id' in patch) await loadProfiles();
    return data.prefs;
  }

  /* ------------------------------------------------------------ new chats */
  async function renderNewChat(panel) {
    await Promise.all([loadProfiles(), loadOverview()]);
    const prefs = st.prefs;
    const line = statusLine();
    const select = h('select', {id: 'pi-default-profile'},
      h('option', {value: '', text: 'No profile: saved new-chat template / built-in defaults', selected: !st.defaultProfileId}),
      st.profiles.filter(p => p.effective).map(p => h('option', {value: p.id, text: p.name + (p.builtin ? ' (built-in)' : ''), selected: p.id === st.defaultProfileId})));
    const inherit = h('input', {type: 'radio', name: 'pi-newchat-mode', value: 'inherit', checked: prefs.new_chat_mode !== 'profile'});
    const fromProfile = h('input', {type: 'radio', name: 'pi-newchat-mode', value: 'profile', checked: prefs.new_chat_mode === 'profile'});
    const workdir = h('input', {type: 'text', id: 'pi-default-workdir', value: prefs.default_workdir || '', spellcheck: 'false', placeholder: 'C:\\Users\\me\\Projects'});
    const chatDir = st.chat && st.chat.config && (st.chat.config.workdirs || [])[0];
    panel.replaceChildren(
      h('h3', {text: 'New chats'}),
      h('div', {class: 'pi-field'}, h('label', {for: 'pi-default-profile', text: 'Default profile'}), select,
        h('small', {text: 'Used for new chats without a source chat, and by "New chat" when chosen below.'})),
      h('div', {class: 'pi-field', role: 'radiogroup', 'aria-labelledby': 'pi-newchat-mode-label'},
        h('span', {class: 'pi-label', id: 'pi-newchat-mode-label', text: 'The New chat button'}),
        h('label', {class: 'pi-check'}, inherit, h('span', {text: 'copies the settings of the chat that is open (as before)'})),
        h('label', {class: 'pi-check'}, fromProfile, h('span', {text: 'starts from the default profile'}))),
      h('div', {class: 'pi-field'}, h('label', {for: 'pi-default-workdir', text: 'Default working directory'}),
        h('div', {class: 'pi-row'}, h('div', {class: 'pi-grow'}, workdir),
          chatDir ? h('button', {type: 'button', class: 'pi-btn', text: 'Use this chat\'s folder', onclick: () => { workdir.value = chatDir; }}) : null),
        h('small', {text: 'Given to new chats that have no working directory yet. Leave empty for none.'})),
      h('div', {class: 'pi-actions'}, h('button', {type: 'button', class: 'pi-btn primary', text: 'Save', onclick: async () => {
        try {
          await setPrefs({default_profile_id: select.value, new_chat_mode: fromProfile.checked ? 'profile' : 'inherit', default_workdir: workdir.value.trim()});
          status(line, 'Saved.', 'ok');
        } catch (error) { status(line, error.message, 'err'); }
      }})), line);
  }

  /* ------------------------------------------------------------ appearance */
  function themeNames() {
    try {
      const list = window.PiTheme && window.PiTheme.list ? window.PiTheme.list() : [];
      return (list || []).map(t => (typeof t === 'string' ? {name: t, label: t} : {name: t.name || t.id, label: t.label || t.name || t.id}));
    } catch (_) { return []; }
  }

  function applyZoom(zoom) {
    const value = Number(zoom) || 100;
    document.documentElement.style.zoom = value === 100 ? '' : String(value / 100);
    document.documentElement.style.setProperty('--pi-zoom', String(value / 100));
  }

  function applyAppearance(prefs) {
    applyZoom(prefs.zoom);
    if (prefs.theme && window.PiTheme && typeof window.PiTheme.apply === 'function') {
      try { if (!window.PiTheme.current || window.PiTheme.current() !== prefs.theme) window.PiTheme.apply(prefs.theme); } catch (_) { /* unknown theme */ }
    }
  }

  let prefTimer = null;
  function savePrefSoon(patch, line) {
    clearTimeout(prefTimer);
    prefTimer = setTimeout(() => setPrefs(patch).then(() => status(line, 'Saved.', 'ok')).catch(e => status(line, e.message, 'err')), 300);
  }

  async function renderAppearance(panel) {
    await loadOverview();
    const line = statusLine();
    const themes = themeNames();
    let current = st.prefs.theme || '';
    try { if (window.PiTheme && window.PiTheme.current) current = window.PiTheme.current() || current; } catch (_) { /* ignore */ }
    const theme = h('select', {id: 'pi-theme', disabled: !themes.length},
      themes.length ? themes.map(t => h('option', {value: t.name, text: t.label, selected: t.name === current}))
                    : h('option', {value: '', text: 'Default theme'}));
    theme.addEventListener('change', () => {
      try { window.PiTheme.apply(theme.value); } catch (_) { /* ignore */ }
      savePrefSoon({theme: theme.value}, line);
    });
    const zoomValue = h('output', {for: 'pi-zoom', text: (st.prefs.zoom || 100) + '%'});
    const zoom = h('input', {type: 'range', id: 'pi-zoom', min: '80', max: '150', step: '5', value: String(st.prefs.zoom || 100)});
    zoom.addEventListener('input', () => { zoomValue.textContent = zoom.value + '%'; applyZoom(zoom.value); savePrefSoon({zoom: Number(zoom.value)}, line); });
    panel.replaceChildren(
      h('h3', {text: 'Appearance'}),
      h('div', {class: 'pi-field'}, h('label', {for: 'pi-theme', text: 'Theme'}), theme,
        themes.length ? null : h('small', {text: 'No theme switcher is available in this window.'})),
      h('div', {class: 'pi-field'}, h('label', {for: 'pi-zoom'}, 'Text and interface size ', zoomValue), zoom,
        h('div', {class: 'pi-actions'}, h('button', {type: 'button', class: 'pi-btn', text: 'Reset to 100%', onclick: () => {
          zoom.value = '100'; zoom.dispatchEvent(new Event('input'));
        }}))),
      line);
  }

  /* ------------------------------------------------------------ insights */
  async function renderInsights(panel) {
    const info = await req(API + '/insights');
    const line = statusLine();
    const mb = n => (n / 1048576).toFixed(n < 10485760 ? 1 : 0) + ' MB';
    const summary = h('p', {class: 'pi-muted', id: 'pi-insights-summary'});
    const describe = d => {
      summary.textContent = d.chats ? `${d.chats} chat${d.chats === 1 ? '' : 's'} recorded, ${mb(d.bytes)}.` : 'Nothing recorded yet.';
    };
    describe(info);
    const box = h('input', {type: 'checkbox', id: 'pi-insights', checked: !!info.enabled});
    box.addEventListener('change', () => setPrefs({insights: box.checked})
      .then(() => status(line, box.checked ? 'Recording from the next request on.' : 'Recording stopped.', 'ok'))
      .catch(e => { box.checked = !box.checked; status(line, e.message, 'err'); }));
    const build = h('button', {type: 'button', class: 'pi-btn', id: 'pi-insights-report', text: 'Build and open report'});
    build.addEventListener('click', async () => {
      build.disabled = true;
      status(line, 'Building the report…');
      try {
        const r = await req(API + '/insights/report', {method: 'POST', body: {open: true}});
        describe(r);
        status(line, r.chats ? 'Opened ' + r.html : 'Nothing recorded yet.', r.chats ? 'ok' : '');
      } catch (e) { status(line, e.message, 'err'); }
      build.disabled = false;
    });
    const openFolder = () => req(API + '/open-folder', {method: 'POST', body: {which: 'insights'}})
      .then(r => status(line, 'Opened ' + r.path, 'ok')).catch(e => status(line, e.message, 'err'));
    panel.replaceChildren(
      h('h3', {text: 'Insights'}),
      h('label', {class: 'pi-check'}, box, h('span', {}, 'Record usage insights ',
        h('small', {class: 'pi-muted', text: '(off by default)'}))),
      h('p', {class: 'pi-muted', text: 'Records, for every chat, how the context window grows and shrinks, what the chat was doing '
        + 'when it compacted or curated its context, every tool call (name, outcome, error class, result size and duration; '
        + 'paths and flags, file contents and edit text only as sizes) and the tokens and cost of every request. '
        + 'Short excerpts of your messages, the answers and compaction summaries are kept so the moments can be read later. '
        + 'Everything stays in the folder below on this computer.'}),
      summary,
      h('div', {class: 'pi-field'}, h('span', {class: 'pi-label', text: 'Insights folder'}),
        h('div', {class: 'pi-row'}, h('code', {class: 'pi-grow', text: info.folder}),
          h('button', {type: 'button', class: 'pi-btn', text: 'Open', onclick: openFolder}))),
      h('div', {class: 'pi-actions'}, build),
      h('p', {class: 'pi-muted', text: 'The report shows the context window over time per chat, the compaction and curation moments, '
        + 'token use per model and turn, and statistics per tool, with sections for editing and inspection and, per model, '
        + 'the tools that fail most. It is written as report.html and report.md into the insights folder.'}),
      line);
  }

  /* ------------------------------------------------------------ about */
  async function renderAbout(panel) {
    const o = await loadOverview();
    const line = statusLine();
    const openFolder = which => req(API + '/open-folder', {method: 'POST', body: {which}})
      .then(r => status(line, 'Opened ' + r.path, 'ok')).catch(e => status(line, e.message, 'err'));
    panel.replaceChildren(
      h('h3', {text: o.product || 'CTXAgent'}),
      h('p', {}, 'Version ', h('b', {text: o.version}), h('span', {class: 'pi-muted', text: ` · backend ${o.backend_build}`})),
      h('div', {class: 'pi-field'}, h('span', {class: 'pi-label', text: 'Data folder (chats, profiles, settings)'}),
        h('div', {class: 'pi-row'}, h('code', {class: 'pi-grow', text: o.data_dir}),
          h('button', {type: 'button', class: 'pi-btn', text: 'Open', onclick: () => openFolder('data')}))),
      h('div', {class: 'pi-field'}, h('span', {class: 'pi-label', text: 'Logs folder'}),
        h('div', {class: 'pi-row'}, h('code', {class: 'pi-grow', text: o.logs_dir}),
          h('button', {type: 'button', class: 'pi-btn', text: 'Open logs folder', onclick: () => openFolder('logs')}))),
      h('div', {class: 'pi-field'}, h('span', {class: 'pi-label', text: 'API keys stored in'}), h('div', {text: o.keystore.label})),
      line);
  }

  /* ------------------------------------------------------------ profile slot */
  let slot = null, slotButton = null, menu = null;

  function mountSlot() {
    const host = document.getElementById('profile-slot');
    if (!host || slot) return;
    host.classList.add('pi-profile-slot');
    slotButton = h('button', {type: 'button', class: 'pi-profile-btn', 'aria-haspopup': 'menu', 'aria-expanded': 'false', hidden: true,
      onclick: () => (menu.hidden ? openMenu() : closeMenu())});
    menu = h('div', {class: 'pi-profile-menu', role: 'menu', 'aria-label': 'Chat profile', hidden: true});
    menu.addEventListener('keydown', e => {
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeMenu(true); return; }
      if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
      if (e.target.matches('input, select')) return;
      const items = [...menu.querySelectorAll('[role="menuitem"]:not(:disabled)')];
      const i = items.indexOf(document.activeElement);
      e.preventDefault();
      const next = items[(i + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length];
      if (next) next.focus();
    });
    document.addEventListener('mousedown', e => { if (menu && !menu.hidden && !host.contains(e.target)) closeMenu(); });
    host.replaceChildren(slotButton, menu);
    slot = host;
  }

  function renderSlot() {
    mountSlot();
    if (!slot) return;
    const s = st.chatStatus;
    if (!st.chat || !s) { slotButton.hidden = true; closeMenu(); return; }
    slotButton.hidden = false;
    const name = s.profile ? s.profile.name : (s.missing ? 'deleted profile' : 'No profile');
    slotButton.replaceChildren(...[h('span', {class: 'pi-profile-label', text: 'Profile'}), h('span', {class: 'pi-profile-name', text: name}),
      s.modified ? h('span', {class: 'pi-profile-mod', text: '(modified)'}) : null,
      s.missing ? h('span', {class: 'pi-profile-missing', text: '(deleted)'}) : null].filter(Boolean));
    slotButton.title = s.modified ? 'This chat differs from its profile in: ' + s.differences.join(', ') : 'Chat settings profile';
    slotButton.setAttribute('aria-label', `Profile: ${name}${s.modified ? ' (modified)' : ''}${s.missing ? ' (deleted)' : ''}`);
  }

  async function refreshSlot(chatId) {
    const id = chatId || (st.chat && st.chat.id);
    if (!id) { renderSlot(); return; }
    const seq = ++st.slotSeq;
    try {
      const s = await req(`${API}/chats/${encodeURIComponent(id)}/profile`);
      if (seq !== st.slotSeq || !st.chat || st.chat.id !== id) return;
      st.chatStatus = s;
    } catch (_) {
      if (seq !== st.slotSeq) return;
      st.chatStatus = null;
    }
    renderSlot();
    if (menu && !menu.hidden) buildMenu();
  }

  function closeMenu(focusButton) {
    if (!menu || menu.hidden) return;
    menu.hidden = true;
    slotButton.setAttribute('aria-expanded', 'false');
    if (focusButton) slotButton.focus();
  }

  async function openMenu() {
    if (!st.chatStatus) return;
    try { await loadProfiles(); } catch (_) { /* keep stale list */ }
    buildMenu();
    menu.hidden = false;
    slotButton.setAttribute('aria-expanded', 'true');
    const first = menu.querySelector('[role="menuitem"]:not(:disabled)');
    if (first) first.focus();
  }

  function chatChanged(chatId) {
    window.dispatchEvent(new CustomEvent('pi:chat-config-changed', {detail: {chatId}}));
    refreshSlot(chatId);
  }

  function buildMenu() {
    const s = st.chatStatus, chat = st.chat;
    if (!s || !chat) return;
    const line = statusLine();
    const inline = h('div', {class: 'pi-inline'});
    const p = s.profile;
    const busy = !!chat.running, worker = !!chat.delegated;
    const item = (text, onclick, disabled) => h('button', {type: 'button', role: 'menuitem', text, disabled, onclick});
    const head = h('div', {class: 'pi-menu-head'},
      h('div', {}, h('b', {text: p ? p.name : (s.missing ? 'Profile deleted' : 'No profile')}),
        p && p.builtin ? h('span', {class: 'pi-badge', text: 'built-in'}) : null),
      p && p.description ? h('div', {class: 'pi-muted pi-small', text: p.description}) : null,
      s.modified ? h('div', {class: 'pi-small'}, h('span', {class: 'pi-profile-mod', text: 'Modified: '}),
        s.differences.slice(0, 8).join(', ') + (s.differences.length > 8 ? ` and ${s.differences.length - 8} more` : '')) : null,
      !p && !s.missing ? h('div', {class: 'pi-muted pi-small', text: 'This chat was not created from a profile.'}) : null,
      worker ? h('div', {class: 'pi-muted pi-small', text: 'Worker-owned chat: settings come from the parent agent.'}) : null);

    const reset = item('Reset chat to profile', async () => {
      if (!await confirmInline(inline, `Replace this chat's settings with "${p.name}"? History, usage and working directories are kept.`, 'Reset')) return;
      try {
        await req(`${API}/chats/${encodeURIComponent(chat.id)}/profile/reset`, {method: 'POST', body: {}});
        closeMenu(true); chatChanged(chat.id);
      } catch (error) { status(line, error.message, 'err'); }
    }, !p || !s.modified || busy || worker);
    const saveInto = item('Save chat settings into profile', async () => {
      if (!await confirmInline(inline, `Overwrite "${p.name}" with this chat's saved settings? Other chats keep their own settings.`, 'Overwrite')) return;
      try {
        await req(`${API}/chats/${encodeURIComponent(chat.id)}/profile/save`, {method: 'POST'});
        await refreshSlot(chat.id); status(line, 'Profile updated.', 'ok');
      } catch (error) { status(line, error.message, 'err'); }
    }, !p || p.builtin || !s.modified);
    if (p && p.builtin) saveInto.title = 'Built-in profiles are read-only: save as a new profile instead.';
    const saveAs = item('Save as new profile…', () => {
      inline.replaceChildren();
      const nameInput = h('input', {type: 'text', 'aria-label': 'New profile name', value: (p ? p.name + ' (custom)' : (chat.title || 'Chat') + ' settings'), maxlength: '80'});
      const go = async () => {
        try {
          const r = await req(`${API}/chats/${encodeURIComponent(chat.id)}/profile/save-as`, {method: 'POST', body: {name: nameInput.value}});
          acceptProfiles(r);
          if (r.linked) { closeMenu(true); chatChanged(chat.id); } else { await refreshSlot(chat.id); status(line, 'Profile created; the chat is busy, so it was not linked.', 'busy'); }
        } catch (error) { status(line, error.message, 'err'); }
      };
      nameInput.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); go(); } });
      inline.append(h('label', {class: 'pi-small', text: 'Name'}), nameInput, h('div', {class: 'pi-actions'},
        h('button', {type: 'button', class: 'pi-btn primary', text: 'Create', onclick: go}),
        h('button', {type: 'button', class: 'pi-btn', text: 'Cancel', onclick: () => inline.replaceChildren()})));
      nameInput.select(); nameInput.focus();
    });
    const switchTo = item('Apply another profile…', () => {
      inline.replaceChildren();
      const pick = h('select', {'aria-label': 'Profile to apply'}, st.profiles.filter(x => x.effective)
        .map(x => h('option', {value: x.id, text: x.name + (x.builtin ? ' (built-in)' : ''), selected: p && x.id === p.id})));
      inline.append(pick, h('p', {class: 'pi-muted pi-small', text: 'Replaces this chat\'s settings; history, usage and working directories are kept.'}),
        h('div', {class: 'pi-actions'},
          h('button', {type: 'button', class: 'pi-btn primary', text: 'Apply', onclick: async () => {
            try {
              await req(`${API}/chats/${encodeURIComponent(chat.id)}/profile/reset`, {method: 'POST', body: {profile_id: pick.value}});
              closeMenu(true); chatChanged(chat.id);
            } catch (error) { status(line, error.message, 'err'); }
          }}),
          h('button', {type: 'button', class: 'pi-btn', text: 'Cancel', onclick: () => inline.replaceChildren()})));
      pick.focus();
    }, busy || worker);
    const newChat = item('New chat from this profile', () => { closeMenu(); newChatFromProfile(p.id); }, !p);
    const manage = item('Manage profiles…', () => { closeMenu(); open('profiles'); });
    menu.replaceChildren(head, reset, saveInto, saveAs, switchTo, newChat, manage, inline, line);
  }

  /* ------------------------------------------------------------ wiring */
  function observeConfigSaves() {
    // Per-chat Settings/Engine/Tools panels save with PATCH /api/chats/{id}/config;
    // refresh the "(modified)" state afterwards. Observation only.
    const original = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      const response = await original(input, init);
      try {
        const url = typeof input === 'string' ? input : input.url;
        const method = ((init && init.method) || (typeof input === 'object' && input.method) || 'GET').toUpperCase();
        const m = /\/api\/chats\/([^/?#]+)\/(config|defaults)(?:[?#]|$)/.exec(url || '');
        if (m && method !== 'GET' && response.ok && st.chat && decodeURIComponent(m[1]) === st.chat.id) setTimeout(() => refreshSlot(st.chat.id), 0);
      } catch (_) { /* never break the app's request */ }
      return response;
    };
  }

  function augmentBanner() {
    const banner = document.getElementById('banner');
    if (!banner) return;
    const addButton = () => {
      if (!/API_KEY/.test(banner.textContent || '') || banner.querySelector('.pi-banner-action')) return;
      const target = banner.firstElementChild || banner;
      target.append(h('button', {type: 'button', class: 'pi-btn primary pi-banner-action', text: 'Add a key in Settings', onclick: () => open('keys')}));
    };
    new MutationObserver(addButton).observe(banner, {childList: true, subtree: true});
    addButton();
  }

  function init() {
    mountSlot();
    const opener = document.getElementById('open-settings');
    if (opener) opener.addEventListener('click', () => open());
    const onChat = chat => {
      st.chat = chat && chat.id ? chat : null;
      st.chatStatus = null;
      renderSlot();
      if (st.chat) refreshSlot(st.chat.id);
    };
    window.addEventListener('pi:chat-loaded', e => onChat(e.detail || null));
    // A chat that loaded before this deferred script ran.
    if (window.PiCurrentChat) onChat(window.PiCurrentChat);
    observeConfigSaves();
    augmentBanner();
    loadOverview().then(o => applyAppearance(o.prefs || {})).catch(() => { /* backend not ready */ });
    window.PiSettings = {open, close, refreshSlot: () => refreshSlot()};
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
