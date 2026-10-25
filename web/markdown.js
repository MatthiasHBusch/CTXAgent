/* Chat Markdown: local CommonMark/GFM parser, no source HTML or automatic remote media.
 * Vendored dependency/provenance: vendor/README.md. No runtime CDN requests. */
(() => {
  'use strict';
  const escape = value => String(value).replace(/[&<>"']/g, c => ({
    '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
  }[c]));
  const safeURL = value => {
    const url = String(value || '').trim();
    // Relative paths resolve against this application, not the author's machine/source.
    // Allow only explicit public web/mail links; no javascript/data/file/custom schemes.
    if (!/^(https?:\/\/|mailto:)/i.test(url) || /[\u0000-\u0020\u007f]/.test(url)) return false;
    try {
      const parsed = new URL(url);
      return ['http:', 'https:', 'mailto:'].includes(parsed.protocol) &&
        (parsed.protocol === 'mailto:' || (!!parsed.hostname && !parsed.username && !parsed.password));
    } catch (_) { return false; }
  };
  const parser = typeof window.markdownit === 'function' ? window.markdownit({
    html: false, breaks: true, linkify: true, typographer: false, maxNesting: 50
  }) : null;

  if (parser) {
    // Keep the parser's maintained dangerous-scheme validator. Apply the stricter
    // app policy at render time, so rejected credential/relative links are not
    // reparsed by linkify as misleading partial URLs.
    parser.renderer.rules.link_open = (tokens, index, options, env, renderer) => {
      const token = tokens[index], allowed = safeURL(token.attrGet('href'));
      (env.chatLinks ||= []).push(allowed);
      if (!allowed) return '<span class="markdown-disabled-link">';
      token.attrSet('target', '_blank');
      token.attrSet('rel', 'noopener noreferrer');
      token.attrSet('referrerpolicy', 'no-referrer');
      return renderer.renderToken(tokens, index, options);
    };
    parser.renderer.rules.link_close = (_tokens, _index, _options, env) => env.chatLinks?.pop() ? '</a>' : '</span>';
    // Loading model-supplied remote images would silently disclose the user's IP
    // and could carry tracking/query data. Keep alt text and an explicit safe link.
    parser.renderer.rules.image = (tokens, index, _options, env) => {
      const token = tokens[index], src = token.attrGet('src');
      const alt = token.content || 'Image';
      const label = escape(`[Image: ${alt}]`);
      return safeURL(src) && !env.chatLinks?.includes(true) ? `<a class="markdown-image-link" href="${escape(src)}" target="_blank" rel="noopener noreferrer" referrerpolicy="no-referrer">${label}</a>` : label;
    };
    parser.renderer.rules.table_open = () => '<div class="markdown-table" role="region" aria-label="Markdown table" tabindex="0"><table>\n';
    parser.renderer.rules.table_close = () => '</table></div>\n';
    parser.renderer.rules.task_checkbox = (tokens, index) =>
      `<input type="checkbox" class="markdown-task" disabled tabindex="-1" aria-label="${tokens[index].meta.checked ? 'Completed' : 'Incomplete'} task"${tokens[index].meta.checked ? ' checked' : ''}> `;
    // Task markers are recognized only as literal text at a parsed list item's
    // beginning, never by substituting HTML into source strings/code blocks.
    parser.core.ruler.after('inline', 'chat_task_lists', state => {
      for (let i = 2; i < state.tokens.length; i++) {
        const inline = state.tokens[i];
        if (inline.type !== 'inline' || state.tokens[i - 1].type !== 'paragraph_open' ||
            state.tokens[i - 2].type !== 'list_item_open' || inline.children?.[0]?.type !== 'text') continue;
        const first = inline.children[0], marker = /^\[([ xX])\][ \t]+/.exec(first.content);
        if (!marker) continue;
        first.content = first.content.slice(marker[0].length);
        const checkbox = new state.Token('task_checkbox', 'input', 0);
        checkbox.meta = {checked: marker[1].toLowerCase() === 'x'};
        inline.children.unshift(checkbox);
        state.tokens[i - 2].attrJoin('class', 'markdown-task-item');
      }
    });
  }

  // Chat prose often spaces every explicit pipe row with a blank line. Repair
  // only unambiguous top-level pipe tables with a real delimiter row. Do not
  // change source/history, fenced code, indented code, or separated prose.
  function pipeCells(line) {
    if (/^(?: {4}|\t)/.test(line)) return null;
    const text = line.trim();
    if (!text.startsWith('|') || !text.endsWith('|') || text.length < 3) return null;
    const cells = []; let start = 1, code = 0;
    for (let i = 1; i < text.length; i++) {
      if (text[i] === '\\') { i++; continue; }
      if (text[i] === '`') {
        let end = i + 1;
        while (text[end] === '`') end++;
        const width = end - i;
        if (!code) code = width; else if (width === code) code = 0;
        i = end - 1; continue;
      }
      if (text[i] === '|' && !code) { cells.push(text.slice(start, i).trim()); start = i + 1; }
    }
    return start === text.length && cells.length ? cells : null;
  }

  function normalizeTables(source) {
    source = source.replace(/\r\n?/g, '\n');
    if (!/\|[^\n]*\n[ \t]*\n/.test(source)) return source;
    const lines = source.split('\n'), protectedLines = new Set();
    // Use the parser's own block ranges, including nested/unfinished fences and
    // indented code. Guessing fences with regex corrupts literal Markdown.
    for (const token of parser.parse(source, {})) {
      if ((token.type === 'fence' || token.type === 'code_block') && token.map) {
        for (let line = token.map[0]; line < token.map[1]; line++) protectedLines.add(line);
      }
    }
    const out = [];
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (protectedLines.has(i)) { out.push(line); continue; }
      const header = pipeCells(line);
      let sep = i + 1;
      while (sep < lines.length && !lines[sep].trim()) sep++;
      const delimiter = sep < lines.length ? pipeCells(lines[sep]) : null;
      if (!header || !delimiter || delimiter.length !== header.length || !delimiter.every(c => /^:?-{3,}:?$/.test(c))) {
        out.push(line); continue;
      }
      out.push(line, lines[sep]); i = sep;
      while (i + 1 < lines.length) {
        let next = i + 1;
        while (next < lines.length && !lines[next].trim()) next++;
        const row = next < lines.length ? pipeCells(lines[next]) : null;
        if (!row || row.length !== header.length) break;
        // A following delimiter starts a distinct table; preserve its separation.
        let after = next + 1;
        while (after < lines.length && !lines[after].trim()) after++;
        const nextDelimiter = after < lines.length ? pipeCells(lines[after]) : null;
        if (nextDelimiter?.length === header.length && nextDelimiter.every(c => /^:?-{3,}:?$/.test(c))) break;
        out.push(lines[next]); i = next;
      }
    }
    return out.join('\n');
  }

  function render(source) {
    const text = String(source ?? '');
    if (!parser) return `<pre class="markdown-fallback">${escape(text)}</pre>`;
    return parser.render(normalizeTables(text));
  }
  window.ChatMarkdown = Object.freeze({render, version: 'markdown-it-15.0.2'});
})();
