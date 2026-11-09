/* CTXAgent theme registry. Loaded synchronously in <head>, before any
   stylesheet paints, so the saved theme is applied without a flash.

   window.PiTheme
     list()       -> [{id, label, dark}]   ('system' reports the OS preference)
     current()    -> the chosen id, possibly 'system'
     resolved()   -> the concrete theme id in effect ('system' -> 'light'/'dark')
     apply(name)  -> persists the choice, sets <html data-theme>, returns the
                     resolved id; unknown names are ignored (returns null)
   Fires window 'pi:theme-changed' {detail: {id, resolved}} after each change. */
(function () {
  'use strict';
  var KEY = 'pi.theme';
  var THEMES = [
    {id: 'system', label: 'System', dark: null},
    {id: 'light', label: 'Light', dark: false},
    {id: 'dark', label: 'Dark', dark: true},
    {id: 'contrast', label: 'High contrast', dark: true},
    {id: 'paper', label: 'Paper', dark: false},
    {id: 'nord', label: 'Nord', dark: true},
    {id: 'ember', label: 'Ember', dark: true},
  ];
  var media = null;
  try { media = window.matchMedia('(prefers-color-scheme: dark)'); } catch (_) {}

  function find(id) {
    for (var i = 0; i < THEMES.length; i++) if (THEMES[i].id === id) return THEMES[i];
    return null;
  }
  function systemDark() { return !!(media && media.matches); }
  function load() {
    try { return localStorage.getItem(KEY); } catch (_) { return null; }
  }
  function save(id) {
    try { localStorage.setItem(KEY, id); } catch (_) {}
  }

  var choice = find(load()) ? load() : 'system';

  function resolve(id) { return id === 'system' ? (systemDark() ? 'dark' : 'light') : id; }

  function paint(notify) {
    var root = document.documentElement, resolved = resolve(choice);
    root.setAttribute('data-theme', resolved);
    root.setAttribute('data-theme-choice', choice);
    if (notify) {
      try {
        window.dispatchEvent(new CustomEvent('pi:theme-changed', {detail: {id: choice, resolved: resolved}}));
      } catch (_) {}
    }
    return resolved;
  }

  function apply(name) {
    if (!find(name)) return null;
    choice = name;
    save(name);
    return paint(true);
  }

  function list() {
    return THEMES.map(function (t) {
      return {id: t.id, label: t.label, dark: t.dark === null ? systemDark() : t.dark};
    });
  }

  if (media) {
    var follow = function () { if (choice === 'system') paint(true); };
    if (media.addEventListener) media.addEventListener('change', follow);
    else if (media.addListener) media.addListener(follow);
  }
  // Another window of the same app changed the theme.
  window.addEventListener('storage', function (event) {
    if (event.key !== KEY) return;
    choice = find(event.newValue) ? event.newValue : 'system';
    paint(true);
  });

  window.PiTheme = Object.freeze({
    list: list,
    current: function () { return choice; },
    resolved: function () { return resolve(choice); },
    apply: apply,
  });
  paint(false);
})();
