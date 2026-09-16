/* Settings: what the agent may do, and what the keys do.
 *
 * Two halves, deliberately kept apart because they live in different places.
 *
 *   FEATURES are the agent's, and live in config.json — the same file the
 *   CLI and the daemon read. Switching "Shell commands" off here is the same
 *   act as `/config set tools.enabled_groups`, so the server writes only the
 *   handful of keys it names and nothing else in that file is reachable from
 *   the browser.
 *
 *   KEYBINDS are this window's, and live in localStorage. They rebind what
 *   the page itself does — send, new chat, the drawer, the theme — and they
 *   are per-browser, because they are a property of how someone types, not
 *   of the agent.
 */

const KEYS_STORE = 'symbio.keybinds.v1';

const ACTIONS = [
  { id: 'send', label: 'Send message', fallback: 'Enter', inComposer: true },
  { id: 'newline', label: 'New line', fallback: 'Shift+Enter', inComposer: true, fixed: true },
  { id: 'newChat', label: 'New chat', fallback: 'Mod+N' },
  { id: 'focusComposer', label: 'Focus the composer', fallback: 'Mod+L' },
  { id: 'toggleSidebar', label: 'Show or hide the sidebar', fallback: 'Mod+B' },
  { id: 'toggleWorkspace', label: 'Open or close the workspace', fallback: 'Mod+J' },
  { id: 'toggleTheme', label: 'Light or dark', fallback: 'Mod+Shift+L' },
  { id: 'openSettings', label: 'Settings', fallback: 'Mod+,' },
];

let keybinds = {};

function loadKeybinds() {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(KEYS_STORE)) || {};
  } catch (e) { /* blocked storage: the defaults below still work */ }
  keybinds = {};
  for (const action of ACTIONS) {
    keybinds[action.id] = saved[action.id] || action.fallback;
  }
}

function saveKeybinds() {
  try {
    localStorage.setItem(KEYS_STORE, JSON.stringify(keybinds));
  } catch (e) { /* the binds still hold for this page load */ }
}

/* A keyboard event as the same string a binding is written in.
 * "Mod" is Command on a Mac and Control everywhere else — one binding that
 * reads correctly on both, rather than two tables to keep in step. */
function comboOf(event) {
  const parts = [];
  if (event.metaKey || event.ctrlKey) parts.push('Mod');
  if (event.shiftKey) parts.push('Shift');
  if (event.altKey) parts.push('Alt');
  let key = event.key;
  if (key === ' ') key = 'Space';
  else if (key.length === 1) key = key.toUpperCase();
  if (['Meta', 'Control', 'Shift', 'Alt'].includes(key)) return null;  // modifier alone
  parts.push(key);
  return parts.join('+');
}

function matches(event, binding) {
  const combo = comboOf(event);
  return Boolean(combo) && combo.toLowerCase() === String(binding || '').toLowerCase();
}

function prettyCombo(combo) {
  const mac = navigator.platform.toLowerCase().includes('mac');
  return String(combo || '')
    .replace('Mod', mac ? '⌘' : 'Ctrl')
    .replace('Shift', mac ? '⇧' : 'Shift')
    .replace('Alt', mac ? '⌥' : 'Alt')
    .replace(/\+/g, mac ? '' : '+');
}

// ── the panel ───────────────────────────────────────────────────────

let settingsData = null;
let capturing = null;          // the action id currently listening for a key

async function openSettings() {
  document.getElementById('settings').hidden = false;
  document.body.classList.add('modal-open');
  renderSettings();             // keybinds render immediately, offline
  try {
    const res = await fetch('/api/settings');
    settingsData = await res.json();
  } catch (e) {
    settingsData = { error: 'Could not read the settings file.' };
  }
  renderSettings();
}

function closeSettings() {
  capturing = null;
  document.getElementById('settings').hidden = true;
  document.body.classList.remove('modal-open');
}

function renderSettings() {
  const body = document.getElementById('settings-body');
  const rows = [];

  rows.push('<h3 class="settings-heading">What Symbio may do</h3>');
  if (!settingsData) {
    rows.push('<p class="settings-note">Reading settings…</p>');
  } else if (settingsData.error) {
    rows.push(`<p class="settings-note">${escHtml(settingsData.error)}</p>`);
  } else {
    for (const group of settingsData.groups) {
      rows.push(toggleRow('groups', group));
    }
    rows.push('<h3 class="settings-heading">Behaviour</h3>');
    for (const feature of settingsData.features) {
      rows.push(toggleRow('features', feature));
    }
    rows.push(`<p class="settings-note">${escHtml(settingsData.restart_note)}</p>`);
  }

  rows.push('<h3 class="settings-heading">Keyboard</h3>');
  for (const action of ACTIONS) {
    const listening = capturing === action.id;
    rows.push(`<div class="settings-row">
      <div class="settings-label">
        <span>${escHtml(action.label)}</span>
        ${action.inComposer ? '<span class="settings-hint">in the composer</span>' : ''}
      </div>
      ${action.fixed
        ? `<span class="key-chip fixed">${escHtml(prettyCombo(keybinds[action.id]))}</span>`
        : `<button class="key-chip${listening ? ' listening' : ''}" data-bind="${action.id}">${
            listening ? 'press keys…' : escHtml(prettyCombo(keybinds[action.id]))}</button>`}
    </div>`);
  }
  rows.push('<div class="settings-row"><button class="link-btn" id="reset-keys">Reset keys to defaults</button></div>');

  body.innerHTML = rows.join('');

  body.querySelectorAll('[data-toggle]').forEach(el => {
    el.addEventListener('click', () => flip(el.dataset.section, el.dataset.toggle));
  });
  body.querySelectorAll('[data-bind]').forEach(el => {
    el.addEventListener('click', () => { capturing = el.dataset.bind; renderSettings(); });
  });
  const reset = body.querySelector('#reset-keys');
  if (reset) reset.addEventListener('click', () => {
    for (const action of ACTIONS) keybinds[action.id] = action.fallback;
    saveKeybinds();
    renderSettings();
  });
}

function toggleRow(section, item) {
  return `<div class="settings-row">
    <div class="settings-label">
      <span>${escHtml(item.label)}</span>
      <span class="settings-hint">${escHtml(item.hint)}</span>
    </div>
    <button class="switch${item.on ? ' on' : ''}" data-section="${section}"
            data-toggle="${escHtml(item.key)}" role="switch"
            aria-checked="${item.on ? 'true' : 'false'}"><span></span></button>
  </div>`;
}

async function flip(section, key) {
  if (!settingsData || settingsData.error) return;
  const list = settingsData[section] || [];
  const item = list.find(i => i.key === key);
  if (!item) return;
  item.on = !item.on;          // optimistic: the row moves under the finger
  renderSettings();
  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [section]: { [key]: item.on } }),
    });
    const out = await res.json();
    if (!out.ok) throw new Error(out.error || 'refused');
  } catch (e) {
    // Put it back rather than showing a switch that lies about the file.
    item.on = !item.on;
    settingsData.error = `That change was not saved: ${e.message}`;
    renderSettings();
  }
}

// ── the keys themselves ─────────────────────────────────────────────

window.addEventListener('keydown', (event) => {
  // Capture mode: the next combo becomes the binding, whatever it is.
  if (capturing) {
    const combo = comboOf(event);
    if (!combo) return;         // a lone modifier, still waiting
    event.preventDefault();
    if (combo !== 'Escape') {
      keybinds[capturing] = combo;
      saveKeybinds();
    }
    capturing = null;
    renderSettings();
    return;
  }

  if (event.key === 'Escape' && !document.getElementById('settings').hidden) {
    return closeSettings();
  }

  const typing = event.target === document.getElementById('input');
  if (typing && matches(event, keybinds.send)) {
    event.preventDefault();
    return submit();
  }
  // Everything else is a window-level action, and a bare letter must not fire
  // one while someone is typing a message.
  if (typing && !(event.metaKey || event.ctrlKey || event.altKey)) return;

  const fire = {
    newChat: () => newChat(),
    focusComposer: () => document.getElementById('input').focus(),
    toggleSidebar: () => document.body.classList.toggle('sidebar-hidden'),
    toggleWorkspace: () => (document.body.classList.contains('workspace-open')
      ? closeWorkspace() : openWorkspace('mindmap')),
    toggleTheme: () => document.getElementById('btn-theme').click(),
    openSettings: () => (document.getElementById('settings').hidden
      ? openSettings() : closeSettings()),
  };
  for (const [id, run] of Object.entries(fire)) {
    if (matches(event, keybinds[id])) {
      event.preventDefault();
      run();
      return;
    }
  }
});

loadKeybinds();
document.getElementById('btn-settings').addEventListener('click', openSettings);
document.getElementById('btn-close-settings').addEventListener('click', closeSettings);
document.getElementById('settings').addEventListener('click', (event) => {
  if (event.target.id === 'settings') closeSettings();
});

// The composer hint says what the keys ARE, so a rebind is visible without
// opening this panel.
function refreshComposerHint() {
  const hint = document.querySelector('.composer-hint');
  if (!hint) return;
  hint.textContent = `${prettyCombo(keybinds.send)} to send · `
    + `${prettyCombo(keybinds.newline)} for a new line · runs entirely on this Mac`;
}
refreshComposerHint();
