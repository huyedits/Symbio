/* The chat client: one conversation, streamed, with the agent's own work
 * shown rather than hidden.
 *
 * The old shell put the conversation in a resizable strip above a mind map,
 * which said the graph was the product and the talking was a side panel. It
 * is the other way round — the graph is something you ask to see.
 *
 * Two things here are not cosmetic:
 *
 *   Tool activity is a card, not a scroll. The server sends every [Tool: …],
 *   [Vision], [Security] and progress line as it happens. Dropping them makes
 *   the agent look like it guessed; printing them raw buries the answer. They
 *   collapse into one line that says what it is doing, and open on click.
 *
 *   A confirmation is a blocking card. The server's _confirm_fn is a real
 *   thread waiting on an answer — the same gate as the terminal's y/N — so
 *   the UI must not let the turn continue around it.
 */

const CONVERSATIONS_KEY = 'symbio.conversations.v1';
const LIVE_KEY = 'symbio.live.v1';
const THEME_KEY = 'symbio.theme';

let ws = null;
let reconnectDelay = 500;
let live = { id: null, title: 'New chat', messages: [] };
let viewingId = null;          // null = the live conversation
let streamEl = null;           // the assistant bubble currently being written
let streamText = '';
let activityCard = null;
let pendingConfirm = false;

const thread = document.getElementById('thread');
const input = document.getElementById('input');
const sendBtn = document.getElementById('btn-send');

// ── storage ─────────────────────────────────────────────────────────

function loadConversations() {
  try {
    return JSON.parse(localStorage.getItem(CONVERSATIONS_KEY)) || [];
  } catch (e) {
    return [];
  }
}

function saveConversations(list) {
  try {
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(list.slice(0, 50)));
  } catch (e) { /* private window, full disk: the chat still works */ }
}

function archiveLive() {
  if (!live.messages.length) return;
  const list = loadConversations().filter(c => c.id !== live.id);
  list.unshift({ id: live.id, title: live.title, at: Date.now(),
                 messages: live.messages });
  saveConversations(list);
  renderChatList();
}

function renderChatList() {
  const box = document.getElementById('chat-list');
  const saved = loadConversations();
  const rows = [];
  rows.push(row(live.id, live.title || 'New chat', viewingId === null, true));
  for (const c of saved) {
    if (c.id === live.id) continue;
    rows.push(row(c.id, c.title, viewingId === c.id, false));
  }
  box.innerHTML = rows.join('');
  box.querySelectorAll('.chat-row').forEach(el => {
    el.addEventListener('click', (e) => {
      if (e.target.closest('.row-delete')) return;
      openConversation(el.dataset.id);
    });
    const del = el.querySelector('.row-delete');
    if (del) del.addEventListener('click', () => {
      saveConversations(loadConversations().filter(c => c.id !== el.dataset.id));
      if (viewingId === el.dataset.id) backToLive();
      else renderChatList();
    });
  });

  function row(id, title, active, isLive) {
    return `<div class="chat-row${active ? ' active' : ''}" data-id="${id}">
      <span class="row-title">${escHtml(title || 'New chat')}</span>
      ${isLive ? '<span class="row-live">live</span>'
               : '<button class="row-delete" title="Delete">✕</button>'}
    </div>`;
  }
}

// ── conversations ───────────────────────────────────────────────────

function newChat() {
  archiveLive();
  live = { id: `c${Date.now()}`, title: 'New chat', messages: [] };
  rememberLive();
  viewingId = null;
  streamEl = null;
  activityCard = null;
  renderThread([]);
  setTitle('New chat');
  document.getElementById('read-only-banner').hidden = true;
  input.disabled = false;
  // A new conversation is a new server-side session: the socket owns the
  // history, so reconnecting is what actually clears it. Pretending
  // otherwise would show an empty thread over a model that still remembers.
  if (ws && ws.readyState === WebSocket.OPEN) ws.close(4000, 'new chat');
  renderChatList();
  input.focus();
}

function openConversation(id) {
  if (id === live.id) return backToLive();
  const found = loadConversations().find(c => c.id === id);
  if (!found) return;
  viewingId = id;
  renderThread(found.messages);
  setTitle(found.title);
  document.getElementById('read-only-banner').hidden = false;
  // Not disabled: typing here sends to the live conversation, which is what
  // someone reading an old thread and starting to type actually wants.
  input.disabled = false;
  input.placeholder = 'Message Symbio (sends to the live chat)…';
  renderChatList();
}

function backToLive() {
  viewingId = null;
  renderThread(live.messages);
  setTitle(live.title);
  document.getElementById('read-only-banner').hidden = true;
  input.disabled = false;
  input.placeholder = 'Message Symbio…';
  renderChatList();
  input.focus();
}

function setTitle(text) {
  document.getElementById('conversation-title').textContent = text || 'New chat';
}

// ── rendering ───────────────────────────────────────────────────────

/* The empty window is a greeting with the composer under it, in the middle
 * of the page; the first message moves the composer to the bottom where a
 * conversation needs it. One function owns each direction, so no path can
 * leave the greeting on screen above a reply. */
function greetingText() {
  const h = new Date().getHours();
  const part = h < 5 ? 'Up late' : h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening';
  const name = window.__userName && window.__userName !== 'user' ? `, ${window.__userName}` : '';
  return `${part}${name}`;
}

function showWelcome() {
  thread.innerHTML = `<div class="welcome">
    <h1 class="greeting"><span class="mark" aria-hidden="true"></span><span>${escHtml(greetingText())}</span></h1>
  </div>`;
  document.getElementById('main').classList.add('empty');
}

function clearWelcome() {
  const welcome = thread.querySelector('.welcome');
  if (welcome) welcome.remove();
  document.getElementById('main').classList.remove('empty');
}

function renderThread(messages) {
  thread.innerHTML = '';
  if (!messages.length) {
    showWelcome();
    return;
  }
  document.getElementById('main').classList.remove('empty');
  for (const m of messages) addBubble(m.role, m.text, false, m.thought);
  markLastAssistant();
  scrollToEnd();
}

function addBubble(role, text, store = true, thought = '') {
  clearWelcome();
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  // The reasoning is KEPT, not dropped: it is the most interesting thing a
  // local model produces and the reason to trust or distrust the answer. It
  // is folded shut because it is not what was said — the answer is.
  const lines = thought ? thought.trim().split(/\s+/).length : 0;
  el.innerHTML =
    (thought
      ? `<details class="thought"><summary>Thought process · ${lines} words</summary>`
        + `<div class="thought-body">${escHtml(thought.trim())}</div></details>`
      : '')
    + `<div class="msg-body">${renderMarkdown(text)}</div>`;
  if (role === 'assistant') {
    const actions = document.createElement('div');
    actions.className = 'msg-actions';
    const copy = document.createElement('button');
    copy.className = 'copy-btn';
    copy.type = 'button';
    copy.textContent = 'Copy';
    copy.addEventListener('click', () => {
      navigator.clipboard.writeText(text).then(() => {
        copy.textContent = 'Copied';
        setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
      });
    });
    const retry = document.createElement('button');
    retry.className = 'retry-btn';
    retry.type = 'button';
    retry.textContent = 'Retry';
    retry.title = 'Ask the same thing again';
    retry.addEventListener('click', retryLast);
    actions.append(copy, retry);
    el.appendChild(actions);
  }
  thread.appendChild(el);
  if (store && viewingId === null) {
    live.messages.push(thought ? { role, text, thought } : { role, text });
    if (live.messages.length === 1) {
      live.title = text.slice(0, 48);
      setTitle(live.title);
      renderChatList();
    }
    archiveLive();
    rememberLive();
  }
  scrollToEnd();
  return el;
}

/* Between pressing enter and the first token there can be twenty seconds of
 * prompt processing, and the window used to show nothing at all for it: no
 * bubble, no clock, no sign the machine had heard. "Nothing is happening" is
 * indistinguishable from "it is working" unless something moves, so this puts
 * a placeholder in the thread immediately, counts the seconds out loud, and
 * says what it is waiting for if the wait gets long.
 */
let waitTimer = null;
let waitStarted = 0;
let lastFrameAt = 0;
// While the model loads, what the wait is actually for. The server sends a
// `waking` line every couple of seconds; it wins over the guesses below.
let wakingText = '';

function startWaiting() {
  waitStarted = Date.now();
  lastFrameAt = Date.now();
  clearWelcome();
  let pending = document.getElementById('pending');
  if (!pending) {
    pending = document.createElement('div');
    pending.id = 'pending';
    pending.className = 'msg assistant pending';
    pending.innerHTML = '<div class="msg-body"><span class="dots">'
      + '<span></span><span></span><span></span></span>'
      + '<span class="wait-label"></span></div>';
    thread.appendChild(pending);
  }
  scrollToEnd();
  clearInterval(waitTimer);
  waitTimer = setInterval(tickWaiting, 1000);
  tickWaiting();
}

function tickWaiting() {
  const seconds = Math.round((Date.now() - waitStarted) / 1000);
  const quiet = Math.round((Date.now() - lastFrameAt) / 1000);
  if (wakingText) {
    setStatus('busy', wakingText);
    const waking = document.querySelector('#pending .wait-label');
    if (waking) waking.textContent = `${wakingText} — your message goes as soon as it is up`;
    return;
  }
  setStatus('busy', `Thinking… ${seconds}s`);
  const label = document.querySelector('#pending .wait-label');
  if (!label) return;
  // What it is actually waiting on, rather than a spinner that means nothing.
  // A 14B reading a 5,000-token prompt is silent for a while by design; a
  // silence past half a minute more likely means the resident model is busy
  // with another window, since it serves one at a time.
  if (quiet > 30) {
    label.textContent = `still nothing after ${quiet}s — the resident model `
      + 'serves one window at a time, so it may be busy with another';
  } else if (seconds > 3) {
    label.textContent = `reading the prompt… ${seconds}s`;
  } else {
    label.textContent = '';
  }
}

function stopWaiting() {
  clearInterval(waitTimer);
  waitTimer = null;
  const pending = document.getElementById('pending');
  if (pending) pending.remove();
}

function ensureStream() {
  stopWaiting();
  if (streamEl) return streamEl;
  clearWelcome();
  streamText = '';
  streamEl = document.createElement('div');
  streamEl.className = 'msg assistant streaming';
  streamEl.innerHTML = '<div class="msg-body"></div>';
  thread.appendChild(streamEl);
  scrollToEnd();
  return streamEl;
}

/* The model thinks out loud before it answers, and the CLI prints that block
 * behind a [Reasoning] marker. Streamed straight into the bubble it reads as
 * the reply — the same failure as a truncated <think> block being shown as
 * the answer — so it is split off and folded away, with the answer left as
 * the message. */
function splitReasoning(text, final = false) {
  // The session's marker is "  [Reasoning] " — indented, as the terminal
  // prints it. Tested for flush left, it never matched, and the whole
  // thought was shown as the reply.
  const lead = text.match(/^\s*/)[0].length;
  if (!text.startsWith('[Reasoning]', lead)) return { thought: '', answer: text };
  text = text.slice(lead);
  // The answer starts at the speaker label the session prints after the
  // reasoning ("Symbio : Hi!"). Splitting at the first blank line instead
  // showed every paragraph of reasoning after the first as the reply — seen
  // live 2026-09-26, with the label left in the middle of the bubble.
  const name = (window.__assistantName || 'Symbio').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const label = new RegExp(`\\n\\s*${name}\\s*:\\s*`);
  const found = label.exec(text);
  if (found) {
    return { thought: text.slice('[Reasoning]'.length, found.index).trim(),
             answer: text.slice(found.index + found[0].length) };
  }
  // No label yet: still thinking, unless the turn is over — then the old
  // first-blank-line split is the best guess there is.
  const end = text.indexOf('\n\n');
  if (!final || end === -1) return { thought: text.slice('[Reasoning]'.length).trim(), answer: '' };
  return {
    thought: text.slice('[Reasoning]'.length, end).trim(),
    answer: text.slice(end + 2),
  };
}

function appendToken(text) {
  const el = ensureStream();
  streamText += text;
  const { thought, answer } = splitReasoning(streamText);
  let thoughtEl = el.querySelector('.thought');
  if (thought && !thoughtEl) {
    thoughtEl = document.createElement('details');
    thoughtEl.className = 'thought';
    thoughtEl.innerHTML = '<summary>Thinking…</summary><div class="thought-body"></div>';
    el.insertBefore(thoughtEl, el.firstChild);
  }
  if (thoughtEl) thoughtEl.querySelector('.thought-body').textContent = thought;
  el.querySelector('.msg-body').innerHTML = renderMarkdown(answer);
  scrollToEnd();
}

function markLastAssistant() {
  thread.querySelectorAll('.msg.assistant').forEach(m => m.classList.remove('last'));
  const all = thread.querySelectorAll('.msg.assistant');
  if (all.length) all[all.length - 1].classList.add('last');
}

function retryLast() {
  if (document.body.classList.contains('busy') || viewingId !== null) return;
  const lastUser = [...live.messages].reverse().find(m => m.role === 'user');
  if (!lastUser) return;
  input.value = lastUser.text;
  input.dispatchEvent(new Event('input'));
  submit();
}

function finishStream(finalText) {
  const raw = finalText || streamText || '';
  const split = splitReasoning(raw, true);
  // The reasoning is kept out of the stored transcript as well: it is not
  // what was said, and replaying it as the assistant's message would teach
  // the next read of this conversation that it was.
  const text = (split.answer || (split.thought ? '' : raw)).trim();
  if (streamEl) {
    streamEl.remove();
    streamEl = null;
  }
  streamText = '';
  if (activityCard) {
    activityCard.classList.add('done');
    activityCard = null;
  }
  if (text || split.thought) addBubble('assistant', text, true, split.thought);
  markLastAssistant();
}

// Everything the turn did on its way to the answer, folded into one line.
function addActivity(text) {
  const clean = (text || '').replace(/\s+$/, '');
  if (!clean.trim()) return;
  // A message that arrives with no turn in flight — "no resident model is
  // running" on connect — is about the window, not about a reply. As a card
  // it pushed the greeting off an empty window; it belongs above the composer.
  if (!document.body.classList.contains('busy') && !activityCard) {
    showNotice(clean);
    return;
  }
  if (!activityCard) {
    clearWelcome();
    activityCard = document.createElement('div');
    activityCard.className = 'activity';
    activityCard.innerHTML = `
      <button class="activity-head" type="button">
        <span class="spinner"></span>
        <span class="activity-label">Working…</span>
        <span class="activity-chevron">▾</span>
      </button>
      <pre class="activity-log"></pre>`;
    activityCard.querySelector('.activity-head').addEventListener('click', () => {
      activityCard.classList.toggle('open');
    });
    // Before the streaming bubble, so the answer always sits last.
    if (streamEl) thread.insertBefore(activityCard, streamEl);
    else thread.appendChild(activityCard);
  }
  const log = activityCard.querySelector('.activity-log');
  log.textContent += (log.textContent ? '\n' : '') + clean;
  const label = clean.trim().replace(/^\[|\]$/g, '');
  activityCard.querySelector('.activity-label').textContent =
    label.length > 80 ? label.slice(0, 80) + '…' : label;
  scrollToEnd();
}

function showNotice(text) {
  const box = document.getElementById('notice');
  document.getElementById('notice-text').innerHTML = renderMarkdown(text.trim());
  box.hidden = false;
}

function hideNotice() {
  document.getElementById('notice').hidden = true;
}

function showConfirm(prompt) {
  pendingConfirm = true;
  const card = document.createElement('div');
  card.className = 'confirm-card';
  card.innerHTML = `
    <div class="confirm-text">${escHtml(prompt)}</div>
    <div class="confirm-actions">
      <button class="btn deny">Deny</button>
      <button class="btn allow">Allow</button>
    </div>`;
  thread.appendChild(card);
  scrollToEnd();
  const answer = (approved) => {
    if (!pendingConfirm) return;
    pendingConfirm = false;
    send({ type: 'confirm_response', approved });
    card.classList.add('answered');
    card.querySelector('.confirm-actions').innerHTML =
      `<span class="confirm-verdict">${approved ? 'Allowed' : 'Denied'}</span>`;
  };
  card.querySelector('.allow').addEventListener('click', () => answer(true));
  card.querySelector('.deny').addEventListener('click', () => answer(false));
}

function nearBottom() {
  return thread.scrollHeight - thread.scrollTop - thread.clientHeight < 80;
}

function scrollToEnd(force = false) {
  if (force || nearBottom() || !document.body.classList.contains('busy')) {
    thread.scrollTop = thread.scrollHeight;
  }
  updateToBottom();
}

function updateToBottom() {
  const btn = document.getElementById('btn-to-bottom');
  if (btn) btn.hidden = nearBottom();
}

function setStatus(state, text) {
  const dot = document.getElementById('status-dot');
  if (dot) dot.className = 'status-dot ' + state;
  const label = document.getElementById('status-text');
  if (label) label.textContent = text;
}

// ── markdown, enough of it ──────────────────────────────────────────

function escHtml(str) {
  return String(str == null ? '' : str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/* Escape first, then mark up the escaped text. A model's reply routinely
 * contains the tags it was asked about, and a renderer that parses before it
 * escapes turns that into markup — including, in an agent that can be asked
 * to write HTML, script. */
function renderMarkdown(src) {
  let out = escHtml(src || '');
  const blocks = [];
  out = out.replace(/```(\w*)\n?([\s\S]*?)```/g, (_m, lang, code) => {
    blocks.push(`<div class="code-block"><div class="code-head"><span>${lang || 'code'}</span>`
      + `<button type="button" class="code-copy">Copy</button></div>`
      + `<pre class="code"${lang ? ` data-lang="${lang}"` : ''}><code>${code.replace(/\n$/, '')}</code></pre></div>`);
    return ` ${blocks.length - 1} `;
  });
  out = out.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  out = out.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
  out = out.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
  out = out.replace(/(https?:\/\/[^\s<]+)/g,
                    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
  out = out.replace(/(?:^|\n)((?:[-*]\s+.+(?:\n|$))+)/g, (_m, items) => {
    const lis = items.trim().split('\n')
      .map(line => `<li>${line.replace(/^[-*]\s+/, '')}</li>`).join('');
    return `<ul>${lis}</ul>`;
  });
  out = out.split(/\n{2,}/).map(part =>
    /^\s*<(h\d|ul|pre|div| )/.test(part) || part.includes(' ')
      ? part : `<p>${part.replace(/\n/g, '<br>')}</p>`).join('');
  return out.replace(/ (\d+) /g, (_m, i) => blocks[Number(i)]);
}

// ── socket ──────────────────────────────────────────────────────────

function connect() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${scheme}://${location.host}/ws/chat`);

  ws.onopen = () => {
    reconnectDelay = 500;
    setStatus('ok', 'Connected');
  };

  ws.onmessage = (event) => {
    let data;
    try { data = JSON.parse(event.data); } catch (e) { return; }
    lastFrameAt = Date.now();
    switch (data.type) {
      case 'connected':
        window.__assistantName = data.assistant_name || 'Symbio';
        window.__userName = data.user_name || '';
        input.placeholder = `Message ${window.__assistantName}…`;
        document.getElementById('model-chip').textContent =
          (data.model_name || '').split('/').pop() || '—';
        // Asleep is not an error: the first message wakes it.
        if (data.model_state === 'down') {
          setStatus('off', `${data.assistant_name} · asleep — waking up`);
        } else if (data.model_state === 'loading') {
          setStatus('busy', `${data.assistant_name} · waking up`);
        } else {
          setStatus('ok', `${data.assistant_name} · ready`);
        }
        if (!live.messages.length) {
          const title = document.querySelector('.welcome h1');
          if (title) title.textContent = window.__assistantName;
        }
        break;
      case 'token':   wakingText = ''; appendToken(data.text); break;
      case 'waking':
        wakingText = data.text || 'Waking Symbio…';
        if (waitTimer) tickWaiting(); else setStatus('busy', wakingText);
        break;
      case 'asleep':
        // The model could not be woken (the reason follows as a system line).
        wakingText = '';
        setStatus('off', `${window.__assistantName || 'Symbio'} · asleep — send a message to try again`);
        break;
      case 'awake':
        wakingText = '';
        if (!waitTimer) setStatus('ok', `${window.__assistantName || 'Symbio'} · ready`);
        break;
      case 'system':
      case 'progress': addActivity(data.text); break;
      case 'confirm': showConfirm(data.prompt); break;
      case 'done':
        wakingText = '';
        stopWaiting();
        finishStream(data.text);
        setStatus('ok', `Ready · last reply took ${Math.round((Date.now() - waitStarted) / 1000)}s`);
        setBusy(false);
        break;
      case 'error':
        stopWaiting();
        finishStream('');
        addActivity(data.text);
        addBubble('assistant', `**Something failed.** ${String(data.text).split('\n')[0]}`);
        setBusy(false);
        break;
      case 'quit':
        setStatus('off', 'Session ended');
        break;
    }
  };

  ws.onclose = () => {
    stopWaiting();
    setStatus('off', 'Reconnecting…');
    setBusy(false);
    // Backs off rather than hammering: the server is a local process that
    // may be restarting, and a tight loop would just fill its log.
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 8000);
  };

  ws.onerror = () => setStatus('off', 'Connection problem');
}

function send(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
}

function setBusy(busy) {
  document.body.classList.toggle('busy', busy);
  sendBtn.disabled = busy || !input.value.trim();
}

function submit() {
  const text = input.value.trim();
  if (!text) return;
  // Typing while a saved transcript is open used to return here and do
  // nothing at all: the message vanished, the composer cleared nothing, and
  // the window looked broken. Send it to the live conversation instead —
  // that is plainly what was meant.
  if (viewingId !== null) backToLive();
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    addActivity('Not connected to the window server yet — nothing was sent. '
                + 'Retrying the connection.');
    return;
  }
  addBubble('user', text);
  input.value = '';
  input.style.height = 'auto';
  setBusy(true);
  startWaiting();
  send({ type: 'chat', message: text });
}

// ── wiring ──────────────────────────────────────────────────────────

document.getElementById('composer').addEventListener('submit', (e) => {
  e.preventDefault();
  submit();
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    submit();
  }
});

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 200) + 'px';
  sendBtn.disabled = !input.value.trim() || document.body.classList.contains('busy');
});

document.getElementById('btn-new-chat').addEventListener('click', newChat);
thread.addEventListener('scroll', updateToBottom);
document.getElementById('btn-notice-close').addEventListener('click', hideNotice);
document.getElementById('btn-to-bottom').addEventListener('click', () => scrollToEnd(true));
thread.addEventListener('click', (e) => {
  const btn = e.target.closest('.code-copy');
  if (!btn) return;
  const code = btn.closest('.code-block').querySelector('code').textContent;
  navigator.clipboard.writeText(code).then(() => {
    btn.textContent = 'Copied';
    setTimeout(() => { btn.textContent = 'Copy'; }, 1200);
  });
});
document.addEventListener('keydown', (e) => {
  // Cmd/Ctrl+K: new chat, the shortcut the desktop chat apps share.
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    newChat();
    input.focus();
  }
});
document.getElementById('btn-back-live').addEventListener('click', backToLive);

document.querySelectorAll('.suggestion').forEach(btn => {
  btn.addEventListener('click', () => {
    input.value = btn.dataset.send;
    input.dispatchEvent(new Event('input'));
    submit();
  });
});

const setSidebar = (hidden) => {
  document.body.classList.toggle('sidebar-hidden', hidden);
  try { localStorage.setItem('symbio.sidebar', hidden ? 'hidden' : 'shown'); } catch (e) {}
};
document.getElementById('btn-collapse').addEventListener('click', () => setSidebar(true));
document.getElementById('btn-show-sidebar').addEventListener('click', () => setSidebar(false));

const applyTheme = (theme) => {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem(THEME_KEY, theme); } catch (e) {}
};
document.getElementById('btn-theme').addEventListener('click', () => {
  const now = document.documentElement.dataset.theme
    || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  applyTheme(now === 'dark' ? 'light' : 'dark');
});

try {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved) document.documentElement.dataset.theme = saved;
  if (localStorage.getItem('symbio.sidebar') === 'hidden') {
    document.body.classList.add('sidebar-hidden');
  }
} catch (e) { /* storage blocked; the default theme is fine */ }

/* A refresh is not a new conversation. The window used to mint a fresh id on
 * every load and open on an empty thread, so a reload looked exactly like
 * losing the conversation — and the server, which now keeps the daemon
 * session across reconnects, was still in the middle of it. */
function restoreLive() {
  try {
    const saved = JSON.parse(localStorage.getItem(LIVE_KEY));
    if (saved && Array.isArray(saved.messages) && saved.messages.length) {
      live = { id: saved.id, title: saved.title || 'New chat', messages: saved.messages };
      renderThread(live.messages);
      setTitle(live.title);
      return;
    }
  } catch (e) { /* blocked storage: a fresh conversation is the fallback */ }
  live.id = `c${Date.now()}`;
}

function rememberLive() {
  try {
    localStorage.setItem(LIVE_KEY, JSON.stringify(live));
  } catch (e) { /* the conversation still works, it just will not survive */ }
}

restoreLive();
renderChatList();
connect();
