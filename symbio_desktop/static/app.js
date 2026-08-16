/* ═══════════════════════════════════════════════════════════════════════
   Symbio Desktop — Chat + Mind Map + Dashboard
   ═══════════════════════════════════════════════════════════════════════ */

// ── State ──────────────────────────────────────────────────────────

let ecosystem = null;
let svg = null;
let simulation = null;
let selectedNode = null;
let ws = null;
let streamingMsgEl = null;
let chatCollapsed = false;

// ── WebSocket Chat ─────────────────────────────────────────────────

function connectChat() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${protocol}//${location.host}/ws/chat`;

  ws = new WebSocket(url);

  ws.onopen = () => {
    setStatus('connected', 'Connected');
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleWsMessage(data);
  };

  ws.onclose = () => {
    setStatus('error', 'Disconnected');
    // Reconnect after 3s
    setTimeout(connectChat, 3000);
  };

  ws.onerror = () => {
    setStatus('error', 'Connection error');
  };
}

function handleWsMessage(data) {
  switch (data.type) {
    case 'connected':
      document.getElementById('chat-model-badge').textContent =
        data.model_name ? data.model_name.split('/').pop() : 'Symbio';
      addSystemMsg(`Connected to ${data.assistant_name} (${data.model_name || 'unknown'})`);
      break;

    case 'token':
      if (!streamingMsgEl) {
        streamingMsgEl = addAssistantMsg('');
        streamingMsgEl.classList.add('streaming');
      }
      streamingMsgEl.querySelector('.msg-bubble').textContent += data.text;
      scrollChat();
      break;

    case 'done':
      clearProgress();
      if (streamingMsgEl) {
        streamingMsgEl.classList.remove('streaming');
        streamingMsgEl = null;
      }
      break;

    case 'system':
      clearProgress();
      addSystemMsg(data.text);
      break;

    case 'progress':
      showProgress(data.text);
      break;

    case 'confirm':
      showConfirm(data.prompt);
      break;

    case 'error':
      addSystemMsg('Error: ' + data.text);
      if (streamingMsgEl) {
        streamingMsgEl.classList.remove('streaming');
        streamingMsgEl = null;
      }
      break;

    case 'quit':
      addSystemMsg('Session ended.');
      break;

    case 'pong':
      break;
  }
}

function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

  addUserMsg(text);
  input.value = '';
  input.style.height = 'auto';
  document.getElementById('btn-send').disabled = true;

  ws.send(JSON.stringify({ type: 'chat', message: text }));
}

function sendConfirm(approved) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: 'confirm_response', approved }));
  hideConfirm();
}

// ── Chat UI helpers ────────────────────────────────────────────────

function addUserMsg(text) {
  const el = document.createElement('div');
  el.className = 'chat-msg user';
  el.innerHTML = `<div class="msg-sender">You</div><div class="msg-bubble">${escHtml(text)}</div>`;
  document.getElementById('chat-messages').appendChild(el);
  scrollChat();
  return el;
}

function addAssistantMsg(text) {
  const el = document.createElement('div');
  el.className = 'chat-msg assistant';
  el.innerHTML = `<div class="msg-sender">Symbio</div><div class="msg-bubble">${escHtml(text)}</div>`;
  document.getElementById('chat-messages').appendChild(el);
  scrollChat();
  return el;
}

function addSystemMsg(text) {
  const el = document.createElement('div');
  el.className = 'chat-msg system';
  el.innerHTML = `<div class="msg-bubble">${escHtml(text)}</div>`;
  document.getElementById('chat-messages').appendChild(el);
  scrollChat();
  return el;
}

function scrollChat() {
  const container = document.getElementById('chat-messages');
  container.scrollTop = container.scrollHeight;
}

let progressEl = null;

function showProgress(text) {
  if (!text) return;
  if (!progressEl) {
    progressEl = document.createElement('div');
    progressEl.className = 'chat-msg system progress-msg';
    progressEl.innerHTML = '<div class="msg-bubble"></div>';
    document.getElementById('chat-messages').appendChild(progressEl);
  }
  // Clean up tqdm progress bars and spinner frames into a readable line.
  // tqdm format: "Fetching 7 files: 100%|████| 7/7 [00:00<00:00, 7909it/s]"
  let clean = text
    .replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '')  // strip ANSI escapes
    .replace(/[▀-▟]+/g, '')        // strip tqdm bar characters
    .replace(/\s+/g, ' ')
    .trim();
  // Extract the meaningful part from tqdm lines
  const tqdmMatch = clean.match(/^(.+?):\s*(\d+%)\s*\|/);
  if (tqdmMatch) {
    clean = tqdmMatch[1] + ' ' + tqdmMatch[2];
  }
  // Collapse spinner frames
  clean = clean.replace(/^[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]\s*/, '');
  if (clean) {
    progressEl.querySelector('.msg-bubble').textContent = clean;
    scrollChat();
  }
}

function clearProgress() {
  if (progressEl) {
    progressEl.remove();
    progressEl = null;
  }
}

function showConfirm(prompt) {
  document.getElementById('confirm-text').textContent = prompt;
  document.getElementById('chat-confirm').style.display = 'flex';
}

function hideConfirm() {
  document.getElementById('chat-confirm').style.display = 'none';
}

// ── Navigation ─────────────────────────────────────────────────────

document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const view = btn.dataset.view;
    document.querySelectorAll('#bottom-panel .view').forEach(v => v.classList.remove('active'));
    const target = document.getElementById(`view-${view}`);
    if (target) target.classList.add('active');
    if (view === 'mindmap' && ecosystem) renderMindMap();
  });
});

// ── Data Fetching ──────────────────────────────────────────────────

async function fetchEcosystem() {
  try {
    const res = await fetch('/api/ecosystem');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    ecosystem = await res.json();
    renderAll();
  } catch (err) {
    console.error('Failed to fetch ecosystem:', err);
  }
}

function setStatus(state, text) {
  const dot = document.getElementById('status-dot');
  const label = document.getElementById('status-text');
  dot.className = 'status-dot ' + state;
  label.textContent = text;
}

// ── Render All ─────────────────────────────────────────────────────

function renderAll() {
  renderMindMap();
  renderSkills();
  renderRag();
  renderHealth();
}

// ═════════════════════════════════════════════════════════════════════
// MIND MAP
// ═════════════════════════════════════════════════════════════════════

function renderMindMap() {
  const container = document.getElementById('mindmap');
  svg = d3.select('#mindmap-svg');
  svg.selectAll('*').remove();

  const width = container.clientWidth;
  const height = container.clientHeight;

  svg.attr('viewBox', [0, 0, width, height]);

  // Defs for glow filter
  const defs = svg.append('defs');
  const filter = defs.append('filter').attr('id', 'glow').attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
  filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur');
  filter.append('feMerge').selectAll('feMergeNode').data(['blur', 'SourceGraphic']).enter().append('feMergeNode').attr('in', d => d);

  // Build graph
  const nodes = [];
  const edges = [];

  const hm = ecosystem.headmaster;
  nodes.push({
    id: 'headmaster', label: 'Headmaster', type: 'headmaster',
    radius: 26, fx: width / 2, fy: height / 2, data: hm,
  });

  nodes.push({ id: 'rag', label: 'RAG', type: 'rag', radius: 18, data: ecosystem.rag });
  edges.push({ source: 'headmaster', target: 'rag', type: 'rag-edge' });

  nodes.push({ id: 'training', label: 'Training', type: 'training', radius: 18, data: ecosystem.training });
  edges.push({ source: 'headmaster', target: 'training', type: 'training-edge' });

  ecosystem.workers.forEach(w => {
    nodes.push({ id: `worker-${w.role}`, label: w.role, type: 'worker', radius: 15, data: w });
    edges.push({ source: 'headmaster', target: `worker-${w.role}` });
  });

  ecosystem.skills.forEach(s => {
    nodes.push({
      id: `skill-${s.role}`, label: s.skill_name || s.role, type: 'skill',
      radius: 11 + Math.min(s.error_count, 5), data: s,
    });
    edges.push({ source: `skill-${s.role}`, target: 'rag', type: 'rag-edge' });
    edges.push({ source: `skill-${s.role}`, target: 'training', type: 'training-edge' });
  });

  const typeColors = {
    headmaster: '#7c5cfc', worker: '#38bdf8', skill: '#34d399',
    rag: '#f59e0b', training: '#f472b6',
  };

  simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(edges).id(d => d.id).distance(d => {
      if (d.type === 'rag-edge' || d.type === 'training-edge') return 90;
      return 70;
    }).strength(d => {
      if (d.type === 'rag-edge' || d.type === 'training-edge') return 0.08;
      return 0.25;
    }))
    .force('charge', d3.forceManyBody().strength(d => {
      if (d.type === 'headmaster') return -350;
      if (d.type === 'rag' || d.type === 'training') return -180;
      if (d.type === 'worker') return -130;
      return -70;
    }))
    .force('center', d3.forceCenter(width / 2, height / 2).strength(0.04))
    .force('collision', d3.forceCollide().radius(d => d.radius + 7))
    .force('x', d3.forceX(width / 2).strength(0.02))
    .force('y', d3.forceY(height / 2).strength(0.02));

  const edgeGroup = svg.append('g').attr('class', 'edges');
  const edge = edgeGroup.selectAll('line')
    .data(edges)
    .join('line')
    .attr('class', d => `edge-line ${d.type || ''}`);

  const nodeGroup = svg.append('g').attr('class', 'nodes');
  const node = nodeGroup.selectAll('g')
    .data(nodes)
    .join('g')
    .attr('class', 'node')
    .call(d3.drag()
      .on('start', (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x; d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        if (d.type !== 'headmaster') { d.fx = null; d.fy = null; }
      })
    );

  // Glow for headmaster
  node.filter(d => d.type === 'headmaster')
    .append('circle')
    .attr('r', d => d.radius + 5)
    .attr('fill', 'none')
    .attr('stroke', '#7c5cfc')
    .attr('stroke-width', 2)
    .attr('opacity', 0.3)
    .attr('filter', 'url(#glow)');

  node.append('circle')
    .attr('class', d => `node-circle ${d.type}`)
    .attr('r', d => d.radius)
    .attr('fill', d => typeColors[d.type])
    .attr('stroke', d => d.type === 'headmaster' ? '#7c5cfc' : 'rgba(255,255,255,0.12)')
    .attr('stroke-width', d => d.type === 'headmaster' ? 2 : 1)
    .on('click', (event, d) => { event.stopPropagation(); showDetail(d); });

  node.append('text')
    .attr('class', d => `node-label ${d.type}`)
    .attr('dy', d => d.radius + 12)
    .text(d => d.label);

  simulation.on('tick', () => {
    edge.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  svg.call(d3.zoom().scaleExtent([0.3, 3]).on('zoom', (event) => {
    svg.selectAll('g').attr('transform', event.transform);
  }));

  svg.on('click', (event) => {
    if (event.target === svg.node()) hideDetail();
  });

  document.getElementById('btn-zoom-in').onclick = () => zoomBy(1.3);
  document.getElementById('btn-zoom-out').onclick = () => zoomBy(0.7);
  document.getElementById('btn-zoom-fit').onclick = () => zoomFit();
  document.getElementById('btn-refresh').onclick = () => fetchEcosystem();
}

function zoomBy(factor) {
  svg.transition().duration(300).call(d3.zoom().scaleBy, factor);
}

function zoomFit() {
  if (!svg) return;
  const container = document.getElementById('mindmap');
  svg.transition().duration(500).call(
    d3.zoom().transform,
    d3.zoomIdentity.translate(container.clientWidth / 2, container.clientHeight / 2).scale(1).translate(-container.clientWidth / 2, -container.clientHeight / 2)
  );
}

// ── Detail Panel ───────────────────────────────────────────────────

function showDetail(d) {
  selectedNode = d;
  const panel = document.getElementById('detail-panel');
  panel.classList.remove('hidden');

  const data = d.data || {};
  let html = '';

  if (d.type === 'headmaster') {
    html = `
      <div class="detail-title">🧠 Headmaster</div>
      <div class="detail-type">Main Model</div>
      <div class="detail-row"><span class="label">Model</span><span class="value">${escHtml(data.model_name || '—')}</span></div>
      <div class="detail-row"><span class="label">Adapter</span><span class="value">${data.has_weights ? '✓ Trained' : '✗ None'}</span></div>
      <div class="detail-row"><span class="label">Rank</span><span class="value">${data.rank || '—'}</span></div>
      <div class="detail-row"><span class="label">Size</span><span class="value">${data.size_mb ? data.size_mb + ' MB' : '—'}</span></div>
    `;
  } else if (d.type === 'worker') {
    html = `
      <div class="detail-title">⚡ ${escHtml(d.label)}</div>
      <div class="detail-type">Worker</div>
      <div class="detail-row"><span class="label">Model</span><span class="value">${escHtml(data.model_name || '—')}</span></div>
      <div class="detail-row"><span class="label">Adapter</span><span class="value">${data.has_weights ? '✓ Trained' : '✗ None'}</span></div>
      <div class="detail-row"><span class="label">Size</span><span class="value">${data.size_mb ? data.size_mb + ' MB' : '—'}</span></div>
      ${data.description ? `<div class="detail-section"><h4>Description</h4><div class="detail-prompt">${escHtml(data.description)}</div></div>` : ''}
    `;
  } else if (d.type === 'skill') {
    html = `
      <div class="detail-title">🔧 ${escHtml(data.skill_name || d.label)}</div>
      <div class="detail-type">Skill Adapter</div>
      <div class="detail-row"><span class="label">Role</span><span class="value">${escHtml(data.role)}</span></div>
      <div class="detail-row"><span class="label">Adapter</span><span class="value">${data.has_weights ? '✓ Trained' : '✗ None'}</span></div>
      <div class="detail-row"><span class="label">Errors</span><span class="value" style="color:${data.error_count > 0 ? 'var(--danger)' : 'var(--success)'}">${data.error_count}</span></div>
      <div class="detail-row"><span class="label">Corrections</span><span class="value">${data.correction_count}</span></div>
      ${data.routing_rationale ? `<div class="detail-section"><h4>Routing</h4><div class="detail-prompt">${escHtml(data.routing_rationale)}</div></div>` : ''}
      ${data.system_prompt ? `<div class="detail-section"><h4>System Prompt</h4><div class="detail-prompt">${escHtml(data.system_prompt)}</div></div>` : ''}
    `;
  } else if (d.type === 'rag') {
    html = `
      <div class="detail-title">📚 RAG Corpus</div>
      <div class="detail-type">Retrieval System</div>
      <div class="detail-row"><span class="label">Notes</span><span class="value">${data.notes_count}</span></div>
      <div class="detail-row"><span class="label">Training Samples</span><span class="value">${data.training_samples}</span></div>
      <div class="detail-row"><span class="label">Corpus Size</span><span class="value">${data.training_size_mb} MB</span></div>
      <div class="detail-row"><span class="label">Sessions</span><span class="value">${data.sessions_count}</span></div>
    `;
  } else if (d.type === 'training') {
    html = `
      <div class="detail-title">🏋️ Training</div>
      <div class="detail-type">Fine-tuning Pipeline</div>
      <div class="detail-row"><span class="label">Samples</span><span class="value">${data.samples}</span></div>
      <div class="detail-row"><span class="label">Corpus Size</span><span class="value">${data.size_mb} MB</span></div>
      <div class="detail-row"><span class="label">Auto-train</span><span class="value">${data.auto_train ? 'On' : 'Off'}</span></div>
    `;
  }

  panel.innerHTML = html;
}

function hideDetail() {
  selectedNode = null;
  document.getElementById('detail-panel').classList.add('hidden');
}

// ═════════════════════════════════════════════════════════════════════
// SKILLS VIEW
// ═════════════════════════════════════════════════════════════════════

function renderSkills() {
  if (!ecosystem) return;
  const grid = document.getElementById('skills-grid');
  document.getElementById('skill-count').textContent = ecosystem.skills.length;

  grid.innerHTML = ecosystem.skills.map(s => {
    const hasAdapter = s.has_weights;
    const steps = extractSteps(s.system_prompt || '');
    return `
      <div class="skill-card" onclick="selectSkill('${escHtml(s.role)}')">
        <div class="skill-card-header">
          <span class="skill-card-name">${escHtml(s.skill_name || s.role)}</span>
          <div class="skill-card-badges">
            ${s.error_count > 0 ? `<span class="skill-badge errors">${s.error_count} err</span>` : ''}
            ${s.correction_count > 0 ? `<span class="skill-badge corrections">${s.correction_count} fix</span>` : ''}
            <span class="skill-badge ${hasAdapter ? 'trained' : 'untrained'}">${hasAdapter ? 'trained' : 'new'}</span>
          </div>
        </div>
        ${steps ? `<div class="skill-card-steps">${escHtml(steps)}</div>` : ''}
      </div>
    `;
  }).join('');
}

function extractSteps(prompt) {
  if (!prompt) return '';
  const match = prompt.match(/Steps:\s*\n([\s\S]*?)(?:\n\n|Reply with|$)/);
  if (!match) return '';
  return match[1].trim();
}

function selectSkill(role) {
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.querySelector('[data-view="mindmap"]').classList.add('active');
  document.querySelectorAll('#bottom-panel .view').forEach(v => v.classList.remove('active'));
  document.getElementById('view-mindmap').classList.add('active');

  if (!ecosystem) return;
  renderMindMap();

  setTimeout(() => {
    const nodeData = simulation.nodes().find(n => n.id === `skill-${role}`);
    if (nodeData) {
      showDetail(nodeData);
      const container = document.getElementById('mindmap');
      const tx = container.clientWidth / 2 - nodeData.x;
      const ty = container.clientHeight / 2 - nodeData.y;
      svg.transition().duration(600).call(
        d3.zoom().transform,
        d3.zoomIdentity.translate(tx, ty).scale(1.5)
      );
    }
  }, 100);
}

// ═════════════════════════════════════════════════════════════════════
// RAG VIEW
// ═════════════════════════════════════════════════════════════════════

function renderRag() {
  if (!ecosystem) return;
  const rag = ecosystem.rag;
  const container = document.getElementById('rag-stats');

  container.innerHTML = `
    <div class="stat-card">
      <div class="stat-value">${rag.notes_count}</div>
      <div class="stat-label">Notes in Corpus</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${rag.training_samples}</div>
      <div class="stat-label">Training Samples</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${rag.training_size_mb}<span style="font-size:14px"> MB</span></div>
      <div class="stat-label">Corpus Size</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${rag.sessions_count}</div>
      <div class="stat-label">Past Sessions</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${ecosystem.skills.length}</div>
      <div class="stat-label">Skill Adapters</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${ecosystem.workers.length}</div>
      <div class="stat-label">Worker Models</div>
    </div>
  `;
}

// ═════════════════════════════════════════════════════════════════════
// HEALTH VIEW
// ═════════════════════════════════════════════════════════════════════

function renderHealth() {
  if (!ecosystem) return;
  const skills = ecosystem.skills;
  const totalErrors = skills.reduce((sum, s) => sum + s.error_count, 0);
  const totalCorrections = skills.reduce((sum, s) => sum + s.correction_count, 0);
  const skillsWithErrors = skills.filter(s => s.error_count > 0);
  const skillsWithCorrections = skills.filter(s => s.correction_count > 0);
  const trainedCount = skills.filter(s => s.has_weights).length;

  const container = document.getElementById('health-content');
  container.innerHTML = `
    <div class="health-summary">
      <div class="health-card">
        <div class="value ${totalErrors === 0 ? 'ok' : totalErrors < 5 ? 'warn' : 'bad'}">${totalErrors}</div>
        <div class="label">Total Errors</div>
      </div>
      <div class="health-card">
        <div class="value ${totalCorrections === 0 ? 'ok' : 'warn'}">${totalCorrections}</div>
        <div class="label">Corrections</div>
      </div>
      <div class="health-card">
        <div class="value ok">${trainedCount}/${skills.length}</div>
        <div class="label">Trained Adapters</div>
      </div>
      <div class="health-card">
        <div class="value ok">${skills.length}</div>
        <div class="label">Total Skills</div>
      </div>
    </div>
    ${skillsWithErrors.length > 0 ? `
    <div class="health-issues">
      <h3>⚠️ Skills with Errors (${skillsWithErrors.length})</h3>
      ${skillsWithErrors.map(s => `
        <div class="health-issue-row">
          <span class="health-issue-dot error"></span>
          <span style="flex:1">${escHtml(s.skill_name || s.role)}</span>
          <span style="color:var(--danger)">${s.error_count} error${s.error_count > 1 ? 's' : ''}</span>
        </div>
      `).join('')}
    </div>
    ` : `
    <div class="health-issues">
      <h3>✅ All Clear</h3>
      <div class="health-issue-row" style="color:var(--text-muted)">No skills have recorded errors.</div>
    </div>
    `}
    ${skillsWithCorrections.length > 0 ? `
    <div class="health-issues">
      <h3>🔧 Skills with Corrections (${skillsWithCorrections.length})</h3>
      ${skillsWithCorrections.map(s => `
        <div class="health-issue-row">
          <span class="health-issue-dot correction"></span>
          <span style="flex:1">${escHtml(s.skill_name || s.role)}</span>
          <span style="color:var(--accent-4)">${s.correction_count} fix${s.correction_count > 1 ? 'es' : ''}</span>
        </div>
      `).join('')}
    </div>
    ` : ''}
  `;
}

// ── Utilities ───────────────────────────────────────────────────────

function escHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ═════════════════════════════════════════════════════════════════════
// EVENT BINDINGS
// ═════════════════════════════════════════════════════════════════════

// Send button
document.getElementById('btn-send').addEventListener('click', sendMessage);

// Enter to send, Shift+Enter for newline
document.getElementById('chat-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Auto-resize textarea
document.getElementById('chat-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
  document.getElementById('btn-send').disabled = !this.value.trim();
});

// Confirm buttons
document.getElementById('btn-confirm-allow').addEventListener('click', () => sendConfirm(true));
document.getElementById('btn-confirm-deny').addEventListener('click', () => sendConfirm(false));

// Toggle chat panel
document.getElementById('btn-toggle-chat').addEventListener('click', () => {
  chatCollapsed = !chatCollapsed;
  const panel = document.getElementById('chat-panel');
  const btn = document.getElementById('btn-toggle-chat');
  if (chatCollapsed) {
    panel.classList.add('collapsed');
    btn.textContent = '+';
  } else {
    panel.classList.remove('collapsed');
    btn.textContent = '−';
  }
  // Re-render mind map if visible (size changed)
  setTimeout(() => {
    if (document.getElementById('view-mindmap').classList.contains('active') && ecosystem) {
      renderMindMap();
    }
  }, 300);
});

// Clear chat
document.getElementById('btn-clear-chat').addEventListener('click', () => {
  const container = document.getElementById('chat-messages');
  container.innerHTML = `
    <div class="chat-welcome">
      <div class="welcome-icon">🧠</div>
      <div class="welcome-text">Symbio Desktop</div>
      <div class="welcome-sub">Local-first AI agent with self-finetuning adapters</div>
    </div>
  `;
  streamingMsgEl = null;
});

// Resize handle
const resizeHandle = document.getElementById('resize-handle');
const chatPanel = document.getElementById('chat-panel');
let resizeActive = false;
let resizeStartY = 0;
let resizeStartHeight = 0;

resizeHandle.addEventListener('mousedown', (e) => {
  resizeActive = true;
  resizeStartY = e.clientY;
  resizeStartHeight = chatPanel.offsetHeight;
  resizeHandle.classList.add('active');
  document.body.style.cursor = 'row-resize';
  document.body.style.userSelect = 'none';
});

document.addEventListener('mousemove', (e) => {
  if (!resizeActive) return;
  const delta = e.clientY - resizeStartY;
  const newHeight = Math.max(120, Math.min(resizeStartHeight + delta, window.innerHeight * 0.7));
  chatPanel.style.height = newHeight + 'px';
  chatPanel.style.minHeight = '120px';
  chatPanel.classList.remove('collapsed');
  chatCollapsed = false;
  document.getElementById('btn-toggle-chat').textContent = '−';
});

document.addEventListener('mouseup', () => {
  if (!resizeActive) return;
  resizeActive = false;
  resizeHandle.classList.remove('active');
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
  // Re-render mind map
  setTimeout(() => {
    if (document.getElementById('view-mindmap').classList.contains('active') && ecosystem) {
      renderMindMap();
    }
  }, 100);
});

// ── Init ────────────────────────────────────────────────────────────

connectChat();
fetchEcosystem();
setInterval(fetchEcosystem, 30000);
