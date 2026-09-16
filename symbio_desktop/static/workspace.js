/* Workspace views — the mind map, skills, RAG corpus and health panels.
 *
 * Lifted unchanged out of the single-file app.js when the shell became
 * chat-first: the chat is the application now and these are what it can be
 * asked to show, so they live behind the workspace drawer rather than under
 * a permanently half-height conversation. Element ids are the ones the old
 * markup used, which is why the rendering below needed no edits.
 */

let ecosystem = null;
let svg = null;
let simulation = null;
let selectedNode = null;

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

async function renderSessions() {
  const list = document.getElementById('session-list');
  list.innerHTML = '<div class="detail-placeholder">Reading the session store…</div>';
  let data;
  try {
    data = await (await fetch('/api/sessions')).json();
  } catch (e) {
    list.innerHTML = '<div class="detail-placeholder">Could not read the session store.</div>';
    return;
  }
  const sessions = data.sessions || [];
  document.getElementById('session-count').textContent = sessions.length;
  if (!sessions.length) {
    list.innerHTML = `<div class="detail-placeholder">${escHtml(data.reason || 'No sessions yet.')}</div>`;
    return;
  }
  list.innerHTML = sessions.map(s => `<button class="session-row" data-id="${escHtml(s.id)}">
      <span class="session-when">${escHtml((s.started || s.id).replace('T', ' ').slice(0, 16))}</span>
      <span class="session-opening">${escHtml(s.opening || '(no user turn)')}</span>
      <span class="session-turns">${s.turns}</span>
    </button>`).join('');
  list.querySelectorAll('.session-row').forEach(row => {
    row.addEventListener('click', () => openSession(row.dataset.id, row));
  });
}

async function openSession(id, row) {
  document.querySelectorAll('.session-row').forEach(r => r.classList.remove('active'));
  if (row) row.classList.add('active');
  const detail = document.getElementById('session-detail');
  detail.innerHTML = '<div class="detail-placeholder">Reading…</div>';
  let data;
  try {
    data = await (await fetch(`/api/sessions?id=${encodeURIComponent(id)}`)).json();
  } catch (e) {
    detail.innerHTML = '<div class="detail-placeholder">Could not read that session.</div>';
    return;
  }
  // Every turn here was written by a past conversation — other people's text
  // in the Telegram case. Rendered as escaped plain text, never as markup.
  detail.innerHTML = (data.turns || []).map(t => `<div class="session-turn ${escHtml(t.role)}">
      <div class="session-role">${escHtml(t.role)}</div>
      <div class="session-text">${escHtml(t.text || '')}</div>
    </div>`).join('') || '<div class="detail-placeholder">That session has no turns.</div>';
}

function renderAll() {
  if (!ecosystem) return;
  if (document.getElementById('view-mindmap').classList.contains('active')) renderMindMap();
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


// ── Drawer ──────────────────────────────────────────────────────────

function openWorkspace(view) {
  document.body.classList.add('workspace-open');
  document.querySelectorAll('.ws-tab').forEach(b => {
    b.classList.toggle('active', b.dataset.view === view);
  });
  document.querySelectorAll('#workspace .view').forEach(v => v.classList.remove('active'));
  const target = document.getElementById(`view-${view}`);
  if (target) target.classList.add('active');
  if (view === 'sessions') return renderSessions();
  if (!ecosystem) fetchEcosystem();
  // The mind map sizes itself to its container, which was 0x0 while closed.
  else if (view === 'mindmap') setTimeout(renderMindMap, 60);
  else renderAll();
}

function closeWorkspace() {
  document.body.classList.remove('workspace-open');
}

document.querySelectorAll('[data-open-view]').forEach(btn => {
  btn.addEventListener('click', () => openWorkspace(btn.dataset.openView));
});
document.querySelectorAll('.ws-tab').forEach(btn => {
  btn.addEventListener('click', () => openWorkspace(btn.dataset.view));
});
document.getElementById('btn-close-workspace').addEventListener('click', closeWorkspace);
document.getElementById('btn-refresh').addEventListener('click', fetchEcosystem);
document.getElementById('btn-zoom-in').addEventListener('click', () => zoomBy(1.3));
document.getElementById('btn-zoom-out').addEventListener('click', () => zoomBy(1 / 1.3));
document.getElementById('btn-zoom-fit').addEventListener('click', zoomFit);

window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && document.body.classList.contains('workspace-open')) {
    closeWorkspace();
  }
});

fetchEcosystem();
setInterval(() => { if (document.body.classList.contains('workspace-open')) fetchEcosystem(); }, 30000);


/* Where the panel sits, and which side the conversation is on. Taken from
 * OpenClaw's Control UI, which lets a thread move its side panel left, right
 * or below and swap it with the chat — the useful half of a dashboard is
 * being able to put it where the work is. */
const LAYOUTS = ['right', 'left', 'below'];

function applyLayout() {
  let layout = 'right';
  let swapped = false;
  try {
    layout = localStorage.getItem('symbio.layout') || 'right';
    swapped = localStorage.getItem('symbio.swapped') === '1';
  } catch (e) { /* storage blocked: the default layout is fine */ }
  document.body.dataset.layout = LAYOUTS.includes(layout) ? layout : 'right';
  document.body.classList.toggle('swapped', swapped);
}

document.getElementById('btn-layout').addEventListener('click', () => {
  const next = LAYOUTS[(LAYOUTS.indexOf(document.body.dataset.layout || 'right') + 1) % LAYOUTS.length];
  try { localStorage.setItem('symbio.layout', next); } catch (e) {}
  applyLayout();
  if (ecosystem && document.getElementById('view-mindmap').classList.contains('active')) {
    setTimeout(renderMindMap, 200);   // the container just changed size
  }
});

document.getElementById('btn-swap').addEventListener('click', () => {
  const swapped = !document.body.classList.contains('swapped');
  try { localStorage.setItem('symbio.swapped', swapped ? '1' : '0'); } catch (e) {}
  applyLayout();
});

applyLayout();
