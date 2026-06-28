const { ipcRenderer } = require('electron');

const WS_URL = 'ws://127.0.0.1:7432/ws';
const RECONNECT_MS = 2500;

// ── DOM refs ──
const hud          = document.getElementById('hud');
const cmdInput     = document.getElementById('cmd-input');
const expandPanel  = document.getElementById('expand-panel');
const planStrip    = document.getElementById('plan-strip');
const outputArea   = document.getElementById('output-area');
const outputText   = document.getElementById('output-text');
const thinkingEl   = document.getElementById('thinking-indicator');
const ghostHint    = document.getElementById('ghost-hint');
const safeDot      = document.getElementById('safe-dot');
const safeLabel    = document.getElementById('safe-label');
const tasksBadge   = document.getElementById('tasks-badge');
const memoryPanel  = document.getElementById('memory-panel');
const tasksPanel   = document.getElementById('tasks-panel');
const memoryList   = document.getElementById('memory-list');
const tasksList    = document.getElementById('tasks-list');
const tasksBtn     = document.getElementById('tasks-btn');
const memoryBtn    = document.getElementById('memory-btn');
const closeBtn     = document.getElementById('close-btn');
const toastRoot    = document.getElementById('toast-root');

// ── State ──
let ws = null;
let bufferTimer = null;
let memoryOpen  = false;
let tasksOpen   = false;
let expanded    = false;
let lastCommand = '';

// ── IPC ──
ipcRenderer.on('focus-input', () => cmdInput.focus());

closeBtn.addEventListener('click', () => {
  // Just hide by sending the toggle shortcut — main.js owns visibility
  ipcRenderer.send('hide-window');
});

// ── Height sync ──
function syncHeight() {
  const h = hud.scrollHeight;
  ipcRenderer.send('resize', h + 2);
}

// ── Expand / collapse ──
function openPanel() {
  expanded = true;
  expandPanel.classList.add('open');
  setTimeout(syncHeight, 16);
}

function collapsePanel() {
  expanded = false;
  expandPanel.classList.remove('open');
  outputArea.classList.remove('visible');
  planStrip.classList.remove('visible');
  outputText.innerHTML = '';
  planStrip.innerHTML = '';
  setTimeout(syncHeight, 16);
}

// ── WebSocket ──
function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    cmdInput.placeholder = 'Ask or command…';
  };

  ws.onmessage = (e) => {
    try { handleMessage(JSON.parse(e.data)); }
    catch (_) {}
  };

  ws.onclose = () => {
    cmdInput.placeholder = 'Reconnecting…';
    setTimeout(connect, RECONNECT_MS);
  };

  ws.onerror = () => ws.close();
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// ── Message router ──
function handleMessage(msg) {
  switch (msg.type) {
    case 'status':      onStatus(msg);       break;
    case 'thinking':    onThinking();        break;
    case 'reply':       onReply(msg);        break;
    case 'anticipation':onAnticipation(msg); break;
    case 'memory':      onMemory(msg.rows);  break;
    case 'tasks':       onTasks(msg.rows);   break;
    case 'reminder':    onReminder(msg);     break;
    case 'error':       onError(msg.text);   break;
  }
}

// ── Handlers ──

function onStatus(msg) {
  if (msg.safe_mode !== undefined) {
    const safe = msg.safe_mode;
    safeDot.classList.toggle('unsafe', !safe);
    safeLabel.textContent = safe ? 'SAFE' : 'UNSAFE';
  }
  if (msg.tasks_count !== undefined) {
    tasksBadge.textContent = msg.tasks_count;
    tasksBadge.classList.toggle('visible', msg.tasks_count > 0);
  }
}

function onThinking() {
  thinkingEl.classList.add('active');
  hud.classList.remove('anticipating');
  ghostHint.textContent = '';
}

function onReply(msg) {
  thinkingEl.classList.remove('active');
  hud.classList.remove('anticipating');
  ghostHint.textContent = '';

  const plan  = Array.isArray(msg.plan) ? msg.plan : [];
  const text  = msg.text || '';
  const cache = msg.from_cache;

  openPanel();

  if (plan.length) {
    planStrip.innerHTML = plan.map(p => `<div class="plan-item">${esc(p)}</div>`).join('');
    planStrip.classList.add('visible');
  } else {
    planStrip.classList.remove('visible');
  }

  if (text) {
    let html = '';
    if (lastCommand) html += `<div class="cmd-echo">› ${esc(lastCommand)}</div>`;
    if (cache) html += `<div class="cache-tag">⚡ cached</div>\n`;
    html += esc(text);
    outputText.innerHTML = html;
    outputArea.classList.add('visible');
  } else if (!plan.length) {
    collapsePanel();
    return;
  }

  setTimeout(syncHeight, 16);
}

function onAnticipation(msg) {
  if (msg.text !== cmdInput.value.trim()) return;
  hud.classList.remove('anticipating');
  const d = msg.data || {};
  if (d.plan && d.plan.length) {
    ghostHint.textContent = '→ ' + d.plan[0];
  } else if (d.reply) {
    ghostHint.textContent = d.reply.slice(0, 62) + (d.reply.length > 62 ? '…' : '');
  }
}

function onMemory(rows) {
  if (!rows || !rows.length) {
    memoryList.innerHTML = '<div class="empty-hint">No memories yet.</div>';
    return;
  }
  memoryList.innerHTML = rows.map(r => {
    const preview = r.text.slice(0, 90) + (r.text.length > 90 ? '…' : '');
    return `<div class="panel-item"><span class="role-tag">${esc(r.role)}</span>${esc(preview)}</div>`;
  }).join('');
  setTimeout(syncHeight, 16);
}

function onTasks(rows) {
  if (!rows || !rows.length) {
    tasksList.innerHTML = '<div class="empty-hint">No pending tasks.</div>';
    return;
  }
  tasksList.innerHTML = rows.map(r => {
    const due = r.due ? new Date(r.due * 1000).toLocaleString() : '—';
    return `<div class="panel-item">
      <span class="task-id">#${r.id}</span>${esc(r.title)}
      <span class="task-due">${due}</span>
    </div>`;
  }).join('');
  setTimeout(syncHeight, 16);
}

function onReminder(msg) {
  hud.classList.add('reminder-flash');
  setTimeout(() => hud.classList.remove('reminder-flash'), 2000);
  showToast(`⏰  ${msg.title}`);
  send({ type: 'get_status' });
}

function onError(text) {
  thinkingEl.classList.remove('active');
  openPanel();
  outputText.innerHTML = `<span class="error-text">⚠  ${esc(text)}</span>`;
  outputArea.classList.add('visible');
  setTimeout(syncHeight, 16);
}

// ── Toast ──
function showToast(text) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = text;
  toastRoot.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

// ── Input ──
cmdInput.addEventListener('input', () => {
  const val = cmdInput.value;
  ghostHint.textContent = '';

  if (!val.trim()) {
    hud.classList.remove('anticipating');
    return;
  }

  hud.classList.add('anticipating');

  clearTimeout(bufferTimer);
  bufferTimer = setTimeout(() => {
    send({ type: 'buffer', text: val });
  }, 90);
});

cmdInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const text = cmdInput.value.trim();
    if (!text) return;
    lastCommand = text;
    cmdInput.value = '';
    ghostHint.textContent = '';
    hud.classList.remove('anticipating');
    clearTimeout(bufferTimer);
    send({ type: 'input', text });
    return;
  }

  if (e.key === 'Escape') {
    if (expanded) {
      collapsePanel();
      if (memoryOpen) toggleMemory();
      if (tasksOpen)  toggleTasks();
    } else {
      cmdInput.value = '';
      ghostHint.textContent = '';
      hud.classList.remove('anticipating');
    }
  }
});

// ── Panel toggles ──
function toggleMemory() {
  memoryOpen = !memoryOpen;
  if (memoryOpen) {
    openPanel();
    memoryPanel.classList.add('open');
    memoryBtn.classList.add('active');
    send({ type: 'input', text: '/memory' });
  } else {
    memoryPanel.classList.remove('open');
    memoryBtn.classList.remove('active');
    if (!tasksOpen && !outputArea.classList.contains('visible')) collapsePanel();
  }
  setTimeout(syncHeight, 16);
}

function toggleTasks() {
  tasksOpen = !tasksOpen;
  if (tasksOpen) {
    openPanel();
    tasksPanel.classList.add('open');
    tasksBtn.classList.add('active');
    send({ type: 'input', text: '/tasks' });
  } else {
    tasksPanel.classList.remove('open');
    tasksBtn.classList.remove('active');
    if (!memoryOpen && !outputArea.classList.contains('visible')) collapsePanel();
  }
  setTimeout(syncHeight, 16);
}

memoryBtn.addEventListener('click', toggleMemory);
tasksBtn.addEventListener('click', toggleTasks);

// Keyboard shortcuts for panel toggles
document.addEventListener('keydown', (e) => {
  if (document.activeElement === cmdInput) return;
  if (e.key === 'm' || e.key === 'M') toggleMemory();
  if (e.key === 't' || e.key === 'T') toggleTasks();
});

// ── Utility ──
function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Boot ──
connect();
cmdInput.focus();
