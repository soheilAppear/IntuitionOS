/**
 * HUD presentation and input controller.
 *
 * The renderer owns the editable draft and selected visible candidate. The
 * backend owns ranking, correction feedback, and capability enforcement;
 * Electron's main process owns native window visibility and size.
 *
 * Three IDs serve different purposes: inputRevision rejects stale wire replies,
 * a resolution's token/revision binds displayed text in the core, and a separate
 * confirmation token identifies an action parked by the capability gate.
 */

/**
 * @typedef {Object} CorrectionCandidate
 * @property {string} text Full replacement command, including unchanged arguments.
 * @property {string} token Replacement command/subcommand text; not an approval.
 * @property {[number, number]} span Changed span in the original command.
 * @property {string} [reason] Explanation supplied by the shared resolver.
 */

/**
 * @typedef {Object} ResolutionMessage
 * @property {string} original Exact draft for which the snapshot was generated.
 * @property {'exact'|'incomplete'|'correction'|'ambiguous'|'unsupported'} status
 * @property {CorrectionCandidate[]} candidates Ordered by the core resolver.
 * @property {string} token One-use correction commitment, not execution permission.
 * @property {number} revision Core correction-session revision.
 * @property {number|null} client_revision Renderer draft revision echoed by the server.
 * @property {string} [reason]
 */

const { ipcRenderer } = require('electron');

const WS_URL = 'ws://127.0.0.1:7432/ws';
const RECONNECT_MS = 2500;

// ── DOM refs ──
const hud = document.getElementById('hud');
const cmdInput = document.getElementById('cmd-input');
const expandPanel = document.getElementById('expand-panel');
const planStrip = document.getElementById('plan-strip');
const outputArea = document.getElementById('output-area');
const outputText = document.getElementById('output-text');
const thinkingEl = document.getElementById('thinking-indicator');
const ghostHint = document.getElementById('ghost-hint');
const safeDot = document.getElementById('safe-dot');
const safeLabel = document.getElementById('safe-label');
const tasksBadge = document.getElementById('tasks-badge');
const memoryPanel = document.getElementById('memory-panel');
const tasksPanel = document.getElementById('tasks-panel');
const memoryList = document.getElementById('memory-list');
const tasksList = document.getElementById('tasks-list');
const tasksBtn = document.getElementById('tasks-btn');
const memoryBtn = document.getElementById('memory-btn');
const closeBtn = document.getElementById('close-btn');
const toastRoot = document.getElementById('toast-root');
const micBtn = document.getElementById('mic-btn');
const recordingBar = document.getElementById('recording-bar');
const recLabel = document.getElementById('rec-label');
const confirmBar = document.getElementById('confirm-bar');
const confirmTitle = document.getElementById('confirm-title');
const confirmDetail = document.getElementById('confirm-detail');
const confirmAllow = document.getElementById('confirm-allow');
const confirmDeny = document.getElementById('confirm-deny');
const correctionBar = document.getElementById('correction-bar');
const correctionLabel = document.getElementById('correction-label');
const correctionChoices = document.getElementById('correction-choices');

// ── State ──
let ws = null;
let bufferTimer = null;
let memoryOpen = false;
let tasksOpen = false;
let expanded = false;
let lastCommand = '';
let isRecording = false;
/** @type {{token: string, capability: string}|null} */
let pendingConfirm = null;
// Increment on edits; accepting a reply never advances this renderer-owned ID.
let inputRevision = 0;
/** @type {ResolutionMessage|null} */
let resolution = null;
let selectedCorrection = null; // null explicitly keeps the original

// ── IPC ──
ipcRenderer.on('focus-input', () => cmdInput.focus());

closeBtn.addEventListener('click', () => {
  // Hiding preserves the renderer; main.js owns the native window lifecycle.
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
/** Connect to the local backend and discard stale UI commitments on disconnect. */
function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    cmdInput.placeholder = 'Ask or command…';
  };

  ws.onmessage = (e) => {
    try {
      handleMessage(JSON.parse(e.data));
    } catch (_) {}
  };

  ws.onclose = () => {
    clearResolution();
    clearConfirm();
    cmdInput.placeholder = 'Reconnecting…';
    setTimeout(connect, RECONNECT_MS);
  };

  ws.onerror = () => ws.close();
}

/** Send one protocol message when connected; the backend validates its contents. */
function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// ── Message router ──
function handleMessage(msg) {
  switch (msg.type) {
    case 'status':
      onStatus(msg);
      break;
    case 'thinking':
      onThinking();
      break;
    case 'reply':
      onReply(msg);
      break;
    case 'anticipation':
      onAnticipation(msg);
      break;
    case 'memory':
      onMemory(msg.rows);
      break;
    case 'tasks':
      onTasks(msg.rows);
      break;
    case 'reminder':
      onReminder(msg);
      break;
    case 'error':
      onError(msg.text);
      break;
    case 'voice_recording':
      onVoiceRecording(msg);
      break;
    case 'voice_transcribing':
      onVoiceTranscribing();
      break;
    case 'voice_text':
      onVoiceText(msg.text);
      break;
    case 'confirm_request':
      onConfirmRequest(msg);
      break;
    case 'resolution':
      onResolution(msg);
      break;
    case 'input_invalidated':
      clearResolution();
      clearConfirm();
      break;
    case 'exit':
      ipcRenderer.send('hide-window');
      break;
    case 'token':
      onToken(msg.text);
      break;
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
  clearStream();
  thinkingEl.classList.add('active');
  hud.classList.remove('anticipating');
  clearGhost();
}

function clearGhost() {
  ghostHint.textContent = '';
  ghostHint.title = '';
  ghostHint.style.opacity = '';
  ghostHint.classList.remove('confident');
}

function onReply(msg) {
  thinkingEl.classList.remove('active');
  clearStream();
  if (pendingConfirm) clearConfirm();
  hud.classList.remove('anticipating');
  clearGhost();

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
  if (resolution && resolution.candidates.length) return;
  if (msg.text !== cmdInput.value.trim()) return;
  hud.classList.remove('anticipating');
  const d = msg.data || {};

  // A next-action hint can stand on its own when there is no cheap result to warm.
  let hint = '';
  if (d.reply) {
    hint = d.reply.slice(0, 62) + (d.reply.length > 62 ? '…' : '');
  } else if (d.action) {
    hint = d.action;
  } else if (d.plan && d.plan.length) {
    hint = d.plan[0];
  }
  if (!hint) return;

  ghostHint.textContent = '→ ' + hint;
  ghostHint.title = d.why || '';

  // Next-action confidence controls hint emphasis. Deterministic correction
  // scores are ranked separately and never rendered as probabilities.
  const floor = typeof msg.reveal_threshold === 'number' ? msg.reveal_threshold : 0.7;
  const conf  = typeof d.confidence === 'number' ? d.confidence : floor;
  const span  = Math.max(1e-6, 1 - floor);
  const t     = Math.max(0, Math.min(1, (conf - floor) / span));

  ghostHint.style.opacity = (0.42 + 0.58 * t).toFixed(3);
  ghostHint.classList.toggle('confident', t > 0.6);
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
  clearStream();
  openPanel();
  outputText.innerHTML = `<span class="error-text">⚠  ${esc(text)}</span>`;
  outputArea.classList.add('visible');
  setTimeout(syncHeight, 16);
}

// ── Streaming ──
//
// A turn can now span several tool iterations, so the panel shows tokens as they
// land rather than staying blank for the whole round trip. onReply overwrites
// this with the final text, which is the authoritative version.

let streamBuffer = '';

function onToken(piece) {
  if (!piece) return;
  if (!streamBuffer) {
    openPanel();
    outputArea.classList.add('visible');
  }
  streamBuffer += piece;
  outputText.innerHTML = `<span class="streaming">${esc(streamBuffer)}</span>`;
  setTimeout(syncHeight, 16);
}

function clearStream() {
  streamBuffer = '';
}

// ── Confirmation ──
//
// The gate parked an action. Nothing has run and nothing will until this is
// answered, so the bar stays up and Enter/Escape are borrowed for the answer
// rather than submitting a new command on top of a pending one.

/** Display a parked action only if it still belongs to the current draft. */
function onConfirmRequest(msg) {
  // An edit may have happened while dispatch was awaiting the capability gate.
  if (msg.client_revision !== undefined && msg.client_revision !== null && msg.client_revision !== inputRevision) {
    send({ type: 'confirm', token: msg.token, granted: false });
    return;
  }
  thinkingEl.classList.remove('active');
  pendingConfirm = { token: msg.token, capability: msg.capability };

  const irreversible = msg.reversibility === 'irreversible';
  confirmBar.classList.toggle('irreversible', irreversible);
  confirmTitle.textContent = (irreversible ? 'Cannot be undone: ' : 'Confirm: ') + msg.capability;

  const args = msg.args || {};
  const argText = Object.keys(args).length
    ? Object.entries(args).map(([k, v]) => `${k}=${String(v)}`).join('  ')
    : (msg.summary || msg.reason || '');
  confirmDetail.textContent = argText;

  confirmBar.classList.add('active');
  openPanel();
  cmdInput.focus();
  setTimeout(syncHeight, 16);
}

/** Answer the gate token; selecting a correction never calls this implicitly. */
function answerConfirm(granted) {
  if (!pendingConfirm) return;
  send({ type: 'confirm', token: pendingConfirm.token, granted });
  clearConfirm();
  if (granted) onThinking();
}

function clearConfirm() {
  pendingConfirm = null;
  confirmBar.classList.remove('active', 'irreversible');
  confirmTitle.textContent = '';
  confirmDetail.textContent = '';
  setTimeout(syncHeight, 16);
}

confirmAllow.addEventListener('click', () => answerConfirm(true));
confirmDeny.addEventListener('click',  () => answerConfirm(false));

// ── Voice ──

function startVoice() {
  if (isRecording) {
    stopVoice();
    return;
  }
  isRecording = true;
  micBtn.classList.add('recording');
  recordingBar.classList.add('active');
  recLabel.textContent = 'Listening...';
  send({ type: 'voice_start' });
  setTimeout(syncHeight, 16);
}

function stopVoice() {
  if (!isRecording) return;
  send({ type: 'voice_stop' });
}

function onVoiceRecording(msg) {
  if (msg.active) {
    recLabel.textContent = 'Listening…';
  } else {
    // mic closed — transcription in progress
    isRecording = false;
    micBtn.classList.remove('recording');
    micBtn.classList.add('transcribing');
    recLabel.textContent = 'Transcribing…';
  }
}

function onVoiceTranscribing() {
  recLabel.textContent = 'Transcribing…';
}

function onVoiceText(text) {
  micBtn.classList.remove('recording', 'transcribing');
  recordingBar.classList.remove('active');
  isRecording = false;
  if (text) {
    cmdInput.value = text;
    inputChanged();
    cmdInput.focus();
    showToast('Speech is ready to review. Enter submits the selected command.');
  }
  setTimeout(syncHeight, 16);
}

micBtn.addEventListener('click', () => {
  if (isRecording) stopVoice(); else startVoice();
});

// IPC from main process (Alt+V global shortcut)
ipcRenderer.on('voice-toggle', () => {
  if (isRecording) stopVoice(); else startVoice();
});

// ── Toast ──
function showToast(text) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = text;
  toastRoot.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

// ── Visible command resolution ──
/** Remove rendered choices whenever their draft or server state is invalidated. */
function clearResolution() {
  resolution = null;
  selectedCorrection = null;
  correctionChoices.replaceChildren();
  correctionBar.classList.remove('active');
  setTimeout(syncHeight, 16);
}

/** @param {ResolutionMessage} msg Snapshot must match both text and draft ID. */
function onResolution(msg) {
  if (msg.original !== cmdInput.value || msg.client_revision !== inputRevision) return;
  resolution = msg;
  selectedCorrection = msg.candidates.length ? 0 : null;
  clearGhost();
  renderResolution();
}

/** Render full commands as text nodes; highlight only the provider's token span. */
function renderResolution() {
  if (!resolution) return;
  correctionChoices.replaceChildren();
  correctionLabel.textContent = resolution.candidates.length
    ? `${resolution.status}: review the highlighted change`
    : (resolution.status === 'exact' ? 'Exact command · Enter submits unchanged' : resolution.reason || 'Enter submits unchanged');
  const choices = [...resolution.candidates.map((c, index) => ({ ...c, index })),
    { text: resolution.original, index: null }];
  for (const candidate of choices) {
    const button = document.createElement('button');
    button.className = 'correction-choice';
    button.classList.toggle('selected', candidate.index === selectedCorrection);
    button.setAttribute('role', 'option');
    button.setAttribute('aria-selected', String(candidate.index === selectedCorrection));
    const caption = document.createElement('span');
    caption.className = 'choice-caption';
    caption.textContent = candidate.index === null ? 'Keep original' : `Suggestion ${candidate.index + 1}`;
    button.appendChild(caption);
    if (candidate.index !== null && candidate.span) {
      const start = candidate.span[0];
      const end = start + candidate.token.length;
      button.appendChild(document.createTextNode(candidate.text.slice(0, start)));
      const changed = document.createElement('mark');
      changed.textContent = candidate.text.slice(start, end);
      button.appendChild(changed);
      button.appendChild(document.createTextNode(candidate.text.slice(end)));
      button.title = candidate.reason || '';
    } else {
      button.appendChild(document.createTextNode(candidate.text));
    }
    button.addEventListener('click', () => {
      selectedCorrection = candidate.index;
      renderResolution();
      cmdInput.focus();
    });
    correctionChoices.appendChild(button);
  }
  correctionBar.classList.toggle('active', !!resolution.original.trim());
  setTimeout(syncHeight, 16);
}

/** Cycle through ranked candidates, followed by the explicit original choice. */
function cycleCorrection(delta) {
  if (!resolution || !resolution.candidates.length) return;
  const count = resolution.candidates.length;
  const current = selectedCorrection === null ? count : selectedCorrection;
  const next = (current + delta + count + 1) % (count + 1);
  selectedCorrection = next === count ? null : next;
  renderResolution();
}

// ── Input ──
/** Revoke local commitments and notify the server immediately for every edit. */
function inputChanged() {
  const val = cmdInput.value;
  inputRevision += 1;
  clearResolution();
  clearConfirm();
  clearGhost();
  hud.classList.toggle('anticipating', !!val.trim());
  clearTimeout(bufferTimer);
  // Send even empty edits immediately: the server must revoke approvals before
  // any following Enter/click can submit a stale command.
  send({ type: 'buffer', text: val, client_revision: inputRevision });
}
cmdInput.addEventListener('input', inputChanged);

cmdInput.addEventListener('keydown', (e) => {
  // A parked action owns Enter and Escape until it is answered.
  if (pendingConfirm) {
    if (e.key === 'Enter') {
      e.preventDefault();
      answerConfirm(true);
      return;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      answerConfirm(false);
      return;
    }
  }

  if ((e.key === 'ArrowDown' || (e.ctrlKey && e.key.toLowerCase() === 'n')) && resolution) {
    e.preventDefault();
    cycleCorrection(1);
    return;
  }
  if ((e.key === 'ArrowUp' || (e.ctrlKey && e.key.toLowerCase() === 'p')) && resolution) {
    e.preventDefault();
    cycleCorrection(-1);
    return;
  }
  if (e.key === 'Escape' && resolution && resolution.candidates.length) {
    e.preventDefault();
    selectedCorrection = null;
    renderResolution();
    return;
  }

  if (e.key === 'Enter') {
    e.preventDefault();
    const text = cmdInput.value;
    if (!text.trim()) return;
    if (!resolution || resolution.original !== text || resolution.client_revision !== inputRevision) {
      send({ type: 'resolve', text, client_revision: inputRevision });
      return;
    }
    const selectedText = selectedCorrection === null ? text : resolution.candidates[selectedCorrection].text;
    lastCommand = selectedText;
    send({ type: 'input', text, selected_text: selectedText, token: resolution.token,
      revision: resolution.revision, candidate_index: selectedCorrection, client_revision: inputRevision });
    // Clearing a submitted draft is not a new edit: an arriving approval still
    // belongs to this revision. The next real input event will invalidate it.
    cmdInput.value = '';
    clearResolution();
    clearGhost();
    hud.classList.remove('anticipating');
    clearTimeout(bufferTimer);
    return;
  }

  if (e.key === 'Escape') {
    if (expanded) {
      collapsePanel();
      if (memoryOpen) toggleMemory();
      if (tasksOpen)  toggleTasks();
    } else {
      cmdInput.value = '';
      inputChanged();
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
/** Escape text for the response panels that use HTML templates. */
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
