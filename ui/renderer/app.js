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
const { readCameraPreview } = require('./camera-preview.cjs');
const { requestBrainbit } = require('./brainbit-http.cjs');

const WS_URL = 'ws://127.0.0.1:7432/ws';
const GESTURE_CONTROL_URL = 'http://127.0.0.1:7432/gestures';
const RECONNECT_MS = 2500;
const DISCONNECTED_MESSAGE = 'The IntuitionOS backend is disconnected. Start the project launcher and wait for reconnection. Your text is preserved.';

// ── DOM refs ──
const hud = document.getElementById('hud');
const cmdInput = document.getElementById('cmd-input');
const expandPanel = document.getElementById('expand-panel');
const planStrip = document.getElementById('plan-strip');
const outputArea = document.getElementById('output-area');
const outputText = document.getElementById('output-text');
const thinkingEl = document.getElementById('thinking-indicator');
const ghostHint = document.getElementById('ghost-hint');
const safeToggle = document.getElementById('safe-toggle');
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
const gestureToggle = document.getElementById('gesture-toggle');
const gestureLabel = document.getElementById('gesture-label');
const gestureStrip = document.getElementById('gesture-strip');
const gestureStatusText = document.getElementById('gesture-status');
const gestureHelp = document.getElementById('gesture-help');
const gestureFeedbackText = document.getElementById('gesture-feedback');
const gestureMotion = document.getElementById('gesture-motion');
const gestureProgressText = document.getElementById('gesture-progress-text');
const gestureMeter = document.getElementById('gesture-meter');
const gestureMeterFill = document.getElementById('gesture-meter-fill');
const gestureClose = document.getElementById('gesture-close');
const gestureGuide = document.getElementById('gesture-guide');
const gestureModeText = document.getElementById('gesture-mode');
const gestureDesktopMode = document.getElementById('gesture-desktop-mode');
const gestureTravel = document.getElementById('gesture-travel');
const gestureTravelValue = document.getElementById('gesture-travel-value');
const gestureSaveSettings = document.getElementById('gesture-save-settings');
const gestureSettingsStatus = document.getElementById('gesture-settings-status');
const handModeMouse = document.getElementById('hand-mode-mouse');
const handModeDesktop = document.getElementById('hand-mode-desktop');
const handModeStatus = document.getElementById('hand-mode-status');
const handClickSound = document.getElementById('hand-click-sound');
const handClickFeedback = document.getElementById('hand-click-feedback');
const gestureModel = document.getElementById('gesture-model');
const gestureBendClick = document.getElementById('gesture-bend-click');
const gesturePreviewToggle = document.getElementById('gesture-preview-toggle');
const gesturePreview = document.getElementById('gesture-preview');
const gesturePreviewCanvas = document.getElementById('gesture-preview-canvas');
const gesturePreviewPose = document.getElementById('gesture-preview-pose');
const gesturePreviewHint = document.getElementById('gesture-preview-hint');
const gesturePreviewModel = document.getElementById('gesture-preview-model');
const gesturePreviewReason = document.getElementById('gesture-preview-reason');
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
const brainbitPanel = document.getElementById('brainbit-panel');
const brainbitSummary = document.getElementById('brainbit-summary');
const brainbitStatusText = document.getElementById('brainbit-status');
const brainbitErrorText = document.getElementById('brainbit-error');
const brainbitDevices = document.getElementById('brainbit-devices');
const brainbitDiscover = document.getElementById('brainbit-discover');
const brainbitConnect = document.getElementById('brainbit-connect');
const brainbitDisconnect = document.getElementById('brainbit-disconnect');
const brainbitRefresh = document.getElementById('brainbit-refresh');
const brainbitDeviceInfo = document.getElementById('brainbit-device-info');

// ── State ──
let ws = null;
let reconnectTimer = null;
let hasConnected = false;
let connectionErrorVisible = false;
let safeMode = null;
let requestedSafeMode = null;
let voiceAvailable = null;
let voiceStatusText = '';
let gestureStatus = null;
let requestedGestures = null;
let gestureRequestId = 0;
let gestureFeedback = '';
let gestureError = '';
let cameraMayBeRunning = false;
let gestureSettingsPending = false;
let handModeError = '';
let clickSoundStatus = null;
let requestedClickSound = null;
let clickSoundRequestId = 0;
let handClickTimer = null;
const seenHandClicks = new Set();
let gestureProgress = null;
let gestureCloseRequest = null;
let gesturePreviewActive = false;
let gesturePreviewEpoch = 0;
let gesturePreviewTimer = null;
let gesturePreviewStaleTimer = null;
let gesturePreviewInFlight = false;
let gesturePreviewFrame = 0;
let bufferTimer = null;
let memoryOpen = false;
let tasksOpen = false;
let expanded = false;
let lastCommand = '';
let isRecording = false;
/** The complete parked action, retained so mode changes can refresh its prompt. */
let pendingConfirm = null;
// Increment on edits; accepting a reply never advances this renderer-owned ID.
let inputRevision = 0;
/** @type {ResolutionMessage|null} */
let resolution = null;
let selectedCorrection = null; // null explicitly keeps the original
let brainbitStatus = null;
let brainbitReady = false;
let brainbitError = '';
let brainbitAction = null;
let brainbitActionId = 0;
let brainbitGeneration = 0;
let brainbitRevision = -1;
let brainbitDeviceList = '';
let brainbitPolling = false;
let brainbitPollTimer = null;
let brainbitReadId = 0;
let brainbitReadInFlight = false;

// ── IPC ──
ipcRenderer.on('focus-input', () => cmdInput.focus());
ipcRenderer.on('desktop-visibility', (_event, status) => {
  if (!status || typeof status.pinned !== 'boolean') return;
  document.getElementById('hud-desktop-indicator').textContent = status.pinned
    ? 'All desktops' : status.state === 'unavailable' ? 'Desktop visibility needs attention' : 'All desktops: preparing…';
  document.getElementById('desktop-visibility-warning').hidden = status.state !== 'unavailable';
  document.getElementById('desktop-visibility-text').textContent = status.text || 'Could not keep the HUD on all desktops.';
  setTimeout(syncHeight, 16);
});
document.getElementById('desktop-visibility-retry').addEventListener('click', () => {
  ipcRenderer.send('desktop-visibility-retry');
});

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
  const socket = new WebSocket(WS_URL);
  ws = socket;
  updateConnectionUI();

  socket.onopen = () => {
    if (socket !== ws) return;
    // A restarted backend cannot honor the previous connection's tokens.
    clearResolution();
    clearConfirm();
    if (hasConnected) inputRevision += 1;
    hasConnected = true;
    voiceAvailable = null;
    voiceStatusText = '';
    gestureStatus = null;
    handModeError = '';
    clickSoundStatus = null;
    requestedClickSound = null;
    ++clickSoundRequestId;
    requestedGestures = null;
    gestureRequestId += 1;
    resetBrainbitConnection();
    cmdInput.placeholder = 'Ask or command…';
    updateConnectionUI();
    if (connectionErrorVisible) {
      outputText.innerHTML = 'Backend connected. Review your command, then press Enter.';
      connectionErrorVisible = false;
    }
    // Refresh the exact current draft after recovery; never replay a submission
    // or approval whose delivery became uncertain during a disconnect.
    send({ type: 'buffer', text: cmdInput.value, client_revision: inputRevision });
  };

  socket.onmessage = (e) => {
    if (socket !== ws || socket.readyState !== WebSocket.OPEN) return;
    try {
      handleMessage(JSON.parse(e.data));
    } catch (_) {}
  };

  socket.onclose = () => disconnected(socket);
  socket.onerror = () => socket.close();
}

function isConnected() {
  return ws && ws.readyState === WebSocket.OPEN;
}

/** Connection state remains visible even when the draft hides the placeholder. */
function updateConnectionUI() {
  const connected = isConnected();
  const modeKnown = typeof safeMode === 'boolean';
  safeLabel.textContent = connected
    ? (requestedSafeMode !== null ? 'CHANGING…' : !modeKnown ? 'CONNECTED' : safeMode ? 'SAFE ON' : 'SAFE OFF')
    : 'OFFLINE';
  safeToggle.disabled = !connected || !modeKnown || requestedSafeMode !== null;
  safeToggle.setAttribute('aria-checked', String(safeMode === true));
  safeToggle.setAttribute('aria-busy', String(requestedSafeMode !== null));
  safeToggle.title = !connected ? 'Safe Mode unavailable: backend disconnected'
    : !modeKnown ? 'Waiting for Safe Mode status'
    : requestedSafeMode !== null ? 'Updating Safe Mode…'
    : safeMode ? 'Turn Safe Mode off. Commands still require approval.' : 'Turn Safe Mode on';
  safeDot.classList.toggle('unsafe', connected && safeMode === false);
  safeDot.style.opacity = connected ? '' : '0.35';
  safeDot.title = connected ? 'Backend connected' : 'Backend disconnected';
  const voiceBlocked = !connected || voiceAvailable === false;
  // Keep the click handler reachable so an unavailable mic can explain why.
  micBtn.setAttribute('aria-disabled', String(voiceBlocked));
  micBtn.style.opacity = voiceBlocked ? '0.45' : '';
  micBtn.title = !connected ? 'Voice unavailable: backend disconnected'
    : voiceStatusText || 'Voice input (Alt+V)';
  updateGestureUI();
  renderBrainbitUI();
  syncBrainbitPolling();
}

/** Change only the mode; a pending action still needs its own explicit answer. */
safeToggle.addEventListener('click', () => {
  if (safeToggle.disabled || !isConnected() || typeof safeMode !== 'boolean') return;
  requestedSafeMode = !safeMode;
  updateConnectionUI();
  if (!send({ type: 'set_safe_mode', enabled: requestedSafeMode })) {
    requestedSafeMode = null;
    showDisconnected();
  }
});

function disconnected(socket) {
  if (socket !== ws) return;
  resetBrainbitConnection();
  safeMode = null;
  requestedSafeMode = null;
  cameraMayBeRunning = cameraMayBeRunning || requestedGestures === true;
  gestureStatus = null;
  clickSoundStatus = null;
  requestedClickSound = null;
  ++clickSoundRequestId;
  requestedGestures = null;
  gestureRequestId += 1;
  gestureFeedback = '';
  gestureError = '';
  gestureSettingsPending = false;
  clearGhost();
  hud.classList.remove('anticipating');
  cmdInput.placeholder = 'Backend disconnected — reconnecting…';
  showDisconnected();
  if (reconnectTimer === null) {
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, RECONNECT_MS);
  }
}

function showDisconnected() {
  clearResolution();
  clearConfirm();
  resetVoice();
  updateConnectionUI();
  onError(DISCONNECTED_MESSAGE);
  connectionErrorVisible = true;
}

/** Return false on lost delivery so callers retain drafts and stop busy states. */
function send(obj) {
  if (!isConnected()) return false;
  try {
    ws.send(JSON.stringify(obj));
    return true;
  } catch (_) {
    const socket = ws;
    socket.close();
    disconnected(socket);
    return false;
  }
}

// ── Message router ──
function handleMessage(msg) {
  switch (msg.type) {
    case 'status':
      onStatus(msg);
      break;
    case 'thinking':
      onThinking(msg);
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
      if (msg.source === 'gestures') {
        requestedGestures = null;
        gestureError = msg.text || 'Could not change hand tracking.';
        updateGestureUI();
        break;
      }
      if (msg.source === 'safe_mode') {
        requestedSafeMode = null;
        updateConnectionUI();
      }
      onError(msg.text, msg.source);
      break;
    case 'voice_status':
      onVoiceStatus(msg);
      break;
    case 'gesture_status':
      onGestureStatus(msg);
      break;
    case 'brainbit_status':
      onBrainbitStatus(msg);
      break;
    case 'gesture':
      onGesture(msg);
      break;
    case 'gesture_progress':
      onGestureProgress(msg);
      break;
    case 'gesture_click':
      onHandClick(msg);
      break;
    case 'gesture_sound':
      onClickSound(msg);
      break;
    case 'gesture_close':
      onGestureClose(msg);
      break;
    case 'gesture_desktop':
      onGestureDesktop(msg);
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
  if (msg.voice) onVoiceStatus(msg.voice);
  if (msg.gestures) onGestureStatus(msg.gestures);
  if (msg.brainbit) onBrainbitStatus(msg.brainbit);
  if (typeof msg.safe_mode === 'boolean') {
    safeMode = msg.safe_mode;
    if (safeMode === requestedSafeMode) requestedSafeMode = null;
    updateConnectionUI();
    renderConfirm();
  }
  if (msg.tasks_count !== undefined) {
    tasksBadge.textContent = msg.tasks_count;
    tasksBadge.classList.toggle('visible', msg.tasks_count > 0);
  }
}

function onThinking(msg = {}) {
  clearStream();
  thinkingEl.classList.add('active');
  hud.classList.remove('anticipating');
  clearGhost();
  // Replace the previous result while the model loads or produces its first token.
  openPanel();
  planStrip.innerHTML = '';
  planStrip.classList.remove('visible');
  const commandEcho = lastCommand ? `<div class="cmd-echo">› ${esc(lastCommand)}</div>` : '';
  const statusText = typeof msg.text === 'string' && msg.text ? msg.text : 'Thinking…';
  outputText.innerHTML = `${commandEcho}<div role="status" aria-live="polite">${esc(statusText)}</div>`;
  outputArea.classList.add('visible');
  setTimeout(syncHeight, 16);
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

function onError(text, source) {
  if (source === 'voice' || /^(Voice |Microphone |No speech detected|Transcription )/i.test(text || '')) {
    resetVoice();
  }
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
  pendingConfirm = { ...msg };
  renderConfirm();
  confirmBar.classList.add('active');
  openPanel();
  cmdInput.focus();
  setTimeout(syncHeight, 16);
}

/** A parked request's original mode must not override newer backend status. */
function renderConfirm() {
  if (!pendingConfirm) return;
  const msg = pendingConfirm;
  const irreversible = msg.reversibility === 'irreversible';
  const requiresSafeModeOff = msg.requires_safe_mode_off === true && safeMode !== false;
  pendingConfirm.allow_safe_mode_change = requiresSafeModeOff;
  confirmBar.classList.toggle('irreversible', irreversible);
  confirmTitle.textContent = requiresSafeModeOff ? 'Turn off Safe Mode and run?'
    : (irreversible ? 'Cannot be undone: ' : 'Confirm: ') + msg.capability;
  confirmAllow.textContent = requiresSafeModeOff ? 'Yes, turn off & run' : 'Yes, run';
  confirmDeny.textContent = 'No, cancel';

  const args = msg.args || {};
  const argText = Object.keys(args).length
    ? Object.entries(args).map(([k, v]) => `${k}=${String(v)}`).join('  ')
    : (msg.summary || (msg.requires_safe_mode_off ? '' : msg.reason) || '');
  const permissionDetail = requiresSafeModeOff
    ? 'Safe Mode will stay off until you turn it on. '
      + (irreversible ? 'This action cannot be undone. ' : '')
    : '';
  confirmDetail.textContent = permissionDetail ? `${permissionDetail.trim()}\n${argText}` : argText;

  setTimeout(syncHeight, 16);
}

/** Answer the gate token; selecting a correction never calls this implicitly. */
function answerConfirm(granted) {
  if (!pendingConfirm) return;
  const answer = { type: 'confirm', token: pendingConfirm.token, granted };
  if (granted && pendingConfirm.requires_safe_mode_off === true) {
    // Bind consent to the wording actually shown, even if another client
    // changes the mode before the backend receives this answer.
    answer.allow_safe_mode_change = pendingConfirm.allow_safe_mode_change;
  }
  if (!send(answer)) {
    showDisconnected();
    return;
  }
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

// ── Camera hand tracking ──

function onGestureStatus(msg) {
  if (msg.click_sound) onClickSound(msg.click_sound);
  if (typeof msg.running !== 'boolean') return;
  const previousState = gestureStatus?.state;
  gestureStatus = { ...msg, state: msg.state || (msg.running ? 'running' : 'off') };
  cameraMayBeRunning = msg.running || ['starting', 'stopping'].includes(gestureStatus.state);
  if ((requestedGestures === true && ['starting', 'running', 'error', 'unavailable'].includes(gestureStatus.state))
      || (requestedGestures === false && ['off', 'stopping', 'error', 'unavailable'].includes(gestureStatus.state))) {
    requestedGestures = null;
  }
  // A failed start may leave the authoritative state "off". Keep its error
  // visible until a retry or a successful transition, including other HUDs.
  if (previousState !== gestureStatus.state
      && ['starting', 'running', 'stopping'].includes(gestureStatus.state)) gestureError = '';
  if (!msg.running) {
    gestureFeedback = '';
    gestureProgress = null;
    gestureCloseRequest = null;
  }
  const trackerBackend = msg.tracker_backend || msg.settings?.tracker_backend || 'mediapipe';
  gestureModel.value = ['rtmpose', 'wilor'].includes(trackerBackend)
    ? trackerBackend : String(msg.model_complexity ?? msg.settings?.model_complexity ?? 1);
  gestureBendClick.checked = (msg.bend_click ?? msg.settings?.bend_click) === true;
  if (msg.settings) {
    gestureDesktopMode.value = msg.settings.desktop_mode || 'auto';
    gestureTravel.value = String(msg.settings.travel_palms || 1.2);
    gestureTravelValue.textContent = `${Number(gestureTravel.value).toFixed(1)} palms`;
  }
  if (msg.close?.pending && msg.running) gestureCloseRequest = msg.close;
  if (previousState !== gestureStatus.state) {
    if (msg.running && previousState !== 'running') ipcRenderer.send('gesture-release-focus');
  }
  updateGestureUI();
}

/** Reflect backend acknowledgement; losing a socket does not turn off its camera. */
function gestureCleanupPending(status = gestureStatus) {
  return status?.tracker_cleanup_pending === true
    || status?.navigation_cleanup_pending === true || status?.desktop?.cleanup_pending === true;
}

function updateGestureUI() {
  const connected = isConnected();
  const known = connected && gestureStatus !== null;
  const state = known ? gestureStatus.state : 'unknown';
  const running = known && gestureStatus.running;
  const cleanupPending = known && gestureCleanupPending();
  const canStop = running || state === 'starting' || cleanupPending;
  const pending = requestedGestures !== null;
  const busy = pending || ['starting', 'stopping'].includes(state);
  const problem = known && (['error', 'unavailable'].includes(state) || !!gestureError);
  const blocked = !known || pending || gestureSettingsPending || state === 'stopping'
    || (gestureStatus?.available === false && !canStop);
  gestureToggle.setAttribute('aria-checked', String(!!running));
  gestureToggle.setAttribute('aria-disabled', String(blocked));
  gestureToggle.setAttribute('aria-busy', String(busy));
  gestureToggle.classList.toggle('active', !!running);
  gestureToggle.classList.toggle('busy', busy);
  gestureToggle.classList.toggle('error', !!problem);
  gestureLabel.textContent = !known ? 'CAMERA ?'
    : requestedGestures === false && cleanupPending ? 'CLEANING…'
    : requestedGestures === false || state === 'stopping' ? 'STOPPING…'
    : requestedGestures === true || state === 'starting' ? 'STARTING…'
    : running ? 'CAMERA ON'
    : cleanupPending ? 'Retry cleanup'
    : state === 'unavailable' ? 'CAMERA N/A'
    : problem ? 'CAMERA ERROR' : 'CAMERA OFF';
  gestureToggle.title = !connected ? 'Camera status unknown: backend disconnected'
    : !known ? 'Waiting for camera status'
    : pending ? 'Waiting for camera acknowledgement…'
    : cleanupPending ? 'Retry releasing hand input and stopping hand tracking.'
    : canStop ? 'Stop hand tracking and release the camera'
    : state === 'stopping' ? 'Releasing the camera…'
    : problem ? `${gestureError || gestureStatus.text} Click to ${gestureStatus.available === false ? 'see details' : 'retry'}.`
    : 'Start hand tracking with your webcam';

  const unknownActive = !known && cameraMayBeRunning;
  gestureStrip.classList.toggle('visible', !!(canStop || state === 'stopping' || problem || unknownActive));
  gestureStrip.classList.toggle('error', !!(problem || unknownActive));
  gestureStatusText.textContent = unknownActive
    ? 'Camera state unknown. It may still be running; press Ctrl+C in the launcher to stop it.'
    : gestureError || (known ? gestureStatus.text || 'Hand tracking is off.' : 'Waiting for camera status.');
  gestureHelp.hidden = !running || state !== 'running';
  const mouseMode = (gestureStatus?.input_mode || gestureStatus?.settings?.input_mode) === 'mouse';
  const bendClick = (gestureStatus?.bend_click ?? gestureStatus?.settings?.bend_click) === true;
  gestureHelp.textContent = mouseMode
    ? 'Point briefly to begin, then relax your fingers to move. Pinch and release to click; hold to drag.'
      + (bendClick ? ' Index bend click is also enabled.' : '')
      + ' Fully open four fingers and hold still, then move to navigate desktops.'
    : 'Hold an open palm still until ready, then move. Hold briefly at 100% to finish automatically; a fist or pinch cancels before completion.';
  document.getElementById('gesture-guide-intro').textContent = mouseMode
    ? 'Hand Mouse: point briefly to begin, then relax your fingers to move. Fully open four fingers for navigation. Pinch and release to click; hold to drag. Make a full fist or lower your hand to pause.'
    : 'Desktop gestures: hold an open palm still until ready. Move sideways for desktops, up for Task View, or down to show the desktop.';
  document.getElementById('mouse-gesture-guide').hidden = !mouseMode;
  document.getElementById('mouse-bend-guide').hidden = !mouseMode || !bendClick;
  document.getElementById('mouse-gesture-note').hidden = !mouseMode;
  document.getElementById('desktop-gesture-guide').hidden = mouseMode;
  document.getElementById('desktop-gesture-note').hidden = mouseMode;
  gestureModeText.hidden = false;
  gestureFeedbackText.textContent = gestureFeedback;
  if (!running) {
    gestureProgress = null;
    gestureCloseRequest = null;
  }
  gestureMotion.hidden = !running || !gestureProgress;
  gestureClose.hidden = !running || !gestureCloseRequest;
  gestureClose.textContent = gestureCloseRequest
    ? `Close “${gestureCloseRequest.title}”?\nHold thumbs-up within ${Math.ceil(gestureCloseRequest.remaining_s || 6)} seconds. Fist or lower hand cancels.` : '';
  const desktop = gestureStatus?.desktop;
  gestureModeText.textContent = desktop?.preference !== 'shortcut'
    && (desktop?.mode === 'native' || desktop?.native_available) && !desktop?.fallback
    ? 'Sideways movement follows your hand through Windows touchpad input. Windows Settings → Touchpad → Four-finger gestures controls desktop switching. Windows handles the animation and final snap. Up/down uses a shortcut after completion for Task View or Show desktop.'
    : desktop?.preference === 'shortcut' || desktop?.fallback || desktop?.text?.startsWith('Measured shortcut fallback')
      ? 'Measured steps: reach 100% and hold briefly to request the action once. There is no live desktop movement in this mode; Windows animates after completion. Up/down uses shortcuts for Task View or Show desktop.'
      : 'Smooth mode checks Windows support for sideways swipes; measured steps take over if unavailable. Up/down always uses a shortcut after completion for Task View or Show desktop.';
  const settingsBlocked = !known || busy || running || cleanupPending || gestureSettingsPending;
  gestureDesktopMode.disabled = settingsBlocked;
  gestureModel.disabled = settingsBlocked;
  gestureBendClick.disabled = settingsBlocked || !mouseMode;
  gestureTravel.disabled = settingsBlocked;
  gestureSaveSettings.disabled = settingsBlocked;
  handModeMouse.disabled = settingsBlocked;
  handModeDesktop.disabled = settingsBlocked;
  handModeMouse.setAttribute('aria-pressed', String(!!known && mouseMode));
  handModeDesktop.setAttribute('aria-pressed', String(!!known && !mouseMode));
  handModeStatus.textContent = !known ? 'Waiting for hand controls…'
    : gestureSettingsPending ? 'Applying hand controls…'
    : cleanupPending ? 'Finish cleanup to change modes'
    : handModeError || (busy || running ? 'Camera off to change modes'
      : mouseMode ? 'Camera on → point to move' : 'Camera on → open palm to swipe');
  if (!known || !running || !mouseMode) {
    clearTimeout(handClickTimer);
    handClickFeedback.hidden = true;
  }
  updateClickSoundUI();
  syncGesturePreview();
  setTimeout(syncHeight, 16);
}

gestureToggle.addEventListener('click', async () => {
  if (!isConnected()) { showDisconnected(); return; }
  if (!gestureStatus || requestedGestures !== null || gestureSettingsPending || gestureStatus.state === 'stopping') return;
  const enabled = !(gestureStatus.running || gestureStatus.state === 'starting'
    || gestureCleanupPending());
  if (enabled && gestureStatus.available === false) {
    gestureError = gestureStatus.text || 'Hand tracking is unavailable.';
    updateGestureUI();
    return;
  }
  requestedGestures = enabled;
  gestureError = '';
  gestureFeedback = '';
  updateGestureUI();
  const requestId = ++gestureRequestId;
  try {
    // A model reply can occupy the command socket. Camera controls must still
    // reach the backend immediately, without submitting or interrupting a draft.
    const response = await fetch(GESTURE_CONTROL_URL, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }), signal: AbortSignal.timeout(5000),
    });
    const result = await response.json();
    if (requestId !== gestureRequestId) return;
    requestedGestures = null;
    if (result.gestures) onGestureStatus(result.gestures);
    if (!response.ok || result.error) {
      gestureError = result.error || 'Could not change hand tracking. Click to retry.';
    }
    updateGestureUI();
  } catch (_) {
    if (requestId !== gestureRequestId) return;
    requestedGestures = null;
    gestureError = 'Camera change could not be confirmed. Click to retry, or press Ctrl+C in the launcher to stop it.';
    updateGestureUI();
  }
});

function onGesture(msg) {
  if (!isConnected() || !gestureStatus?.running) return;
  const names = { swipe_left: 'Swipe left', swipe_right: 'Swipe right', swipe_up: 'Swipe up',
    swipe_down: 'Swipe down', two_finger: 'Two fingers', fist: 'Fist', thumbs_up: 'Thumbs up',
    restore_window: 'One finger', four_finger: 'Four fingers',
    close_request: 'Pinch', close_confirm: 'Thumbs up' };
  const name = names[msg.gesture] || 'Hand gesture';
  gestureFeedback = `${name} — ${msg.ok ? '' : 'Action failed: '}${msg.text || (msg.ok ? 'Done' : 'Please try again')}`;
  updateGestureUI();
}

function onGestureProgress(msg) {
  if (!isConnected() || !gestureStatus?.running) return;
  gestureProgress = msg.state === 'idle' ? null : msg;
  const value = Number.isFinite(msg.progress) ? Math.max(-1, Math.min(1, msg.progress)) : 0;
  const directional = ['desktop', 'overview'].includes(msg.state)
    || (['uncertain', 'committing'].includes(msg.state) && ['horizontal', 'vertical'].includes(msg.axis));
  const navigation = msg.gesture === 'hand_navigation' || directional;
  const mouseProgress = !navigation && (msg.input_mode === 'mouse' || String(msg.state || '').startsWith('mouse_'));
  const showMeter = !['completed', 'cancelled', 'error'].includes(msg.state)
    && (!mouseProgress || ['mouse_arming', 'mouse_pinch', 'mouse_bend'].includes(msg.state));
  const commitReady = msg.state === 'committing';
  const meterWasHidden = gestureMeter.hidden;
  gestureMeter.hidden = !showMeter;
  gestureMeter.classList.toggle('directional', directional);
  gestureMeter.classList.toggle('commit-ready', commitReady);
  gestureMeter.setAttribute('aria-label', navigation ? 'Navigation travel' : 'Hand movement');
  gestureMeter.setAttribute('aria-valuemin', directional ? '-100' : '0');
  gestureMeter.setAttribute('aria-valuenow', String(Math.round(value * 100)));
  gestureMeterFill.style.left = directional ? `${50 + Math.min(0, value) * 50}%` : '0%';
  gestureMeterFill.style.width = `${Math.abs(value) * (directional ? 50 : 100)}%`;
  const releaseHint = ['desktop', 'overview'].includes(msg.state)
    ? ' · Reach 100% and hold briefly' : '';
  gestureProgressText.textContent = `${msg.text || 'Watching your hand'}${gestureProgress && showMeter ? ` · ${Math.round(Math.abs(value) * 100)}%` : ''}${releaseHint}`;
  gestureMeter.setAttribute('aria-valuetext', gestureProgressText.textContent);
  // Native window resize is only needed when the meter appears/disappears.
  const wasHidden = gestureMotion.hidden;
  gestureMotion.hidden = !gestureProgress;
  if (wasHidden !== gestureMotion.hidden || meterWasHidden !== gestureMeter.hidden) setTimeout(syncHeight, 16);
}

function onHandClick(msg) {
  if (!isConnected() || !gestureStatus?.running
      || (gestureStatus.input_mode || gestureStatus.settings?.input_mode) !== 'mouse'
      || typeof msg.id !== 'string' || !msg.id || msg.id.length > 160
      || !['bend', 'pinch'].includes(msg.source) || seenHandClicks.has(msg.id)) return;
  seenHandClicks.add(msg.id);
  if (seenHandClicks.size > 128) seenHandClicks.delete(seenHandClicks.values().next().value);
  clearTimeout(handClickTimer);
  handClickFeedback.textContent = msg.source === 'bend' ? 'Clicked · index bend' : 'Clicked · pinch';
  handClickFeedback.hidden = false;
  handClickTimer = setTimeout(() => { handClickFeedback.hidden = true; }, 650);
}

function updateClickSoundUI() {
  const known = isConnected() && clickSoundStatus && typeof clickSoundStatus.enabled === 'boolean';
  handClickSound.disabled = !known || requestedClickSound !== null || clickSoundStatus?.available === false;
  handClickSound.setAttribute('aria-pressed', String(!!known && clickSoundStatus.enabled));
  handClickSound.textContent = !known ? 'Sound …' : clickSoundStatus.available === false ? 'Sound N/A'
    : clickSoundStatus.error ? 'Sound error' : requestedClickSound !== null ? 'Sound …'
      : clickSoundStatus.enabled ? 'Sound on' : 'Sound off';
  handClickSound.title = !known ? 'Waiting for click sound status'
    : clickSoundStatus.error || (clickSoundStatus.enabled ? 'Mute hand click sounds' : 'Enable hand click sounds');
}

function onClickSound(msg) {
  if (typeof msg.enabled !== 'boolean') return;
  clickSoundStatus = { enabled: msg.enabled, available: msg.available !== false, error: msg.error || '' };
  if (requestedClickSound === msg.enabled) requestedClickSound = null;
  updateClickSoundUI();
}

handClickSound.addEventListener('click', async () => {
  if (!isConnected() || !clickSoundStatus || requestedClickSound !== null || clickSoundStatus.available === false) return;
  requestedClickSound = !clickSoundStatus.enabled;
  const requestId = ++clickSoundRequestId;
  updateClickSoundUI();
  try {
    const response = await fetch(`${GESTURE_CONTROL_URL}/sound`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: requestedClickSound }), signal: AbortSignal.timeout(3000),
    });
    const result = await response.json();
    if (requestId !== clickSoundRequestId) return;
    requestedClickSound = null;
    if (result.click_sound) onClickSound(result.click_sound);
    if (!response.ok || result.error) clickSoundStatus.error = result.error || 'Could not change click sound.';
  } catch (_) {
    if (requestId !== clickSoundRequestId) return;
    requestedClickSound = null;
    clickSoundStatus.error = 'Sound change could not be confirmed. Click to retry.';
  }
  updateClickSoundUI();
});

function onGestureClose(msg) {
  if (!isConnected() || !gestureStatus?.running) return;
  gestureCloseRequest = msg.pending ? msg : null;
  if (msg.pending) ipcRenderer.send('gesture-attention');
  else if (msg.text) gestureFeedback = msg.text;
  updateGestureUI();
}

function onGestureDesktop(msg) {
  if (!isConnected() || !gestureStatus?.running) return;
  gestureStatus.desktop = msg;
  if (msg.error) gestureFeedback = msg.error;
  else if (msg.completed) gestureFeedback = (msg.axis !== 'vertical' && msg.mode === 'native'
    ? 'Desktop swipe completed. Windows handles the final transition.'
    : msg.direction === 'up' ? 'Task View requested.'
      : msg.direction === 'down' ? 'Show desktop toggle requested.' : 'Desktop switch requested.')
    + ' Lower your hand or make a fist before another swipe.';
  updateGestureUI();
}

gestureGuide.addEventListener('toggle', () => { syncGesturePreview(); setTimeout(syncHeight, 16); });
gestureTravel.addEventListener('input', () => {
  gestureTravelValue.textContent = `${Number(gestureTravel.value).toFixed(1)} palms`;
});
async function applyGestureSettings(inputMode) {
  if (!isConnected() || !gestureStatus || gestureStatus.running || gestureSettingsPending
      || gestureCleanupPending()
      || ['starting', 'stopping'].includes(gestureStatus.state)) return;
  gestureSettingsPending = true;
  handModeError = '';
  gestureSettingsStatus.textContent = 'Applying…';
  updateGestureUI();
  const requestId = ++gestureRequestId;
  const trackerBackend = ['rtmpose', 'wilor'].includes(gestureModel.value)
    ? gestureModel.value : 'mediapipe';
  const modelComplexity = trackerBackend !== 'mediapipe'
    ? (gestureStatus.model_complexity ?? gestureStatus.settings?.model_complexity ?? 1)
    : Number(gestureModel.value);
  try {
    const response = await fetch(`${GESTURE_CONTROL_URL}/settings`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ desktop_mode: gestureDesktopMode.value,
        travel_palms: Number(gestureTravel.value), model_complexity: modelComplexity,
        tracker_backend: trackerBackend, bend_click: gestureBendClick.checked === true,
        ...(inputMode ? { input_mode: inputMode } : {}) }), signal: AbortSignal.timeout(5000),
    });
    const result = await response.json();
    if (requestId !== gestureRequestId) return;
    if (result.gestures) onGestureStatus(result.gestures);
    gestureSettingsStatus.textContent = response.ok && !result.error
      ? 'Applied for this session. Turn on the camera when ready.'
      : result.error || 'Could not apply settings.';
    if (inputMode && (!response.ok || result.error)) handModeError = result.error || 'Could not change mode.';
  } catch (_) {
    if (requestId !== gestureRequestId) return;
    gestureSettingsStatus.textContent = 'Settings could not be confirmed. Reconnect and try again.';
    if (inputMode) handModeError = 'Mode change not confirmed. Reconnect and try again.';
  } finally {
    if (requestId === gestureRequestId) {
      gestureSettingsPending = false;
      updateGestureUI();
    }
  }
}
gestureSaveSettings.addEventListener('click', () => applyGestureSettings());
handModeMouse.addEventListener('click', () => applyGestureSettings('mouse'));
handModeDesktop.addEventListener('click', () => applyGestureSettings('desktop'));

// BrainBit controls have their own status and request lifetime. Opening the
// panel reads cached metadata; only these explicit buttons mutate a connection.
const BRAINBIT_STATES = ['disabled', 'unavailable', 'disconnected', 'scanning',
  'connecting', 'connected', 'disconnecting', 'error'];
const BRAINBIT_LABELS = { disabled: 'Disabled', unavailable: 'Unavailable',
  disconnected: 'Disconnected', scanning: 'Discovering devices', connecting: 'Connecting',
  connected: 'Connected', disconnecting: 'Disconnecting', error: 'Needs attention' };
const BRAINBIT_ACTION_LABELS = { discover: 'Discovering devices', connect: 'Connecting',
  disconnect: 'Disconnecting', refresh: 'Refreshing status' };

function brainbitText(value, fallback = '') {
  return typeof value === 'string' && value.trim() ? value.slice(0, 400) : fallback;
}

function validBrainbitStatus(status) {
  return status && BRAINBIT_STATES.includes(status.state)
    && typeof status.available === 'boolean' && typeof status.busy === 'boolean'
    && Array.isArray(status.devices)
    && (status.device === null || (typeof status.device === 'object' && !Array.isArray(status.device)));
}

function resetBrainbitConnection() {
  ++brainbitActionId;
  ++brainbitGeneration;
  ++brainbitReadId;
  brainbitRevision = -1;
  brainbitStatus = null;
  brainbitReady = false;
  brainbitAction = null;
  brainbitError = '';
  brainbitReadInFlight = false;
  brainbitPolling = false;
  clearTimeout(brainbitPollTimer);
  brainbitPollTimer = null;
}

function onBrainbitStatus(status) {
  if (!isConnected() || !validBrainbitStatus(status)) return false;
  if (Number.isInteger(status.revision) && status.revision < brainbitRevision) return false;
  if (Number.isInteger(status.revision)) brainbitRevision = status.revision;
  ++brainbitGeneration;
  brainbitStatus = status;
  brainbitReady = true;
  brainbitError = brainbitText(status.error);
  renderBrainbitUI();
  return true;
}

function renderBrainbitUI() {
  const online = isConnected();
  const known = online && brainbitReady && brainbitStatus !== null;
  const state = known ? brainbitStatus.state : null;
  const connected = state === 'connected';
  const busy = known && (!!brainbitAction || brainbitStatus.busy
    || ['scanning', 'connecting', 'disconnecting'].includes(state));
  const usable = known && brainbitStatus.available && !['disabled', 'unavailable'].includes(state);
  const label = !online ? 'Backend offline' : !known ? 'Status unavailable'
    : BRAINBIT_ACTION_LABELS[brainbitAction] || BRAINBIT_LABELS[state];
  brainbitSummary.textContent = label;
  const statusText = !online
    ? 'Backend offline. Device connection is unconfirmed; waiting for reconnection.'
    : !known ? 'Waiting for current device status. Controls will return when the backend responds.'
    : brainbitAction ? `${label}...`
    : brainbitText(brainbitStatus.text, BRAINBIT_LABELS[state]);
  // Repeated cached polls should not repeat identical live announcements.
  if (brainbitStatusText.textContent !== statusText) brainbitStatusText.textContent = statusText;
  brainbitStatusText.setAttribute('aria-busy', String(!!busy));
  brainbitPanel.classList.toggle('connected', connected);
  brainbitPanel.classList.toggle('busy', !!busy);
  brainbitPanel.classList.toggle('unavailable', !known || ['disabled', 'unavailable', 'error'].includes(state));
  if (brainbitErrorText.textContent !== brainbitError) brainbitErrorText.textContent = brainbitError;
  brainbitErrorText.hidden = !brainbitError || !online;

  const devices = (brainbitStatus?.devices || []).filter(device => device
    && typeof device.id === 'string' && device.id).slice(0, 50);
  const signature = JSON.stringify(devices.map(device => [device.id, device.name, device.family]));
  // Do not rebuild identical options on every poll: keep selection and focus.
  if (signature !== brainbitDeviceList) {
    brainbitDeviceList = signature;
    const selected = brainbitDevices.value;
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = devices.length ? 'Select a discovered device' : 'Discover devices first';
    const options = devices.map((device, index) => {
      const option = document.createElement('option');
      option.value = device.id;
      option.textContent = `${index + 1}. ${brainbitText(device.name, 'BrainBit')}`
        + (brainbitText(device.family) ? ` (${brainbitText(device.family)})` : '');
      return option;
    });
    brainbitDevices.replaceChildren(placeholder, ...options);
    brainbitDevices.value = devices.some(device => device.id === selected) ? selected : '';
  }
  brainbitDiscover.disabled = !usable || busy || connected;
  brainbitDevices.disabled = !usable || busy || connected || !devices.length;
  brainbitConnect.disabled = !usable || busy || connected
    || !devices.some(device => device.id === brainbitDevices.value);
  brainbitDisconnect.disabled = !usable || brainbitAction === 'disconnect' || state === 'disconnecting'
    || (!connected && !busy && !brainbitStatus?.device);
  brainbitDisconnect.textContent = busy && !connected && brainbitAction !== 'disconnect'
    && state !== 'disconnecting' ? 'Cancel / disconnect' : 'Disconnect';
  brainbitRefresh.disabled = !usable || busy;
  brainbitDeviceInfo.hidden = !connected;
  const device = connected ? brainbitStatus.device : null;
  document.getElementById('brainbit-name').textContent = device ? brainbitText(device.name, 'BrainBit') : '';
  document.getElementById('brainbit-family').textContent = device ? brainbitText(device.family, 'Unknown') : '';
  document.getElementById('brainbit-battery').textContent = device
    ? Number.isFinite(device.battery) && device.battery >= 0 && device.battery <= 100
      ? `${Math.round(device.battery)}%` : 'Unknown' : '';
  document.getElementById('brainbit-firmware').textContent = device
    ? brainbitText(device.firmware, 'Unknown') : '';
  setTimeout(syncHeight, 16);
}

function failBrainbitRequest(message) {
  ++brainbitGeneration;
  brainbitReady = false;
  brainbitAction = null;
  brainbitError = message;
  // Do not claim a device disconnected when only its status request failed.
  // Selection can survive recovery, but device metadata and busy state cannot.
  if (brainbitStatus) brainbitStatus = { ...brainbitStatus, device: null, busy: false };
  renderBrainbitUI();
}

function failBrainbitAction(message, generation) {
  const newerTerminalStatus = brainbitReady && generation !== brainbitGeneration
    && !brainbitStatus.busy && !['scanning', 'connecting', 'disconnecting'].includes(brainbitStatus.state);
  if (newerTerminalStatus) {
    // A completed WS update is newer evidence than this older request's
    // transport failure. Keep its connection state and report the uncertainty.
    brainbitAction = null;
    brainbitError = message;
    renderBrainbitUI();
  } else {
    failBrainbitRequest(message);
  }
}

function syncBrainbitPolling() {
  const active = !!(brainbitPanel.open && document.hidden !== true && isConnected());
  if (!active) {
    if (brainbitPolling) {
      ++brainbitReadId;
      brainbitReadInFlight = false;
    }
    brainbitPolling = false;
    clearTimeout(brainbitPollTimer);
    brainbitPollTimer = null;
    return;
  }
  if (!brainbitPolling) {
    brainbitPolling = true;
    pollBrainbitStatus();
  }
}

async function pollBrainbitStatus() {
  if (!brainbitPolling || brainbitReadInFlight) return;
  brainbitPollTimer = null;
  brainbitReadInFlight = true;
  const readId = ++brainbitReadId;
  const generation = brainbitGeneration;
  try {
    const response = await requestBrainbit('status');
    if (readId !== brainbitReadId || generation !== brainbitGeneration || !isConnected()) return;
    if (!response.ok || !validBrainbitStatus(response.data)) {
      failBrainbitRequest(brainbitText(response.data?.error,
        'Device status could not be confirmed. Waiting for the backend to recover.'));
      return;
    }
    onBrainbitStatus(response.data);
  } catch (_) {
    if (readId === brainbitReadId && generation === brainbitGeneration && isConnected())
      failBrainbitRequest('Device status could not be confirmed. Waiting for the backend to recover.');
  } finally {
    if (readId === brainbitReadId) {
      brainbitReadInFlight = false;
      if (brainbitPolling) brainbitPollTimer = setTimeout(pollBrainbitStatus, 1000);
    }
  }
}

async function runBrainbitAction(action) {
  const button = { discover: brainbitDiscover, connect: brainbitConnect,
    disconnect: brainbitDisconnect, refresh: brainbitRefresh }[action];
  if (!button || button.disabled || !isConnected() || !brainbitReady) return;
  // Disconnect is also cancellation. Supersede the old action's eventual reply.
  const actionId = ++brainbitActionId;
  const generation = ++brainbitGeneration;
  brainbitAction = action;
  brainbitError = '';
  renderBrainbitUI();
  try {
    const response = await requestBrainbit(action, action === 'connect' ? brainbitDevices.value : undefined);
    if (actionId !== brainbitActionId || !isConnected()) return;
    if (!response.ok || !validBrainbitStatus(response.data)) {
      failBrainbitAction(brainbitText(response.data?.error,
        'Connection change could not be confirmed. Waiting for current device status.'), generation);
      return;
    }
    brainbitAction = null;
    // A newer WS snapshot wins over an unversioned HTTP snapshot. The next
    // cached GET reconciles the final state if an intermediate event won.
    if (Number.isInteger(response.data.revision) || generation === brainbitGeneration || !brainbitReady)
      onBrainbitStatus(response.data);
    renderBrainbitUI();
  } catch (_) {
    if (actionId === brainbitActionId && isConnected())
      failBrainbitAction('Connection change could not be confirmed. Waiting for current device status.', generation);
  } finally {
    if (actionId === brainbitActionId && brainbitPolling && !brainbitReadInFlight) {
      clearTimeout(brainbitPollTimer);
      pollBrainbitStatus();
    }
  }
}

brainbitPanel.addEventListener('toggle', () => { syncBrainbitPolling(); setTimeout(syncHeight, 16); });
// Native select type-ahead must not trigger the HUD's M/T command shortcuts.
brainbitPanel.addEventListener('keydown', event => event.stopPropagation());
document.addEventListener('visibilitychange', syncBrainbitPolling);
brainbitDevices.addEventListener('change', renderBrainbitUI);
brainbitDiscover.addEventListener('click', () => runBrainbitAction('discover'));
brainbitConnect.addEventListener('click', () => runBrainbitAction('connect'));
brainbitDisconnect.addEventListener('click', () => runBrainbitAction('disconnect'));
brainbitRefresh.addEventListener('click', () => runBrainbitAction('refresh'));

// Preview pulls reuse the recognizer's camera. Opening this panel never starts it.
const HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
  [13,17],[0,17],[17,18],[18,19],[19,20]];
const HAND_POSE_NAMES = { open_palm: 'Open palm', fist: 'Fist', point: 'One finger',
  pinch: 'Pinch', two_finger: 'Two fingers', four_finger: 'Four fingers',
  thumbs_up: 'Thumbs up', none: 'Pose unclear' };
const HAND_CANCEL_REASONS = { hand_lost: 'Hand left the camera view',
  pose_unclear: 'Finger pose stayed unclear too long', tracking_gap: 'Tracking paused too long',
  tracking_jump: 'Hand position jumped', stopped: 'Camera stopped',
  input_failed: 'Windows input failed' };

function clearGesturePreview(text) {
  ++gesturePreviewFrame; // Invalidate an image that is still decoding.
  clearTimeout(gesturePreviewStaleTimer);
  const context = gesturePreviewCanvas.getContext('2d');
  context.clearRect(0, 0, gesturePreviewCanvas.width, gesturePreviewCanvas.height);
  gesturePreviewPose.textContent = text;
  gesturePreviewHint.textContent = '';
  gesturePreviewModel.textContent = '';
  gesturePreviewReason.textContent = '';
}

function syncGesturePreview() {
  gesturePreview.hidden = !gesturePreviewToggle.checked;
  const active = !!(gesturePreviewToggle.checked && gestureGuide.open
    && document.hidden !== true && isConnected() && gestureStatus?.running);
  if (!active) {
    if (gesturePreviewActive) ++gesturePreviewEpoch;
    gesturePreviewActive = false;
    clearTimeout(gesturePreviewTimer);
    gesturePreviewTimer = null;
    clearGesturePreview(!isConnected() ? 'Backend disconnected'
      : !gestureStatus?.running ? 'Turn on the camera to see your hand' : 'Preview hidden');
    return;
  }
  if (!gesturePreviewActive) {
    gesturePreviewActive = true;
    ++gesturePreviewEpoch;
    clearGesturePreview('Waiting for a camera frame…');
    pollGesturePreview();
  }
}

function renderGesturePreview(snapshot, epoch) {
  if (!gesturePreviewActive || epoch !== gesturePreviewEpoch) return;
  if (!snapshot.running || !snapshot.image || !Number.isFinite(snapshot.age_ms)
      || snapshot.age_ms < 0 || snapshot.age_ms > 700) {
    clearGesturePreview(snapshot.running ? 'Waiting for a fresh camera frame…' : 'Camera is off');
    return;
  }
  if (!/^data:image\/jpeg;base64,[A-Za-z0-9+/=]+$/.test(snapshot.image)) {
    clearGesturePreview('Camera frame could not be displayed');
    return;
  }
  const frame = ++gesturePreviewFrame;
  const received = Date.now();
  const picture = new Image();
  picture.onload = () => {
    if (!gesturePreviewActive || epoch !== gesturePreviewEpoch || frame !== gesturePreviewFrame) return;
    if (snapshot.age_ms + Date.now() - received > 700) {
      clearGesturePreview('Waiting for a fresh camera frame…'); return;
    }
    if (Number.isInteger(snapshot.width) && snapshot.width > 0 && snapshot.width <= 480
        && Number.isInteger(snapshot.height) && snapshot.height > 0 && snapshot.height <= 960) {
      gesturePreviewCanvas.width = snapshot.width;
      gesturePreviewCanvas.height = snapshot.height;
    }
    const context = gesturePreviewCanvas.getContext('2d');
    const width = gesturePreviewCanvas.width, height = gesturePreviewCanvas.height;
    context.clearRect(0, 0, width, height);
    context.drawImage(picture, 0, 0, width, height);
    const region = snapshot.control_region;
    if (snapshot.input_mode === 'mouse' && Array.isArray(region) && region.length === 4
        && region.every(v => Number.isFinite(v) && v >= 0 && v <= 1)
        && region[0] < region[2] && region[1] < region[3]) {
      context.strokeStyle = '#93c5fd'; context.lineWidth = 1; context.setLineDash([5, 5]);
      context.strokeRect(region[0] * width, region[1] * height,
        (region[2] - region[0]) * width, (region[3] - region[1]) * height);
      context.setLineDash([]); context.fillStyle = '#bfdbfe'; context.font = '12px sans-serif';
      context.fillText('Pointer area', region[0] * width + 6, region[1] * height + 16);
    }
    const points = snapshot.landmarks;
    if (snapshot.tracked && Array.isArray(points) && points.length === 21
        && points.every(p => Array.isArray(p) && Number.isFinite(p[0]) && Number.isFinite(p[1]))) {
      context.strokeStyle = '#5eead4'; context.lineWidth = 2;
      context.beginPath();
      for (const [a, b] of HAND_CONNECTIONS) {
        context.moveTo(points[a][0] * width, points[a][1] * height);
        context.lineTo(points[b][0] * width, points[b][1] * height);
      }
      context.stroke(); context.fillStyle = '#fef3c7';
      for (const p of points) {
        context.beginPath(); context.arc(p[0] * width, p[1] * height, 3, 0, Math.PI * 2); context.fill();
      }
    }
    const raw = HAND_POSE_NAMES[snapshot.raw_pose] || 'Pose unclear';
    const effective = snapshot.effective_pose || snapshot.pose;
    const mouseStateNames = { mouse_pointer: 'Pointer active', mouse_arming: 'Hold to start',
      mouse_bend: 'Index bend', mouse_clicked: 'Clicked', mouse_pinch: 'Pinching',
      mouse_dragging: 'Dragging', mouse_recovering: 'Pointer frozen · recovering',
      mouse_paused: 'Paused', mouse_error: 'Input needs attention',
      error: 'Input needs attention' };
    gesturePreviewPose.textContent = !snapshot.tracked ? 'No hand detected'
      : snapshot.input_mode === 'mouse' && mouseStateNames[snapshot.state]
        ? mouseStateNames[snapshot.state]
        : `${raw}${effective && effective !== snapshot.raw_pose && effective !== 'none' ? ' · holding swipe' : ''}`;
    const progress = ['desktop', 'overview', 'uncertain', 'committing'].includes(snapshot.state) && Number.isFinite(snapshot.progress)
      ? ` · ${Math.round(Math.abs(snapshot.progress) * 100)}%` : '';
    gesturePreviewHint.textContent = (snapshot.hint || 'Face one open palm toward the camera.') + progress;
    const fps = Number.isFinite(snapshot.fps) ? `${snapshot.fps.toFixed(0)} tracking FPS` : '';
    gesturePreviewModel.textContent = [snapshot.model_name || snapshot.model, fps].filter(Boolean).join(' · ');
    const reason = snapshot.last_cancel_reason || snapshot.last_reason;
    gesturePreviewReason.textContent = reason ? `Last cancellation: ${HAND_CANCEL_REASONS[reason] || reason}` : '';
    clearTimeout(gesturePreviewStaleTimer);
    gesturePreviewStaleTimer = setTimeout(() => {
      if (gesturePreviewActive && epoch === gesturePreviewEpoch && frame === gesturePreviewFrame)
        clearGesturePreview('Waiting for a fresh camera frame…');
    }, Math.max(1, 700 - snapshot.age_ms - (Date.now() - received)));
  };
  picture.onerror = () => {
    if (gesturePreviewActive && epoch === gesturePreviewEpoch && frame === gesturePreviewFrame)
      clearGesturePreview('Camera frame could not be displayed');
  };
  picture.src = snapshot.image;
}

async function pollGesturePreview() {
  if (!gesturePreviewActive || gesturePreviewInFlight) return;
  gesturePreviewInFlight = true;
  const epoch = gesturePreviewEpoch;
  try {
    renderGesturePreview(await readCameraPreview(), epoch);
  } catch (_) {
    if (gesturePreviewActive && epoch === gesturePreviewEpoch)
      clearGesturePreview('Preview unavailable. Restart the backend to load the update.');
  } finally {
    gesturePreviewInFlight = false;
    if (gesturePreviewActive) gesturePreviewTimer = setTimeout(pollGesturePreview, 125);
  }
}

gesturePreviewToggle.addEventListener('change', () => { syncGesturePreview(); setTimeout(syncHeight, 16); });
document.addEventListener('visibilitychange', syncGesturePreview);

// ── Voice ──

function resetVoice() {
  isRecording = false;
  micBtn.classList.remove('recording', 'transcribing');
  recordingBar.classList.remove('active');
}

function onVoiceStatus(msg) {
  voiceAvailable = msg.available;
  voiceStatusText = msg.text || '';
  if (['ready', 'disabled', 'error'].includes(msg.state)) resetVoice();
  updateConnectionUI();
}

function startVoice() {
  if (isRecording) {
    stopVoice();
    return;
  }
  if (!isConnected()) {
    showDisconnected();
    return;
  }
  if (voiceAvailable === false) {
    onError(voiceStatusText || 'Voice input is unavailable.', 'voice');
    return;
  }
  if (!send({ type: 'voice_start' })) {
    showDisconnected();
    return;
  }
  isRecording = true;
  micBtn.classList.add('recording');
  recordingBar.classList.add('active');
  recLabel.textContent = 'Listening...';
  setTimeout(syncHeight, 16);
}

function stopVoice() {
  if (!isRecording) return;
  if (!send({ type: 'voice_stop' })) showDisconnected();
}

function onVoiceRecording(msg) {
  if (msg.active) {
    isRecording = true;
    micBtn.classList.add('recording');
    recordingBar.classList.add('active');
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
  resetVoice();
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
    if (!isConnected()) {
      showDisconnected();
      return;
    }
    if (!resolution || resolution.original !== text || resolution.client_revision !== inputRevision) {
      if (!send({ type: 'resolve', text, client_revision: inputRevision })) showDisconnected();
      return;
    }
    const selectedText = selectedCorrection === null ? text : resolution.candidates[selectedCorrection].text;
    if (!send({ type: 'input', text, selected_text: selectedText, token: resolution.token,
      revision: resolution.revision, candidate_index: selectedCorrection, client_revision: inputRevision })) {
      showDisconnected();
      return;
    }
    lastCommand = selectedText;
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
