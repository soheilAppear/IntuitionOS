// Experimental display/controller. Sensor frames remain transient in this renderer.
const { requestMultimodal } = require('./multimodal-http.cjs');
const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
  [13,17],[0,17],[17,18],[18,19],[19,20]];

function createMultimodalPanel({ document, isConnected, isCameraRunning = () => false,
  isBrainbitConnected = () => false, onResize = () => {}, request = requestMultimodal,
  imageFactory = () => new Image(), schedule = setTimeout, cancel = clearTimeout, now = Date.now }) {
  const ids = ['panel','summary','status','error','start','stop','contact','arm','start-hint',
    'camera','eeg','camera-status','eeg-status','motion','diagnostics','contact-status',
    'left','right','reset','trial','calibration-status'];
  const el = Object.fromEntries(ids.map(id => [id, document.getElementById(`multimodal-${id}`)]));
  let status = null, known = false, error = '', pending = null, epoch = 0, actionId = 0;
  let generation = 0, revision = -1, readId = 0, reading = false, polling = false;
  let pollTimer = null, staleTimer = null, frameId = 0, receivedAt = 0;
  let lastOnline = false;
  const text = (value, fallback = '') => typeof value === 'string' && value.trim() ? value.slice(0, 600) : fallback;
  const put = (id, value) => { if (el[id].textContent !== value) el[id].textContent = value; };
  const visible = () => el.panel.open && document.hidden !== true;
  const valid = data => data && typeof data.state === 'string' && typeof data.running === 'boolean'
    && typeof data.armed === 'boolean';
  const age = value => Number.isFinite(value) && value >= 0 ? value + (now() - receivedAt) / 1000 : Infinity;
  const cameraFresh = () => known && status?.running && status.camera?.running
    && age(status.camera.age_ms / 1000) <= 0.5;
  const eegFresh = () => known && status?.running && status.eeg?.state === 'running'
    && status.eeg?.mode === 'signal' && age(status.eeg.stats?.age_seconds) <= 0.75;
  const fresh = () => cameraFresh() && eegFresh();
  const number = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : 'unknown';
  const microvolts = value => Number.isFinite(value) ? value * 1e6 : NaN;
  const count = value => Number.isFinite(value) && value >= 0 ? String(Math.floor(value)) : 'unknown';
  const seconds = value => Number.isFinite(value) ? `${Math.max(0, value).toFixed(1)} s` : 'unknown';

  function clearCamera() {
    ++frameId;
    el.camera.getContext('2d').clearRect(0, 0, el.camera.width, el.camera.height);
  }
  function clearEeg() { el.eeg.getContext('2d').clearRect(0, 0, el.eeg.width, el.eeg.height); }
  function clearFrames() { clearCamera(); clearEeg(); cancel(staleTimer); staleTimer = null; }

  function render() {
    const online = !!isConnected(), ready = online && known;
    const running = ready && status.running;
    const busy = ready && (!!pending || ['starting','stopping','contact'].includes(status?.state));
    const stopping = pending === 'stop';
    const armed = ready && status.armed && fresh() && !stopping && pending !== 'calibrate';
    put('summary', !online ? 'Backend offline' : !ready ? 'Status unconfirmed'
      : stopping ? 'Stopping / disarming' : `${running ? 'Running' : text(status.state, 'Stopped')} / ${armed ? 'armed' : 'disarmed'}`);
    const labels = { start: 'Starting preview...', stop: 'Stopping preview and disarming...',
      contact: 'Checking contacts for 5 seconds...', arm: 'Updating swipe arming...',
      calibrate: 'Preparing a 3-second trial...', reset_calibration: 'Clearing calibration...' };
    put('status', !online ? 'Backend offline. Preview and arming are unconfirmed; frames hidden.'
      : !ready ? 'Waiting for current preview status. Stop & disarm remains available.'
      : labels[pending] || text(status.reason, text(status.state, 'Stopped')));
    el.status.setAttribute('aria-busy', String(online && busy));
    put('error', error || (ready ? text(status.error) : ''));
    el.error.hidden = !el.error.textContent;
    el.start.disabled = !ready || busy || running || status?.cleanup_pending || isCameraRunning() || !isBrainbitConnected();
    // Stop remains reachable through in-flight operations and unconfirmed reads.
    el.stop.disabled = !online;
    el.contact.disabled = !ready || busy || running || status?.cleanup_pending || !isBrainbitConnected();
    el.arm.checked = !!armed;
    el.arm.disabled = !ready || busy || !running || (!armed && (!status.can_arm || !fresh()));
    const trial = ready ? status.calibration?.pending : null;
    el.left.disabled = el.right.disabled = !ready || busy || !running || !fresh() || !status.can_arm || !!trial;
    el.reset.disabled = !ready || busy || !!trial;
    put('start-hint', isCameraRunning()
      ? 'Stop the normal hand camera before starting this preview.'
      : !isBrainbitConnected() ? 'Connect BrainBit in the panel above before starting.'
      : 'Start preview is explicit. Desktop swipes remain disarmed until you arm them.');
    if (!ready || stopping || !visible()) clearFrames();
    else {
      if (!cameraFresh()) clearCamera();
      if (!eegFresh()) clearEeg();
    }
    put('camera-status', !running ? 'Camera preview is off.' : !cameraFresh()
      ? 'Camera frame is stale or unavailable.'
      : `${status.camera.tracked ? 'Hand tracked' : 'No hand tracked'} · ${number(status.camera.fps, 0)} FPS · frame age ${seconds(age(status.camera.age_ms / 1000))}`);
    const stats = ready ? status.eeg?.stats || {} : {};
    put('eeg-status', !running ? 'EEG preview is off.' : !eegFresh() ? 'Waiting for fresh EEG samples.'
      : `${number(stats.received_rate_hz, 0)} Hz received / ${number(status.eeg.nominal_hz, 0)} Hz nominal · age ${seconds(age(stats.age_seconds))} · µV, auto scale per channel`);
    const motion = ready ? status.motion || {} : {};
    put('motion', ready ? `Webcam direction: ${['left','right','neutral'].includes(motion.direction) ? motion.direction : 'neutral'} · ${motion.ready ? 'ready' : 'waiting for neutral hand'}${motion.last_event?.direction ? ` · last event: ${text(motion.last_event.direction)}` : ''}` : '');
    const channelStats = Array.isArray(stats.channels) ? stats.channels.slice(0, 8).map(channel =>
      `${text(channel.name, 'Channel')}: RMS ${number(microvolts(channel.rms_v))} µV, peak-to-peak ${number(microvolts(channel.peak_to_peak_v))} µV`).join(' | ') : '';
    put('diagnostics', ready ? `Counter discontinuities ${count(stats.gaps)} · duplicates ${count(stats.duplicates)} · nonfinite ${count(stats.nonfinite)} · channel mismatches ${count(stats.channel_mismatches)} · queue drops ${count(stats.queue_drops)}${channelStats ? `\n${channelStats}` : ''}` : '');
    const contact = ready ? status.eeg?.contact_precheck : null;
    const contactAge = contact ? age(contact.age_seconds) : Infinity;
    const values = Array.isArray(contact?.values) ? contact.values.slice(0, 8).map((value, index) => {
      const channel = contact.channels?.[index];
      return `${text(typeof channel === 'string' ? channel : channel?.name, `Channel ${index + 1}`)} ${Number.isFinite(value) ? `${number(value / 1000)} kΩ` : 'unavailable'}`;
    }).join(' · ') : '';
    put('contact-status', !contact ? 'No contact precheck yet. Stop preview to run a 5-second check.'
      : `Contact precheck ${contactAge > 30 ? '(stale; historical)' : '(historical)'} · age ${seconds(contactAge)}: ${values || 'No readings'}${text(contact.unit_note) ? ` · ${text(contact.unit_note)}` : ''}. Values are descriptive; no good/bad cutoff.`);
    const calibration = ready ? status.calibration || {} : {};
    put('trial', trial ? `Move ${text(trial.label)} now with your hand. ${number(trial.remaining_seconds, 1)} s remaining. Desktop swipes are disarmed.`
      : text(calibration.last_result?.reason || calibration.last_result?.text || calibration.last_result, 'Choose left or right to begin a trial.'));
    const evaluation = calibration.evaluation || {};
    let evaluationText = text(evaluation.reason, 'More valid trials are needed for held-out evaluation.');
    if (evaluation.state === 'evaluated' && Number.isFinite(evaluation.accuracy)
      && Number.isFinite(evaluation.baseline_accuracy) && evaluation.n_train >= 12 && evaluation.n_test >= 6) {
      evaluationText = `Training ${count(evaluation.n_train)} · held-out ${count(evaluation.n_test)} · accuracy ${number(evaluation.accuracy * 100, 0)}% · majority baseline ${number(evaluation.baseline_accuracy * 100, 0)}%${Number.isFinite(evaluation.balanced_accuracy) ? ` · balanced accuracy ${number(evaluation.balanced_accuracy * 100, 0)}%` : ''}. Experimental evidence only; EEG does not control swipes.`;
    }
    put('calibration-status', ready ? `Valid trials: left ${count(calibration.counts?.left ?? 0)} · right ${count(calibration.counts?.right ?? 0)}. ${evaluationText}` : 'Calibration status unavailable.');
    onResize();
  }

  function drawCamera(camera) {
    clearCamera();
    if (!cameraFresh() || !visible()) return;
    // Do not truncate image data with the display-text helper.
    if (typeof camera.image !== 'string' || !/^data:image\/jpeg;base64,[A-Za-z0-9+/=]+$/.test(camera.image)) {
      put('camera-status', 'Camera frame could not be displayed.'); return;
    }
    const id = frameId, currentEpoch = epoch, image = imageFactory();
    image.onload = () => {
      if (id !== frameId || currentEpoch !== epoch || !cameraFresh() || !visible() || !isConnected()) return;
      const context = el.camera.getContext('2d');
      context.drawImage(image, 0, 0, 480, 360);
      const points = camera.landmarks;
      if (Array.isArray(points) && points.length === 21 && points.every(p => Array.isArray(p)
        && Number.isFinite(p[0]) && Number.isFinite(p[1]))) {
        context.strokeStyle = '#5eead4'; context.lineWidth = 2; context.beginPath();
        for (const [a, b] of HAND_EDGES) {
          context.moveTo(points[a][0] * 480, points[a][1] * 360);
          context.lineTo(points[b][0] * 480, points[b][1] * 360);
        }
        context.stroke();
      }
    };
    image.onerror = () => { if (id === frameId) { clearCamera(); put('camera-status', 'Camera frame could not be displayed.'); } };
    image.src = camera.image;
  }

  function drawEeg(eeg) {
    clearEeg();
    if (!eegFresh() || !visible()) return;
    const channels = Array.isArray(eeg.channels) ? eeg.channels.slice(0, 8) : [];
    const rows = Array.isArray(eeg.waveform) ? eeg.waveform.slice(-250) : [];
    if (!channels.length || !rows.length) { put('eeg-status', 'No EEG samples in this preview yet.'); return; }
    if (eeg.units !== 'V') { put('eeg-status', 'EEG units are unconfirmed; waveform hidden.'); return; }
    const context = el.eeg.getContext('2d'), width = 480, height = 360;
    const lane = height / channels.length;
    context.font = '11px sans-serif'; context.lineWidth = 1;
    channels.forEach((channel, index) => {
      const values = rows.map(row => Number.isFinite(row.samples?.[index]) ? row.samples[index] * 1e6 : null);
      const finite = values.filter(Number.isFinite);
      if (!finite.length) {
        context.fillStyle = '#94a3b8';
        context.fillText(`${text(channel.name, `Ch ${index + 1}`)} · no finite samples`, 8, lane * index + 13);
        return;
      }
      const center = finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : 0;
      const span = Math.max(1, ...finite.map(value => Math.abs(value - center)));
      const middle = lane * (index + 0.5);
      context.fillStyle = '#cbd5e1';
      context.fillText(`${text(channel.name, `Ch ${index + 1}`)} · ±${number(span)} µV around ${number(center)} µV`, 8, lane * index + 13);
      context.strokeStyle = '#263445'; context.beginPath();
      context.moveTo(8, middle); context.lineTo(width - 8, middle); context.stroke();
      context.strokeStyle = ['#5eead4','#93c5fd','#fbbf24','#c4b5fd'][index % 4]; context.beginPath();
      let pen = false;
      values.forEach((value, rowIndex) => {
        if (value === null) { pen = false; return; }
        const x = 8 + rowIndex * (width - 16) / Math.max(1, rows.length - 1);
        const y = middle - (value - center) / span * Math.max(4, lane * 0.3);
        if (pen) context.lineTo(x, y); else context.moveTo(x, y);
        pen = true;
      });
      context.stroke();
    });
  }

  function accept(data, raw = false) {
    if (!isConnected() || !valid(data)) return false;
    if (Number.isInteger(data.revision) && data.revision < revision) return false;
    if (Number.isInteger(data.revision)) revision = data.revision;
    ++generation;
    // Retain metadata only; do not collect a recording in the controller.
    const { image, landmarks, ...camera } = data.camera || {};
    const { waveform, samples, ...eeg } = data.eeg || {};
    status = { ...data, camera, eeg };
    receivedAt = now(); known = true; error = '';
    render();
    if (raw && data.running && pending !== 'stop' && visible()) {
      drawCamera(data.camera || {});
      drawEeg(data.eeg || {});
    }
    cancel(staleTimer);
    if (data.running && visible()) staleTimer = schedule(() => {
      clearFrames();
      render();
    }, Math.max(1, Math.min(500 - (data.camera?.age_ms || 0), 750 - ((data.eeg?.stats?.age_seconds || 0) * 1000)) + 1));
    return true;
  }

  function failure(message) {
    ++generation; known = false; pending = null; error = message;
    clearFrames(); render();
  }
  function resetConnection() {
    ++epoch; ++generation; ++actionId; ++readId; revision = -1;
    status = null; known = false; pending = null; error = ''; reading = false;
    polling = false; cancel(pollTimer); pollTimer = null; clearFrames();
  }
  function syncPolling() {
    const active = !!(isConnected() && visible());
    if (!active) {
      if (polling) { ++readId; reading = false; }
      polling = false; cancel(pollTimer); pollTimer = null; clearFrames(); return;
    }
    if (!polling) { polling = true; poll(); }
  }
  async function poll() {
    if (!polling || reading) return;
    reading = true; pollTimer = null;
    const id = ++readId, initialGeneration = generation, currentEpoch = epoch;
    const action = known && status.running ? 'preview' : 'status';
    try {
      const response = await request(action);
      if (id !== readId || currentEpoch !== epoch || initialGeneration !== generation || !isConnected()) return;
      if (!response.ok || !valid(response.data)) { failure('Preview status could not be confirmed. Frames hidden; use Stop & disarm if needed.'); return; }
      accept(response.data, action === 'preview');
    } catch (_) {
      if (id === readId && currentEpoch === epoch && initialGeneration === generation && isConnected())
        failure('Preview connection failed. Frames hidden; use Stop & disarm if needed.');
    } finally {
      if (id === readId) {
        reading = false;
        if (polling) pollTimer = schedule(poll, known && status.running ? 125 : 1000);
      }
    }
  }
  async function run(action, payload = {}) {
    const button = { start: el.start, stop: el.stop, contact: el.contact, arm: el.arm,
      calibrate: payload.label === 'left' ? el.left : el.right, reset_calibration: el.reset }[action];
    if (!button || button.disabled || !isConnected() || (pending && action !== 'stop') || pending === 'stop') { render(); return; }
    const id = ++actionId, currentEpoch = epoch;
    ++generation; pending = action; error = '';
    if (action === 'stop' || action === 'calibrate') clearFrames();
    render();
    try {
      const response = await request(action, payload);
      if (id !== actionId || currentEpoch !== epoch || !isConnected()) return;
      pending = null;
      if (!response.ok || !valid(response.data)) { failure(text(response.data?.error, 'Preview change could not be confirmed. Use Stop & disarm to recover.')); return; }
      if (!accept(response.data)) render();
    } catch (_) {
      if (id === actionId && currentEpoch === epoch && isConnected())
        failure('Preview change could not be confirmed. Use Stop & disarm to recover.');
    } finally {
      if (id === actionId && polling && !reading) { cancel(pollTimer); poll(); }
    }
  }

  el.panel.addEventListener('toggle', () => { syncPolling(); render(); });
  el.panel.addEventListener('keydown', event => event.stopPropagation());
  document.addEventListener('visibilitychange', () => { syncPolling(); render(); });
  el.start.addEventListener('click', () => run('start'));
  el.stop.addEventListener('click', () => run('stop'));
  el.contact.addEventListener('click', () => run('contact'));
  el.arm.addEventListener('change', () => { const enabled = el.arm.checked; render(); run('arm', { enabled }); });
  el.left.addEventListener('click', () => run('calibrate', { label: 'left' }));
  el.right.addEventListener('click', () => run('calibrate', { label: 'right' }));
  el.reset.addEventListener('click', () => run('reset_calibration'));
  render();
  return {
    connectionChanged() {
      const online = !!isConnected();
      if (online !== lastOnline) { resetConnection(); lastOnline = online; }
      render(); syncPolling();
    },
    acceptStatus: data => accept(data),
    refreshControls: render,
  };
}

module.exports = { createMultimodalPanel };
