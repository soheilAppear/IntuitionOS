// Experimental display/controller. Sensor frames remain transient in this renderer.
const { requestMultimodal } = require('./multimodal-http.cjs');
const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
  [13,17],[0,17],[17,18],[18,19],[19,20]];

function createMultimodalPanel({ document, isConnected, isCameraRunning = () => false,
  isBrainbitConnected = () => false, onResize = () => {}, request = requestMultimodal,
  startEegGuard = async () => ({ ok: false, error: 'Global Escape stop is unavailable.' }),
  stopEegGuard = async () => ({ ok: true }),
  imageFactory = () => new Image(), schedule = setTimeout, cancel = clearTimeout, now = Date.now }) {
  const ids = ['panel','summary','status','error','start','stop','contact','arm','start-hint',
    'camera','eeg','camera-status','eeg-status','motion','diagnostics','contact-status',
    'left','right','reset','trial','calibration-status','control-mode','eeg-arm','control-status','prediction',
    'eeg-phase','eeg-left','eeg-right','eeg-rest','eeg-train','eeg-reset','eeg-trial','eeg-model-status',
    'eeg-errors','eeg-evaluation','eeg-gates','train-left','train-right','train-rest',
    'validate-left','validate-right','validate-rest'];
  const el = Object.fromEntries(ids.map(id => [id, document.getElementById(`multimodal-${id}`)]));
  let status = null, known = false, error = '', pending = null, epoch = 0, actionId = 0;
  let generation = 0, revision = -1, readId = 0, reading = false, polling = false;
  let pollTimer = null, staleTimer = null, frameId = 0, receivedAt = 0;
  let lastOnline = false;
  let stopRequested = false, wasActive = false, eegActionError = '';
  let guardToken = null;
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
  const controlMode = () => status?.control_mode === 'eeg' ? 'eeg' : 'webcam';
  const armingSuppressed = () => stopRequested || ['stop', 'arm', 'eeg_arm', 'calibrate', 'eeg_trial', 'eeg_train', 'eeg_reset', 'control_mode'].includes(pending);
  const eegEligible = decoder => !!decoder && decoder.trained === true && decoder.passed_validation === true
    && decoder.arm_eligible === true && decoder.evaluation?.passed === true
    && ['left', 'right', 'rest'].every(label => decoder.train_counts?.[label] >= 8 && decoder.validation_counts?.[label] >= 8);
  function releaseGuard(token = guardToken, unconditional = false) {
    if (token === guardToken || unconditional) guardToken = null;
    // Main owns the lease/shortcut. Scoped cleanup cannot cancel a newer lease.
    try { Promise.resolve(stopEegGuard(unconditional ? undefined : token)).catch(() => {}); } catch (_) {}
  }

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
    const stopping = pending === 'stop' || stopRequested;
    const armed = ready && controlMode() === 'webcam' && status.armed && fresh() && !armingSuppressed();
    const eegArmed = ready && controlMode() === 'eeg' && status.eeg_control?.armed && eegFresh()
      && eegEligible(status.eeg_control?.decoder) && !armingSuppressed();
    put('summary', !online ? 'Backend offline' : !ready ? 'Status unconfirmed'
      : stopping ? 'Stopping / disarming' : running && !visible() ? 'Running / open to refresh status'
      : `${running ? 'Running' : text(status.state, 'Stopped')} / ${eegArmed ? 'EEG armed' : armed ? 'webcam armed' : 'prediction only'}`);
    const labels = { start: 'Starting preview...', stop: 'Stopping preview and disarming...',
      contact: 'Checking contacts for 5 seconds...', arm: 'Updating swipe arming...',
      calibrate: 'Preparing a 3-second trial...', reset_calibration: 'Clearing calibration...',
      eeg_trial: 'Preparing a guided EEG trial...', eeg_train: 'Training and freezing the EEG model...',
      eeg_reset: 'Resetting EEG model and trials...', eeg_arm: 'Updating EEG swipe arming...',
      control_mode: 'Changing control source and disarming both modes...' };
    put('status', !online ? 'Backend offline. Preview and arming are unconfirmed; frames hidden.'
      : !ready ? 'Waiting for current preview status. Stop & disarm remains available.'
      : labels[pending] || text(status.reason, text(status.state, 'Stopped')));
    el.status.setAttribute('aria-busy', String(online && busy));
    put('error', error || (ready ? text(status.error) : ''));
    el.error.hidden = !el.error.textContent;
    el.start.disabled = !ready || busy || running || stopping || status?.cleanup_pending || isCameraRunning() || !isBrainbitConnected();
    // Stop remains reachable through in-flight operations and unconfirmed reads.
    el.stop.disabled = !online;
    el.contact.disabled = !ready || busy || running || stopping || status?.cleanup_pending || !isBrainbitConnected();
    el.arm.checked = !!armed;
    el.arm.disabled = !ready || busy || !running || stopping || controlMode() !== 'webcam'
      || (!armed && (!status.can_arm || !fresh()));
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
    put('camera-status', !running ? 'Camera preview is off.' : controlMode() === 'eeg' && !status.camera?.running
      ? 'Camera is off. EEG inference uses EEG only.' : !cameraFresh()
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
    renderEegControl({ ready, running, busy, armed, eegArmed, stopping });
    onResize();
  }

  function renderEegControl({ ready, running, busy, armed, eegArmed, stopping }) {
    const control = ready ? status.eeg_control || {} : {};
    const decoder = control.decoder || {};
    const trial = control.pending;
    const labels = ['left', 'right', 'rest'];
    const quota = Number.isInteger(decoder.min_per_class) && decoder.min_per_class >= 8 ? decoder.min_per_class : 8;
    const trainCounts = decoder.train_counts || {}, validationCounts = decoder.validation_counts || {};
    const trainingComplete = labels.every(label => Number.isInteger(trainCounts[label]) && trainCounts[label] >= quota);
    const validationComplete = labels.every(label => Number.isInteger(validationCounts[label]) && validationCounts[label] >= quota);
    const eligible = eegEligible(decoder) && trainingComplete && validationComplete;
    const mode = controlMode();
    el['control-mode'].value = mode;
    el['control-mode'].disabled = !ready || busy || !!trial || stopping || status?.cleanup_pending;
    el['eeg-arm'].checked = !!eegArmed;
    el['eeg-arm'].disabled = !ready || busy || !running || stopping || mode !== 'eeg' || !!trial
      || (!eegArmed && (!control.can_arm || !eligible || !eegFresh()));
    const hasDecoder = ready && !!status.eeg_control?.decoder;
    // Frozen training data is never extended by a validation click.
    el['eeg-phase'].value = decoder.trained ? 'validate' : 'train';
    el['eeg-phase'].disabled = true;
    for (const label of labels) {
      put(`train-${label}`, `${count(trainCounts[label] ?? 0)} / ${quota}`);
      put(`validate-${label}`, `${count(validationCounts[label] ?? 0)} / ${quota}`);
      el[`eeg-${label}`].disabled = !hasDecoder || busy || !running || stopping || !!trial
        || mode !== 'webcam' || !fresh() || status.camera?.tracked !== true
        || (decoder.trained ? validationCounts[label] : trainCounts[label]) >= quota;
    }
    el['eeg-train'].disabled = !hasDecoder || busy || stopping || !!trial || !running
      || !eegFresh() || decoder.trained === true || !trainingComplete || decoder.can_train === false;
    el['eeg-reset'].disabled = !hasDecoder || busy || stopping || !!trial;
    put('control-status', !ready ? 'Control state unconfirmed. Arming is disabled.'
      : stopping ? 'Stop requested. Both arming controls are cleared while the backend confirms.'
      : eegArmed ? 'EEG swipes armed: EEG inference chooses direction. The webcam does not choose EEG actions.'
      : armed ? 'Webcam swipes armed: hand motion chooses direction. EEG predictions do not choose actions.'
      : `Prediction only. Both desktop controls are disarmed. Selected source: ${mode === 'eeg' ? 'EEG classifier' : 'webcam direction'}.`);
    const prediction = control.prediction;
    const predicted = prediction && prediction.valid === true && eegFresh()
      && Number.isFinite(prediction.confidence);
    put('prediction', predicted
      ? `EEG prediction: ${labels.includes(prediction.label) ? prediction.label : 'uncertain'} · uncalibrated score ${number(prediction.confidence, 3)}${Number.isFinite(prediction.margin) ? ` · margin ${number(prediction.margin, 3)}` : ''}${eegArmed ? ' · EEG control armed' : ' · display only'}${prediction.ood ? ' · outside training range' : ''}`
      : `EEG prediction unavailable. ${!eegFresh() ? 'Waiting for fresh EEG.' : text(prediction?.reason, decoder.trained ? 'Waiting for a complete EEG window.' : 'Collect training trials and train a model first.')}`);
    const trialCue = trial?.label === 'rest' ? 'keep your hand still for rest'
      : trial?.remaining_seconds > 2.5 ? `hold still for the first half-second; prepare to move ${text(trial?.label)}`
      : trial?.remaining_seconds <= 0.5 ? 'hold still for the final half-second'
      : `move your hand ${text(trial?.label)} once now`;
    put('eeg-trial', trial
      ? `${trial.phase === 'validate' ? 'Validation' : 'Training'} trial: ${trialCue}. ${number(trial.remaining_seconds, 1)} s remaining. Both modes are disarmed.`
      : text(control.last_result, decoder.trained
        ? 'Model frozen. Collect new validation trials; these do not retrain the model.'
        : 'Choose left, right, or rest to collect a training trial.'));
    put('eeg-model-status', !hasDecoder ? 'EEG classifier status unavailable.'
      : decoder.trained ? `Frozen model · ${text(decoder.state, 'collecting_validation')}. Training data is separate from new validation trials.`
      : trainingComplete ? 'Training quota reached. Select Train & freeze model before collecting validation.'
      : 'Collect at least 8 accepted training trials per class. Rejected trials do not increase counts.');
    const trainingError = text(decoder.training_error || decoder.train_error);
    const validationError = text(decoder.validation_error);
    put('eeg-errors', [eegActionError, trainingError && `Training: ${trainingError}`, validationError && `Validation: ${validationError}`].filter(Boolean).join('\n'));
    el['eeg-errors'].hidden = !el['eeg-errors'].textContent;
    const evaluation = decoder.evaluation || {};
    renderEegEvaluation(evaluation, validationComplete);
    const gateReason = text(control.reason, text(evaluation.reason, 'Collect and validate the frozen EEG model before arming.'));
    const thresholds = evaluation.thresholds || {};
    const gateNames = { enough_trials: '8 validation trials per class', balanced_accuracy: `Balanced accuracy ≥ ${number((thresholds.balanced_accuracy ?? 0.75) * 100, 0)}%`,
      each_recall: `Each recall ≥ ${number((thresholds.each_recall ?? 0.70) * 100, 0)}%`, each_precision: `Each precision ≥ ${number((thresholds.each_precision ?? 0.70) * 100, 0)}%`,
      rest_false_activation_rate: `Rest false activations ≤ ${number((thresholds.rest_false_activation_rate ?? 0.10) * 100, 0)}%` };
    const gateDetails = Object.entries(gateNames).map(([key, name]) => `${name}: ${evaluation.gates?.[key] === true ? 'passed' : evaluation.gates?.[key] === false ? 'not passed' : 'pending'}`).join(' · ');
    put('eeg-gates', `${eligible ? 'Validation gates passed' : 'EEG arming locked'}: ${gateReason}\n${gateDetails}\nEngineering gates are not a safety guarantee. EEG may classify motion or muscle artifacts, not intention.${control.neutral_ready === false ? ' Rest must be recognized before a new EEG swipe.' : ''}`);
  }

  function renderEegEvaluation(evaluation, complete) {
    const labels = ['left', 'right', 'rest'];
    const percent = value => Number.isFinite(value) ? `${number(value * 100, 1)}%` : 'not available';
    if (!complete) {
      put('eeg-evaluation', 'Validation is incomplete. Collect at least 8 new trials per class against the frozen model. Arming remains locked.');
      return;
    }
    const lines = [`Frozen-model validation: accuracy ${percent(evaluation.accuracy)} · balanced accuracy ${percent(evaluation.balanced_accuracy)}.`];
    for (const label of labels) {
      const metrics = evaluation.per_class?.[label] || {};
      lines.push(`${label}: precision ${percent(metrics.precision)} · recall ${percent(metrics.recall)}.`);
    }
    const matrix = evaluation.confusion_matrix || evaluation.confusion;
    if (matrix) {
      lines.push('Confusion rows = actual; columns = predicted left / right / rest / uncertain:');
      labels.forEach((label, index) => {
        const row = Array.isArray(matrix) ? matrix[index] : matrix[label];
        if (row) lines.push(`${label}: ${[...labels, 'uncertain'].map((column, columnIndex) => count(Array.isArray(row) ? row[columnIndex] : row[column])).join(' / ')}`);
      });
    }
    const restFalse = evaluation.rest_false_activations;
    lines.push(`Rest false activations: ${count(restFalse)} / ${count(evaluation.rest_trials)} held-out rest trials before debounce${Number.isFinite(evaluation.rest_false_activation_rate) ? ` (${percent(evaluation.rest_false_activation_rate)})` : ''}.`);
    lines.push(`Uncertain predictions: ${count(evaluation.abstentions)}.`);
    if (Number.isFinite(evaluation.baseline_accuracy)) lines.push(`Majority-class baseline: ${percent(evaluation.baseline_accuracy)}.`);
    put('eeg-evaluation', lines.join('\n'));
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
    if (guardToken && !data.eeg_control?.armed && pending !== 'eeg_arm') releaseGuard();
    wasActive = wasActive || data.running || data.armed || data.eeg_control?.armed === true;
    if (!data.running && !data.armed && !data.eeg_control?.armed && !data.cleanup_pending) {
      stopRequested = false;
      wasActive = false;
    }
    receivedAt = now(); known = true; error = '';
    render();
    if (raw && data.running && pending !== 'stop' && visible()) {
      drawCamera(data.camera || {});
      drawEeg(data.eeg || {});
    }
    cancel(staleTimer);
    const staleIn = controlMode() === 'eeg'
      ? 750 - ((data.eeg?.stats?.age_seconds || 0) * 1000)
      : Math.min(500 - (data.camera?.age_ms || 0), 750 - ((data.eeg?.stats?.age_seconds || 0) * 1000));
    if (data.running && visible()) staleTimer = schedule(() => {
      if (controlMode() === 'eeg' && !eegFresh() && guardToken) releaseGuard();
      clearFrames();
      render();
    }, Math.max(1, staleIn + 1));
    return true;
  }

  function failure(message) {
    // A failed cached read during guard preparation must revoke that future
    // commitment before its asynchronously prepared token can be used.
    if (pending === 'eeg_arm' || pending === 'arm') {
      ++actionId;
      if (pending === 'eeg_arm') releaseGuard(undefined, true);
    }
    if (guardToken) releaseGuard();
    ++generation; known = false; pending = null; error = message;
    clearFrames(); render();
  }
  function resetConnection() {
    if (guardToken || pending === 'eeg_arm') releaseGuard(undefined, true);
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
      calibrate: payload.label === 'left' ? el.left : el.right, reset_calibration: el.reset,
      eeg_trial: el[`eeg-${payload.label}`], eeg_train: el['eeg-train'], eeg_reset: el['eeg-reset'],
      eeg_arm: el['eeg-arm'], control_mode: el['control-mode'] }[action];
    if (!button || button.disabled || !isConnected() || (pending && action !== 'stop') || pending === 'stop') { render(); return; }
    const id = ++actionId, currentEpoch = epoch;
    ++generation; pending = action; error = '';
    const preparingGuard = action === 'eeg_arm' && payload.enabled === true;
    let preparedToken = null;
    if (['stop','control_mode','eeg_reset','eeg_train','eeg_trial','calibrate','reset_calibration'].includes(action)
      || (action === 'eeg_arm' && payload.enabled === false)) releaseGuard(undefined, true);
    if (action === 'stop') { stopRequested = true; wasActive = true; }
    if (action.startsWith('eeg_')) eegActionError = '';
    if (action === 'stop' || action === 'calibrate' || action === 'eeg_trial') clearFrames();
    render();
    try {
      if (preparingGuard) {
        const prepared = await startEegGuard();
        preparedToken = prepared?.token;
        if (id !== actionId || currentEpoch !== epoch || !isConnected()) {
          if (preparedToken) releaseGuard(preparedToken);
          return;
        }
        if (prepared?.ok !== true || typeof preparedToken !== 'string') {
          eegActionError = text(prepared?.error, 'Global Escape could not be registered. EEG controls remain disarmed.');
          failure(eegActionError); return;
        }
        // Registering Escape is asynchronous. Freshness, mode, and model
        // eligibility must still hold immediately before the arm request.
        if (!known || !status.running || controlMode() !== 'eeg' || !eegFresh()
          || !status.eeg_control?.can_arm || !eegEligible(status.eeg_control?.decoder) || stopRequested) {
          releaseGuard(preparedToken);
          pending = null;
          eegActionError = 'EEG readiness changed before arming. Review the latest status and arm again explicitly.';
          render();
          return;
        }
        guardToken = preparedToken;
        payload = { enabled: true, guard_token: preparedToken };
      }
      const response = await request(action, payload);
      if (id !== actionId || currentEpoch !== epoch || !isConnected()) {
        if (preparedToken) releaseGuard(preparedToken);
        return;
      }
      pending = null;
      if (Number.isInteger(response.data?.revision) && response.data.revision < revision) {
        if (preparedToken && (!known || !status?.eeg_control?.armed || guardToken !== preparedToken))
          releaseGuard(preparedToken);
        render();
        return;
      }
      if (!response.ok || !valid(response.data)) {
        if (preparedToken) releaseGuard(preparedToken);
        const message = text(response.data?.error, 'Preview change could not be confirmed. Use Stop & disarm to recover.');
        if (action.startsWith('eeg_')) eegActionError = message;
        failure(message); return;
      }
      if (!accept(response.data)) render();
      if (preparedToken && !status?.eeg_control?.armed) releaseGuard(preparedToken);
    } catch (_) {
      if (preparedToken) releaseGuard(preparedToken);
      if (id === actionId && currentEpoch === epoch && isConnected()) {
        if (action.startsWith('eeg_')) eegActionError = 'EEG operation could not be confirmed. Check status before trying again.';
        failure('Preview change could not be confirmed. Use Stop & disarm to recover.');
      }
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
  el['eeg-arm'].addEventListener('change', () => { const enabled = el['eeg-arm'].checked; render(); run('eeg_arm', { enabled }); });
  el['control-mode'].addEventListener('change', () => { const mode = el['control-mode'].value; render(); run('control_mode', { mode }); });
  for (const label of ['left', 'right', 'rest']) el[`eeg-${label}`].addEventListener('click', () =>
    run('eeg_trial', { label, phase: status?.eeg_control?.decoder?.trained ? 'validate' : 'train' }));
  el['eeg-train'].addEventListener('click', () => run('eeg_train'));
  el['eeg-reset'].addEventListener('click', () => run('eeg_reset'));
  el.left.addEventListener('click', () => run('calibrate', { label: 'left' }));
  el.right.addEventListener('click', () => run('calibrate', { label: 'right' }));
  el.reset.addEventListener('click', () => run('reset_calibration'));
  // Document capture sees Escape while this HUD has focus, including in its
  // input field. It does not register a global shortcut or intercept terminals.
  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape' || (typeof document.hasFocus === 'function' && !document.hasFocus())
      || (!wasActive && !pending && !el.panel.open)) return;
    event.preventDefault();
    event.stopImmediatePropagation?.();
    event.stopPropagation();
    stopRequested = true;
    clearFrames();
    if (isConnected()) run('stop');
    else {
      releaseGuard(undefined, true);
      ++actionId; ++generation;
      known = false; pending = null;
      error = 'Stop requested while offline. Device state is unconfirmed; no operation will be replayed after reconnection.';
      render();
    }
  }, true);
  render();
  return {
    connectionChanged() {
      const online = !!isConnected();
      if (online !== lastOnline) { resetConnection(); lastOnline = online; }
      render(); syncPolling();
    },
    acceptStatus: data => accept(data),
    refreshControls: render,
    emergencyStop(reason = 'escape') {
      ++actionId; ++generation; ++readId;
      reading = false; pending = null; known = false; stopRequested = true; wasActive = true;
      releaseGuard(undefined, true);
      clearFrames();
      error = `Emergency stop requested (${text(reason, 'escape')}). Waiting for confirmed stopped status.`;
      render();
      if (polling && isConnected()) { cancel(pollTimer); poll(); }
    },
  };
}

module.exports = { createMultimodalPanel };
