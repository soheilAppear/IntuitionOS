// Fast double-blink controls. Sensor frames are transient and never archived here.
const { requestMultimodal } = require('./multimodal-http.cjs');
const { drawHandOverlay } = require('./hand-overlay.cjs');
const LABELS = ['rest', 'select'];
const QUOTA = 8;
const TRAINING_LIMIT = 48;

function createMultimodalPanel({ document, isConnected, isCameraRunning = () => false,
  isBrainbitConnected = () => false, onResize = () => {}, onCameraPrivacy = () => {}, request = requestMultimodal,
  startEegGuard = async () => ({ ok: false, error: 'Global Escape stop is unavailable.' }),
  stopEegGuard = async () => ({ ok: true }),
  imageFactory = () => new Image(), schedule = setTimeout, cancel = clearTimeout, now = Date.now }) {
  const ids = ['panel','summary','status','error','start','stop','start-hint','camera','eeg',
    'camera-status','eeg-status','diagnostics','eeg-arm','eeg-arm-label','control-status','prediction',
    'camera-container','hand-pointer-status','selection-counts','eeg-camera',
    'train-rest','train-select','validate-rest','validate-select','eeg-rest','eeg-select',
    'eeg-train','eeg-reset','eeg-load-saved','eeg-save','eeg-saved-status','eeg-trial','eeg-model-status',
    'eeg-errors','eeg-evaluation','eeg-gates','eeg-batch','eeg-batch-cancel','eeg-batch-status','eeg-guided-cue'];
  const el = Object.fromEntries(ids.map(id => [id, document.getElementById(`multimodal-${id}`)]));
  let status = null, known = false, error = '', pending = null, epoch = 0, actionId = 0;
  let generation = 0, revision = -1, readId = 0, reading = false, polling = false;
  let pollTimer = null, staleTimer = null, frameId = 0, receivedAt = 0, lastOnline = false;
  let stopRequested = false, wasActive = false, eegActionError = '', guardToken = null;
  // The queue schedules labels only; the server owns countdown, samples, and trial acceptance.
  let batch = null, batchTimer = null, batchNotice = '';
  const text = (value, fallback = '') => typeof value === 'string' && value.trim() ? value.slice(0, 600) : fallback;
  const put = (id, value) => { if (el[id].textContent !== value) el[id].textContent = value; };
  const visible = () => el.panel.open && document.hidden !== true;
  const valid = data => data && typeof data.state === 'string' && typeof data.running === 'boolean'
    && typeof data.eeg_control?.armed === 'boolean';
  // Fixed metadata is a protocol check, never a selectable legacy mode.
  const workflowKnown = () => status?.control_mode === 'eeg_click'
    && status.eeg_control?.method === 'cnn_blink_fast' && status.eeg_control?.target === 'mouse';
  const age = value => Number.isFinite(value) && value >= 0 ? value + Math.max(0, now() - receivedAt) / 1000 : Infinity;
  const cameraFresh = () => known && status?.running && status.camera?.running
    && Number.isFinite(status.camera.age_ms) && age(status.camera.age_ms / 1000) <= 0.5;
  const eegFresh = () => known && status?.running && status.eeg?.state === 'running'
    && status.eeg?.mode === 'signal' && age(status.eeg.stats?.age_seconds) <= 0.75;
  const cameraMouseEnabled = () => status?.eeg_control?.camera_mouse_enabled === true;
  const cameraDisplayEnabled = () => cameraMouseEnabled() && pending !== 'eeg_camera';
  const number = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : 'unknown';
  const microvolts = value => Number.isFinite(value) ? value * 1e6 : NaN;
  const count = value => Number.isInteger(value) && value >= 0 ? String(value) : 'unknown';
  const seconds = value => Number.isFinite(value) ? `${Math.max(0, value).toFixed(1)} s` : 'unknown';
  const complete = counts => LABELS.every(label => Number.isInteger(counts?.[label]) && counts[label] >= QUOTA);
  // Initial and retained arming both require this exact frozen profile and its own held-out evidence.
  const modelEligible = decoder => !!decoder && decoder.trained === true && decoder.passed_validation === true
    && decoder.arm_eligible === true && decoder.evaluation?.passed === true
    && decoder.feature_profile === 'cnn_blink_fast_endpoint_v1' && decoder.window_seconds === 1
    && decoder.validation_scope === 'held_out_one_second_trials'
    && Array.isArray(decoder.labels) && decoder.labels.length === 2 && LABELS.every(label => decoder.labels.includes(label))
    && complete(decoder.train_counts) && complete(decoder.validation_counts);
  const armingSuppressed = () => stopRequested || ['stop','eeg_arm','eeg_trial','eeg_train',
    'eeg_reset','eeg_camera','eeg_load_saved'].includes(pending);
  function releaseGuard(token = guardToken, unconditional = false) {
    if (token === guardToken || unconditional) guardToken = null;
    // A late token can revoke itself, but cannot revoke a newer main-process lease.
    try { Promise.resolve(stopEegGuard(unconditional ? undefined : token)).catch(() => {}); } catch (_) {}
  }
  const batchCounts = phase => status?.eeg_control?.decoder?.[phase === 'train' ? 'train_counts' : 'validation_counts'] || {};
  const trialName = label => label === 'rest' ? 'No command' : 'Double blink';
  const trainingCount = counts => LABELS.every(label => Number.isInteger(counts[label]) && counts[label] >= 0)
    ? counts.rest + counts.select : null;
  const trainingLimit = () => Number.isInteger(status?.eeg_control?.decoder?.max_training_trials)
    && status.eeg_control.decoder.max_training_trials >= 16 ? status.eeg_control.decoder.max_training_trials : TRAINING_LIMIT;
  function negativeCue(phase) {
    const index = batchCounts(phase).rest;
    return ['Keep your eyes open and read normally', 'Make one gentle single blink', 'Make one gentle head movement'][Number.isInteger(index) && index >= 0 ? index % 3 : 0];
  }
  function trialDisplay(trial) {
    // Preparation never borrows the recording instruction, even when old metadata lacks a cue.
    if (trial.stage === 'prepare') return `${trialName(trial.label)} starts in ${seconds(trial.prepare_remaining_seconds)} — ${text(trial.cue, 'Keep your eyes open and wait for the recording cue.')}`;
    if (trial.remaining_seconds <= 0) return 'Finishing signal collection… keep your eyes open and wait for acceptance.';
    const cue = text(trial.cue, trial.label === 'select'
      ? 'Make two quick blinks now, finishing both within this second; then keep your eyes open.'
      : `${negativeCue(trial.phase)} now; do not make a double blink.`);
    return `${trialName(trial.label)} · ${seconds(trial.remaining_seconds)} — ${cue}`;
  }
  function calibrationChanged(previous, current) {
    const before = previous.eeg_control || {}, after = current.eeg_control || {};
    return previous.eeg?.session_id !== current.eeg?.session_id
      || ['method','target','camera_mouse_enabled','saved_model_source'].some(key => before[key] !== after[key])
      || before.decoder?.trained !== after.decoder?.trained
      || ['train_counts','validation_counts'].some(phase => LABELS.some(label => after.decoder?.[phase]?.[label] < before.decoder?.[phase]?.[label]));
  }
  function stopBatch(message) {
    cancel(batchTimer); batchTimer = null; batch = null;
    batchNotice = message;  // Cancelling this local queue never resets or aborts the backend's currently collecting trial.
  }
  function batchMatchesSession(job) {
    // These comparisons prevent a queued click from applying its label to another model or a newly restarted acquisition.
    return isConnected() && known && visible() && status.running && workflowKnown()
      && cameraMouseEnabled() === job.cameraMouse
      && status.eeg?.session_id === job.session && status.eeg_control?.saved_model_source === job.source
      && (status.eeg_control?.decoder?.trained === true) === (job.phase === 'validate')
      && ['rest', 'select'].every(label => Number.isInteger(batchCounts(job.phase)[label])
        && batchCounts(job.phase)[label] >= job.minimum[label])
      && status.eeg_control?.armed !== true && !stopRequested;
  }
  function queueNextBatchTrial() {
    const job = batch;
    if (!job) return;
    const counts = batchCounts(job.phase);
    if (!['rest', 'select'].every(label => Number.isInteger(counts[label]) && counts[label] >= 0)) {
      stopBatch('Guided collection paused: trial counts could not be confirmed. Existing data is retained.'); return;
    }
    const remaining = ['rest', 'select'].filter(label => counts[label] < job.quota);
    if (!remaining.length) {
      stopBatch(job.phase === 'train' ? 'Training collection complete. Click Train & freeze fast CNN when ready; validation is a separate step.'
        : 'Validation collection complete. Review the result before arming. Controls remain disarmed.'); return;
    }
    if (job.phase === 'train' && trainingCount(counts) >= trainingLimit()) {
      stopBatch(`Training collection is full (${trainingLimit()} trials). Existing data is retained.`); return;
    }
    // Alternate unfinished classes; a class already at quota is never collected again.
    job.label = remaining.find(label => label !== job.lastLabel) || remaining[0];
    job.before = counts[job.label];  // The next request must increase this class by exactly one before any later cue is scheduled.
    job.minimum = { rest: counts.rest, select: counts.select };  // A reset or replacement that removes observed evidence cancels this queue.
    // Fast trials have one authoritative preparation interval in the backend,
    // shared by manual and guided collection. Do not add a second countdown.
    job.stage = 'preparing';
    cancel(batchTimer);
    batchTimer = schedule(() => runBatchCountdown(job), 0);
  }
  function runBatchCountdown(job) {
    batchTimer = null;
    if (batch !== job) return;  // An old timer cannot revive a cancelled or replaced batch.
    if (!batchMatchesSession(job) || !eegFresh() || status?.eeg_control?.pending || pending) {
      stopBatch('Guided collection paused because preview readiness changed. Existing data is retained.'); render(); return;
    }
    if (job.phase === 'train' && trainingCount(batchCounts('train')) >= trainingLimit()) {
      stopBatch(`Training collection is full (${trainingLimit()} trials). Existing data is retained.`); render(); return;
    }
    job.stage = 'requesting';  // Set this before POST so an early status response cannot enqueue a duplicate trial.
    render();
    run('eeg_trial', { label: job.label, phase: job.phase }, job);
  }
  function advanceBatchFromStatus() {
    const job = batch;
    if (!job) return;
    if (!batchMatchesSession(job)) {
      stopBatch('Guided collection stopped because the acquisition, camera option, or calibration changed. Existing data is retained.'); return;
    }
    const trial = status.eeg_control?.pending;
    if (job.stage === 'preparing') {
      if (trial) stopBatch('Guided collection paused: another trial started. Existing data is retained.');
      return;
    }
    if (trial) {
      if (trial.label !== job.label || trial.phase !== job.phase) {
        stopBatch('Guided collection paused: another trial is active. Existing data is retained.'); return;
      }
      job.stage = 'collecting'; return;  // Remaining time can be zero while the backend finishes receiving the signal.
    }
    if (pending === 'eeg_trial') return;  // Wait for the POST acknowledgment even if a concurrent read still shows no trial.
    const accepted = batchCounts(job.phase)[job.label];
    if (accepted !== job.before + 1) {
      stopBatch(`Guided collection paused: ${text(status.eeg_control?.last_result, 'the trial was not accepted.')} Existing data is retained. Retry when ready.`); return;
    }
    job.lastLabel = job.label;
    queueNextBatchTrial();  // Only a confirmed count increment can schedule the next cue.
  }
  function startBatch() {
    if (el['eeg-batch'].disabled || batch) return;
    const phase = status.eeg_control.decoder.trained ? 'validate' : 'train';
    batchNotice = '';
    // Capture phase and source once; a later status change cancels the queue instead of silently changing what it collects.
    batch = { phase, cameraMouse: cameraMouseEnabled(), session: status.eeg.session_id, source: status.eeg_control.saved_model_source,
      quota: QUOTA, lastLabel: null };
    queueNextBatchTrial(); render();  // Starting a batch neither trains a model nor arms either input source.
  }

  function clearCamera() {
    ++frameId;
    el.camera.getContext('2d').clearRect(0, 0, el.camera.width, el.camera.height);
  }
  function clearEeg() { el.eeg.getContext('2d').clearRect(0, 0, el.eeg.width, el.eeg.height); }
  function clearFrames() { clearCamera(); clearEeg(); cancel(staleTimer); staleTimer = null; }

  function scheduleFrameExpiry() {
    cancel(staleTimer);
    staleTimer = null;
    if (!known || !status.running || !visible() || pending === 'stop' || stopRequested) return;
    const deadlines = [];
    if (cameraDisplayEnabled() && cameraFresh()) deadlines.push(500 - age(status.camera.age_ms / 1000) * 1000);
    if (eegFresh()) deadlines.push(750 - age(status.eeg.stats.age_seconds) * 1000);
    if (!deadlines.length) return;
    staleTimer = schedule(() => {
      staleTimer = null;
      // The backend pauses hybrid input during brief tracking loss; preserve its Escape shortcut for recovery.
      if (!eegFresh() && guardToken) releaseGuard();
      render();
      // Camera and EEG have separate deadlines. Expiring the first must not
      // leave the other sensor's last status displayed as fresh indefinitely.
      scheduleFrameExpiry();
    }, Math.max(1, Math.ceil(Math.min(...deadlines)) + 1));
  }


  function render() {
    const online = !!isConnected(), ready = online && known;
    const control = ready ? status.eeg_control || {} : {}, decoder = control.decoder || {};
    const running = ready && status.running, stopping = pending === 'stop' || stopRequested;
    const busy = !!pending || ready && ['starting','stopping'].includes(status.state);
    const compatible = ready && workflowKnown(), cameraMouse = cameraMouseEnabled();
    const trial = control.pending, eligible = modelEligible(decoder);
    // Acknowledged camera/EEG quality pauses keep the lease; readiness is checked only when initially arming.
    const armed = compatible && running && control.armed === true && eegFresh() && eligible && !armingSuppressed();
    put('summary', !online ? 'Backend offline' : !ready ? 'Status unconfirmed' : stopping ? 'Stopping / disarming'
      : `${running ? 'Running' : text(status.state, 'Stopped')} / ${armed ? 'armed' : 'disarmed'}`);
    const operations = { start: 'Starting EEG acquisition…', stop: 'Stopping and disarming…',
      eeg_trial: 'Preparing a guided EEG trial…', eeg_train: 'Training and freezing the CNN…',
      eeg_reset: 'Resetting CNN and trials…', eeg_arm: 'Updating arming…',
      eeg_camera: 'Changing camera mouse and disarming; EEG and the CNN are retained…',
      eeg_load_saved: 'Loading the saved CNN; controls remain disarmed…', eeg_save: 'Saving CNN and trials on this desktop…' };
    put('status', !online ? 'Backend offline. Device status is unconfirmed; frames hidden.'
      : !ready ? 'Waiting for current status. Stop & disarm remains available.'
      : !compatible ? 'The backend has not confirmed the fast double-blink workflow. Stop remains available.'
      : operations[pending] || text(status.reason, text(status.state, 'Stopped')));
    el.status.setAttribute('aria-busy', String(online && busy));
    put('error', error || (ready ? text(status.error) : '')); el.error.hidden = !el.error.textContent;
    // Even EEG-only operation needs exclusive input-controller ownership.
    el.start.disabled = !compatible || busy || running || stopping || status?.cleanup_pending || isCameraRunning() || !isBrainbitConnected();
    put('start', cameraMouse ? 'Start EEG + camera' : 'Start EEG only');
    el.stop.disabled = !online; // Emergency stop is reachable through every pending operation.
    put('start-hint', !isBrainbitConnected() ? 'Connect BrainBit in the panel above before starting.'
      : isCameraRunning() ? 'Stop the separate Hand controls first so this panel can own desktop input.'
      : cameraMouse ? 'Camera mouse uses the same Hand controls tracker, model, and settings. Start does not arm desktop input.'
      : 'Camera mouse is off. EEG clicks use the current cursor; no camera or visible hand is required. Start does not arm clicks.');
    el['eeg-camera'].checked = cameraMouse;
    el['eeg-camera'].disabled = !compatible || typeof control.camera_mouse_enabled !== 'boolean'
      || busy || !!trial || stopping || !!status?.cleanup_pending;
    el['camera-container'].hidden = !cameraDisplayEnabled();
    if (!ready || stopping || !visible()) clearFrames();
    else { if (!cameraDisplayEnabled() || !cameraFresh()) clearCamera(); if (!eegFresh()) clearEeg(); }
    put('camera-status', !cameraDisplayEnabled() ? 'Camera mouse is off or changing.'
      : !running ? 'Camera preview is off.' : !cameraFresh() ? 'Camera frame is stale or unavailable.'
      : `${status.camera.tracked ? 'Hand tracked' : 'No hand tracked'} · ${number(status.camera.fps, 0)} FPS · frame age ${seconds(age(status.camera.age_ms / 1000))}`);
    const pointer = control.hand_pointer || {};
    put('hand-pointer-status', !running ? 'Start EEG + camera, then explicitly arm to move the cursor.'
      : !armed ? 'Camera mouse is disarmed. Explicitly arm, then point briefly to engage hand aiming.'
      : !cameraFresh() || status.camera.tracked !== true ? 'Camera mouse paused: show your hand in a fresh frame. EEG blinks still click at the current cursor without a visible hand.'
      : `Camera mouse ${pointer.ready === true ? 'ready' : 'waiting'}: ${text(pointer.hint, 'Point briefly, then move your hand to aim.')} Pinch clicks and drags remain available.`);
    const stats = ready ? status.eeg?.stats || {} : {};
    put('eeg-status', !running ? 'EEG preview is off.' : !eegFresh() ? 'Waiting for fresh EEG samples.'
      : `${number(stats.received_rate_hz, 0)} Hz received / ${number(status.eeg.nominal_hz, 0)} Hz nominal · age ${seconds(age(stats.age_seconds))} · µV, auto scale per channel`);
    const channelStats = Array.isArray(stats.channels) ? stats.channels.slice(0, 8).map(channel =>
      `${text(channel.name, 'Channel')}: RMS ${number(microvolts(channel.rms_v))} µV, peak-to-peak ${number(microvolts(channel.peak_to_peak_v))} µV`).join(' | ') : '';
    put('diagnostics', ready ? `Counter discontinuities ${count(stats.gaps)} · duplicates ${count(stats.duplicates)} · nonfinite ${count(stats.nonfinite)} · channel mismatches ${count(stats.channel_mismatches)} · queue drops ${count(stats.queue_drops)}${channelStats ? `\n${channelStats}` : ''}` : '');
    put('selection-counts', `EEG clicks: ${count(control.click_count ?? 0)}${cameraMouse ? ` · Finger clicks: ${count(control.finger_click_count ?? 0)}` : ''}`);
    put('eeg-arm-label', cameraMouse ? 'Arm camera mouse + EEG clicks' : 'Arm EEG mouse clicks');
    el['eeg-arm'].checked = !!armed;
    el['eeg-arm'].disabled = !compatible || busy || !running || stopping || !!trial || !!batch
      || (!armed && (control.can_arm !== true || !eligible || !eegFresh()));
    const paused = armed && cameraMouse && control.signal_quality_paused === true;
    put('control-status', !ready ? 'Control state unconfirmed. Arming is disabled.'
      : stopping ? 'Stop requested. Controls are disarmed while the backend confirms.'
      : paused ? `Camera mouse active; EEG clicks paused: ${text(control.signal_quality_reason, text(control.prediction?.reason, 'Waiting for clean EEG.'))} Camera aiming, pinch clicks, and drags remain available. EEG clicks wait for a full clean 1-second window and eyes-open rest. Escape stops both controls.`
      : armed ? `${cameraMouse ? 'Camera mouse + EEG clicks armed: point briefly, then move your hand to aim. Finger clicks and drags take priority.' : 'EEG mouse clicks armed: aim with your usual mouse or touchpad.'} One recognized double blink sends one left click at the current cursor, even without a visible hand. Escape stops controls from any app.`
      : 'Prediction only. Desktop controls are disarmed. A validated CNN and a separate explicit arm are required.');
    const readiness = decoder.trained ? describeCnnReadiness(control, decoder, control.prediction, eligible) : '';
    if (armed && !paused) put('control-status', `${el['control-status'].textContent} ${readiness}`);
    if (!armed && control.armed === false && text(control.last_disarm_reason))
      put('control-status', `${el['control-status'].textContent}\nLast disarm: ${text(control.last_disarm_reason)}`);
    const prediction = control.prediction;
    put('prediction', prediction?.valid === true && eegFresh() && Number.isFinite(prediction.confidence)
      ? `EEG blink detection: ${LABELS.includes(prediction.label) ? trialName(prediction.label).toLowerCase() : 'uncertain'} · uncalibrated score ${number(prediction.confidence, 3)} · ${armed ? 'control armed' : 'display only'}${readiness ? `\n${readiness}` : ''}`
      : `EEG prediction unavailable. ${!eegFresh() ? 'Waiting for fresh EEG.' : text(prediction?.reason, decoder.trained ? 'Waiting for a complete EEG window.' : 'Collect training trials and train the CNN first.')}${readiness ? `\n${readiness}` : ''}`);
    renderTraining({ compatible, running, busy, stopping, control, decoder, trial, eligible });
    onResize();
  }

  function renderTraining({ compatible, running, busy, stopping, control, decoder, trial, eligible }) {
    const trainCounts = decoder.train_counts || {}, validationCounts = decoder.validation_counts || {};
    const trained = decoder.trained === true, trainingComplete = complete(trainCounts), validationComplete = complete(validationCounts);
    const phase = trained ? 'validate' : 'train', activeCounts = trained ? validationCounts : trainCounts;
    const total = trainingCount(trainCounts), limit = trainingLimit(), full = !trained && (total === null || total >= limit);
    // Loaded inference-bound archives keep historical examples immutable; resetting is a separate action.
    const bound = decoder.inference_context_bound === true;
    const blocked = !compatible || busy || !running || stopping || !!trial || !!batch || !eegFresh() || bound;
    for (const label of LABELS) {
      put(`train-${label}`, `${count(trainCounts[label])} / ${QUOTA}`);
      put(`validate-${label}`, `${count(validationCounts[label])} / ${QUOTA}`);
      // Extra training can repair a failed fit, but never extends frozen held-out validation.
      el[`eeg-${label}`].disabled = blocked || full || !Number.isInteger(activeCounts[label]) || activeCounts[label] < 0
        || ((trained || !trainingComplete) && activeCounts[label] >= QUOTA);
    }
    put('eeg-rest', !trained && trainingComplete ? 'Collect another no-command trial' : 'Collect no-command trial');
    put('eeg-select', !trained && trainingComplete ? 'Collect another double-blink trial' : 'Collect double-blink trial');
    put('eeg-batch', trained ? 'Collect remaining validation' : 'Collect remaining training');
    el['eeg-batch'].disabled = blocked || !visible() || control.armed === true || (trained ? validationComplete : trainingComplete)
      || full || trainingCount(activeCounts) === null || typeof status?.eeg?.session_id !== 'string' || !status.eeg.session_id;
    el['eeg-batch-cancel'].disabled = !batch;
    put('eeg-batch-status', batch ? `Guided ${batch.phase === 'train' ? 'training' : 'validation'} · No command ${count(activeCounts.rest)} / 8 · Double blink ${count(activeCounts.select)} / 8. Cancel stops future cues; the current trial can finish.`
      : batchNotice || (bound ? 'Saved examples are retained. Reset explicitly before collecting a new model.'
        : control.armed === true ? 'Disarm controls before guided collection. Existing trials are kept.'
        : 'Collect the remaining trials with one click. Each server countdown lasts 3 seconds before a 1-second recording. Existing trials are kept.'));
    put('eeg-guided-cue', trial ? trialDisplay(trial) : batch
      ? `Preparing ${trialName(batch.label)}… wait for the countdown and recording cue.`
      : `Next No command trial: ${negativeCue(phase)} during recording; do not make a double blink. Double blink: complete both quick blinks during the 1-second recording cue.`);
    el['eeg-train'].disabled = !compatible || busy || stopping || !!trial || !!batch || !running || !eegFresh()
      || trained || !trainingComplete || decoder.can_train === false || bound;
    el['eeg-reset'].disabled = !compatible || busy || stopping || !!trial;
    el['eeg-load-saved'].disabled = !compatible || busy || !running || stopping || !!trial || !!batch || status?.cleanup_pending || !eegFresh();
    el['eeg-save'].disabled = !compatible || busy || stopping || !!trial || !!batch || !trained;
    put('eeg-saved-status', control.saved_model_file ? `CNN, training, and validation saved locally: ${text(control.saved_model_file)}.`
      : control.saved_model_source ? `Saved CNN loaded: ${text(control.saved_model_source)}. Its original validation result is retained.`
      : 'Use saved fast CNN restores this one-second profile and its own validation. Loading never arms controls or bypasses failed validation.');
    put('eeg-trial', trial ? `${trained ? 'Validation' : 'Training'}: ${trialDisplay(trial)} Controls are disarmed.`
      : text(control.last_result, trained ? 'CNN frozen. New validation trials do not retrain it.' : 'Collect at least 8 accepted examples per class, then train and freeze the CNN.'));
    put('eeg-model-status', trained ? `Frozen CNN (1-second window) · ${text(decoder.state, 'collecting_validation')}. Training and validation remain separate.`
      : trainingComplete ? `Training quota reached. Select Train & freeze fast CNN before validation.${full ? ` Training collection is full (${limit} trials).` : ` You can add either class before freezing (${total}/${limit} retained).`}`
      : full ? `Training collection is unavailable or full (${limit} trials). Confirm counts before continuing.`
      : 'Collect at least 8 accepted training trials per class. Rejected trials do not increase counts.');
    put('eeg-errors', [eegActionError, text(decoder.training_error || decoder.train_error), text(decoder.validation_error)].filter(Boolean).join('\n'));
    el['eeg-errors'].hidden = !el['eeg-errors'].textContent;
    const evaluation = decoder.evaluation || {}, percent = value => Number.isFinite(value) ? `${number(value * 100, 0)}%` : 'pending';
    const lines = [validationComplete ? `Held-out one-second trials · balanced accuracy ${percent(evaluation.balanced_accuracy)}. These results do not measure physical blink-to-click latency or continuous false clicks.`
      : 'Collect 8 new validation trials per class against the frozen CNN.'];
    for (const label of LABELS) lines.push(`${trialName(label)}: precision ${percent(evaluation.per_class?.[label]?.precision)} · recall ${percent(evaluation.per_class?.[label]?.recall)}.`);
    const matrix = evaluation.confusion_matrix || evaluation.confusion;
    if (matrix) {
      lines.push('Confusion rows = actual; columns = predicted No command / Double blink / uncertain:');
      LABELS.forEach((label, index) => {
        const row = Array.isArray(matrix) ? matrix[index] : matrix[label];
        if (row) lines.push(`${trialName(label)}: ${[...LABELS, 'uncertain'].map((name, column) => count(Array.isArray(row) ? row[column] : row[name])).join(' / ')}`);
      });
    }
    lines.push(`No-command false activations: ${count(evaluation.rest_false_activations)} / ${count(evaluation.rest_trials)} held-out trials (${percent(evaluation.rest_false_activation_rate)}).`);
    put('eeg-evaluation', lines.join('\n'));
    const thresholds = evaluation.thresholds || {};
    // The fast service publishes measured metrics and an overall result, not legacy per-gate flags.
    const requirements = ['8 validation trials per class', `balanced accuracy ≥ ${percent(thresholds.balanced_accuracy ?? .75)}`,
      `each recall ≥ ${percent(thresholds.each_recall ?? .70)}`, `each precision ≥ ${percent(thresholds.each_precision ?? .70)}`,
      `no-command false activations ≤ ${percent(thresholds.rest_false_activation_rate ?? .10)}`];
    put('eeg-gates', `${eligible ? 'Validation gates passed' : 'Arming locked until CNN validation passes'}: ${text(control.reason, text(evaluation.reason, 'Collect and validate the frozen CNN.'))}\nRequired: ${requirements.join(' · ')}`);
  }
  function describeCnnReadiness(control, decoder, prediction, eligible) {
    if (!eegFresh()) return 'CNN paused: waiting for fresh EEG.';
    if (!eligible) return 'CNN clicks locked: the frozen model must pass separate validation.';
    if (control.signal_quality_paused === true) {
      return `EEG clicks paused: ${text(control.signal_quality_reason,
        text(prediction?.reason, 'Waiting for a full clean EEG window and eyes-open rest.'))}`;
    }
    const reason = text(prediction?.reason, text(decoder.reason, text(control.eeg_click_reason, text(control.reason))));
    if (prediction?.valid !== true) return `CNN paused. ${reason || 'Waiting for a usable EEG window.'}`;
    const remaining = Number.isFinite(prediction?.suppressed_evidence_remaining_seconds)
      ? prediction.suppressed_evidence_remaining_seconds : decoder.suppressed_evidence_remaining_seconds;
    if (Number.isFinite(remaining) && remaining > 0) {
      // This is remaining EEG evidence, not a wall-clock countdown or permission to click.
      return `CNN recovery: ${seconds(remaining)} of newer EEG still needed after finger input. ${reason}`;
    }
    const readyForBlink = prediction?.ready_for_blink !== false && decoder.ready_for_blink !== false
      && (prediction?.ready_for_blink === true || decoder.ready_for_blink === true);
    if (readyForBlink && control.can_eeg_click === true) return 'CNN ready for a new double blink.';
    if (readyForBlink) return `CNN waiting. ${text(control.eeg_click_reason, text(control.reason, reason || 'Waiting for click readiness.'))}`;
    if (prediction?.label === 'select' && prediction?.event_id) {
      return 'CNN recognized a double blink. Keep your eyes open until the detector is ready for the next one.';
    }
    return `CNN waiting. ${reason || 'Keep your eyes open until the detector recognizes rest.'}`;
  }


  function drawCamera(camera) {
    clearCamera(); // Invalidate every earlier image decode, including frames from a stopped camera.
    if (!cameraDisplayEnabled() || !cameraFresh() || !visible()) return;
    if (typeof camera.image !== 'string' || !/^data:image\/jpeg;base64,[A-Za-z0-9+/=]+$/.test(camera.image)) {
      put('camera-status', 'Camera frame could not be displayed.'); return;
    }
    const id = frameId, currentEpoch = epoch, image = imageFactory();
    image.onload = () => {
      if (id !== frameId || currentEpoch !== epoch || !cameraDisplayEnabled() || !cameraFresh() || !visible() || !isConnected()) return;
      const width = Number.isInteger(camera.width) && camera.width > 0 && camera.width <= 480 ? camera.width : 480;
      const height = Number.isInteger(camera.height) && camera.height > 0 && camera.height <= 960 ? camera.height : 360;
      el.camera.width = width; el.camera.height = height;
      const context = el.camera.getContext('2d');
      context.drawImage(image, 0, 0, width, height);
      drawHandOverlay(context, camera, width, height); // The standalone camera uses this same joint renderer.
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
    const previousStatus = status;
    const previousCameraMouse = cameraMouseEnabled();
    status = { ...data, camera, eeg };
    // Once collection has stopped, discard guidance from an old phase/source/reset; active queues publish their own cancellation reason below.
    if (!batch && previousStatus && calibrationChanged(previousStatus, status)) batchNotice = '';
    // Freshness belongs to this snapshot's arrival, not the preceding response's age.
    receivedAt = now(); known = true;
    onCameraPrivacy(data.camera_privacy); // Aggregate ownership has its own revision in the privacy controller.
    // Tracking pauses keep Escape active; a confirmed backend disarm or stopped run releases it.
    if (guardToken && (previousCameraMouse !== cameraMouseEnabled() || !workflowKnown() || !modelEligible(status.eeg_control?.decoder)
      || (!data.eeg_control?.armed || !eegFresh()) && pending !== 'eeg_arm')) releaseGuard();
    wasActive = wasActive || data.running || data.eeg_control?.armed === true;
    if (!data.running && !data.eeg_control?.armed && !data.cleanup_pending) {
      stopRequested = false;
      wasActive = false;
    }
    error = '';
    advanceBatchFromStatus();  // Count changes and pending state, not a local recording timer, determine acceptance.
    render();
    if (raw && data.running && pending !== 'stop' && visible()) {
      drawCamera(data.camera || {});
      drawEeg(data.eeg || {});
    }
    scheduleFrameExpiry();
    return true;
  }

  function failure(message) {
    if (batch) stopBatch('Guided collection paused: connection or trial acknowledgment failed. Existing data is retained.');
    // A failed cached read during guard preparation must revoke that future
    // commitment before its asynchronously prepared token can be used.
    if (pending === 'eeg_arm') {
      ++actionId;
      releaseGuard(undefined, true);
    }
    if (guardToken) releaseGuard();
    ++generation; known = false; pending = null; error = message;
    clearFrames(); render();
  }
  function resetConnection() {
    if (batch) stopBatch('Guided collection stopped when the connection changed. Existing data is retained.');
    else batchNotice = '';  // A new backend connection cannot inherit a stale training-complete instruction.
    if (guardToken || pending === 'eeg_arm') releaseGuard(undefined, true);
    ++epoch; ++generation; ++actionId; ++readId; revision = -1;
    status = null; known = false; pending = null; error = ''; reading = false;
    polling = false; cancel(pollTimer); pollTimer = null; clearFrames();
  }
  function syncPolling() {
    const active = !!(isConnected() && visible());
    if (!active) {
      if (batch) stopBatch('Guided collection paused because the panel is hidden. The current trial can finish; existing data is retained.');
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
  async function run(action, payload = {}, guidedJob = null) {
    const button = { start: el.start, stop: el.stop, eeg_trial: el[`eeg-${payload.label}`],
      eeg_train: el['eeg-train'], eeg_reset: el['eeg-reset'], eeg_load_saved: el['eeg-load-saved'],
      eeg_save: el['eeg-save'], eeg_arm: el['eeg-arm'], eeg_camera: el['eeg-camera'] }[action];
    const guidedTrial = guidedJob && guidedJob === batch && action === 'eeg_trial';
    if (!button || (button.disabled && !guidedTrial) || !isConnected() || (pending && action !== 'stop') || pending === 'stop') { render(); return; }
    if (batch && !guidedTrial) stopBatch('Guided collection stopped by another control. Existing data is retained.');
    const id = ++actionId, currentEpoch = epoch;
    ++generation; pending = action; error = '';
    const preparingGuard = action === 'eeg_arm' && payload.enabled === true;
    // Bind asynchronous arming to the output the user actually selected at click time.
    const requestedCameraMouse = cameraMouseEnabled();
    let preparedToken = null;
    // A mutating control revokes the old global shortcut lease before changing output.
    if (['stop','eeg_camera','eeg_reset','eeg_train','eeg_trial','eeg_load_saved'].includes(action)
      || (action === 'eeg_arm' && payload.enabled === false)) releaseGuard(undefined, true);
    if (action === 'stop') { stopRequested = true; wasActive = true; }
    if (action.startsWith('eeg_')) eegActionError = '';
    if (action === 'stop' || action === 'eeg_trial' || action === 'eeg_camera') clearFrames();
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
        // Registering Escape is asynchronous. Freshness, camera option, and model
        // eligibility must still hold immediately before the arm request.
        // The fixed workflow must still be valid when asynchronous guard preparation finishes.
        if (!known || !status.running || !workflowKnown() || !eegFresh()
          || cameraMouseEnabled() !== requestedCameraMouse || status.eeg_control?.can_arm !== true
          || !modelEligible(status.eeg_control?.decoder) || !!status.eeg_control?.pending || stopRequested) {
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
  el['eeg-arm'].addEventListener('change', () => { const enabled = el['eeg-arm'].checked; render(); run('eeg_arm', { enabled }); });
  el['eeg-camera'].addEventListener('change', () => { const enabled = el['eeg-camera'].checked; render(); run('eeg_camera', { enabled }); });
  // The frozen state chooses the phase; neither a UI selector nor the trial payload can retrain validation.
  for (const label of LABELS) el[`eeg-${label}`].addEventListener('click', () =>
    run('eeg_trial', { label, phase: status?.eeg_control?.decoder?.trained ? 'validate' : 'train' }));
  el['eeg-train'].addEventListener('click', () => run('eeg_train'));
  el['eeg-load-saved'].addEventListener('click', () => run('eeg_load_saved'));
  el['eeg-save'].addEventListener('click', () => run('eeg_save'));
  el['eeg-reset'].addEventListener('click', () => run('eeg_reset'));
  el['eeg-batch'].addEventListener('click', startBatch);
  el['eeg-batch-cancel'].addEventListener('click', () => {
    if (batch) stopBatch('Queued trials cancelled. The current trial can finish; all accepted data is retained.');
    render();
  });
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
      if (batch) stopBatch('Guided collection stopped by Escape. Existing data is retained.');
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
      if (batch) stopBatch('Guided collection stopped by the emergency stop. Existing data is retained.');
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
