// Experimental display/controller. Sensor frames remain transient in this renderer.
const { requestMultimodal } = require('./multimodal-http.cjs');
// This is the backend's accepted-training archive capacity, not the eight-trial class quota or the separate validation quota.
const BLINK_TRAINING_LIMIT = 80;
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
    'validate-left','validate-right','validate-rest',
    // Selection mode reuses the rest controls and adds only its second class.
    'eeg-select','train-select','validate-select','column-left','column-right','column-rest','column-select',
    // These labels and containers change without replacing live input elements.
    'mode-note','calibration-heading','calibration-instructions','session-note','camera-container',
    'target-controls','eeg-target','webcam-arm-label','eeg-arm-label','selection-counts','hand-pointer-status',
    'eeg-load-saved','eeg-save','eeg-saved-status','validation-note','method-controls','eeg-method',
    'eeg-guided-controls','eeg-batch','eeg-batch-cancel','eeg-batch-status','eeg-guided-cue'];
  const el = Object.fromEntries(ids.map(id => [id, document.getElementById(`multimodal-${id}`)]));
  let status = null, known = false, error = '', pending = null, epoch = 0, actionId = 0;
  let generation = 0, revision = -1, readId = 0, reading = false, polling = false;
  let pollTimer = null, staleTimer = null, frameId = 0, receivedAt = 0;
  let lastOnline = false;
  let stopRequested = false, wasActive = false, eegActionError = '';
  let guardToken = null;
  // The local queue owns only the next cue/countdown; all samples, acceptance decisions, and model state belong to the backend.
  let batch = null, batchTimer = null, batchNotice = '';
  const text = (value, fallback = '') => typeof value === 'string' && value.trim() ? value.slice(0, 600) : fallback;
  const put = (id, value) => { if (el[id].textContent !== value) el[id].textContent = value; };
  const visible = () => el.panel.open && document.hidden !== true;
  const valid = data => data && typeof data.state === 'string' && typeof data.running === 'boolean'
    && typeof data.armed === 'boolean';
  const age = value => Number.isFinite(value) && value >= 0 ? value + (now() - receivedAt) / 1000 : Infinity;
  const cameraFresh = () => known && status?.running && status.camera?.running
    && Number.isFinite(status.camera.age_ms) && age(status.camera.age_ms / 1000) <= 0.5;
  const eegFresh = () => known && status?.running && status.eeg?.state === 'running'
    && status.eeg?.mode === 'signal' && age(status.eeg.stats?.age_seconds) <= 0.75;
  const fresh = () => cameraFresh() && eegFresh();
  const number = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : 'unknown';
  const microvolts = value => Number.isFinite(value) ? value * 1e6 : NaN;
  const count = value => Number.isFinite(value) && value >= 0 ? String(Math.floor(value)) : 'unknown';
  const seconds = value => Number.isFinite(value) ? `${Math.max(0, value).toFixed(1)} s` : 'unknown';
  // Old backend snapshots have no mode; retain their original webcam behavior.
  const controlMode = () => !status ? 'eeg_click' : ['eeg', 'eeg_click'].includes(status.control_mode) ? status.control_mode : 'webcam';
  // Both EEG modes need the same freshness and global Escape guard checks.
  const eegMode = () => ['eeg', 'eeg_click'].includes(controlMode());
  // Click arming starts hand engagement, so it needs fresh tracking but not an already engaged pointer.
  const clickHandFresh = () => controlMode() !== 'eeg_click' || (cameraFresh()
    && status.camera?.tracked === true && status.eeg_control?.hand_pointer?.required === true);
  // Hand controls can start without a trained/fresh EEG signal; the backend gates the EEG add-on separately.
  const eegControlFresh = () => controlMode() === 'eeg_click' ? clickHandFresh() : eegFresh();
  // An unconfirmed target must never appear armed or gain a new Escape lease.
  const eegTargetKnown = () => controlMode() !== 'eeg_click' || ['practice', 'mouse'].includes(status?.eeg_control?.target);
  // Older backends omit the method field; only that omission defaults to the existing eyes-closed model.
  const eegMethod = () => status?.eeg_control?.method === undefined ? 'eyes_closed' : status.eeg_control.method;
  // Unknown method names must not inherit permission to issue a different kind of EEG click.
  const eegMethodKnown = () => controlMode() !== 'eeg_click' || ['eyes_closed', 'double_blink'].includes(eegMethod());
  const doubleBlinkMode = () => controlMode() === 'eeg_click' && eegMethod() === 'double_blink';
  // The mode determines the expected vocabulary; arbitrary server labels cannot unlock arming.
  const eegLabels = () => controlMode() === 'eeg_click' ? ['rest', 'select'] : ['left', 'right', 'rest'];
  // Switching the target immediately clears the displayed arm before the POST returns.
  const armingSuppressed = () => stopRequested || ['stop', 'arm', 'eeg_arm', 'calibrate', 'eeg_trial', 'eeg_train', 'eeg_reset', 'eeg_target', 'eeg_method', 'eeg_load_saved', 'control_mode'].includes(pending);
  const eegEligible = decoder => !!decoder && decoder.trained === true && decoder.passed_validation === true
    && decoder.arm_eligible === true && decoder.evaluation?.passed === true
    // Click mode must explicitly confirm the binary vocabulary; legacy snapshots may omit it.
    && (controlMode() !== 'eeg_click' && !decoder.labels || Array.isArray(decoder.labels)
      && decoder.labels.length === eegLabels().length && eegLabels().every(label => decoder.labels.includes(label)))
    // A successful flag alone never substitutes for the per-class held-out quotas.
    && eegLabels().every(label => decoder.train_counts?.[label] >= 8 && decoder.validation_counts?.[label] >= 8);
  // Recoverable camera/EEG pauses retain the shared hand arm and Escape stop lease.
  const armModelEligible = () => controlMode() === 'eeg_click' || eegEligible(status?.eeg_control?.decoder);
  const armedInputsAvailable = () => controlMode() === 'eeg_click' ? status?.running === true : eegControlFresh();
  function releaseGuard(token = guardToken, unconditional = false) {
    if (token === guardToken || unconditional) guardToken = null;
    // Main owns the lease/shortcut. Scoped cleanup cannot cancel a newer lease.
    try { Promise.resolve(stopEegGuard(unconditional ? undefined : token)).catch(() => {}); } catch (_) {}
  }

  const batchCounts = phase => status?.eeg_control?.decoder?.[phase === 'train' ? 'train_counts' : 'validation_counts'] || {};
  const trialName = label => label === 'rest' ? 'No command' : 'Double blink';
  // Invalid or incomplete count metadata must not make an archive look as though it has free slots.
  const blinkTrainingCount = counts => ['rest', 'select'].every(label => Number.isInteger(counts[label]) && counts[label] >= 0)
    ? counts.rest + counts.select : null;
  function calibrationChanged(previous, current) {
    const before = previous.eeg_control || {}, after = current.eeg_control || {};
    // A completed/paused cue notice belongs to one method, acquisition, saved source, and training/validation phase.
    return previous.control_mode !== current.control_mode || previous.eeg?.session_id !== current.eeg?.session_id
      || ['method', 'saved_model_source', 'saved_calibration_source'].some(key => before[key] !== after[key])
      || before.decoder?.trained !== after.decoder?.trained
      // Resetting or replacing evidence can also occur without changing the method or the acquisition identity.
      || ['train_counts', 'validation_counts'].some(phase => ['rest', 'select'].some(label =>
        after.decoder?.[phase]?.[label] < before.decoder?.[phase]?.[label]));
  }
  function stopBatch(message) {
    cancel(batchTimer); batchTimer = null; batch = null;
    batchNotice = message;  // Cancelling this local queue never resets or aborts the backend's currently collecting trial.
  }
  function batchMatchesSession(job) {
    // These comparisons prevent a queued click from applying its label to another model or a newly restarted acquisition.
    return isConnected() && known && visible() && status.running && doubleBlinkMode()
      && status.eeg?.session_id === job.session && status.eeg_control?.saved_model_source === job.source
      && status.eeg_control?.saved_calibration_source === job.calibrationSource
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
      stopBatch(job.phase === 'train' ? 'Training collection complete. Click Train & freeze model when ready; validation is a separate step.'
        : 'Validation collection complete. Review the result before arming. Controls remain disarmed.'); return;
    }
    if (job.phase === 'train' && blinkTrainingCount(counts) >= BLINK_TRAINING_LIMIT) {
      stopBatch('Training collection is full (80 trials). Existing data is retained.'); return;
    }
    // Alternate unfinished classes; a class already at quota is never collected again.
    job.label = remaining.find(label => label !== job.lastLabel) || remaining[0];
    job.before = counts[job.label];  // The next request must increase this class by exactly one before any later cue is scheduled.
    job.minimum = { rest: counts.rest, select: counts.select };  // A reset or replacement that removes observed evidence cancels this queue.
    job.stage = 'preparing'; job.deadline = now() + 3000;  // Preparation time is local UI time, never evidence that a signal trial finished.
    cancel(batchTimer);
    batchTimer = schedule(() => runBatchCountdown(job), 250);
  }
  function runBatchCountdown(job) {
    batchTimer = null;
    if (batch !== job) return;  // An old timer cannot revive a cancelled or replaced batch.
    if (!batchMatchesSession(job) || !fresh() || status?.eeg_control?.pending || pending) {
      stopBatch('Guided collection paused because preview readiness changed. Existing data is retained.'); render(); return;
    }
    if (job.phase === 'train' && blinkTrainingCount(batchCounts('train')) >= BLINK_TRAINING_LIMIT) {
      stopBatch('Training collection is full (80 trials). Existing data is retained.'); render(); return;
    }
    if (now() < job.deadline) {
      render(); batchTimer = schedule(() => runBatchCountdown(job), 250); return;
    }
    job.stage = 'requesting';  // Set this before POST so an early status response cannot enqueue a duplicate trial.
    render();
    run('eeg_trial', { label: job.label, phase: job.phase }, job);
  }
  function advanceBatchFromStatus() {
    const job = batch;
    if (!job) return;
    if (!batchMatchesSession(job)) {
      stopBatch('Guided collection stopped because the preview, method, or calibration changed. Existing data is retained.'); return;
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
    batch = { phase, session: status.eeg.session_id, source: status.eeg_control.saved_model_source,
      calibrationSource: status.eeg_control.saved_calibration_source,
      quota: Math.max(8, status.eeg_control.decoder.min_per_class || 8), lastLabel: null };
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
    if (cameraFresh()) deadlines.push(500 - age(status.camera.age_ms / 1000) * 1000);
    if (eegFresh()) deadlines.push(750 - age(status.eeg.stats.age_seconds) * 1000);
    if (!deadlines.length) return;
    staleTimer = schedule(() => {
      staleTimer = null;
      // The backend pauses hybrid input during brief tracking loss; preserve its Escape shortcut for recovery.
      if (eegMode() && !armedInputsAvailable() && guardToken) releaseGuard();
      render();
      // Camera and EEG have separate deadlines. Expiring the first must not
      // leave the other sensor's last status displayed as fresh indefinitely.
      scheduleFrameExpiry();
    }, Math.max(1, Math.ceil(Math.min(...deadlines)) + 1));
  }

  function render() {
    const online = !!isConnected(), ready = online && known;
    const running = ready && status.running;
    const busy = ready && (!!pending || ['starting','stopping','contact'].includes(status?.state));
    const stopping = pending === 'stop' || stopRequested;
    const armed = ready && controlMode() === 'webcam' && status.armed && fresh() && !armingSuppressed();
    // An acknowledged hybrid arm survives temporary sensor pauses; EEG clicks still require backend readiness.
    const eegArmed = ready && eegMode() && status.eeg_control?.armed && armedInputsAvailable()
      && eegTargetKnown() && eegMethodKnown() && armModelEligible() && !armingSuppressed();
    put('summary', !online ? 'Backend offline' : !ready ? 'Status unconfirmed'
      : stopping ? 'Stopping / disarming' : running && !visible() ? 'Running / open to refresh status'
      : `${running ? 'Running' : text(status.state, 'Stopped')} / ${eegArmed ? 'EEG armed' : armed ? 'webcam armed' : 'prediction only'}`);
    const labels = { start: 'Starting preview...', stop: 'Stopping preview and disarming...',
      contact: 'Checking contacts for 5 seconds...', arm: 'Updating swipe arming...',
      calibrate: 'Preparing a 3-second trial...', reset_calibration: 'Clearing calibration...',
      eeg_trial: 'Preparing a guided EEG trial...', eeg_train: 'Training and freezing the EEG model...',
      // Keep pending-action wording accurate for both selection and swipe targets.
      eeg_reset: 'Resetting EEG model and trials...', eeg_arm: 'Updating EEG arming...',
      eeg_target: 'Changing selection target and disarming...',
      eeg_method: 'Changing EEG click method and disarming; both models are retained...',
      eeg_load_saved: 'Loading the saved EEG model; controls remain disarmed...',
      eeg_save: 'Saving EEG model, training, and validation on this desktop...',
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
      // Viewing signals never implicitly starts practice output or desktop actions.
      : 'Start preview is explicit. Hand movement, finger clicks, and EEG actions remain disarmed until you arm them.');
    if (!ready || stopping || !visible()) clearFrames();
    else {
      if (!cameraFresh()) clearCamera();
      if (!eegFresh()) clearEeg();
    }
    // Only the advanced EEG swipe mode can operate without a camera.
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
    // The active mode selects the two-state or legacy three-class workflow.
    const labels = eegLabels();
    const quota = Number.isInteger(decoder.min_per_class) && decoder.min_per_class >= 8 ? decoder.min_per_class : 8;
    const trainCounts = decoder.train_counts || {}, validationCounts = decoder.validation_counts || {};
    const trainingComplete = labels.every(label => Number.isInteger(trainCounts[label]) && trainCounts[label] >= quota);
    const validationComplete = labels.every(label => Number.isInteger(validationCounts[label]) && validationCounts[label] >= quota);
    const eligible = eegEligible(decoder) && trainingComplete && validationComplete;
    const mode = controlMode();
    // The default target routes EEG selections to a counter; armed finger input keeps its normal behavior.
    const clickMode = mode === 'eeg_click', target = control.target === 'mouse' ? 'mouse' : 'practice';
    // A blink model is separate from the older sustained-eyes-closed model, even though both use rest/select labels on the wire.
    const blinkMode = doubleBlinkMode();
    const blinkTrainingTotal = blinkTrainingCount(trainCounts);
    // The cap applies to all unfrozen blink training, including an unfinished no-command quota; validation has its own slots.
    const blinkTrainingFull = blinkMode && decoder.trained === false
      && (blinkTrainingTotal === null || blinkTrainingTotal >= BLINK_TRAINING_LIMIT);
    // Only acknowledged, recognized targets can enable selection arming.
    const targetKnown = eegTargetKnown();
    el['control-mode'].value = mode;
    el['control-mode'].disabled = !ready || busy || !!trial || stopping || status?.cleanup_pending;
    // Hybrid selection keeps the hand camera visible while hiding unrelated swipe controls.
    el['camera-container'].hidden = false;
    el.motion.hidden = clickMode;
    el['webcam-arm-label'].hidden = clickMode;
    // Pointer feedback explains deliberate engagement, tracking loss and held positions next to the camera.
    el['hand-pointer-status'].hidden = !clickMode;
    const pointer = control.hand_pointer || {};
    put('hand-pointer-status', !ready ? 'Hand pointer status unconfirmed.'
      : !running ? 'Hand pointer is off. Start preview, then explicitly arm.'
      : !clickHandFresh() ? 'Hand pointer paused: keep your hand visible in a fresh camera frame. Tracking resumes automatically while armed.'
      : `${pointer.ready === true ? 'Hand pointer ready' : 'Hand pointer waiting'}: ${text(pointer.hint,
        eegArmed ? 'Point your index finger briefly, then move your hand to aim.' : 'Controls are disarmed.')}`);
    // Target selection is independent of starting acquisition and always disarms output.
    el['target-controls'].hidden = !clickMode;
    el['eeg-target'].value = target;
    el['eeg-target'].disabled = !ready || !clickMode || busy || !!trial || stopping || status?.cleanup_pending;
    // Method changes preserve each model but require a fresh explicit arm for the newly selected gesture.
    el['method-controls'].hidden = !clickMode;
    el['eeg-method'].value = eegMethodKnown() ? eegMethod() : '';
    el['eeg-method'].disabled = !ready || !clickMode || busy || !!trial || stopping || status?.cleanup_pending;
    // Name the exact output that the next explicit arm would enable.
    put('eeg-arm-label', clickMode ? target === 'mouse' ? 'Arm hand controls + EEG clicks' : 'Arm hand controls + EEG practice' : 'Arm EEG swipes');
    // Counters come from the backend; a predicted select is not counted as an action.
    el['selection-counts'].hidden = !clickMode;
    put('selection-counts', ready ? `EEG practice: ${count(control.selection_count ?? 0)} · EEG clicks: ${count(control.click_count ?? 0)} · Finger clicks: ${count(control.finger_click_count ?? 0)}` : 'Selection counters unconfirmed.');
    // Finger gestures remain active; the target selector changes only the EEG selection destination.
    put('mode-note', blinkMode
      ? 'Move your hand to aim; click with your fingers or one deliberate double blink. Pinch and release to click; hold a pinch to drag. Optional index bend clicks keep your Hand controls setting. This learns the blink waveform recorded by the EEG headset locally; it is a blink gesture, not thought reading. A single blink is not a click command. Practice sends EEG selections to the counter, while finger clicks and drags still affect the desktop.'
      : clickMode
      ? 'Move your hand to aim; click with your fingers or the trained EEG pattern. After arming, point your index finger briefly to engage. Pinch and release to click; hold a pinch to drag. Index bend clicks follow your existing optional Hand controls setting. Practice sends EEG selections to the counter, while finger clicks and drags still affect the desktop. For EEG select, gently close your eyes; keep them open for rest. This detects a trained signal pattern, not thought reading.'
      : 'The camera labels and checks guided trials; the EEG classifier uses EEG-only features and inference. Prediction alone does not control the desktop. Alignment uses approximate host timestamps, not native sensor timestamps.');
    // The simple workflow presents only the two states used by its decoder.
    put('calibration-heading', blinkMode ? 'Double-blink EEG calibration (separate model)' : clickMode ? 'Two-state EEG calibration' : 'Three-class EEG calibration');
    // Counting four seconds remains usable when the user cannot see the open-eyes cue.
    put('calibration-instructions', blinkMode
      ? '1. Collect 8 No command and 8 Double blink training trials. Use Collect remaining training or the manual buttons. No command: read normally; natural single blinks are allowed. Double blink: blink twice gently when prompted, then keep your eyes open. Each trial lasts about 5 seconds. 2. Click Train & freeze model. 3. Choose Collect remaining validation for 8 new trials of each. Cancel queued trials keeps accepted data. Your eyes-closed model is preserved.'
      : clickMode
      ? '1. Collect 8 training trials each for rest and select, alternating them. Each trial lasts 5 seconds. Rest: keep your eyes open and relax. Select: gently close your eyes, count four seconds, then reopen. Keep your head and jaw still. 2. Train & freeze model. Collect 8 new validation trials each for rest and select without retraining. Keep the camera running; pointing is not needed during calibration. Hand movement is disarmed during trials.'
      : '1. In Webcam direction mode, collect at least 8 training trials each for left, right, and rest. Then train and freeze the model. 2. Collect 8 new validation trials per class without retraining. Each trial is 3 seconds: hold still for the first half-second, move once during the middle 2 seconds for left/right, then hold still for the final half-second. For rest, stay still throughout. The camera checks labels only.');
    // Actual clicks follow the return to rest, so the user can reopen their eyes first.
    put('session-note', blinkMode
      ? 'Choose Mouse left click and arm hand controls + EEG clicks to use the Windows cursor. Move your hand to aim, then make one deliberate double blink for one left click. Keep your eyes open afterwards; there is no long eyes-closed hold or rest/select/rest sequence. Single blinks are ignored as click commands. Finger clicks and drags remain active and take priority over EEG. Tracking resumes automatically after brief hand loss. Double-blink EEG clicks require this separate model to pass validation. Switching method or target disarms and preserves both models; Escape stops both controls.'
      : clickMode
      ? 'Choose Mouse left click and arm hand controls + EEG clicks to use the Windows cursor. Point briefly, then move your hand to aim; pinch/release clicks and pinch-hold drags remain active. Tracking resumes automatically after a brief hand loss. EEG uncertainty pauses only EEG clicks. For EEG, use eyes-open rest, sustained select, then eyes-open rest again; a confirmed selection holds the target until the click. Finger gestures cancel a pending EEG selection. Use saved EEG model restores your saved training and validation without retraining. Changing target disarms; Escape stops both controls.'
      : 'Keep acquisition running after training and validation: switch from Webcam direction to EEG classifier without stopping or restarting. Restarting acquisition requires resetting and repeating calibration. Switching control source disarms both modes.');
    el['eeg-arm'].checked = !!eegArmed;
    // The hand arm is independent of calibration; actual EEG clicks are gated separately in the backend.
    el['eeg-arm'].disabled = !ready || busy || !running || stopping || !eegMode() || !targetKnown || !eegMethodKnown() || !!trial
      || !!batch
      || (!eegArmed && (!control.can_arm || !armModelEligible() || !eegControlFresh()));
    // Loading is a separate explicit action, never an implicit reset, train, or arm.
    el['eeg-load-saved'].hidden = !clickMode;
    el['eeg-load-saved'].disabled = !ready || !clickMode || busy || !running || stopping || !!trial || status?.cleanup_pending || !eegFresh();
    el['eeg-save'].hidden = !clickMode;
    el['eeg-save'].disabled = !ready || !clickMode || busy || stopping || !!trial || decoder.trained !== true;
    // Every model action routes to the selected method; naming it prevents accidental resets of the other calibration.
    put('eeg-load-saved', blinkMode ? 'Use saved double-blink model' : 'Use saved EEG model');
    put('eeg-save', blinkMode ? 'Save double-blink model & trials' : 'Save EEG model & trials');
    put('eeg-reset', blinkMode ? 'Reset double-blink model & trials' : 'Reset EEG model & trials');
    put('eeg-saved-status', !clickMode ? '' : control.saved_model_file
      ? `Model, training, and validation saved locally: ${text(control.saved_model_file)}.` : control.saved_model_source
      ? `Saved model loaded: ${text(control.saved_model_source)}. Training and historical validation are retained.`
      : blinkMode ? 'Use saved double-blink model restores only a blink backup; passing validation is still required for EEG clicks. Your eyes-closed model is retained.'
      : 'Use saved EEG model restores the latest local backup without repeating training. Loading leaves controls disarmed.');
    const hasDecoder = ready && !!status.eeg_control?.decoder;
    // Frozen training data is never extended by a validation click.
    el['eeg-phase'].value = decoder.trained ? 'validate' : 'train';
    el['eeg-phase'].disabled = true;
    // Reset every column and button on mode changes so no stale direction control survives.
    for (const label of ['left', 'right', 'rest', 'select']) {
      const activeLabel = labels.includes(label);
      el[`column-${label}`].hidden = !activeLabel;
      el[`train-${label}`].hidden = el[`validate-${label}`].hidden = !activeLabel;
      el[`eeg-${label}`].hidden = !activeLabel;
      put(`train-${label}`, `${count(trainCounts[label] ?? 0)} / ${quota}`);
      put(`validate-${label}`, `${count(validationCounts[label] ?? 0)} / ${quota}`);
      // Binary labels come from EEG cues: the camera stays running, but no pointing pose labels a trial.
      const trialReady = clickMode ? fresh() : mode === 'webcam' && fresh() && status.camera?.tracked === true;
      // Extra command examples are available only after both base quotas are complete, preserving room for no-command training.
      const additionalBlinkTraining = blinkMode && label === 'select' && decoder.trained === false && trainingComplete;
      // The extra-example exception never applies to held-out validation, eyes-closed training, or the no-command class.
      const atTrialLimit = blinkTrainingFull || (!additionalBlinkTraining
        && (decoder.trained ? validationCounts[label] : trainCounts[label]) >= quota);
      el[`eeg-${label}`].disabled = !activeLabel || !hasDecoder || busy || !running || stopping || !!trial
        || !!batch
        || !trialReady
        || atTrialLimit;
    }
    // Human-facing labels clarify blink training without changing the backend's fixed rest/select vocabulary.
    put('column-rest', blinkMode ? 'No command' : 'Rest');
    put('column-select', blinkMode ? 'Double blink' : 'Select');
    put('eeg-rest', blinkMode ? 'Collect no-command trial' : 'Collect rest trial');
    put('eeg-select', blinkMode ? decoder.trained === false && trainCounts.select >= quota
      ? 'Collect another double-blink trial' : 'Collect double-blink trial' : 'Collect select trial');
    // Guided buttons fill only the currently selected phase; training and validation remain separate deliberate actions.
    el['eeg-guided-controls'].hidden = !blinkMode;
    el['eeg-guided-cue'].hidden = !blinkMode;
    const batchPhase = decoder.trained ? 'validate' : 'train';
    const phaseComplete = decoder.trained ? validationComplete : trainingComplete;
    put('eeg-batch', decoder.trained ? 'Collect remaining validation' : 'Collect remaining training');
    el['eeg-batch'].disabled = !blinkMode || !ready || !visible() || busy || !running || stopping || !!trial || !!batch
      || !fresh() || control.armed === true || !hasDecoder || phaseComplete || blinkTrainingFull
      || typeof status.eeg?.session_id !== 'string' || !status.eeg.session_id;
    el['eeg-batch-cancel'].disabled = !batch;
    const activeCounts = decoder.trained ? validationCounts : trainCounts;
    put('eeg-batch-status', batch ? `Guided ${batch.phase === 'train' ? 'training' : 'validation'} · No command ${count(activeCounts.rest ?? 0)} / ${quota} · Double blink ${count(activeCounts.select ?? 0)} / ${quota}. Cancel stops future cues; the current trial can finish.`
      : batchNotice || (control.armed === true ? 'Disarm hand controls before guided collection. Existing trials will be kept.'
        : `Collect the remaining ${batchPhase === 'train' ? 'training' : 'validation'} trials with one click. Follow each cue; there is a 3-second preparation interval. Existing trials are kept. Manual collection is also available.`));
    const preparing = batch?.stage === 'preparing';
    put('eeg-guided-cue', trial ? trial.remaining_seconds <= 0
      ? 'Finishing signal collection… keep your eyes open and wait for acceptance.'
      : `${trialName(trial.label)} · ${Math.ceil(trial.remaining_seconds)} s — ${text(trial.cue, trial.label === 'rest' ? 'Read normally; natural single blinks are allowed.' : 'Blink twice gently when prompted, then keep your eyes open.')}`
      : preparing ? `${trialName(batch.label)} starts in ${Math.max(1, Math.ceil((batch.deadline - now()) / 1000))}… Keep your eyes open and relax.`
      : batch?.stage === 'requesting' ? `Starting ${trialName(batch.label)}… wait for the cue.`
      : 'Follow the cue here. Blink gently twice for Double blink; read normally for No command.');
    el['eeg-train'].disabled = !hasDecoder || busy || stopping || !!trial || !running
      || !eegFresh() || decoder.trained === true || !trainingComplete || decoder.can_train === false;
    el['eeg-reset'].disabled = !hasDecoder || busy || stopping || !!trial;
    put('control-status', !ready ? 'Control state unconfirmed. Arming is disabled.'
      : stopping ? 'Stop requested. Both arming controls are cleared while the backend confirms.'
      // Practice affects EEG only; the explicit shared arm also enables ordinary finger clicks and drags.
      : eegArmed && blinkMode ? target === 'mouse'
        ? 'Hand controls + EEG clicks armed: move your hand to aim. One deliberate double blink produces one left click after recognition. Single blinks are not click commands. Pinch/release clicks and pinch-hold drags remain active; optional bend clicks follow your setting. Finger gestures take priority over EEG.'
        : 'Hand controls + EEG practice armed: one deliberate double blink increments the EEG counter after recognition. Finger clicks and drags still affect the desktop. Your hand controls remain active while the EEG add-on waits for validation or a usable signal.'
      : eegArmed && clickMode ? target === 'mouse'
        ? 'Hand controls + EEG clicks armed: point briefly, then move your hand to aim. Pinch/release clicks and pinch-hold drags remain active; optional bend clicks follow your setting. EEG rest, select, then rest produces one click. Finger gestures cancel a pending EEG selection.'
        : 'Hand controls + EEG practice armed: point briefly, then move your hand to aim. EEG rest, select, then rest increments the counter. Finger clicks and drags still affect the desktop; optional bend clicks follow your setting.'
      : eegArmed ? 'EEG swipes armed: EEG inference chooses direction. The webcam does not choose EEG actions.'
      : armed ? 'Webcam swipes armed: hand motion chooses direction. EEG predictions do not choose actions.'
      : `Prediction only. Both desktop controls are disarmed. Selected source: ${clickMode ? 'EEG selection / click' : mode === 'eeg' ? 'EEG classifier' : 'webcam direction'}.`);
    if (eegArmed && clickMode) {
      // The shared arm enables the hand immediately; this line distinguishes the independently gated EEG add-on.
      put('control-status', `${el['control-status'].textContent} ${control.can_eeg_click === true && (!blinkMode || eligible)
        ? 'EEG click add-on is ready.' : `EEG click add-on is paused. ${text(control.eeg_click_reason, text(control.reason, blinkMode ? 'Waiting for a validated double-blink model and fresh usable EEG.' : 'Waiting for a usable model and fresh rest/select/rest signal.'))}`}`);
    }
    const prediction = control.prediction;
    const predicted = prediction && prediction.valid === true && eegFresh()
      && Number.isFinite(prediction.confidence);
    const predictionLabel = labels.includes(prediction?.label) ? blinkMode
      ? prediction.label === 'select' ? 'double blink' : 'no command' : prediction.label : 'uncertain';
    put('prediction', predicted
      ? `EEG ${blinkMode ? 'blink detection' : 'prediction'}: ${predictionLabel} · uncalibrated score ${number(prediction.confidence, 3)}${Number.isFinite(prediction.margin) ? ` · margin ${number(prediction.margin, 3)}` : ''}${eegArmed ? ' · EEG control armed' : ' · display only'}${prediction.ood ? ' · outside training range' : ''}`
      : `EEG prediction unavailable. ${!eegFresh() ? 'Waiting for fresh EEG.' : text(prediction?.reason, decoder.trained ? 'Waiting for a complete EEG window.' : 'Collect training trials and train a model first.')}`);
    // Prefer the backend's timed cue; the fallback keeps old snapshots understandable.
    // Blink cues ask for a brief pair near the trial center; they never inherit the eyes-closed four-second hold.
    const blinkCue = text(trial?.cue, trial?.label === 'rest' ? 'read normally; natural single blinks are allowed; do not make a deliberate double blink'
      : trial?.remaining_seconds > 3.5 ? 'keep your eyes open; prepare to blink twice near the middle of the trial'
      : trial?.remaining_seconds <= 1.5 ? 'keep your eyes open; do not repeat the double blink'
      : 'blink twice deliberately now, then keep your eyes open');
    const clickCue = text(trial?.cue, trial?.label === 'rest' ? 'keep your eyes open and relax without moving'
      : trial?.remaining_seconds <= 1 ? 'open your eyes and rest'
      : 'gently close your eyes, count four seconds from the start, then reopen; keep your head and jaw still');
    const trialCue = blinkMode ? blinkCue : clickMode ? clickCue : trial?.label === 'rest' ? 'keep your hand still for rest'
      : trial?.remaining_seconds > 2.5 ? `hold still for the first half-second; prepare to move ${text(trial?.label)}`
      : trial?.remaining_seconds <= 0.5 ? 'hold still for the final half-second'
      : `move your hand ${text(trial?.label)} once now`;
    put('eeg-trial', trial
      ? `${trial.phase === 'validate' ? 'Validation' : 'Training'} trial: ${trialCue}. ${number(trial.remaining_seconds, 1)} s remaining. Both modes are disarmed.`
      : text(control.last_result, decoder.trained
        ? 'Model frozen. Collect new validation trials; these do not retrain the model.'
        // No movement directions are requested from the binary selection decoder.
        : blinkMode ? 'Choose No command or Double blink to collect a 5-second training trial. Alternate the two states.'
        : clickMode ? 'Choose rest or select to collect a 5-second training trial. Alternate the two states.'
        : 'Choose left, right, or rest to collect a training trial.'));
    // Version 3 may retain an unusual command trial without using it to learn the waveform; report the backend's actual fit.
    const fit = decoder.training_fit;
    // Require coherent integer counts from a frozen v3 model; missing/older metadata uses the existing generic status instead.
    const fitKnown = blinkMode && decoder.trained === true && decoder.detector_version === 3
      && Number.isInteger(fit?.supported_select_trials) && Number.isInteger(fit?.total_select_trials)
      && Number.isInteger(fit?.excluded_select_trials) && fit.supported_select_trials > 0
      && fit.total_select_trials === trainCounts.select && fit.supported_select_trials <= fit.total_select_trials
      && fit.excluded_select_trials === fit.total_select_trials - fit.supported_select_trials;
    // Fitting evidence is distinct from held-out performance: these counts never unlock arming or satisfy validation.
    const fitSummary = fitKnown ? `Model learned from ${fit.supported_select_trials} of ${fit.total_select_trials} double-blink trials.`
      + (fit.excluded_select_trials ? ` ${fit.excluded_select_trials} unusual trial${fit.excluded_select_trials === 1 ? '' : 's'} kept but not used.` : '')
      + (validationComplete ? '' : ' Collect validation next.') : '';
    put('eeg-model-status', !hasDecoder ? 'EEG classifier status unavailable.'
      : decoder.trained ? fitSummary || `Frozen model · ${text(decoder.state, 'collecting_validation')}. Training data is separate from new validation trials.`
      : blinkMode && blinkTrainingTotal >= BLINK_TRAINING_LIMIT && !trainingComplete
        ? 'Training collection is full (80 trials), but the required examples for both classes are incomplete.'
      : trainingComplete ? blinkMode
        ? 'Training quota reached. Select Train & freeze model before collecting validation.'
          + (blinkTrainingTotal < BLINK_TRAINING_LIMIT ? ' You can add another double-blink example if needed.' : ' Training collection is full (80 trials).')
        : 'Training quota reached. Select Train & freeze model before collecting validation.'
      : 'Collect at least 8 accepted training trials per class. Rejected trials do not increase counts.');
    const trainingError = text(decoder.training_error || decoder.train_error);
    const validationError = text(decoder.validation_error);
    put('eeg-errors', [eegActionError, trainingError && `Training: ${trainingError}`, validationError && `Validation: ${validationError}`].filter(Boolean).join('\n'));
    el['eeg-errors'].hidden = !el['eeg-errors'].textContent;
    const evaluation = decoder.evaluation || {};
    // Evaluation rows follow this decoder's vocabulary, including binary confusion matrices.
    renderEegEvaluation(evaluation, validationComplete, labels, decoder.validation_scope === 'saved_acquisition');
    const gateReason = text(control.reason, text(evaluation.reason, 'Collect and validate the frozen EEG model before arming.'));
    const thresholds = evaluation.thresholds || {};
    const gateNames = { enough_trials: '8 validation trials per class', balanced_accuracy: `Balanced accuracy ≥ ${number((thresholds.balanced_accuracy ?? 0.75) * 100, 0)}%`,
      each_recall: `Each recall ≥ ${number((thresholds.each_recall ?? 0.70) * 100, 0)}%`, each_precision: `Each precision ≥ ${number((thresholds.each_precision ?? 0.70) * 100, 0)}%`,
      rest_false_activation_rate: `${blinkMode ? 'No-command' : 'Rest'} false activations ≤ ${number((thresholds.rest_false_activation_rate ?? 0.10) * 100, 0)}%` };
    const gateDetails = Object.entries(gateNames).map(([key, name]) => `${name}: ${evaluation.gates?.[key] === true ? 'passed' : evaluation.gates?.[key] === false ? 'not passed' : 'pending'}`).join(' · ');
    // Keep the readiness instruction consistent with the currently selected output.
    const gateHeading = eligible ? 'Validation gates passed' : clickMode && !blinkMode && control.experimental_model === true
      ? 'Experimental saved model: validation did not pass' : clickMode ? 'EEG clicks waiting; hand controls remain available' : 'EEG arming locked';
    put('eeg-gates', `${gateHeading}: ${gateReason}\n${gateDetails}\nEngineering gates are not a safety guarantee. EEG may classify motion or muscle artifacts, not intention.${control.neutral_ready === false ? blinkMode ? ' Waiting for the blink detector to become ready for a new pair.' : clickMode ? ' Eyes-open rest must be recognized before a new selection.' : ' Rest must be recognized before a new EEG swipe.' : ''}`);
    put('validation-note', blinkMode
      ? 'Double-blink EEG clicks require a frozen model and passing separate validation, including no-command trials. Loading a saved blink model keeps this requirement; failed validation does not enable EEG clicks. The detector learns your local blink waveform and requires a deliberate pair rather than a single blink. Hand movement and finger clicks remain available independently. Your eyes-closed model is preserved. Escape stops both controls from any app.'
      : clickMode
      ? 'Hand movement and finger clicks work independently of EEG readiness. A saved model keeps its original validation result; loading it explicitly enables experimental EEG use even when that result failed. EEG clicks still require fresh usable signals and a complete rest/select/rest sequence. Escape stops both controls from any app.'
      : 'Validation uses new trials against the frozen model. Scores are uncalibrated model scores, not probabilities or intention measurements. EEG swipes require passing validation in the current acquisition session and a separate explicit arm. Escape stops and disarms from any app.');
  }

  // The caller supplies the active, allowlisted labels instead of assuming three classes.
  function renderEegEvaluation(evaluation, complete, labels, historical = false) {
    const percent = value => Number.isFinite(value) ? `${number(value * 100, 1)}%` : 'not available';
    // Display names change with the gesture, while matrix indexing stays tied to the server's rest/select keys.
    const blinkMode = doubleBlinkMode();
    const displayLabel = label => blinkMode ? ({ rest: 'no command', select: 'double blink' }[label] || label) : label;
    if (!complete) {
      put('eeg-evaluation', `Validation is incomplete. Collect at least 8 new trials per class against the frozen model. ${controlMode() === 'eeg_click' ? 'Hand controls remain available.' : 'Arming remains locked.'}`);
      return;
    }
    const lines = [`Frozen-model validation${historical ? ' (historical saved acquisition)' : ''}: accuracy ${percent(evaluation.accuracy)} · balanced accuracy ${percent(evaluation.balanced_accuracy)}.`];
    for (const label of labels) {
      const metrics = evaluation.per_class?.[label] || {};
      lines.push(`${displayLabel(label)}: precision ${percent(metrics.precision)} · recall ${percent(metrics.recall)}.`);
    }
    const matrix = evaluation.confusion_matrix || evaluation.confusion;
    if (matrix) {
      // Display the exact order used to read array-based confusion matrices below.
      lines.push(`Confusion rows = actual; columns = predicted ${[...labels, 'uncertain'].map(displayLabel).join(' / ')}:`);
      labels.forEach((label, index) => {
        const row = Array.isArray(matrix) ? matrix[index] : matrix[label];
        if (row) lines.push(`${displayLabel(label)}: ${[...labels, 'uncertain'].map((column, columnIndex) => count(Array.isArray(row) ? row[columnIndex] : row[column])).join(' / ')}`);
      });
    }
    const restFalse = evaluation.rest_false_activations;
    lines.push(`${blinkMode ? 'No-command' : 'Rest'} false activations: ${count(restFalse)} / ${count(evaluation.rest_trials)} held-out ${blinkMode ? 'no-command trials' : 'rest trials before debounce'}${Number.isFinite(evaluation.rest_false_activation_rate) ? ` (${percent(evaluation.rest_false_activation_rate)})` : ''}.`);
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
    const previousStatus = status;
    const previousMethod = eegMethod();  // A new method must never retain a lease obtained for the old gesture.
    status = { ...data, camera, eeg };
    // Once collection has stopped, discard guidance from an old phase/source/reset; active queues publish their own cancellation reason below.
    if (!batch && previousStatus && calibrationChanged(previousStatus, status)) batchNotice = '';
    // Freshness belongs to this snapshot's arrival, not the preceding response's age.
    receivedAt = now(); known = true;
    // Tracking pauses keep Escape active; a confirmed backend disarm or stopped run releases it.
    if (guardToken && (previousMethod !== eegMethod() || !eegMethodKnown()
      || (!data.eeg_control?.armed || !armedInputsAvailable()) && pending !== 'eeg_arm')) releaseGuard();
    wasActive = wasActive || data.running || data.armed || data.eeg_control?.armed === true;
    if (!data.running && !data.armed && !data.eeg_control?.armed && !data.cleanup_pending) {
      stopRequested = false;
      wasActive = false;
    }
    error = '';
    advanceBatchFromStatus();  // Count changes and pending state, not a local five-second timeout, determine acceptance.
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
    if (pending === 'eeg_arm' || pending === 'arm') {
      ++actionId;
      if (pending === 'eeg_arm') releaseGuard(undefined, true);
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
    const button = { start: el.start, stop: el.stop, contact: el.contact, arm: el.arm,
      calibrate: payload.label === 'left' ? el.left : el.right, reset_calibration: el.reset,
      eeg_trial: el[`eeg-${payload.label}`], eeg_train: el['eeg-train'], eeg_reset: el['eeg-reset'],
      eeg_load_saved: el['eeg-load-saved'],
      eeg_save: el['eeg-save'],
      // Target selection uses its own guarded input and cannot masquerade as arming.
      eeg_arm: el['eeg-arm'], eeg_target: el['eeg-target'], eeg_method: el['eeg-method'], control_mode: el['control-mode'] }[action];
    const guidedTrial = guidedJob && guidedJob === batch && action === 'eeg_trial';
    if (!button || (button.disabled && !guidedTrial) || !isConnected() || (pending && action !== 'stop') || pending === 'stop') { render(); return; }
    if (batch && !guidedTrial) stopBatch('Guided collection stopped by another control. Existing data is retained.');
    const id = ++actionId, currentEpoch = epoch;
    ++generation; pending = action; error = '';
    const preparingGuard = action === 'eeg_arm' && payload.enabled === true;
    // Bind asynchronous arming to the output the user actually selected at click time.
    const requestedEegMode = controlMode(), requestedEegTarget = status?.eeg_control?.target, requestedEegMethod = eegMethod();
    let preparedToken = null;
    // A target change revokes the old global shortcut lease before changing output.
    if (['stop','control_mode','eeg_target','eeg_method','eeg_reset','eeg_train','eeg_trial','eeg_load_saved','calibrate','reset_calibration'].includes(action)
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
        // Both EEG modes must still be valid when asynchronous guard preparation finishes.
        if (!known || !status.running || !eegMode() || !eegControlFresh() || !eegTargetKnown() || !eegMethodKnown()
          // A new status cannot silently turn a pending practice arm into real mouse output.
          || controlMode() !== requestedEegMode || (requestedEegMode === 'eeg_click'
            && (status.eeg_control?.target !== requestedEegTarget || eegMethod() !== requestedEegMethod))
          || !status.eeg_control?.can_arm || !armModelEligible() || stopRequested) {
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
  // Read the requested target before rendering restores the last acknowledged selection.
  el['eeg-target'].addEventListener('change', () => { const target = el['eeg-target'].value; render(); run('eeg_target', { target }); });
  // Preserve the last acknowledged method until the backend has disarmed and selected the requested model.
  el['eeg-method'].addEventListener('change', () => { const method = el['eeg-method'].value; render(); run('eeg_method', { method }); });
  // Hidden labels remain disabled; only the active mode's trial buttons can send requests.
  for (const label of ['left', 'right', 'rest', 'select']) el[`eeg-${label}`].addEventListener('click', () =>
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
