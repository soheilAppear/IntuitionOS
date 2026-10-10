// Synthetic samples and mock DOM: never opens hardware, Electron, or a socket.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createMultimodalPanel } = require('../ui/renderer/multimodal-panel.cjs');

function harness({ guardMode = 'ready' } = {}) {
  const elements = new Map(), timers = [], requests = [], images = [], drawing = [];
  class Element {
    constructor(id) { this.id = id; this.textContent = ''; this.open = false; this.listeners = {}; this.attributes = {}; this.width = 480; this.height = 360; }
    addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
    dispatch(name, args = {}) { for (const fn of this.listeners[name] || []) fn({ preventDefault() {}, stopImmediatePropagation() {}, stopPropagation() {}, ...args }); }
    setAttribute(key, value) { this.attributes[key] = value; }
    getContext() { return new Proxy({}, { get: (_target, name) => (...args) => drawing.push([this.id, name, ...args]), set: () => true }); }
  }
  const document = new Element('document');
  document.getElementById = id => { if (!elements.has(id)) elements.set(id, new Element(id)); return elements.get(id); };
  let online = true, handCamera = false, brainbit = true, time = 0;
  const guards = [], guardStops = [];
  const panel = createMultimodalPanel({ document, isConnected: () => online,
    isCameraRunning: () => handCamera, isBrainbitConnected: () => brainbit,
    request: (action, payload) => new Promise((resolve, reject) => requests.push({ action, payload, resolve, reject })),
    startEegGuard: () => {
      if (guardMode === 'pending') return new Promise(resolve => guards.push({ resolve }));
      guards.push({});
      return Promise.resolve(guardMode === 'ready'
        ? { ok: true, token: '11111111-2222-4333-8444-555555555555' }
        : { ok: false, error: 'Escape is already registered by another app.' });
    },
    stopEegGuard: token => { guardStops.push(token); return Promise.resolve({ ok: true }); },
    imageFactory: () => { const picture = {}; images.push(picture); return picture; },
    schedule: (fn, delay) => { timers.push({ fn, delay }); return timers.length; },
    cancel: id => { if (timers[id - 1]) timers[id - 1].cancelled = true; }, now: () => time,
  });
  panel.connectionChanged();
  const el = id => document.getElementById(`multimodal-${id}`);
  const open = value => { el('panel').open = value; el('panel').dispatch('toggle'); };
  return { panel, document, requests, images, drawing, timers, el, open, guards, guardStops,
    online(value) { online = value; panel.connectionChanged(); },
    camera(value) { handCamera = value; panel.refreshControls(); },
    brainbit(value) { brainbit = value; panel.refreshControls(); },
    tick(delay) { const timer = timers.find(t => !t.cancelled && !t.ran && (delay === undefined || t.delay === delay));
      assert.ok(timer, `Expected ${delay}ms timer`); timer.ran = true; time += timer.delay; timer.fn(); },
    advance(ms) { time += ms; },
  };
}

const stopped = (extra = {}) => ({ state: 'stopped', running: false, armed: false, can_arm: false,
  reason: 'Preview stopped.', camera: {}, eeg: {}, motion: {}, calibration: { counts: { left: 0, right: 0 } }, ...extra });
const running = (extra = {}) => stopped({ state: 'running', running: true, can_arm: true, reason: 'Preview running.',
  camera: { running: true, tracked: true, image: 'data:image/jpeg;base64,YWJj', age_ms: 0, fps: 30, width: 480, height: 360, sequence: 1 },
  eeg: { state: 'running', mode: 'signal', units: 'V', nominal_hz: 250, channels: [{ name: 'T3' }, { name: 'T4' }],
    stats: { age_seconds: 0, received_rate_hz: 250, gaps: 0, duplicates: 0, nonfinite: 0, channel_mismatches: 0, queue_drops: 0 },
    waveform: Array.from({ length: 10 }, (_, i) => ({ samples: [i * 0.000001, -i * 0.000001] })) }, ...extra });
const flush = () => new Promise(resolve => setImmediate(resolve));
async function respond(request, data, ok = true) { request.resolve({ ok, status: ok ? 200 : 503, data }); await flush(); }

test('real markup starts collapsed and states local experimental scope', () => {
  const html = fs.readFileSync(path.join(__dirname, '../ui/renderer/index.html'), 'utf8');
  const markup = html.match(/<details class="multimodal-panel"[\s\S]*?<\/details>/)[0];
  assert.doesNotMatch(markup.split('>')[0], /\bopen\b/);
  assert.match(markup, /No recordings saved; stays on this desktop/);
  assert.match(markup, /approximate host timestamps/);
  assert.match(markup, /EEG-only features and inference/);
  assert.match(markup, /not thought reading/);
  for (const id of ['start','stop','contact','arm','camera','eeg','left','right','reset']) assert.match(markup, new RegExp(`id="multimodal-${id}"`));
});

test('opening fetches status only and never starts sensors; hidden panels stop raw pulls', async () => {
  const r = harness(); assert.equal(r.requests.length, 0); assert.equal(r.el('arm').checked, false);
  r.open(true); assert.equal(r.requests[0].action, 'status');
  await respond(r.requests[0], stopped()); r.tick(1000);
  assert.equal(r.requests[1].action, 'status');
  await respond(r.requests[1], running()); r.tick(125);
  assert.equal(r.requests[2].action, 'preview');
  r.open(false); await respond(r.requests[2], running({ armed: true }));
  assert.equal(r.images.length, 0, 'late frame from hidden panel is discarded');
  assert.ok(r.requests.every(q => ['status', 'preview'].includes(q.action)));
});

test('start requires connected BrainBit and stopped normal hand camera', async () => {
  const r = harness(); r.panel.acceptStatus(stopped());
  r.camera(true); r.el('start').dispatch('click'); assert.equal(r.requests.length, 0);
  assert.match(r.el('start-hint').textContent, /Stop the normal hand camera/);
  r.camera(false); r.brainbit(false); assert.equal(r.el('start').disabled, true);
  r.brainbit(true); r.el('start').dispatch('click'); r.el('start').dispatch('click');
  assert.equal(r.requests.length, 1); assert.equal(r.requests[0].action, 'start');
  await respond(r.requests[0], running()); assert.equal(r.el('arm').checked, false);
});

test('arming is explicit and stop supersedes pending start or arm responses', async () => {
  const r = harness(); r.panel.acceptStatus(running());
  r.el('arm').checked = true; r.el('arm').dispatch('change');
  assert.equal(r.requests[0].action, 'arm'); assert.deepEqual(r.requests[0].payload, { enabled: true });
  assert.equal(r.el('arm').checked, false, 'wait for acknowledgment');
  r.el('stop').dispatch('click'); assert.equal(r.requests[1].action, 'stop');
  assert.equal(r.el('stop').disabled, false);
  await respond(r.requests[1], stopped()); await respond(r.requests[0], running({ armed: true }));
  assert.equal(r.el('arm').checked, false); assert.match(r.el('summary').textContent, /stopped/i);
});

test('failed reads hide frames and arm while stop remains available', async () => {
  const r = harness(); r.panel.acceptStatus(running({ armed: true })); r.open(true);
  r.requests[0].reject(new Error('network')); await flush();
  assert.equal(r.el('arm').checked, false); assert.equal(r.el('arm').disabled, true);
  assert.equal(r.el('stop').disabled, false); assert.match(r.el('error').textContent, /failed/);
  r.el('stop').dispatch('click'); assert.equal(r.requests[1].action, 'stop');
  await respond(r.requests[1], stopped()); assert.equal(r.el('arm').checked, false);
});

test('delayed arm response cannot restore arming over a newer dropout snapshot', async () => {
  const r = harness(); r.panel.acceptStatus(running({ revision: 1 }));
  r.el('arm').checked = true; r.el('arm').dispatch('change');
  r.open(true);
  await respond(r.requests[1], running({ revision: 3, armed: false, can_arm: false,
    reason: 'Packet gap detected; controls disarmed.' }));
  await respond(r.requests[0], running({ revision: 2, armed: true }));
  assert.equal(r.el('arm').checked, false);
  assert.equal(r.el('arm').disabled, true);
  assert.match(r.el('status').textContent, /Packet gap/);
  assert.equal(r.el('status').attributes['aria-busy'], 'false');
});

test('raw draws are bounded to250 samples, volts become µV and delayed camera decode is rejected after stop', async () => {
  const r = harness(); r.panel.acceptStatus(running()); r.open(true);
  const frame = running(); frame.eeg.waveform = Array.from({ length: 400 }, (_, i) => ({ samples: [i * 1e-6, null] }));
  await respond(r.requests[0], frame);
  assert.equal(r.images.length, 1);
  const labels = r.drawing.filter(row => row[0] === 'multimodal-eeg' && row[1] === 'fillText').map(row => row[2]);
  assert.ok(labels.some(label => label.includes('274.5 µV')), 'mean uses latest250 raw volts converted to microvolts');
  assert.ok(r.drawing.filter(row => row[0] === 'multimodal-eeg' && row[1] === 'lineTo').length <= 252);
  r.el('stop').dispatch('click'); const before = r.drawing.filter(row => row[1] === 'drawImage').length;
  r.images[0].onload(); assert.equal(r.drawing.filter(row => row[1] === 'drawImage').length, before);
});

test('stale frame deadline clears displays and disables arming without posting', async () => {
  const r = harness(); r.panel.acceptStatus(running({ armed: true })); r.open(true);
  await respond(r.requests[0], running({ armed: true }));
  r.tick(501);
  assert.equal(r.el('arm').checked, false); assert.equal(r.el('arm').disabled, true);
  assert.match(r.el('camera-status').textContent, /stale/);
  assert.ok(r.requests.every(q => ['status', 'preview'].includes(q.action)));
});

test('camera expiry retains fresh EEG then independently expires its status and waveform', async () => {
  const r = harness(); r.panel.acceptStatus(running()); r.open(true);
  await respond(r.requests[0], running());
  const eegClears = () => r.drawing.filter(row => row[0] === 'multimodal-eeg' && row[1] === 'clearRect').length;
  const before = eegClears();
  r.tick(501);
  assert.match(r.el('camera-status').textContent, /stale/);
  assert.match(r.el('eeg-status').textContent, /250 Hz received/);
  assert.equal(eegClears(), before, 'fresh EEG waveform survives camera expiry');
  r.tick(250);
  assert.match(r.el('eeg-status').textContent, /Waiting for fresh EEG/);
  assert.ok(eegClears() > before);
});

test('offline discards delayed replies and never replays a sensor or arm operation', async () => {
  const r = harness(); r.panel.acceptStatus(stopped()); r.el('start').dispatch('click');
  r.online(false); assert.equal(r.el('arm').checked, false); assert.equal(r.el('stop').disabled, true);
  r.online(true); await respond(r.requests[0], running({ armed: true }));
  assert.equal(r.el('arm').checked, false); assert.equal(r.requests.length, 1);
  assert.equal(r.el('stop').disabled, false);
});

test('contact precheck shows age and descriptive units without cutoffs', async () => {
  const r = harness(); r.panel.acceptStatus(stopped()); r.el('contact').dispatch('click');
  assert.equal(r.requests[0].action, 'contact');
  await respond(r.requests[0], stopped({ eeg: { contact_precheck: { age_seconds: 61, values: [12000, null],
    channels: [{ name: 'T3' }, { name: 'T4' }], units: 'ohm' } } }));
  assert.match(r.el('contact-status').textContent, /stale; historical/);
  assert.match(r.el('contact-status').textContent, /12.0 kΩ/);
  assert.match(r.el('contact-status').textContent, /T4 unavailable/);
  assert.match(r.el('contact-status').textContent, /no good\/bad cutoff/);
});

test('calibration guides movement and shows accuracy only with enough heldout evidence', async () => {
  const r = harness(); r.panel.acceptStatus(running({ armed: true }));
  r.el('left').dispatch('click'); assert.deepEqual(r.requests[0].payload, { label: 'left' });
  assert.equal(r.el('arm').checked, false);
  await respond(r.requests[0], running({ calibration: { counts: { left: 1, right: 0 }, pending: { label: 'left', remaining_seconds: 2.4 } } }));
  assert.match(r.el('trial').textContent, /Move left now/); assert.equal(r.el('right').disabled, true);
  r.panel.acceptStatus(running({ calibration: { counts: { left: 2, right: 2 }, evaluation: { state: 'evaluated', n_train: 2, n_test: 2, accuracy: 1, baseline_accuracy: 0.5 } } }));
  assert.doesNotMatch(r.el('calibration-status').textContent, /accuracy 100/);
  r.panel.acceptStatus(running({ calibration: { counts: { left: 10, right: 10 }, evaluation: { state: 'evaluated', n_train: 14, n_test: 6, accuracy: 0.67, balanced_accuracy: 0.65, baseline_accuracy: 0.5 } } }));
  assert.match(r.el('calibration-status').textContent, /accuracy 67%/);
  assert.match(r.el('calibration-status').textContent, /majority baseline 50%/);
  r.el('reset').dispatch('click'); assert.equal(r.requests[1].action, 'reset_calibration');
});

test('camera errors and empty waveform remain readable and never manufacture signal', async () => {
  const r = harness(); r.panel.acceptStatus(running()); r.open(true);
  const frame = running(); frame.camera.image = '<img onerror=bad>'; frame.eeg.waveform = [];
  await respond(r.requests[0], frame);
  assert.equal(r.images.length, 0); assert.match(r.el('camera-status').textContent, /could not be displayed/);
  assert.match(r.el('eeg-status').textContent, /No EEG samples/);
  r.panel.acceptStatus(stopped({ state: 'error', error: '<script>camera failed</script>' }));
  assert.equal(r.el('error').textContent, '<script>camera failed</script>'); assert.equal(r.el('stop').disabled, false);
});

const classifier = (extra = {}) => ({ state: 'collecting_training', trained: false, arm_eligible: false,
  passed_validation: false, can_train: false, min_per_class: 8,
  train_counts: { left: 0, right: 0, rest: 0 }, validation_counts: { left: 0, right: 0, rest: 0 },
  evaluation: { state: 'incomplete', passed: false }, ...extra });
const passedClassifier = (extra = {}) => classifier({ state: 'validated', trained: true,
  arm_eligible: true, passed_validation: true, train_counts: { left: 8, right: 8, rest: 8 },
  validation_counts: { left: 8, right: 8, rest: 8 }, evaluation: { passed: true, complete: true,
    accuracy: 1, balanced_accuracy: 1, baseline_accuracy: 1 / 3, rest_false_activations: 0,
    rest_trials: 8, rest_false_activation_rate: 0, abstentions: 0,
    per_class: Object.fromEntries(['left','right','rest'].map(label => [label, { precision: 1, recall: 1 }])),
    confusion_matrix: Object.fromEntries(['left','right','rest'].map(label => [label,
      Object.fromEntries(['left','right','rest','uncertain'].map(predicted => [predicted, predicted === label ? 8 : 0]))])),
    gates: { enough_trials: true, balanced_accuracy: true, each_recall: true, each_precision: true, rest_false_activation_rate: true } }, ...extra });
const eegState = (extra = {}) => running({ control_mode: 'eeg', camera: { running: false },
  eeg_control: { armed: false, can_arm: true, neutral_ready: false, decoder: passedClassifier(),
    prediction: { valid: true, label: 'rest', confidence: 0.93, margin: 0.50 }, ...extra } });

test('EEG training trials use explicit three-class labels and frozen validation phase', async () => {
  const r = harness();
  r.panel.acceptStatus(running({ control_mode: 'webcam', eeg_control: { decoder: classifier() } }));
  assert.equal(r.el('eeg-phase').value, 'train');
  r.el('eeg-rest').dispatch('click');
  assert.equal(r.requests[0].action, 'eeg_trial');
  assert.deepEqual(r.requests[0].payload, { label: 'rest', phase: 'train' });
  await respond(r.requests[0], running({ eeg_control: { decoder: classifier(),
    pending: { label: 'rest', phase: 'train', remaining_seconds: 2.5 } } }));
  assert.match(r.el('eeg-trial').textContent, /keep your hand still/);
  assert.equal(r.el('eeg-left').disabled, true);
  r.panel.acceptStatus(running({ eeg_control: { decoder: classifier({ trained: true,
    state: 'collecting_validation', train_counts: { left: 8, right: 8, rest: 8 } }) } }));
  assert.equal(r.el('eeg-phase').value, 'validate');
  r.el('eeg-right').dispatch('click');
  assert.deepEqual(r.requests[1].payload, { label: 'right', phase: 'validate' });
  assert.equal(r.el('eeg-train').disabled, true, 'frozen model cannot be retrained during validation');
});

test('train requires quotas and validation cannot be extended until results look favorable', async () => {
  const r = harness(); r.panel.acceptStatus(running({ eeg_control: { decoder: classifier() } }));
  r.el('eeg-train').dispatch('click'); assert.equal(r.requests.length, 0);
  r.panel.acceptStatus(running({ eeg_control: { decoder: classifier({ can_train: true,
    train_counts: { left: 8, right: 8, rest: 8 } }) } }));
  r.el('eeg-train').dispatch('click'); assert.equal(r.requests[0].action, 'eeg_train');
  await respond(r.requests[0], running({ eeg_control: { decoder: passedClassifier({
    state: 'validation_failed', arm_eligible: false, passed_validation: false,
    evaluation: { passed: false, reason: 'Insufficient held-out precision.' } }) } }));
  for (const label of ['left','right','rest']) assert.equal(r.el(`eeg-${label}`).disabled, true);
  r.el('eeg-reset').dispatch('click'); assert.equal(r.requests[1].action, 'eeg_reset');
});

test('insufficient or failed validation never enables EEG arming even when can_arm is inconsistent', () => {
  for (const decoder of [classifier(), passedClassifier({ validation_counts: { left: 8, right: 8, rest: 7 } }),
    passedClassifier({ passed_validation: false }), passedClassifier({ evaluation: { passed: false } })]) {
    const r = harness(); r.panel.acceptStatus(eegState({ decoder }));
    assert.equal(r.el('eeg-arm').disabled, true); r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
    assert.equal(r.guards.length, 0); assert.equal(r.requests.length, 0);
    assert.match(r.el('eeg-gates').textContent, /arming locked/);
  }
});

test('EEG arming needs a registered Escape guard and does not depend on camera at inference time', async () => {
  const r = harness(); r.panel.acceptStatus(eegState());
  assert.equal(r.el('eeg-arm').disabled, false); assert.equal(r.el('arm').disabled, true);
  assert.match(r.el('prediction').textContent, /uncalibrated score 0.930/);
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
  assert.equal(r.requests.length, 0, 'guard preparation precedes the arm POST'); await flush();
  assert.equal(r.guards.length, 1);
  assert.deepEqual(r.requests[0].payload, { enabled: true, guard_token: '11111111-2222-4333-8444-555555555555' });
  await respond(r.requests[0], eegState({ armed: true }));
  assert.equal(r.el('eeg-arm').checked, true); assert.equal(r.el('arm').checked, false);
  r.el('eeg-arm').checked = false; r.el('eeg-arm').dispatch('change');
  assert.deepEqual(r.requests[1].payload, { enabled: false });
  assert.ok(r.guardStops.length);
});

test('failed Escape registration blocks the EEG arm POST and reports the blocker', async () => {
  const r = harness({ guardMode: 'unavailable' }); r.panel.acceptStatus(eegState());
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  assert.equal(r.requests.length, 0); assert.equal(r.el('eeg-arm').checked, false);
  assert.match(r.el('eeg-errors').textContent, /Escape is already registered/);
});

test('mode changes disarm both sources and send no implicit arm operation', async () => {
  const r = harness(); r.panel.acceptStatus(running({ armed: true, control_mode: 'webcam', eeg_control: { decoder: passedClassifier() } }));
  r.el('control-mode').value = 'eeg'; r.el('control-mode').dispatch('change');
  assert.deepEqual(r.requests[0].payload, { mode: 'eeg' });
  assert.equal(r.el('arm').checked, false); assert.equal(r.el('eeg-arm').checked, false);
  await respond(r.requests[0], eegState());
  assert.equal(r.el('arm').disabled, true); assert.equal(r.el('eeg-arm').checked, false);
  assert.equal(r.requests.length, 1); assert.equal(r.guards.length, 0);
});

test('late prepared guard is scoped-cleaned after stop and cannot send an arm POST', async () => {
  const r = harness({ guardMode: 'pending' }); r.panel.acceptStatus(eegState());
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
  r.el('stop').dispatch('click'); assert.equal(r.requests[0].action, 'stop');
  await respond(r.requests[0], stopped());
  const token = '22222222-2222-4333-8444-555555555555'; r.guards[0].resolve({ ok: true, token }); await flush();
  assert.equal(r.requests.length, 1); assert.ok(r.guardStops.includes(token));
  assert.equal(r.el('eeg-arm').checked, false);
});

test('newer dropout status beats a late successful EEG arm response', async () => {
  const r = harness(); r.panel.acceptStatus({ ...eegState(), revision: 1 });
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  r.panel.acceptStatus({ ...eegState({ armed: false, can_arm: false, reason: 'EEG dropout' }), revision: 3 });
  await respond(r.requests[0], { ...eegState({ armed: true }), revision: 2 });
  assert.equal(r.el('eeg-arm').checked, false); assert.equal(r.el('eeg-arm').disabled, true);
  assert.ok(r.guardStops.length);
});

test('EEG-only freshness expiry clears prediction and arming without replay', async () => {
  const r = harness(); r.panel.acceptStatus(eegState({ armed: true })); r.open(true);
  await respond(r.requests[0], eegState({ armed: true })); r.tick(751);
  assert.equal(r.el('eeg-arm').checked, false); assert.equal(r.el('eeg-arm').disabled, true);
  assert.match(r.el('prediction').textContent, /Waiting for fresh EEG/);
  assert.ok(r.requests.every(q => ['status','preview'].includes(q.action)));
});

test('Escape stops through a pending operation; offline Escape clears local arms without replay', async () => {
  const r = harness({ guardMode: 'pending' }); r.panel.acceptStatus(eegState());
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
  let prevented = false;
  r.document.dispatch('keydown', { key: 'Escape', preventDefault() { prevented = true; } });
  assert.equal(prevented, true); assert.equal(r.requests[0].action, 'stop');
  assert.equal(r.el('eeg-arm').checked, false);
  r.online(false); r.document.dispatch('keydown', { key: 'Escape' });
  assert.equal(r.requests.length, 1); assert.match(r.el('error').textContent, /offline/);
  r.online(true); r.panel.acceptStatus(stopped());
  assert.equal(r.requests.length, 1);
  const idle = harness(); idle.document.hasFocus = () => false; idle.open(true);
  idle.document.dispatch('keydown', { key: 'Escape' });
  assert.equal(idle.requests.length, 1, 'local handler does not act when another app has focus');
});

test('main emergency-stop notification invalidates an in-flight arm response', async () => {
  const r = harness(); r.panel.acceptStatus({ ...eegState(), revision: 1 });
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  r.panel.emergencyStop('guard-lease-lost');
  await respond(r.requests[0], { ...eegState({ armed: true }), revision: 2 });
  assert.equal(r.el('eeg-arm').checked, false); assert.match(r.el('error').textContent, /Emergency stop requested/);
  assert.equal(r.requests.length, 1, 'main owns the emergency stop POST');
});

test('frozen validation shows per-class metrics, uncertain outcomes, and engineering limits', () => {
  const r = harness(); r.panel.acceptStatus(eegState());
  const output = r.el('eeg-evaluation').textContent;
  assert.match(output, /left: precision 100.0% · recall 100.0%/);
  assert.match(output, /predicted left \/ right \/ rest \/ uncertain/);
  assert.match(output, /Rest false activations: 0 \/ 8 held-out rest trials before debounce/);
  assert.match(r.el('eeg-gates').textContent, /Balanced accuracy ≥ 75%: passed/);
  assert.match(r.el('eeg-gates').textContent, /not a safety guarantee/);
});

test('brief uncertain EEG transition keeps the acknowledged arm and its Escape guard', async () => {
  const r = harness(); r.panel.acceptStatus(eegState());
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  await respond(r.requests[0], eegState({ armed: true }));
  const releases = r.guardStops.length;
  r.panel.acceptStatus(eegState({ armed: true, can_arm: false,
    prediction: { valid: true, label: null, confidence: 0.61, margin: 0.08, reason: 'Transition uncertain' } }));
  assert.equal(r.el('eeg-arm').checked, true);
  assert.equal(r.el('eeg-arm').disabled, false, 'user can always explicitly disarm');
  assert.match(r.el('prediction').textContent, /prediction: uncertain/);
  assert.equal(r.guardStops.length, releases);
  r.panel.acceptStatus(eegState({ armed: false, can_arm: false }));
  assert.ok(r.guardStops.length > releases, 'confirmed disarm releases the shortcut lease');
});

test('read failure cancels a pending guard before it can post EEG arm', async () => {
  const r = harness({ guardMode: 'pending' }); r.panel.acceptStatus({ ...eegState(), revision: 1 });
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); r.open(true);
  assert.equal(r.requests[0].action, 'preview');
  r.requests[0].reject(new Error('Cached preview unavailable')); await flush();
  assert.equal(r.el('eeg-arm').checked, false);
  assert.ok(r.guardStops.length, 'pending native guard is cancelled immediately');
  r.guards[0].resolve({ ok: true, token: '11111111-2222-4333-8444-555555555555' }); await flush();
  assert.equal(r.requests.length, 1, 'late guard must not issue EEG arm');
  assert.ok(r.guardStops.includes('11111111-2222-4333-8444-555555555555'));
});

test('stale revisioned arm error cannot replace a newer disarmed snapshot', async () => {
  const r = harness(); r.panel.acceptStatus({ ...eegState(), revision: 1 });
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  r.panel.acceptStatus({ ...eegState({ armed: false, can_arm: false, reason: 'Latest dropout status' }), revision: 3 });
  await respond(r.requests[0], { ...eegState(), revision: 2, error: 'Old arm request failed' }, false);
  assert.equal(r.el('eeg-arm').checked, false);
  assert.match(r.el('eeg-gates').textContent, /Latest dropout status/);
  assert.equal(r.el('error').hidden, true);
  assert.ok(r.guardStops.length);
});

test('guided trial cue keeps movement inside the central EEG window', () => {
  const r = harness();
  for (const [remaining_seconds, expected] of [[2.9, /hold still for the first/], [1.7, /move your hand left once now/], [0.3, /hold still for the final/]]) {
    r.panel.acceptStatus(running({ eeg_control: { decoder: classifier(), pending: { label: 'left', phase: 'train', remaining_seconds } } }));
    assert.match(r.el('eeg-trial').textContent, expected);
  }
});

test('read failure revokes an in-flight EEG arm acknowledgment after releasing its guard', async () => {
  const r = harness(); r.panel.acceptStatus({ ...eegState(), revision: 1 });
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  r.open(true); r.requests[1].reject(new Error('GET failure')); await flush();
  assert.ok(r.guardStops.length);
  await respond(r.requests[0], { ...eegState({ armed: true }), revision: 3 });
  assert.equal(r.el('eeg-arm').checked, false);
  assert.equal(r.el('eeg-arm').disabled, true);
  assert.match(r.el('summary').textContent, /unconfirmed/);
});

test('guard preparation rechecks newer dropout status and elapsed EEG freshness before posting arm', async () => {
  for (const dropout of [true, false]) {
    const r = harness({ guardMode: 'pending' }); r.panel.acceptStatus({ ...eegState(), revision: 1 });
    r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
    if (dropout) r.panel.acceptStatus({ ...eegState({ can_arm: false, reason: 'New dropout' }), revision: 3 });
    else r.advance(800);
    r.guards[0].resolve({ ok: true, token: '11111111-2222-4333-8444-555555555555' }); await flush();
    assert.equal(r.requests.length, 0);
    assert.equal(r.el('eeg-arm').checked, false);
    assert.ok(r.guardStops.includes('11111111-2222-4333-8444-555555555555'));
    assert.match(r.el('eeg-errors').textContent, /readiness changed/);
  }
});

test('older HTTP arm acknowledgment retains the same guard confirmed by newer armed status', async () => {
  const r = harness(); r.panel.acceptStatus({ ...eegState(), revision: 1 });
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  r.panel.acceptStatus({ ...eegState({ armed: true }), revision: 3 });
  await respond(r.requests[0], { ...eegState({ armed: true }), revision: 2 });
  assert.equal(r.el('eeg-arm').checked, true);
  assert.equal(r.guardStops.length, 0, 'the current confirmed armed lease remains registered');
});
