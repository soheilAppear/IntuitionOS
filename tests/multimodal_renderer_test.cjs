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
  assert.match(markup, /Raw camera and EEG streams are not recorded/);
  assert.match(markup, /learned waveform features and trial\/validation metadata and stay on this desktop/);
  assert.match(markup, /derived pulse shapes, not full EEG recordings/);
  assert.match(markup, /Move your hand to aim; click with your fingers or the trained EEG pattern/);
  assert.match(markup, /trained signal pattern/);
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

// Binary fixtures contain only rest/select counts; the camera remains active for hand aiming.
const clickClassifier = (extra = {}) => classifier({ labels: ['rest', 'select'],
  train_counts: { rest: 0, select: 0 }, validation_counts: { rest: 0, select: 0 }, ...extra });
// A completed binary validation must unlock only the two intended labels.
const passedClickClassifier = (extra = {}) => clickClassifier({ trained: true, state: 'validated',
  arm_eligible: true, passed_validation: true, train_counts: { rest: 8, select: 8 },
  validation_counts: { rest: 8, select: 8 }, evaluation: { passed: true, complete: true,
    accuracy: 1, balanced_accuracy: 1, rest_false_activations: 0, rest_trials: 8,
    rest_false_activation_rate: 0, abstentions: 0,
    per_class: { rest: { precision: 1, recall: 1 }, select: { precision: 1, recall: 1 } },
    confusion_matrix: { rest: { rest: 8, select: 0, uncertain: 0 }, select: { rest: 0, select: 8, uncertain: 0 } } }, ...extra });
// Practice remains the explicit initial target even when the classifier is already validated.
const clickState = (extra = {}) => running({ control_mode: 'eeg_click',
  eeg_control: { armed: false, can_arm: true, target: 'practice', selection_count: 0, click_count: 0, finger_click_count: 0,
    hand_pointer: { required: true, ready: false, state: 'mouse_paused', hint: 'Point your index finger briefly to engage.' },
    neutral_ready: false, decoder: passedClickClassifier(),
    prediction: { valid: true, label: 'rest', confidence: 0.93, margin: 0.50 }, ...extra } });

test('default markup introduces hand aiming and practice before EEG mouse clicks', () => {
  const html = fs.readFileSync(path.join(__dirname, '../ui/renderer/index.html'), 'utf8');
  // The first visible choice is selection; advanced legacy modes are still available.
  assert.match(html, /value="eeg_click" selected>EEG selection \/ click/);
  assert.match(html, /Webcam direction \(advanced\)/);
  assert.match(html, /value="practice">Practice counter/);
  assert.match(html, /count four seconds, then reopen/);
  assert.match(html, /Move your hand to aim; click with your fingers or the trained EEG pattern/);
  assert.match(html, /Pinch and release to click; hold a pinch to drag/);
  assert.match(html, /Index bend clicks follow your existing optional Hand controls setting/);
  assert.match(html, /Arm hand controls \+ EEG practice/);
  assert.doesNotMatch(html, /id="multimodal-camera-container" hidden/);
  const r = harness();
  assert.equal(r.el('control-mode').value, 'eeg_click');
  assert.equal(r.el('eeg-arm').disabled, true, 'unknown backend state cannot arm');
});

test('binary trials keep the camera visible but do not require a pointing hand', async () => {
  const r = harness();
  const state = clickState({ decoder: clickClassifier() });
  state.camera.tracked = false; // EEG cues label calibration while hand output remains disarmed.
  r.panel.acceptStatus(state);
  // Hidden left/right buttons remain disabled even if an event is dispatched programmatically.
  for (const label of ['left', 'right']) {
    assert.equal(r.el(`eeg-${label}`).hidden, true);
    assert.equal(r.el(`column-${label}`).hidden, true);
    assert.equal(r.el(`train-${label}`).hidden, true);
    r.el(`eeg-${label}`).dispatch('click');
  }
  assert.equal(r.requests.length, 0);
  assert.equal(r.el('camera-container').hidden, false);
  assert.equal(r.el('webcam-arm-label').hidden, true);
  assert.equal(r.el('eeg-rest').disabled, false);
  assert.equal(r.el('eeg-select').disabled, false);
  assert.match(r.el('calibration-heading').textContent, /Two-state/);
  r.el('eeg-select').dispatch('click');
  assert.deepEqual(r.requests[0].payload, { label: 'select', phase: 'train' });
  await respond(r.requests[0], clickState({ decoder: clickClassifier(), pending: {
    label: 'select', phase: 'train', remaining_seconds: 4.5, cue: 'Gently close your eyes; keep your head and jaw still.' } }));
  assert.match(r.el('eeg-trial').textContent, /Gently close your eyes/);
  assert.equal(r.el('eeg-rest').disabled, true, 'overlapping trials are blocked');
});

test('binary validation freezes training and renders only the active confusion matrix', async () => {
  const r = harness(); r.panel.acceptStatus(clickState({ decoder: clickClassifier({ trained: true,
    train_counts: { rest: 8, select: 8 }, validation_counts: { rest: 0, select: 0 } }) }));
  // Subsequent trial clicks always enter held-out validation instead of extending training.
  r.el('eeg-rest').dispatch('click');
  assert.deepEqual(r.requests[0].payload, { label: 'rest', phase: 'validate' });
  await respond(r.requests[0], clickState());
  assert.equal(r.el('eeg-train').disabled, true);
  assert.equal(r.el('eeg-rest').disabled, true);
  assert.equal(r.el('eeg-select').disabled, true);
  assert.match(r.el('eeg-evaluation').textContent, /predicted rest \/ select \/ uncertain/);
  assert.match(r.el('eeg-evaluation').textContent, /select: precision 100.0%/);
  assert.doesNotMatch(r.el('eeg-evaluation').textContent, /left|right/);
});

test('hand arming stays independent of EEG training and validation', () => {
  for (const extra of [{ decoder: passedClickClassifier({ labels: undefined }) },
    { decoder: passedClickClassifier({ labels: ['rest'] }) },
    { decoder: passedClickClassifier({ labels: ['left', 'right', 'rest'] }) },
    { decoder: passedClickClassifier({ validation_counts: { rest: 7, select: 8 } }) },
    { decoder: clickClassifier() }]) {
    const r = harness(); r.panel.acceptStatus(clickState(extra));
    assert.equal(r.el('eeg-arm').disabled, false, 'the backend gates the EEG add-on independently');
  }
  const r = harness(); r.panel.acceptStatus(clickState({ target: undefined }));
  assert.equal(r.el('eeg-arm').disabled, true, 'the exact intended output must still be known');
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
  assert.equal(r.guards.length, 0);
  assert.equal(r.requests.length, 0);
});

test('practice and mouse targets name exact output, show counts and disarm before target changes', async () => {
  const r = harness(); r.panel.acceptStatus(clickState({ selection_count: 3, finger_click_count: 2 }));
  assert.equal(r.el('eeg-arm-label').textContent, 'Arm hand controls + EEG practice');
  assert.match(r.el('selection-counts').textContent, /EEG practice: 3 · EEG clicks: 0 · Finger clicks: 2/);
  // Practice requires the same Escape guard as actual mouse output.
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  assert.equal(r.guards.length, 1);
  await respond(r.requests[0], clickState({ armed: true, selection_count: 3 }));
  assert.equal(r.el('eeg-arm').checked, true);
  assert.match(r.el('control-status').textContent, /Finger clicks and drags still affect the desktop/);
  r.el('eeg-target').value = 'mouse'; r.el('eeg-target').dispatch('change');
  assert.deepEqual(r.requests[1].payload, { target: 'mouse' });
  assert.equal(r.requests[1].action, 'eeg_target');
  assert.equal(r.el('eeg-arm').checked, false);
  assert.ok(r.guardStops.length, 'changing output releases the old lease immediately');
  await respond(r.requests[1], clickState({ target: 'mouse', selection_count: 3, click_count: 1 }));
  assert.equal(r.el('eeg-arm-label').textContent, 'Arm hand controls + EEG clicks');
  assert.equal(r.el('eeg-arm').checked, false, 'a target change never implicitly rearms');
  assert.match(r.el('selection-counts').textContent, /EEG clicks: 1/);
  assert.equal(r.guards.length, 1);
});

test('hybrid freshness expiry pauses feedback while preserving the shared Escape guard', async () => {
  const r = harness(); r.panel.acceptStatus(clickState());
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  await respond(r.requests[0], clickState({ armed: true }));
  r.open(true); await respond(r.requests[1], clickState({ armed: true }));
  const releases = r.guardStops.length;
  // Input suppression and recovery happen in core; Escape remains available during a sensor pause.
  r.tick(501);
  assert.equal(r.el('eeg-arm').checked, true);
  assert.equal(r.el('eeg-arm').disabled, false, 'the user can still disarm while tracking is paused');
  assert.equal(r.guardStops.length, releases);
  assert.match(r.el('hand-pointer-status').textContent, /fresh camera frame/);
  r.tick(250); // The independent EEG deadline still clears its prediction afterwards.
  assert.match(r.el('prediction').textContent, /Waiting for fresh EEG/);
  assert.equal(r.requests.length, 2);
});

test('selection trial cues say to reopen and mode switches restore legacy controls', () => {
  const r = harness(); r.panel.acceptStatus(clickState({ decoder: clickClassifier(), pending: {
    label: 'select', phase: 'train', remaining_seconds: 0.5, cue: 'Open your eyes and rest.' } }));
  assert.match(r.el('eeg-trial').textContent, /Open your eyes and rest/);
  // Switching away clears selection-only UI instead of leaving stale binary state visible.
  r.panel.acceptStatus(running({ control_mode: 'webcam', eeg_control: { decoder: classifier() } }));
  assert.equal(r.el('eeg-select').hidden, true);
  assert.equal(r.el('eeg-left').hidden, false);
  assert.equal(r.el('target-controls').hidden, true);
  assert.equal(r.el('selection-counts').hidden, true);
  assert.equal(r.el('camera-container').hidden, false);
});

test('pending practice arm cannot become a mouse arm after a newer target snapshot', async () => {
  const r = harness({ guardMode: 'pending' }); r.panel.acceptStatus({ ...clickState(), revision: 1 });
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
  // Another client changing targets must not inherit this user's practice authorization.
  r.panel.acceptStatus({ ...clickState({ target: 'mouse' }), revision: 2 });
  const token = '11111111-2222-4333-8444-555555555555';
  r.guards[0].resolve({ ok: true, token }); await flush();
  assert.equal(r.requests.length, 0);
  assert.equal(r.el('eeg-arm').checked, false);
  assert.ok(r.guardStops.includes(token));
  assert.match(r.el('eeg-errors').textContent, /readiness changed/);
});

test('hybrid arming requires a fresh tracked camera and the hand pointer contract', () => {
  for (const change of [state => { state.camera.running = false; },
    state => { state.camera.tracked = false; }, state => { state.camera.age_ms = 501; },
    state => { delete state.eeg_control.hand_pointer; }]) {
    const state = clickState(); change(state);
    const r = harness(); r.panel.acceptStatus(state);
    assert.equal(r.el('eeg-arm').disabled, true, 'backend can_arm alone cannot bypass the hand checks');
    r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
    assert.equal(r.guards.length, 0);
    assert.equal(r.requests.length, 0);
  }
});

test('hybrid hand engagement begins after arming and shows controller feedback', () => {
  const r = harness(); r.panel.acceptStatus(clickState());
  assert.equal(r.el('eeg-arm').disabled, false, 'point engagement is not required before the arm starts it');
  assert.match(r.el('hand-pointer-status').textContent, /Point your index finger/);
  r.panel.acceptStatus(clickState({ armed: true,
    hand_pointer: { required: true, ready: true, state: 'mouse_pointer', hint: 'Move your hand to aim.' } }));
  assert.match(r.el('hand-pointer-status').textContent, /Hand pointer ready: Move your hand to aim/);
  assert.match(r.el('control-status').textContent, /move your hand to aim/);
});

test('armed hybrid output preserves finger clicks and explains their priority over EEG', () => {
  const r = harness(); r.panel.acceptStatus(clickState({ armed: true, target: 'mouse', finger_click_count: 4 }));
  assert.match(r.el('control-status').textContent, /Pinch\/release clicks and pinch-hold drags remain active/);
  assert.match(r.el('control-status').textContent, /optional bend clicks follow your setting/);
  assert.match(r.el('control-status').textContent, /Finger gestures cancel a pending EEG selection/);
  assert.match(r.el('selection-counts').textContent, /Finger clicks: 4/);
  assert.match(r.el('mode-note').textContent, /Practice sends EEG selections to the counter, while finger clicks and drags still affect the desktop/);
  assert.doesNotMatch(r.el('mode-note').textContent, /pinch.*cannot|bending.*does not click/i);
});

test('hybrid camera loss blocks calibration while an untracked fresh camera permits EEG cues', () => {
  for (const camera of [{ running: false }, { running: true, tracked: true, age_ms: 501 }]) {
    const r = harness(); r.panel.acceptStatus({ ...clickState({ decoder: clickClassifier() }), camera });
    assert.equal(r.el('eeg-rest').disabled, true);
    assert.equal(r.el('eeg-select').disabled, true);
  }
});

test('temporary hand loss keeps the acknowledged arm and Escape lease for automatic recovery', async () => {
  const r = harness(); r.panel.acceptStatus(clickState());
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  await respond(r.requests[0], clickState({ armed: true }));
  const releases = r.guardStops.length;
  const state = clickState({ armed: true }); state.camera.tracked = false;
  r.panel.acceptStatus(state);
  assert.equal(r.el('eeg-arm').checked, true);
  assert.equal(r.el('eeg-arm').disabled, false);
  assert.equal(r.guardStops.length, releases);
  assert.match(r.el('hand-pointer-status').textContent, /resumes automatically/);
  r.panel.acceptStatus(clickState({ armed: false }));
  assert.ok(r.guardStops.length > releases, 'a confirmed core disarm still releases the lease');
});

test('hybrid guard preparation rechecks hand loss and camera expiration before arming', async () => {
  for (const handLost of [false, true]) {
    const r = harness({ guardMode: 'pending' }); r.panel.acceptStatus({ ...clickState(), revision: 1 });
    r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
    if (handLost) {
      const state = { ...clickState(), revision: 2 }; state.camera.tracked = false;
      r.panel.acceptStatus(state);
    } else r.advance(501); // EEG remains fresh here; camera freshness alone must prevent the arm.
    const token = '11111111-2222-4333-8444-555555555555';
    r.guards[0].resolve({ ok: true, token }); await flush();
    assert.equal(r.requests.length, 0);
    assert.ok(r.guardStops.includes(token));
    assert.equal(r.el('eeg-arm').checked, false);
    assert.match(r.el('eeg-errors').textContent, /readiness changed/);
  }
});

test('saved model restore is explicit, preserves failed validation, and does not arm output', async () => {
  const r = harness(); r.panel.acceptStatus(clickState({ decoder: clickClassifier() }));
  assert.equal(r.requests.length, 0, 'status never implicitly loads a personal model');
  r.el('eeg-load-saved').dispatch('click');
  assert.equal(r.requests[0].action, 'eeg_load_saved');
  assert.deepEqual(r.requests[0].payload, {});
  const failed = passedClickClassifier({ passed_validation: false, arm_eligible: false, state: 'validation_failed', validation_scope: 'saved_acquisition',
    evaluation: { passed: false, complete: true, accuracy: 0.875, balanced_accuracy: 0.875,
      gates: { rest_false_activation_rate: false }, rest_false_activation_rate: 0.125 } });
  await respond(r.requests[0], clickState({ decoder: failed, experimental_model: true,
    saved_model_source: 'model-existing.json' }));
  assert.equal(r.el('eeg-arm').checked, false);
  assert.equal(r.el('eeg-arm').disabled, false, 'hand controls remain available');
  assert.equal(r.guards.length, 0, 'restoring never requests an Escape lease or arm');
  assert.match(r.el('eeg-gates').textContent, /validation did not pass/);
  assert.match(r.el('eeg-gates').textContent, /Rest false activations.*not passed/);
  assert.match(r.el('eeg-evaluation').textContent, /87.5%/);
  assert.match(r.el('eeg-evaluation').textContent, /historical saved acquisition/);
  assert.match(r.el('eeg-saved-status').textContent, /model-existing.json/);
});

test('save model sends no paths and keeps an existing hand arm and Escape lease', async () => {
  const r = harness(); r.panel.acceptStatus(clickState());
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  await respond(r.requests[0], clickState({ armed: true }));
  const releases = r.guardStops.length;
  r.el('eeg-save').dispatch('click');
  assert.equal(r.requests[1].action, 'eeg_save');
  assert.deepEqual(r.requests[1].payload, {});
  assert.equal(r.guardStops.length, releases);
  await respond(r.requests[1], clickState({ armed: true, saved_model_file: 'model-backup.json' }));
  assert.equal(r.el('eeg-arm').checked, true);
  assert.match(r.el('eeg-saved-status').textContent, /model-backup.json/);
});

test('EEG signal loss pauses only the add-on and does not prevent hand arming or recovery', async () => {
  const r = harness();
  r.panel.acceptStatus({ ...clickState({ decoder: clickClassifier(), can_eeg_click: false }), eeg: { state: 'stopped' } });
  assert.equal(r.el('eeg-arm').disabled, false);
  assert.equal(r.el('eeg-load-saved').disabled, true, 'load requires a fresh compatible EEG acquisition');
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  await respond(r.requests[0], { ...clickState({ armed: true, can_eeg_click: false }), eeg: { state: 'stopped' } });
  assert.equal(r.el('eeg-arm').checked, true);
  assert.match(r.el('control-status').textContent, /EEG click add-on is paused/);
  const releases = r.guardStops.length;
  r.panel.acceptStatus(clickState({ armed: true, can_eeg_click: true }));
  assert.match(r.el('control-status').textContent, /EEG click add-on is ready/);
  assert.equal(r.guardStops.length, releases);
});

test('EEG method selector defaults to the existing model and does not change anything on status reads', () => {
  const html = fs.readFileSync(path.join(__dirname, '../ui/renderer/index.html'), 'utf8');
  assert.match(html, /id="multimodal-eeg-method" disabled/);
  assert.match(html, /value="eyes_closed" selected>Eyes closed \(existing model\)/);
  assert.match(html, /value="double_blink">Deliberate double blink/);
  const r = harness(); r.panel.acceptStatus(clickState());
  assert.equal(r.el('eeg-method').value, 'eyes_closed', 'older backend snapshots retain the established workflow');
  assert.equal(r.requests.length, 0);
  assert.equal(r.el('method-controls').hidden, false);
  r.panel.acceptStatus(eegState());
  assert.equal(r.el('method-controls').hidden, true, 'advanced swipes never expose a click method');
});

test('changing EEG method disarms immediately and restores the separate acknowledged model counts', async () => {
  const r = harness(); r.panel.acceptStatus(clickState());
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  await respond(r.requests[0], clickState({ armed: true }));
  const releases = r.guardStops.length;
  r.el('eeg-method').value = 'double_blink'; r.el('eeg-method').dispatch('change');
  assert.equal(r.requests[1].action, 'eeg_method');
  assert.deepEqual(r.requests[1].payload, { method: 'double_blink' });
  assert.equal(r.el('eeg-arm').checked, false);
  assert.ok(r.guardStops.length > releases);
  assert.equal(r.el('eeg-method').value, 'eyes_closed', 'wait for backend acknowledgment');
  await respond(r.requests[1], clickState({ method: 'double_blink', decoder: clickClassifier() }));
  assert.equal(r.el('train-rest').textContent, '0 / 8');
  assert.equal(r.el('eeg-arm').checked, false);
  assert.equal(r.el('eeg-reset').textContent, 'Reset double-blink model & trials');
  r.el('eeg-method').value = 'eyes_closed'; r.el('eeg-method').dispatch('change');
  await respond(r.requests[2], clickState({ method: 'eyes_closed' }));
  assert.equal(r.el('train-rest').textContent, '8 / 8');
  assert.equal(r.el('validate-select').textContent, '8 / 8');
  assert.ok(r.requests.every(request => ['eeg_arm', 'eeg_method'].includes(request.action)), 'method changes never issue reset or retrain');
  assert.equal(r.guards.length, 1, 'switching does not silently rearm');
});

test('pending arm cannot authorize another EEG method after guard preparation', async () => {
  const r = harness({ guardMode: 'pending' });
  r.panel.acceptStatus({ ...clickState({ method: 'eyes_closed' }), revision: 1 });
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
  r.panel.acceptStatus({ ...clickState({ method: 'double_blink' }), revision: 2 });
  const token = '11111111-2222-4333-8444-555555555555';
  r.guards[0].resolve({ ok: true, token }); await flush();
  assert.equal(r.requests.length, 0);
  assert.ok(r.guardStops.includes(token));
  assert.equal(r.el('eeg-arm').checked, false);
  assert.match(r.el('eeg-errors').textContent, /readiness changed/);
});

test('double-blink calibration uses brief paired-blink cues and the existing fixed trial payload', async () => {
  const r = harness(); r.panel.acceptStatus(clickState({ method: 'double_blink', decoder: clickClassifier() }));
  assert.equal(r.el('column-rest').textContent, 'No command');
  assert.equal(r.el('column-select').textContent, 'Double blink');
  assert.match(r.el('calibration-heading').textContent, /separate model/);
  assert.match(r.el('calibration-instructions').textContent, /8 No command and 8 Double blink training trials/);
  assert.match(r.el('calibration-instructions').textContent, /natural single blinks are allowed/);
  assert.match(r.el('calibration-instructions').textContent, /Collect remaining validation for 8 new trials of each/);
  assert.doesNotMatch(r.el('calibration-instructions').textContent, /count four seconds|hold.*closed/);
  r.el('eeg-select').dispatch('click');
  assert.deepEqual(r.requests[0].payload, { label: 'select', phase: 'train' });
  await respond(r.requests[0], clickState({ method: 'double_blink', decoder: clickClassifier(),
    pending: { label: 'select', phase: 'train', remaining_seconds: 2.5 } }));
  assert.match(r.el('eeg-trial').textContent, /blink twice deliberately now, then keep your eyes open/);
  assert.doesNotMatch(r.el('eeg-trial').textContent, /count four seconds/);
  assert.equal(r.el('eeg-method').disabled, true, 'active trial cannot be relabeled by switching methods');
  r.panel.acceptStatus(clickState({ method: 'double_blink', decoder: clickClassifier({ trained: true,
    train_counts: { rest: 8, select: 8 } }) }));
  r.el('eeg-rest').dispatch('click');
  assert.deepEqual(r.requests[1].payload, { label: 'rest', phase: 'validate' });
  await respond(r.requests[1], clickState({ method: 'double_blink', pending: { label: 'rest', phase: 'validate',
    remaining_seconds: 3, cue: 'Read normally and allow natural single blinks.' } }));
  assert.match(r.el('eeg-trial').textContent, /Read normally and allow natural single blinks/);
});

test('failed blink validation never appears ready through the old experimental-model flag', () => {
  const decoder = passedClickClassifier({ state: 'validation_failed', passed_validation: false, arm_eligible: false,
    evaluation: { passed: false, complete: true, accuracy: 0.875, balanced_accuracy: 0.875,
      gates: { rest_false_activation_rate: false }, rest_false_activations: 1, rest_trials: 8, rest_false_activation_rate: 0.125 } });
  const r = harness(); r.panel.acceptStatus(clickState({ method: 'double_blink', decoder,
    armed: true, target: 'mouse', experimental_model: true, can_eeg_click: true }));
  assert.equal(r.el('eeg-arm').checked, true, 'hand controls remain independently armed');
  assert.match(r.el('control-status').textContent, /EEG click add-on is paused/);
  assert.doesNotMatch(r.el('control-status').textContent, /add-on is ready/);
  assert.doesNotMatch(r.el('eeg-gates').textContent, /Experimental saved model/);
  assert.match(r.el('validation-note').textContent, /failed validation does not enable EEG clicks/);
  assert.match(r.el('eeg-evaluation').textContent, /No-command false activations: 1 \/ 8/);
  assert.match(r.el('eeg-evaluation').textContent, /87.5%/);
});

test('validated double-blink controls explain one left click and preserve finger input without long holds', () => {
  const r = harness(); r.panel.acceptStatus(clickState({ method: 'double_blink', armed: true,
    target: 'mouse', can_eeg_click: true, prediction: { valid: true, label: 'select', confidence: 0.92 } }));
  assert.match(r.el('control-status').textContent, /One deliberate double blink produces one left click/);
  assert.match(r.el('control-status').textContent, /Pinch\/release clicks and pinch-hold drags remain active/);
  assert.match(r.el('control-status').textContent, /add-on is ready/);
  assert.match(r.el('prediction').textContent, /EEG blink detection: double blink/);
  assert.match(r.el('session-note').textContent, /Single blinks are ignored as click commands/);
  assert.doesNotMatch(r.el('control-status').textContent, /rest, select, then rest|count four seconds/);
  assert.match(r.el('eeg-evaluation').textContent, /predicted no command \/ double blink \/ uncertain/);
  assert.match(r.el('eeg-evaluation').textContent, /double blink: precision 100.0%/);
  assert.equal(r.el('eeg-load-saved').textContent, 'Use saved double-blink model');
  assert.equal(r.el('eeg-save').textContent, 'Save double-blink model & trials');
});

test('unknown method names cannot receive an arm and changing methods revokes an existing lease', async () => {
  const r = harness(); r.panel.acceptStatus(clickState());
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  await respond(r.requests[0], clickState({ armed: true }));
  const releases = r.guardStops.length;
  r.panel.acceptStatus(clickState({ method: 'unknown-method', armed: true }));
  assert.equal(r.el('eeg-arm').disabled, true);
  assert.equal(r.el('eeg-arm').checked, false);
  assert.ok(r.guardStops.length > releases);
  assert.equal(r.requests.length, 1, 'a new status does not manufacture another arm');
});

// The guided runner binds to this session and fills only the unfinished class quotas.
function guidedState({ rest = 0, select = 0, trained = false, validation = { rest: 0, select: 0 }, ...control } = {}) {
  const state = clickState({ method: 'double_blink', decoder: clickClassifier({ trained,
    train_counts: { rest, select }, validation_counts: validation, can_train: !trained && rest >= 8 && select >= 8 }), ...control });
  state.eeg.session_id = 'guided-test-session';
  return state;
}
function guidedHarness(state) {
  const r = harness();
  r.el('panel').open = true;  // Keep the cues visible without introducing mock background poll requests.
  r.panel.acceptStatus(state);
  return r;
}
function finishPreparation(r, state) {
  for (let i = 0; i < 12; i++) {
    r.panel.acceptStatus(state);  // A running preview keeps sensor metadata fresh throughout the countdown.
    r.tick(250);
  }
}

test('guided collection fills remaining quotas and never trains or starts validation automatically', async () => {
  const state = guidedState({ rest: 7, select: 7 });
  const r = guidedHarness(state);
  assert.equal(r.el('eeg-batch').disabled, false);
  r.el('eeg-batch').dispatch('click');
  assert.match(r.el('eeg-guided-cue').textContent, /No command starts in 3/);
  assert.equal(r.el('eeg-rest').disabled, true, 'manual trial buttons cannot overlap a guided batch');
  assert.equal(r.el('eeg-arm').disabled, true);
  finishPreparation(r, state);
  assert.deepEqual(r.requests[0].payload, { label: 'rest', phase: 'train' });
  assert.equal(r.requests.length, 1);
  // Read metadata arriving before the POST acknowledges its trial cannot queue a duplicate.
  r.panel.acceptStatus(state);
  assert.equal(r.requests.length, 1);
  await respond(r.requests[0], guidedState({ rest: 7, select: 7,
    pending: { label: 'rest', phase: 'train', remaining_seconds: 5, cue: 'Read normally.' } }));
  r.advance(5000);
  const waiting = guidedState({ rest: 7, select: 7, pending: { label: 'rest', phase: 'train',
    remaining_seconds: 0, samples_collected: 1170, samples_required: 1250, cue: 'Finishing signal collection…' } });
  r.panel.acceptStatus(waiting);
  assert.match(r.el('eeg-guided-cue').textContent, /Finishing signal collection/);
  assert.equal(r.requests.length, 1, 'zero on the countdown is not backend acceptance');
  r.advance(500); r.panel.acceptStatus(waiting);
  assert.equal(r.requests.length, 1);
  const firstAccepted = guidedState({ rest: 8, select: 7, last_result: 'Accepted train no-command trial.' });
  r.panel.acceptStatus(firstAccepted);
  assert.match(r.el('eeg-guided-cue').textContent, /Double blink starts in 3/);
  finishPreparation(r, firstAccepted);
  assert.deepEqual(r.requests[1].payload, { label: 'select', phase: 'train' });
  await respond(r.requests[1], guidedState({ rest: 8, select: 7,
    pending: { label: 'select', phase: 'train', remaining_seconds: 5, cue: 'Wait for the blink cue.' } }));
  r.panel.acceptStatus(guidedState({ rest: 8, select: 8, last_result: 'Accepted train double-blink trial.' }));
  assert.match(r.el('eeg-batch-status').textContent, /Training collection complete/);
  assert.match(r.el('eeg-batch-status').textContent, /validation is a separate step/);
  assert.equal(r.el('eeg-batch').disabled, true);
  assert.equal(r.el('eeg-train').disabled, false);
  assert.ok(r.requests.every(request => request.action === 'eeg_trial'));
  assert.equal(r.guards.length, 0);
});

test('guided collection skips completed no-command quota and preserves partially collected double blinks', () => {
  const state = guidedState({ rest: 8, select: 2 });
  const r = guidedHarness(state);
  r.el('eeg-batch').dispatch('click');
  assert.match(r.el('eeg-guided-cue').textContent, /Double blink starts in 3/);
  assert.match(r.el('eeg-batch-status').textContent, /No command 8 \/ 8 · Double blink 2 \/ 8/);
  finishPreparation(r, state);
  assert.deepEqual(r.requests[0].payload, { label: 'select', phase: 'train' });
  assert.equal(r.el('train-select').textContent, '2 / 8');
});

test('validation guided collection requires its own click and sends only held-out trial payloads', async () => {
  const state = guidedState({ rest: 8, select: 8, trained: true, validation: { rest: 7, select: 8 } });
  const r = guidedHarness(state);
  assert.equal(r.requests.length, 0);
  assert.equal(r.el('eeg-batch').textContent, 'Collect remaining validation');
  r.el('eeg-batch').dispatch('click'); finishPreparation(r, state);
  assert.deepEqual(r.requests[0].payload, { label: 'rest', phase: 'validate' });
  await respond(r.requests[0], guidedState({ rest: 8, select: 8, trained: true,
    validation: { rest: 7, select: 8 }, pending: { label: 'rest', phase: 'validate', remaining_seconds: 5 } }));
  r.panel.acceptStatus(guidedState({ rest: 8, select: 8, trained: true, validation: { rest: 8, select: 8 } }));
  assert.match(r.el('eeg-batch-status').textContent, /Validation collection complete/);
  assert.equal(r.requests.length, 1);
  assert.equal(r.guards.length, 0);
});

test('rejected guided trial pauses visibly and does not retry, reset, or discard counts', async () => {
  const state = guidedState({ rest: 8, select: 2 });
  const r = guidedHarness(state);
  r.el('eeg-batch').dispatch('click'); finishPreparation(r, state);
  await respond(r.requests[0], guidedState({ rest: 8, select: 2,
    pending: { label: 'select', phase: 'train', remaining_seconds: 5 } }));
  r.panel.acceptStatus(guidedState({ rest: 8, select: 2, last_result: 'Cue needs one clear double blink.' }));
  assert.match(r.el('eeg-batch-status').textContent, /paused: Cue needs one clear double blink/);
  assert.match(r.el('eeg-batch-status').textContent, /Existing data is retained/);
  assert.equal(r.el('train-select').textContent, '2 / 8');
  assert.equal(r.el('eeg-select').disabled, false, 'manual retry is available after a rejected batch trial');
  assert.equal(r.requests.length, 1);
  assert.equal(r.el('eeg-batch-cancel').disabled, true);
});

test('cancel queued trials preserves the in-progress trial and prevents later timer actions', async () => {
  const state = guidedState({ rest: 8, select: 2 });
  const r = guidedHarness(state);
  r.el('eeg-batch').dispatch('click');
  const oldTimer = r.timers.find(timer => !timer.cancelled && timer.delay === 250);
  r.el('eeg-batch-cancel').dispatch('click');
  oldTimer.fn();
  assert.equal(r.requests.length, 0, 'even an already queued timer cannot restart a cancelled batch');
  r.el('eeg-batch').dispatch('click'); finishPreparation(r, state);
  await respond(r.requests[0], guidedState({ rest: 8, select: 2,
    pending: { label: 'select', phase: 'train', remaining_seconds: 5 } }));
  r.el('eeg-batch-cancel').dispatch('click');
  assert.match(r.el('eeg-batch-status').textContent, /current trial can finish/);
  assert.equal(r.requests.length, 1, 'cancel does not send Stop, Reset, or a destructive trial operation');
  r.panel.acceptStatus(guidedState({ rest: 8, select: 3, last_result: 'Accepted train double-blink trial.' }));
  assert.equal(r.el('train-select').textContent, '3 / 8');
  assert.equal(r.requests.length, 1);
  assert.equal(r.el('eeg-select').disabled, false);
});

test('session changes, hidden cues, stopped preview, and phase changes cancel future guided requests', () => {
  for (const change of [
    (r, state) => { state.eeg.session_id = 'replacement-session'; r.panel.acceptStatus(state); },
    (r) => { r.document.hidden = true; r.document.dispatch('visibilitychange'); },
    (r, state) => { state.running = false; state.state = 'stopped'; r.panel.acceptStatus(state); },
    (r, state) => { state.eeg_control.decoder.trained = true; r.panel.acceptStatus(state); },
    (r, state) => { state.eeg_control.decoder.train_counts = { rest: 0, select: 0 }; r.panel.acceptStatus(state); },
    (r, state) => { state.eeg_control.saved_model_source = 'double-blink-replacement.json'; r.panel.acceptStatus(state); },
    (r, state) => { state.eeg_control.saved_calibration_source = 'calibration-replacement.json'; r.panel.acceptStatus(state); },
    (r) => r.panel.emergencyStop('escape'),
    (r) => r.online(false),
  ]) {
    const state = guidedState({ rest: 8, select: 2 }), r = guidedHarness(state);
    r.el('eeg-batch').dispatch('click');
    const oldTimer = r.timers.find(timer => !timer.cancelled && timer.delay === 250);
    change(r, state); oldTimer.fn();
    assert.equal(r.requests.length, 0);
    assert.equal(r.el('eeg-batch-cancel').disabled, true);
  }
});

test('explicit method switch or Stop cancels preparation and cannot be followed by a delayed trial', () => {
  for (const action of ['method', 'stop']) {
    const r = guidedHarness(guidedState({ rest: 8, select: 2 }));
    r.el('eeg-batch').dispatch('click');
    const oldTimer = r.timers.find(timer => !timer.cancelled && timer.delay === 250);
    if (action === 'method') {
      r.el('eeg-method').value = 'eyes_closed'; r.el('eeg-method').dispatch('change');
    } else r.el('stop').dispatch('click');
    oldTimer.fn();
    assert.equal(r.requests.length, 1);
    assert.equal(r.requests[0].action, action === 'method' ? 'eeg_method' : 'stop');
    assert.equal(r.el('eeg-batch-cancel').disabled, true);
  }
});

test('stale sensors and failed trial acknowledgments stop guided requests without retrying', async () => {
  const state = guidedState({ rest: 8, select: 2 });
  const stale = guidedHarness(state); stale.el('eeg-batch').dispatch('click');
  stale.advance(1000); stale.tick(250);
  assert.equal(stale.requests.length, 0);
  assert.match(stale.el('eeg-batch-status').textContent, /readiness changed/);
  const r = guidedHarness(state); r.el('eeg-batch').dispatch('click'); finishPreparation(r, state);
  r.requests[0].reject(new Error('connection interrupted')); await flush();
  r.panel.acceptStatus(state);
  assert.equal(r.requests.length, 1);
  assert.match(r.el('eeg-batch-status').textContent, /acknowledgment failed/);
  assert.equal(r.el('train-select').textContent, '2 / 8');
});

test('cancelling before a trial acknowledgment cannot revive a batch or start overlapping requests', async () => {
  const state = guidedState({ rest: 8, select: 2 }), r = guidedHarness(state);
  r.el('eeg-batch').dispatch('click'); finishPreparation(r, state);
  r.el('eeg-batch-cancel').dispatch('click');
  r.el('eeg-batch').dispatch('click');
  assert.equal(r.requests.length, 1, 'an unacknowledged POST keeps new collection disabled');
  await respond(r.requests[0], guidedState({ rest: 8, select: 2,
    pending: { label: 'select', phase: 'train', remaining_seconds: 0 } }));
  assert.match(r.el('eeg-guided-cue').textContent, /Finishing signal collection/);
  r.panel.acceptStatus(guidedState({ rest: 8, select: 3 }));
  assert.equal(r.el('eeg-batch-cancel').disabled, true);
  assert.equal(r.el('train-select').textContent, '3 / 8');
  assert.equal(r.requests.length, 1, 'late acknowledgment and count updates do not restore the cancelled queue');
});

test('blink collection stays collect, train, validate without draft resume controls', () => {
  const html = fs.readFileSync(path.join(__dirname, '../ui/renderer/index.html'), 'utf8');
  assert.doesNotMatch(html, /multimodal-eeg-(?:save-progress|load-progress|progress-controls|progress-status)/);
  const state = guidedState({ rest: 8, select: 2, saved_calibration_source: 'internal-backup.json' });
  const r = guidedHarness(state);
  assert.equal(r.el('train-rest').textContent, '8 / 8');
  assert.equal(r.el('train-select').textContent, '2 / 8');
  assert.equal(r.el('eeg-batch').textContent, 'Collect remaining training');
  assert.equal(r.el('eeg-select').disabled, false);
  assert.equal(r.el('eeg-train').disabled, true, 'collect the unfinished quota before training');
  assert.equal(r.el('eeg-save').disabled, true, 'saved-model controls remain restricted to trained models');
  assert.match(r.el('calibration-instructions').textContent, /2\. Click Train & freeze model\. 3\. Choose Collect remaining validation/);
  assert.doesNotMatch(r.el('calibration-instructions').textContent, /resume|acquisition|draft/i);
  r.panel.acceptStatus(guidedState({ rest: 8, select: 8, trained: true }));
  assert.equal(r.el('eeg-batch').textContent, 'Collect remaining validation');
  assert.equal(r.el('eeg-save').disabled, false);
  assert.equal(r.el('eeg-load-saved').textContent, 'Use saved double-blink model');
  assert.equal(r.el('eeg-arm').checked, false);
  assert.equal(r.requests.length, 0, 'status changes never collect, train, load backups, or arm automatically');
});

test('frozen blink fit reports retained outliers honestly and still requires separate validation', () => {
  const fit = { supported_select_trials: 7, total_select_trials: 8, excluded_select_trials: 1 };
  const state = guidedState({ rest: 8, select: 8, trained: true });
  Object.assign(state.eeg_control.decoder, { detector_version: 3, training_fit: fit });
  const r = guidedHarness(state);
  assert.equal(r.el('eeg-model-status').textContent,
    'Model learned from 7 of 8 double-blink trials. 1 unusual trial kept but not used. Collect validation next.');
  assert.equal(r.el('train-select').textContent, '8 / 8', 'a fit outlier is retained in collected evidence');
  assert.equal(r.el('eeg-batch').textContent, 'Collect remaining validation');
  assert.equal(r.el('eeg-arm').checked, false);
  assert.equal(r.requests.length, 0);
  state.eeg_control.decoder.validation_counts = { rest: 8, select: 8 };
  r.panel.acceptStatus(state);
  assert.doesNotMatch(r.el('eeg-model-status').textContent, /Collect validation next/);
  for (const change of [
    { trained: false }, { detector_version: 2 },
    { training_fit: { ...fit, excluded_select_trials: 0 } },
    { training_fit: { ...fit, supported_select_trials: 9 } },
  ]) {
    r.panel.acceptStatus({ ...state, eeg_control: { ...state.eeg_control,
      decoder: { ...state.eeg_control.decoder, ...change } } });
    assert.doesNotMatch(r.el('eeg-model-status').textContent, /Model learned from/);
  }
});

test('manual blink training can add clearer examples beyond eight without extending guided or validation quotas', async () => {
  const r = guidedHarness(guidedState({ rest: 8, select: 8 }));
  assert.equal(r.el('eeg-select').disabled, false);
  assert.equal(r.el('eeg-select').textContent, 'Collect another double-blink trial');
  assert.equal(r.el('eeg-rest').disabled, true, 'no-command training keeps its existing quota');
  assert.equal(r.el('eeg-batch').disabled, true, 'guided training still stops at eight per class');
  assert.match(r.el('eeg-model-status').textContent, /add another double-blink example if needed/);
  r.el('eeg-select').dispatch('click');
  assert.deepEqual(r.requests[0].payload, { label: 'select', phase: 'train' });
  await respond(r.requests[0], guidedState({ rest: 8, select: 9 }));
  assert.equal(r.el('train-select').textContent, '9 / 8', 'accepted prior examples remain counted');
  assert.equal(r.el('eeg-select').disabled, false);
  r.panel.acceptStatus(guidedState({ rest: 8, select: 71 }));
  assert.equal(r.el('eeg-select').disabled, false, 'the eightieth training trial can be collected');
  r.panel.acceptStatus(guidedState({ rest: 8, select: 72 }));
  assert.equal(r.el('eeg-select').disabled, true, 'training cannot exceed eighty accepted trials');
  assert.match(r.el('eeg-model-status').textContent, /collection is full \(80 trials\)/);
  assert.doesNotMatch(r.el('eeg-model-status').textContent, /add another/);
  r.panel.acceptStatus(guidedState({ rest: 8, select: 9, trained: true, validation: { rest: 8, select: 8 } }));
  assert.equal(r.el('eeg-select').disabled, true, 'frozen validation cannot be extended after eight');
  assert.equal(r.el('eeg-select').textContent, 'Collect double-blink trial');
  r.el('eeg-select').dispatch('click');
  assert.equal(r.requests.length, 1);
  r.panel.acceptStatus(clickState({ decoder: clickClassifier({ trained: false,
    train_counts: { rest: 8, select: 8 } }) }));
  assert.equal(r.el('eeg-select').disabled, true, 'eyes-closed training retains its existing quota');
});

test('freezing a completed guided training batch replaces its old train instruction with validation guidance', async () => {
  const state = guidedState({ rest: 8, select: 7 }), r = guidedHarness(state);
  r.el('eeg-batch').dispatch('click'); finishPreparation(r, state);
  await respond(r.requests[0], guidedState({ rest: 8, select: 8 }));
  assert.match(r.el('eeg-batch-status').textContent, /Training collection complete/);
  r.el('eeg-train').dispatch('click');
  await respond(r.requests[1], guidedState({ rest: 8, select: 8, trained: true }));
  assert.equal(r.el('eeg-batch').textContent, 'Collect remaining validation');
  assert.doesNotMatch(r.el('eeg-batch-status').textContent, /Training collection complete|Click Train & freeze/);
  assert.match(r.el('eeg-batch-status').textContent, /remaining validation trials/);
  assert.equal(r.requests.length, 2, 'freezing never starts the next phase automatically');
});

test('eighty blink training trials close all collection paths without blocking later validation', () => {
  const base = guidedHarness(guidedState({ rest: 7, select: 8 }));
  assert.equal(base.el('eeg-select').disabled, true, 'finish the base no-command quota before adding extra command examples');
  assert.equal(base.el('eeg-rest').disabled, false);
  base.panel.acceptStatus(guidedState({ rest: 8, select: 8 }));
  assert.equal(base.el('eeg-select').disabled, false, 'extra command examples become available after both base quotas');
  const r = guidedHarness(guidedState({ rest: 7, select: 73 }));
  assert.match(r.el('eeg-model-status').textContent, /collection is full \(80 trials\)/);
  assert.equal(r.el('eeg-rest').disabled, true, 'the shared training cap also applies to an unfinished rest quota');
  assert.equal(r.el('eeg-select').disabled, true);
  assert.equal(r.el('eeg-batch').disabled, true, 'guided collection cannot queue a trial the backend will reject at capacity');
  r.el('eeg-rest').dispatch('click'); r.el('eeg-batch').dispatch('click');
  assert.equal(r.requests.length, 0);
  r.panel.acceptStatus(guidedState({ rest: 8, select: 72, trained: true }));
  assert.equal(r.el('eeg-rest').disabled, false, 'held-out validation has a separate fixed quota');
  assert.equal(r.el('eeg-select').disabled, false);
  assert.equal(r.el('eeg-batch').disabled, false);
});

test('old completion guidance clears after a count reset, source replacement, or connection change', async () => {
  for (const change of ['reset', 'source', 'session', 'connection']) {
    const state = guidedState({ rest: 8, select: 7 }), r = guidedHarness(state);
    r.el('eeg-batch').dispatch('click'); finishPreparation(r, state);
    await respond(r.requests[0], guidedState({ rest: 8, select: 8 }));
    assert.match(r.el('eeg-batch-status').textContent, /Training collection complete/);
    const next = guidedState({ rest: change === 'reset' ? 0 : 8, select: change === 'reset' ? 0 : 8 });
    if (change === 'source') next.eeg_control.saved_model_source = 'different-model.json';
    if (change === 'session') next.eeg.session_id = 'different-acquisition';
    if (change === 'connection') { r.online(false); r.online(true); }
    r.panel.acceptStatus(next);
    assert.doesNotMatch(r.el('eeg-batch-status').textContent, /Training collection complete/);
    assert.equal(r.requests.filter(request => request.action === 'eeg_trial').length, 1, 'an identity change never queues another trial');
  }
});
