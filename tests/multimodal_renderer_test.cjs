// Synthetic samples and mock DOM: never opens hardware, Electron, or a socket.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createMultimodalPanel } = require('../ui/renderer/multimodal-panel.cjs');

function harness({ guardMode = 'ready' } = {}) {
  const elements = new Map(), timers = [], requests = [], images = [], drawing = [], privacy = [];
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
    onCameraPrivacy: data => privacy.push(data),
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
    privacy,
    advance(ms) { time += ms; },
  };
}

const TOKEN = '11111111-2222-4333-8444-555555555555';
const points = Array.from({ length: 21 }, (_, i) => [i / 30, i / 40, 0]);
// Fixed-profile fixtures intentionally include the evidence required to arm.
function running({ decoder = {}, eeg_control = {}, camera = {}, eeg = {}, ...extra } = {}) {
  return { state: 'running', running: true, control_mode: 'eeg_click',
    reason: 'Acquisition running.',
    camera: { running: true, tracked: true, image: 'data:image/jpeg;base64,YWJj', age_ms: 0,
      fps: 30, width: 480, height: 360, landmarks: points, ...camera },
    eeg: { state: 'running', mode: 'signal', units: 'V', session_id: 'session-1', nominal_hz: 250,
      channels: [{ name: 'T3' }, { name: 'T4' }],
      stats: { age_seconds: 0, received_rate_hz: 250, gaps: 0, duplicates: 0, nonfinite: 0, channel_mismatches: 0, queue_drops: 0 },
      waveform: Array.from({ length: 10 }, (_, i) => ({ samples: [i * 1e-6, -i * 1e-6] })), ...eeg },
    eeg_control: { method: 'cnn_blink_fast', target: 'mouse', armed: false, can_arm: true,
      can_eeg_click: true, camera_mouse_enabled: true, hand_pointer: { enabled: true, required: false, ready: true },
      prediction: { valid: true, ready: true, ready_for_blink: true, label: 'rest', confidence: .9 },
      decoder: { trained: true, passed_validation: true, arm_eligible: true, evaluation: { passed: true },
        feature_profile: 'cnn_blink_fast_endpoint_v1', window_seconds: 1,
        validation_scope: 'held_out_one_second_trials', labels: ['rest', 'select'],
        train_counts: { rest: 8, select: 8 }, validation_counts: { rest: 8, select: 8 },
        ready_for_blink: true, ...decoder }, ...eeg_control }, ...extra };
}
const stopped = extra => running({ state: 'stopped', running: false, reason: 'Acquisition stopped.', ...extra });
const collecting = (extra = {}) => running({ ...extra, decoder: { trained: false, passed_validation: false,
  arm_eligible: false, train_counts: { rest: 0, select: 0 }, validation_counts: { rest: 0, select: 0 }, ...extra.decoder } });
const flush = () => new Promise(resolve => setImmediate(resolve));
async function respond(request, data, ok = true) { request.resolve({ ok, status: ok ? 200 : 503, data }); await flush(); }
const action = (r, name) => r.requests.filter(request => request.action === name).at(-1);
async function arm(r, data = running()) {
  r.panel.acceptStatus(data); r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  const armed = structuredClone(data); armed.eeg_control.armed = true;
  await respond(action(r, 'eeg_arm'), armed);
}
async function preview(r, data = running()) {
  r.panel.acceptStatus(data); r.open(true); await respond(action(r, 'preview'), data);
}

test('real markup contains one workflow, accurate storage scope, and persistent camera privacy status', () => {
  const html = fs.readFileSync(path.join(__dirname, '../ui/renderer/index.html'), 'utf8');
  const markup = html.match(/<details class="multimodal-panel"[\s\S]*?<\/details>/)[0];
  assert.doesNotMatch(markup.split('>')[0], /\bopen\b/);
  assert.doesNotMatch(markup, /<select|practice|EEG swipes|eyes.closed|5.second|automatic double|legacy/i);
  assert.match(markup, /processed one-second training and validation waveforms/);
  assert.match(markup, /pinch-hold drags/); assert.match(markup, /not thought reading/);
  assert.match(markup, /Desktop swipes: raise index, middle, ring, and pinky/);
  assert.match(markup, /left for the next desktop or right for the previous/);
  assert.match(markup, /combined mode supports horizontal swipes/);
  assert.match(markup, /does not measure physical blink-to-click latency/);
  assert.match(html, /id="camera-privacy"[^>]*role="status"/);
  for (const id of ['start','stop','eeg-camera','eeg-arm','camera','eeg','eeg-rest','eeg-select','eeg-train','eeg-save','eeg-load-saved'])
    assert.match(markup, new RegExp(`id="multimodal-${id}"`));
  const source = fs.readFileSync(path.join(__dirname, '../ui/renderer/multimodal-panel.cjs'), 'utf8');
  const ids = source.match(/const ids = \[([\s\S]*?)\];/)[1].match(/'[^']+'/g).map(id => id.slice(1, -1));
  for (const id of ids) assert.match(markup, new RegExp(`id="multimodal-${id}"`), `DOM contains ${id}`);
});

test('opening is read-only; collapsed panels discard pending raw previews', async () => {
  const r = harness(); assert.equal(r.requests.length, 0);
  r.open(true); assert.equal(r.requests[0].action, 'status');
  await respond(r.requests[0], stopped()); r.tick(1000);
  assert.equal(r.requests[1].action, 'status'); await respond(r.requests[1], running()); r.tick(125);
  assert.equal(r.requests[2].action, 'preview'); r.open(false);
  await respond(r.requests[2], running()); assert.equal(r.images.length, 0);
  assert.ok(r.requests.every(request => ['status','preview'].includes(request.action)));
});

test('Start requires BrainBit, exclusive input ownership, confirmed fast workflow and finished cleanup', async () => {
  const r = harness(); r.panel.acceptStatus(stopped());
  r.camera(true); r.el('start').dispatch('click'); assert.equal(r.requests.length, 0);
  assert.match(r.el('start-hint').textContent, /Stop the separate Hand controls/);
  r.camera(false); r.brainbit(false); assert.equal(r.el('start').disabled, true);
  r.brainbit(true); r.panel.acceptStatus(stopped({ cleanup_pending: true })); assert.equal(r.el('start').disabled, true);
  r.panel.acceptStatus(stopped({ eeg_control: { method: 'cnn_blink' } })); assert.equal(r.el('start').disabled, true);
  r.panel.acceptStatus(stopped()); r.el('start').dispatch('click'); r.el('start').dispatch('click');
  assert.equal(r.requests.length, 1); assert.equal(r.requests[0].action, 'start');
  await respond(r.requests[0], running()); assert.equal(r.el('eeg-arm').checked, false);
});

test('acknowledged camera-off changes Start copy without requiring a camera or hand', () => {
  const r = harness(); r.panel.acceptStatus(stopped({ eeg_control: { camera_mouse_enabled: false } }));
  assert.equal(r.el('start').textContent, 'Start EEG only'); assert.equal(r.el('start').disabled, false);
  r.panel.acceptStatus(running({ eeg_control: { camera_mouse_enabled: false }, camera: { running: false, tracked: false } }));
  assert.equal(r.el('eeg-arm').disabled, false); assert.equal(r.el('camera-container').hidden, true);
  assert.equal(r.el('eeg-arm-label').textContent, 'Arm EEG mouse clicks');
});

test('Stop supersedes pending Start and never accepts its late running response', async () => {
  const r = harness(); r.panel.acceptStatus(stopped()); r.el('start').dispatch('click'); const start = action(r, 'start');
  r.el('stop').dispatch('click'); assert.equal(r.el('stop').disabled, false);
  await respond(action(r, 'stop'), stopped()); await respond(start, running());
  assert.match(r.el('summary').textContent, /stopped/i); assert.equal(r.el('eeg-arm').checked, false);
});

for (const [name, decoder] of [
  ['failed validation', { passed_validation: false }], ['failed evaluation', { evaluation: { passed: false } }],
  ['insufficient held-out class', { validation_counts: { rest: 8, select: 7 } }],
  ['wrong profile', { feature_profile: 'cnn_blink_v1' }], ['five-second window', { window_seconds: 5 }],
  ['historical incompatible scope', { validation_scope: 'saved_acquisition' }],
  ['wrong label vocabulary', { labels: ['left','right'] }], ['missing train count', { train_counts: { rest: 8 } }],
]) test(`arming rejects ${name}`, () => {
  const r = harness(); r.panel.acceptStatus(running({ decoder, eeg_control: { experimental_model: true, can_arm: true } }));
  assert.equal(r.el('eeg-arm').disabled, true); r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
  assert.equal(r.guards.length, 0); assert.equal(r.requests.length, 0);
});

test('valid arm waits for native Escape registration and backend acknowledgment', async () => {
  const r = harness(); r.panel.acceptStatus(running()); r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
  assert.equal(r.el('eeg-arm').checked, false); await flush();
  assert.deepEqual(action(r, 'eeg_arm').payload, { enabled: true, guard_token: TOKEN });
  await respond(action(r, 'eeg_arm'), running({ eeg_control: { armed: true } }));
  assert.equal(r.el('eeg-arm').checked, true); assert.match(r.el('control-status').textContent, /one left click at the current cursor/);
});

test('fast overall validation result does not invent pending legacy per-gate flags', () => {
  const r = harness(); r.panel.acceptStatus(running());
  assert.match(r.el('eeg-gates').textContent, /Validation gates passed/);
  assert.match(r.el('eeg-gates').textContent, /Required: 8 validation trials per class/);
  assert.doesNotMatch(r.el('eeg-gates').textContent, /pending/);
});

test('failed validation explains why fresh camera and EEG cannot arm either mouse control', () => {
  const r = harness();
  r.panel.acceptStatus(running({ eeg_control: { can_arm: false, can_eeg_click: false },
    decoder: { state: 'validation_failed', passed_validation: false, arm_eligible: false,
      evaluation: { complete: true, passed: false, accuracy: 13 / 16, balanced_accuracy: 13 / 16,
        rest_false_activations: 1, rest_trials: 8, rest_false_activation_rate: .125,
        confusion: { rest: { rest: 7, select: 1, uncertain: 0 }, select: { rest: 2, select: 6, uncertain: 0 } } } } }));
  assert.equal(r.el('eeg-camera').checked, true);
  assert.match(r.el('camera-status').textContent, /Hand tracked/);
  assert.equal(r.el('eeg-arm').disabled, true); assert.equal(r.el('eeg-arm').checked, false);
  for (const id of ['control-status', 'hand-pointer-status', 'prediction']) {
    assert.match(r.el(id).textContent, /CNN validation failed/);
    assert.match(r.el(id).textContent, /no-command false selections: 1 \/ 8 \(12\.5%\)/);
    assert.match(r.el(id).textContent, /missed double blinks: 2 \/ 8/);
    assert.doesNotMatch(r.el(id).textContent, /Explicitly arm, then point/);
  }
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
  assert.equal(r.guards.length, 0); assert.equal(r.requests.length, 0);
});

for (const [name, evaluation] of [
  ['missing', null],
  ['malformed', { complete: true, passed: false, rest_false_activations: '1', rest_trials: 8,
    confusion: { select: { rest: -2, select: 6, uncertain: 0 } } }],
]) test(`failed validation stays clear and locked with ${name} evaluation details`, () => {
  const r = harness();
  r.panel.acceptStatus(running({ decoder: { state: 'validation_failed', passed_validation: false,
    arm_eligible: false, evaluation } }));
  for (const id of ['control-status', 'hand-pointer-status', 'prediction']) {
    assert.match(r.el(id).textContent, /CNN validation failed/);
    assert.doesNotMatch(r.el(id).textContent, /false selections:|missed double blinks:|NaN|undefined/);
  }
  assert.equal(r.el('eeg-arm').disabled, true); assert.equal(r.el('eeg-arm').checked, false);
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
  assert.equal(r.guards.length, 0); assert.equal(r.requests.length, 0);
});

test('unavailable global Escape prevents the arm POST', async () => {
  const r = harness({ guardMode: 'unavailable' }); r.panel.acceptStatus(running());
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  assert.equal(action(r, 'eeg_arm'), undefined); assert.match(r.el('eeg-errors').textContent, /Escape/);
});

for (const change of ['stop','offline','camera','stale','failed-model','trial']) test(`pending native arm cannot survive ${change}`, async () => {
  const r = harness({ guardMode: 'pending' }); r.panel.acceptStatus(running());
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change');
  if (change === 'stop') r.el('stop').dispatch('click');
  if (change === 'offline') r.online(false);
  if (change === 'camera') r.panel.acceptStatus(running({ eeg_control: { camera_mouse_enabled: false } }));
  if (change === 'stale') r.advance(751);
  if (change === 'failed-model') r.panel.acceptStatus(running({ decoder: { passed_validation: false } }));
  if (change === 'trial') r.panel.acceptStatus(running({ eeg_control: { pending: { phase: 'train', label: 'rest' } } }));
  r.guards[0].resolve({ ok: true, token: TOKEN }); await flush();
  assert.equal(action(r, 'eeg_arm'), undefined); assert.ok(r.guardStops.includes(TOKEN));
});

test('late arm response cannot resurrect input after Stop', async () => {
  const r = harness(); r.panel.acceptStatus(running()); r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); await flush();
  const arming = action(r, 'eeg_arm'); r.el('stop').dispatch('click'); await respond(action(r, 'stop'), stopped());
  await respond(arming, running({ eeg_control: { armed: true } }));
  assert.equal(r.el('eeg-arm').checked, false); assert.match(r.el('summary').textContent, /stopped/i);
});

test('usable high-amplitude EEG keeps the acknowledged arm and normal CNN readiness', async () => {
  const r = harness(); await arm(r); const revocations = r.guardStops.length;
  r.panel.acceptStatus(running({ eeg_control: { armed: true, can_arm: true, can_eeg_click: true,
    signal_quality_paused: false, signal_quality_reason: null,
    prediction: { valid: true, ready: true, ready_for_blink: true, label: 'rest', confidence: .1 } },
    eeg: { stats: { ...running().eeg.stats, channels: [{ name: 'O2', peak_to_peak_v: .04, rms_v: .014 }] } } }));
  assert.equal(r.el('eeg-arm').checked, true); assert.equal(r.el('eeg-arm').disabled, false);
  assert.match(r.el('control-status').textContent, /CNN ready for a new double blink/);
  assert.match(r.el('diagnostics').textContent, /40,000|40000/);
  assert.equal(r.guardStops.length, revocations);
});

test('transient CNN recovery preserves backend arm while hard disarm shows its sticky cause', async () => {
  const r = harness(); await arm(r); const revocations = r.guardStops.length;
  r.panel.acceptStatus(running({ eeg_control: { armed: true, can_arm: false, can_eeg_click: false,
    prediction: { valid: true, ready: false, ready_for_blink: false, reason: 'Keep eyes open until rest.' } } }));
  assert.equal(r.el('eeg-arm').checked, true); assert.match(r.el('prediction').textContent, /Keep eyes open until rest/);
  assert.equal(r.guardStops.length, revocations);
  r.panel.acceptStatus(running({ eeg_control: { armed: false, last_disarm_reason: 'EEG sample counter changed.' } }));
  assert.equal(r.el('eeg-arm').checked, false); assert.ok(r.guardStops.length > revocations);
  assert.match(r.el('control-status').textContent, /Last disarm: EEG sample counter changed/);
});

test('desktop navigation shows its live hint without calling the pointer ready or waiting', async () => {
  const r = harness(); await arm(r); const revocations = r.guardStops.length;
  for (const hint of ['Hold four fingers still', 'Next desktop: 70%', 'Lower your hand or make a fist before another swipe']) {
    r.panel.acceptStatus(running({ eeg_control: { armed: true, can_arm: false, can_eeg_click: false,
      hand_pointer: { state: 'desktop_swipe', ready: false, hint },
      prediction: { valid: true, ready: false, ready_for_blink: false, reason: 'Desktop navigation has priority.' } } }));
    assert.equal(r.el('eeg-arm').checked, true);
    assert.equal(r.el('hand-pointer-status').textContent,
      `Desktop swipe: ${hint} EEG clicks wait until navigation and recovery finish.`);
    assert.doesNotMatch(r.el('hand-pointer-status').textContent, /Camera mouse (ready|waiting)/);
    assert.equal(r.guardStops.length, revocations);
  }
  // Brief tracking loss can retain navigation ownership until its grace/reset completes.
  r.panel.acceptStatus(running({ camera: { tracked: false }, eeg_control: { armed: true,
    can_eeg_click: false, hand_pointer: { state: 'desktop_swipe', ready: false, hint: 'Tracking lost: navigation paused.' } } }));
  assert.match(r.el('hand-pointer-status').textContent, /Desktop swipe: Tracking lost/);
  assert.doesNotMatch(r.el('hand-pointer-status').textContent, /EEG blinks still click/);
  r.panel.acceptStatus(running({ eeg_control: { armed: true, can_arm: false, can_eeg_click: false,
    hand_pointer: { state: 'mouse_paused', ready: false, hint: 'Point briefly to resume aiming.' },
    prediction: { valid: true, ready: false, ready_for_blink: false,
      suppressed_evidence_remaining_seconds: .8, reason: 'Keep eyes open until rest.' } } }));
  assert.match(r.el('hand-pointer-status').textContent, /Camera mouse waiting: Point briefly to resume aiming/);
  assert.match(r.el('prediction').textContent, /0.8 s of newer EEG still needed after hand input/);
  assert.equal(r.el('eeg-arm').checked, true); assert.equal(r.guardStops.length, revocations);
});

test('stale or disarmed metadata cannot advertise an active desktop swipe', () => {
  const r = harness(), pointer = { state: 'desktop_swipe', ready: false, hint: 'Next desktop: 70%' };
  r.panel.acceptStatus(running({ eeg_control: { armed: false, hand_pointer: pointer } }));
  assert.match(r.el('hand-pointer-status').textContent, /disarmed/);
  assert.doesNotMatch(r.el('hand-pointer-status').textContent, /Desktop swipe:/);
  r.panel.acceptStatus(running({ camera: { age_ms: 900 }, eeg_control: { armed: true, hand_pointer: pointer } }));
  assert.match(r.el('hand-pointer-status').textContent, /Camera mouse paused/);
  assert.doesNotMatch(r.el('hand-pointer-status').textContent, /Desktop swipe:/);
});

test('camera expiry clears only camera while EEG expiry revokes the active guard', async () => {
  const r = harness(); await arm(r); r.open(true); const revocations = r.guardStops.length;
  await respond(action(r, 'preview'), running({ eeg_control: { armed: true } }));
  r.tick(501); assert.equal(r.el('eeg-arm').checked, true); assert.equal(r.guardStops.length, revocations);
  assert.match(r.el('camera-status').textContent, /stale/);
  r.tick(250); assert.equal(r.el('eeg-arm').checked, false); assert.ok(r.guardStops.length > revocations);
});

test('closing the panel and hiding the document do not revoke an acknowledged main-process guard', async () => {
  const r = harness(); await arm(r); r.open(true); const before = r.guardStops.length;
  r.open(false); r.document.hidden = true; r.document.dispatch('visibilitychange');
  r.panel.acceptStatus(running({ eeg_control: { armed: true } }));
  assert.equal(r.guardStops.length, before); assert.equal(r.el('eeg-arm').checked, true);
});

test('network failure hides output and revokes guard while leaving Stop available', async () => {
  const r = harness(); await arm(r); r.open(true); const before = r.guardStops.length;
  action(r, 'preview').reject(new Error('offline')); await flush();
  assert.equal(r.el('eeg-arm').checked, false); assert.equal(r.el('eeg-arm').disabled, true);
  assert.equal(r.el('stop').disabled, false); assert.ok(r.guardStops.length > before);
});

test('failed read during async guard preparation revokes the future token', async () => {
  const r = harness({ guardMode: 'pending' }); r.panel.acceptStatus(running());
  r.el('eeg-arm').checked = true; r.el('eeg-arm').dispatch('change'); r.open(true);
  action(r, 'preview').reject(new Error('read failure')); await flush();
  r.guards[0].resolve({ ok: true, token: TOKEN }); await flush();
  assert.equal(action(r, 'eeg_arm'), undefined); assert.ok(r.guardStops.includes(TOKEN));
});

test('lower revisions and stale read generations cannot overwrite newer stopped status', async () => {
  const r = harness(); r.panel.acceptStatus(running({ revision: 4 })); r.open(true);
  r.panel.acceptStatus(stopped({ revision: 6 }));
  await respond(action(r, 'preview'), running({ revision: 5, eeg_control: { armed: true } }));
  assert.equal(r.panel.acceptStatus(running({ revision: 3 })), false);
  assert.match(r.el('summary').textContent, /stopped/i); assert.equal(r.images.length, 0);
});

test('camera toggle disarms first, preserves EEG/model, and never posts Start or Stop', async () => {
  const r = harness(); await arm(r); const before = r.guardStops.length;
  r.el('eeg-camera').checked = false; r.el('eeg-camera').dispatch('change');
  assert.equal(r.el('eeg-arm').checked, false); assert.ok(r.guardStops.length > before);
  assert.deepEqual(action(r, 'eeg_camera').payload, { enabled: false });
  await respond(action(r, 'eeg_camera'), running({ eeg_control: { camera_mouse_enabled: false } }));
  assert.equal(r.el('eeg-camera').checked, false); assert.equal(r.el('eeg-arm').disabled, false);
  assert.equal(r.el('train-rest').textContent, '8 / 8');
  assert.equal(action(r, 'start'), undefined); assert.equal(action(r, 'stop'), undefined);
});

test('camera changes are blocked during a trial or cleanup, and Stop defeats a late toggle response', async () => {
  const r = harness(); r.panel.acceptStatus(running({ eeg_control: { pending: { label: 'rest', phase: 'train' } } }));
  assert.equal(r.el('eeg-camera').disabled, true);
  r.panel.acceptStatus(stopped({ cleanup_pending: true })); assert.equal(r.el('eeg-camera').disabled, true);
  r.panel.acceptStatus(running()); r.el('eeg-camera').checked = false; r.el('eeg-camera').dispatch('change');
  const changing = action(r, 'eeg_camera'); r.el('stop').dispatch('click'); await respond(action(r, 'stop'), stopped());
  await respond(changing, running({ eeg_control: { camera_mouse_enabled: false } })); assert.match(r.el('summary').textContent, /stopped/i);
});

test('both camera views import the same 21-joint overlay helper', async () => {
  const r = harness(); await preview(r); r.images[0].onload();
  assert.equal(r.drawing.filter(row => row[0] === 'multimodal-camera' && row[1] === 'arc').length, 21);
  assert.equal(r.drawing.filter(row => row[0] === 'multimodal-camera' && row[1] === 'lineTo').length, 21);
});

for (const change of ['camera-off','stop','hidden','offline','stale']) test(`late image decode cannot repaint after ${change}`, async () => {
  const r = harness(); await preview(r); const image = r.images[0]; r.drawing.length = 0;
  if (change === 'camera-off') r.panel.acceptStatus(running({ eeg_control: { camera_mouse_enabled: false } }));
  if (change === 'stop') r.el('stop').dispatch('click');
  if (change === 'hidden') r.open(false);
  if (change === 'offline') r.online(false);
  if (change === 'stale') r.advance(501);
  image.onload(); assert.equal(r.drawing.filter(row => row[1] === 'drawImage').length, 0);
});

test('camera-off never decodes a supplied frame; untracked camera never draws joint marks', async () => {
  const r = harness(); await preview(r, running({ eeg_control: { camera_mouse_enabled: false } })); assert.equal(r.images.length, 0);
  const u = harness(); await preview(u, running({ camera: { tracked: false } })); u.images[0].onload();
  assert.equal(u.drawing.filter(row => row[1] === 'arc').length, 0);
});

test('EEG drawing bounds rows and channels, rejects unknown units, and tolerates nonfinite samples', async () => {
  const r = harness(); await preview(r, running({ eeg: { waveform: Array.from({ length: 400 }, (_, i) => ({ samples: [i === 399 ? NaN : i * 1e-6, i * 1e-6] })) } }));
  const lines = r.drawing.filter(row => row[0] === 'multimodal-eeg' && row[1] === 'lineTo');
  assert.ok(lines.length <= 500); assert.ok(lines.every(row => row.slice(2).every(Number.isFinite)));
  assert.match(r.el('eeg-status').textContent, /µV/);
  const u = harness(); await preview(u, running({ eeg: { units: 'unknown' } }));
  assert.match(u.el('eeg-status').textContent, /units are unconfirmed/);
  assert.equal(u.drawing.filter(row => row[0] === 'multimodal-eeg' && row[1] === 'lineTo').length, 0);
});

test('aggregate camera privacy is forwarded without retaining sensor frames', () => {
  const r = harness(); const privacy = { instance_id: 'a', revision: 1, state: 'on' };
  r.panel.acceptStatus(running({ camera_privacy: privacy })); assert.deepEqual(r.privacy.at(-1), privacy);
});

test('renderer accepts the real stopped service contract without a legacy top-level arm field', () => {
  // The paired Python test checks this fixture against actual service constructors with devices that reject all hardware access.
  const states = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/multimodal-service-status.json'), 'utf8'));
  for (const state of states) {
    assert.equal(Object.hasOwn(state, 'armed'), false);
    const r = harness(); assert.equal(r.panel.acceptStatus(state), true);
    assert.equal(r.el('start').disabled, false); assert.equal(r.el('eeg-arm').disabled, true);
    assert.equal(r.el('eeg-camera').checked, state.eeg_control.camera_mouse_enabled);
    assert.equal(r.el('start').textContent, state.eeg_control.camera_mouse_enabled ? 'Start EEG + camera' : 'Start EEG only');
    assert.equal(r.el('train-rest').textContent, '0 / 8'); assert.equal(r.el('validate-select').textContent, '0 / 8');
    assert.equal(r.requests.length, 0); assert.equal(r.guards.length, 0);
  }
});

test('manual collection posts fixed labels and server-owned phases, disarming before collection', async () => {
  const r = harness(); r.panel.acceptStatus(collecting()); r.el('eeg-select').dispatch('click');
  assert.deepEqual(action(r, 'eeg_trial').payload, { label: 'select', phase: 'train' }); assert.ok(r.guardStops.length);
  await respond(action(r, 'eeg_trial'), collecting({ eeg_control: { pending: { label: 'select', phase: 'train', stage: 'prepare', prepare_remaining_seconds: 3, remaining_seconds: 1 } } }));
  assert.match(r.el('eeg-trial').textContent, /starts in 3.0 s/); assert.doesNotMatch(r.el('eeg-trial').textContent, /blinks now/);
  assert.equal(r.el('eeg-arm').checked, false);
  const v = harness(); v.panel.acceptStatus(running({ decoder: { validation_counts: { rest: 0, select: 0 } } })); v.el('eeg-rest').dispatch('click');
  assert.deepEqual(action(v, 'eeg_trial').payload, { label: 'rest', phase: 'validate' });
});

test('guided batch uses one backend countdown, alternates only after accepted counts, and waits for samples', async () => {
  const r = harness(); await preview(r, collecting()); r.el('eeg-batch').dispatch('click');
  assert.equal(action(r, 'eeg_trial'), undefined); r.tick(0);
  assert.deepEqual(action(r, 'eeg_trial').payload, { label: 'rest', phase: 'train' });
  await respond(action(r, 'eeg_trial'), collecting({ eeg_control: { pending: { label: 'rest', phase: 'train', stage: 'prepare', prepare_remaining_seconds: 2.5, remaining_seconds: 1, cue: 'Prepare to read normally.' } } }));
  assert.match(r.el('eeg-guided-cue').textContent, /starts in 2.5 s.*Prepare to read/);
  r.panel.acceptStatus(collecting({ eeg_control: { pending: { label: 'rest', phase: 'train', stage: 'record', remaining_seconds: 0 } } }));
  assert.match(r.el('eeg-guided-cue').textContent, /Finishing signal collection/);
  assert.equal(r.requests.filter(q => q.action === 'eeg_trial').length, 1);
  r.panel.acceptStatus(collecting({ decoder: { train_counts: { rest: 1, select: 0 } } })); r.tick(0);
  assert.deepEqual(action(r, 'eeg_trial').payload, { label: 'select', phase: 'train' });
  assert.equal(action(r, 'eeg_train'), undefined); assert.equal(action(r, 'eeg_arm'), undefined);
});

test('recording cue is authoritative and one-second fallback never instructs a long hold', () => {
  const r = harness(); r.panel.acceptStatus(collecting({ eeg_control: { pending: { phase: 'train', label: 'select', stage: 'record', remaining_seconds: .8, cue: 'Blink twice now.' } } }));
  assert.match(r.el('eeg-guided-cue').textContent, /0.8 s.*Blink twice now/);
  r.panel.acceptStatus(collecting({ eeg_control: { pending: { phase: 'train', label: 'select', stage: 'record', remaining_seconds: 1 } } }));
  assert.match(r.el('eeg-trial').textContent, /finishing both within this second/); assert.doesNotMatch(r.el('eeg-trial').textContent, /hold|four|middle/);
});

test('rejected guided trial pauses visibly and never silently retries', async () => {
  const r = harness(); await preview(r, collecting()); r.el('eeg-batch').dispatch('click'); r.tick(0);
  await respond(action(r, 'eeg_trial'), collecting({ eeg_control: { last_result: 'Signal artifact rejected.' } }));
  assert.match(r.el('eeg-batch-status').textContent, /paused.*Signal artifact rejected/);
  assert.equal(r.el('eeg-batch-cancel').disabled, true); assert.equal(r.requests.filter(q => q.action === 'eeg_trial').length, 1);
});

for (const change of ['cancel','hidden','camera','source','session','reset','offline']) test(`queued guided trial cannot survive ${change}`, async () => {
  const r = harness(); await preview(r, collecting()); r.el('eeg-batch').dispatch('click');
  if (change === 'cancel') r.el('eeg-batch-cancel').dispatch('click');
  if (change === 'hidden') r.open(false);
  if (change === 'camera') { r.el('eeg-camera').checked = false; r.el('eeg-camera').dispatch('change'); }
  if (change === 'source') r.panel.acceptStatus(collecting({ eeg_control: { saved_model_source: 'replacement.json' } }));
  if (change === 'session') r.panel.acceptStatus(collecting({ eeg: { session_id: 'session-2' } }));
  if (change === 'reset') r.el('eeg-reset').dispatch('click');
  if (change === 'offline') r.online(false);
  for (const timer of r.timers.filter(t => t.delay === 0)) timer.fn();
  assert.equal(action(r, 'eeg_trial'), undefined); assert.equal(r.el('eeg-batch-cancel').disabled, true);
});

test('guided quotas do not freeze or validate automatically and manual extra training remains possible', async () => {
  const r = harness(); await preview(r, collecting({ decoder: { train_counts: { rest: 8, select: 7 } } }));
  r.el('eeg-batch').dispatch('click'); r.tick(0); assert.equal(action(r, 'eeg_trial').payload.label, 'select');
  await respond(action(r, 'eeg_trial'), collecting({ decoder: { train_counts: { rest: 8, select: 8 } } }));
  assert.match(r.el('eeg-batch-status').textContent, /Training collection complete/);
  assert.equal(r.el('eeg-batch').disabled, true); assert.equal(r.el('eeg-train').disabled, false);
  assert.equal(r.el('eeg-rest').disabled, false); assert.equal(r.el('eeg-select').disabled, false);
  assert.match(r.el('eeg-rest').textContent, /another/); assert.equal(action(r, 'eeg_train'), undefined);
});

test('training cap, malformed count data, frozen validation quota, and loaded archives block extra collection', () => {
  const r = harness();
  for (const data of [collecting({ decoder: { train_counts: { rest: 24, select: 24 } } }),
    collecting({ decoder: { train_counts: { rest: NaN, select: 8 } } }), running(),
    collecting({ decoder: { inference_context_bound: true } })]) {
    r.panel.acceptStatus(data); assert.equal(r.el('eeg-rest').disabled, true); assert.equal(r.el('eeg-select').disabled, true);
  }
  r.panel.acceptStatus(collecting({ decoder: { train_counts: { rest: 20, select: 20 }, max_training_trials: 40 } }));
  assert.equal(r.el('eeg-rest').disabled, true);
});

test('save retains active guard; load and reset explicitly disarm without rearming', async () => {
  const r = harness(); await arm(r); const before = r.guardStops.length;
  r.el('eeg-save').dispatch('click'); assert.equal(r.guardStops.length, before);
  await respond(action(r, 'eeg_save'), running({ eeg_control: { armed: true, saved_model_file: 'local.json' } }));
  assert.equal(r.el('eeg-arm').checked, true); assert.match(r.el('eeg-saved-status').textContent, /local.json/);
  r.el('eeg-load-saved').dispatch('click'); assert.ok(r.guardStops.length > before); assert.equal(r.el('eeg-arm').checked, false);
  await respond(action(r, 'eeg_load_saved'), running({ decoder: { passed_validation: false }, eeg_control: { saved_model_source: 'failed.json' } }));
  assert.equal(r.el('eeg-arm').disabled, true); assert.equal(r.el('eeg-arm').checked, false);
  r.el('eeg-reset').dispatch('click'); assert.ok(action(r, 'eeg_reset')); assert.equal(r.requests.filter(q => q.action === 'eeg_arm').length, 1);
});

test('Escape cancels pending work; offline reconnect never replays an operation', async () => {
  const r = harness(); r.panel.acceptStatus(running()); r.document.dispatch('keydown', { key: 'Escape' });
  assert.equal(action(r, 'stop').action, 'stop'); await respond(action(r, 'stop'), stopped());
  r.online(false); r.open(true); r.document.dispatch('keydown', { key: 'Escape' }); r.online(true);
  assert.ok(r.requests.slice(1).every(q => ['status','preview'].includes(q.action)));
  assert.equal(r.el('eeg-arm').checked, false);
});

test('native emergency stop invalidates in-flight reads and clears local frames and lease', async () => {
  const r = harness(); await arm(r); r.open(true); const read = action(r, 'preview');
  r.panel.emergencyStop('escape'); await respond(read, running({ eeg_control: { armed: true } }));
  assert.equal(r.el('eeg-arm').checked, false); assert.match(r.el('error').textContent, /Emergency stop requested/);
  assert.equal(r.images.length, 0);
});
