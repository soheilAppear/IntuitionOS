// Synthetic samples and mock DOM: never opens hardware, Electron, or a socket.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createMultimodalPanel } = require('../ui/renderer/multimodal-panel.cjs');

function harness() {
  const elements = new Map(), timers = [], requests = [], images = [], drawing = [];
  class Element {
    constructor(id) { this.id = id; this.textContent = ''; this.open = false; this.listeners = {}; this.attributes = {}; this.width = 480; this.height = 360; }
    addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
    dispatch(name) { for (const fn of this.listeners[name] || []) fn({ stopPropagation() {} }); }
    setAttribute(key, value) { this.attributes[key] = value; }
    getContext() { return new Proxy({}, { get: (_target, name) => (...args) => drawing.push([this.id, name, ...args]), set: () => true }); }
  }
  const document = new Element('document');
  document.getElementById = id => { if (!elements.has(id)) elements.set(id, new Element(id)); return elements.get(id); };
  let online = true, handCamera = false, brainbit = true, time = 0;
  const panel = createMultimodalPanel({ document, isConnected: () => online,
    isCameraRunning: () => handCamera, isBrainbitConnected: () => brainbit,
    request: (action, payload) => new Promise((resolve, reject) => requests.push({ action, payload, resolve, reject })),
    imageFactory: () => { const picture = {}; images.push(picture); return picture; },
    schedule: (fn, delay) => { timers.push({ fn, delay }); return timers.length; },
    cancel: id => { if (timers[id - 1]) timers[id - 1].cancelled = true; }, now: () => time,
  });
  panel.connectionChanged();
  const el = id => document.getElementById(`multimodal-${id}`);
  const open = value => { el('panel').open = value; el('panel').dispatch('toggle'); };
  return { panel, document, requests, images, drawing, timers, el, open,
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
  assert.match(markup, /EEG is not used to choose or trigger/);
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
