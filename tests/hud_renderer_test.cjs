// Renderer behavior without Electron installation, devices or a running backend.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function renderer({ connected = true } = {}) {
  const drawing = [];
  const canvasContext = new Proxy({}, { get: (_target, name) => (...args) => drawing.push([name, ...args]),
    set: () => true });
  class Element {
    constructor(tag = 'div') {
      this.tagName = tag;
      this.value = '';
      this.textContent = '';
      this.innerHTML = '';
      this.children = [];
      this.listeners = {};
      this.style = {};
      this.attributes = {};
      const classes = new Set();
      this.classList = {
        add: (...names) => names.forEach(n => classes.add(n)),
        remove: (...names) => names.forEach(n => classes.delete(n)),
        contains: name => classes.has(name),
        toggle: (name, force = !classes.has(name)) => force ? classes.add(name) : classes.delete(name),
      };
    }
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
    dispatch(type, args = {}) { for (const fn of this.listeners[type] || []) fn({ preventDefault() {}, ...args }); }
    appendChild(child) { this.children.push(child); return child; }
    replaceChildren(...children) { this.children = children; }
    setAttribute(key, value) { this.attributes[key] = value; }
    getContext() { return canvasContext; }
    focus() {}
    remove() {}
  }
  const elements = new Map();
  const document = new Element('document');
  document.getElementById = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  document.createElement = tag => new Element(tag);
  document.createTextNode = text => Object.assign(new Element('#text'), { textContent: text });
  const sent = [];
  const cameraRequests = [];
  const brainbitRequests = [];
  const sockets = [];
  const timers = [];
  const ipcHandlers = {};
  const ipcSent = [];
  const images = [];
  class Image { constructor() { images.push(this); } }
  class WebSocket {
    static OPEN = 1;
    constructor() { this.readyState = 0; sockets.push(this); }
    open() { this.readyState = WebSocket.OPEN; this.onopen?.(); }
    send(value) {
      if (this.readyState !== WebSocket.OPEN || this.failSend) throw new Error('socket disconnected');
      sent.push(JSON.parse(value));
    }
    close() { this.readyState = 3; this.onclose?.(); }
  }
  const context = vm.createContext({
    document, WebSocket, Image,
    require: name => name === 'electron'
      ? { ipcRenderer: { on: (name, callback) => { ipcHandlers[name] = callback; }, send: (...args) => ipcSent.push(args) } }
      : name === './camera-preview.cjs' ? { readCameraPreview: () => new Promise((resolve, reject) => {
        cameraRequests.push({ url: 'http://127.0.0.1:7432/gestures/preview', method: 'GET',
          resolve: async response => response.ok ? resolve(await response.json()) : reject(new Error('Unavailable')), reject });
      }) }
      : name === './brainbit-http.cjs' ? { requestBrainbit: (action, deviceId) => new Promise((resolve, reject) => {
        brainbitRequests.push({ action, deviceId, resolve, reject });
      }) }
      : name === './multimodal-panel.cjs' ? { createMultimodalPanel: () => ({ connectionChanged() {}, acceptStatus() {}, refreshControls() {} }) }
      : require(name),
    setTimeout: (callback, delay) => { timers.push({ callback, delay }); return timers.length; },
    clearTimeout: id => { if (timers[id - 1]) timers[id - 1].cancelled = true; }, console,
    AbortSignal,
    fetch: (url, options) => new Promise((resolve, reject) => {
      cameraRequests.push({ url, method: options.method, body: options.body ? JSON.parse(options.body) : undefined, resolve, reject });
    }),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../ui/renderer/app.js'), 'utf8'), context);
  if (connected) {
    sockets[0].open();
    sent.length = 0;
  }
  const input = document.getElementById('cmd-input');
  const change = text => { input.value = text; input.dispatch('input'); };
  const key = (key, extra = {}) => input.dispatch('keydown', { key, ...extra });
  const message = value => vm.runInContext(`handleMessage(${JSON.stringify(value)})`, context);
  const show = (original, candidates, revision = 1) => message({
    type: 'resolution', original, candidates, token: 'visible-token', revision,
    client_revision: vm.runInContext('inputRevision', context), status: candidates.length ? 'correction' : 'exact',
  });
  const reconnect = () => {
    const timer = timers.find(timer => timer.delay === 2500 && !timer.cancelled && !timer.ran);
    assert.ok(timer, 'disconnect must schedule a reconnect');
    timer.ran = true;
    timer.callback();
    sockets.at(-1).open();
  };
  return {
    elements, sent, context, input, change, key, message, show, reconnect, sockets, cameraRequests, brainbitRequests, images, drawing, timers,
    voiceToggle: () => ipcHandlers['voice-toggle'](),
    nativeMessage: (name, value) => ipcHandlers[name]({}, value), ipcSent,
    get socket() { return sockets.at(-1); },
  };
}

function textOf(element) { return element.textContent + element.children.map(textOf).join(''); }

function openCameraPreview(r) {
  r.message({ type: 'gesture_status', running: true, state: 'running', available: true });
  r.elements.get('gesture-guide').open = true;
  r.elements.get('gesture-preview-toggle').checked = true;
  r.elements.get('gesture-preview-toggle').dispatch('change');
}

function previewFrame(extra = {}) {
  return { running: true, image: 'data:image/jpeg;base64,YWJj', width: 480, height: 360,
    age_ms: 10, tracked: true, raw_pose: 'open_palm', effective_pose: 'open_palm',
    landmarks: Array.from({ length: 21 }, (_, i) => [i / 30, 0.5, 0]),
    fps: 25, model_name: 'MediaPipe Hands Full', state: 'desktop', progress: -0.6,
    hint: 'Next desktop; make a fist to release', last_cancel_reason: '', ...extra };
}

test('preview is opt in and does not start or double-open a camera', async () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { running: false, state: 'off', available: true } });
  r.elements.get('gesture-guide').open = true;
  r.elements.get('gesture-preview-toggle').checked = true;
  r.elements.get('gesture-preview-toggle').dispatch('change');
  assert.equal(r.cameraRequests.length, 0);
  assert.match(r.elements.get('gesture-preview-pose').textContent, /Turn on the camera/);
  openCameraPreview(r);
  assert.equal(r.cameraRequests.length, 1);
  assert.equal(r.cameraRequests[0].method, 'GET');
  assert.equal(r.cameraRequests[0].url, 'http://127.0.0.1:7432/gestures/preview');
  r.elements.get('gesture-preview-toggle').dispatch('change');
  assert.equal(r.cameraRequests.length, 1);
  r.cameraRequests[0].resolve({ ok: true, json: async () => previewFrame() });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(r.images.length, 1);
  r.images[0].onload();
  assert.match(r.elements.get('gesture-preview-hint').textContent, /60%/);
  assert.match(r.elements.get('gesture-preview-model').textContent, /25 tracking FPS/);
  assert.equal(r.drawing.filter(call => call[0] === 'arc').length, 21);
  assert.equal(r.drawing.filter(call => call[0] === 'drawImage').length, 1);
});

for (const stop of ['camera', 'disconnect', 'close guide', 'uncheck']) {
  test(`preview discards a pending frame after ${stop}`, async () => {
    const r = renderer(); openCameraPreview(r);
    r.cameraRequests[0].resolve({ ok: true, json: async () => previewFrame() });
    await new Promise(resolve => setImmediate(resolve));
    if (stop === 'camera') r.message({ type: 'gesture_status', running: false, state: 'off', available: true });
    if (stop === 'disconnect') r.socket.close();
    if (stop === 'close guide') { r.elements.get('gesture-guide').open = false; r.elements.get('gesture-guide').dispatch('toggle'); }
    if (stop === 'uncheck') { r.elements.get('gesture-preview-toggle').checked = false; r.elements.get('gesture-preview-toggle').dispatch('change'); }
    r.images[0].onload();
    assert.equal(r.drawing.filter(call => call[0] === 'drawImage').length, 0);
    assert.equal(vm.runInContext('gesturePreviewActive', r.context), false);
    assert.equal(r.elements.get('gesture-preview-model').textContent, '');
  });
}

test('preview clears stale video and safely reports unknown visible finger poses', async () => {
  const r = renderer(); openCameraPreview(r);
  r.cameraRequests[0].resolve({ ok: true, json: async () => previewFrame({ age_ms: 900 }) });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(r.images.length, 0);
  assert.match(r.elements.get('gesture-preview-pose').textContent, /fresh camera frame/);
  const snapshot = previewFrame({ raw_pose: 'unknown', last_cancel_reason: 'pose_unclear', hint: '<Keep palm open>' });
  vm.runInContext(`renderGesturePreview(${JSON.stringify(snapshot)}, gesturePreviewEpoch)`, r.context);
  r.images[0].onload();
  assert.equal(r.elements.get('gesture-preview-pose').textContent, 'Pose unclear · holding swipe');
  assert.match(r.elements.get('gesture-preview-reason').textContent, /Finger pose/);
  assert.match(r.elements.get('gesture-preview-hint').textContent, /<Keep palm open>/);
  assert.equal(r.elements.get('gesture-preview-hint').innerHTML, '');
  const staleTimer = r.timers.findLast(timer => timer.delay > 500 && timer.delay <= 700);
  staleTimer.callback();
  assert.match(r.elements.get('gesture-preview-pose').textContent, /fresh camera frame/);
});

test('preview ignores an old request after closing and reopening the guide', async () => {
  const r = renderer(); openCameraPreview(r);
  r.elements.get('gesture-guide').open = false;
  r.elements.get('gesture-guide').dispatch('toggle');
  r.elements.get('gesture-guide').open = true;
  r.elements.get('gesture-guide').dispatch('toggle');
  assert.equal(r.cameraRequests.length, 1, 'only one preview request at a time');
  r.cameraRequests[0].resolve({ ok: true, json: async () => previewFrame() });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(r.images.length, 0);
  r.timers.findLast(timer => timer.delay === 125 && !timer.cancelled).callback();
  assert.equal(r.cameraRequests.length, 2);
});

test('mouse preview marks the pointer area and displays movement instructions', async () => {
  const r = renderer(); openCameraPreview(r);
  r.cameraRequests[0].resolve({ ok: true, json: async () => previewFrame({
    input_mode: 'mouse', control_region: [0.15, 0.12, 0.85, 0.82],
    state: 'mouse_pointer', raw_pose: 'point', effective_pose: 'point',
    hint: 'Point to move; pinch to click or hold to drag', progress: 1,
  }) });
  await new Promise(resolve => setImmediate(resolve));
  r.images[0].onload();
  assert.equal(r.drawing.filter(call => call[0] === 'strokeRect').length, 1);
  assert.equal(r.elements.get('gesture-preview-pose').textContent, 'Pointer active');
  assert.match(r.elements.get('gesture-preview-hint').textContent, /Point to move/);
  assert.doesNotMatch(r.elements.get('gesture-preview-hint').textContent, /100%/);
});

test('mouse navigation preview names four fingers and reports vertical travel', async () => {
  const r = renderer(); openCameraPreview(r);
  r.cameraRequests[0].resolve({ ok: true, json: async () => previewFrame({
    input_mode: 'mouse', state: 'overview', raw_pose: 'four_finger', effective_pose: 'four_finger',
    hint: 'Task View', progress: -0.7,
  }) });
  await new Promise(resolve => setImmediate(resolve));
  r.images[0].onload();
  assert.equal(r.elements.get('gesture-preview-pose').textContent, 'Four fingers');
  assert.equal(r.elements.get('gesture-preview-hint').textContent, 'Task View · 70%');
});

test('all-desktop status is independent of camera state and retries through the native shell', () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { running: false, state: 'off', available: true } });
  r.change('preserve my draft');
  r.nativeMessage('desktop-visibility', { state: 'pinned', pinned: true, text: 'HUD visible on all virtual desktops.' });
  assert.equal(r.elements.get('hud-desktop-indicator').textContent, 'All desktops');
  assert.equal(r.elements.get('desktop-visibility-warning').hidden, true);
  r.nativeMessage('desktop-visibility', { state: 'unavailable', pinned: false, text: 'Could not pin <HUD>. Retry.' });
  assert.equal(r.elements.get('desktop-visibility-warning').hidden, false);
  assert.equal(r.elements.get('desktop-visibility-text').textContent, 'Could not pin <HUD>. Retry.');
  assert.equal(r.elements.get('desktop-visibility-text').innerHTML, '');
  r.elements.get('desktop-visibility-retry').dispatch('click');
  assert.deepEqual(r.ipcSent.at(-1), ['desktop-visibility-retry']);
  assert.equal(r.input.value, 'preserve my draft');
  assert.equal(r.cameraRequests.length, 0);
});

test('signed hand progress reverses continuously without touching a command draft', () => {
  const r = renderer();
  r.change('keep this draft');
  r.message({ type: 'status', gestures: { running: true, state: 'running', available: true } });
  r.message({ type: 'gesture_progress', state: 'desktop', progress: -0.8, text: 'Next desktop' });
  assert.equal(r.elements.get('gesture-meter-fill').style.left, '10%');
  assert.equal(r.elements.get('gesture-meter-fill').style.width, '40%');
  assert.match(r.elements.get('gesture-progress-text').textContent, /80%/);
  r.message({ type: 'gesture_progress', state: 'desktop', progress: -0.2, text: 'Next desktop' });
  assert.equal(r.elements.get('gesture-meter-fill').style.left, '40%');
  assert.equal(r.elements.get('gesture-meter-fill').style.width, '10%');
  r.message({ type: 'gesture_progress', state: 'cancelled', progress: 0 });
  assert.equal(r.elements.get('gesture-motion').hidden, false);
  assert.equal(r.elements.get('gesture-meter').hidden, true);
  assert.equal(r.input.value, 'keep this draft');
  assert.ok(!r.sent.some(m => ['input', 'confirm'].includes(m.type)));
});

for (const [state, axis, direction, sign] of [
  ['desktop', 'horizontal', 'left', -1], ['overview', 'vertical', 'up', -1],
  ['overview', 'vertical', 'down', 1],
]) test(`mouse navigation keeps its signed meter and reverses the ${direction} commit threshold`, () => {
  const r = renderer();
  r.change('keep this draft');
  r.message({ type: 'gesture_status', running: true, state: 'running', input_mode: 'mouse' });
  const progress = (value, phase = state, extra = {}) => r.message({ type: 'gesture_progress', input_mode: 'mouse',
    gesture: 'hand_navigation', state: phase, axis, direction, progress: value, text: 'Navigation', ...extra });
  const meter = r.elements.get('gesture-meter');
  const text = r.elements.get('gesture-progress-text');
  progress(sign * 0.8);
  assert.equal(meter.hidden, false);
  assert.equal(meter.attributes['aria-valuemin'], '-100');
  assert.equal(meter.attributes['aria-valuenow'], String(sign * 80));
  assert.equal(r.elements.get('gesture-meter-fill').style.left, sign < 0 ? '10%' : '50%');
  assert.equal(r.elements.get('gesture-meter-fill').style.width, '40%');
  assert.match(text.textContent, /80%.*Reach 100% and hold briefly/);
  progress(sign, 'committing', { commit_progress: 0, text: 'Hold briefly to finish' });
  assert.equal(meter.classList.contains('commit-ready'), true);
  assert.match(text.textContent, /Hold briefly to finish.*100%/);
  progress(sign * 0.95, 'committing', { commit_progress: 0.5, text: 'Hold briefly to finish' });
  assert.equal(meter.classList.contains('commit-ready'), true, 'completion dwell tolerates small tremors');
  assert.equal(meter.attributes['aria-valuenow'], String(sign * 95));
  progress(sign * 0.95, 'uncertain', { text: 'Tracking paused; keep your hand visible' });
  assert.equal(meter.classList.contains('directional'), true);
  assert.equal(meter.classList.contains('commit-ready'), false, 'tracking loss cannot promise completion');
  assert.equal(meter.attributes['aria-valuemin'], '-100');
  assert.equal(meter.attributes['aria-valuenow'], String(sign * 95));
  progress(sign * 0.2);
  assert.equal(meter.classList.contains('commit-ready'), false);
  assert.match(text.textContent, /20%.*Reach 100% and hold briefly/);
  assert.doesNotMatch(text.textContent, /fist to finish/);
  assert.equal(r.elements.get('gesture-meter-fill').style.width, '10%');
  r.message({ type: 'gesture_progress', input_mode: 'mouse', gesture: 'hand_navigation',
    state: 'cancelled', progress: 0, text: 'Swipe cancelled; lower your hand or make a fist' });
  assert.equal(r.elements.get('gesture-motion').hidden, false);
  assert.equal(meter.hidden, true);
  assert.equal(text.textContent, 'Swipe cancelled; lower your hand or make a fist');
  assert.equal(r.input.value, 'keep this draft');
  assert.equal(r.cameraRequests.length, 0);
  assert.ok(!r.sent.some(m => ['input', 'confirm'].includes(m.type)));
});

test('navigation arming in mouse mode shows its hold meter without promising a commit', () => {
  const r = renderer();
  r.message({ type: 'gesture_status', running: true, state: 'running', input_mode: 'mouse' });
  r.message({ type: 'gesture_progress', input_mode: 'mouse', gesture: 'hand_navigation',
    state: 'arming', progress: 1, text: 'Hold four fingers still' });
  assert.equal(r.elements.get('gesture-meter').hidden, false);
  assert.equal(r.elements.get('gesture-meter').classList.contains('commit-ready'), false);
  assert.doesNotMatch(r.elements.get('gesture-progress-text').textContent, /finish|Reach 100%/);
  r.message({ type: 'gesture_progress', input_mode: 'mouse', state: 'mouse_pointer', progress: 1,
    text: 'Move your hand' });
  assert.equal(r.elements.get('gesture-meter').hidden, true);
});

test('navigation completion keeps the rearm hint without a live movement meter or input action', () => {
  const r = renderer();
  r.change('keep this draft');
  r.message({ type: 'gesture_status', running: true, state: 'running', input_mode: 'mouse' });
  r.message({ type: 'gesture_progress', input_mode: 'mouse', gesture: 'hand_navigation',
    state: 'committing', axis: 'horizontal', progress: -1, commit_progress: 0.5,
    text: 'Hold briefly to finish' });
  r.message({ type: 'gesture_progress', input_mode: 'mouse', gesture: 'hand_navigation',
    state: 'completed', axis: 'horizontal', progress: -1,
    text: 'Gesture complete; lower your hand or make a fist before another swipe' });
  assert.equal(r.elements.get('gesture-motion').hidden, false);
  assert.equal(r.elements.get('gesture-meter').hidden, true);
  assert.equal(r.elements.get('gesture-meter').classList.contains('commit-ready'), false);
  assert.equal(r.elements.get('gesture-progress-text').textContent,
    'Gesture complete; lower your hand or make a fist before another swipe');
  assert.equal(r.input.value, 'keep this draft');
  assert.equal(r.cameraRequests.length, 0);
  assert.ok(!r.sent.some(m => ['input', 'confirm'].includes(m.type)));
  r.message({ type: 'gesture_progress', input_mode: 'mouse', gesture: 'hand_navigation',
    state: 'error', progress: 0, text: 'Navigation input could not release; turn the camera off to retry' });
  assert.equal(r.elements.get('gesture-meter').hidden, true);
  assert.equal(r.elements.get('gesture-motion').hidden, false);
  assert.match(r.elements.get('gesture-progress-text').textContent, /could not release/);
  assert.doesNotMatch(r.elements.get('gesture-progress-text').textContent, /%/);
  r.message({ type: 'gesture_progress', input_mode: 'mouse', gesture: 'hand_navigation',
    state: 'idle', progress: 0, text: 'Ready' });
  assert.equal(r.elements.get('gesture-motion').hidden, true);
});

test('navigation controls are available in idle mouse mode and blocked throughout live capture', () => {
  const r = renderer();
  const controls = ['gesture-desktop-mode', 'gesture-travel'];
  for (const id of controls) assert.equal(r.elements.get(id).disabled, true);
  r.message({ type: 'gesture_status', running: false, state: 'off', input_mode: 'mouse',
    settings: { desktop_mode: 'auto', travel_palms: 1.8 } });
  for (const id of controls) assert.equal(r.elements.get(id).disabled, false);
  assert.equal(r.elements.get('gesture-mode').hidden, false);
  for (const state of ['starting', 'running', 'stopping']) {
    r.message({ type: 'gesture_status', running: state === 'running', state, input_mode: 'mouse' });
    for (const id of controls) assert.equal(r.elements.get(id).disabled, true, `${id} during ${state}`);
  }
  r.message({ type: 'gesture_status', running: false, state: 'off', input_mode: 'mouse',
    tracker_cleanup_pending: true });
  for (const id of controls) assert.equal(r.elements.get(id).disabled, true);
  r.message({ type: 'gesture_status', running: false, state: 'off', input_mode: 'mouse' });
  for (const id of controls) assert.equal(r.elements.get(id).disabled, false);
  r.elements.get('gesture-save-settings').dispatch('click');
  for (const id of controls) assert.equal(r.elements.get(id).disabled, true, `${id} while applying`);
});

test('hand close prompt safely displays the target and stays separate from command approval', () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { running: true, state: 'running', available: true } });
  r.message({ type: 'confirm_request', token: 'command-token', capability: 'typed_action' });
  r.message({ type: 'gesture_close', pending: true, title: '<Document & notes>', remaining_s: 6 });
  assert.match(r.elements.get('gesture-close').textContent, /<Document & notes>/);
  assert.equal(r.elements.get('gesture-close').innerHTML, '');
  assert.equal(r.elements.get('gesture-close').hidden, false);
  r.message({ type: 'gesture_close', pending: false, text: 'Close cancelled.' });
  assert.equal(r.elements.get('gesture-close').hidden, true);
  assert.equal(vm.runInContext('pendingConfirm.token', r.context), 'command-token');
  assert.ok(!r.sent.some(m => m.type === 'confirm'));
});

test('camera stop and disconnect clear hand progress and approval without replay', () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { running: true, state: 'running', available: true } });
  r.message({ type: 'gesture_close', pending: true, title: 'Document', remaining_s: 6 });
  r.message({ type: 'gesture_progress', state: 'holding', progress: 0.8, text: 'Hold thumbs-up' });
  r.socket.close();
  assert.equal(r.elements.get('gesture-motion').hidden, true);
  assert.equal(r.elements.get('gesture-close').hidden, true);
  r.message({ type: 'gesture_progress', state: 'holding', progress: 1 });
  assert.equal(r.elements.get('gesture-motion').hidden, true);
  assert.ok(!r.sent.some(m => m.type === 'confirm'));
});

test('hand settings apply independently while idle and cannot change live tracking', async () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { running: false, state: 'off', available: true,
    settings: { desktop_mode: 'auto', travel_palms: 1.8 } } });
  r.elements.get('gesture-desktop-mode').value = 'shortcut';
  r.elements.get('gesture-travel').value = '2.2';
  r.elements.get('gesture-save-settings').dispatch('click');
  assert.equal(r.cameraRequests.length, 1);
  assert.equal(r.cameraRequests[0].url, 'http://127.0.0.1:7432/gestures/settings');
  assert.deepEqual(r.cameraRequests[0].body, { desktop_mode: 'shortcut', travel_palms: 2.2,
    model_complexity: 1, tracker_backend: 'mediapipe', bend_click: false });
  r.elements.get('gesture-toggle').dispatch('click');
  assert.equal(r.cameraRequests.length, 1);
  r.cameraRequests[0].resolve({ ok: true, json: async () => ({ gestures: {
    running: false, state: 'off', available: true,
    settings: { desktop_mode: 'shortcut', travel_palms: 2.2 },
  } }) });
  await new Promise(resolve => setImmediate(resolve));
  assert.match(r.elements.get('gesture-settings-status').textContent, /Applied/);
  r.message({ type: 'gesture_status', running: true, state: 'running', available: true });
  assert.equal(r.elements.get('gesture-save-settings').disabled, true);
  r.elements.get('gesture-save-settings').dispatch('click');
  assert.equal(r.cameraRequests.length, 1);
});

for (const backend of ['rtmpose', 'wilor']) {
test(`${backend} selection preserves the MediaPipe tier and never starts the camera`, async () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { running: false, state: 'off', available: true,
    settings: { desktop_mode: 'auto', travel_palms: 1.8, model_complexity: 0 } } });
  assert.equal(r.elements.get('gesture-model').value, '0');
  r.elements.get('gesture-model').value = backend;
  r.elements.get('gesture-save-settings').dispatch('click');
  assert.deepEqual(r.cameraRequests[0].body, { desktop_mode: 'auto', travel_palms: 1.8,
    model_complexity: 0, tracker_backend: backend, bend_click: false });
  r.cameraRequests[0].resolve({ ok: true, json: async () => ({ gestures: {
    running: false, state: 'off', available: true, tracker_backend: backend, model_complexity: 0,
    settings: { desktop_mode: 'auto', travel_palms: 1.8, model_complexity: 0, tracker_backend: backend },
  } }) });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(r.elements.get('gesture-model').value, backend);
  assert.equal(r.elements.get('gesture-label').textContent, 'CAMERA OFF');
  assert.equal(r.cameraRequests.length, 1);
  assert.ok(!r.sent.some(m => m.type === 'input'));
  r.elements.get('gesture-model').value = '1';
  r.elements.get('gesture-save-settings').dispatch('click');
  assert.equal(r.cameraRequests[1].body.tracker_backend, 'mediapipe');
  assert.equal(r.cameraRequests[1].body.model_complexity, 1);
});

test(`${backend} selector uses acknowledged backend and defaults legacy messages to MediaPipe`, () => {
  const r = renderer();
  r.message({ type: 'gesture_status', running: false, state: 'off', available: true,
    settings: { tracker_backend: backend, model_complexity: 1 } });
  assert.equal(r.elements.get('gesture-model').value, backend);
  r.message({ type: 'gesture_status', running: false, state: 'off', available: true,
    settings: { model_complexity: 0 } });
  assert.equal(r.elements.get('gesture-model').value, '0');
  r.message({ type: 'gesture_status', running: true, state: 'running', available: true,
    tracker_backend: backend, model_complexity: 1 });
  assert.equal(r.elements.get('gesture-model').value, backend);
  assert.equal(r.elements.get('gesture-model').disabled, true);
  r.elements.get('gesture-save-settings').dispatch('click');
  assert.equal(r.cameraRequests.length, 0);
});
}

test('WiLoR preview names the selected model without falling back to another tracker label', async () => {
  const r = renderer(); openCameraPreview(r);
  r.cameraRequests[0].resolve({ ok: true, json: async () => previewFrame({
    tracker_backend: 'wilor', model_name: 'WiLoR + AnyHand · GPU',
  }) });
  await new Promise(resolve => setImmediate(resolve));
  r.images[0].onload();
  assert.equal(r.elements.get('gesture-preview-model').textContent,
    'WiLoR + AnyHand · GPU · 25 tracking FPS');
});

test('preview displays the actual tracker model reported by the backend', async () => {
  const r = renderer(); openCameraPreview(r);
  r.cameraRequests[0].resolve({ ok: true, json: async () => previewFrame({
    tracker_backend: 'rtmpose', model_name: 'RTMPose Hand5 · CPU',
  }) });
  await new Promise(resolve => setImmediate(resolve));
  r.images[0].onload();
  assert.equal(r.elements.get('gesture-preview-model').textContent,
    'RTMPose Hand5 · CPU · 25 tracking FPS');
});

test('Hand Mouse mode is selected explicitly and waits for backend acknowledgement', async () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { running: false, state: 'off', available: true,
    input_mode: 'desktop', settings: { desktop_mode: 'auto', travel_palms: 1.8, model_complexity: 1 } } });
  r.change('keep my draft');
  r.elements.get('hand-mode-mouse').dispatch('click');
  assert.equal(r.cameraRequests.length, 1);
  assert.equal(r.cameraRequests[0].body.input_mode, 'mouse');
  assert.equal(r.elements.get('hand-mode-mouse').attributes['aria-pressed'], 'false');
  assert.equal(r.elements.get('hand-mode-desktop').disabled, true);
  r.elements.get('gesture-toggle').dispatch('click');
  assert.equal(r.cameraRequests.length, 1, 'camera cannot start midway through changing input mode');
  r.cameraRequests[0].resolve({ ok: true, json: async () => ({ gestures: {
    running: false, state: 'off', available: true, input_mode: 'mouse',
    settings: { desktop_mode: 'auto', travel_palms: 1.8, model_complexity: 1, input_mode: 'mouse' },
  } }) });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(r.elements.get('hand-mode-mouse').attributes['aria-pressed'], 'true');
  assert.equal(r.elements.get('mouse-gesture-guide').hidden, false);
  assert.equal(r.elements.get('desktop-gesture-guide').hidden, true);
  assert.match(r.elements.get('gesture-help').textContent, /relax your fingers/);
  assert.match(r.elements.get('gesture-help').textContent, /Fully open four fingers/);
  assert.match(r.elements.get('gesture-guide-intro').textContent, /Fully open four fingers for navigation/);
  assert.match(r.elements.get('gesture-help').textContent, /Pinch and release to click/);
  assert.equal(r.elements.get('mouse-bend-guide').hidden, true);
  assert.match(r.elements.get('hand-mode-status').textContent, /point to move/);
  assert.equal(r.input.value, 'keep my draft');
  r.message({ type: 'gesture_status', running: true, state: 'running', input_mode: 'mouse' });
  r.message({ type: 'gesture_progress', input_mode: 'mouse', state: 'mouse_pointer', progress: 1, text: 'Point to move' });
  assert.equal(r.elements.get('gesture-meter').hidden, true);
  assert.equal(r.elements.get('gesture-progress-text').textContent, 'Point to move');
  r.message({ type: 'gesture_progress', input_mode: 'mouse', state: 'mouse_pinch', progress: 0.5, text: 'Hold to drag' });
  assert.equal(r.elements.get('gesture-meter').hidden, false);
  assert.match(r.elements.get('gesture-progress-text').textContent, /50%/);
  assert.equal(r.elements.get('hand-mode-desktop').disabled, true);
  r.elements.get('hand-mode-desktop').dispatch('click');
  assert.equal(r.cameraRequests.length, 1);
});

test('index bend click is opt-in and its guide follows backend acknowledgement', async () => {
  const r = renderer();
  const status = { running: false, state: 'off', available: true, input_mode: 'mouse',
    settings: { desktop_mode: 'auto', travel_palms: 1.8, bend_click: false } };
  r.message({ type: 'gesture_status', ...status });
  const checkbox = r.elements.get('gesture-bend-click');
  assert.equal(checkbox.checked, false);
  assert.equal(checkbox.disabled, false);
  assert.equal(r.elements.get('mouse-bend-guide').hidden, true);
  checkbox.checked = true;
  r.elements.get('gesture-save-settings').dispatch('click');
  assert.equal(r.cameraRequests[0].body.bend_click, true);
  assert.equal(checkbox.disabled, true);
  assert.equal(r.elements.get('mouse-bend-guide').hidden, true, 'a draft setting must not change active instructions');
  r.cameraRequests[0].resolve({ ok: true, json: async () => ({ gestures: {
    ...status, settings: { ...status.settings, bend_click: true },
  } }) });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(checkbox.checked, true);
  assert.equal(checkbox.disabled, false);
  assert.equal(r.elements.get('mouse-bend-guide').hidden, false);
  assert.match(r.elements.get('gesture-help').textContent, /Index bend click is also enabled/);
  assert.equal(r.cameraRequests.length, 1, 'applying a gesture option never starts the camera');
  r.message({ type: 'gesture_status', ...status, bend_click: false,
    settings: { ...status.settings, bend_click: true } });
  assert.equal(checkbox.checked, false, 'top-level acknowledged value takes precedence');
  assert.equal(r.elements.get('mouse-bend-guide').hidden, true);
  assert.doesNotMatch(r.elements.get('gesture-help').textContent, /Index bend/);
});

test('index bend setting is disabled outside idle mouse mode and legacy status defaults it off', () => {
  const r = renderer();
  const checkbox = r.elements.get('gesture-bend-click');
  assert.equal(checkbox.disabled, true);
  r.message({ type: 'gesture_status', running: false, state: 'off', input_mode: 'desktop' });
  assert.equal(checkbox.checked, false);
  assert.equal(checkbox.disabled, true);
  r.message({ type: 'gesture_status', running: false, state: 'off', input_mode: 'mouse', bend_click: true });
  assert.equal(checkbox.checked, true);
  assert.equal(checkbox.disabled, false);
  for (const state of ['starting', 'running', 'stopping']) {
    r.message({ type: 'gesture_status', running: state === 'running', state,
      input_mode: 'mouse', bend_click: true });
    assert.equal(checkbox.disabled, true);
    r.elements.get('gesture-save-settings').dispatch('click');
    assert.equal(r.cameraRequests.length, 0);
  }
  r.message({ type: 'gesture_status', running: false, state: 'off', input_mode: 'mouse' });
  assert.equal(checkbox.checked, false);
  assert.equal(r.elements.get('mouse-bend-guide').hidden, true);
  r.socket.close();
  assert.equal(checkbox.disabled, true);
});

test('failed mouse mode selection stays on desktop and exposes the error outside the guide', async () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { running: false, state: 'off', available: true,
    input_mode: 'desktop', settings: { desktop_mode: 'auto', travel_palms: 1.8, model_complexity: 1 } } });
  r.elements.get('hand-mode-mouse').dispatch('click');
  r.cameraRequests[0].resolve({ ok: false, json: async () => ({ error: 'Camera is still stopping.' }) });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(r.elements.get('hand-mode-mouse').attributes['aria-pressed'], 'false');
  assert.equal(r.elements.get('mouse-gesture-guide').hidden, true);
  assert.match(r.elements.get('hand-mode-status').textContent, /still stopping/);
});

test('completed clicks get a transient badge without replaying progress or duplicate events', () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { running: true, state: 'running', input_mode: 'mouse',
    click_sound: { enabled: true, available: true } } });
  r.message({ type: 'gesture_progress', state: 'mouse_bend', input_mode: 'mouse', progress: 0.5, text: 'Hold bend' });
  assert.equal(r.elements.get('hand-click-feedback').hidden, true);
  assert.equal(r.elements.get('gesture-meter').hidden, false);
  r.message({ type: 'gesture_click', id: 'session:1', source: 'bend' });
  assert.equal(r.elements.get('hand-click-feedback').hidden, false);
  assert.equal(r.elements.get('hand-click-feedback').textContent, 'Clicked · index bend');
  r.message({ type: 'gesture_click', id: 'session:1', source: 'bend' });
  assert.equal(r.timers.filter(t => t.delay === 650).length, 1);
  r.timers.find(t => t.delay === 650).callback();
  assert.equal(r.elements.get('hand-click-feedback').hidden, true);
  r.message({ type: 'gesture_status', running: false, state: 'off', input_mode: 'mouse' });
  r.message({ type: 'gesture_click', id: 'session:2', source: 'bend' });
  assert.equal(r.elements.get('hand-click-feedback').hidden, true);
  assert.equal(r.cameraRequests.length, 0, 'click feedback cannot trigger another input request');
});

test('click sound can be muted live without stopping pointer control', async () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { running: true, state: 'running', input_mode: 'mouse',
    click_sound: { enabled: true, available: true } } });
  r.elements.get('hand-click-sound').dispatch('click');
  assert.equal(r.cameraRequests[0].url, 'http://127.0.0.1:7432/gestures/sound');
  assert.deepEqual(r.cameraRequests[0].body, { enabled: false });
  assert.equal(r.elements.get('hand-click-sound').disabled, true);
  r.cameraRequests[0].resolve({ ok: true, json: async () => ({ click_sound: { enabled: false, available: true } }) });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(r.elements.get('hand-click-sound').textContent, 'Sound off');
  assert.equal(r.elements.get('gesture-label').textContent, 'CAMERA ON');
  r.message({ type: 'gesture_click', id: 'session:1', source: 'pinch' });
  assert.equal(r.elements.get('hand-click-feedback').textContent, 'Clicked · pinch');
  assert.equal(r.cameraRequests.length, 1);
});

test('late sound responses cannot overwrite a reconnected session', async () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { running: true, state: 'running', input_mode: 'mouse',
    click_sound: { enabled: true, available: true } } });
  r.elements.get('hand-click-sound').dispatch('click');
  r.message({ type: 'gesture_click', id: 'old:1', source: 'bend' });
  r.socket.close();
  assert.equal(r.elements.get('hand-click-feedback').hidden, true);
  r.reconnect();
  r.message({ type: 'status', gestures: { running: false, state: 'off', input_mode: 'mouse',
    click_sound: { enabled: true, available: true } } });
  r.cameraRequests[0].resolve({ ok: true, json: async () => ({ click_sound: { enabled: false, available: true } }) });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(r.elements.get('hand-click-sound').textContent, 'Sound on');
});

test('desktop modes explain horizontal animation separately from vertical shortcuts', () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { running: true, state: 'running', available: true, input_mode: 'mouse' } });
  r.message({ type: 'gesture_desktop', mode: 'shortcut', preference: 'shortcut', completed: false });
  assert.match(r.elements.get('gesture-mode').textContent, /reach 100%/);
  assert.match(r.elements.get('gesture-mode').textContent, /no live desktop movement/);
  assert.equal(r.elements.get('gesture-mode').hidden, false);
  r.message({ type: 'gesture_desktop', mode: 'native', preference: 'auto', completed: true });
  assert.match(r.elements.get('gesture-mode').textContent, /Four-finger gestures/);
  assert.match(r.elements.get('gesture-mode').textContent, /Up\/down uses a shortcut/);
  assert.doesNotMatch(r.elements.get('gesture-mode').textContent, /Three-finger gestures/);
  assert.match(r.elements.get('gesture-feedback').textContent, /Windows handles/);
  r.message({ type: 'gesture_desktop', mode: 'native', preference: 'auto', fallback: true,
    error: 'Native input failed. Next swipe uses measured steps.' });
  assert.match(r.elements.get('gesture-mode').textContent, /reach 100%/);
  r.message({ type: 'gesture_desktop', mode: 'shortcut', preference: 'auto', native_available: true,
    axis: 'vertical', direction: 'up', completed: true });
  assert.match(r.elements.get('gesture-mode').textContent, /Sideways movement follows your hand/);
  assert.equal(r.elements.get('gesture-feedback').textContent, 'Task View requested. Lower your hand or make a fist before another swipe.');
  r.message({ type: 'gesture_desktop', mode: 'shortcut', preference: 'shortcut', native_available: true,
    axis: 'vertical', direction: 'down', completed: true });
  assert.match(r.elements.get('gesture-mode').textContent, /no live desktop movement/);
  assert.equal(r.elements.get('gesture-feedback').textContent, 'Show desktop toggle requested. Lower your hand or make a fist before another swipe.');
});

test('camera toggle waits for status and acknowledgement without submitting or editing a draft', () => {
  const r = renderer();
  const button = r.elements.get('gesture-toggle');
  button.dispatch('click');
  assert.equal(r.sent.length, 0);
  assert.equal(r.cameraRequests.length, 0);
  r.message({ type: 'status', gestures: { available: true, running: false, state: 'off', text: 'Gestures are off.' } });
  r.change('git status');
  r.show('git status', []);
  r.message({ type: 'confirm_request', token: 'waiting', capability: 'run_command', client_revision: 1 });
  const count = r.sent.length;
  button.dispatch('click');
  button.dispatch('click');
  assert.equal(r.cameraRequests.length, 1);
  assert.deepEqual(r.cameraRequests[0].body, { enabled: true });
  assert.equal(r.sent.length, count);
  assert.equal(button.attributes['aria-checked'], 'false');
  assert.equal(button.attributes['aria-busy'], 'true');
  assert.equal(r.elements.get('gesture-label').textContent, 'STARTING…');
  assert.equal(r.input.value, 'git status');
  assert.ok(vm.runInContext('resolution !== null && pendingConfirm.token === "waiting"', r.context));
  r.message({ type: 'gesture_status', available: true, running: true, state: 'running', text: 'Watching for gestures.' });
  assert.equal(button.attributes['aria-checked'], 'true');
  assert.equal(button.attributes['aria-busy'], 'false');
  assert.equal(r.elements.get('gesture-label').textContent, 'CAMERA ON');
  button.dispatch('click');
  assert.deepEqual(r.cameraRequests.at(-1).body, { enabled: false });
  assert.equal(button.attributes['aria-checked'], 'true', 'camera stays visibly on until release is acknowledged');
  r.message({ type: 'gesture_status', available: true, running: false, state: 'off', text: 'Gestures are off.' });
  assert.equal(button.attributes['aria-checked'], 'false');
  assert.equal(r.elements.get('gesture-label').textContent, 'CAMERA OFF');
});

test('camera startup can be cancelled and stopping blocks a second start', () => {
  const r = renderer();
  const button = r.elements.get('gesture-toggle');
  r.message({ type: 'gesture_status', available: true, running: false, state: 'starting', text: 'Starting gesture recognition.' });
  assert.equal(button.attributes['aria-disabled'], 'false');
  button.dispatch('click');
  assert.deepEqual(r.cameraRequests.at(-1).body, { enabled: false });
  r.message({ type: 'gesture_status', available: true, running: false, state: 'stopping', text: 'Releasing the camera.' });
  const count = r.cameraRequests.length;
  button.dispatch('click');
  assert.equal(r.cameraRequests.length, count);
  assert.equal(r.elements.get('gesture-label').textContent, 'STOPPING…');
});

for (const [cleanupType, cleanupFields] of [
  ['tracker', { tracker_cleanup_pending: true }],
  ['navigation', { navigation_cleanup_pending: true }],
  ['native desktop', { desktop: { cleanup_pending: true } }],
]) for (const available of [true, false]) {
  test(`unfinished ${cleanupType} cleanup retries stop and blocks settings (available=${available})`, async () => {
    const r = renderer();
    const button = r.elements.get('gesture-toggle');
    const cleanupStatus = { available, running: false, state: 'error',
      ...cleanupFields, tracker_backend: 'rtmpose', input_mode: 'mouse',
      text: 'Camera released; hand input cleanup failed.' };
    r.message({ type: 'gesture_status', ...cleanupStatus });
    assert.equal(r.elements.get('gesture-label').textContent, 'Retry cleanup');
    assert.equal(button.attributes['aria-disabled'], 'false');
    assert.equal(button.attributes['aria-checked'], 'false');
    assert.match(button.title, /Retry releasing hand input/);
    for (const id of ['gesture-model', 'gesture-bend-click', 'gesture-desktop-mode', 'gesture-travel',
      'gesture-save-settings', 'hand-mode-mouse', 'hand-mode-desktop']) {
      assert.equal(r.elements.get(id).disabled, true);
    }
    r.elements.get('gesture-save-settings').dispatch('click');
    r.elements.get('hand-mode-desktop').dispatch('click');
    assert.equal(r.cameraRequests.length, 0);
    button.dispatch('click');
    assert.deepEqual(r.cameraRequests[0].body, { enabled: false });
    assert.equal(r.elements.get('gesture-label').textContent, 'CLEANING…');
    r.cameraRequests[0].resolve({ ok: false, json: async () => ({
      error: 'Cleanup still pending.', gestures: cleanupStatus,
    }) });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(r.elements.get('gesture-label').textContent, 'Retry cleanup');
    button.dispatch('click');
    assert.deepEqual(r.cameraRequests[1].body, { enabled: false });
    r.cameraRequests[1].resolve({ ok: true, json: async () => ({ gestures: {
      available: true, running: false, state: 'off', tracker_cleanup_pending: false,
      navigation_cleanup_pending: false, desktop: { cleanup_pending: false },
      tracker_backend: 'rtmpose', input_mode: 'mouse',
    } }) });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(r.elements.get('gesture-label').textContent, 'CAMERA OFF');
    assert.equal(r.elements.get('gesture-model').disabled, false);
    assert.equal(r.elements.get('gesture-desktop-mode').disabled, false);
    assert.equal(r.elements.get('gesture-travel').disabled, false);
    assert.equal(r.elements.get('hand-mode-desktop').disabled, false);
    assert.equal(r.cameraRequests.length, 2, 'successful cleanup does not auto-start the camera');
    button.dispatch('click');
    assert.deepEqual(r.cameraRequests[2].body, { enabled: true });
  });
}

test('camera availability errors are visible and recoverable startup errors allow retry', () => {
  const r = renderer();
  const button = r.elements.get('gesture-toggle');
  r.message({ type: 'status', gestures: { available: false, running: false, state: 'unavailable', text: 'Install the pinned requirements.' } });
  button.dispatch('click');
  assert.equal(r.sent.length, 0);
  assert.equal(r.cameraRequests.length, 0);
  assert.match(r.elements.get('gesture-status').textContent, /pinned requirements/);
  assert.ok(r.elements.get('gesture-strip').classList.contains('visible'));
  r.message({ type: 'gesture_status', available: true, running: false, state: 'error', text: 'No camera at index 0.' });
  button.dispatch('click');
  assert.deepEqual(r.cameraRequests.at(-1).body, { enabled: true });
  r.message({ type: 'error', source: 'gestures', text: 'Camera is in use.' });
  r.message({ type: 'gesture_status', available: true, running: false, state: 'off', text: 'Gestures are off.' });
  assert.equal(button.attributes['aria-busy'], 'false');
  assert.match(r.elements.get('gesture-status').textContent, /Camera is in use/);
  assert.equal(r.elements.get('gesture-label').textContent, 'CAMERA ERROR');
});

test('disconnect keeps camera state unknown and reconnect never replays its toggle', () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { available: true, running: true, state: 'running', text: 'Watching for gestures.' } });
  r.elements.get('gesture-toggle').dispatch('click');
  r.socket.close();
  assert.equal(r.elements.get('gesture-label').textContent, 'CAMERA ?');
  assert.match(r.elements.get('gesture-status').textContent, /may still be running/);
  const count = r.cameraRequests.length;
  r.reconnect();
  assert.equal(r.cameraRequests.length, count);
  assert.equal(r.elements.get('gesture-toggle').attributes['aria-disabled'], 'true');
  r.message({ type: 'status', gestures: { available: true, running: false, state: 'off', text: 'Gestures are off.' } });
  assert.equal(r.elements.get('gesture-label').textContent, 'CAMERA OFF');
  assert.ok(!r.elements.get('gesture-strip').classList.contains('visible'));
});

test('camera request failure clears pending controls without pretending the camera stopped', async () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { available: true, running: true, state: 'running' } });
  r.elements.get('gesture-toggle').dispatch('click');
  r.cameraRequests[0].reject(new Error('network failure'));
  await new Promise(resolve => setImmediate(resolve));
  r.message({ type: 'gesture_status', available: true, running: true, state: 'running' });
  assert.equal(vm.runInContext('requestedGestures', r.context), null);
  assert.equal(r.elements.get('gesture-label').textContent, 'CAMERA ON');
  assert.match(r.elements.get('gesture-status').textContent, /could not be confirmed/);
});

test('camera stop uses its own request while a command reply is pending', async () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { available: true, running: true, state: 'running' } });
  r.change('Explain something slowly');
  r.show(r.input.value, []);
  r.key('Enter');
  r.message({ type: 'thinking' });
  const count = r.sent.length;
  r.elements.get('gesture-toggle').dispatch('click');
  assert.equal(r.sent.length, count, 'camera stop must bypass the busy command socket');
  const request = r.cameraRequests[0];
  assert.equal(request.url, 'http://127.0.0.1:7432/gestures');
  assert.equal(request.method, 'POST');
  assert.deepEqual(request.body, { enabled: false });
  request.resolve({ ok: true, json: async () => ({ gestures: {
    available: true, running: false, state: 'off', text: 'The camera is released.',
  } }) });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(r.elements.get('gesture-label').textContent, 'CAMERA OFF');
  assert.ok(r.elements.get('thinking-indicator').classList.contains('active'));
});

test('a late camera response cannot overwrite fresh status after reconnection', async () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { available: true, running: false, state: 'off' } });
  r.elements.get('gesture-toggle').dispatch('click');
  const oldRequest = r.cameraRequests[0];
  r.socket.close();
  r.reconnect();
  r.message({ type: 'status', gestures: { available: true, running: false, state: 'off' } });
  oldRequest.resolve({ ok: true, json: async () => ({ gestures: {
    available: true, running: true, state: 'running',
  } }) });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(r.elements.get('gesture-label').textContent, 'CAMERA OFF');
  assert.equal(r.cameraRequests.length, 1, 'reconnect must not replay the camera start');
});

test('gesture feedback uses text and only appears while tracking is active', () => {
  const r = renderer();
  r.message({ type: 'status', gestures: { available: true, running: true, state: 'running' } });
  r.message({ type: 'gesture', gesture: 'swipe_left', ok: true, text: 'Moved <window> left.' });
  assert.equal(r.elements.get('gesture-feedback').textContent, 'Swipe left — Moved <window> left.');
  assert.equal(r.elements.get('gesture-feedback').innerHTML, '');
  r.message({ type: 'gesture', gesture: 'two_finger', ok: false, text: 'No window available.' });
  assert.match(r.elements.get('gesture-feedback').textContent, /Action failed: No window available/);
  r.message({ type: 'gesture_status', available: true, running: false, state: 'off' });
  r.message({ type: 'gesture', gesture: 'fist', ok: true, text: 'Late event' });
  assert.equal(r.elements.get('gesture-feedback').textContent, '');
});

test('Safe Mode toggle waits for backend acknowledgement and preserves the draft and approval', () => {
  const r = renderer();
  const toggle = r.elements.get('safe-toggle');
  assert.equal(toggle.disabled, true, 'unknown mode must not be guessed');
  toggle.dispatch('click');
  assert.equal(r.sent.length, 0);
  r.message({ type: 'status', safe_mode: true });
  assert.equal(toggle.disabled, false);
  assert.equal(toggle.attributes['aria-checked'], 'true');
  r.change('git status');
  r.show('git status', []);
  r.message({ type: 'confirm_request', token: 'waiting', capability: 'run_command', client_revision: 1 });
  const count = r.sent.length;
  toggle.dispatch('click');
  toggle.dispatch('click');
  assert.deepEqual(r.sent.slice(count), [{ type: 'set_safe_mode', enabled: false }]);
  assert.equal(toggle.disabled, true);
  assert.equal(toggle.attributes['aria-checked'], 'true', 'show acknowledged state while waiting');
  assert.equal(r.input.value, 'git status');
  assert.ok(vm.runInContext('resolution !== null && pendingConfirm.token === "waiting"', r.context));
  r.message({ type: 'status', safe_mode: false });
  assert.equal(toggle.disabled, false);
  assert.equal(toggle.attributes['aria-checked'], 'false');
  assert.ok(r.elements.get('safe-dot').classList.contains('unsafe'));
  assert.equal(r.elements.get('safe-label').textContent, 'SAFE OFF');
  assert.ok(!r.sent.some(message => message.type === 'confirm'));
  toggle.dispatch('click');
  assert.deepEqual(r.sent.at(-1), { type: 'set_safe_mode', enabled: true });
  r.message({ type: 'status', safe_mode: true });
  assert.equal(r.elements.get('safe-label').textContent, 'SAFE ON');
  assert.ok(!r.elements.get('safe-dot').classList.contains('unsafe'));
});

test('a disconnected mode change is never replayed and must wait for fresh status', () => {
  const r = renderer();
  const toggle = r.elements.get('safe-toggle');
  r.message({ type: 'status', safe_mode: true });
  r.socket.failSend = true;
  toggle.dispatch('click');
  assert.equal(toggle.disabled, true);
  assert.equal(vm.runInContext('requestedSafeMode', r.context), null);
  r.reconnect();
  assert.equal(toggle.disabled, true);
  assert.ok(!r.sent.some(message => message.type === 'set_safe_mode'));
  r.message({ type: 'status', safe_mode: false });
  assert.equal(toggle.disabled, false);
  assert.equal(toggle.attributes['aria-checked'], 'false');
});

test('Safe Mode permission explains the mode change and grants only the displayed token', () => {
  const r = renderer();
  r.message({ type: 'status', safe_mode: true });
  r.message({ type: 'confirm_request', token: 'permission', capability: 'run_command',
    reversibility: 'irreversible', requires_safe_mode_off: true, args: { command: 'echo <hello>' } });
  assert.equal(r.elements.get('confirm-title').textContent, 'Turn off Safe Mode and run?');
  assert.match(r.elements.get('confirm-detail').textContent, /Safe Mode will stay off until you turn it on/);
  assert.match(r.elements.get('confirm-detail').textContent, /command=echo <hello>/);
  assert.equal(r.elements.get('confirm-allow').textContent, 'Yes, turn off & run');
  assert.equal(r.sent.length, 0, 'showing the question must do nothing');
  r.elements.get('confirm-allow').dispatch('click');
  r.elements.get('confirm-allow').dispatch('click');
  assert.deepEqual(r.sent, [{ type: 'confirm', token: 'permission', granted: true, allow_safe_mode_change: true }]);
  assert.equal(r.elements.get('safe-toggle').attributes['aria-checked'], 'true');
});

test('declining Safe Mode permission leaves mode unchanged and cancels the action', () => {
  const r = renderer();
  r.message({ type: 'status', safe_mode: true });
  r.message({ type: 'confirm_request', token: 'permission', capability: 'run_command',
    reversibility: 'irreversible', requires_safe_mode_off: true });
  r.elements.get('confirm-deny').dispatch('click');
  assert.deepEqual(r.sent, [{ type: 'confirm', token: 'permission', granted: false }]);
  assert.equal(r.elements.get('safe-toggle').attributes['aria-checked'], 'true');
  assert.ok(!r.elements.get('thinking-indicator').classList.contains('active'));
  r.message({ type: 'confirm_request', token: 'normal', capability: 'write_file' });
  assert.equal(r.elements.get('confirm-allow').textContent, 'Yes, run');
  assert.ok(!r.elements.get('confirm-detail').textContent.includes('Safe Mode will stay off'));
});

test('turning Safe Mode off refreshes a waiting permission prompt without approving it', () => {
  const r = renderer();
  r.message({ type: 'status', safe_mode: true });
  r.message({ type: 'confirm_request', token: 'waiting', capability: 'run_command',
    reversibility: 'irreversible', requires_safe_mode_off: true, args: { cmd: 'echo hello' } });
  r.elements.get('safe-toggle').dispatch('click');
  r.message({ type: 'status', safe_mode: false });
  assert.equal(r.elements.get('confirm-title').textContent, 'Cannot be undone: run_command');
  assert.equal(r.elements.get('confirm-allow').textContent, 'Yes, run');
  assert.equal(r.elements.get('confirm-detail').textContent, 'cmd=echo hello');
  assert.ok(r.elements.get('confirm-bar').classList.contains('active'));
  assert.ok(!r.sent.some(message => message.type === 'confirm'));
  r.elements.get('confirm-allow').dispatch('click');
  assert.deepEqual(r.sent.at(-1), { type: 'confirm', token: 'waiting', granted: true, allow_safe_mode_change: false });
});

test('an older mode-change prompt arriving after SAFE OFF cannot ask to disable it again', () => {
  const r = renderer();
  r.message({ type: 'status', safe_mode: false });
  r.message({ type: 'confirm_request', token: 'delayed', capability: 'os_shutdown_computer',
    reversibility: 'irreversible', requires_safe_mode_off: true,
    reason: 'os_shutdown_computer requires turning off Safe Mode and confirmation' });
  assert.equal(r.elements.get('confirm-allow').textContent, 'Yes, run');
  assert.ok(!r.elements.get('confirm-title').textContent.includes('Safe Mode'));
  assert.ok(!r.elements.get('confirm-detail').textContent.includes('Safe Mode'));
  r.message({ type: 'status', safe_mode: true });
  assert.equal(r.elements.get('confirm-title').textContent, 'Turn off Safe Mode and run?');
  assert.equal(r.elements.get('confirm-allow').textContent, 'Yes, turn off & run');
  assert.ok(!r.sent.some(message => message.type === 'confirm'));
});

test('renderer highlights only the changed token and submits exact visible arguments', () => {
  const r = renderer();
  const raw = '  pyhton\ttrain.py --key "<secret>"  ';
  const corrected = '  python\ttrain.py --key "<secret>"  ';
  r.change(raw);
  r.show(raw, [{ text: corrected, token: 'python', span: [2, 8] }]);
  const first = r.elements.get('correction-choices').children[0];
  assert.equal(first.children.find(child => child.tagName === 'mark').textContent, 'python');
  assert.ok(textOf(first).endsWith(corrected));
  r.key('Enter');
  const submission = r.sent.at(-1);
  assert.equal(submission.type, 'input');
  assert.equal(submission.text, raw);
  assert.equal(submission.selected_text, corrected);
  assert.equal(submission.token, 'visible-token');
  assert.equal(submission.candidate_index, 0);
});

test('Enter after an unseen edit requests review without executing', () => {
  const r = renderer();
  r.change('gti status');
  r.key('Enter');
  assert.equal(r.sent.at(-1).type, 'resolve');
  assert.ok(!r.sent.some(message => message.type === 'input'));
});

test('stale resolution cannot apply after argument edits', () => {
  const r = renderer();
  r.change('gti status');
  r.change('gti diff');
  r.message({ type: 'resolution', original: 'gti status', client_revision: 1,
    candidates: [{ text: 'git status', token: 'git', span: [0, 3] }], token: 'stale', revision: 1 });
  r.key('Enter');
  assert.equal(r.sent.at(-1).type, 'resolve');
  assert.equal(r.sent.at(-1).text, 'gti diff');
});

test('every edit including empty input revokes local confirmation and informs server', () => {
  const r = renderer();
  r.change('gti status');
  r.message({ type: 'confirm_request', token: 'pending', capability: 'run_command', client_revision: 1 });
  assert.ok(r.elements.get('confirm-bar').classList.contains('active'));
  r.change('');
  assert.ok(!r.elements.get('confirm-bar').classList.contains('active'));
  assert.equal(r.sent.at(-1).type, 'buffer');
  assert.equal(r.sent.at(-1).text, '');
  r.message({ type: 'confirm_request', token: 'late', capability: 'run_command', client_revision: 1 });
  assert.deepEqual(r.sent.at(-1), { type: 'confirm', token: 'late', granted: false });
});

test('alternatives and keep-original selection are explicit and keyboard accessible', () => {
  const r = renderer();
  r.change('gti status');
  r.show('gti status', [
    { text: 'git status', token: 'git', span: [0, 3] },
    { text: 'ghi status', token: 'ghi', span: [0, 3] },
  ]);
  r.key('n', { ctrlKey: true });
  assert.equal(vm.runInContext('selectedCorrection', r.context), 1);
  r.key('Escape');
  r.key('Enter');
  assert.equal(r.sent.at(-1).candidate_index, null);
  assert.equal(r.sent.at(-1).selected_text, 'gti status');
});

test('voice fills the editable draft and never submits by itself', () => {
  const r = renderer();
  r.message({ type: 'voice_text', text: 'gti status' });
  assert.equal(r.input.value, 'gti status');
  assert.equal(r.sent.at(-1).type, 'buffer');
  assert.ok(!r.sent.some(message => message.type === 'input'));
});

test('offline Enter shows an actionable error and preserves the complete draft', () => {
  const r = renderer({ connected: false });
  const raw = '  gti status --secret "keep this"  ';
  r.change(raw);
  r.key('Enter');
  assert.equal(r.input.value, raw);
  assert.equal(r.sent.length, 0);
  assert.equal(r.elements.get('safe-label').textContent, 'OFFLINE');
  assert.ok(r.elements.get('output-area').classList.contains('visible'));
  assert.match(r.elements.get('output-text').innerHTML, /backend is disconnected/i);
  assert.match(r.elements.get('output-text').innerHTML, /launcher/i);
});

test('offline mic click and shortcut explain disconnection without pretending to record', () => {
  const r = renderer({ connected: false });
  r.elements.get('mic-btn').dispatch('click');
  r.voiceToggle();
  assert.equal(r.sent.length, 0);
  assert.equal(vm.runInContext('isRecording', r.context), false);
  assert.ok(!r.elements.get('recording-bar').classList.contains('active'));
  assert.equal(r.elements.get('mic-btn').attributes['aria-disabled'], 'true');
  assert.match(r.elements.get('output-text').innerHTML, /backend is disconnected/i);
});

test('send failure retains a reviewed command instead of clearing an unsent draft', () => {
  const r = renderer();
  r.change('gti status');
  r.show('gti status', [{ text: 'git status', token: 'git', span: [0, 3] }]);
  r.socket.failSend = true;
  r.key('Enter');
  assert.equal(r.input.value, 'gti status');
  assert.ok(!r.sent.some(message => message.type === 'input'));
  assert.equal(vm.runInContext('resolution', r.context), null);
  assert.match(r.elements.get('output-text').innerHTML, /backend is disconnected/i);
});

test('disconnect clears approvals, stale corrections and recording indicators', () => {
  const r = renderer();
  r.change('gti status');
  r.show('gti status', [{ text: 'git status', token: 'git', span: [0, 3] }]);
  r.message({ type: 'confirm_request', token: 'pending', capability: 'run_command', client_revision: 1 });
  r.elements.get('mic-btn').dispatch('click');
  r.socket.close();
  assert.equal(vm.runInContext('resolution', r.context), null);
  assert.equal(vm.runInContext('pendingConfirm', r.context), null);
  assert.equal(vm.runInContext('isRecording', r.context), false);
  assert.ok(!r.elements.get('recording-bar').classList.contains('active'));
  assert.ok(!r.elements.get('thinking-indicator').classList.contains('active'));
  assert.equal(r.input.value, 'gti status');
});

test('reconnect reviews the latest offline draft and never replays input or approvals', () => {
  const r = renderer();
  r.change('gti status');
  r.show('gti status', [{ text: 'git status', token: 'git', span: [0, 3] }]);
  const oldSocket = r.socket;
  oldSocket.close();
  r.change('gti diff  --stat');
  r.key('Enter');
  r.reconnect();
  const refreshed = r.sent.at(-1);
  assert.equal(refreshed.type, 'buffer');
  assert.equal(refreshed.text, 'gti diff  --stat');
  assert.ok(refreshed.client_revision > 2);
  assert.ok(!r.sent.some(message => message.type === 'input' || message.type === 'confirm'));
  oldSocket.onmessage({ data: JSON.stringify({ type: 'resolution', original: r.input.value,
    client_revision: refreshed.client_revision, candidates: [], token: 'old', revision: 1 }) });
  assert.equal(vm.runInContext('resolution', r.context), null);
  r.key('Enter');
  assert.equal(r.sent.at(-1).type, 'resolve');
  r.show('gti diff  --stat', [{ text: 'git diff  --stat', token: 'git', span: [0, 3] }]);
  r.key('Enter');
  assert.equal(r.sent.at(-1).type, 'input');
  assert.equal(r.sent.at(-1).selected_text, 'git diff  --stat');
});

test('voice loading and setup failures explain availability, and ready status enables retry', () => {
  const r = renderer();
  r.message({ type: 'status', voice: { state: 'loading', available: false, text: 'Voice model is loading.' } });
  r.elements.get('mic-btn').dispatch('click');
  assert.ok(!r.sent.some(message => message.type === 'voice_start'));
  assert.match(r.elements.get('output-text').innerHTML, /Voice model is loading/);
  r.message({ type: 'voice_status', state: 'ready', available: true, text: 'Voice ready.' });
  r.elements.get('mic-btn').dispatch('click');
  assert.equal(r.sent.at(-1).type, 'voice_start');
  assert.ok(r.elements.get('recording-bar').classList.contains('active'));
  r.message({ type: 'error', source: 'voice', text: 'Microphone access was denied.' });
  assert.equal(vm.runInContext('isRecording', r.context), false);
  assert.ok(!r.elements.get('recording-bar').classList.contains('active'));
  assert.match(r.elements.get('output-text').innerHTML, /Microphone access was denied/);
  r.message({ type: 'voice_status', state: 'error', available: true, text: 'Try your microphone again.' });
  r.elements.get('mic-btn').dispatch('click');
  assert.equal(r.sent.at(-1).type, 'voice_start');
});

test('transcription errors stop the busy microphone and preserve typed text', () => {
  const r = renderer();
  r.change('/help');
  r.elements.get('mic-btn').dispatch('click');
  r.message({ type: 'voice_recording', active: false });
  assert.ok(r.elements.get('mic-btn').classList.contains('transcribing'));
  r.message({ type: 'error', source: 'voice', text: 'Could not load the local speech model.' });
  assert.ok(!r.elements.get('mic-btn').classList.contains('transcribing'));
  assert.ok(!r.elements.get('recording-bar').classList.contains('active'));
  assert.equal(r.input.value, '/help');
});

test('a backend started after the HUD automatically reviews the waiting draft', () => {
  const r = renderer({ connected: false });
  r.change('/help');
  r.socket.close();
  r.reconnect();
  assert.equal(r.input.value, '/help');
  assert.deepEqual(r.sent.at(-1), { type: 'buffer', text: '/help', client_revision: 1 });
  assert.ok(!r.sent.some(message => message.type === 'input'));
  r.show('/help', []);
  r.key('Enter');
  assert.equal(r.sent.at(-1).selected_text, '/help');
  assert.equal(r.input.value, '');
});

test('an approval that cannot be sent is discarded without showing execution in progress', () => {
  const r = renderer();
  r.change('git status');
  r.message({ type: 'confirm_request', token: 'pending', capability: 'run_command', client_revision: 1 });
  // The socket can close before its onclose callback reaches the renderer.
  r.socket.readyState = 3;
  r.key('Enter');
  assert.ok(!r.sent.some(message => message.type === 'confirm'));
  assert.equal(vm.runInContext('pendingConfirm', r.context), null);
  assert.ok(!r.elements.get('thinking-indicator').classList.contains('active'));
  assert.match(r.elements.get('output-text').innerHTML, /backend is disconnected/i);
});

test('thinking immediately replaces stale output with the safely escaped submitted request', () => {
  const r = renderer();
  r.message({ type: 'reply', text: 'Old help response', plan: ['Old plan'] });
  const raw = 'How are you <today> & "well"?';
  r.change(raw);
  r.show(raw, []);
  r.key('Enter');
  r.message({ type: 'thinking' });
  const output = r.elements.get('output-text').innerHTML;
  assert.match(output, /Thinking…/);
  assert.match(output, /How are you &lt;today&gt; &amp; &quot;well&quot;\?/);
  assert.ok(!output.includes('Old help response'));
  assert.ok(!output.includes('<today>'));
  assert.equal(r.elements.get('plan-strip').innerHTML, '');
  assert.ok(!r.elements.get('plan-strip').classList.contains('visible'));
  assert.ok(r.elements.get('expand-panel').classList.contains('open'));
  assert.ok(r.elements.get('output-area').classList.contains('visible'));
  assert.ok(r.elements.get('thinking-indicator').classList.contains('active'));
});

test('streaming and final replies replace the visible pending state', () => {
  const r = renderer();
  r.change('How are you today?');
  r.show('How are you today?', []);
  r.key('Enter');
  r.message({ type: 'thinking' });
  r.message({ type: 'token', text: 'A partial response' });
  assert.match(r.elements.get('output-text').innerHTML, /A partial response/);
  assert.ok(!r.elements.get('output-text').innerHTML.includes('Thinking…'));
  r.message({ type: 'reply', text: 'The final response' });
  const output = r.elements.get('output-text').innerHTML;
  assert.match(output, /How are you today\?/);
  assert.match(output, /The final response/);
  assert.ok(!output.includes('partial response'));
  assert.ok(!r.elements.get('thinking-indicator').classList.contains('active'));
});

test('browser progress displays the action and a launch failure clears the busy state', () => {
  const r = renderer();
  r.change('open chrome and go to google.com');
  r.show(r.input.value, []);
  r.key('Enter');
  r.message({ type: 'thinking', text: 'Opening browser…' });
  assert.match(r.elements.get('output-text').innerHTML, /Opening browser…/);
  assert.ok(r.elements.get('thinking-indicator').classList.contains('active'));
  r.message({ type: 'reply', text: 'Chrome was not found. Install Chrome or use your default browser.' });
  assert.ok(!r.elements.get('thinking-indicator').classList.contains('active'));
  assert.match(r.elements.get('output-text').innerHTML, /Chrome was not found/);
});
