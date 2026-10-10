// EEG guard lifecycle with fake shortcuts, transport, and manually driven timers.
// No Electron application, network connection, or device is started.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { EventEmitter } = require('node:events');
const { EegGuard } = require('../ui/eeg-guard.cjs');

async function flush() {
  for (let turn = 0; turn < 12; turn += 1) await Promise.resolve();
}

function harness({ registration = true, nativeRegistered = true } = {}) {
  const requests = [], registrations = [], unregistrations = [], notifications = [], timers = [], trace = [];
  const callbacks = new Map();
  const shortcutState = { registered: nativeRegistered };
  let token = 0;
  const makeTimer = (kind, callback, delay) => {
    const timer = { kind, callback, delay, active: true, unref() {} };
    timers.push(timer);
    return timer;
  };
  const clearTimer = timer => { if (timer) timer.active = false; };
  const guard = new EegGuard({
    globalShortcut: {
      register(accelerator, callback) {
        registrations.push(accelerator);
        trace.push(`register:${accelerator}`);
        if (registration instanceof Error) throw registration;
        if (registration) callbacks.set(accelerator, callback);
        return registration;
      },
      unregister(accelerator) {
        unregistrations.push(accelerator);
        trace.push(`unregister:${accelerator}`);
        callbacks.delete(accelerator);
      },
      isRegistered: accelerator => shortcutState.registered && callbacks.has(accelerator),
    },
    request(action, payload) {
      let resolve, reject;
      const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
      const request = {
        action, payload, settled: false,
        resolve(value = { ok: true }) { this.settled = true; resolve(value); },
        reject(error = new Error('Synthetic transport failure')) { this.settled = true; reject(error); },
      };
      requests.push(request);
      trace.push(`request:${action}`);
      return promise;
    },
    notify: reason => { notifications.push(reason); trace.push(`notify:${reason}`); },
    randomUUID: () => `synthetic-guard-${++token}`,
    setInterval: (callback, delay) => makeTimer('interval', callback, delay),
    clearInterval: clearTimer,
    setTimeout: (callback, delay) => makeTimer('timeout', callback, delay),
    clearTimeout: clearTimer,
  });
  const active = kind => timers.filter(timer => timer.active && timer.kind === kind);
  const fire = timer => {
    assert.ok(timer?.active, 'Only a live timer should be fired normally');
    if (timer.kind === 'timeout') timer.active = false;
    timer.callback();
  };
  const latest = action => requests.filter(request => request.action === action).at(-1);
  const finishCleanup = async () => {
    for (let turn = 0; turn < 3; turn += 1) {
      for (const request of requests) {
        if (!request.settled && request.action !== 'eeg_guard') request.resolve();
      }
      await flush();
    }
  };
  const start = async () => {
    const pending = guard.start();
    await flush();
    latest('eeg_guard').resolve();
    return pending;
  };
  return { guard, requests, registrations, unregistrations, notifications, timers, trace,
    callbacks, shortcutState, active, fire, latest, finishCleanup, start };
}

test('readiness follows Escape registration and acknowledged lease; ready starts reuse the token', async () => {
  const h = harness();
  const pending = h.guard.start();
  await flush();
  assert.deepEqual(h.registrations, ['Escape']);
  assert.deepEqual(h.trace.slice(0, 2), ['register:Escape', 'request:eeg_guard']);
  assert.deepEqual(h.latest('eeg_guard').payload, { token: 'synthetic-guard-1' });
  assert.equal(h.active('interval').length, 0);
  assert.equal(h.active('timeout').length, 1);
  assert.equal(h.active('timeout')[0].delay, 1500);
  h.latest('eeg_guard').resolve();
  assert.deepEqual(await pending, { ok: true, token: 'synthetic-guard-1' });
  assert.deepEqual(await h.guard.start(), { ok: true, token: 'synthetic-guard-1' });
  assert.equal(h.requests.length, 1);
  assert.equal(h.active('timeout').length, 0);
  assert.equal(h.active('interval').length, 1);
  assert.equal(h.active('interval')[0].delay, 1000);
});

for (const [name, registration] of [['unavailable', false], ['throws', new Error('Shortcut unavailable')]]) {
  test(`Escape registration ${name} prevents the lease and readiness`, async () => {
    const h = harness({ registration });
    const pending = h.guard.start();
    await flush();
    await h.finishCleanup();
    const result = await pending;
    assert.equal(result.ok, false);
    assert.ok(result.error);
    assert.equal(h.requests.filter(request => request.action === 'eeg_guard').length, 0);
    assert.equal(h.active('interval').length, 0);
    assert.equal(h.active('timeout').length, 0);
    assert.equal(h.callbacks.has('Escape'), false);
  });
}

for (const failure of ['response', 'data-error', 'rejection', 'timeout']) {
  test(`initial lease ${failure} releases Escape and never renews after a delayed reply`, async () => {
    const h = harness();
    const pending = h.guard.start();
    await flush();
    const lease = h.latest('eeg_guard');
    if (failure === 'response') lease.resolve({ ok: false, error: 'Synthetic refusal' });
    else if (failure === 'data-error') lease.resolve({ ok: true, data: { error: 'Lease was refused' } });
    else if (failure === 'rejection') lease.reject();
    else h.fire(h.active('timeout')[0]);
    await flush();
    assert.deepEqual(h.latest('eeg_arm').payload, { enabled: false });
    await h.finishCleanup();
    const result = await pending;
    assert.equal(result.ok, false);
    assert.ok(result.error);
    assert.ok(h.unregistrations.includes('Escape'));
    assert.equal(h.callbacks.has('Escape'), false);
    assert.equal(h.active('interval').length, 0);
    if (!lease.settled) lease.resolve();
    await flush();
    assert.equal(h.active('interval').length, 0);
    assert.equal(h.requests.filter(request => request.action === 'eeg_guard').length, 1);
    assert.equal(h.active('timeout').length, 0);
  });
}

test('heartbeat renews the same token without overlapping requests', async () => {
  const h = harness();
  await h.start();
  const timer = h.active('interval')[0];
  h.fire(timer);
  await flush();
  h.fire(timer);
  await flush();
  assert.equal(h.requests.length, 2, 'A pending renewal must exclude another renewal');
  assert.deepEqual(h.latest('eeg_guard').payload, { token: 'synthetic-guard-1' });
  h.latest('eeg_guard').resolve();
  await flush();
  h.fire(timer);
  await flush();
  assert.equal(h.requests.length, 3);
  h.latest('eeg_guard').resolve();
  await flush();
  assert.equal(h.active('timeout').length, 0);
});

for (const failure of ['response', 'data-error', 'rejection', 'timeout']) {
  test(`heartbeat ${failure} unregisters Escape, notifies the renderer, and stops`, async () => {
    const h = harness();
    await h.start();
    const timer = h.active('interval')[0];
    h.fire(timer);
    await flush();
    const renewal = h.latest('eeg_guard');
    if (failure === 'response') renewal.resolve({ ok: false });
    else if (failure === 'data-error') renewal.resolve({ ok: true, data: { error: 'Lease was refused' } });
    else if (failure === 'rejection') renewal.reject();
    else h.fire(h.active('timeout')[0]);
    await flush();
    assert.equal(h.active('interval').length, 0);
    assert.equal(h.callbacks.has('Escape'), false);
    assert.ok(h.unregistrations.includes('Escape'));
    assert.equal(h.notifications.length, 1);
    assert.deepEqual(h.latest('stop').payload, {});
    await h.finishCleanup();
    const requestCount = h.requests.length;
    if (!renewal.settled) renewal.resolve();
    timer.callback(); // A queued callback from the retired interval is harmless.
    await flush();
    assert.equal(h.requests.length, requestCount);
    assert.equal(h.active('interval').length, 0);
    assert.equal(h.active('timeout').length, 0);
  });
}

test('Escape immediately retires the guard and coalesces an emergency stop in flight', async () => {
  const h = harness();
  await h.start();
  const escape = h.callbacks.get('Escape');
  const timer = h.active('interval')[0];
  escape();
  await flush();
  assert.equal(h.active('interval').length, 0);
  assert.equal(h.callbacks.has('Escape'), false);
  assert.deepEqual(h.notifications, ['escape']);
  assert.deepEqual(h.latest('stop').payload, {});
  const stopIndex = h.trace.indexOf('request:stop');
  assert.ok(h.trace.indexOf('unregister:Escape') < stopIndex);
  assert.ok(h.trace.indexOf('notify:escape') < stopIndex);
  escape();
  await flush();
  assert.equal(h.requests.filter(request => request.action === 'stop').length, 1);
  await h.finishCleanup();
  const requestCount = h.requests.length;
  timer.callback();
  await flush();
  assert.equal(h.requests.length, requestCount);
  assert.equal(h.active('timeout').length, 0);
});

test('explicit stop releases the shortcut and heartbeat before disarming, including when inactive', async () => {
  const h = harness();
  await h.start();
  const stopping = h.guard.stop();
  await flush();
  assert.equal(h.active('interval').length, 0);
  assert.equal(h.callbacks.has('Escape'), false);
  assert.deepEqual(h.latest('eeg_arm').payload, { enabled: false });
  assert.ok(h.trace.indexOf('unregister:Escape') < h.trace.indexOf('request:eeg_arm'));
  h.latest('eeg_arm').resolve({ ok: true, data: { armed: false } });
  assert.equal((await stopping).ok, true);
  const inactiveStop = h.guard.stop();
  await flush();
  assert.equal(h.requests.filter(request => request.action === 'eeg_arm').length, 2);
  h.latest('eeg_arm').resolve();
  assert.equal((await inactiveStop).ok, true);
  assert.equal(h.active('timeout').length, 0);
});

test('concurrent starts share one pending lease and one Escape registration', async () => {
  const h = harness();
  const first = h.guard.start(), second = h.guard.start();
  await flush();
  assert.deepEqual(h.registrations, ['Escape']);
  assert.equal(h.requests.length, 1);
  h.latest('eeg_guard').resolve();
  const results = await Promise.all([first, second]);
  assert.deepEqual(results, [
    { ok: true, token: 'synthetic-guard-1' },
    { ok: true, token: 'synthetic-guard-1' },
  ]);
  assert.equal(h.active('interval').length, 1);
});

for (const retirement of ['escape', 'stop', 'dispose']) {
  test(`${retirement} during pending start makes the delayed lease acknowledgement stale`, async () => {
    const h = harness();
    const starting = h.guard.start();
    await flush();
    const lease = h.latest('eeg_guard');
    let stopping;
    if (retirement === 'escape') h.callbacks.get('Escape')();
    else stopping = h.guard[retirement]();
    await flush();
    assert.equal(h.callbacks.has('Escape'), false);
    await h.finishCleanup();
    if (stopping) await stopping;
    lease.resolve();
    const result = await starting;
    assert.equal(result.ok, false);
    assert.ok(result.error);
    assert.equal(h.active('interval').length, 0);
    assert.equal(h.active('timeout').length, 0);
    assert.equal(h.requests.filter(request => request.action === 'eeg_guard').length, 1);
    assert.ok(!h.requests.some(request => request.action === 'eeg_arm' && request.payload.enabled));
    if (retirement === 'dispose') assert.deepEqual(h.notifications, ['shutdown']);
  });
}

test('a retired heartbeat failure cannot stop a newly acknowledged guard', async () => {
  const h = harness();
  await h.start();
  h.fire(h.active('interval')[0]);
  await flush();
  const oldRenewal = h.latest('eeg_guard');
  const stopping = h.guard.stop();
  await flush();
  await h.finishCleanup();
  await stopping;
  const restarted = await h.start();
  assert.deepEqual(restarted, { ok: true, token: 'synthetic-guard-2' });
  const requestCount = h.requests.length;
  oldRenewal.reject();
  await flush();
  assert.equal(h.requests.length, requestCount);
  assert.equal(h.active('interval').length, 1);
  assert.equal(h.callbacks.has('Escape'), true);
  assert.deepEqual(h.notifications, []);
  h.fire(h.active('interval')[0]);
  await flush();
  assert.deepEqual(h.latest('eeg_guard').payload, { token: 'synthetic-guard-2' });
  h.latest('eeg_guard').resolve();
  await flush();
});

test('shutdown cleanup stays bounded when the emergency stop transport never answers', async () => {
  const h = harness();
  await h.start();
  const disposing = h.guard.dispose();
  await flush();
  assert.deepEqual(h.notifications, ['shutdown']);
  assert.deepEqual(h.latest('stop').payload, {});
  assert.equal(h.active('interval').length, 0);
  assert.equal(h.callbacks.has('Escape'), false);
  assert.equal(h.active('timeout').length, 1);
  assert.equal(h.active('timeout')[0].delay, 1500);
  h.fire(h.active('timeout')[0]);
  assert.equal((await disposing).ok, false);
  h.latest('stop').resolve();
  await flush();
  assert.equal(h.active('interval').length, 0);
  assert.equal(h.active('timeout').length, 0);
});

test('unconfirmed native Escape ownership prevents a lease despite register reporting success', async () => {
  const h = harness({ nativeRegistered: false });
  const pending = h.guard.start();
  await flush();
  await h.finishCleanup();
  assert.equal((await pending).ok, false);
  assert.equal(h.requests.filter(request => request.action === 'eeg_guard').length, 0);
  assert.equal(h.callbacks.has('Escape'), false);
  assert.equal(h.active('interval').length, 0);
});

test('Escape ownership lost while the initial lease is pending prevents readiness', async () => {
  const h = harness();
  const pending = h.guard.start();
  await flush();
  h.shortcutState.registered = false;
  h.latest('eeg_guard').resolve();
  await flush();
  await h.finishCleanup();
  assert.equal((await pending).ok, false);
  assert.equal(h.callbacks.has('Escape'), false);
  assert.equal(h.active('interval').length, 0);
});

test('lost native Escape registration stops immediately without renewing the lease', async () => {
  const h = harness();
  await h.start();
  h.shortcutState.registered = false;
  h.fire(h.active('interval')[0]);
  await flush();
  assert.equal(h.requests.filter(request => request.action === 'eeg_guard').length, 1);
  assert.deepEqual(h.latest('stop').payload, {});
  assert.equal(h.notifications.length, 1);
  assert.equal(h.active('interval').length, 0);
  assert.equal(h.callbacks.has('Escape'), false);
  await h.finishCleanup();
  assert.equal(h.active('timeout').length, 0);
});

test('token-scoped stale cleanup cannot disarm a newer session or issue requests while inactive', async () => {
  const h = harness();
  assert.deepEqual(await h.guard.stop('retired-token'), { ok: true, stale: true });
  assert.equal(h.requests.length, 0);
  const oldSession = await h.start();
  const stopping = h.guard.stop(oldSession.token);
  await flush();
  assert.deepEqual(h.latest('eeg_arm').payload, { enabled: false });
  await h.finishCleanup();
  await stopping;
  const newSession = await h.start();
  const requestCount = h.requests.length;
  const unregisterCount = h.unregistrations.length;
  assert.notEqual(newSession.token, oldSession.token);
  assert.deepEqual(await h.guard.stop(oldSession.token), { ok: true, stale: true });
  assert.equal(h.requests.length, requestCount);
  assert.equal(h.unregistrations.length, unregisterCount);
  assert.equal(h.callbacks.has('Escape'), true);
  assert.equal(h.active('interval').length, 1);
  const finalStop = h.guard.stop(newSession.token);
  await flush();
  await h.finishCleanup();
  assert.equal((await finalStop).ok, true);
  assert.equal(h.callbacks.has('Escape'), false);
  assert.equal(h.active('interval').length, 0);
});

async function mainHarness() {
  const app = new EventEmitter(), ipcMain = new EventEmitter(), handlers = new Map();
  const guardCalls = [], sent = [], registrations = [], windows = [];
  let guardOptions, resolveDispose, desktopStops = 0, quitCalls = 0, unregisterAllCalls = 0;
  app.whenReady = () => Promise.resolve();
  app.quit = () => { quitCalls += 1; };
  ipcMain.handle = (channel, callback) => handlers.set(channel, callback);
  class FakeWindow extends EventEmitter {
    constructor() {
      super();
      this.destroyed = false;
      this.hidden = false;
      this.webContents = new EventEmitter();
      this.webContents.destroyed = false;
      this.webContents.isDestroyed = () => this.webContents.destroyed;
      this.webContents.send = (channel, payload) => sent.push({ channel, payload });
      windows.push(this);
    }
    isDestroyed() { return this.destroyed; }
    setBackgroundMaterial() {}
    loadFile() {}
    hide() { this.hidden = true; }
    isVisible() { return !this.hidden; }
    show() { this.hidden = false; }
    showInactive() { this.hidden = false; }
    focus() {}
    isFocused() { return false; }
    setSize() {}
  }
  class FakeGuard {
    constructor(options) { guardOptions = options; }
    start() { guardCalls.push('start'); return Promise.resolve({ ok: true, token: 'native-guard' }); }
    stop(token) { guardCalls.push(token === undefined ? 'stop' : `stop:${token}`); return Promise.resolve({ ok: true }); }
    emergencyStop(reason) { guardCalls.push(`emergency:${reason}`); return Promise.resolve({ ok: true }); }
    dispose() {
      guardCalls.push('dispose');
      return new Promise(resolve => { resolveDispose = resolve; });
    }
  }
  class FakeVisibility {
    constructor() { this.status = {}; }
    start() {}
    stop() { desktopStops += 1; }
    refresh() {}
  }
  const shortcuts = {
    register: (key, callback) => { registrations.push({ key, callback }); return true; },
    unregisterAll: () => { unregisterAllCalls += 1; },
  };
  const filename = path.join(__dirname, '../ui/main.js');
  vm.runInNewContext(fs.readFileSync(filename, 'utf8'), {
    require(name) {
      if (name === 'electron') return { app, BrowserWindow: FakeWindow, globalShortcut: shortcuts,
        ipcMain, screen: { getPrimaryDisplay: () => ({ workAreaSize: { width: 1920, height: 1080 } }) } };
      if (name === 'path') return path;
      if (name === './desktop-visibility') return { DesktopVisibility: FakeVisibility };
      if (name === './eeg-guard.cjs') return { EegGuard: FakeGuard };
      if (name === './renderer/multimodal-http.cjs') return {
        requestMultimodal() { assert.fail('The main lifecycle test must not make a transport request'); },
      };
      assert.fail(`Unexpected dependency: ${name}`);
    },
    __dirname: path.dirname(filename), process: { env: {} }, Promise,
    setTimeout() { assert.fail('The main lifecycle test must not start a real timer'); },
  }, { filename });
  await flush();
  return { app, handlers, guardCalls, sent, registrations, window: windows[0],
    guardOptions: () => guardOptions, resolveDispose: () => resolveDispose({ ok: true }),
    counts: () => ({ desktopStops, quitCalls, unregisterAllCalls }) };
}

test('native IPC only accepts guard requests from the current live HUD', async () => {
  const h = await mainHarness();
  for (const channel of ['eeg-guard-start', 'eeg-guard-stop']) {
    assert.equal((await h.handlers.get(channel)({ sender: {} })).ok, false);
  }
  assert.deepEqual(h.guardCalls, []);
  assert.equal((await h.handlers.get('eeg-guard-start')({ sender: h.window.webContents })).token, 'native-guard');
  assert.equal((await h.handlers.get('eeg-guard-stop')({ sender: h.window.webContents })).ok, true);
  assert.deepEqual(h.guardCalls, ['start', 'stop']);
  h.window.destroyed = true;
  for (const channel of ['eeg-guard-start', 'eeg-guard-stop']) {
    assert.equal((await h.handlers.get(channel)({ sender: h.window.webContents })).ok, false);
  }
  assert.deepEqual(h.guardCalls, ['start', 'stop']);
});

test('native cleanup IPC validates and forwards its optional session token', async () => {
  const h = await mainHarness();
  const cleanup = h.handlers.get('eeg-guard-stop');
  const event = { sender: h.window.webContents };
  for (const payload of [null, [], 'token', { token: 7 }, { token: '' },
    { token: 'not-a-uuid' }, { enabled: false }, { token: 'retired', extra: true }]) {
    assert.equal((await cleanup(event, payload)).ok, false);
  }
  assert.deepEqual(h.guardCalls, []);
  const token = '01234567-89ab-4cde-8f01-23456789abcd';
  assert.equal((await cleanup(event, { token })).ok, true);
  assert.deepEqual(h.guardCalls, [`stop:${token}`]);
});

test('hiding the HUD retains its guard; renderer or window destruction invokes emergency stop', async () => {
  const h = await mainHarness();
  let prevented = 0;
  h.window.emit('close', { preventDefault: () => { prevented += 1; } });
  assert.equal(prevented, 1);
  assert.equal(h.window.hidden, true);
  assert.deepEqual(h.guardCalls, []);
  h.window.webContents.emit('render-process-gone');
  h.window.webContents.emit('destroyed');
  h.window.emit('closed');
  assert.deepEqual(h.guardCalls, [
    'emergency:renderer-crash', 'emergency:renderer-destroyed', 'emergency:window-destroyed',
  ]);
  assert.equal(h.counts().desktopStops, 1);
});

test('native emergency notification targets the live HUD and ignores a destroyed renderer', async () => {
  const h = await mainHarness();
  h.guardOptions().notify('escape');
  assert.equal(h.sent.length, 1);
  assert.equal(h.sent[0].channel, 'eeg-emergency-stop');
  assert.equal(h.sent[0].payload.reason, 'escape');
  h.window.webContents.destroyed = true;
  h.guardOptions().notify('shutdown');
  assert.equal(h.sent.length, 1);
});

test('application quit awaits guard disposal once and rejects new starts during cleanup', async () => {
  const h = await mainHarness();
  let prevented = 0;
  const quitting = { preventDefault: () => { prevented += 1; } };
  h.app.emit('before-quit', quitting);
  h.app.emit('before-quit', quitting);
  assert.deepEqual(h.guardCalls, ['dispose']);
  assert.equal(h.counts().quitCalls, 0);
  assert.equal((await h.handlers.get('eeg-guard-start')({ sender: h.window.webContents })).ok, false);
  h.resolveDispose();
  await flush();
  assert.equal(h.counts().quitCalls, 1);
  assert.equal(prevented, 2);
  h.app.emit('before-quit', quitting);
  assert.equal(prevented, 2, 'The completed cleanup must allow the final quit');
  h.app.emit('will-quit');
  assert.equal(h.counts().unregisterAllCalls, 1);
  h.registrations.find(item => item.key === 'CommandOrControl+Q').callback();
  assert.equal(h.counts().quitCalls, 2, 'The exit shortcut must use normal quit cleanup');
});
