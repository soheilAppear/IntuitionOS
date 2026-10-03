// Desktop pinning lifecycle with fake windows, child processes and timers only.
const test = require('node:test');
const assert = require('node:assert/strict');
const { DesktopVisibility, nativeHandle } = require('../ui/desktop-visibility');

function harness({ platform = 'win32', hwnd = 0xfedcba9876543210n, workspaceVisible = true } = {}) {
  const calls = [], timers = [], statuses = [], workspaceCalls = [];
  const handle = Buffer.alloc(8);
  handle.writeBigUInt64LE(hwnd);
  const window = {
    destroyed: false,
    isDestroyed() { return this.destroyed; },
    getNativeWindowHandle: () => handle,
    setVisibleOnAllWorkspaces: (...args) => workspaceCalls.push(args),
    isVisibleOnAllWorkspaces: () => workspaceVisible,
    focus() { assert.fail('Desktop pinning must not focus the HUD'); },
    show() { assert.fail('Desktop pinning must not show the HUD'); },
    showInactive() { assert.fail('Desktop pinning must not change its current visibility'); },
    hide() { assert.fail('Desktop pinning must not hide the HUD'); },
  };
  const run = (executable, args, options, callback) => {
    const child = { killed: false, kill() { this.killed = true; } };
    calls.push({ executable, args, options, callback, child });
    return child;
  };
  const visibility = new DesktopVisibility({
    window, platform, python: 'C:\\Intuition OS\\.venv\\Scripts\\python.exe',
    projectRoot: 'C:\\Intuition OS', pid: 4321, onStatus: status => statuses.push(status), run,
    later: (callback, delay) => {
      const timer = { callback, delay, cancelled: false, fired: false };
      timers.push(timer);
      return timer;
    },
    cancel: timer => { timer.cancelled = true; },
  });
  const success = (index = calls.length - 1, overrides = {}) => calls[index].callback(null, JSON.stringify({
    ok: true, pinned: true, hwnd: hwnd.toString(), pid: 4321, ...overrides,
  }));
  const fire = (index = timers.length - 1) => {
    const timer = timers[index];
    assert.ok(!timer.cancelled, 'Test must not deliver a cancelled timer');
    timer.fired = true;
    timer.callback();
  };
  return { visibility, window, handle, calls, timers, statuses, workspaceCalls, success, fire };
}

test('native HWND conversion preserves all unsigned 64 bits and supports 32-bit handles', () => {
  const h = harness();
  assert.equal(nativeHandle(h.window), '18364758544493064720');
  const handle32 = Buffer.alloc(4);
  handle32.writeUInt32LE(0xf1234567);
  assert.equal(nativeHandle({ getNativeWindowHandle: () => handle32 }), '4045620583');
});

test('Windows checks only its exact HWND and process using a bounded hidden execFile call', () => {
  const h = harness();
  h.visibility.start();
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0].executable, 'C:\\Intuition OS\\.venv\\Scripts\\python.exe');
  assert.deepEqual(h.calls[0].args, [
    '-m', 'core.hud_desktops', '--hwnd', '18364758544493064720', '--pid', '4321',
  ]);
  assert.deepEqual(h.calls[0].options, {
    cwd: 'C:\\Intuition OS', windowsHide: true, timeout: 6000, maxBuffer: 16384,
  });
  assert.equal(h.visibility.status.pinned, false);
  h.success();
  assert.deepEqual(h.visibility.status, {
    state: 'pinned', pinned: true, text: 'HUD visible on all virtual desktops.',
  });
  assert.deepEqual(h.timers.map(t => t.delay), [30000]);
  assert.equal(h.workspaceCalls.length, 0);
});

for (const [label, overrides] of [
  ['wrong window handle', { hwnd: '18364758544493064721' }],
  ['rounded numeric handle', { hwnd: Number(0xfedcba9876543210n) }],
  ['wrong process', { pid: 5432 }],
  ['string process', { pid: '4321' }],
  ['non-boolean success', { ok: 1 }],
  ['unconfirmed pin', { pinned: false }],
  ['string pin', { pinned: 'true' }],
]) {
  test(`pinning rejects ${label}`, () => {
    const h = harness();
    h.visibility.start();
    h.success(0, overrides);
    assert.equal(h.visibility.status.pinned, false);
    assert.ok(!h.statuses.some(status => status.state === 'pinned'));
    assert.equal(h.timers.at(-1).delay, 1000);
  });
}

test('malformed helper output and process errors never claim the HUD is pinned', () => {
  for (const [error, stdout] of [
    [null, 'not JSON'], [null, 'null'], [null, '[]'],
    [new Error('helper timed out'), JSON.stringify({ ok: true, pinned: true, hwnd: '12', pid: 4321 })],
  ]) {
    const h = harness({ hwnd: 12n });
    h.visibility.start();
    h.calls[0].callback(error, stdout);
    assert.equal(h.visibility.status.pinned, false);
    assert.equal(h.timers.at(-1).delay, 1000);
  }
});

test('early failures retry with backoff, explain persistent failure, and recover', () => {
  const h = harness();
  h.visibility.start();
  for (const delay of [1000, 2000, 3000, 30000]) {
    h.calls.at(-1).callback(new Error('Explorer is not ready'), '');
    assert.equal(h.timers.at(-1).delay, delay);
    if (delay >= 3000) assert.equal(h.visibility.status.state, 'unavailable');
    h.fire();
  }
  h.success();
  assert.equal(h.visibility.status.pinned, true);
  assert.equal(h.visibility.failures, 0);
  assert.equal(h.timers.at(-1).delay, 30000);
  h.fire();
  assert.equal(h.calls.length, 6, 'A successful pin must still be checked after a shell restart');
  h.calls.at(-1).callback(new Error('Explorer restarted'), '');
  assert.equal(h.visibility.status.pinned, false);
  assert.equal(h.timers.at(-1).delay, 1000);
});

test('start is idempotent and manual refresh cannot overlap a helper', () => {
  const h = harness();
  h.visibility.start();
  h.visibility.start();
  h.visibility.refresh();
  assert.equal(h.calls.length, 1);
  h.success();
  const periodic = h.timers[0];
  h.visibility.refresh();
  assert.ok(periodic.cancelled);
  assert.equal(h.calls.length, 2);
});

test('a synchronous launch error releases busy state and schedules a retry', () => {
  const h = harness();
  h.visibility.run = () => { throw new Error('Python is missing'); };
  h.visibility.start();
  assert.equal(h.visibility.busy, false);
  assert.equal(h.visibility.child, null);
  assert.equal(h.visibility.status.pinned, false);
  assert.equal(h.timers[0].delay, 1000);
});

test('stop kills an in-flight helper and ignores its late success', () => {
  const h = harness();
  h.visibility.start();
  h.visibility.stop();
  assert.ok(h.calls[0].child.killed);
  const count = h.statuses.length;
  h.success();
  assert.equal(h.statuses.length, count);
  assert.equal(h.timers.length, 0);
  assert.equal(h.visibility.child, null);
  h.visibility.refresh();
  assert.equal(h.calls.length, 1);
});

test('stop cancels periodic verification and destroyed windows cannot restart a helper', () => {
  const h = harness();
  h.visibility.start();
  h.success();
  h.visibility.stop();
  assert.ok(h.timers[0].cancelled);
  h.window.destroyed = true;
  h.visibility.start();
  assert.equal(h.calls.length, 1);
});

test('stop then restart starts a fresh helper and ignores the prior generation callback', () => {
  const h = harness();
  h.visibility.start();
  const previous = h.calls[0];
  h.visibility.stop();
  h.visibility.start();
  assert.equal(h.calls.length, 2, 'Restart must not remain busy with the killed child');
  const current = h.calls[1].child;
  const count = h.statuses.length;
  previous.callback(null, JSON.stringify({ ok: true, pinned: true,
    hwnd: '18364758544493064720', pid: 4321 }));
  assert.equal(h.statuses.length, count);
  assert.equal(h.visibility.child, current);
  assert.equal(h.visibility.busy, true);
  assert.equal(h.timers.length, 0);
  h.success(1);
  assert.equal(h.visibility.status.pinned, true);
});

test('non-Windows uses Electron workspace visibility without launching or focusing anything', () => {
  const h = harness({ platform: 'darwin' });
  h.visibility.start();
  assert.deepEqual(h.workspaceCalls, [[true, { visibleOnFullScreen: true }]]);
  assert.equal(h.visibility.status.pinned, true);
  assert.equal(h.calls.length, 0);
  assert.equal(h.timers.length, 0);
});

test('non-Windows requires acknowledgement and reports unsupported workspace APIs', () => {
  const h = harness({ platform: 'linux', workspaceVisible: false });
  h.visibility.start();
  assert.equal(h.visibility.status.state, 'unavailable');
  assert.equal(h.visibility.status.pinned, false);
  assert.equal(h.calls.length, 0);
  h.window.setVisibleOnAllWorkspaces = () => { throw new Error('Unsupported'); };
  h.visibility.refresh();
  assert.equal(h.visibility.status.pinned, false);
});
