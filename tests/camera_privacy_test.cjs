const test = require('node:test');
const assert = require('node:assert/strict');
const { createCameraPrivacyIndicator } = require('../ui/renderer/camera-privacy.cjs');

function fixture() {
  let time = 0, connected = true;
  const timers = [], requests = [], classes = new Set(), attributes = {};
  const element = { classList: { toggle: (name, on) => on ? classes.add(name) : classes.delete(name) },
    setAttribute: (key, value) => attributes[key] = value };
  const indicator = createCameraPrivacyIndicator({ element, now: () => time, isConnected: () => connected,
    request: () => new Promise((resolve, reject) => requests.push({ resolve, reject })),
    schedule: fn => { timers.push({ fn }); return timers.length; },
    cancel: id => { if (timers[id - 1]) timers[id - 1].cancelled = true; } });
  indicator.connectionChanged();
  return { indicator, element, classes, requests, attributes,
    advance: milliseconds => { time += milliseconds; const timer = timers.find(t => !t.cancelled && !t.ran);
      if (timer) { timer.ran = true; timer.fn(); } },
    elapseHidden: milliseconds => { time += milliseconds; },
    connect: value => { connected = value; indicator.connectionChanged(); } };
}
const data = (state, revision = 1, extra = {}) => ({ state, revision, instance_id: 'server1',
  in_use: state === 'off' ? false : state === 'starting' ? null : true,
  release_confirmed: state === 'off', owners: state === 'off' ? [] : ['eeg controls'], ...extra });
const flush = () => new Promise(resolve => setImmediate(resolve));

test('actual ownership covers either camera and only confirmed aggregate release turns yellow off', () => {
  const f = fixture();
  assert.equal(f.element.textContent, 'CAMERA ?');
  for (const owner of ['hand controls', 'eeg controls']) {
    f.indicator.accept(data('on', owner === 'hand controls' ? 1 : 2, { owners: [owner] }));
    assert.equal(f.element.textContent, 'CAMERA ON');
    assert.ok(f.classes.has('in-use'));
    assert.match(f.element.title, /Closing a preview does not stop/);
  }
  f.indicator.accept(data('off', 3));
  assert.equal(f.element.textContent, 'CAMERA OFF');
  assert.ok(!f.classes.has('in-use') && !f.classes.has('uncertain'));
});

test('opening stopping and failed cleanup stay yellow without pretending that tracking is running', () => {
  const f = fixture();
  f.indicator.accept(data('starting'));
  assert.equal(f.element.textContent, 'CAMERA STARTING');
  assert.ok(f.classes.has('uncertain'));
  f.indicator.accept(data('stopping', 2));
  assert.equal(f.element.textContent, 'CAMERA STOPPING');
  assert.ok(f.classes.has('in-use'));
  f.indicator.accept(data('unknown', 3));
  assert.equal(f.element.textContent, 'CAMERA ON');
  assert.match(f.element.title, /release is unconfirmed/i);
  assert.ok(f.classes.has('uncertain'));
});

test('panel-independent polling never starts capture and stale or disconnected status cannot claim OFF', async () => {
  const f = fixture();
  await flush();
  assert.equal(f.requests.length, 1);
  f.requests[0].resolve({ camera_privacy: data('off') });
  await flush();
  assert.equal(f.element.textContent, 'CAMERA OFF');
  f.advance(4000);
  assert.equal(f.element.textContent, 'CAMERA ?');
  await flush();
  assert.equal(f.requests.length, 2);
  f.requests[1].resolve({ camera_privacy: data('on', 2) });
  await flush();
  assert.equal(f.element.textContent, 'CAMERA ON');
  f.connect(false);
  assert.equal(f.element.textContent, 'CAMERA ?');
  assert.ok(f.classes.has('uncertain'));
});

test('late HTTP replies cannot overwrite newer websocket ownership or survive a connection epoch', async () => {
  const f = fixture();
  await flush();
  f.indicator.accept(data('on', 5));
  f.requests[0].resolve({ camera_privacy: data('off', 4) });
  await flush();
  assert.equal(f.element.textContent, 'CAMERA ON');
  f.advance(1000); await flush();
  f.connect(false); f.connect(true); await flush();
  f.requests[1].resolve({ camera_privacy: data('off', 6) });
  await flush();
  assert.equal(f.element.textContent, 'CAMERA ?');
  f.requests[2].resolve({ camera_privacy: data('on', 1, { instance_id: 'server2' }) });
  await flush();
  assert.equal(f.element.textContent, 'CAMERA ON');
  assert.equal(f.indicator.accept(data('off', 99)), false);
});

test('malformed contradictory or same-revision conflicting release messages cannot extinguish yellow', () => {
  const f = fixture();
  f.indicator.accept(data('on', 5));
  for (const invalid of [data('off', 5), data('off', 6, { in_use: true }),
    data('off', 6, { release_confirmed: false }), data('off', 6, { revision: NaN }),
    data('off', 6, { owners: ['eeg'] }),
    data('off', 6, { owners: 'hand controls' }), null]) {
    assert.equal(f.indicator.accept(invalid), false);
    assert.equal(f.element.textContent, 'CAMERA ON');
  }
  // Object key order and extra display fields do not invalidate an unchanged heartbeat.
  assert.equal(f.indicator.accept({ ...data('on', 5), optional: true }), true);
});

test('request failure is unknown, with no overlapping metadata request', async () => {
  const f = fixture();
  await flush();
  f.indicator.accept(data('off'));
  f.advance(1000); await flush();
  assert.equal(f.requests.length, 1);
  f.requests[0].reject(new Error('backend unavailable')); await flush();
  assert.equal(f.element.textContent, 'CAMERA ?');
  assert.ok(f.classes.has('uncertain'));
});

test('showing the HUD invalidates stale OFF immediately even when hidden timers did not run', async () => {
  const f = fixture();
  await flush();
  f.requests[0].resolve({ camera_privacy: data('off') }); await flush();
  f.elapseHidden(10000);
  f.indicator.refresh();
  assert.equal(f.element.textContent, 'CAMERA ?');
  await flush();
  assert.equal(f.requests.length, 2);
  f.requests[1].resolve({ camera_privacy: data('on', 2) }); await flush();
  assert.equal(f.element.textContent, 'CAMERA ON');
});
