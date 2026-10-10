// Synthetic transport only: no HTTP connection or acquisition is started.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { EventEmitter } = require('node:events');

function transport(action, payload) {
  const req = new EventEmitter(), res = new EventEmitter(), calls = [], timers = [];
  req.end = body => { req.body = body; };
  req.destroy = () => { req.destroyed = true; req.emit('close'); };
  res.statusCode = 200; res.complete = true;
  let receive;
  const module = { exports: {} };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../ui/renderer/multimodal-http.cjs'), 'utf8'), {
    module, Buffer, require: () => ({ request: (options, callback) => { calls.push(options); receive = callback; return req; } }),
    setTimeout: (fn, delay) => { timers.push({ fn, delay }); return timers.length; },
    clearTimeout: id => { if (timers[id - 1]) timers[id - 1].cancelled = true; },
  });
  const promise = module.exports.requestMultimodal(action, payload);
  return { promise, req, res, calls, timers, start: () => receive(res),
    complete(data, status = 200) { res.statusCode = status; receive(res); res.emit('data', Buffer.from(JSON.stringify(data))); res.emit('end'); } };
}

const guardToken = '11111111-2222-4333-8444-555555555555';
for (const [action, payload] of [['status', {}], ['preview', {}], ['start', {}], ['stop', {}],
  ['contact', {}], ['arm', { enabled: true }], ['calibrate', { label: 'left' }], ['reset_calibration', {}],
  ['eeg_trial', { label: 'rest', phase: 'train' }], ['eeg_trial', { label: 'right', phase: 'validate' }],
  ['eeg_train', {}], ['eeg_reset', {}], ['eeg_arm', { enabled: true, guard_token: guardToken }],
  ['eeg_arm', { enabled: false }], ['eeg_guard', { token: guardToken }], ['control_mode', { mode: 'eeg' }]]) {
  test(`local ${action} request has constrained method, payload and native header`, async () => {
    const t = transport(action, payload), get = ['status', 'preview'].includes(action), options = t.calls[0];
    assert.equal(options.hostname, '127.0.0.1'); assert.equal(options.port, 7432);
    assert.equal(options.path, `/multimodal/${action}`); assert.equal(options.method, get ? 'GET' : 'POST');
    assert.equal(options.headers['X-Intuition-Multimodal'], '1');
    assert.ok(!Object.keys(options.headers).some(key => key.toLowerCase() === 'origin'));
    if (get) assert.equal(t.req.body, undefined); else assert.deepEqual(JSON.parse(t.req.body), payload);
    assert.equal(t.timers[0].delay, get ? 5000 : 30000);
    t.complete({ state: 'stopped' });
    assert.equal((await t.promise).ok, true); assert.equal(t.timers[0].cancelled, true);
  });
}

test('arbitrary routes and extra action fields are rejected before network access', async () => {
  for (const [action, payload] of [['record', {}], ['../start', {}], ['http://host', {}], ['arm', {}],
    ['arm', { enabled: 'true' }], ['arm', { enabled: true, save: true }], ['calibrate', { label: 'up' }],
    ['calibrate', { label: 'left', samples: [] }], ['status', { raw: true }], ['stop', { anything: 1 }],
    ['eeg_trial', { label: 'rest' }], ['eeg_trial', { label: 'up', phase: 'train' }],
    ['eeg_trial', { label: 'left', phase: 'test' }], ['eeg_trial', { label: 'rest', phase: 'train', raw: [] }],
    ['eeg_train', { save: true }], ['eeg_reset', { record: true }],
    ['eeg_arm', { enabled: true }], ['eeg_arm', { enabled: true, guard_token: 'not-a-uuid' }],
    ['eeg_arm', { enabled: false, guard_token: guardToken }], ['eeg_guard', { token: guardToken, arm: true }],
    ['eeg_guard', { token: 'bad' }], ['eeg_guard', { token: '11111111-2222-1333-8444-555555555555' }],
    ['eeg_guard', { token: '11111111-2222-4333-7444-555555555555' }],
    ['control_mode', { mode: 'both' }], ['control_mode', { mode: 'eeg', enabled: true }]]) {
    const t = transport(action, payload);
    await assert.rejects(t.promise, /Invalid preview/); assert.equal(t.calls.length, 0);
  }
});

test('HTTP errors stay errors, redirects are not followed and malformed data rejects', async () => {
  for (const code of [302, 400, 409, 503]) {
    const t = transport('start'); t.complete({ error: 'Synthetic camera error' }, code);
    const result = await t.promise;
    assert.equal(result.ok, false); assert.equal(result.status, code); assert.equal(t.calls.length, 1);
  }
  const t = transport(); t.start(); t.res.emit('data', Buffer.from('not-json')); t.res.emit('end');
  await assert.rejects(t.promise);
});

test('preview response limit and deadline bound a hung or oversized stream', async () => {
  const big = transport('preview'); big.start(); big.res.emit('data', Buffer.alloc(512 * 1024 + 1));
  await assert.rejects(big.promise, /too large/); assert.equal(big.req.destroyed, true);
  const slow = transport('preview'); slow.start(); slow.res.emit('data', Buffer.from('{'));
  slow.timers[0].fn(); await assert.rejects(slow.promise, /timed out/); assert.equal(slow.req.destroyed, true);
});

test('network errors and interrupted responses reject without retaining a deadline', async () => {
  const broken = transport('stop'); broken.req.emit('error', new Error('offline'));
  await assert.rejects(broken.promise, /offline/); assert.equal(broken.timers[0].cancelled, true);
  const aborted = transport('preview'); aborted.start(); aborted.res.emit('aborted');
  await assert.rejects(aborted.promise, /interrupted/); assert.equal(aborted.timers[0].cancelled, true);
});
