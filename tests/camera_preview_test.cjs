const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { EventEmitter } = require('node:events');

function transport() {
  const request = new EventEmitter();
  request.destroy = error => { request.emit('error', error); request.emit('close'); };
  const response = new EventEmitter();
  response.statusCode = 200;
  response.resume = () => {};
  let options, receive, deadline, cleared = false;
  const module = { exports: {} };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../ui/renderer/camera-preview.cjs'), 'utf8'), {
    module, Buffer, require: () => ({ get: (opts, callback) => { options = opts; receive = callback; return request; } }),
    setTimeout: (fn, delay) => { assert.equal(delay, 1500); deadline = fn; return 1; },
    clearTimeout: () => { cleared = true; },
  });
  const promise = module.exports.readCameraPreview();
  return { promise, request, response, options, start: () => receive(response),
    expire: () => deadline(), get cleared() { return cleared; } };
}

test('preview transport is fixed to localhost and assembles chunked JSON', async () => {
  const t = transport();
  assert.equal(t.options.hostname, '127.0.0.1');
  assert.equal(t.options.path, '/gestures/preview');
  assert.equal(t.options.headers['X-Intuition-Preview'], '1');
  assert.equal(t.options.headers.Origin, undefined);
  t.start();
  t.response.emit('data', Buffer.from('{"tracked":'));
  t.response.emit('data', Buffer.from('true}'));
  t.response.emit('end'); t.request.emit('close');
  assert.equal((await t.promise).tracked, true);
  assert.equal(t.cleared, true);
});

test('preview transport rejects oversized and interrupted responses', async () => {
  const big = transport(); big.start();
  const bigResult = assert.rejects(big.promise, /too large/);
  big.response.emit('data', Buffer.alloc(1024 * 1024 + 1));
  await bigResult;
  const aborted = transport(); aborted.start();
  const abortedResult = assert.rejects(aborted.promise, /interrupted/);
  aborted.response.emit('aborted'); aborted.request.emit('close');
  await abortedResult;
});

test('preview transport has an absolute deadline and rejects HTTP or JSON failures', async () => {
  const slow = transport(); slow.start();
  const slowResult = assert.rejects(slow.promise, /timed out/);
  slow.expire(); await slowResult;
  const denied = transport(); denied.response.statusCode = 403;
  const deniedResult = assert.rejects(denied.promise, /unavailable/);
  denied.start(); denied.request.emit('close'); await deniedResult;
  const invalid = transport(); invalid.start();
  const invalidResult = assert.rejects(invalid.promise);
  invalid.response.emit('data', Buffer.from('not json')); invalid.response.emit('end');
  invalid.request.emit('close'); await invalidResult;
});
