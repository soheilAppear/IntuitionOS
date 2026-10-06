// The transport is injected: no backend, Bluetooth device, or Electron launch.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { EventEmitter } = require('node:events');

function transport(action, deviceId) {
  const request = new EventEmitter();
  request.destroy = () => { request.destroyed = true; request.emit('close'); };
  request.end = body => { request.body = body; };
  const response = new EventEmitter();
  response.statusCode = 200;
  response.complete = true;
  const calls = [];
  const timers = [];
  const module = { exports: {} };
  let receive;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../ui/renderer/brainbit-http.cjs'), 'utf8'), {
    module, Buffer,
    require: name => {
      assert.equal(name, 'node:http');
      return { request: (options, callback) => { calls.push(options); receive = callback; return request; } };
    },
    setTimeout: (callback, delay) => { timers.push({ callback, delay, cleared: false }); return timers.length; },
    clearTimeout: id => { timers[id - 1].cleared = true; },
  });
  const promise = module.exports.requestBrainbit(action, deviceId);
  return { promise, request, response, calls, timers,
    get options() { return calls[0]; },
    start: () => receive(response),
    complete: (data, status = 200) => {
      response.statusCode = status;
      receive(response);
      response.emit('data', Buffer.from(JSON.stringify(data)));
      response.emit('end');
      response.emit('close');
      request.emit('close');
    },
    expire: () => timers[0].callback(),
  };
}

test('default status reads fixed localhost with the native header and no body or browser Origin', async () => {
  const t = transport();
  assert.equal(t.options.hostname, '127.0.0.1');
  assert.equal(t.options.port, 7432);
  assert.equal(t.options.path, '/brainbit/status');
  assert.equal(t.options.method, 'GET');
  assert.equal(t.options.headers['X-Intuition-Brainbit'], '1');
  assert.equal(t.options.headers['Content-Type'], undefined);
  assert.ok(!Object.keys(t.options.headers).some(name => name.toLowerCase() === 'origin'));
  assert.equal(t.request.body, undefined);
  assert.equal(t.timers[0].delay, 5000);
  t.start();
  t.response.emit('data', Buffer.from('{"state":'));
  t.response.emit('data', Buffer.from('"disconnected"}'));
  t.response.emit('end');
  const result = await t.promise;
  assert.equal(result.ok, true);
  assert.equal(result.status, 200);
  assert.equal(result.data.state, 'disconnected');
  assert.equal(t.timers[0].cleared, true);
});

for (const action of ['discover', 'connect', 'disconnect', 'refresh']) {
  test(`${action} uses a fixed POST path and the expected JSON body`, async () => {
    const deviceId = 'device/<id> & "name"';
    const t = transport(action, deviceId);
    assert.equal(t.options.hostname, '127.0.0.1');
    assert.equal(t.options.port, 7432);
    assert.equal(t.options.path, `/brainbit/${action}`);
    assert.equal(t.options.method, 'POST');
    assert.equal(t.options.headers['X-Intuition-Brainbit'], '1');
    assert.equal(t.options.headers['Content-Type'], 'application/json');
    assert.ok(!Object.keys(t.options.headers).some(name => name.toLowerCase() === 'origin'));
    assert.deepEqual(JSON.parse(t.request.body), action === 'connect' ? { device_id: deviceId } : {});
    assert.equal(t.timers[0].delay, 30000);
    t.complete({ state: 'pending' }, 202);
    const result = await t.promise;
    assert.equal(result.ok, true);
    assert.equal(result.status, 202);
    assert.equal(result.data.state, 'pending');
    assert.equal(t.calls.length, 1);
    assert.equal(t.timers[0].cleared, true);
  });
}

test('invalid actions and missing or blank connection IDs reject before network access', async () => {
  for (const action of ['../connect', 'status?discover', 'http://example.com', 'GET', '', null, {}, 1]) {
    const t = transport(action);
    await assert.rejects(t.promise, /Invalid BrainBit action/);
    assert.equal(t.calls.length, 0);
    assert.equal(t.timers.length, 0);
  }
  for (const deviceId of [undefined, null, '', ' \t\n', 0, {}, ['device']]) {
    const t = transport('connect', deviceId);
    await assert.rejects(t.promise, /device ID is required/);
    assert.equal(t.calls.length, 0);
    assert.equal(t.timers.length, 0);
  }
});

test('HTTP failures retain parsed error details and redirects are never followed', async () => {
  for (const status of [302, 400, 403, 409, 500]) {
    const t = transport('refresh');
    t.response.headers = { location: 'https://example.com/other' };
    t.complete({ error: 'Device unavailable', brainbit: { state: 'error' } }, status);
    const result = await t.promise;
    assert.equal(result.ok, false);
    assert.equal(result.status, status);
    assert.equal(result.data.error, 'Device unavailable');
    assert.equal(result.data.brainbit.state, 'error');
    assert.equal(t.calls.length, 1);
    assert.equal(t.timers[0].cleared, true);
  }
});

test('the response limit accepts 128 KiB and rejects a byte more across chunks', async () => {
  const limit = 128 * 1024;
  const exact = transport();
  exact.start();
  exact.response.emit('data', Buffer.from('"' + 'a'.repeat(limit - 2) + '"'));
  exact.response.emit('end');
  assert.equal((await exact.promise).data.length, limit - 2);

  const big = transport();
  const rejected = assert.rejects(big.promise, /too large/);
  big.start();
  big.response.emit('data', Buffer.alloc(limit));
  big.response.emit('data', Buffer.from('x'));
  await rejected;
  assert.equal(big.request.destroyed, true);
  assert.equal(big.timers[0].cleared, true);
});

for (const action of ['status', 'discover']) {
  test(`${action} times out on an absolute deadline even after partial data`, async () => {
    const t = transport(action);
    const rejected = assert.rejects(t.promise, /timed out/);
    t.start();
    t.response.emit('data', Buffer.from('{'));
    t.response.emit('data', Buffer.from('"state":'));
    assert.equal(t.timers.length, 1, 'incoming data must not reset the deadline');
    t.expire();
    await rejected;
    assert.equal(t.request.destroyed, true);
    assert.equal(t.timers[0].cleared, true);
    t.response.emit('data', Buffer.from('"connected"}'));
    t.response.emit('end');
    assert.equal(t.calls.length, 1);
  });
}

test('invalid JSON, empty bodies, and invalid error bodies reject', async () => {
  for (const [body, status] of [['not json', 200], ['', 204], ['<html>error</html>', 500]]) {
    const t = transport();
    const rejected = assert.rejects(t.promise);
    t.response.statusCode = status;
    t.start();
    t.response.emit('data', Buffer.from(body));
    t.response.emit('end');
    await rejected;
    assert.equal(t.timers[0].cleared, true);
  }
});

test('network and response errors reject and clear the deadline', async () => {
  for (const source of ['request', 'response']) {
    const t = transport();
    const rejected = assert.rejects(t.promise, /connection failed/);
    if (source === 'response') t.start();
    t[source].emit('error', new Error('connection failed'));
    await rejected;
    assert.equal(t.timers[0].cleared, true);
  }
});

test('aborted responses and premature closes reject without waiting for timeout', async () => {
  for (const event of ['aborted', 'response close', 'request close', 'incomplete end']) {
    const t = transport();
    const rejected = assert.rejects(t.promise, /interrupted/);
    t.start();
    t.response.emit('data', Buffer.from('{}'));
    if (event === 'aborted') t.response.emit('aborted');
    if (event === 'response close') t.response.emit('close');
    if (event === 'request close') t.request.emit('close');
    if (event === 'incomplete end') { t.response.complete = false; t.response.emit('end'); }
    await rejected;
    assert.equal(t.timers[0].cleared, true);
  }
});
