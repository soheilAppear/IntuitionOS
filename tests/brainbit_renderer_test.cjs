// Connection/status UI only: no Electron, SDK, network or device is started.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function renderer() {
  class Element {
    constructor(tag = 'div') {
      Object.assign(this, { tagName: tag, value: '', textContent: '', innerHTML: '',
        children: [], listeners: {}, attributes: {}, style: {}, open: false });
      const classes = new Set();
      this.classList = { add: (...names) => names.forEach(n => classes.add(n)),
        remove: (...names) => names.forEach(n => classes.delete(n)),
        contains: n => classes.has(n),
        toggle: (n, force = !classes.has(n)) => force ? classes.add(n) : classes.delete(n) };
    }
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
    dispatch(type) { for (const fn of this.listeners[type] || []) fn({ preventDefault() {} }); }
    appendChild(child) { this.children.push(child); return child; }
    replaceChildren(...children) { this.children = children; }
    setAttribute(key, value) { this.attributes[key] = value; }
    getContext() { return { clearRect() {} }; }
    focus() {}
    remove() {}
  }
  const elements = new Map();
  const document = new Element();
  document.getElementById = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  document.createElement = tag => new Element(tag);
  document.createTextNode = text => Object.assign(new Element('#text'), { textContent: text });
  const requests = [], timers = [], sockets = [], sent = [];
  class WebSocket {
    static OPEN = 1;
    constructor() { this.readyState = 0; sockets.push(this); }
    open() { this.readyState = 1; this.onopen(); }
    close() { this.readyState = 3; this.onclose(); }
    send(value) { sent.push(JSON.parse(value)); }
  }
  const context = vm.createContext({ document, WebSocket, console, AbortSignal,
    require: name => name === 'electron' ? { ipcRenderer: { on() {}, send() {} } }
      : name === './camera-preview.cjs' ? { readCameraPreview() { throw new Error('Unexpected camera access'); } }
      : name === './brainbit-http.cjs' ? { requestBrainbit: (action, deviceId) => new Promise((resolve, reject) => {
        requests.push({ action, deviceId, resolve, reject });
      }) } : require(name),
    fetch() { throw new Error('Unexpected browser fetch'); },
    setTimeout: (callback, delay) => { timers.push({ callback, delay }); return timers.length; },
    clearTimeout: id => { if (timers[id - 1]) timers[id - 1].cancelled = true; },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../ui/renderer/app.js'), 'utf8'), context);
  sockets[0].open();
  const el = id => document.getElementById(`brainbit-${id}`);
  const message = data => vm.runInContext(`handleMessage(${JSON.stringify({ type: 'brainbit_status', ...data })})`, context);
  const open = value => { el('panel').open = value; el('panel').dispatch('toggle'); };
  const select = id => { el('devices').value = id; el('devices').dispatch('change'); };
  const tick = (delay = 1000) => {
    const timer = timers.find(t => t.delay === delay && !t.cancelled && !t.ran);
    assert.ok(timer, `Expected ${delay}ms timer`);
    timer.ran = true;
    timer.callback();
  };
  return { context, document, elements, requests, timers, sockets, sent, el, message, open, select, tick,
    get socket() { return sockets.at(-1); } };
}

const device = { id: 'opaque-device-1', name: 'BrainBit', family: 'BrainBit2' };
const status = (extra = {}) => ({ state: 'disconnected', available: true, busy: false,
  text: 'Ready to discover devices.', devices: [], device: null, ...extra });
const connected = (extra = {}) => status({ state: 'connected', text: 'Device connected.', devices: [device],
  device: { name: 'BrainBit', family: 'BrainBit2', battery: 83, firmware: '1.2.3' }, ...extra });
const flush = () => new Promise(resolve => setImmediate(resolve));
async function respond(request, data, ok = true) { request.resolve({ ok, status: ok ? 200 : 503, data }); await flush(); }

test('real markup has all BrainBit controls, starts closed and states limited scope', () => {
  const html = fs.readFileSync(path.join(__dirname, '../ui/renderer/index.html'), 'utf8');
  const markup = html.match(/<details class="brainbit-panel"[\s\S]*?<\/details>/)?.[0];
  assert.ok(markup);
  assert.doesNotMatch(markup.split('>')[0], /\bopen\b/);
  assert.match(markup, /Connection and device status only/);
  for (const id of ['panel', 'summary', 'status', 'error', 'devices', 'discover', 'connect',
    'disconnect', 'refresh', 'device-info', 'name', 'family', 'battery', 'firmware'])
    assert.match(markup, new RegExp(`id="brainbit-${id}"`));
  assert.match(markup, /label for="brainbit-devices"/);
});

test('opening, visibility and polling only read cached status and never connect', async () => {
  const r = renderer();
  assert.equal(r.requests.length, 0);
  assert.equal(r.el('panel').open, false);
  r.open(true);
  assert.deepEqual(r.requests.map(q => q.action), ['status']);
  r.open(true);
  assert.equal(r.requests.length, 1, 'one cached read at a time');
  await respond(r.requests[0], status());
  r.tick();
  assert.equal(r.requests[1].action, 'status');
  await respond(r.requests[1], status());
  r.open(false);
  assert.equal(r.timers.filter(t => t.delay === 1000 && !t.cancelled && !t.ran).length, 0);
  r.open(true);
  r.document.hidden = true;
  r.document.dispatch('visibilitychange');
  await respond(r.requests[2], connected());
  assert.equal(r.el('summary').textContent, 'Disconnected', 'hidden stale read ignored');
  r.document.hidden = false;
  r.document.dispatch('visibilitychange');
  assert.equal(r.requests[3].action, 'status');
  assert.ok(r.requests.every(q => q.action === 'status'));
});

test('explicit discover, select, connect and disconnect show busy state and metadata', async () => {
  const r = renderer();
  r.message(status());
  r.el('discover').dispatch('click');
  r.el('discover').dispatch('click');
  assert.equal(r.requests.length, 1);
  assert.equal(r.requests[0].action, 'discover');
  assert.equal(r.el('connect').disabled, true);
  assert.equal(r.el('disconnect').disabled, false);
  assert.equal(r.el('status').attributes['aria-busy'], 'true');
  await respond(r.requests[0], status({ devices: [device] }));
  assert.equal(r.el('connect').disabled, true, 'selection is explicit');
  r.select(device.id);
  r.el('connect').dispatch('click');
  r.el('connect').dispatch('click');
  assert.equal(r.requests.length, 2);
  assert.equal(r.requests[1].action, 'connect');
  assert.equal(r.requests[1].deviceId, device.id);
  await respond(r.requests[1], connected());
  assert.equal(r.el('summary').textContent, 'Connected');
  assert.equal(r.el('battery').textContent, '83%');
  assert.equal(r.el('firmware').textContent, '1.2.3');
  assert.equal(r.el('device-info').hidden, false);
  assert.equal(r.el('devices').disabled, true);
  r.el('disconnect').dispatch('click');
  assert.equal(r.requests[2].action, 'disconnect');
  assert.equal(r.el('disconnect').disabled, true);
  await respond(r.requests[2], status({ devices: [device] }));
  assert.equal(r.el('summary').textContent, 'Disconnected');
  assert.equal(r.el('device-info').hidden, true);
  assert.equal(r.el('battery').textContent, '');
});

test('disconnect cancels a pending connect and ignores its delayed response', async () => {
  const r = renderer();
  r.message(status({ devices: [device] }));
  r.select(device.id);
  r.el('connect').dispatch('click');
  r.el('disconnect').dispatch('click');
  assert.deepEqual(r.requests.map(q => q.action), ['connect', 'disconnect']);
  await respond(r.requests[1], status({ devices: [device] }));
  await respond(r.requests[0], connected());
  assert.equal(r.el('summary').textContent, 'Disconnected');
  assert.equal(r.el('status').attributes['aria-busy'], 'false');
});

test('selection stays stable on cached updates and text-only device values do not inject markup', () => {
  const r = renderer();
  const hostile = { id: 'opaque-id-do-not-show', name: '<img src=x onerror=alert(1)>', family: '<script>family</script>' };
  r.message(status({ devices: [hostile] }));
  r.select(hostile.id);
  const options = r.el('devices').children;
  r.message(status({ devices: [hostile] }));
  assert.equal(r.el('devices').children, options, 'no rebuild for identical list');
  assert.equal(r.el('devices').value, hostile.id);
  assert.equal(options[1].innerHTML, '');
  assert.match(options[1].textContent, /<img/);
  assert.ok(!options[1].textContent.includes(hostile.id));
  r.message(connected({ device: { ...hostile, battery: null, firmware: '<b>firmware</b>', serial: 'private-serial' } }));
  assert.equal(r.el('name').textContent, hostile.name);
  assert.equal(r.el('name').innerHTML, '');
  assert.equal(r.el('battery').textContent, 'Unknown');
  assert.equal(r.el('firmware').textContent, '<b>firmware</b>');
  assert.ok(![...r.elements.values()].some(el => el.textContent.includes('private-serial')));
});

test('fresh websocket status wins over delayed GET and old backend revision', async () => {
  const r = renderer();
  r.open(true);
  r.message(connected({ revision: 7 }));
  await respond(r.requests[0], status({ revision: 6 }));
  assert.equal(r.el('summary').textContent, 'Connected');
  r.message(status({ revision: 3 }));
  assert.equal(r.el('summary').textContent, 'Connected');
  r.message(status({ revision: 8 }));
  assert.equal(r.el('summary').textContent, 'Disconnected');
});

test('HTTP failure clears confirmed connection and busy until cached status recovers', async () => {
  const r = renderer();
  r.message(connected());
  r.open(true);
  await respond(r.requests[0], { error: 'Dongle unavailable.' }, false);
  assert.equal(r.el('summary').textContent, 'Status unavailable');
  assert.equal(r.el('device-info').hidden, true);
  assert.equal(r.el('status').attributes['aria-busy'], 'false');
  for (const id of ['discover', 'connect', 'disconnect', 'refresh', 'devices'])
    assert.equal(r.el(id).disabled, true);
  assert.equal(r.el('error').textContent, 'Dongle unavailable.');
  r.tick();
  await respond(r.requests[1], status({ devices: [device] }));
  assert.equal(r.el('discover').disabled, false);
  assert.equal(r.el('error').hidden, true);
});

test('offline invalidates busy actions and delayed errors cannot break a new connection', async () => {
  const r = renderer();
  r.message(status({ devices: [device] }));
  r.select(device.id);
  r.el('connect').dispatch('click');
  r.socket.close();
  assert.equal(r.el('summary').textContent, 'Backend offline');
  assert.equal(r.el('status').attributes['aria-busy'], 'false');
  assert.equal(r.el('disconnect').disabled, true);
  r.tick(2500);
  r.socket.open();
  r.message(connected({ revision: 1 }));
  r.requests[0].reject(new Error('Late timeout'));
  await flush();
  assert.equal(r.el('summary').textContent, 'Connected');
  assert.equal(r.el('error').hidden, true);
  assert.equal(r.requests.length, 1, 'reconnect does not start a device operation');
});

test('malformed status and action timeouts clear busy with recovery read only', async () => {
  const r = renderer();
  r.message(status());
  r.open(true);
  await respond(r.requests[0], { state: 'connected' });
  assert.equal(r.el('summary').textContent, 'Status unavailable');
  r.tick();
  await respond(r.requests[1], status());
  r.el('discover').dispatch('click');
  r.requests[2].reject(new Error('Timeout'));
  await flush();
  assert.equal(r.el('status').attributes['aria-busy'], 'false');
  assert.equal(r.el('discover').disabled, true);
  assert.equal(r.requests[3].action, 'status');
  await respond(r.requests[3], status({ state: 'scanning', busy: true }));
  assert.equal(r.el('disconnect').disabled, false, 'recovered busy operation can be cancelled');
});

test('unavailable SDK explains status and disables connection actions', () => {
  const r = renderer();
  r.message(status({ state: 'unavailable', available: false, text: 'BrainBit SDK is not installed.' }));
  assert.match(r.el('status').textContent, /SDK is not installed/);
  for (const id of ['discover', 'connect', 'disconnect', 'refresh']) {
    assert.equal(r.el(id).disabled, true);
    r.el(id).dispatch('click');
  }
  assert.equal(r.requests.length, 0);
});

test('a cached read failure cannot discard a later successful connection response', async () => {
  const r = renderer();
  r.message(status({ devices: [device], revision: 1 }));
  r.select(device.id);
  r.el('connect').dispatch('click');
  r.open(true);
  r.requests[1].reject(new Error('Cached read interrupted'));
  await flush();
  assert.equal(r.el('summary').textContent, 'Status unavailable');
  assert.equal(r.el('disconnect').disabled, true);
  await respond(r.requests[0], connected({ revision: 3 }));
  assert.equal(r.el('summary').textContent, 'Connected');
  assert.equal(r.el('disconnect').disabled, false);
  assert.equal(r.el('error').hidden, true);
});

test('a delayed action error preserves a newer completed websocket status', async () => {
  const r = renderer();
  r.message(status({ devices: [device], revision: 1 }));
  r.select(device.id);
  r.el('connect').dispatch('click');
  r.message(connected({ revision: 3 }));
  r.requests[0].reject(new Error('Response transport interrupted'));
  await flush();
  assert.equal(r.el('summary').textContent, 'Connected');
  assert.equal(r.el('disconnect').disabled, false);
  assert.equal(r.el('status').attributes['aria-busy'], 'false');
  assert.match(r.el('error').textContent, /could not be confirmed/);
});

test('refresh is explicit and BrainBit updates preserve draft and hand state', async () => {
  const r = renderer();
  r.elements.get('cmd-input').value = 'Keep this draft';
  vm.runInContext('handleMessage({type:"gesture_status",state:"off",running:false,available:true,input_mode:"mouse"})', r.context);
  const handLabel = r.elements.get('hand-mode-status').textContent;
  r.message(connected());
  r.el('refresh').dispatch('click');
  assert.equal(r.requests[0].action, 'refresh');
  await respond(r.requests[0], connected({ device: { name: 'BrainBit', battery: 79 } }));
  assert.equal(r.el('battery').textContent, '79%');
  assert.equal(r.elements.get('cmd-input').value, 'Keep this draft');
  assert.equal(r.elements.get('hand-mode-status').textContent, handLabel);
  assert.ok(r.sent.every(message => message.type === 'buffer'));
});
