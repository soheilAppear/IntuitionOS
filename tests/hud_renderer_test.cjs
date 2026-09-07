// Renderer behavior without Electron installation, devices or a running backend.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function renderer({ connected = true } = {}) {
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
  const sockets = [];
  const timers = [];
  const ipcHandlers = {};
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
    document, WebSocket,
    require: name => name === 'electron'
      ? { ipcRenderer: { on: (name, callback) => { ipcHandlers[name] = callback; }, send() {} } }
      : require(name),
    setTimeout: (callback, delay) => { timers.push({ callback, delay }); return timers.length; },
    clearTimeout: id => { if (timers[id - 1]) timers[id - 1].cancelled = true; }, console,
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
    elements, sent, context, input, change, key, message, show, reconnect, sockets,
    voiceToggle: () => ipcHandlers['voice-toggle'](),
    get socket() { return sockets.at(-1); },
  };
}

function textOf(element) { return element.textContent + element.children.map(textOf).join(''); }

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
