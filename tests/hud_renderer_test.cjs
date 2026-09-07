// Renderer behavior without Electron installation, devices or a running backend.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function renderer() {
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
  class WebSocket {
    static OPEN = 1;
    constructor() { this.readyState = 1; }
    send(value) { sent.push(JSON.parse(value)); }
    close() {}
  }
  const context = vm.createContext({
    document, WebSocket,
    require: name => name === 'electron' ? { ipcRenderer: { on() {}, send() {} } } : require(name),
    setTimeout: () => 1, clearTimeout() {}, console,
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../ui/renderer/app.js'), 'utf8'), context);
  const input = document.getElementById('cmd-input');
  const change = text => { input.value = text; input.dispatch('input'); };
  const key = (key, extra = {}) => input.dispatch('keydown', { key, ...extra });
  const message = value => vm.runInContext(`handleMessage(${JSON.stringify(value)})`, context);
  const show = (original, candidates, revision = 1) => message({
    type: 'resolution', original, candidates, token: 'visible-token', revision,
    client_revision: vm.runInContext('inputRevision', context), status: candidates.length ? 'correction' : 'exact',
  });
  return { elements, sent, context, input, change, key, message, show };
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
