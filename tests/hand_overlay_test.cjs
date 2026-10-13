// Drawing-only tests: no camera, image decoder, or native mouse input.
const test = require('node:test');
const assert = require('node:assert/strict');
const { drawHandOverlay } = require('../ui/renderer/hand-overlay.cjs');
const points = Array.from({ length: 21 }, (_, i) => [i / 30, i / 40, 0]);
function canvas() {
  const calls = [], context = new Proxy({}, { get: (_target, name) => (...args) => calls.push([name, ...args]), set: () => true });
  return { calls, context };
}
test('tracked hand has exactly the standalone 21 connections and 21 visible joints', () => {
  const c = canvas(); assert.equal(drawHandOverlay(c.context, { tracked: true, landmarks: points }, 480, 360), true);
  assert.equal(c.calls.filter(row => row[0] === 'lineTo').length, 21);
  assert.equal(c.calls.filter(row => row[0] === 'arc').length, 21);
  assert.deepEqual(c.calls.find(row => row[0] === 'arc'), ['arc', 0, 0, 3, 0, Math.PI * 2]);
});
test('untracked, incomplete, nonfinite, and invalid-size overlays draw no joints', () => {
  for (const [snapshot, width, height] of [
    [{ tracked: false, landmarks: points }, 480, 360], [{ tracked: true, landmarks: points.slice(1) }, 480, 360],
    [{ tracked: true, landmarks: [[NaN, 0], ...points.slice(1)] }, 480, 360],
    [{ tracked: true, landmarks: points }, Infinity, 360], [{ tracked: true, landmarks: points }, 480, 0],
  ]) { const c = canvas(); assert.equal(drawHandOverlay(c.context, snapshot, width, height), false); assert.equal(c.calls.length, 0); }
});
test('pointer region matches normalized standalone region and resets line dashes', () => {
  const c = canvas(); drawHandOverlay(c.context, { input_mode: 'mouse', control_region: [.1,.2,.9,.8] }, 400, 200);
  assert.deepEqual(c.calls.find(row => row[0] === 'strokeRect'), ['strokeRect', 40, 40, 320, 120.00000000000001]);
  assert.deepEqual(c.calls.filter(row => row[0] === 'setLineDash'), [['setLineDash', [5,5]], ['setLineDash', []]]);
  const bad = canvas(); drawHandOverlay(bad.context, { input_mode: 'mouse', control_region: [.8,.2,.1,.8] }, 400, 200);
  assert.equal(bad.calls.length, 0);
});
