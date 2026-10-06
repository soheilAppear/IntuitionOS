// Local native transport only. Raw camera/EEG previews never use browser CORS.
const http = require('node:http');
const ACTIONS = new Set(['status', 'preview', 'start', 'stop', 'contact', 'arm', 'calibrate', 'reset_calibration']);

async function requestMultimodal(action = 'status', payload = {}) {
  if (!ACTIONS.has(action)) throw new Error('Invalid preview action');
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) throw new Error('Invalid preview arguments');
  const keys = Object.keys(payload);
  if (action === 'arm' ? keys.length !== 1 || typeof payload.enabled !== 'boolean'
    : action === 'calibrate' ? keys.length !== 1 || !['left', 'right'].includes(payload.label)
    : keys.length !== 0) throw new Error('Invalid preview arguments');
  const method = ['status', 'preview'].includes(action) ? 'GET' : 'POST';
  const headers = { 'X-Intuition-Multimodal': '1' };
  if (method === 'POST') headers['Content-Type'] = 'application/json';
  return new Promise((resolve, reject) => {
    let request, timeout, settled = false;
    const finish = (error, result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      if (error) reject(error); else resolve(result);
    };
    const fail = error => { finish(error); request?.destroy(); };
    request = http.request({ hostname: '127.0.0.1', port: 7432,
      path: `/multimodal/${action}`, method, headers }, response => {
      const chunks = [];
      let size = 0, ended = false;
      response.on('data', chunk => {
        if (settled) return;
        size += chunk.length;
        if (size > 512 * 1024) { fail(new Error('Preview response too large')); return; }
        chunks.push(chunk);
      });
      response.on('error', fail);
      response.on('aborted', () => fail(new Error('Preview response interrupted')));
      response.on('close', () => { if (!ended && !settled) fail(new Error('Preview response interrupted')); });
      response.on('end', () => {
        ended = true;
        if (settled) return;
        if (response.complete === false) { fail(new Error('Preview response interrupted')); return; }
        try {
          const data = JSON.parse(Buffer.concat(chunks).toString('utf8'));
          finish(null, { ok: response.statusCode >= 200 && response.statusCode < 300,
            status: response.statusCode, data });
        } catch (error) { fail(error); }
      });
    });
    request.on('error', finish);
    request.on('close', () => { if (!settled) finish(new Error('Preview response interrupted')); });
    timeout = setTimeout(() => fail(new Error('Preview request timed out')), method === 'GET' ? 5000 : 30000);
    request.end(method === 'POST' ? JSON.stringify(payload) : undefined);
  });
}

module.exports = { requestMultimodal };
