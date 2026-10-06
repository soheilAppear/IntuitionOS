// Native local requests keep BrainBit controls independent of browser CORS.
const http = require('node:http');

const ACTIONS = new Set(['status', 'discover', 'connect', 'disconnect', 'refresh']);
const MAX_RESPONSE_BYTES = 128 * 1024;

async function requestBrainbit(action = 'status', deviceId) {
  if (!ACTIONS.has(action)) throw new Error('Invalid BrainBit action');
  if (action === 'connect' && (typeof deviceId !== 'string' || !deviceId.trim())) {
    throw new Error('A BrainBit device ID is required');
  }

  const method = action === 'status' ? 'GET' : 'POST';
  const headers = { 'X-Intuition-Brainbit': '1' };
  const body = method === 'POST'
    ? JSON.stringify(action === 'connect' ? { device_id: deviceId } : {}) : undefined;
  if (method === 'POST') headers['Content-Type'] = 'application/json';

  return new Promise((resolve, reject) => {
    let request;
    let timeout;
    let settled = false;
    const finish = (error, result) => {
      if (settled) return;
      settled = true;
      if (timeout !== undefined) clearTimeout(timeout);
      if (error) reject(error);
      else resolve(result);
    };
    const fail = error => {
      finish(error);
      request?.destroy();
    };

    request = http.request({ hostname: '127.0.0.1', port: 7432,
      path: `/brainbit/${action}`, method, headers }, response => {
      let size = 0;
      let ended = false;
      const chunks = [];
      response.on('data', chunk => {
        if (settled) return;
        size += chunk.length;
        if (size > MAX_RESPONSE_BYTES) {
          fail(new Error('BrainBit response too large'));
          return;
        }
        chunks.push(chunk);
      });
      response.on('error', fail);
      response.on('aborted', () => fail(new Error('BrainBit response interrupted')));
      response.on('close', () => {
        if (!ended && !settled) fail(new Error('BrainBit response interrupted'));
      });
      response.on('end', () => {
        ended = true;
        if (settled) return;
        if (response.complete === false) {
          fail(new Error('BrainBit response interrupted'));
          return;
        }
        try {
          const data = JSON.parse(Buffer.concat(chunks).toString('utf8'));
          const status = response.statusCode;
          finish(null, { ok: status >= 200 && status < 300, status, data });
        } catch (error) {
          fail(error);
        }
      });
    });
    request.on('error', error => finish(error));
    request.on('close', () => {
      if (!settled) finish(new Error('BrainBit response interrupted'));
    });
    // A fixed deadline also bounds a server that keeps sending partial chunks.
    timeout = setTimeout(() => fail(new Error('BrainBit request timed out')),
      method === 'GET' ? 5000 : 30000);
    request.end(body);
  });
}

module.exports = { requestBrainbit };
