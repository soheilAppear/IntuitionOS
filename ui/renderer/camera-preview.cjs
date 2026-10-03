// Native local request: browsers must not be able to read webcam frames through CORS.
const http = require('node:http');

function readCameraPreview() {
  return new Promise((resolve, reject) => {
    const request = http.get({ hostname: '127.0.0.1', port: 7432, path: '/gestures/preview',
      headers: { 'X-Intuition-Preview': '1' } }, response => {
      if (response.statusCode !== 200) {
        response.resume(); reject(new Error('Camera preview unavailable')); return;
      }
      let size = 0;
      const chunks = [];
      response.on('data', chunk => {
        size += chunk.length;
        if (size > 1024 * 1024) { request.destroy(new Error('Camera preview too large')); return; }
        chunks.push(chunk);
      });
      response.on('error', reject);
      response.on('aborted', () => reject(new Error('Camera preview interrupted')));
      response.on('end', () => {
        try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
        catch (error) { reject(error); }
      });
    });
    // An absolute deadline also bounds a server that trickles a response forever.
    const timeout = setTimeout(() => request.destroy(new Error('Camera preview timed out')), 1500);
    request.on('close', () => clearTimeout(timeout));
    request.on('error', reject);
  });
}

module.exports = { readCameraPreview };
